# -*- coding: utf-8 -*-
"""Acionamento **seletivo** da verificação por LLM (protocolo §7.2 item 8, RQ2).

    .venv/bin/python -m experiments.policy --factorial experiments/results/<...>__factorial-hard.json
    .venv/bin/python -m experiments.policy --factorial A.json B.json --metric ndcg@10 --cv 5
    .venv/bin/python -m experiments.policy --factorial A.json --train-on A.json --test-on B.json

A pergunta (RQ2): dá para decidir, **antes** de pagar pela confirmação, se ela
vai melhorar aquele ranking? A decisão só pode usar `experiments/features.py` —
sinais da consulta, do ranking inicial, da evidência textual e da operação.
Nada da resposta que se quer prever.

Políticas comparadas
--------------------
| política | o que faz | papel |
|---|---|---|
| `nunca` | não chama | piso de custo |
| `sempre` | chama em toda consulta (= produção hoje) | teto de custo |
| `aleatoria@f` | chama uma fração `f` sorteada | **o controle que importa**: mesmo orçamento, sem inteligência |
| `limiar` | chama quando um único sinal cruza um corte | baseline obrigatório do protocolo |
| `regras` | margem pequena **ou** evidência melhor abaixo do #1 | heurística legível |
| `aprendida` | regressão logística regularizada sobre os sinais, calibrada | a proposta |
| `oraculo` | chama só quando o ganho real seria positivo | **teto analítico, não implementável** |

Uma política só "aprendeu quando chamar" se vencer `aleatoria@f` **na mesma
fração acionada** — vencer `sempre` gastando menos é fácil e não prova nada
sobre a decisão.

Aviso de escopo
---------------
Rodado sobre os *splits* históricos, isto é **exploratório** (protocolo §6): as
consultas já foram usadas em decisões de projeto. O resultado que conta vem do
teste prospectivo lacrado. O script imprime esse aviso junto com os números.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

SEED = 20260918


# ------------------------------------------------------------------- dados


def _metric(rank: Optional[int], name: str) -> float:
    if name == "success@1":
        return 1.0 if rank == 1 else 0.0
    if name == "mrr":
        return 1.0 / rank if rank else 0.0
    if name.startswith("ndcg@"):
        k = int(name.split("@")[1])
        if rank is None or rank > k:
            return 0.0
        return 1.0 / math.log2(rank + 1)
    raise ValueError(f"métrica desconhecida: {name}")


def _ledger_costs(path: str) -> dict[str, dict]:
    """(qid) -> custo/latência da chamada de confirmação daquela consulta."""
    out: dict[str, dict] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("stage") != "confirm" or rec.get("repeat") != 0 or rec.get("from_cache"):
                continue
            out[rec["qid"]] = {
                "cost_usd": float(rec.get("cost_usd") or 0.0),
                "latency_ms": float(rec.get("latency_ms") or 0.0),
                "tokens_in": int(rec.get("tokens_in") or 0),
                "tokens_out": int(rec.get("tokens_out") or 0),
            }
    return out


def load_dataset(paths: Sequence[str], before: str, after: str, metric: str) -> list[dict]:
    """Uma linha por consulta: sinais pré-chamada + ganho real da confirmação."""
    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        res = payload.get("results") or {}
        if before not in res or after not in res:
            raise SystemExit(f"{os.path.basename(path)}: faltam as condições {before}/{after} (tem: {', '.join(res)})")
        ledger_rel = (payload.get("run") or {}).get("llm", {}).get("ledger")
        costs = _ledger_costs(os.path.join(_ROOT, ledger_rel) if ledger_rel else "")
        by_qid_after = {r["qid"]: r for r in res[after]["per_query"]}
        split = (payload.get("run") or {}).get("split", "?")
        for r in res[before]["per_query"]:
            a = by_qid_after.get(r["qid"])
            if a is None:
                continue
            m_before = _metric(r.get("rank"), metric)
            m_after = _metric(a.get("rank"), metric)
            c = costs.get(r["qid"], {})
            rows.append(
                {
                    "qid": r["qid"],
                    "split": split,
                    "features": r.get("features") or {},
                    "before": m_before,
                    "after": m_after,
                    "gain": m_after - m_before,
                    "cost_usd": c.get("cost_usd", 0.0),
                    "latency_ms": c.get("latency_ms", 0.0),
                    "tokens_in": c.get("tokens_in", 0),
                    "error_type": a.get("error_type"),
                    "false_confirmation": a.get("false_confirmation"),
                }
            )
    return rows


def feature_matrix(rows: Sequence[dict], names: Sequence[str]) -> np.ndarray:
    return np.array([[float(r["features"].get(n, 0.0)) for n in names] for r in rows], dtype=np.float64)


def feature_names(rows: Sequence[dict]) -> list[str]:
    keys: set = set()
    for r in rows:
        keys.update(r["features"].keys())
    return sorted(keys)


# ---------------------------------------------------------------- políticas


def evaluate(rows: Sequence[dict], call: np.ndarray, lam: float, gamma: float, beta: float) -> dict:
    """Qualidade, custo e utilidade líquida de um vetor de decisões (0/1)."""
    call = np.asarray(call, dtype=bool)
    quality = float(np.mean([r["after"] if c else r["before"] for r, c in zip(rows, call)]))
    baseline = float(np.mean([r["before"] for r in rows]))
    gains = np.array([r["gain"] for r in rows])
    cost = float(sum(r["cost_usd"] for r, c in zip(rows, call) if c))
    lat = [r["latency_ms"] for r, c in zip(rows, call) if c]
    lat_sorted = sorted(lat)
    helped = int(np.sum((gains > 0) & call))
    hurt = int(np.sum((gains < 0) & call))
    missed = int(np.sum((gains > 0) & ~call))
    return {
        "fraction_called": round(float(call.mean()), 4),
        "quality": round(quality, 4),
        "delta_vs_never": round(quality - baseline, 4),
        "helped": helped,
        "hurt": hurt,
        "missed_gain": missed,
        "precision_of_calls": round(helped / max(1, int(call.sum())), 4),
        "cost_usd": round(cost, 6),
        "cost_usd_per_1k": round(cost / len(rows) * 1000, 4) if rows else 0.0,
        "latency_p50_ms": round(lat_sorted[len(lat_sorted) // 2], 1) if lat_sorted else 0.0,
        "latency_p95_ms": round(lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))], 1)
        if lat_sorted
        else 0.0,
        # utilidade líquida com pesos DECLARADOS (protocolo §3); custo em US$/1k
        # consultas para as três parcelas ficarem na mesma ordem de grandeza.
        "net_utility": round(
            (quality - baseline)
            - lam * (cost / len(rows) * 1000 if rows else 0.0)
            - gamma * (float(np.mean(lat)) / 1000.0 if lat else 0.0)
            - beta * (hurt / len(rows) if rows else 0.0),
            4,
        ),
    }


def policy_never(rows) -> np.ndarray:
    return np.zeros(len(rows), dtype=bool)


def policy_always(rows) -> np.ndarray:
    return np.ones(len(rows), dtype=bool)


def policy_random(rows, fraction: float, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(rows)
    k = int(round(fraction * n))
    call = np.zeros(n, dtype=bool)
    if k > 0:
        call[rng.choice(n, size=min(k, n), replace=False)] = True
    return call


def policy_oracle(rows) -> np.ndarray:
    return np.array([r["gain"] > 0 for r in rows], dtype=bool)


def policy_rules(rows) -> np.ndarray:
    """Legível e sem treino: chama quando o ranking está **inseguro** (margem
    relativa pequena entre #1 e #2) **ou** quando há candidato abaixo do #1 com
    evidência textual melhor — que é exatamente a situação da hipótese H1."""
    out = []
    for r in rows:
        f = r["features"]
        insecure = float(f.get("r_margin_1_2_rel", 1.0)) < 0.08
        better_below = float(f.get("e_cov_gap", 0.0)) > 0.0
        out.append(bool(insecure or better_below))
    return np.array(out, dtype=bool)


def fit_threshold(rows, feature: str, invert: bool, grid: int = 40) -> float:
    """Melhor corte de um único sinal, por utilidade de qualidade no treino."""
    vals = np.array([float(r["features"].get(feature, 0.0)) for r in rows])
    best_t, best_q = float(vals.min()), -1e9
    for t in np.linspace(vals.min(), vals.max(), grid):
        call = vals <= t if not invert else vals >= t
        q = float(np.mean([r["after"] if c else r["before"] for r, c in zip(rows, call)]))
        # empate: prefere chamar menos
        if q > best_q + 1e-12 or (abs(q - best_q) <= 1e-12 and call.mean() < (vals <= best_t).mean()):
            best_q, best_t = q, float(t)
    return best_t


def apply_threshold(rows, feature: str, t: float, invert: bool) -> np.ndarray:
    vals = np.array([float(r["features"].get(feature, 0.0)) for r in rows])
    return vals >= t if invert else vals <= t


class LearnedPolicy:
    """Regressão logística regularizada sobre P(a confirmação ajuda).

    Simples de propósito: com dezenas a poucas centenas de consultas rotuladas,
    um modelo maior aprende o ruído do conjunto. O que a proposta precisa mostrar
    não é capacidade, é que os sinais **baratos** carregam informação."""

    def __init__(self, names: Sequence[str], C: float = 0.5, seed: int = SEED):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.names = list(names)
        self.model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)),
            ]
        )
        self.fitted = False

    def fit(self, rows) -> "LearnedPolicy":
        X = feature_matrix(rows, self.names)
        y = np.array([1 if r["gain"] > 0 else 0 for r in rows], dtype=int)
        if len(set(y.tolist())) < 2:
            self.fitted = False
            return self
        self.model.fit(X, y)
        self.fitted = True
        return self

    def proba(self, rows) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(rows))
        return self.model.predict_proba(feature_matrix(rows, self.names))[:, 1]

    def coefficients(self) -> dict[str, float]:
        if not self.fitted:
            return {}
        coef = self.model.named_steps["clf"].coef_[0]
        return {n: round(float(c), 4) for n, c in sorted(zip(self.names, coef), key=lambda kv: -abs(kv[1]))}


def brier(prob: np.ndarray, rows) -> float:
    y = np.array([1.0 if r["gain"] > 0 else 0.0 for r in rows])
    return round(float(np.mean((prob - y) ** 2)), 4)


def reliability(prob: np.ndarray, rows, bins: int = 5) -> list[dict]:
    y = np.array([1.0 if r["gain"] > 0 else 0.0 for r in rows])
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        sel = (prob >= edges[i]) & (prob < edges[i + 1] if i < bins - 1 else prob <= 1.0)
        if not sel.any():
            continue
        out.append(
            {
                "bin": f"[{edges[i]:.1f},{edges[i + 1]:.1f})",
                "n": int(sel.sum()),
                "predito": round(float(prob[sel].mean()), 3),
                "observado": round(float(y[sel].mean()), 3),
            }
        )
    return out


# ------------------------------------------------------------------ fronteira


def frontier(rows, score: np.ndarray, lam: float, gamma: float, beta: float, steps: int = 21) -> list[dict]:
    """Curva qualidade–custo: aciona as consultas de maior score, variando a fração."""
    order = np.argsort(-np.asarray(score), kind="stable")
    n = len(rows)
    out = []
    for f in np.linspace(0, 1, steps):
        k = int(round(f * n))
        call = np.zeros(n, dtype=bool)
        call[order[:k]] = True
        row = evaluate(rows, call, lam, gamma, beta)
        row["target_fraction"] = round(float(f), 3)
        out.append(row)
    return out


# ----------------------------------------------------------------------- CLI


def _folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [idx[i::k] for i in range(k)]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.policy", description="Política de acionamento seletivo.")
    ap.add_argument("--factorial", nargs="+", required=True, help="JSONs de experiments.factorial")
    ap.add_argument("--before", default="C10", help="condição SEM confirmação (default: C10)")
    ap.add_argument("--after", default="C11", help="condição COM confirmação (default: C11)")
    ap.add_argument("--metric", default="success@1")
    ap.add_argument("--cv", type=int, default=5, help="folds; 0 = treina e avalia no mesmo conjunto (só diagnóstico)")
    ap.add_argument("--lam", type=float, default=1.0, help="peso do custo (US$/1k consultas) na utilidade")
    ap.add_argument("--gamma", type=float, default=0.02, help="peso da latência média (s) na utilidade")
    ap.add_argument("--beta", type=float, default=0.5, help="peso da fração de consultas PIORADAS")
    ap.add_argument("--threshold-feature", default="r_margin_1_2_rel")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    rows = load_dataset(args.factorial, args.before, args.after, args.metric)
    if not rows:
        raise SystemExit("nenhuma consulta carregada")
    names = [n for n in feature_names(rows) if n not in ("o_n_sent", "e_n_candidates")]
    n = len(rows)
    gains = np.array([r["gain"] for r in rows])
    n_help, n_hurt = int((gains > 0).sum()), int((gains < 0).sum())

    print(f"» {n} consultas · {args.before} → {args.after} · métrica {args.metric}")
    print(f"  a confirmação ajudou em {n_help}, prejudicou em {n_hurt}, não mudou em {n - n_help - n_hurt}")
    if n_help == 0:
        print("  (sem nenhum ganho observado, não há o que uma política aprenda a acionar)")

    # ---- probabilidade prevista, fora da amostra quando há folds suficientes
    prob = np.zeros(n)
    coefs: dict[str, float] = {}
    if args.cv and args.cv > 1 and n_help >= args.cv and n_hurt + (n - n_help - n_hurt) >= args.cv:
        for fold in _folds(n, args.cv, args.seed):
            mask = np.ones(n, dtype=bool)
            mask[fold] = False
            train = [rows[i] for i in range(n) if mask[i]]
            test = [rows[i] for i in fold]
            model = LearnedPolicy(names, seed=args.seed).fit(train)
            prob[fold] = model.proba(test)
        coefs = LearnedPolicy(names, seed=args.seed).fit(rows).coefficients()
        mode = f"validação cruzada {args.cv}-fold (predição fora da amostra)"
    else:
        model = LearnedPolicy(names, seed=args.seed).fit(rows)
        prob = model.proba(rows)
        coefs = model.coefficients()
        mode = "treino = teste (DIAGNÓSTICO, número otimista)"

    # ---- políticas
    t = fit_threshold(rows, args.threshold_feature, invert=False)
    policies = {
        "nunca": policy_never(rows),
        "sempre": policy_always(rows),
        "limiar": apply_threshold(rows, args.threshold_feature, t, invert=False),
        "regras": policy_rules(rows),
        "aprendida@0.5": prob >= 0.5,
        "oraculo": policy_oracle(rows),
    }
    # controle de orçamento igual: aleatória com a MESMA fração da política aprendida
    f_learned = float((prob >= 0.5).mean())
    policies[f"aleatoria@{f_learned:.2f}"] = policy_random(rows, f_learned, seed=args.seed)

    table = {name: evaluate(rows, call, args.lam, args.gamma, args.beta) for name, call in policies.items()}

    print(f"\n### Políticas — {mode}")
    print("| política | fração acionada | qualidade | Δ vs nunca | ajudou/piorou | US$/1k | p95 (ms) | utilidade |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("nunca", "sempre", f"aleatoria@{f_learned:.2f}", "limiar", "regras", "aprendida@0.5", "oraculo"):
        r = table[name]
        print(
            f"| {name} | {r['fraction_called']:.2f} | {r['quality']:.3f} | {r['delta_vs_never']:+.3f} | "
            f"{r['helped']}/{r['hurt']} | {r['cost_usd_per_1k']:.3f} | {r['latency_p95_ms']:.0f} | {r['net_utility']:+.3f} |"
        )

    print(f"\nlimiar ajustado: {args.threshold_feature} ≤ {t:.4f}")
    if coefs:
        top = list(coefs.items())[:8]
        print("sinais mais pesados na política aprendida: " + ", ".join(f"{k} ({v:+.2f})" for k, v in top))
    print(f"calibração (Brier, menor é melhor): {brier(prob, rows)}")

    front = frontier(rows, prob, args.lam, args.gamma, args.beta)
    print("\n### Fronteira qualidade–custo (aciona os de maior probabilidade prevista)")
    print("| fração | qualidade | Δ vs nunca | US$/1k |")
    print("|---:|---:|---:|---:|")
    for row in front[:: max(1, len(front) // 6)]:
        print(
            f"| {row['fraction_called']:.2f} | {row['quality']:.3f} | {row['delta_vs_never']:+.3f} | {row['cost_usd_per_1k']:.3f} |"
        )

    print("\nPORTÃO (protocolo §8.4): a política só 'aprendeu quando chamar' se vencer")
    print("`aleatoria@f` na MESMA fração, no teste prospectivo lacrado. Sobre split histórico,")
    print("isto é exploratório — as consultas já participaram de decisões de projeto.")

    payload = {
        "run": {
            "kind": "selective-policy",
            "sources": [os.path.relpath(p, _ROOT) for p in args.factorial],
            "before": args.before,
            "after": args.after,
            "metric": args.metric,
            "n": n,
            "mode": mode,
            "cv": args.cv,
            "seed": args.seed,
            "utility_weights": {"lambda_cost": args.lam, "gamma_latency": args.gamma, "beta_regression": args.beta},
            "threshold": {"feature": args.threshold_feature, "value": round(t, 6)},
            "exploratory": True,
        },
        "policies": table,
        "coefficients": coefs,
        "calibration": {"brier": brier(prob, rows), "reliability": reliability(prob, rows)},
        "frontier": front,
        "per_query": [
            {
                "qid": r["qid"],
                "split": r["split"],
                "gain": r["gain"],
                "p_help": round(float(p), 4),
                "error_type": r["error_type"],
            }
            for r, p in zip(rows, prob)
        ],
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\n» JSON: {os.path.relpath(args.out, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
