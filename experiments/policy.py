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


def permutation_null(rows: Sequence[dict], k: int, n_perm: int = 10_000, seed: int = SEED) -> np.ndarray:
    """Distribuição da qualidade sob seleção **aleatória de exatamente `k`** consultas.

    Um único sorteio pode, por acaso, empatar com a política — com n=30 e k=14 a
    variância é grande. O controle correto para "a política aprendeu a escolher?"
    é a distribuição inteira sob o mesmo orçamento de chamadas: a política só
    demonstra escolher bem se cair na cauda superior dela.

    Devolve `n_perm` valores da métrica sob sorteio sem reposição."""
    rng = np.random.default_rng(seed)
    n = len(rows)
    before = np.array([r["before"] for r in rows], dtype=np.float64)
    after = np.array([r["after"] for r in rows], dtype=np.float64)
    if k <= 0:
        return np.full(n_perm, before.mean())
    if k >= n:
        return np.full(n_perm, after.mean())
    # k índices distintos por linha, sem laço Python: ordena ruído e pega o topo.
    noise = rng.random((n_perm, n))
    picks = np.argpartition(noise, k - 1, axis=1)[:, :k]
    quality = np.tile(before, (n_perm, 1))
    np.put_along_axis(quality, picks, after[picks], axis=1)
    return quality.mean(axis=1)


def null_position(null: np.ndarray, observed: float) -> dict:
    """Onde a política cai na distribuição nula, e o p unilateral."""
    n_perm = int(null.size)
    p_right = float((null >= observed).sum() + 1) / (n_perm + 1)  # estimador sem viés
    return {
        "n_perm": n_perm,
        "null_mean": round(float(null.mean()), 4),
        "null_sd": round(float(null.std(ddof=1)), 4),
        "null_ci_low": round(float(np.percentile(null, 2.5)), 4),
        "null_ci_high": round(float(np.percentile(null, 97.5)), 4),
        "observed": round(float(observed), 4),
        "percentile": round(float((null < observed).mean() * 100), 1),
        "p_one_sided": round(p_right, 4),
    }


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


# Hiperparâmetros FIXOS, escolhidos a priori e nunca ajustados contra o
# resultado — se fossem ajustados, a validação cruzada estaria contaminada e o
# número reportado seria otimista.
LOGREG = {
    "C": 0.5,  # regularização L2; não ajustada
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 2000,
    "class_weight": "balanced",  # os positivos são minoria
}
DECISION_THRESHOLD = 0.5  # limiar padrão; NÃO ajustado (ver docstring)


class LearnedPolicy:
    """Regressão logística regularizada sobre P(a confirmação ajuda).

    Simples de propósito: com dezenas a poucas centenas de consultas rotuladas,
    um modelo maior aprende o ruído do conjunto. O que a proposta precisa mostrar
    não é capacidade, é que os sinais **baratos** carregam informação.

    Protocolo, para o experimento ser auditável:

    * **Rótulo positivo**: `gain > 0`, isto é, a confirmação **melhorou** a
      métrica naquela consulta. Consultas inalteradas (`gain == 0`) entram como
      negativas — a política deve aprender a não gastar chamada nelas tanto
      quanto deve evitar as que pioram.
    * **Padronização dentro do fold**: `StandardScaler` está dentro do
      `Pipeline`, e o `Pipeline` é ajustado **dentro de cada partição de
      treino**. Nenhuma estatística do conjunto de teste entra na padronização.
    * **Seleção de características**: não há. Todos os sinais de
      `experiments/features.py` entram, exceto dois constantes por construção
      (`o_n_sent`, `e_n_candidates`), excluídos **a priori** por nome — exclusão
      fixa, não guiada por dado.
    * **Regularização**: `C` fixo em 0.5, nunca ajustado contra o resultado.
      Não há busca de hiperparâmetro, logo não há a contaminação que uma busca
      interna mal feita introduziria.
    * **Limiar de decisão**: 0,5, o padrão. A fração acionada (47% no `hard`) é
      **consequência** do limiar, não um alvo escolhido. Varrer o limiar para
      maximizar a métrica seria ajustar contra o teste; a varredura aparece
      apenas como a fronteira qualidade--custo, que é descritiva."""

    def __init__(self, names: Sequence[str], C: float = LOGREG["C"], seed: int = SEED):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.names = list(names)
        self.model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        penalty=LOGREG["penalty"],
                        solver=LOGREG["solver"],
                        max_iter=LOGREG["max_iter"],
                        class_weight=LOGREG["class_weight"],
                        random_state=seed,
                    ),
                ),
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
    ap.add_argument("--n-perm", type=int, default=10_000, help="sorteios da distribuição nula")
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
    learned_call = prob >= DECISION_THRESHOLD
    policies = {
        "nunca": policy_never(rows),
        "sempre": policy_always(rows),
        "limiar": apply_threshold(rows, args.threshold_feature, t, invert=False),
        "regras": policy_rules(rows),
        f"aprendida@{DECISION_THRESHOLD}": learned_call,
        "oraculo": policy_oracle(rows),
    }
    f_learned = float(learned_call.mean())
    k_learned = int(learned_call.sum())
    policies[f"aleatoria@{f_learned:.2f}"] = policy_random(rows, f_learned, seed=args.seed)

    table = {name: evaluate(rows, call, args.lam, args.gamma, args.beta) for name, call in policies.items()}

    # ---- controle de orçamento igual, como DISTRIBUIÇÃO, não sorteio único ----
    null = permutation_null(rows, k_learned, n_perm=args.n_perm, seed=args.seed)
    pos = null_position(null, table[f"aprendida@{DECISION_THRESHOLD}"]["quality"])
    null_rules = permutation_null(rows, int(policy_rules(rows).sum()), n_perm=args.n_perm, seed=args.seed)
    pos_rules = null_position(null_rules, table["regras"]["quality"])

    print(f"\n### Políticas — {mode}")
    print("| política | fração acionada | qualidade | Δ vs nunca | ajudou/piorou | US$/1k | p95 (ms) | utilidade |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    ordem = (
        "nunca",
        "sempre",
        f"aleatoria@{f_learned:.2f}",
        "limiar",
        "regras",
        f"aprendida@{DECISION_THRESHOLD}",
        "oraculo",
    )
    for name in ordem:
        r = table[name]
        print(
            f"| {name} | {r['fraction_called']:.2f} | {r['quality']:.3f} | {r['delta_vs_never']:+.3f} | "
            f"{r['helped']}/{r['hurt']} | {r['cost_usd_per_1k']:.3f} | {r['latency_p95_ms']:.0f} | {r['net_utility']:+.3f} |"
        )

    print(f"\n### Controle de orçamento igual — distribuição de permutação ({args.n_perm} sorteios de {k_learned}/{n})")
    print("| política | acionadas | qualidade | nulo: média [IC 95%] | percentil no nulo | p (unilateral) |")
    print("|---|---:|---:|---|---:|---:|")
    for nome, pp, kk in (
        (f"aprendida@{DECISION_THRESHOLD}", pos, k_learned),
        ("regras", pos_rules, int(policy_rules(rows).sum())),
    ):
        print(
            f"| {nome} | {kk}/{n} | {pp['observed']:.3f} | "
            f"{pp['null_mean']:.3f} [{pp['null_ci_low']:.3f}; {pp['null_ci_high']:.3f}] | "
            f"P{pp['percentile']:.0f} | {pp['p_one_sided']:.3f} |"
        )
    print("\nUm sorteio único pode empatar com a política por acaso; a distribuição inteira")
    print("é o controle honesto. A política só demonstra ESCOLHER se cair na cauda superior.")

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
            "audit": {
                "positive_label": "gain > 0 (a confirmação melhorou a métrica); gain == 0 conta como negativo",
                "n_positive": n_help,
                "n_negative": n - n_help,
                "standardization": "StandardScaler dentro do Pipeline, ajustado DENTRO de cada fold",
                "feature_selection": "nenhuma; exclusão fixa a priori de o_n_sent e e_n_candidates (constantes)",
                "n_features": len(names),
                "features": names,
                "hyperparameters": LOGREG,
                "hyperparameter_search": "nenhuma — C fixo, não ajustado contra o resultado",
                "decision_threshold": DECISION_THRESHOLD,
                "decision_threshold_choice": "padrão 0,5; a fração acionada é consequência, não alvo",
                "cv_folds": args.cv,
                "permutation_control": {"n_perm": args.n_perm, "k": k_learned},
            },
        },
        "policies": table,
        "permutation_null": {f"aprendida@{DECISION_THRESHOLD}": pos, "regras": pos_rules},
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
