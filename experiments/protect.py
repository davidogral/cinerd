# -*- coding: utf-8 -*-
"""Proteção do primeiro resultado: confirmar evidência **sem** autoridade para reordenar o #1.

    .venv/bin/python -m experiments.protect --factorial experiments/results/<...>__factorial-object.json
    .venv/bin/python -m experiments.protect --factorial A.json B.json --cv 5 --out results/protect.json

De onde vem a hipótese
----------------------
Do contraste de verificadores (27B hospedado × 8B local) no *split* `hard`: as
**seis** regressões do verificador local ocorreram todas em consultas cujo #1 já
estava correto, e a única regressão do hospedado tem o mesmo mecanismo — o alvo
**foi** confirmado, mas junto com outro candidato que veio antes na ordem de
confiança, e a promoção múltipla rebaixou um #1 certo. O dano não vem de
confirmação falsa; vem de dar ao verificador **autoridade de reordenação**.

Daí o espaço de ação de três valores, em vez do binário chamar/não chamar:

| ação | o que faz | custo |
|---|---|---|
| `pular` | não chama o verificador | zero |
| `travar` | chama, usa a confirmação como evidência, mas o #1 do primeiro estágio **fica** | igual ao de chamar |
| `abrir` | chama e deixa reordenar — é o que está em produção hoje | igual ao de chamar |

`pular` economiza; `travar` não economiza nada, só limita o estrago. São
respostas a perguntas diferentes e o script mede as duas.

Por que não custa cota
----------------------
`travar` é uma regra de **reordenação posterior**: não muda prompt, candidato
nem resposta do modelo. O resultado sai por recomputação do que o
`experiments/factorial.py` já gravou por consulta (`rank_first_stage`, `rank`,
`n_picks`), sem nenhuma chamada nova.

O que é exato e o que é intervalo
---------------------------------
**`Success@1` é exato.** Travado, a posição 1 é ocupada pelo #1 do primeiro
estágio por construção: o alvo está em #1 **se e somente se** já estava.

`nDCG@10` sai como **intervalo**. `_promote` põe os confirmados na frente na
ordem de confiança; travar devolve o item travado à posição 1, e quem estava
antes dele desce uma. A posição do travado na lista promovida não é gravada, só
limitada (`≤ n_picks+1`), então a posição do alvo fica determinada em alguns
casos e em outros vale `r` ou `r+1`. O intervalo é reportado, não escondido —
reconstruir a ordem inteira exigiria replay do primeiro estágio, que é
determinístico mas caro, e não muda o desfecho primário.

Reuso adaptativo
----------------
A hipótese **nasceu** do `hard`. Medi-la ali é o reuso adaptativo que o próprio
artigo critica, então o script **recusa** entrada do `hard` sem
`--permitir-split-gerador`. Em `entity`/`object` ela é exploratória sobre
consultas que não a sugeriram; o teste que conta é o prospectivo lacrado (§6.1).
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

SEED = 20260919
GENERATING_SPLIT = "hard"


# --------------------------------------------------------------------- dados


def load_rows(paths: Sequence[str], cond: str, before: str, allow_generating: bool) -> list[dict]:
    """Uma linha por consulta, com as duas posições que a trava precisa."""
    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        run = payload.get("run") or {}
        if run.get("valid") is False:
            raise SystemExit(f"{os.path.basename(path)} está marcada INVÁLIDA — não entra em análise.")
        split = run.get("split", "?")
        if split == GENERATING_SPLIT and not allow_generating:
            raise SystemExit(
                f"recusado: {os.path.basename(path)} é do split `{GENERATING_SPLIT}`, que **gerou** esta hipótese.\n"
                "Medir a trava nele é reuso adaptativo. Use `entity`/`object`, ou\n"
                "--permitir-split-gerador se o objetivo for descrever o mecanismo, não estimar o efeito."
            )
        res = payload.get("results") or {}
        if cond not in res:
            raise SystemExit(f"{os.path.basename(path)}: falta a condição {cond} (tem: {', '.join(res)})")
        by_before = {r["qid"]: r for r in (res.get(before) or {}).get("per_query", [])}
        for r in res[cond]["per_query"]:
            rank_first = r.get("rank_first_stage")
            b = by_before.get(r["qid"])
            if b is not None and b.get("rank") != rank_first:
                raise SystemExit(
                    f"{r['qid']}: {before}.rank={b.get('rank')} difere de {cond}.rank_first_stage={rank_first} — "
                    "as duas condições não compartilham o mesmo estado de primeiro estágio."
                )
            rows.append(
                {
                    "qid": r["qid"],
                    "split": split,
                    "rank_first": rank_first,
                    "rank_open": r.get("rank"),
                    "n_picks": int(r.get("n_picks") or 0),
                    "confirmed_target": r.get("confirmed_target"),
                    "false_confirmation": r.get("false_confirmation"),
                    "features": r.get("features") or {},
                }
            )
    return rows


# --------------------------------------------------------------- posições


def lock_bounds(rank_first: Optional[int], rank_open: Optional[int], n_picks: int) -> tuple:
    """Posição do alvo com o #1 travado: `(mínima, máxima)`, iguais quando exata."""
    if rank_first is None:
        return None, None
    if not n_picks:  # nada foi confirmado: travar não muda nada
        return rank_first, rank_first
    if rank_first == 1:  # o alvo É o item travado
        return 1, 1
    if rank_open is None:
        return None, None
    # o item travado ocupa a posição 1; o alvo não pode estar lá.
    lo = max(int(rank_open), 2)
    hi = lo if int(rank_open) > n_picks + 1 else lo + 1
    return lo, hi


def success(rank: Optional[int]) -> float:
    return 1.0 if rank == 1 else 0.0


def ndcg(rank: Optional[int], k: int = 10) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def outcomes(rows: Sequence[dict]) -> dict:
    """Vetores por consulta para cada ação, prontos para combinar."""
    n = len(rows)
    out = {
        "s_skip": np.zeros(n),
        "s_open": np.zeros(n),
        "s_lock": np.zeros(n),
        "n_lo": np.zeros(n),
        "n_hi": np.zeros(n),
        "n_open": np.zeros(n),
        "n_skip": np.zeros(n),
    }
    for i, r in enumerate(rows):
        lo, hi = lock_bounds(r["rank_first"], r["rank_open"], r["n_picks"])
        out["s_skip"][i] = success(r["rank_first"])
        out["s_open"][i] = success(r["rank_open"])
        out["s_lock"][i] = success(lo)
        out["n_skip"][i] = ndcg(r["rank_first"])
        out["n_open"][i] = ndcg(r["rank_open"])
        out["n_lo"][i] = ndcg(hi)  # pior caso da trava
        out["n_hi"][i] = ndcg(lo)  # melhor caso da trava
    return out


def evaluate(rows: Sequence[dict], action: np.ndarray, o: dict) -> dict:
    """`action`: 0=pular, 1=travar, 2=abrir."""
    a = np.asarray(action)
    called = a > 0
    s = np.where(a == 0, o["s_skip"], np.where(a == 1, o["s_lock"], o["s_open"]))
    n_lo = np.where(a == 0, o["n_skip"], np.where(a == 1, o["n_lo"], o["n_open"]))
    n_hi = np.where(a == 0, o["n_skip"], np.where(a == 1, o["n_hi"], o["n_open"]))
    base = o["s_skip"]
    return {
        "fraction_called": round(float(called.mean()), 4),
        "fraction_locked": round(float((a == 1).mean()), 4),
        "success@1": round(float(s.mean()), 4),
        "delta_vs_never": round(float(s.mean() - base.mean()), 4),
        "ndcg@10_lo": round(float(n_lo.mean()), 4),
        "ndcg@10_hi": round(float(n_hi.mean()), 4),
        "helped": int(np.sum(s > base)),
        "hurt": int(np.sum(s < base)),
    }


# ------------------------------------------------------------------ políticas


def conf(rows: Sequence[dict], feature: str) -> np.ndarray:
    return np.array([float(r["features"].get(feature, 0.0)) for r in rows], dtype=np.float64)


def a_never(n: int) -> np.ndarray:
    return np.zeros(n, dtype=int)


def a_always(n: int) -> np.ndarray:
    return np.full(n, 2, dtype=int)


def a_always_locked(n: int) -> np.ndarray:
    return np.ones(n, dtype=int)


def a_lock_above(c: np.ndarray, tau: float) -> np.ndarray:
    """Trava onde o primeiro estágio está confiante; abre no resto."""
    return np.where(c >= tau, 1, 2)


def a_skip_above(c: np.ndarray, tau: float) -> np.ndarray:
    """Portão: nem chama onde o primeiro estágio está confiante."""
    return np.where(c >= tau, 0, 2)


def a_oracle(o: dict) -> np.ndarray:
    """Teto analítico: a melhor das três ações por consulta. Não implementável."""
    stack = np.vstack([o["s_skip"], o["s_lock"], o["s_open"]])
    return np.argmax(stack, axis=0).astype(int)


def fit_tau(rows, o: dict, c: np.ndarray, kind: str, grid: int = 60) -> float:
    """Corte que maximiza Success@1 no conjunto de treino; empate prefere travar mais."""
    build = a_lock_above if kind == "trava" else a_skip_above
    best_t, best_q, best_frac = float(c.max()) + 1.0, -1e9, 0.0
    for t in np.linspace(float(c.min()), float(c.max()) + 1e-9, grid):
        act = build(c, float(t))
        q = float(evaluate(rows, act, o)["success@1"])
        frac = float((act != 2).mean())
        if q > best_q + 1e-12 or (abs(q - best_q) <= 1e-12 and frac > best_frac):
            best_q, best_t, best_frac = q, float(t), frac
    return best_t


def permutation_null(o: dict, k: int, n_perm: int = 10_000, seed: int = SEED) -> np.ndarray:
    """Qualidade sob **travar k consultas ao acaso**, o resto aberto.

    É o controle que importa: a trava só demonstra usar o sinal de confiança se
    cair na cauda superior de travar o mesmo número de consultas por sorteio."""
    rng = np.random.default_rng(seed)
    s_open, s_lock = o["s_open"], o["s_lock"]
    n = int(s_open.size)
    if k <= 0:
        return np.full(n_perm, float(s_open.mean()))
    if k >= n:
        return np.full(n_perm, float(s_lock.mean()))
    noise = rng.random((n_perm, n))
    picks = np.argpartition(noise, k - 1, axis=1)[:, :k]
    q = np.tile(s_open, (n_perm, 1))
    np.put_along_axis(q, picks, s_lock[picks], axis=1)
    return q.mean(axis=1)


def null_position(null: np.ndarray, observed: float) -> dict:
    n_perm = int(null.size)
    return {
        "n_perm": n_perm,
        "null_mean": round(float(null.mean()), 4),
        "null_ci_low": round(float(np.percentile(null, 2.5)), 4),
        "null_ci_high": round(float(np.percentile(null, 97.5)), 4),
        "observed": round(float(observed), 4),
        "percentile": round(float((null < observed).mean() * 100), 1),
        "p_one_sided": round(float((null >= observed).sum() + 1) / (n_perm + 1), 4),
    }


def _folds(n: int, k: int, seed: int) -> list:
    idx = np.random.default_rng(seed).permutation(n)
    return [idx[i::k] for i in range(k)]


# ------------------------------------------------------------------ mecanismo


def mechanism(rows: Sequence[dict], o: dict) -> dict:
    """A contabilidade exata do que a trava recupera e do que ela custa.

    Em `Success@1` a trava só muda duas classes de consulta, e nenhuma outra:
    recupera as que a reordenação rebaixou de um #1 já certo, e perde as que a
    reordenação promoveu a #1 vindas de baixo."""
    rec = [r["qid"] for i, r in enumerate(rows) if o["s_skip"][i] == 1 and o["s_open"][i] == 0]
    lost = [r["qid"] for i, r in enumerate(rows) if o["s_skip"][i] == 0 and o["s_open"][i] == 1]
    rec_fc = sum(
        1 for i, r in enumerate(rows) if o["s_skip"][i] == 1 and o["s_open"][i] == 0 and r["false_confirmation"]
    )
    return {
        "recuperaveis": len(rec),
        "recuperaveis_qids": rec,
        "recuperaveis_com_confirmacao_falsa": rec_fc,
        "perdiveis": len(lost),
        "perdiveis_qids": lost,
        "nota": "a trava incondicional recupera todas as recuperáveis e perde todas as perdíveis; "
        "o sinal de confiança existe para separar as duas.",
    }


# ----------------------------------------------------------------------- CLI


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--factorial", nargs="+", required=True)
    ap.add_argument("--cond", default="C11", help="condição COM confirmação (default C11)")
    ap.add_argument("--before", default="C10", help="condição SEM confirmação, só para conferência (default C10)")
    ap.add_argument("--feature", default="r_margin_1_2_rel", help="sinal de confiança do primeiro estágio")
    ap.add_argument(
        "--cv", type=int, default=5, help="folds para escolher o corte; 0 = ajusta e avalia no mesmo conjunto"
    )
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--permitir-split-gerador", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    rows = load_rows(args.factorial, args.cond, args.before, args.permitir_split_gerador)
    if not rows:
        raise SystemExit("nenhuma consulta carregada.")
    o = outcomes(rows)
    c = conf(rows, args.feature)
    n = len(rows)
    splits = sorted({r["split"] for r in rows})

    print(f"» {n} consultas ({', '.join(splits)}), condição {args.cond}, sinal `{args.feature}`")
    print("» EXPLORATÓRIO (protocolo §6): splits históricos. O desfecho que conta é o prospectivo lacrado.\n")

    mech = mechanism(rows, o)
    print("### O que a trava pode mexer (exato, Success@1)")
    print(f"  recuperáveis (o #1 já estava certo e a reordenação estragou): {mech['recuperaveis']}")
    print(f"     destas, com confirmação falsa: {mech['recuperaveis_com_confirmacao_falsa']}")
    print(f"  perdíveis (a reordenação trouxe o alvo para #1 de baixo):     {mech['perdiveis']}")
    print("  qualquer outra consulta é indiferente à trava.\n")

    fixed = {
        "nunca (não chama)": a_never(n),
        "sempre, com autoridade (produção hoje)": a_always(n),
        "sempre, travado": a_always_locked(n),
        "oráculo de 3 ações (teto, não implementável)": a_oracle(o),
    }
    report: dict = {"policies": {}}
    print("### Políticas")
    print("| política | chamadas | travadas | Success@1 | Δ vs nunca | nDCG@10 [pior; melhor] |")
    print("|---|---:|---:|---:|---:|---|")
    for name, act in fixed.items():
        m = evaluate(rows, act, o)
        report["policies"][name] = m
        print(
            f"| {name} | {m['fraction_called']:.2f} | {m['fraction_locked']:.2f} | {m['success@1']:.3f} | "
            f"{m['delta_vs_never']:+.3f} | [{m['ndcg@10_lo']:.3f}; {m['ndcg@10_hi']:.3f}] |"
        )

    # ---- cortes, escolhidos fora do conjunto em que são avaliados
    for kind, build in (("trava", a_lock_above), ("portao", a_skip_above)):
        if args.cv and args.cv > 1:
            act = np.full(n, 2, dtype=int)
            taus = []
            for fold in _folds(n, args.cv, args.seed):
                te = np.zeros(n, dtype=bool)
                te[fold] = True
                tr_rows = [r for i, r in enumerate(rows) if not te[i]]
                tr_o = {k: v[~te] for k, v in o.items()}
                t = fit_tau(tr_rows, tr_o, c[~te], kind)
                taus.append(t)
                act[te] = build(c, t)[te]
            label = f"{kind}@corte (cv{args.cv})"
            tau_note = f"cortes por fold: {', '.join(f'{t:.3f}' for t in taus)}"
        else:
            t = fit_tau(rows, o, c, kind)
            act = build(c, t)
            label = f"{kind}@{t:.3f} (ajustado no mesmo conjunto)"
            tau_note = f"corte: {t:.3f}"
        m = evaluate(rows, act, o)
        m["tau_note"] = tau_note
        if kind == "trava":
            k = int((act == 1).sum())
            null = permutation_null(o, k, args.n_perm, args.seed)
            m["null"] = null_position(null, m["success@1"])
        report["policies"][label] = m
        print(
            f"| {label} | {m['fraction_called']:.2f} | {m['fraction_locked']:.2f} | {m['success@1']:.3f} | "
            f"{m['delta_vs_never']:+.3f} | [{m['ndcg@10_lo']:.3f}; {m['ndcg@10_hi']:.3f}] |"
        )

    for label, m in report["policies"].items():
        if "null" in m:
            nz = m["null"]
            print(f"\n### Controle de permutação — travar {m['fraction_locked']:.0%} ao acaso ({label})")
            print(f"  nulo: média {nz['null_mean']:.3f}  IC 95% [{nz['null_ci_low']:.3f}; {nz['null_ci_high']:.3f}]")
            print(f"  observado {nz['observed']:.3f} → percentil {nz['percentile']:.0f}, p = {nz['p_one_sided']:.3f}")
            if nz["percentile"] < 95:
                print("  → NÃO distinguível de travar ao acaso o mesmo número de consultas.")

    report["run"] = {
        "kind": "protect",
        "n": n,
        "splits": splits,
        "cond": args.cond,
        "feature": args.feature,
        "cv": args.cv,
        "seed": args.seed,
        "inputs": [os.path.relpath(p, _ROOT) for p in args.factorial],
        "exploratory": True,
    }
    report["mechanism"] = mech
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\n» JSON: {os.path.relpath(args.out, _ROOT)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
