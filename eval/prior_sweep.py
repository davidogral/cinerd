# -*- coding: utf-8 -*-
"""Varredura do *prior* de popularidade, estratificada por popularidade do alvo.

    .venv/bin/python -m eval.prior_sweep                      # split de teste
    .venv/bin/python -m eval.prior_sweep --split all          # os 5 splits juntos
    .venv/bin/python -m eval.prior_sweep --split all --out eval/results/prior-sweep.json

Por que existe
--------------
A fusão treinável (`eval/train_fusion.py`) sobe o *prior* de popularidade de
`0,35` para `0,90` e ganha no *split* `hard`. Isso é **hipótese, não decisão de
produção**: se o conjunto de avaliação for enviesado para filmes conhecidos,
subir o *prior* explora o viés do conjunto em vez de melhorar a busca. O peso
final é curadoria, não saída de otimizador.

O que este script mede, então, não é "qual peso é melhor" — é **em quem o peso
mexe**. Cada consulta é estratificada pelo percentil do `vote_count` do filme
alvo **dentro do catálogo inteiro** (não dentro do conjunto de consultas, que já
é enviesado), e o efeito de cada peso é reportado por estrato, com intervalo de
confiança por *bootstrap* pareado.

Método
------
Os canais são calculados **uma vez** por consulta (`eval.pipelines.raw_channels`,
sem peso) e recombinados analiticamente para cada valor do *prior*. A
reconstrução é exata: reproduz `_synopsis_scores` até o arredondamento de
`float32`, incluindo o peso lexical adaptativo por consulta. Determinístico e
sem rede, como o resto de `eval/`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from typing import Optional

os.environ.setdefault("RECOMENDAI_TMDB_NAMES", "0")

import numpy as np  # noqa: E402

from eval.dataset import dataset_sha1, load_queries  # noqa: E402
from eval.pipelines import CHANNELS, PRODUCTION_WEIGHTS, make_ctx, raw_channels  # noqa: E402
from eval.stats import paired_bootstrap  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Grade do sweep: o valor de produção, o aprendido no dev, e os pontos entre e
# além, para ver se o efeito é monotônico ou tem joelho.
GRID = (0.0, 0.15, 0.35, 0.5, 0.7, 0.9, 1.2)
PRODUCTION_PRIOR = PRODUCTION_WEIGHTS["prior"]
LEARNED_PRIOR = 0.9

# Cortes por percentil do catálogo, não do conjunto de consultas.
STRATA = (
    ("cauda longa (<P50)", 0.0, 50.0),
    ("meio (P50–P90)", 50.0, 90.0),
    ("popular (P90–P99)", 90.0, 99.0),
    ("muito popular (≥P99)", 99.0, 100.01),
)


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _ndcg10(rank: Optional[int]) -> float:
    if rank is None or rank > 10:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def collect(split: str, quiet: bool = False) -> list[dict]:
    """Uma entrada por consulta: posição do alvo para cada peso da grade."""
    from retrieval.search_engine import SearchEngine, clean_descriptive_query

    queries = load_queries(split)
    if not quiet:
        print(f"» motor de busca… ({len(queries)} consultas de {split})")
    engine = SearchEngine(rerank=False)
    if not engine.has_synopsis_index:
        raise SystemExit("índice de sinopse ausente — rode `python -m retrieval.index_builder`")

    ids = engine._movie_ids
    votes = np.array([float((engine.catalog.get(int(t)) or {}).get("vote_count") or 0.0) for t in ids])
    row_of = {int(t): i for i, t in enumerate(ids)}

    rows: list[dict] = []
    for i, q in enumerate(queries, 1):
        if not quiet and i % 20 == 0:
            print(f"  · {i}/{len(queries)}", flush=True)
        ctx = make_ctx(engine, q.query)
        raw = raw_channels(engine, ctx)
        w = dict(PRODUCTION_WEIGHTS)
        w["lexical"] = engine._adaptive_lexical_weight(clean_descriptive_query(q.query))
        base = sum(w[c] * raw[c] for c in CHANNELS if c != "prior")
        prior_vec = raw["prior"]

        target_row = row_of.get(int(q.relevant_id))
        target_votes = float(votes[target_row]) if target_row is not None else 0.0
        pct = float((votes < target_votes).mean() * 100.0)

        ranks: dict[str, Optional[int]] = {}
        for p in GRID:
            fused = base + p * prior_vec
            if target_row is None:
                ranks[str(p)] = None
                continue
            ranks[str(p)] = 1 + int(np.count_nonzero(fused > fused[target_row]))
        rows.append(
            {
                "qid": q.qid,
                "split": q.split,
                "title": q.relevant_title,
                "relevant_id": q.relevant_id,
                "target_vote_count": target_votes,
                "target_catalog_pct": round(pct, 1),
                "ranks": ranks,
            }
        )
    return rows


def _stratum(pct: float) -> str:
    for name, lo, hi in STRATA:
        if lo <= pct < hi:
            return name
    return STRATA[-1][0]


def summarize(rows: list[dict]) -> dict:
    out: dict = {"n": len(rows), "by_prior": {}, "by_stratum": {}, "contrast": {}}
    for p in GRID:
        key = str(p)
        ranks = [r["ranks"][key] for r in rows]
        out["by_prior"][key] = {
            "ndcg@10": round(float(np.mean([_ndcg10(x) for x in ranks])), 4),
            "success@1": round(float(np.mean([1.0 if x == 1 else 0.0 for x in ranks])), 4),
            "median_rank": int(np.median([x for x in ranks if x is not None])) if any(ranks) else None,
        }

    a_key, b_key = str(PRODUCTION_PRIOR), str(LEARNED_PRIOR)
    for name, _, _ in STRATA:
        sub = [r for r in rows if _stratum(r["target_catalog_pct"]) == name]
        if not sub:
            continue
        block: dict = {"n": len(sub)}
        for metric, fn in (("ndcg@10", _ndcg10), ("success@1", lambda x: 1.0 if x == 1 else 0.0)):
            a = [fn(r["ranks"][a_key]) for r in sub]
            b = [fn(r["ranks"][b_key]) for r in sub]
            block[metric] = {
                f"prior_{a_key}": round(float(np.mean(a)), 4),
                f"prior_{b_key}": round(float(np.mean(b)), 4),
                **{k: round(v, 4) for k, v in paired_bootstrap(a, b).items() if k in ("delta", "ci_low", "ci_high")},
                "wins": int(sum(1 for x, y in zip(a, b) if y > x)),
                "losses": int(sum(1 for x, y in zip(a, b) if y < x)),
            }
        out["by_stratum"][name] = block

    for metric, fn in (("ndcg@10", _ndcg10), ("success@1", lambda x: 1.0 if x == 1 else 0.0)):
        a = [fn(r["ranks"][a_key]) for r in rows]
        b = [fn(r["ranks"][b_key]) for r in rows]
        out["contrast"][metric] = paired_bootstrap(a, b)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="eval.prior_sweep", description="Prior de popularidade por estrato.")
    ap.add_argument("--split", default="test", help="dev|test|hard|object|entity|all")
    ap.add_argument("--out", help="grava o JSON")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    rows = collect(args.split, quiet=args.quiet)
    summary = summarize(rows)
    pcts = [r["target_catalog_pct"] for r in rows]

    print(f"\n### Composição do conjunto — {args.split} ({len(rows)} consultas)")
    print(
        f"percentil do alvo no catálogo: mediana **P{np.median(pcts):.0f}**, "
        f"mínimo P{min(pcts):.0f}, quartil inferior P{np.percentile(pcts, 25):.0f}"
    )
    counts = {name: sum(1 for p in pcts if _stratum(p) == name) for name, _, _ in STRATA}
    print("| estrato | consultas | % |")
    print("|---|---:|---:|")
    for name, _, _ in STRATA:
        print(f"| {name} | {counts[name]} | {counts[name] / len(rows):.1%} |")

    print("\n### Varredura do prior — agregado")
    print("| prior | nDCG@10 | Success@1 | mediana |")
    print("|---:|---:|---:|---:|")
    for p in GRID:
        m = summary["by_prior"][str(p)]
        mark = "  ← produção" if p == PRODUCTION_PRIOR else ("  ← aprendido no dev" if p == LEARNED_PRIOR else "")
        med = f"#{m['median_rank']}" if m["median_rank"] else "—"
        print(f"| {p} | {m['ndcg@10']:.3f} | {m['success@1']:.3f} | {med} |{mark}")

    print(f"\n### {LEARNED_PRIOR} − {PRODUCTION_PRIOR}, por estrato de popularidade do alvo")
    print("| estrato | n | nDCG@10 (0,35) | nDCG@10 (0,9) | Δ [IC 95%] | melhora/piora |")
    print("|---|---:|---:|---:|---|---:|")
    for name, _, _ in STRATA:
        b = summary["by_stratum"].get(name)
        if not b:
            print(f"| {name} | 0 | — | — | — | — |")
            continue
        m = b["ndcg@10"]
        print(
            f"| {name} | {b['n']} | {m[f'prior_{PRODUCTION_PRIOR}']:.3f} | {m[f'prior_{LEARNED_PRIOR}']:.3f} | "
            f"{m['delta']:+.3f} [{m['ci_low']:+.3f}; {m['ci_high']:+.3f}] | {m['wins']}/{m['losses']} |"
        )
    c = summary["contrast"]["ndcg@10"]
    print(
        f"\nagregado: {c['delta']:+.4f} [{c['ci_low']:+.4f}; {c['ci_high']:+.4f}]  "
        f"({c['wins']} melhoram, {c['losses']} pioram, {c['ties']} iguais)"
    )
    print("\nLeitura: um ganho agregado que desaparece ou inverte no estrato de cauda longa")
    print("é o conjunto recompensando popularidade, não a busca melhorando. O peso final")
    print("é curadoria — ver docs/PROTOCOLO-TOIS.md §7.4.")

    payload = {
        "run": {
            "kind": "prior-sweep",
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": _git_commit(),
            "split": args.split,
            "dataset_sha1": dataset_sha1(),
            "grid": list(GRID),
            "production_prior": PRODUCTION_PRIOR,
            "learned_prior": LEARNED_PRIOR,
            "strata": [{"name": n, "pct_low": lo, "pct_high": hi} for n, lo, hi in STRATA],
            "target_catalog_pct": {
                "median": float(np.median(pcts)),
                "min": float(min(pcts)),
                "p25": float(np.percentile(pcts, 25)),
            },
        },
        "summary": summary,
        "per_query": rows,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\n» JSON: {os.path.relpath(args.out, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
