# -*- coding: utf-8 -*-
"""Treina a **fusão linear simples** (baseline do protocolo, §7.2 item 4).

    .venv/bin/python -m eval.train_fusion                 # treina no dev, grava params
    .venv/bin/python -m eval.train_fusion --budget 400
    .venv/bin/python -m eval.train_fusion --split dev --dry-run

Por que existe
--------------
Os pesos da fusão de produção foram escolhidos **à mão**, iterando contra
dev/test. Um revisor pergunta, com razão: o ganho vem da arquitetura de oito
sinais ou do ajuste manual? Este script dá ao mesmo conjunto de canais um
ajuste **automático, declarado e reprodutível**, treinado **somente no split
dev**, e grava os pesos em `eval/params/fusion_learned.json`. O pipeline
`fusion_learned` (`eval/pipelines.py`) usa esses pesos; a comparação contra
`fusion` passa a isolar arquitetura de ajuste.

Método
------
Busca coordenada determinística a partir dos pesos de produção: cada canal é
varrido numa grade, o melhor valor é mantido, passa-se ao próximo canal, repete
por rodadas até o **orçamento** de avaliações da função-objetivo acabar. Sem
aleatoriedade, sem gradiente — é o análogo automático do que foi feito à mão, e
o orçamento fica registrado no JSON para a comparação ser honesta.

Objetivo: **nDCG@10 médio no dev**. Para cada consulta, os canais são
pré-computados uma vez sobre um *pool* de candidatos (união do top-`POOL_PER_CHANNEL`
de cada canal + o filme alvo); a posição do alvo é calculada dentro do pool.
É uma **aproximação**: um filme fora do top-`POOL_PER_CHANNEL` de todos os canais
não é considerado. Ele teria de ser mediano em tudo e ainda assim superar o alvo,
o que a soma não-negativa torna improvável — e a avaliação final (`eval.run`)
ranqueia o catálogo inteiro, sem aproximação nenhuma.

Determinístico e sem rede, como o resto de `eval/`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from datetime import datetime, timezone

os.environ.setdefault("RECOMENDAI_TMDB_NAMES", "0")

import numpy as np  # noqa: E402

from eval.dataset import dataset_sha1, load_queries  # noqa: E402
from eval.pipelines import CHANNELS, PRODUCTION_WEIGHTS, make_ctx, raw_channels  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
PARAMS_DIR = os.path.join(_HERE, "params")
PARAMS_PATH = os.path.join(PARAMS_DIR, "fusion_learned.json")

POOL_PER_CHANNEL = 500
# Grade por coordenada: cobre de "desligado" ao dobro do maior peso de produção.
GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.1, 1.3)
DEFAULT_BUDGET = 300


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def build_cache(split: str, quiet: bool = False) -> list[dict]:
    """Uma entrada por consulta: matriz (canais × candidatos) e o índice do alvo."""
    from retrieval.search_engine import SearchEngine

    queries = load_queries(split)
    if not quiet:
        print(f"» carregando motor de busca… ({len(queries)} consultas de {split})")
    engine = SearchEngine(rerank=False)
    if not engine.has_synopsis_index:
        raise SystemExit("índice de sinopse ausente — rode `python -m retrieval.index_builder`")

    row_of = {int(t): i for i, t in enumerate(engine._movie_ids)}
    cache: list[dict] = []
    for i, q in enumerate(queries, 1):
        if not quiet and i % 10 == 0:
            print(f"  · {i}/{len(queries)}", flush=True)
        raw = raw_channels(engine, make_ctx(engine, q.query))
        pool: set[int] = set()
        for name in CHANNELS:
            vec = raw[name]
            top = np.argsort(vec, kind="stable")[::-1][:POOL_PER_CHANNEL]
            pool.update(int(x) for x in top)
        target_row = row_of.get(int(q.relevant_id))
        if target_row is None:
            continue  # alvo fora do índice: não pontua em nenhuma configuração
        pool.add(target_row)
        idx = np.array(sorted(pool), dtype=np.int64)
        mat = np.stack([raw[name][idx] for name in CHANNELS]).astype(np.float32)
        cache.append({"qid": q.qid, "mat": mat, "target": int(np.searchsorted(idx, target_row))})
    return cache


def ndcg10(cache: list[dict], weights: np.ndarray) -> float:
    """nDCG@10 médio sob um vetor de pesos (ordem = `CHANNELS`)."""
    total = 0.0
    for row in cache:
        scores = weights @ row["mat"]
        t = scores[row["target"]]
        # posição = 1 + quantos candidatos pontuam estritamente acima do alvo.
        rank = 1 + int(np.count_nonzero(scores > t))
        if rank <= 10:
            total += 1.0 / math.log2(rank + 1)
    return total / len(cache) if cache else 0.0


def coordinate_search(cache: list[dict], budget: int, quiet: bool = False) -> tuple[np.ndarray, float, int, list[dict]]:
    w = np.array([PRODUCTION_WEIGHTS[c] for c in CHANNELS], dtype=np.float64)
    best = ndcg10(cache, w)
    used = 1
    trace = [{"eval": used, "ndcg@10": round(best, 4), "weights": dict(zip(CHANNELS, w.round(3).tolist()))}]
    if not quiet:
        print(f"  início (pesos de produção): nDCG@10={best:.4f}")

    improved = True
    rounds = 0
    while improved and used < budget:
        improved = False
        rounds += 1
        for c, _name in enumerate(CHANNELS):
            for value in GRID:
                if used >= budget:
                    break
                if abs(value - w[c]) < 1e-9:
                    continue
                trial = w.copy()
                trial[c] = value
                score = ndcg10(cache, trial)
                used += 1
                if score > best + 1e-9:
                    best, w, improved = score, trial, True
                    trace.append(
                        {"eval": used, "ndcg@10": round(best, 4), "weights": dict(zip(CHANNELS, w.round(3).tolist()))}
                    )
        if not quiet:
            print(f"  rodada {rounds}: nDCG@10={best:.4f}  ({used}/{budget} avaliações)")
    return w, best, used, trace


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval.train_fusion", description="Fusão linear treinável (baseline).")
    ap.add_argument("--split", default="dev", help="split de TREINO (default: dev — nunca test)")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="avaliações da função-objetivo")
    ap.add_argument("--dry-run", action="store_true", help="não grava o JSON de parâmetros")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.split == "test":
        raise SystemExit("recusado: treinar no split de teste invalida o próprio baseline (protocolo §6).")

    cache = build_cache(args.split, quiet=args.quiet)
    w, best, used, trace = coordinate_search(cache, args.budget, quiet=args.quiet)

    weights = {name: round(float(v), 4) for name, v in zip(CHANNELS, w)}
    base = ndcg10(cache, np.array([PRODUCTION_WEIGHTS[c] for c in CHANNELS], dtype=np.float64))
    payload = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": _git_commit(),
            "train_split": args.split,
            "n_queries": len(cache),
            "dataset_sha1": dataset_sha1(),
            "budget_evals": args.budget,
            "used_evals": used,
            "pool_per_channel": POOL_PER_CHANNEL,
            "grid": list(GRID),
            "objective": "ndcg@10 (aproximado no pool; ver docstring)",
            "note": "peso lexical FIXO (produção usa adaptativo 0.20–0.30) — modelo deliberadamente mais simples",
        },
        "weights": weights,
        "production_weights": PRODUCTION_WEIGHTS,
        "train_ndcg@10": {"production": round(base, 4), "learned": round(best, 4)},
        "trace": trace,
    }

    print()
    print("| canal | produção | aprendido |")
    print("|---|---:|---:|")
    for name in CHANNELS:
        print(f"| {name} | {PRODUCTION_WEIGHTS[name]:.3f} | {weights[name]:.3f} |")
    print()
    print(f"nDCG@10 no TREINO ({args.split}, n={len(cache)}): produção {base:.4f} → aprendido {best:.4f}")
    print("Esse número é de treino — não reportar. O que vale é `eval.run --split test`.")

    if args.dry_run:
        return 0
    os.makedirs(PARAMS_DIR, exist_ok=True)
    with open(PARAMS_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\n» {os.path.relpath(PARAMS_PATH, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
