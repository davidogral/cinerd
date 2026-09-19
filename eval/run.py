# -*- coding: utf-8 -*-
"""Avaliação executável do SRI — substitui o notebook como fonte de verdade.

    .venv/bin/python -m eval.run                     # split de teste, 8 pipelines
    .venv/bin/python -m eval.run --split dev
    .venv/bin/python -m eval.run --fast              # pula o re-ranker (rápido)
    .venv/bin/python -m eval.run --pipelines fusion,fusion_rerank
    .venv/bin/python -m eval.run --split all --out eval/results/full.json

Roda cada variante do pipeline (`eval/pipelines.py`) sobre as consultas do split
escolhido, mede posição do filme relevante + latência, agrega as métricas
(`eval/metrics.py`) e:

1. imprime a **tabela de ablação** em Markdown (a que vai no README);
2. grava um **JSON versionado** em `eval/results/`:
   - `AAAA-MM-DDTHH-MM-SSZ__<split>.json`  (registro imutável da rodada)
   - `latest__<split>.json`                (ponteiro para a última)
   - acrescenta 1 linha-resumo em `history.jsonl` (evolução semana a semana).

Determinismo: o fallback de nome via TMDB é desligado (`RECOMENDAI_TMDB_NAMES=0`)
para a rodada não depender de rede; os pesos/limiares vêm do `search_engine.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

# Rodada reprodutível e offline: sem ida à TMDB para resolver título.
os.environ.setdefault("RECOMENDAI_TMDB_NAMES", "0")

from eval import metrics as M
from eval.dataset import QUERIES_PATH, dataset_sha1, load_queries, split_counts
from eval.pipelines import NEEDS_EMBEDDINGS, PIPELINE_LABELS, PIPELINES, make_ctx, rank_of

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
_ROOT = os.path.dirname(_HERE)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _engine_meta() -> dict:
    """Identidade do índice usado na rodada.

    Além dos metadados declarados (`meta.json`), grava a **impressão digital**
    dos artefatos — SHA-256 por arquivo e um hash combinado. O índice não é
    redistribuível (DVC com credencial privada + licença da fonte), então essa
    impressão é o que permite a um terceiro reconstruir o índice e **verificar**
    se chegou ao mesmo artefato. Ver `eval/fingerprint.py` e
    `docs/PROTOCOLO-TOIS.md` §11.1."""
    from eval.fingerprint import fingerprint

    out: dict = {}
    p = os.path.join(_ROOT, "retrieval", "index", "meta.json")
    try:
        with open(p, encoding="utf-8") as fh:
            m = json.load(fh)
        out = {k: m[k] for k in ("n_movies", "embed_model", "embed_dim", "lexical") if k in m}
    except Exception:
        out = {}
    try:
        index_dir = os.environ.get("RECOMENDAI_INDEX_DIR") or os.path.join(_ROOT, "retrieval", "index")
        if not os.path.isabs(index_dir):
            index_dir = os.path.join(_ROOT, index_dir)
        fp = fingerprint(index_dir)
        if fp:
            out["index_fingerprint"] = {
                "dir": fp["dir"],
                "n_files": fp["n_files"],
                "total_bytes": fp["total_bytes"],
                "combined_sha256": fp["combined_sha256"],
                "files": fp["files"],
            }
    except Exception:
        pass
    return out


def _env_snapshot() -> dict:
    from retrieval.search_engine import (
        DEFAULT_EMBED_WEIGHT,
        DEFAULT_KEYWORD_WEIGHT,
        DEFAULT_POP_PRIOR,
        RERANK_BLEND,
        RERANK_POOL,
    )

    return {
        "embed_weight": DEFAULT_EMBED_WEIGHT,
        "keyword_weight": DEFAULT_KEYWORD_WEIGHT,
        "lexical_weight": "adaptativo (0.20–0.30)",
        "pop_prior": DEFAULT_POP_PRIOR,
        "rerank_pool": RERANK_POOL,
        "rerank_blend": RERANK_BLEND,
        "intent_thresholds": [0.92, 0.85],
        "tmdb_name_fallback": os.environ.get("RECOMENDAI_TMDB_NAMES") not in ("0", "false", "no"),
        "query_slm": os.environ.get("RECOMENDAI_QUERY_SLM", "0") not in ("0", "false", "no"),
    }


def evaluate(split: str, pipeline_keys: list[str], limit: int | None, quiet: bool) -> dict:
    from retrieval.search_engine import SearchEngine

    queries = load_queries(split)
    if limit:
        queries = queries[:limit]
    if not quiet:
        print("» carregando motor de busca…")
    # rerank=True SEMPRE na avaliação: medimos o cross-encoder como variante da
    # ablação, independente de ele estar ligado por padrão em produção
    # (RERANK_ENABLED, hoje off — ver comentário em search_engine.py).
    engine = SearchEngine(rerank=True)
    if not engine.has_synopsis_index:
        sys.exit("índice de sinopse ausente — rode `python -m retrieval.index_builder`")

    # Um contexto (consulta limpa + embedding) por consulta, reaproveitado.
    ctxs = {q.qid: make_ctx(engine, q.query) for q in queries}

    results: dict = {}
    for key in pipeline_keys:
        fn = PIPELINES[key]
        if key in NEEDS_EMBEDDINGS and engine._embeddings is None:
            if not quiet:
                print(f"  · pulando {key} (sem embeddings)")
            continue
        if not quiet:
            print(f"  · {key} … ", end="", flush=True)
        ranks, times_ms, per_query = [], [], []
        for q in queries:
            t0 = time.perf_counter()
            ranked_ids = fn(engine, ctxs[q.qid])
            dt = (time.perf_counter() - t0) * 1000.0
            pos = rank_of(ranked_ids, q.relevant_id)
            ranks.append(pos)
            times_ms.append(dt)
            per_query.append(
                {
                    "qid": q.qid,
                    "source": q.source,
                    "title": q.relevant_title,
                    "relevant_id": q.relevant_id,
                    "rank": pos,
                    "rr": round(M.reciprocal_rank(pos), 4),
                }
            )
        agg = M.aggregate(ranks)
        results[key] = {
            "metrics": agg,
            "latency_ms": M.latency_stats(times_ms),
            "per_query": per_query,
        }
        if not quiet:
            print(
                f"MRR={agg['mrr']:.3f}  nDCG@10={agg['ndcg@10']:.3f}  "
                f"R@50={agg['recall@50']:.3f}  ({agg['found']}/{agg['n']})"
            )

    return {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": _git_commit(),
            "split": split,
            "n_queries": len(queries),
            "dataset_file": os.path.relpath(QUERIES_PATH, _ROOT),
            "dataset_sha1": dataset_sha1(),
            "split_counts": split_counts(),
            "engine": _engine_meta(),
            "config": _env_snapshot(),
            "pipelines": pipeline_keys,
        },
        "results": results,
    }


# ---------------------------------------------------------------------- saída
_COLS = [
    ("ndcg@10", "nDCG@10"),
    ("mrr", "MRR"),
    ("recall@10", "R@10"),
    ("recall@50", "R@50"),
    ("precision@10", "P@10"),
]


def render_table(payload: dict) -> str:
    res = payload["results"]
    lines = [
        "| Pipeline | " + " | ".join(h for _, h in _COLS) + " | mediana | lat. p50 |",
        "|" + "---|" * (len(_COLS) + 3),
    ]
    for key, label in PIPELINE_LABELS:
        if key not in res:
            continue
        m = res[key]["metrics"]
        lat = res[key]["latency_ms"]
        med = f"#{m['median_rank']}" if m["median_rank"] is not None else "—"
        p50 = f"{lat['p50_ms']:.0f} ms" if lat["p50_ms"] is not None else "—"
        cells = " | ".join(f"{m[k]:.3f}" for k, _ in _COLS)
        lines.append(f"| {label} | {cells} | {med} | {p50} |")
    return "\n".join(lines)


def _write_outputs(payload: dict, out: str | None, quiet: bool) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    split = payload["run"]["split"]
    stamp = payload["run"]["timestamp_utc"].replace(":", "-")
    canonical = out or os.path.join(RESULTS_DIR, f"{stamp}__{split}.json")
    with open(canonical, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    latest = os.path.join(RESULTS_DIR, f"latest__{split}.json")
    with open(latest, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    # history.jsonl: 1 linha-resumo por rodada (sem per_query).
    row = {
        "timestamp_utc": payload["run"]["timestamp_utc"],
        "git_commit": payload["run"]["git_commit"],
        "split": split,
        "n_queries": payload["run"]["n_queries"],
        "dataset_sha1": payload["run"]["dataset_sha1"][:12],
    }
    for key in payload["run"]["pipelines"]:
        if key in payload["results"]:
            m = payload["results"][key]["metrics"]
            row[key] = {mk: m[mk] for mk in ("mrr", "ndcg@10", "recall@10", "recall@50", "precision@10", "median_rank")}
    with open(os.path.join(RESULTS_DIR, "history.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not quiet:
        print(f"\n» JSON: {os.path.relpath(canonical, _ROOT)}")
        print(f"        {os.path.relpath(latest, _ROOT)}  (ponteiro)")
        print(f"        {os.path.relpath(os.path.join(RESULTS_DIR, 'history.jsonl'), _ROOT)}  (+1 linha)")


# ------------------------------------------------------- varredura do re-ranker
def sweep_rerank(split: str, pools: list[int], limit: int | None, quiet: bool) -> dict:
    """Fusão + cross-encoder em vários tamanhos de pool (0 = re-ranker desligado).
    Mede a curva **qualidade × latência** — a base da decisão de produção."""
    from retrieval.search_engine import SearchEngine

    queries = load_queries(split)
    if limit:
        queries = queries[:limit]
    if not quiet:
        print("» carregando motor de busca…")
    # rerank=True SEMPRE na avaliação: medimos o cross-encoder como variante da
    # ablação, independente de ele estar ligado por padrão em produção
    # (RERANK_ENABLED, hoje off — ver comentário em search_engine.py).
    engine = SearchEngine(rerank=True)
    ctxs = {q.qid: make_ctx(engine, q.query) for q in queries}
    fusion_only = PIPELINES["fusion"]

    rows = []
    for pool in pools:
        if not quiet:
            print(f"  · pool={(pool or 'off')!s:<4} … ", end="", flush=True)
        ranks, times_ms = [], []
        for q in queries:
            ctx = ctxs[q.qid]
            t0 = time.perf_counter()
            if pool <= 0:
                ids = fusion_only(engine, ctx)
            else:
                ids = engine.synopsis_ranked_ids(ctx.raw, rerank=True, pool=pool)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            ranks.append(rank_of(ids, q.relevant_id))
        agg, lat = M.aggregate(ranks), M.latency_stats(times_ms)
        rows.append({"pool": pool, "metrics": agg, "latency_ms": lat})
        if not quiet:
            print(
                f"nDCG@10={agg['ndcg@10']:.3f}  MRR={agg['mrr']:.3f}  "
                f"R@50={agg['recall@50']:.3f}  p50={lat['p50_ms']:.0f}ms"
            )

    return {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": _git_commit(),
            "mode": "sweep_rerank",
            "split": split,
            "n_queries": len(queries),
            "dataset_sha1": dataset_sha1(),
            "pools": pools,
            "engine": _engine_meta(),
            "config": _env_snapshot(),
        },
        "sweep": rows,
    }


def render_sweep(payload: dict) -> str:
    lines = [
        "| pool cross-encoder | nDCG@10 | MRR | R@10 | R@50 | mediana | lat. p50 | lat. p90 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in payload["sweep"]:
        m, lat = r["metrics"], r["latency_ms"]
        med = f"#{m['median_rank']}" if m["median_rank"] is not None else "—"
        label = "0 (desligado)" if r["pool"] <= 0 else str(r["pool"])
        lines.append(
            f"| {label} | {m['ndcg@10']:.3f} | {m['mrr']:.3f} | {m['recall@10']:.3f} "
            f"| {m['recall@50']:.3f} | {med} | {lat['p50_ms']:.0f} ms | {lat['p90_ms']:.0f} ms |"
        )
    return "\n".join(lines)


def _write_sweep(payload: dict, out: str | None, quiet: bool) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    split = payload["run"]["split"]
    stamp = payload["run"]["timestamp_utc"].replace(":", "-")
    canonical = out or os.path.join(RESULTS_DIR, f"{stamp}__sweep-rerank-{split}.json")
    for path in (canonical, os.path.join(RESULTS_DIR, f"latest__sweep-rerank-{split}.json")):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    if not quiet:
        print(f"\n» JSON: {os.path.relpath(canonical, _ROOT)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Avaliação do SRI (ablação por sinal).")
    ap.add_argument(
        "--split",
        default="test",
        choices=["test", "dev", "hard", "entity", "object", "all"],
        help="qual split avaliar (padrão: test — o único reportado).",
    )
    ap.add_argument(
        "--pipelines", default=None, help=f"lista separada por vírgula; padrão: todos ({', '.join(PIPELINES)})."
    )
    ap.add_argument("--fast", action="store_true", help="pula fusion_rerank (não carrega o cross-encoder).")
    ap.add_argument(
        "--sweep-rerank",
        action="store_true",
        help="varre o tamanho do pool do cross-encoder (curva qualidade × latência).",
    )
    ap.add_argument(
        "--pools", default="0,10,20,50,100,300", help="tamanhos de pool para --sweep-rerank (0 = desligado)."
    )
    ap.add_argument("--limit", type=int, default=None, help="só as N primeiras consultas.")
    ap.add_argument("--out", default=None, help="caminho do JSON canônico.")
    ap.add_argument("--no-write", action="store_true", help="não grava nada em disco.")
    ap.add_argument(
        "--gate-ndcg",
        type=float,
        default=None,
        help="PORTÃO DE CI: sai com código 1 se o nDCG@10 da fusão ficar abaixo deste valor.",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.sweep_rerank:
        pools = sorted({max(0, int(x)) for x in args.pools.split(",")})
        payload = sweep_rerank(args.split, pools, args.limit, args.quiet)
        print(f"\n### Varredura do re-ranker — split `{args.split}` ({payload['run']['n_queries']} consultas)\n")
        print(render_sweep(payload))
        print()
        if not args.no_write:
            _write_sweep(payload, args.out, args.quiet)
        return

    keys = [k.strip() for k in args.pipelines.split(",")] if args.pipelines else list(PIPELINES)
    for k in keys:
        if k not in PIPELINES:
            ap.error(f"pipeline desconhecido: {k!r} (opções: {', '.join(PIPELINES)})")
    if args.fast:
        keys = [k for k in keys if k != "fusion_rerank"]

    payload = evaluate(args.split, keys, args.limit, args.quiet)

    print(f"\n### Ablação por sinal — split `{args.split}` ({payload['run']['n_queries']} consultas)\n")
    print(render_table(payload))
    print()

    if not args.no_write:
        _write_outputs(payload, args.out, args.quiet)

    if args.gate_ndcg is not None:
        got = payload["results"].get("fusion", {}).get("metrics", {}).get("ndcg@10")
        if got is None:
            sys.exit("PORTÃO: pipeline 'fusion' não avaliado — não dá para checar o limiar.")
        status = "OK" if got >= args.gate_ndcg else "FALHOU"
        print(f"PORTÃO nDCG@10 (fusão, split {args.split}): {got:.4f} >= {args.gate_ndcg:.4f} ? {status}")
        if got < args.gate_ndcg:
            sys.exit(1)


if __name__ == "__main__":
    main()
