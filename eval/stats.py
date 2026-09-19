# -*- coding: utf-8 -*-
"""Inferência estatística sobre os resultados por consulta — **determinístico e sem rede**.

Todo número de `eval/run.py` e `eval/ablation_components.py` é estimativa
pontual sobre 30–95 consultas. Este módulo transforma as diferenças entre
pipelines em **diferença + intervalo de confiança**, que é o formato exigido
pelo protocolo do estudo (`docs/PROTOCOLO-TOIS.md`, §8.3).

    .venv/bin/python -m eval.stats compare --split test
    .venv/bin/python -m eval.stats compare --split test --metric success@1 --baseline fusion
    .venv/bin/python -m eval.stats compare --split test --cluster-by movie
    .venv/bin/python -m eval.stats power   --split test --metric success@1 --delta 0.05
    .venv/bin/python -m eval.stats compare --results caminho.json --out eval/results/stats__x.json

Método
------
**Bootstrap pareado**: as duas condições são medidas nas *mesmas* consultas, então
a reamostragem sorteia **consultas** (com reposição) e recalcula a **média das
diferenças pareadas** dentro de cada reamostra. Isso preserva o pareamento entre
configurações e **incorpora a heterogeneidade entre consultas** à estimativa de
incerteza — não a elimina. O pareamento remove o que é comum às duas condições
numa mesma consulta (dificuldade intrínseca); o que resta, e que entra no
intervalo, é a variação da *diferença* de consulta para consulta.

`--cluster-by movie` reamostra **clusters** (todas as consultas do mesmo filme
saem juntas) — necessário quando há dependência entre consultas, como manda o
protocolo. Sem dependência, cluster = consulta e o resultado é idêntico.

O valor-p é bootstrap bicaudal por inversão: `2 · min(P(Δ*≤0), P(Δ*≥0))`, com
correção de Holm dentro da família de comparações da chamada. Ele **não** é o
desfecho — o intervalo é; o p entra só porque revisor pede.

Semente fixa (`SEED`) e `n_boot` registrados no JSON: a mesma entrada sempre
produz o mesmo intervalo.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Callable, Iterable, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
_ROOT = os.path.dirname(_HERE)

SEED = 20260918
N_BOOT = 10_000
ALPHA = 0.05

# --------------------------------------------------------------- métricas


def _success_at(k: int) -> Callable[[Optional[int]], float]:
    def f(rank: Optional[int]) -> float:
        return 1.0 if (rank is not None and rank <= k) else 0.0

    return f


def _ndcg_at(k: int) -> Callable[[Optional[int]], float]:
    def f(rank: Optional[int]) -> float:
        if rank is None or rank > k:
            return 0.0
        return 1.0 / math.log2(rank + 1)

    return f


def _precision_at(k: int) -> Callable[[Optional[int]], float]:
    def f(rank: Optional[int]) -> float:
        return (1.0 / k) if (rank is not None and rank <= k) else 0.0

    return f


def _rr(rank: Optional[int]) -> float:
    return 1.0 / rank if rank else 0.0


def metric_fn(name: str) -> Callable[[Optional[int]], float]:
    """Nome da métrica -> função posição(1-based|None) -> valor por consulta.

    `success@k` e `recall@k` são a mesma coisa neste regime (um único relevante);
    os dois nomes são aceitos porque o protocolo fala em `Success@1` e o
    `eval/metrics.py` fala em `recall@k`."""
    n = name.strip().lower()
    if n in ("mrr", "rr"):
        return _rr
    if "@" in n:
        head, _, tail = n.partition("@")
        k = int(tail)
        if head in ("success", "recall", "hit", "hits"):
            return _success_at(k)
        if head == "ndcg":
            return _ndcg_at(k)
        if head == "precision":
            return _precision_at(k)
    raise ValueError(f"métrica desconhecida: {name!r} (use success@k, recall@k, ndcg@k, precision@k, mrr)")


# ------------------------------------------------------------- carregamento


def load_ranks(path: str, source: Optional[str] = None) -> tuple[dict[str, dict[str, Optional[int]]], dict]:
    """Lê um JSON de `eval/results/` e devolve ({pipeline: {qid: posição}}, meta).

    Aceita tanto a saída de `eval.run` quanto qualquer JSON com a mesma forma
    (`results[<chave>]["per_query"]` com `qid` e `rank`) — é assim que os
    experimentos com LLM em `experiments/` reaproveitam este módulo.

    `source` restringe às consultas daquela origem (ex.: `v2`, o subconjunto do
    teste não usado na calibração original dos pesos)."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    out: dict[str, dict[str, Optional[int]]] = {}
    for key, block in (payload.get("results") or {}).items():
        per_query = block.get("per_query")
        if not per_query:
            continue
        out[key] = {
            row["qid"]: row.get("rank")
            for row in per_query
            if source is None or str(row.get("source", "")).startswith(source)
        }
    return out, payload.get("run", {})


def _clusters_by_movie(path: str) -> dict[str, str]:
    """qid -> chave de cluster = filme relevante (quase-duplicatas do mesmo alvo
    saem juntas na reamostragem)."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    out: dict[str, str] = {}
    for block in (payload.get("results") or {}).values():
        for row in block.get("per_query") or []:
            out[row["qid"]] = f"m{row.get('relevant_id')}"
    return out


# ------------------------------------------------------------- bootstrap


def paired_bootstrap(
    a: Iterable[float],
    b: Iterable[float],
    clusters: Optional[Iterable[str]] = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    alpha: float = ALPHA,
) -> dict:
    """Diferença `média(b) − média(a)` com IC percentil por bootstrap pareado.

    `a`/`b`: valor da métrica por consulta, **na mesma ordem** nas duas listas.
    `clusters`: rótulo de cluster por consulta; reamostra clusters inteiros.
    """
    av = np.asarray(list(a), dtype=np.float64)
    bv = np.asarray(list(b), dtype=np.float64)
    if av.shape != bv.shape:
        raise ValueError(f"listas de tamanhos diferentes: {av.shape} vs {bv.shape}")
    n = int(av.size)
    if n == 0:
        raise ValueError("nenhuma consulta em comum entre as duas condições")

    diff = bv - av
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)

    if clusters is None:
        idx_groups = [np.array([i]) for i in range(n)]
    else:
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(clusters):
            groups.setdefault(c, []).append(i)
        idx_groups = [np.asarray(v) for v in groups.values()]
    n_groups = len(idx_groups)

    boot = np.empty(n_boot, dtype=np.float64)
    for t in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        sample = np.concatenate([idx_groups[j] for j in pick])
        boot[t] = float(diff[sample].mean())

    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    # p bicaudal por inversão do bootstrap; o piso 1/n_boot evita reportar "p=0".
    p_left = float((boot <= 0).mean())
    p_right = float((boot >= 0).mean())
    p = min(1.0, 2 * min(p_left, p_right))
    p = max(p, 1.0 / n_boot)

    return {
        "n": n,
        "n_clusters": n_groups,
        "mean_a": float(av.mean()),
        "mean_b": float(bv.mean()),
        "delta": observed,
        "ci_low": lo,
        "ci_high": hi,
        "ci_halfwidth": (hi - lo) / 2.0,
        "p_value": p,
        "wins": int((diff > 0).sum()),
        "losses": int((diff < 0).sum()),
        "ties": int((diff == 0).sum()),
        "n_boot": n_boot,
        "seed": seed,
        "alpha": alpha,
    }


def holm(p_values: list[float]) -> list[float]:
    """Correção de Holm–Bonferroni; devolve os p ajustados na ordem de entrada."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p_values[i])
        running = max(running, val)  # monotonicidade
        adjusted[i] = running
    return adjusted


# ------------------------------------------------------------------ potência


def power_curve(
    diff: Iterable[float],
    n_grid: Iterable[int],
    target_delta: float,
    n_sim: int = 2000,
    n_boot: int = 2000,
    seed: int = SEED,
    alpha: float = ALPHA,
) -> list[dict]:
    """Simula: com `n` consultas do mesmo tipo das observadas, qual a meia-largura
    típica do IC e a chance de o IC excluir zero?

    As diferenças por consulta observadas (`diff`) são a população de onde cada
    estudo simulado sorteia. Para estimar potência sob um efeito alvo, a
    distribuição é deslocada para ter média `target_delta` — assim a variância é
    a real e só o efeito é hipotético."""
    d = np.asarray(list(diff), dtype=np.float64)
    if d.size == 0:
        raise ValueError("sem diferenças observadas")
    shifted = d - d.mean() + float(target_delta)
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_grid:
        halfwidths = np.empty(n_sim)
        excludes_zero = 0
        for s in range(n_sim):
            study = rng.choice(shifted, size=n, replace=True)
            # bootstrap interno barato: média de reamostras da própria amostra
            boot_idx = rng.integers(0, n, size=(n_boot, n))
            boot = study[boot_idx].mean(axis=1)
            lo = np.percentile(boot, 100 * alpha / 2)
            hi = np.percentile(boot, 100 * (1 - alpha / 2))
            halfwidths[s] = (hi - lo) / 2.0
            if lo > 0 or hi < 0:
                excludes_zero += 1
        rows.append(
            {
                "n": int(n),
                "target_delta": float(target_delta),
                "median_ci_halfwidth": float(np.median(halfwidths)),
                "mean_ci_halfwidth": float(halfwidths.mean()),
                "power": excludes_zero / n_sim,
                "n_sim": n_sim,
            }
        )
    return rows


# --------------------------------------------------------------------- CLI


def _aligned(
    ranks: dict[str, dict[str, Optional[int]]], a_key: str, b_key: str, fn: Callable[[Optional[int]], float]
) -> tuple[list[float], list[float], list[str]]:
    qids = [q for q in ranks[a_key] if q in ranks[b_key]]
    qids.sort()
    return [fn(ranks[a_key][q]) for q in qids], [fn(ranks[b_key][q]) for q in qids], qids


def _fmt(x: float) -> str:
    return f"{x:+.4f}"


def cmd_compare(args: argparse.Namespace) -> int:
    path = args.results or os.path.join(RESULTS_DIR, f"latest__{args.split}.json")
    ranks, meta = load_ranks(path, source=args.source)
    keys = args.pipelines.split(",") if args.pipelines else list(ranks)
    missing = [k for k in keys if k not in ranks]
    if missing:
        raise SystemExit(f"pipeline ausente no JSON: {', '.join(missing)} (tem: {', '.join(ranks)})")
    baseline = args.baseline if args.baseline in ranks else keys[0]
    fn = metric_fn(args.metric)
    cluster_map = _clusters_by_movie(path) if args.cluster_by == "movie" else None

    comparisons = []
    for key in keys:
        if key == baseline:
            continue
        a, b, qids = _aligned(ranks, baseline, key, fn)
        clusters = [cluster_map[q] for q in qids] if cluster_map else None
        row = paired_bootstrap(a, b, clusters=clusters, n_boot=args.n_boot, seed=args.seed)
        row["baseline"] = baseline
        row["pipeline"] = key
        comparisons.append(row)

    for row, adj in zip(comparisons, holm([r["p_value"] for r in comparisons])):
        row["p_holm"] = adj

    payload = {
        "run": {
            "kind": "paired-bootstrap",
            "source": os.path.relpath(path, _ROOT),
            "source_run": {k: meta.get(k) for k in ("timestamp_utc", "git_commit", "split", "dataset_sha1")},
            "metric": args.metric,
            "baseline": baseline,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "alpha": ALPHA,
            "cluster_by": args.cluster_by,
            "source_filter": args.source,
            "multiplicity": "holm",
        },
        "comparisons": comparisons,
    }

    src = f"  ·  origem: {args.source}" if args.source else ""
    print(f"» {os.path.relpath(path, _ROOT)}  ·  métrica: {args.metric}  ·  base: {baseline}{src}")
    if cluster_map:
        print(f"  reamostragem por cluster ({args.cluster_by})")
    print()
    print(f"| Contraste | {args.metric} base | condição | Δ | IC 95% | p (Holm) | ganha/perde/empata |")
    print("|---|---:|---:|---:|---|---:|---|")
    for r in comparisons:
        print(
            f"| {r['pipeline']} − {baseline} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | "
            f"**{_fmt(r['delta'])}** | [{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}] | "
            f"{r['p_holm']:.3f} | {r['wins']}/{r['losses']}/{r['ties']} |"
        )
    print()
    n = comparisons[0]["n"] if comparisons else 0
    print(f"n = {n} consultas · {args.n_boot} reamostras · semente {args.seed}")
    print("IC que cruza zero = diferença indistinguível do ruído amostral neste n.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\n» JSON: {os.path.relpath(args.out, _ROOT)}")
    return 0


def cmd_power(args: argparse.Namespace) -> int:
    path = args.results or os.path.join(RESULTS_DIR, f"latest__{args.split}.json")
    ranks, _ = load_ranks(path)
    fn = metric_fn(args.metric)
    a, b, _ = _aligned(ranks, args.baseline, args.pipeline, fn)
    diff = [y - x for x, y in zip(a, b)]
    grid = [int(x) for x in args.n_grid.split(",")]
    rows = power_curve(diff, grid, args.delta, n_sim=args.n_sim, n_boot=args.n_boot, seed=args.seed)

    print(f"» potência simulada · métrica {args.metric} · efeito alvo Δ={args.delta:+.3f}")
    print(f"  variância vinda de {args.pipeline} − {args.baseline} em {os.path.relpath(path, _ROOT)} (n={len(diff)})")
    print()
    print("| n consultas | meia-largura mediana do IC 95% | potência (IC exclui 0) |")
    print("|---:|---:|---:|")
    for r in rows:
        print(f"| {r['n']} | ±{r['median_ci_halfwidth']:.3f} | {r['power']:.2f} |")
    print()
    print("Leitura: o n necessário é o primeiro em que a meia-largura cabe na precisão")
    print("declarada no protocolo (§8.3), não o primeiro com potência alta.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"run": {"kind": "power", "metric": args.metric, "seed": args.seed}, "grid": rows}, fh, indent=2)
        print(f"\n» JSON: {os.path.relpath(args.out, _ROOT)}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="eval.stats", description="Bootstrap pareado sobre resultados por consulta.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="diferença + IC 95% de cada pipeline contra a base")
    c.add_argument("--split", default="test")
    c.add_argument("--results", help="JSON de resultados (default: eval/results/latest__<split>.json)")
    c.add_argument("--pipelines", help="lista separada por vírgula (default: todos os do JSON)")
    c.add_argument("--baseline", default="fusion", help="condição de referência (default: fusion)")
    c.add_argument("--metric", default="ndcg@10", help="success@1 | ndcg@10 | mrr | recall@k …")
    c.add_argument("--cluster-by", choices=["none", "movie"], default="none")
    c.add_argument("--source", help="restringe a uma origem de consulta (ex.: v2)")
    c.add_argument("--n-boot", type=int, default=N_BOOT)
    c.add_argument("--seed", type=int, default=SEED)
    c.add_argument("--out", help="grava o resultado em JSON")
    c.set_defaults(func=cmd_compare)

    p = sub.add_parser("power", help="quantas consultas para a precisão declarada no protocolo")
    p.add_argument("--split", default="test")
    p.add_argument("--results")
    p.add_argument("--baseline", default="fusion")
    p.add_argument("--pipeline", default="fusion_rerank", help="de onde vem a variância observada")
    p.add_argument("--metric", default="success@1")
    p.add_argument("--delta", type=float, default=0.05, help="efeito alvo a detectar")
    p.add_argument("--n-grid", default="47,100,200,300,500,800")
    p.add_argument("--n-sim", type=int, default=500)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out")
    p.set_defaults(func=cmd_power)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
