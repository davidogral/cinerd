# -*- coding: utf-8 -*-
"""Julgamento humano em *pool* cego, com dois anotadores (protocolo §5).

    # 1. monta o pool a partir dos candidatos de TODOS os métodos comparados
    .venv/bin/python -m experiments.annotate pool --queries experiments/data/prospective_raw.jsonl \\
        --results eval/results/latest__test.json experiments/results/<...>__factorial-hard.json --k 10

    # 2. cada anotador roda a sua tarefa (cega, embaralhada)
    .venv/bin/python -m experiments.annotate run --task experiments/data/pool__ana.jsonl

    # 3. concordância entre os dois
    .venv/bin/python -m experiments.annotate agree --a ...__ana.jsonl --b ...__bruno.jsonl

    # 4. conflitos para o terceiro juiz, e o padrão de verdade final
    .venv/bin/python -m experiments.annotate adjudicate --a ... --b ... --out experiments/data/gold.jsonl
    .venv/bin/python -m experiments.annotate export --gold experiments/data/gold.jsonl

Três decisões do protocolo que este arquivo implementa literalmente:

1. **Pool pela união** dos candidatos de todos os métodos comparados. Julgar só
   o topo de um método premia esse método: o que ele não recupera nunca é
   julgado relevante e some da conta.
2. **Cego e embaralhado.** O anotador não vê de que método veio o candidato, nem
   a posição que ele ocupava.
3. **Cobertura da fonte é rótulo próprio.** "O fato não está na sinopse" não é
   "o fato é falso no filme" — é a diferença entre erro de evidência e erro de
   julgamento, e o estudo inteiro depende de separar os dois.

Relevância graduada: `2` é o filme lembrado, `1` é resposta defensável (mesma
franquia, premissa quase idêntica), `0` não é.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone
from typing import Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(_HERE, "data")

GRADES = {"2": "é o filme lembrado", "1": "resposta defensável", "0": "não é"}
COVERAGE = {"s": "o fato citado ESTÁ no texto", "n": "não está no texto", "p": "parcialmente"}
POOL_SEED = 20260918


# ------------------------------------------------------------------ pool


def _load_queries(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    raise SystemExit("--queries espera um .jsonl (saída de experiments.collect export)")


def _candidates_from_results(paths: Sequence[str], k: int) -> dict[str, list[int]]:
    """qid -> união dos top-k de cada método presente nos JSONs de resultado.

    Os JSONs de `eval.run` guardam a posição do alvo, não a lista inteira; quando
    a lista não estiver disponível, o que entra é o alvo conhecido + os candidatos
    que o fatorial registrou. O pool é sempre a UNIÃO — nunca o topo de um só."""
    pools: dict[str, set] = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        for block in (payload.get("results") or {}).values():
            for row in block.get("per_query") or []:
                qid = row["qid"]
                bucket = pools.setdefault(qid, set())
                if row.get("relevant_id"):
                    bucket.add(int(row["relevant_id"]))
                for tid in (row.get("picks") or [])[:k]:
                    bucket.add(int(tid))
                for tid in (row.get("candidates") or [])[:k]:
                    bucket.add(int(tid))
    return {qid: sorted(ids) for qid, ids in pools.items()}


def cmd_pool(args: argparse.Namespace) -> int:
    from core import catalog

    queries = _load_queries(args.queries)
    extra = _candidates_from_results(args.results or [], args.k)
    rng = random.Random(POOL_SEED)

    tasks: list[dict] = []
    for q in queries:
        ids = set(extra.get(q["qid"], []))
        if q.get("clicked_item_id"):
            ids.add(int(q["clicked_item_id"]))
        if not ids:
            continue
        items = []
        for tid in sorted(ids):
            mv = catalog.get_movie(int(tid)) or {}
            items.append(
                {
                    "tmdb_id": int(tid),
                    "title": mv.get("title"),
                    "year": str(mv.get("release_date") or "")[:4] or None,
                    "overview": mv.get("overview") or "",
                }
            )
        rng.shuffle(items)  # cego: sem ordem de método, sem posição original
        tasks.append({"qid": q["qid"], "query": q["query"], "candidates": items})

    os.makedirs(DATA_DIR, exist_ok=True)
    for name in args.annotators:
        path = os.path.join(DATA_DIR, f"pool__{name}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for t in tasks:
                fh.write(json.dumps({**t, "annotator": name, "judgments": {}}, ensure_ascii=False) + "\n")
        print(f"» {os.path.relpath(path, _ROOT)}  ({len(tasks)} consultas)")
    n_cands = sum(len(t["candidates"]) for t in tasks)
    print(f"» {n_cands} julgamentos por anotador (média {n_cands / max(1, len(tasks)):.1f} candidatos/consulta)")
    print("Guia do anotador: docs/PROTOCOLO-ANOTACAO.md")
    return 0


# -------------------------------------------------------------------- run


def cmd_run(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in open(args.task, encoding="utf-8") if line.strip()]
    todo = [r for r in rows if len(r.get("judgments") or {}) < len(r["candidates"])]
    print(f"» {len(todo)} consultas pendentes de {len(rows)} · anotador: {rows[0].get('annotator')}")
    print("   graus: " + " · ".join(f"[{k}] {v}" for k, v in GRADES.items()))
    print("   cobertura: " + " · ".join(f"[{k}] {v}" for k, v in COVERAGE.items()))
    print("   [s] pula a consulta · [q] salva e sai\n")

    try:
        for r in todo:
            print("=" * 72)
            print(f"CONSULTA: {r['query']}")
            for c in r["candidates"]:
                key = str(c["tmdb_id"])
                if key in (r.get("judgments") or {}):
                    continue
                print(f"\n  {c['title']} ({c['year']})")
                print(f"  {(c['overview'] or '(sem sinopse)')[:400]}")
                grade = input("  relevância [2/1/0/s/q]: ").strip().lower()
                if grade == "q":
                    raise KeyboardInterrupt
                if grade == "s":
                    break
                if grade not in GRADES:
                    print("  (valor inválido, pulado)")
                    continue
                cov = input("  o fato citado está no texto? [s/n/p]: ").strip().lower()
                r.setdefault("judgments", {})[key] = {
                    "grade": int(grade),
                    "coverage": cov if cov in COVERAGE else "p",
                    "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
    except (KeyboardInterrupt, EOFError):
        print("\n» interrompido — progresso salvo")
    finally:
        with open(args.task, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        done = sum(len(r.get("judgments") or {}) for r in rows)
        total = sum(len(r["candidates"]) for r in rows)
        print(f"» {done}/{total} julgamentos em {os.path.relpath(args.task, _ROOT)}")
    return 0


# ------------------------------------------------------------------ agree


def _pairs(a_rows: list[dict], b_rows: list[dict], field: str) -> list[tuple]:
    b_by_qid = {r["qid"]: r for r in b_rows}
    out = []
    for ra in a_rows:
        rb = b_by_qid.get(ra["qid"])
        if not rb:
            continue
        ja, jb = ra.get("judgments") or {}, rb.get("judgments") or {}
        for tid in set(ja) & set(jb):
            out.append((ra["qid"], tid, ja[tid][field], jb[tid][field]))
    return out


def cohen_kappa(pairs: list[tuple], weighted: bool = False) -> Optional[float]:
    """κ de Cohen; `weighted=True` usa peso quadrático (para relevância graduada)."""
    if not pairs:
        return None
    a = [p[2] for p in pairs]
    b = [p[3] for p in pairs]
    labels = sorted({*a, *b}, key=str)
    idx = {v: i for i, v in enumerate(labels)}
    n, k = len(pairs), len(labels)
    if k < 2:
        return 1.0
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1

    def w(i: int, j: int) -> float:
        if not weighted:
            return 0.0 if i == j else 1.0
        return ((i - j) / (k - 1)) ** 2

    ra = [sum(obs[i]) for i in range(k)]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    po = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    pe = sum(w(i, j) * ra[i] * cb[j] for i in range(k) for j in range(k)) / (n * n)
    if pe == 0:
        return 1.0
    return round(1.0 - po / pe, 4)


def cmd_agree(args: argparse.Namespace) -> int:
    a_rows = [json.loads(x) for x in open(args.a, encoding="utf-8") if x.strip()]
    b_rows = [json.loads(x) for x in open(args.b, encoding="utf-8") if x.strip()]
    grade_pairs = _pairs(a_rows, b_rows, "grade")
    cov_pairs = _pairs(a_rows, b_rows, "coverage")
    if not grade_pairs:
        raise SystemExit("nenhum julgamento em comum ainda")

    exact = sum(1 for p in grade_pairs if p[2] == p[3]) / len(grade_pairs)
    print(f"» {len(grade_pairs)} julgamentos em comum")
    print(f"   concordância bruta (relevância): {exact:.3f}")
    print(f"   κ de Cohen (relevância, nominal): {cohen_kappa(grade_pairs)}")
    print(f"   κ ponderado (relevância graduada): {cohen_kappa(grade_pairs, weighted=True)}")
    print(f"   κ de Cohen (cobertura da fonte):   {cohen_kappa(cov_pairs)}")
    disc = [p for p in grade_pairs if p[2] != p[3]]
    print(f"\n   {len(disc)} discordâncias; distribuição: {Counter((p[2], p[3]) for p in disc).most_common(5)}")
    print("   Exemplos (vão para o artigo, protocolo §5):")
    for p in disc[:5]:
        print(f"     {p[0]} · filme {p[1]}: A={p[2]} B={p[3]}")
    return 0


# ------------------------------------------------------------- adjudicate


def cmd_adjudicate(args: argparse.Namespace) -> int:
    a_rows = [json.loads(x) for x in open(args.a, encoding="utf-8") if x.strip()]
    b_rows = [json.loads(x) for x in open(args.b, encoding="utf-8") if x.strip()]
    b_by_qid = {r["qid"]: r for r in b_rows}

    gold, conflicts = [], 0
    for ra in a_rows:
        rb = b_by_qid.get(ra["qid"])
        if not rb:
            continue
        ja, jb = ra.get("judgments") or {}, rb.get("judgments") or {}
        merged = {}
        for tid in set(ja) | set(jb):
            va, vb = ja.get(tid), jb.get(tid)
            if va and vb and va["grade"] == vb["grade"]:
                merged[tid] = {"grade": va["grade"], "coverage": va["coverage"], "source": "consenso"}
            elif va and vb:
                conflicts += 1
                print(f"\nCONFLITO · {ra['query']}")
                cand = next((c for c in ra["candidates"] if str(c["tmdb_id"]) == tid), {})
                print(f"  {cand.get('title')} ({cand.get('year')}): A={va['grade']} B={vb['grade']}")
                if args.auto:
                    # regra pré-definida: conservadora, fica com o MENOR grau
                    g = min(va["grade"], vb["grade"])
                    merged[tid] = {"grade": g, "coverage": va["coverage"], "source": "regra-conservadora"}
                else:
                    print(f"  {(cand.get('overview') or '')[:300]}")
                    g = input("  terceiro juiz [2/1/0]: ").strip()
                    merged[tid] = {
                        "grade": int(g) if g in GRADES else min(va["grade"], vb["grade"]),
                        "coverage": va["coverage"],
                        "source": "terceiro-juiz",
                    }
            else:
                one = va or vb
                merged[tid] = {"grade": one["grade"], "coverage": one["coverage"], "source": "um-anotador"}
        targets = sorted((int(t) for t, v in merged.items() if v["grade"] == 2))
        gold.append({"qid": ra["qid"], "query": ra["query"], "judgments": merged, "targets": targets})

    with open(args.out, "w", encoding="utf-8") as fh:
        for row in gold:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_multi = sum(1 for g in gold if len(g["targets"]) > 1)
    n_none = sum(1 for g in gold if not g["targets"])
    print(f"\n» {len(gold)} consultas · {conflicts} conflitos adjudicados")
    print(f"   {n_multi} com MAIS DE UM filme de grau 2 (o protocolo prevê; relevância graduada cobre)")
    print(f"   {n_none} sem nenhum grau 2 — item ausente do catálogo ou consulta fora da tarefa (§4)")
    print(f"» {os.path.relpath(args.out, _ROOT)}")
    return 0


# ----------------------------------------------------------------- export


def cmd_export(args: argparse.Namespace) -> int:
    """Padrão de verdade final -> formato que `eval.dataset` já lê."""
    gold = [json.loads(x) for x in open(args.gold, encoding="utf-8") if x.strip()]
    out_path = args.out or os.path.join(DATA_DIR, "prospective_queries.jsonl")
    n = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for g in gold:
            if not g["targets"]:
                continue
            fh.write(
                json.dumps(
                    {
                        "qid": g["qid"],
                        "split": g.get("partition", "prospectivo"),
                        "query": g["query"],
                        "title_hint": "",
                        "year": 0,
                        "relevant_tmdb_id": g["targets"][0],
                        "relevant_title": "",
                        "source": "prospectivo",
                        "all_targets": g["targets"],
                        "graded": {t: v["grade"] for t, v in g["judgments"].items() if v["grade"] > 0},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    print(f"» {n} consultas com alvo → {os.path.relpath(out_path, _ROOT)}")
    print("   `relevant_tmdb_id` é o primeiro alvo de grau 2; `all_targets`/`graded` preservam o resto.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.annotate", description="Julgamento humano em pool cego.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pool", help="monta as tarefas de anotação (união dos métodos, embaralhada)")
    p.add_argument("--queries", required=True)
    p.add_argument("--results", nargs="*", help="JSONs de eval.run / experiments.factorial")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--annotators", nargs="+", default=["ana", "bruno"])
    p.set_defaults(func=cmd_pool)

    r = sub.add_parser("run", help="tarefa interativa de um anotador")
    r.add_argument("--task", required=True)
    r.set_defaults(func=cmd_run)

    g = sub.add_parser("agree", help="concordância entre dois anotadores")
    g.add_argument("--a", required=True)
    g.add_argument("--b", required=True)
    g.set_defaults(func=cmd_agree)

    d = sub.add_parser("adjudicate", help="resolve conflitos e grava o padrão de verdade")
    d.add_argument("--a", required=True)
    d.add_argument("--b", required=True)
    d.add_argument("--out", default=os.path.join(DATA_DIR, "gold.jsonl"))
    d.add_argument("--auto", action="store_true", help="usa a regra conservadora em vez do terceiro juiz")
    d.set_defaults(func=cmd_adjudicate)

    x = sub.add_parser("export", help="padrão de verdade -> conjunto avaliável")
    x.add_argument("--gold", default=os.path.join(DATA_DIR, "gold.jsonl"))
    x.add_argument("--out")
    x.set_defaults(func=cmd_export)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
