# -*- coding: utf-8 -*-
"""Variabilidade do verificador entre execuções idênticas (protocolo §7.3, RQ3).

    .venv/bin/python -m experiments.variability experiments/results/<...>__factorial-hard.json
    .venv/bin/python -m experiments.variability A.json B.json --out experiments/results/variabilidade.json

Lê o bloco `repeats` gravado por `experiments/factorial.py --repeats R` e
responde as quatro perguntas que o protocolo §7.3 exige, nesta ordem:

1. **Taxa de mudança de decisão** — com a mesma consulta, os mesmos candidatos e
   temperatura zero, o conjunto confirmado muda entre execuções?
2. **Impacto no ranking** — quando muda, muda a posição do alvo? Quantas
   consultas trocam de `Success@1` só por serem repetidas?
3. **Falhas de API** — quantas chamadas caíram em cada passada.
4. **Sensibilidade à ordem dos candidatos** — o mesmo, quando a única coisa que
   muda é a ordem de apresentação (`factorial.py --ordem embaralhada`).

Por que isto não roda no provedor local
---------------------------------------
Geração gulosa (temperatura zero) com pesos fixos é **determinística**: repetir
cinco vezes devolve cinco respostas idênticas por construção, e a taxa de
mudança medida seria zero por um motivo que não é sobre o modelo. O não
determinismo que interessa aqui é o do **serviço hospedado** — lote, escalonamento
e versão que muda sem aviso — então as repetições de ordem fixa só têm sentido
no provedor de produção.

A sensibilidade à **ordem** é diferente: é uma propriedade do modelo, não do
serviço, e vale nos dois provedores. Rodar a variante embaralhada no local é
legítimo e não gasta cota.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _jaccard(a: Sequence, b: Sequence) -> float:
    sa, sb = {int(x) for x in a}, {int(x) for x in b}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def passes(payload: dict) -> list[dict]:
    """Passadas comparáveis: a original (repeat 0) e cada repetição sem cache."""
    out = []
    base = (payload.get("results") or {}).get("C11")
    if base:
        out.append({"repeat": 0, "cached": True, "per_query": base["per_query"], "metrics": base["metrics"]})
    for rep in payload.get("repeats") or []:
        out.append({"repeat": rep["repeat"], "cached": False, "per_query": rep["per_query"], "metrics": rep["metrics"]})
    return out


def ledger_failures(payload: dict) -> dict:
    """Falhas por passada, lidas do ledger da própria execução (§7.3)."""
    rel = ((payload.get("run") or {}).get("llm") or {}).get("ledger")
    path = os.path.join(_ROOT, rel) if rel else ""
    by_rep: dict = defaultdict(lambda: {"calls": 0, "failed": 0})
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("stage") != "confirm" or r.get("from_cache"):
                continue
            slot = by_rep[int(r.get("repeat") or 0)]
            slot["calls"] += 1
            if not r.get("ok"):
                slot["failed"] += 1
    return {
        k: {**v, "failure_rate": round(v["failed"] / v["calls"], 4) if v["calls"] else 0.0} for k, v in by_rep.items()
    }


def analyse(payload: dict) -> dict:
    ps = passes(payload)
    run = payload.get("run") or {}
    if len(ps) < 2:
        raise SystemExit(
            f"{run.get('run_id', '?')}: só há {len(ps)} passada — rode `factorial.py --repeats R` (R ≥ 5, §7.3)."
        )
    # só as consultas presentes em TODAS as passadas
    common = set.intersection(*[{r["qid"] for r in p["per_query"]} for p in ps])
    picks_by = {p["repeat"]: {r["qid"]: r.get("picks") or [] for r in p["per_query"]} for p in ps}
    rank_by = {p["repeat"]: {r["qid"]: r.get("rank") for r in p["per_query"]} for p in ps}
    reps = sorted(picks_by)

    changed_dec, changed_rank, changed_s1, jaccs = [], [], [], []
    per_query = []
    for qid in sorted(common):
        sets = [tuple(sorted(int(x) for x in picks_by[r][qid])) for r in reps]
        ranks = [rank_by[r][qid] for r in reps]
        s1 = {1 if x == 1 else 0 for x in ranks}
        pair_j = [
            _jaccard(picks_by[reps[i]][qid], picks_by[reps[j]][qid])
            for i in range(len(reps))
            for j in range(i + 1, len(reps))
        ]
        dec = len(set(sets)) > 1
        rk = len(set(ranks)) > 1
        changed_dec.append(dec)
        changed_rank.append(rk)
        changed_s1.append(len(s1) > 1)
        jaccs.append(float(np.mean(pair_j)) if pair_j else 1.0)
        per_query.append(
            {
                "qid": qid,
                "decisao_mudou": dec,
                "posicao_mudou": rk,
                "success1_mudou": len(s1) > 1,
                "jaccard_medio": round(float(np.mean(pair_j)) if pair_j else 1.0, 4),
                "posicoes": ranks,
                "n_confirmados": [len(picks_by[r][qid]) for r in reps],
            }
        )

    agg = {m: [round(float(p["metrics"][m]), 4) for p in ps] for m in ("ndcg@10", "recall@1") if m in ps[0]["metrics"]}
    return {
        "run_id": run.get("run_id"),
        "split": run.get("split"),
        "provider": run.get("providers") or run.get("provider"),
        "ordem": run.get("repeat_order", "fixa"),
        "n_passadas": len(ps),
        "n_consultas": len(common),
        "taxa_mudanca_decisao": round(float(np.mean(changed_dec)), 4),
        "taxa_mudanca_posicao": round(float(np.mean(changed_rank)), 4),
        "taxa_mudanca_success1": round(float(np.mean(changed_s1)), 4),
        "jaccard_medio_entre_passadas": round(float(np.mean(jaccs)), 4),
        "agregado_por_passada": agg,
        "amplitude_agregada": {m: round(max(v) - min(v), 4) for m, v in agg.items()},
        "falhas_por_passada": ledger_failures(payload),
        "per_query": per_query,
    }


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("factorial", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    reports = []
    for path in args.factorial:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        rep = analyse(payload)
        reports.append(rep)
        prov = rep["provider"]
        print(f"\n### {rep['run_id']}  ({rep['n_consultas']} consultas × {rep['n_passadas']} passadas)")
        print(f"  provedor: {prov}   ordem dos candidatos: {rep['ordem']}")
        print(f"  mudou a decisão (conjunto confirmado): {rep['taxa_mudanca_decisao']:.1%} das consultas")
        print(f"  mudou a posição do alvo:               {rep['taxa_mudanca_posicao']:.1%}")
        print(f"  mudou o Success@1:                     {rep['taxa_mudanca_success1']:.1%}")
        print(f"  Jaccard médio entre passadas:          {rep['jaccard_medio_entre_passadas']:.3f}")
        for m, v in rep["agregado_por_passada"].items():
            print(f"  {m} por passada: {v}  (amplitude {rep['amplitude_agregada'][m]:+.3f})")
        for r, f in sorted(rep["falhas_por_passada"].items()):
            if f["failed"]:
                print(f"  passada {r}: {f['failed']}/{f['calls']} chamadas falharam ({f['failure_rate']:.1%})")
        if rep["ordem"] == "fixa" and rep["taxa_mudanca_decisao"] == 0.0:
            print("  → decisão estável a temperatura zero nesta amostra.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"kind": "variabilidade", "reports": reports}, fh, ensure_ascii=False, indent=2)
        print(f"\n» JSON: {os.path.relpath(args.out, _ROOT)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
