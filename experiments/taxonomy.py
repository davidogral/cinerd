# -*- coding: utf-8 -*-
"""Taxonomia de ganhos e danos da verificação por LLM (entrega empírica do plano).

    .venv/bin/python -m experiments.taxonomy experiments/results/*__factorial-*.json
    .venv/bin/python -m experiments.taxonomy A.json --before C10 --after C11 --metric ndcg@10
    .venv/bin/python -m experiments.taxonomy A.json --cases 8

O plano pede, como resultado empírico, uma "taxonomia replicável de ganhos e
danos da verificação por LLM em recuperação de item lembrado parcialmente". Um
agregado ("subiu 0,18 de nDCG@10") não é isso: some a informação de **onde** o
ganho aparece e **o que** acontece quando ele não aparece.

Este módulo lê a saída de `experiments.factorial` e responde quatro perguntas,
todas por consulta:

1. **Onde o verificador ajuda?** Distribuição de ganho por posição inicial do
   alvo — a hipótese H1 diz que ele ajuda quando o alvo já está no *pool* mas
   abaixo de falsos positivos semânticos.
2. **Onde ele prejudica?** Consultas pioradas, com a confirmação falsa que as
   causou — a hipótese H2 diz que isso se concentra em evidência incompleta.
3. **De quem é a culpa quando não resolve?** `recuperacao` (o alvo nem chegou ao
   *pool*), `evidencia` (o texto não contém o fato citado) ou `julgamento` (o
   texto continha e mesmo assim não foi confirmado).
4. **Quanto o teto de recuperação limita tudo isso?** Nenhum verificador
   conserta o que o primeiro estágio não trouxe.

Só lê arquivos; não chama o provedor e não escreve nada.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from typing import Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

ERROR_LABELS = {
    "ok": "resolvido (#1)",
    "recuperacao": "erro de recuperação (fora do pool)",
    "evidencia": "erro de evidência (fato não está no texto)",
    "julgamento": "erro de julgamento (texto tinha, não confirmou)",
}


def _metric(rank: Optional[int], name: str) -> float:
    if name == "success@1":
        return 1.0 if rank == 1 else 0.0
    if name == "mrr":
        return 1.0 / rank if rank else 0.0
    k = int(name.split("@")[1])
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _bucket(rank: Optional[int], pool: int) -> str:
    if rank is None:
        return "não recuperado"
    if rank == 1:
        return "#1 (já certo)"
    if rank <= 5:
        return "#2–5"
    if rank <= pool:
        return f"#6–{pool} (no pool)"
    return f">#{pool} (fora do pool)"


def load(paths: Sequence[str], before: str, after: str, metric: str) -> tuple[list[dict], int, bool]:
    rows: list[dict] = []
    pool = 20
    all_valid = True
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        run = payload.get("run") or {}
        pool = int(run.get("pool") or pool)
        if run.get("valid") is False:
            all_valid = False
            print(f"!! {os.path.basename(path)}: {run.get('invalid_reason')}")
        res = payload.get("results") or {}
        if before not in res or after not in res:
            continue
        by_qid = {r["qid"]: r for r in res[after]["per_query"]}
        for r in res[before]["per_query"]:
            a = by_qid.get(r["qid"])
            if a is None:
                continue
            rows.append(
                {
                    "qid": r["qid"],
                    "split": run.get("split"),
                    "title": r.get("title"),
                    "query": a.get("query_effective") or "",
                    "rank_before": r.get("rank"),
                    "rank_after": a.get("rank"),
                    "gain": _metric(a.get("rank"), metric) - _metric(r.get("rank"), metric),
                    "error_type": a.get("error_type"),
                    "target_in_pool": a.get("target_in_pool"),
                    "cue_cov": a.get("target_cue_coverage"),
                    "confirmed_target": a.get("confirmed_target"),
                    "false_confirmation": a.get("false_confirmation"),
                    "n_picks": a.get("n_picks"),
                    "tipo": a.get("tipo"),
                }
            )
    return rows, pool, all_valid


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.taxonomy", description="Ganhos e danos da verificação por LLM.")
    ap.add_argument("factorial", nargs="+")
    ap.add_argument("--before", default="C10", help="condição sem confirmação (default C10)")
    ap.add_argument("--after", default="C11", help="condição com confirmação (default C11)")
    ap.add_argument("--metric", default="success@1")
    ap.add_argument("--cases", type=int, default=5, help="quantos casos concretos mostrar de cada tipo")
    args = ap.parse_args(argv)

    rows, pool, valid = load(args.factorial, args.before, args.after, args.metric)
    if not rows:
        raise SystemExit(f"nenhuma consulta com as condições {args.before}/{args.after}")
    n = len(rows)
    helped = [r for r in rows if r["gain"] > 0]
    hurt = [r for r in rows if r["gain"] < 0]
    same = n - len(helped) - len(hurt)

    print(f"» {n} consultas · {args.before} → {args.after} · métrica {args.metric} · pool {pool}")
    if not valid:
        print("!! há rodada INVÁLIDA na entrada: os números abaixo estão atenuados.")
    print(f"   melhorou {len(helped)}  ·  piorou {len(hurt)}  ·  inalterado {same}")
    print(f"   ganho médio: {sum(r['gain'] for r in rows) / n:+.4f}")

    # 1. onde ajuda (H1)
    print("\n### 1. Onde a confirmação ajuda — por posição inicial do alvo")
    print("| posição antes | consultas | melhorou | piorou | ganho médio |")
    print("|---|---:|---:|---:|---:|")
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        buckets.setdefault(_bucket(r["rank_before"], pool), []).append(r)
    order = ["#1 (já certo)", "#2–5", f"#6–{pool} (no pool)", f">#{pool} (fora do pool)", "não recuperado"]
    for name in order:
        b = buckets.get(name)
        if not b:
            continue
        g = sum(1 for r in b if r["gain"] > 0)
        h = sum(1 for r in b if r["gain"] < 0)
        print(f"| {name} | {len(b)} | {g} | {h} | {sum(r['gain'] for r in b) / len(b):+.3f} |")
    print("\nH1 prevê concentração do ganho nas faixas de dentro do pool abaixo de #1.")

    # 2. onde prejudica (H2)
    print(f"\n### 2. Onde prejudica — {len(hurt)} consulta(s)")
    if not hurt:
        print("Nenhuma regressão nesta amostra. Com este n, isso limita o que se pode")
        print("afirmar sobre H2: ausência de dano observado não é ausência de dano.")
    else:
        print("| consulta | antes | depois | confirmou o alvo? | confirmados | conf. falsa | cobertura |")
        print("|---|---:|---:|---|---:|---|---:|")
        for r in sorted(hurt, key=lambda x: x["gain"])[: args.cases]:
            q = (r["query"] or "")[:40]
            print(
                f"| {q}… | #{r['rank_before']} | #{r['rank_after']} | "
                f"{'sim' if r['confirmed_target'] else 'não'} | {r['n_picks']} | "
                f"{'sim' if r['false_confirmation'] else 'não'} | {r['cue_cov']} |"
            )
        low = sum(1 for r in hurt if (r["cue_cov"] or 0) < 0.34)
        multi = sum(1 for r in hurt if r["confirmed_target"] and (r["n_picks"] or 0) > 1)
        print(f"\nDas {len(hurt)} regressões, {low} tinham cobertura de pista baixa no alvo (H2).")
        if multi:
            print(
                f"{multi} delas são um mecanismo próprio: o alvo FOI confirmado, mas junto com "
                "outro\ncandidato que veio antes na ordem de confiança — a confirmação múltipla "
                "reordena\ne pode rebaixar um #1 que já estava certo. É dano sem confirmação falsa."
            )

    # 3. de quem é a culpa
    print("\n### 3. Quando não resolve, de quem é a falha")
    counts = Counter(r["error_type"] for r in rows)
    print("| tipo | consultas | % |")
    print("|---|---:|---:|")
    for key in ("ok", "recuperacao", "evidencia", "julgamento"):
        c = counts.get(key, 0)
        print(f"| {ERROR_LABELS[key]} | {c} | {c / n:.1%} |")
    print("\nA leitura que importa para o desenho do sistema: erro de RECUPERAÇÃO é teto")
    print("do primeiro estágio (aumentar o pool ou o recall), erro de EVIDÊNCIA seria teto")
    print("da fonte de texto, e só o erro de JULGAMENTO é do verificador.")
    print("\nRESSALVA: a separação evidência/julgamento é um proxy por cobertura literal de")
    print("pista, e a comparação entre provedores mostrou que ele é fraco — as mesmas")
    print("consultas, com o mesmo texto, deram 1 erro de 'evidência' no modelo de 27B e 10")
    print("no de 8B. Texto insuficiente não depende do modelo; julgamento depende. Ler como")
    print("'candidato a erro de evidência' até existir o rótulo humano (protocolo §5).")

    # 4. teto de recuperação
    in_pool = sum(1 for r in rows if r["target_in_pool"])
    solved = sum(1 for r in rows if r["rank_after"] == 1)
    print("\n### 4. Teto de recuperação")
    print(f"   alvo dentro do pool (top-{pool}): {in_pool}/{n} ({in_pool / n:.1%})")
    print(f"   resolvidas em #1 depois da confirmação: {solved}/{n} ({solved / n:.1%})")
    if in_pool:
        print(f"   aproveitamento do que estava disponível: {solved}/{in_pool} ({solved / in_pool:.1%})")
    print("   Nenhum verificador conserta o que o primeiro estágio não trouxe.")

    # casos concretos de ganho, para o artigo
    if helped and args.cases:
        print("\n### Casos de ganho (para ilustrar no texto)")
        print("| consulta | antes | depois | filme |")
        print("|---|---:|---:|---|")
        for r in sorted(helped, key=lambda x: -x["gain"])[: args.cases]:
            print(f"| {(r['query'] or '')[:44]}… | #{r['rank_before']} | #{r['rank_after']} | {r['title']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
