# -*- coding: utf-8 -*-
"""Desenho fatorial 2×2 das duas etapas de LLM (protocolo §7.1).

    .venv/bin/python -m experiments.factorial --split hard
    .venv/bin/python -m experiments.factorial --split test --conditions C00,C11
    .venv/bin/python -m experiments.factorial --split hard --repeats 5 --sample 10
    .venv/bin/python -m experiments.factorial --split object --prices experiments/prices.json
    .venv/bin/python -m experiments.factorial --split object --resume experiments/results/<ontem>.json

Por que não fica em `eval/`
---------------------------
`eval/` é determinístico e sem rede por regra do projeto. Isto aqui chama o
provedor de LLM, então mora em `experiments/` — mas grava no **mesmo formato**
de `eval/results/`, de modo que `python -m eval.stats compare --results <arquivo>`
funciona direto sobre a saída.

As quatro células
-----------------

| condição | etapa A (entende a consulta) | etapa B (confirma o top-k) |
|---|---|---|
| `C00` | não | não |
| `C10` | sim | não |
| `C01` | não | sim |
| `C11` | sim | sim  ← o sistema em produção |

O que a medição anterior (ad hoc, "antes/depois") não permitia: separar o efeito
de cada etapa e estimar a interação entre elas. `(C11−C01) − (C10−C00)` é a
interação; é a comparação P2 do protocolo.

O que fica congelado entre as células: corpus, índice, pesos da fusão, *prompts*,
modelos, tamanho do *pool* e a ordem dos candidatos entregue ao verificador.

Execução distribuída por dias, sem confundir dia com condição
-------------------------------------------------------------
A cota gratuita do provedor (200 mil *tokens*/dia no modelo de confirmação, a
~1.960 *tokens* por consulta) não comporta o fatorial completo num dia só. O
protocolo (§7.3) aceita distribuir a execução, **desde que as condições sejam
intercaladas** — senão uma mudança do serviço entre um dia e outro entra nos
números como se fosse efeito da condição.

A forma mais forte de intercalar é não intercalar: o laço externo é **por
consulta**, e cada consulta recebe as quatro células na mesma sessão, minutos
umas das outras. Quando a cota diária acaba, a execução para num limite de
consulta, grava o que está completo e é retomada no dia seguinte com
`--resume`. Nenhuma consulta fica com células de dias diferentes; o que varia
entre dias é *quais* consultas, não *quais* condições.

Cada célula registra `measured_utc`, e o *ledger* registra modelo, horário,
*tokens* de entrada e saída, tentativas, falhas e custo por chamada — o que
permite verificar depois se houve deriva entre os dias.

Taxonomia de erro (protocolo §3), registrada por consulta na condição com etapa B:

* `recuperacao` — o alvo nem chegou ao *pool*: nenhum verificador poderia consertar;
* `julgamento` — o alvo estava no *pool* e não foi confirmado (ou outro foi promovido acima);
* `evidencia` — como acima, **mas** o texto entregue não contém as pistas da
  consulta (`cue_coverage` baixa): a falha é da fonte, não do juízo;
* `ok` — o alvo terminou em #1.

O reagrupamento por franquia fica **fora** destas células de propósito: ele
depende do `tipo` vindo da etapa A e contaminaria o fatorial. É avaliado à parte,
como ablação, no caminho determinístico.

Primeiro estágio
----------------
As células usam o caminho de **busca por sinopse** (`_synopsis_scores`), o mesmo
que o pipeline `fusion` de `eval/pipelines.py` — não o `search_combined` inteiro
de produção, que ainda mistura casamento por nome e detecção de intenção. É uma
simplificação deliberada: mantém C00 idêntico à linha `fusion` de `eval.run`, de
modo que as duas medições sejam diretamente comparáveis. Verificado no split
`hard`: C00 dá nDCG@10 = 0,473, o mesmo valor de `eval.run --split hard`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Optional

os.environ.setdefault("RECOMENDAI_TMDB_NAMES", "0")

import numpy as np  # noqa: E402

from eval import metrics as M  # noqa: E402
from eval.dataset import dataset_sha1, load_queries  # noqa: E402
from eval.pipelines import make_ctx, raw_channels  # noqa: E402
from experiments import features as F  # noqa: E402
from experiments import llm_client  # noqa: E402
from experiments.llm_client import LLMRunner, QuotaExhausted, load_prices  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
RESULTS_DIR = os.path.join(_HERE, "results")

CONDITIONS = {
    "C00": (False, False),
    "C10": (True, False),
    "C01": (False, True),
    "C11": (True, True),
}

# Limiar de cobertura de pista abaixo do qual a falha é atribuída à EVIDÊNCIA e
# não ao julgamento. Escolhido antes de olhar resultado; declarado no JSON.
#
# ATENÇÃO — este rótulo é um PROXY PROVISÓRIO, e a comparação entre provedores
# de 2026-09-19 mostrou que ele é fraco. As mesmas 30 consultas, com os mesmos
# textos e portanto a mesma cobertura de pista, produziram 1 erro de "evidência"
# com o modelo de 27B e 10 com o de 8B. Se o texto fosse mesmo insuficiente,
# nenhum dos dois resolveria. Logo o que o proxy chamou de "o fato não está na
# fonte" era, naquelas consultas, falha de JULGAMENTO do modelo menor.
#
# O instrumento correto é o rótulo humano de cobertura da fonte previsto em
# `docs/PROTOCOLO-TOIS.md` §5 e implementado em `experiments/annotate.py`. Até
# ele existir, ler esta coluna como "candidato a erro de evidência", nunca como
# atribuição.
EVIDENCE_COVERAGE_FLOOR = 0.34


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _derive_plan(query: str, plan: Optional[dict]) -> tuple[str, list, Optional[float], Optional[float], str]:
    """Mesma derivação de `core.inference_client._understand_and_rewrite`:
    o texto original **nunca** é substituído, só acrescido (regra dura do projeto)."""
    from core.inference_client import OBJECT_PLOT_LEXICAL_WEIGHT

    if not plan:
        return query, [], None, None, "generico"
    tipo = plan.get("tipo") if plan.get("tipo") in ("pessoa", "objeto", "generico") else "generico"
    pistas_objeto = [str(x) for x in (plan.get("pistas_objeto") or [])]
    pistas_pessoa = [str(x) for x in (plan.get("pistas_pessoa") or [])]
    out, plw, ew = query, None, None
    if tipo == "objeto" and pistas_objeto:
        ql = query.lower()
        extra = [t for t in pistas_objeto if t.lower() not in ql]
        out = f"{query} {' '.join(extra)}".strip() if extra else query
        plw, ew = OBJECT_PLOT_LEXICAL_WEIGHT, 0.0
    return out, (pistas_pessoa if tipo == "pessoa" else []), plw, ew, tipo


def _candidates(engine, ids: list[int], pool: int) -> list[dict]:
    """Top-`pool` com a sinopse **inteira** do catálogo — igual à produção, que
    não manda a versão truncada de exibição para o verificador."""
    from core import catalog

    out = []
    for tid in ids[:pool]:
        mv = catalog.get_movie(int(tid)) or {}
        out.append(
            {
                "tmdb_id": int(tid),
                "title": mv.get("title"),
                "year": (str(mv.get("release_date") or "")[:4] or None),
                "overview": mv.get("overview") or "",
            }
        )
    return out


def _promote(ids: list[int], picks: list[int]) -> list[int]:
    """Confirmados sobem na ordem de confiança; o resto mantém a ordem da fusão."""
    if not picks:
        return ids
    pick_set = {int(p) for p in picks}
    promoted = [int(p) for p in picks if int(p) in set(ids)]
    return promoted + [i for i in ids if int(i) not in pick_set]


def _first_stage(engine, q, plan: Optional[dict], pool: int) -> dict:
    """Primeiro estágio + candidatos + sinais pré-chamada, para um dado plano.

    Só depende de o entendimento de consulta ter rodado ou não, então as duas
    células que compartilham esse estado (C00/C01 e C10/C11) reaproveitam o
    mesmo cálculo."""
    q_eff, pistas, plw, ew, tipo = _derive_plan(q.query, plan)
    scores = engine._synopsis_scores(q_eff, pistas_pessoa=pistas, plot_lexical_weight=plw, entity_weight=ew)
    order = np.argsort(scores, kind="stable")[::-1]
    ids = [int(engine._movie_ids[i]) for i in order]
    rank_first = ids.index(q.relevant_id) + 1 if q.relevant_id in ids else None
    cands = _candidates(engine, ids, pool)
    feats = F.all_features(q_eff, scores, cands, raw=raw_channels(engine, make_ctx(engine, q_eff)), engine=engine)
    return {
        "q_eff": q_eff,
        "tipo": tipo,
        "ids": ids,
        "rank_first": rank_first,
        "cands": cands,
        "features": feats,
    }


def run_query(
    engine, q, runner: LLMRunner, conds: list[str], pool: int, repeat: int, use_cache: bool
) -> dict[str, dict]:
    """**Todas** as células de UMA consulta, na mesma sessão.

    O laço externo é por consulta, não por condição, de propósito. Como o
    orçamento diário do provedor obriga a distribuir a execução por vários dias
    (protocolo §7.3), rodar condição a condição colocaria C00/C10 num dia e
    C01/C11 em outro — e qualquer mudança do serviço entre os dias apareceria
    como se fosse efeito da condição. Assim, ou a consulta tem as quatro células
    do mesmo momento, ou não entra.

    Levanta `QuotaExhausted` se a cota diária acabar no meio: o chamador
    descarta esta consulta e grava o que já estava completo."""
    need_a = any(CONDITIONS[c][0] for c in conds)
    plan: Optional[dict] = None
    if need_a:
        plan, _ = runner.understand(q.qid, q.query, repeat=repeat, use_cache=use_cache)
        if runner.quota_exhausted:
            raise QuotaExhausted(runner.quota_exhausted)

    stages: dict[bool, dict] = {}
    for use_a in {CONDITIONS[c][0] for c in conds}:
        stages[use_a] = _first_stage(engine, q, plan if use_a else None, pool)

    out: dict[str, dict] = {}
    for cond in conds:
        use_a, use_b = CONDITIONS[cond]
        st = stages[use_a]
        rank_final, picks, confirmed_target, false_confirm = st["rank_first"], [], None, None
        if use_b:
            picks, _ = runner.confirm(q.qid, st["q_eff"], st["cands"], repeat=repeat, use_cache=use_cache)
            if runner.quota_exhausted:
                raise QuotaExhausted(runner.quota_exhausted)
            ids_after = _promote(st["ids"], picks)
            rank_final = ids_after.index(q.relevant_id) + 1 if q.relevant_id in ids_after else None
            confirmed_target = q.relevant_id in {int(p) for p in picks}
            false_confirm = bool(picks) and not confirmed_target

        in_pool = any(c["tmdb_id"] == q.relevant_id for c in st["cands"])
        target_text = next((c["overview"] for c in st["cands"] if c["tmdb_id"] == q.relevant_id), "")
        target_cov = F.cue_coverage(st["q_eff"], target_text) if in_pool else 0.0
        if rank_final == 1:
            error_type = "ok"
        elif not in_pool:
            error_type = "recuperacao"
        elif target_cov < EVIDENCE_COVERAGE_FLOOR:
            error_type = "evidencia"
        else:
            error_type = "julgamento"

        out[cond] = {
            "qid": q.qid,
            "source": q.source,
            "title": q.relevant_title,
            "relevant_id": q.relevant_id,
            "rank": rank_final,
            "rank_first_stage": st["rank_first"],
            "rr": round(M.reciprocal_rank(rank_final), 4),
            "query_effective": st["q_eff"],
            "query_changed": st["q_eff"] != q.query,
            "tipo": st["tipo"],
            "n_picks": len(picks),
            "picks": picks,
            "confirmed_target": confirmed_target,
            "false_confirmation": false_confirm,
            "target_in_pool": in_pool,
            "target_cue_coverage": target_cov,
            "error_type": error_type,
            "measured_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "features": st["features"],
        }
    return out


def _aggregate(per_query: list[dict]) -> dict:
    return {"metrics": M.aggregate([r["rank"] for r in per_query]), "per_query": per_query}


def _load_resume(path: str, conds: list[str], force: bool = False) -> tuple[dict[str, list[dict]], set]:
    """Consultas já medidas em TODAS as condições pedidas, de uma execução anterior.

    Recusa retomar de uma execução marcada inválida: herdar células atenuadas em
    silêncio é exatamente o que o portão de validade existe para impedir."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if (payload.get("run") or {}).get("valid") is False and not force:
        raise SystemExit(
            f"recusado: {os.path.basename(path)} está marcada INVÁLIDA "
            f"({(payload.get('run') or {}).get('invalid_reason')}).\n"
            "Refaça do zero, ou use --resume-force se souber por que é seguro."
        )
    res = payload.get("results") or {}
    done: dict[str, list[dict]] = {c: list((res.get(c) or {}).get("per_query") or []) for c in conds}
    sets = [{r["qid"] for r in rows} for rows in done.values()]
    complete = set.intersection(*sets) if sets and all(sets) else set()
    for c in conds:
        done[c] = [r for r in done[c] if r["qid"] in complete]
    return done, complete


def revalidate(path: str, max_failure_rate: float) -> int:
    """Recalcula o portão de validade a partir do *ledger* da execução.

    Existe porque o critério pode ser corrigido depois da execução (foi: a
    chamada que detecta a cota diária estava contando como falha de medição,
    embora a consulta dela seja descartada). Recomputar do *ledger* é auditável;
    editar o veredito à mão não seria. A recomputação fica registrada no próprio
    arquivo."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    run = payload.get("run") or {}
    ledger_rel = (run.get("llm") or {}).get("ledger")
    if not ledger_rel:
        raise SystemExit("execução sem ledger — não dá para recomputar")
    ledger = os.path.join(_ROOT, ledger_rel)
    if not os.path.exists(ledger):
        raise SystemExit(f"ledger ausente: {ledger_rel}")

    graded = failed = quota_stop = 0
    with open(ledger, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("from_cache"):
                continue
            if rec.get("error") == "quota_diaria":
                quota_stop += 1
                continue
            graded += 1
            if not rec.get("ok"):
                failed += 1
    rate = round(failed / graded, 4) if graded else 0.0

    before = run.get("valid")
    run["llm"]["failure_rate"] = rate
    run["llm"]["calls_graded"] = graded
    run["llm"]["calls_quota_stop"] = quota_stop
    run["valid"] = rate <= max_failure_rate
    run["invalid_reason"] = (
        None if run["valid"] else (f"{rate:.1%} das chamadas falharam (teto {max_failure_rate:.1%})")
    )
    run.setdefault("revalidations", []).append(
        {
            "at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "valid_before": before,
            "valid_after": run["valid"],
            "failure_rate": rate,
            "graded_calls": graded,
            "quota_stop_calls": quota_stop,
            "criterion": "chamada que detecta cota diária não conta como falha (consulta descartada)",
        }
    )
    payload["run"] = run
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"» {os.path.relpath(path, _ROOT)}")
    print(f"  {graded} chamadas avaliadas, {failed} falhas, {quota_stop} descartadas por cota")
    print(f"  taxa de falha: {rate:.1%} (teto {max_failure_rate:.1%})")
    print(f"  validade: {before} → {run['valid']}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.factorial", description="2×2 das etapas de LLM.")
    ap.add_argument("--split", default="hard")
    ap.add_argument("--conditions", default="C00,C10,C01,C11")
    ap.add_argument("--pool", type=int, default=None, help="candidatos enviados ao verificador (default: produção)")
    ap.add_argument("--sample", type=int, default=None, help="usa só as N primeiras consultas do split")
    ap.add_argument("--repeats", type=int, default=0, help="repetições SEM cache, para medir variabilidade (§7.3)")
    ap.add_argument("--prices", help="JSON de preço por milhão de tokens, por modelo")
    ap.add_argument(
        "--provider",
        choices=["groq", "local"],
        default="groq",
        help="groq = o modelo de produção; local = MLX nesta máquina (sem cota, pesos fixados por hash)",
    )
    ap.add_argument("--provider-understand", choices=["groq", "local"], help="sobrepõe --provider só na etapa A")
    ap.add_argument(
        "--provider-confirm",
        choices=["groq", "local"],
        help="sobrepõe --provider só na etapa B — é assim que se isola o VERIFICADOR",
    )
    ap.add_argument("--out", help="caminho do JSON (default: experiments/results/<stamp>__factorial-<split>.json)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--resume", help="JSON de uma execução anterior; retoma as consultas que faltam")
    ap.add_argument("--resume-force", action="store_true", help="retoma mesmo de execução marcada inválida")
    ap.add_argument(
        "--max-failure-rate",
        type=float,
        default=0.02,
        help="acima disso a rodada é marcada INVÁLIDA (default 2%%)",
    )
    ap.add_argument("--revalidate", help="recalcula o portão de validade de um JSON a partir do ledger e sai")
    args = ap.parse_args(argv)

    if args.revalidate:
        return revalidate(args.revalidate, args.max_failure_rate)

    from core import query_llm
    from retrieval.search_engine import SearchEngine

    if "groq" in {args.provider, args.provider_understand, args.provider_confirm} and not query_llm.is_configured():
        raise SystemExit("GROQ_API_KEY ausente/desligada — use --provider local ou configure a chave.")

    conds = [c.strip() for c in args.conditions.split(",")]
    for c in conds:
        if c not in CONDITIONS:
            raise SystemExit(f"condição desconhecida: {c} (use {', '.join(CONDITIONS)})")

    pool = args.pool or query_llm.RERANK_POOL
    queries = load_queries(args.split)
    if args.sample:
        queries = queries[: args.sample]

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    provs = {args.provider_understand or args.provider, args.provider_confirm or args.provider}
    sufixo = "" if provs == {"groq"} else ("-local" if provs == {"local"} else "-misto")
    run_id = f"{stamp}__factorial-{args.split}{sufixo}"
    runner = LLMRunner(
        run_id,
        prices=load_prices(args.prices),
        provider=args.provider,
        provider_understand=args.provider_understand,
        provider_confirm=args.provider_confirm,
    )

    print(f"» motor de busca… ({len(queries)} consultas de {args.split}, pool={pool})")
    engine = SearchEngine(rerank=False)
    if not engine.has_synopsis_index:
        raise SystemExit("índice de sinopse ausente — rode `python -m retrieval.index_builder`")

    acc: dict[str, list[dict]] = {c: [] for c in conds}
    already: set = set()
    if args.resume:
        acc, already = _load_resume(args.resume, conds, force=args.resume_force)
        print(f"  retomando: {len(already)} consultas já completas em {os.path.basename(args.resume)}")

    pending = [q for q in queries if q.qid not in already]
    stopped: Optional[str] = None
    for i, q in enumerate(pending, 1):
        try:
            cells = run_query(engine, q, runner, conds, pool, 0, True)
        except QuotaExhausted as exc:
            stopped = str(exc)
            print(f"\n!! cota diária do provedor esgotada em {q.qid} — parando num limite de consulta.")
            print("   O que já estava completo foi preservado; retome amanhã com --resume.")
            break
        for cond, row in cells.items():
            acc[cond].append(row)
        if not args.quiet and i % 5 == 0:
            print(f"    {i}/{len(pending)} consultas (todas as {len(conds)} células)", flush=True)

    results: dict = {c: _aggregate(acc[c]) for c in conds if acc[c]}
    if not results:
        raise SystemExit("nenhuma consulta completa — nada a gravar.")
    for cond in conds:
        if cond in results:
            m = results[cond]["metrics"]
            print(f"  · {cond}: nDCG@10={m['ndcg@10']:.3f}  MRR={m['mrr']:.3f}  Success@1={m['recall@1']:.3f}")

    # --------------------------------------------- repetições (variabilidade)
    repeats: list[dict] = []
    if args.repeats and "C11" in conds and not stopped:
        sub = [q for q in queries if q.qid in {r["qid"] for r in acc["C11"]}][: min(10, len(acc["C11"]))]
        print(f"  · repetições SEM cache: {args.repeats}× em {len(sub)} consultas")
        for r in range(1, args.repeats + 1):
            rows = []
            try:
                for q in sub:
                    rows.append(run_query(engine, q, runner, ["C11"], pool, r, False)["C11"])
            except QuotaExhausted as exc:
                stopped = str(exc)
                print(f"    cota esgotada na repetição {r}; {len(rows)} consultas medidas.")
            if rows:
                repeats.append({"repeat": r, "metrics": M.aggregate([x["rank"] for x in rows]), "per_query": rows})
                print(f"    {r}/{args.repeats}: nDCG@10={repeats[-1]['metrics']['ndcg@10']:.3f}")
            if stopped:
                break

    payload = {
        "run": {
            "kind": "factorial-2x2",
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_id": run_id,
            "git_commit": _git_commit(),
            "split": args.split,
            "n_queries": len(next(iter(results.values()))["per_query"]),
            "n_queries_split": len(queries),
            "resumed_from": os.path.relpath(args.resume, _ROOT) if args.resume else None,
            "partial": bool(stopped),
            "stopped_reason": stopped,
            "query_major": True,
            "dataset_sha1": dataset_sha1(),
            "conditions": conds,
            "pool": pool,
            "provider": args.provider,
            "providers": runner.providers,
            "models": {
                stage: (
                    (query_llm.GROQ_MODEL if stage == "understand" else query_llm.GROQ_RERANK_MODEL)
                    if prov == "groq"
                    else llm_client.LOCAL_MODEL
                )
                for stage, prov in runner.providers.items()
            },
            "runtime": {
                "groq": {
                    "endpoint": llm_client.query_llm._URL,
                    "note": "sem versão exposta pelo provedor; a data da execução é o único identificador",
                    "reasoning_effort_understand": "low",
                    "temperature": 0,
                }
                if "groq" in runner.providers.values()
                else None,
                "local": llm_client.local_runtime() if "local" in runner.providers.values() else None,
            },
            "prompt_sha256": {
                "understand": hashlib.sha256(query_llm._SYSTEM.encode("utf-8")).hexdigest(),
                "confirm": hashlib.sha256(query_llm._RERANK_SYSTEM.encode("utf-8")).hexdigest(),
            },
            "temperature": 0,
            "evidence_coverage_floor": EVIDENCE_COVERAGE_FLOOR,
            "franchise_pullup": False,
            "llm": runner.summary(),
        },
        "results": results,
        "repeats": repeats,
    }

    # Portão de validade: uma chamada recusada vira "nenhum candidato confirmado",
    # ou seja, uma célula com etapa B que se comporta como se não tivesse etapa B.
    # O efeito medido sai atenuado e o JSON pareceria normal — daí o portão ser
    # explícito e gravado dentro do próprio arquivo.
    failure_rate = payload["run"]["llm"]["failure_rate"]
    payload["run"]["valid"] = failure_rate <= args.max_failure_rate
    payload["run"]["max_failure_rate"] = args.max_failure_rate
    if not payload["run"]["valid"]:
        payload["run"]["invalid_reason"] = (
            f"{failure_rate:.1%} das chamadas falharam (teto {args.max_failure_rate:.1%}); "
            "efeito da etapa B sai atenuado — refazer"
        )

    out = args.out or os.path.join(RESULTS_DIR, f"{run_id}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print("\n### Células")
    print("| condição | A | B | nDCG@10 | MRR | Success@1 | mediana |")
    print("|---|---|---|---:|---:|---:|---:|")
    for cond in conds:
        a, b = CONDITIONS[cond]
        m = results[cond]["metrics"]
        med = f"#{m['median_rank']}" if m["median_rank"] is not None else "—"
        print(
            f"| {cond} | {'sim' if a else 'não'} | {'sim' if b else 'não'} | "
            f"{m['ndcg@10']:.3f} | {m['mrr']:.3f} | {m['recall@1']:.3f} | {med} |"
        )
    if {"C00", "C01", "C10", "C11"} <= set(conds):
        g = {c: results[c]["metrics"]["recall@1"] for c in ("C00", "C10", "C01", "C11")}
        print()
        print(
            f"efeito A (entendimento) = {g['C10'] - g['C00']:+.3f}   efeito B (confirmação) = {g['C01'] - g['C00']:+.3f}"
        )
        print(f"interação (C11−C01)−(C10−C00) = {(g['C11'] - g['C01']) - (g['C10'] - g['C00']):+.3f}   [Success@1]")
        print("Sem intervalo, estes são pontos. Rode: python -m eval.stats compare --results <json> --baseline C00")

    print(f"\n» JSON: {os.path.relpath(out, _ROOT)}")
    print(f"» ledger: {payload['run']['llm']['ledger']}")
    print(
        f"» chamadas reais: {payload['run']['llm']['calls_real']}"
        f"  falhas: {failure_rate:.1%}"
        f"  espera por vazão: {sum(b.get('throttled_s', 0) for b in payload['run']['llm']['by_stage'].values()):.0f}s"
        f"  custo declarado: US$ {payload['run']['llm']['cost_usd_total']}"
    )
    if not payload["run"]["valid"]:
        print(f"\n!! RODADA INVÁLIDA — {payload['run']['invalid_reason']}")
        print("   Os números acima estão ATENUADOS e não devem ser reportados.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
