# -*- coding: utf-8 -*-
"""Variantes do pipeline de recuperação por sinopse, para a **tabela de ablação**.

Cada variante recebe `(engine, ctx)` e devolve a **ordenação completa** de
`tmdb_id`s (melhor primeiro). O `ctx` carrega a consulta limpa e o embedding da
consulta, computados uma vez por consulta e reaproveitados entre as variantes.

| chave            | o que é                                                        |
|------------------|----------------------------------------------------------------|
| `bm25`           | só o sinal **lexical** (BM25 Okapi cru sobre as sinopses)      |
| `embedding`      | só o sinal **semântico** (cosseno com o embedding da sinopse)  |
| `thematic`       | só o sinal **temático** (cosseno com o embedding de keywords)  |
| `fusion`         | fusão z-score dos três sinais + prior de popularidade (**pipeline de produção**) |
| `fusion_rerank`  | a fusão acima com o **cross-encoder** re-rankeando o top-`POOL` — variante experimental (off em produção, ver `search_engine.RERANK_ENABLED`) |

Todas operam sobre o mesmo índice, então a comparação isola o efeito de cada
sinal e do re-ranker.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from retrieval.search_engine import RERANK_POOL, clean_descriptive_query

POOL = RERANK_POOL


@dataclass
class QueryCtx:
    raw: str
    clean: str
    q_emb: Optional[np.ndarray]


def make_ctx(engine, query: str) -> QueryCtx:
    clean = clean_descriptive_query(query)
    q_emb = engine._encode(clean) if engine._embeddings is not None else None
    return QueryCtx(raw=query, clean=clean, q_emb=q_emb)


def _order_to_ids(engine, order: np.ndarray) -> list[int]:
    ids = engine._movie_ids
    return [int(ids[i]) for i in order]


def _bm25(engine, ctx: QueryCtx) -> list[int]:
    scores = engine._bm25.scores(ctx.clean)
    return _order_to_ids(engine, np.argsort(scores)[::-1])


def _embedding(engine, ctx: QueryCtx) -> list[int]:
    sims = engine._embeddings @ ctx.q_emb
    return _order_to_ids(engine, np.argsort(sims)[::-1])


def _thematic(engine, ctx: QueryCtx) -> list[int]:
    kw_q = engine._keyword_query_emb(ctx.clean, ctx.q_emb)
    sims = engine._kw_embeddings @ kw_q
    return _order_to_ids(engine, np.argsort(sims)[::-1])


def _fusion(engine, ctx: QueryCtx) -> list[int]:
    fused = engine._synopsis_scores(ctx.raw)
    return _order_to_ids(engine, np.argsort(fused)[::-1])


def _fusion_rerank(engine, ctx: QueryCtx) -> list[int]:
    return engine.synopsis_ranked_ids(ctx.raw, rerank=True, pool=POOL)


# Pseudo-relevance feedback (PRF, "Rocchio"): assume que o topo da 1ª busca está
# ~certo, calcula o centroide dos embeddings de sinopse desses K filmes e o mistura
# de volta no vetor da consulta (peso alpha), depois re-funde. Sem LLM, sem rede —
# só numpy sobre o índice que já existe. Ajuda consulta genérica: "grupo destrói
# objeto poderoso" puxa Vingadores no 1º passo, mas o centroide dos vizinhos certos
# empurra o vetor na direção de fantasia épica.
PRF_K = int(os.environ.get("RECOMENDAI_PRF_K", "8"))
PRF_ALPHA = float(os.environ.get("RECOMENDAI_PRF_ALPHA", "0.35"))


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _fusion_prf(engine, ctx: QueryCtx, k: int = PRF_K, alpha: float = PRF_ALPHA) -> list[int]:
    cq, q_emb = ctx.clean, ctx.q_emb
    comps0 = engine._synopsis_components(cq, q_emb=q_emb)
    fused0 = comps0["lexical"] + comps0["synopsis"] + comps0["keyword"] + comps0["prior"]
    top = np.argsort(fused0)[::-1][:k]
    centroid = _l2(engine._embeddings[top].astype(np.float64).mean(axis=0))
    q_prf = _l2((1.0 - alpha) * q_emb.astype(np.float64) + alpha * centroid).astype(np.float32)
    comps = engine._synopsis_components(cq, q_emb=q_prf)
    fused = comps["lexical"] + comps["synopsis"] + comps["keyword"] + comps["prior"]
    return _order_to_ids(engine, np.argsort(fused)[::-1])


# ---------------------------------------------------------------------------
# Baselines de combinação exigidos pelo protocolo do estudo
# (`docs/PROTOCOLO-TOIS.md`, §7.2): a mesma informação de recuperação, combinada
# por outra regra. Sem isso, "a fusão z-score é melhor" não tem contra-prova —
# só há ablação interna dela contra si mesma.

# Canais que a fusão de produção soma. `plot` e `person_match` ficam de fora: o
# primeiro é 0 por padrão (medido negativo), o segundo depende de pistas vindas
# do LLM e é inerte numa rodada determinística.
CHANNELS: tuple[str, ...] = ("lexical", "synopsis", "keyword", "entity", "plot_lexical", "plot_maxsim", "prior")

# Pesos de produção, para inicializar a busca da fusão treinável e para o
# registro do JSON. O lexical de produção é adaptativo (0.20–0.30 por consulta);
# a fusão treinável aprende **um** valor fixo — é um modelo mais simples de
# propósito, e a diferença está documentada no README.
from retrieval.search_engine import (  # noqa: E402
    DEFAULT_EMBED_WEIGHT,
    DEFAULT_ENTITY_WEIGHT,
    DEFAULT_KEYWORD_WEIGHT,
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_PLOT_BM25_WEIGHT,
    DEFAULT_PLOT_CHUNK_WEIGHT,
    DEFAULT_POP_PRIOR,
)

PRODUCTION_WEIGHTS: dict[str, float] = {
    "lexical": DEFAULT_LEXICAL_WEIGHT,
    "synopsis": DEFAULT_EMBED_WEIGHT,
    "keyword": DEFAULT_KEYWORD_WEIGHT,
    "entity": DEFAULT_ENTITY_WEIGHT,
    "plot_lexical": DEFAULT_PLOT_BM25_WEIGHT,
    "plot_maxsim": DEFAULT_PLOT_CHUNK_WEIGHT,
    "prior": DEFAULT_POP_PRIOR,
}


def raw_channels(engine, ctx: QueryCtx) -> dict[str, np.ndarray]:
    """Os sinais **sem peso** (z-score após ReLU), alinhados a `engine._movie_ids`.

    `_synopsis_components` devolve tudo já multiplicado pelo peso; aqui os pesos
    ajustáveis vão a 1.0 e os dois que só existem como constante de módulo
    (`plot_maxsim`, `prior`) são divididos de volta. Assim qualquer regra de
    combinação (RRF, fusão treinável, política) parte exatamente da mesma
    evidência que a fusão de produção usa."""
    comps = engine._synopsis_components(
        ctx.raw,
        q_emb=ctx.q_emb,
        lexical_weight=1.0,
        embed_weight=1.0,
        keyword_weight=1.0,
        plot_lexical_weight=1.0,
        entity_weight=1.0,
    )
    out = {
        name: np.asarray(comps[name], dtype=np.float64)
        for name in ("lexical", "synopsis", "keyword", "entity", "plot_lexical")
    }
    out["plot_maxsim"] = np.asarray(comps["plot_maxsim"], dtype=np.float64) / (DEFAULT_PLOT_CHUNK_WEIGHT or 1.0)
    out["prior"] = np.asarray(comps["prior"], dtype=np.float64) / (DEFAULT_POP_PRIOR or 1.0)
    return out


RRF_K = int(os.environ.get("RECOMENDAI_RRF_K", "60"))
RRF_DEPTH = int(os.environ.get("RECOMENDAI_RRF_DEPTH", "1000"))


def _rrf(engine, ctx: QueryCtx, k: int = RRF_K, depth: int = RRF_DEPTH) -> list[int]:
    """Reciprocal Rank Fusion sobre os mesmos canais: cada canal vira uma LISTA
    ordenada e o filme soma `1/(k + posição)` em cada lista onde aparece.

    Combina **posição**, não magnitude — por construção imune ao outlier de
    escala que motivou o teto de z-score. É o baseline que diz se o teto resolve
    um problema real da soma de z-scores ou se a soma é que era o problema.

    Profundidade `depth`: fora do top-`depth` de todo canal, o filme não recebe
    contribuição nenhuma — então posições no fundo da lista não são
    interpretáveis (a cauda empata em 0 e é desempatada pela ordem do índice).
    Métricas até @50 não são afetadas."""
    raw = raw_channels(engine, ctx)
    n = len(engine._movie_ids)
    scores = np.zeros(n, dtype=np.float64)
    for name in CHANNELS:
        vec = raw.get(name)
        if vec is None:
            continue
        order = np.argsort(vec, kind="stable")[::-1][:depth]
        if name != "prior":  # canais com ReLU: 0 = "não casou", não entra na lista
            order = order[vec[order] > 0]
        if order.size == 0:
            continue
        scores[order] += 1.0 / (k + np.arange(1, order.size + 1, dtype=np.float64))
    return _order_to_ids(engine, np.argsort(-scores, kind="stable"))


_LEARNED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params", "fusion_learned.json")
_learned_cache: Optional[dict] = None


def learned_weights() -> dict:
    """Pesos da fusão treinável (`eval/train_fusion.py`, ajustados só no dev)."""
    global _learned_cache
    if _learned_cache is None:
        if not os.path.exists(_LEARNED_PATH):
            raise RuntimeError(
                "eval/params/fusion_learned.json ausente — rode `python -m eval.train_fusion` "
                "(treina no split dev; nunca no test)."
            )
        with open(_LEARNED_PATH, encoding="utf-8") as fh:
            _learned_cache = json.load(fh)
    return _learned_cache


def _fusion_learned(engine, ctx: QueryCtx) -> list[int]:
    """Soma linear dos mesmos canais com pesos **ajustados no split dev** por
    busca coordenada, sob orçamento declarado (ver `eval/train_fusion.py`).

    Responde à objeção óbvia à fusão de produção: os pesos dela foram escolhidos
    à mão: quanto do resultado é a arquitetura e quanto é o ajuste manual? Este
    baseline recebe o mesmo orçamento e o mesmo dado."""
    w = learned_weights()["weights"]
    raw = raw_channels(engine, ctx)
    fused = np.zeros(len(engine._movie_ids), dtype=np.float64)
    for name in CHANNELS:
        fused += float(w.get(name, 0.0)) * raw[name]
    return _order_to_ids(engine, np.argsort(-fused, kind="stable"))


PIPELINES: dict[str, Callable] = {
    "bm25": _bm25,
    "embedding": _embedding,
    "thematic": _thematic,
    "fusion": _fusion,
    "fusion_prf": _fusion_prf,
    "rrf": _rrf,
    "fusion_learned": _fusion_learned,
    "fusion_rerank": _fusion_rerank,
}

# Rótulos legíveis para a tabela (ordem = ordem de exibição).
PIPELINE_LABELS: list[tuple[str, str]] = [
    ("bm25", "BM25 puro (lexical)"),
    ("embedding", "Só embedding (semântico)"),
    ("thematic", "Só temático (keywords)"),
    ("fusion", "Fusão (produção)"),
    ("fusion_prf", f"Fusão + PRF (k={PRF_K}, α={PRF_ALPHA})"),
    ("rrf", f"RRF sobre os mesmos canais (k={RRF_K})"),
    ("fusion_learned", "Fusão treinável (pesos ajustados no dev)"),
    ("fusion_rerank", f"Fusão + re-ranker (pool {POOL})"),
]

# Variantes que dependem do modelo de embeddings / cross-encoder (mais lentas).
NEEDS_EMBEDDINGS = {"embedding", "thematic", "fusion", "fusion_prf", "rrf", "fusion_learned", "fusion_rerank"}
NEEDS_RERANKER = {"fusion_rerank"}


def rank_of(ranked_ids: list[int], target_id: int) -> Optional[int]:
    """Posição 1-based de `target_id` em `ranked_ids`, ou None se não estiver lá."""
    try:
        return ranked_ids.index(int(target_id)) + 1
    except ValueError:
        return None
