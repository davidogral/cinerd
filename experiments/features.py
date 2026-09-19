# -*- coding: utf-8 -*-
"""Sinais **baratos e disponíveis antes da chamada** ao LLM (protocolo §4/§7.2).

A tese do estudo depende inteiramente disto: a política só pode usar informação
que já existe *antes* de pagar pela verificação. Toda função aqui é
determinística, sem rede e calculada a partir de (i) o texto da consulta, (ii) o
ranking do primeiro estágio e (iii) o texto de evidência dos candidatos — nada
que dependa da resposta que se quer prever.

Quatro grupos, como no plano:

| grupo | exemplos |
|---|---|
| consulta | comprimento, idioma, nome próprio, termo raro (idf), dígitos, erro de grafia |
| ranking inicial | margem top-1/top-2, entropia dos scores, desacordo entre canais, posição por canal |
| evidência | texto disponível por candidato, cobertura de pistas literais, conflito entre candidatos |
| operação | tokens previstos, custo/latência estimados |

Um erro fácil de cometer aqui é incluir, sem perceber, algo que só se sabe
depois da chamada (por exemplo "o LLM confirmou algum candidato"). Qualquer
função nova precisa receber só `query`, `order/scores` do primeiro estágio e os
textos dos candidatos.
"""

from __future__ import annotations

import math
import re
from typing import Optional, Sequence

import numpy as np

from retrieval.search_engine import _content_tokens, _strip_accents

# Marcadores gramaticais frequentes em pt-BR e ausentes do inglês (e vice-versa).
_PT_MARKERS = {"que", "uma", "não", "com", "para", "dos", "das", "seu", "sua", "ele", "ela", "onde", "quem", "sobre"}
_EN_MARKERS = {"the", "and", "with", "that", "who", "where", "his", "her", "about", "from", "into", "they"}
_WORD = re.compile(r"[A-Za-zÀ-ÿ]+")


def _norm(text: str) -> str:
    return _strip_accents((text or "").lower())


def _tokens(text: str) -> list[str]:
    return _WORD.findall(_norm(text))


# ------------------------------------------------------------------ consulta


def query_features(query: str, engine=None) -> dict:
    raw = query or ""
    toks = _tokens(raw)
    content = _content_tokens(raw)
    words = raw.split()
    # nome próprio: maiúscula fora da primeira palavra (o usuário digita "Toretto",
    # "Dodge Charger" assim mesmo quando escreve o resto em minúscula).
    caps_mid = sum(1 for w in words[1:] if w[:1].isupper())
    pt = sum(1 for t in toks if t in {_strip_accents(m) for m in _PT_MARKERS})
    en = sum(1 for t in toks if t in _EN_MARKERS)

    feats = {
        "q_n_words": len(words),
        "q_n_chars": len(raw),
        "q_n_content_tokens": len(content),
        "q_has_digit": 1.0 if any(c.isdigit() for c in raw) else 0.0,
        "q_caps_mid": float(caps_mid),
        "q_lang_pt": 1.0 if pt >= en else 0.0,
        "q_lang_margin": float(pt - en),
        # proxy de erro de grafia: token fora do vocabulário do índice lexical.
        "q_oov_ratio": 0.0,
        "q_max_idf": 0.0,
        "q_mean_idf": 0.0,
    }
    if engine is not None and getattr(engine, "_bm25", None) is not None:
        vocab = engine._bm25.vectorizer.vocabulary_
        idf = engine._bm25.idf
        known = [vocab[t] for t in content if t in vocab]
        feats["q_oov_ratio"] = round(1.0 - (len(known) / len(content)), 4) if content else 0.0
        if known:
            vals = [float(idf[i]) for i in known]
            feats["q_max_idf"] = round(max(vals), 4)
            feats["q_mean_idf"] = round(sum(vals) / len(vals), 4)
    return feats


# ------------------------------------------------------------- ranking inicial


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def ranking_features(scores: np.ndarray, raw: Optional[dict[str, np.ndarray]] = None, top: int = 20) -> dict:
    """`scores`: vetor fundido do primeiro estágio (alinhado ao índice).
    `raw`: canais sem peso (`eval.pipelines.raw_channels`), para medir desacordo."""
    order = np.argsort(scores, kind="stable")[::-1][:top]
    s = scores[order].astype(np.float64)
    s1 = float(s[0]) if s.size else 0.0
    s2 = float(s[1]) if s.size > 1 else 0.0
    s5 = float(s[4]) if s.size > 4 else s2
    denom = abs(s1) if abs(s1) > 1e-9 else 1.0
    soft = np.exp(s - s.max())
    soft = soft / soft.sum() if soft.sum() > 0 else soft

    feats = {
        "r_top1": round(s1, 4),
        "r_margin_1_2": round(s1 - s2, 4),
        "r_margin_1_2_rel": round((s1 - s2) / denom, 4),
        "r_margin_1_5_rel": round((s1 - s5) / denom, 4),
        "r_entropy_top20": round(_entropy(soft), 4),
        "r_std_top20": round(float(s.std()), 4),
    }
    if raw:
        tops: dict[str, int] = {}
        active = 0
        for name, vec in raw.items():
            if not np.any(vec > 0):
                continue
            active += 1
            tops[name] = int(np.argmax(vec))
        feats["r_channels_active"] = float(active)
        # desacordo: quantos canais discordam do #1 da fusão, e quantos #1 distintos.
        best = int(order[0]) if order.size else -1
        feats["r_channels_agree_top1"] = float(sum(1 for i in tops.values() if i == best))
        feats["r_distinct_channel_top1"] = float(len(set(tops.values())))
        lex, sem = raw.get("lexical"), raw.get("synopsis")
        if lex is not None and sem is not None:
            lex_top = set(np.argsort(lex, kind="stable")[::-1][:top].tolist())
            sem_top = set(np.argsort(sem, kind="stable")[::-1][:top].tolist())
            inter = len(lex_top & sem_top)
            feats["r_lex_sem_overlap"] = round(inter / top, 4)
    return feats


# ----------------------------------------------------------------- evidência


def cue_coverage(query: str, text: str) -> float:
    """Fração dos tokens de conteúdo da consulta que aparecem literalmente no
    texto do candidato. É o proxy determinístico de "a evidência textual sequer
    contém o fato citado" — o rótulo que separa erro de evidência de erro de
    julgamento (protocolo §3)."""
    content = set(_content_tokens(query))
    if not content:
        return 0.0
    hay = set(_tokens(text))
    return round(len(content & hay) / len(content), 4)


def evidence_features(query: str, candidates: Sequence[dict], top: int = 20) -> dict:
    """`candidates`: [{"tmdb_id", "title", "year", "overview"}, ...] na ordem do
    primeiro estágio — exatamente o que seria mandado ao verificador."""
    cands = list(candidates)[:top]
    if not cands:
        return {
            "e_n_candidates": 0.0,
            "e_mean_chars": 0.0,
            "e_min_chars": 0.0,
            "e_empty_ratio": 1.0,
            "e_cov_top1": 0.0,
            "e_cov_max": 0.0,
            "e_cov_mean": 0.0,
            "e_cov_gap": 0.0,
            "e_n_cov_ge_half": 0.0,
        }
    texts = [(c.get("overview") or "") for c in cands]
    lens = [len(t) for t in texts]
    covs = [cue_coverage(query, t) for t in texts]
    top1 = covs[0]
    best = max(covs)
    return {
        "e_n_candidates": float(len(cands)),
        "e_mean_chars": round(sum(lens) / len(lens), 1),
        "e_min_chars": float(min(lens)),
        "e_empty_ratio": round(sum(1 for x in lens if x < 40) / len(lens), 4),
        "e_cov_top1": top1,
        "e_cov_max": best,
        "e_cov_mean": round(sum(covs) / len(covs), 4),
        # quanto o melhor candidato por evidência textual supera o #1 do ranking:
        # positivo = há texto melhor abaixo, que é justamente onde a confirmação
        # tem chance de consertar o ranking (H1).
        "e_cov_gap": round(best - top1, 4),
        "e_n_cov_ge_half": float(sum(1 for c in covs if c >= 0.5)),
    }


# ------------------------------------------------------------------ operação


def cost_features(query: str, candidates: Sequence[dict], top: int = 20, chars_per_token: float = 3.6) -> dict:
    """Tamanho do *prompt* que a chamada teria. `chars_per_token` é uma estimativa
    grosseira; o valor medido de verdade vem do *ledger* (`llm_client`)."""
    cands = list(candidates)[:top]
    body = sum(len(c.get("title") or "") + len(str(c.get("year") or "")) + len(c.get("overview") or "") for c in cands)
    chars = len(query) + body
    return {
        "o_prompt_chars": float(chars),
        "o_est_tokens_in": round(chars / chars_per_token, 1),
        "o_n_sent": float(len(cands)),
    }


def all_features(query: str, scores: np.ndarray, candidates: Sequence[dict], raw=None, engine=None) -> dict:
    """Vetor completo de sinais pré-chamada, achatado num dicionário plano."""
    out: dict = {}
    out.update(query_features(query, engine=engine))
    out.update(ranking_features(scores, raw=raw))
    out.update(evidence_features(query, candidates))
    out.update(cost_features(query, candidates))
    return out


FEATURE_GROUPS = {
    "consulta": "q_",
    "ranking": "r_",
    "evidencia": "e_",
    "operacao": "o_",
}


def group_of(name: str) -> str:
    for group, prefix in FEATURE_GROUPS.items():
        if name.startswith(prefix):
            return group
    return "outro"


def ndcg_at(rank: Optional[int], k: int = 10) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)
