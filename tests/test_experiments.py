# -*- coding: utf-8 -*-
"""`experiments/`: as partes que não chamam rede.

O que estes testes protegem, em ordem de importância:

1. **Nenhum sinal pré-chamada pode depender da resposta do LLM.** É a premissa
   inteira da RQ2; se vazar, a política "aprende" a prever o que já sabe.
2. A contabilidade de uma política (qualidade, dano, fração acionada) tem que
   bater com o que se calcula à mão num exemplo pequeno.
3. Os critérios de exclusão da coleta valem ANTES de olhar resultado, então
   precisam ser determinísticos e testáveis isoladamente.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments import collect, features, policy

# ------------------------------------------------------------------ features


def test_sinais_nao_dependem_da_resposta_do_llm():
    """Assinatura fechada: só consulta, ranking e texto dos candidatos entram."""
    query = "um homem com tatuagens caca o assassino da esposa"
    scores = np.array([5.0, 4.0, 1.0, 0.5, 0.1])
    cands = [{"tmdb_id": i, "title": f"F{i}", "year": 2000, "overview": "tatuagens e vinganca"} for i in range(5)]
    feats = features.all_features(query, scores, cands)
    proibidos = ("pick", "confirm", "rank_after", "gain", "llm", "resposta")
    assert not [k for k in feats if any(p in k.lower() for p in proibidos)]
    assert all(isinstance(v, (int, float)) for v in feats.values())


def test_margem_separa_ranking_seguro_de_inseguro():
    seguro = features.ranking_features(np.array([10.0, 1.0, 0.9, 0.8, 0.7]))
    inseguro = features.ranking_features(np.array([10.0, 9.9, 9.8, 9.7, 9.6]))
    assert seguro["r_margin_1_2_rel"] > inseguro["r_margin_1_2_rel"]
    assert seguro["r_entropy_top20"] < inseguro["r_entropy_top20"]


def test_cobertura_de_pista_e_fracao_de_tokens_de_conteudo():
    assert features.cue_coverage("homem tatuagens memorias", "Um homem com tatuagens e amnésia.") == pytest.approx(
        2 / 3, abs=0.01
    )
    assert features.cue_coverage("nada em comum aqui", "texto totalmente diferente") == 0.0
    assert features.cue_coverage("", "qualquer coisa") == 0.0


def test_cov_gap_aponta_evidencia_melhor_abaixo_do_topo():
    """`e_cov_gap` > 0 é a situação da hipótese H1: há candidato abaixo do #1
    cujo texto cobre melhor as pistas da consulta."""
    query = "carro vermelho explode na ponte"
    cands = [
        {"tmdb_id": 1, "title": "A", "year": 2000, "overview": "um drama familiar"},
        {"tmdb_id": 2, "title": "B", "year": 2001, "overview": "um carro vermelho explode na ponte"},
    ]
    feats = features.evidence_features(query, cands)
    assert feats["e_cov_gap"] > 0
    assert feats["e_cov_max"] > feats["e_cov_top1"]


def test_texto_ausente_conta_como_evidencia_vazia():
    cands = [{"tmdb_id": 1, "title": "A", "year": 2000, "overview": ""}] * 3
    assert features.evidence_features("qualquer consulta", cands)["e_empty_ratio"] == 1.0


# -------------------------------------------------------------------- policy


def _rows(gains, cost=0.001):
    return [
        {
            "qid": f"q{i}",
            "split": "t",
            "features": {"r_margin_1_2_rel": 0.5 - i * 0.1},
            "before": 0.0,
            "after": float(g > 0),
            "gain": float(g),
            "cost_usd": cost,
            "latency_ms": 500.0,
            "error_type": "ok",
            "false_confirmation": False,
        }
        for i, g in enumerate(gains)
    ]


def test_contabilidade_de_politica_bate_na_mao():
    rows = _rows([1, 1, 0, -1])  # 2 ajudam, 1 neutro, 1 piora
    tudo = policy.evaluate(rows, np.ones(4, dtype=bool), lam=0.0, gamma=0.0, beta=0.0)
    assert tudo["fraction_called"] == 1.0
    assert tudo["helped"] == 2 and tudo["hurt"] == 1
    nada = policy.evaluate(rows, np.zeros(4, dtype=bool), lam=0.0, gamma=0.0, beta=0.0)
    assert nada["fraction_called"] == 0.0
    assert nada["cost_usd"] == 0.0
    assert nada["delta_vs_never"] == 0.0


def test_oraculo_e_teto_e_nunca_e_piso():
    rows = _rows([1, 1, 0, -1])
    oraculo = policy.evaluate(rows, policy.policy_oracle(rows), 0.0, 0.0, 0.0)
    sempre = policy.evaluate(rows, policy.policy_always(rows), 0.0, 0.0, 0.0)
    nunca = policy.evaluate(rows, policy.policy_never(rows), 0.0, 0.0, 0.0)
    assert oraculo["quality"] >= sempre["quality"] >= 0
    assert oraculo["hurt"] == 0  # o oráculo nunca aciona onde faz mal
    assert nunca["quality"] <= oraculo["quality"]


def test_aleatoria_respeita_a_fracao_pedida():
    rows = _rows([1] * 10)
    call = policy.policy_random(rows, 0.3, seed=1)
    assert call.sum() == 3
    assert (policy.policy_random(rows, 0.3, seed=1) == call).all()  # determinística


def test_custo_so_conta_consulta_acionada():
    rows = _rows([1, 1, 1, 1], cost=0.01)
    metade = np.array([True, True, False, False])
    r = policy.evaluate(rows, metade, lam=0.0, gamma=0.0, beta=0.0)
    assert r["cost_usd"] == pytest.approx(0.02)


def test_utilidade_penaliza_dano_e_custo():
    rows = _rows([1, -1], cost=0.01)
    sem_peso = policy.evaluate(rows, np.ones(2, dtype=bool), lam=0.0, gamma=0.0, beta=0.0)
    com_peso = policy.evaluate(rows, np.ones(2, dtype=bool), lam=1.0, gamma=0.0, beta=10.0)
    assert com_peso["net_utility"] < sem_peso["net_utility"]


# ------------------------------------------------------------------- collect


@pytest.mark.parametrize(
    "query,motivo",
    [
        ("", "vazia"),
        ("oi", "curta_demais"),
        ("filmes", "curta_demais"),  # curto antes de navegacional: ordem é declarada
        ("me manda em davi@exemplo.com", "pii"),
        ("acesse https://exemplo.com/filme", "pii"),
        ("recomendar filmes bons pra hoje", None),
        ("um homem sem memorias caca o assassino da esposa", None),
    ],
)
def test_criterios_de_exclusao_sao_deterministicos(query, motivo):
    assert collect.classify_exclusion(query) == motivo


def test_pii_e_detectada_antes_de_qualquer_outro_criterio():
    # consulta longa e legítima, mas com e-mail no meio: sai por PII, não entra.
    q = "aquele filme do homem que perde a memoria, manda pra davi@exemplo.com por favor"
    assert collect.find_pii(q) == ["email"]
    assert collect.classify_exclusion(q) == "pii"


def test_sal_deriva_sessao_sem_expor_o_original():
    salt = b"sal-de-teste"
    h1 = collect._sid_hash("sessao-abc", salt)
    h2 = collect._sid_hash("sessao-abc", salt)
    h3 = collect._sid_hash("sessao-abc", b"outro-sal")
    assert h1 == h2 and h1 != h3
    assert "sessao-abc" not in (h1 or "")
    assert collect._sid_hash(None, salt) is None
