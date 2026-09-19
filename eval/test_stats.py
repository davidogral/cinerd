# -*- coding: utf-8 -*-
"""`eval.stats`: o intervalo tem que ser determinístico e cobrir os casos-limite
que aparecem de verdade aqui (muita consulta empatada, diferença nula, n pequeno)."""

from __future__ import annotations

import pytest

from eval.stats import holm, metric_fn, paired_bootstrap, power_curve


def test_metricas_conhecidas():
    assert metric_fn("success@1")(1) == 1.0
    assert metric_fn("success@1")(2) == 0.0
    assert metric_fn("recall@10")(10) == 1.0  # success@k == recall@k (1 relevante)
    assert metric_fn("mrr")(4) == pytest.approx(0.25)
    assert metric_fn("mrr")(None) == 0.0
    assert metric_fn("ndcg@10")(1) == 1.0
    assert metric_fn("ndcg@10")(11) == 0.0
    with pytest.raises(ValueError):
        metric_fn("map@10")


def test_diferenca_nula_tem_ic_degenerado():
    a = [1.0, 0.0, 1.0, 0.5]
    r = paired_bootstrap(a, a, n_boot=200)
    assert r["delta"] == 0.0
    assert r["ci_low"] == 0.0 and r["ci_high"] == 0.0
    assert (r["wins"], r["losses"], r["ties"]) == (0, 0, 4)


def test_ic_captura_diferenca_constante():
    a = [0.0] * 20
    b = [1.0] * 20
    r = paired_bootstrap(a, b, n_boot=500)
    assert r["delta"] == 1.0
    assert r["ci_low"] == 1.0 and r["ci_high"] == 1.0
    assert r["p_value"] <= 2.0 / 500


def test_determinismo_mesma_semente():
    a = [0.0, 1.0, 0.0, 1.0, 0.5, 0.2, 0.9, 0.1]
    b = [1.0, 1.0, 0.0, 0.0, 0.7, 0.2, 0.4, 0.3]
    r1 = paired_bootstrap(a, b, n_boot=300, seed=7)
    r2 = paired_bootstrap(a, b, n_boot=300, seed=7)
    r3 = paired_bootstrap(a, b, n_boot=300, seed=8)
    assert r1 == r2
    assert (r1["ci_low"], r1["ci_high"]) != (r3["ci_low"], r3["ci_high"])


def test_cluster_reduz_unidades_independentes():
    # 8 consultas, 2 clusters: a reamostragem sorteia 2 unidades, não 8.
    a = [0.0] * 8
    b = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    solto = paired_bootstrap(a, b, n_boot=400, seed=3)
    agrupado = paired_bootstrap(a, b, clusters=["x"] * 4 + ["y"] * 4, n_boot=400, seed=3)
    assert solto["n_clusters"] == 8 and agrupado["n_clusters"] == 2
    # menos unidades independentes => intervalo mais largo
    assert agrupado["ci_high"] - agrupado["ci_low"] > solto["ci_high"] - solto["ci_low"]


def test_tamanhos_diferentes_falham():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])


def test_holm_ordena_e_e_monotono():
    adj = holm([0.01, 0.04, 0.03])
    assert adj[0] == pytest.approx(0.03)  # 3 × 0.01
    assert adj[1] >= adj[2] >= adj[0]
    assert all(0.0 <= p <= 1.0 for p in adj)


def test_potencia_cresce_com_n():
    diff = [0.0] * 30 + [1.0] * 5 + [-1.0] * 5
    rows = power_curve(diff, [50, 400], target_delta=0.1, n_sim=40, n_boot=200)
    assert rows[1]["median_ci_halfwidth"] < rows[0]["median_ci_halfwidth"]
    assert rows[1]["power"] >= rows[0]["power"]
