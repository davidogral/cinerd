# -*- coding: utf-8 -*-
"""core.query_llm: nunca deve quebrar/bloquear a busca, mesmo sem chave, com
rede fora, timeout ou resposta malformada do Groq."""

import json

import pytest


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Cache em disco isolado por teste — não toca data/tmdb_cache/ real nem
    vaza estado entre testes (os caches são globais do módulo; vários testes
    de rerank_confirm reusam a MESMA consulta/candidatos, então sem isolar o
    _rerank_cache um teste veria o resultado cacheado pelo anterior)."""
    from core import query_llm

    monkeypatch.setattr(query_llm, "_cache", None)
    monkeypatch.setattr(query_llm, "_rerank_cache", None)
    monkeypatch.setattr("core.db._PROJECT_ROOT", str(tmp_path))
    yield
    monkeypatch.setattr(query_llm, "_cache", None)
    monkeypatch.setattr(query_llm, "_rerank_cache", None)


def test_understand_default_when_not_configured(monkeypatch):
    from core import query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "")
    plan = query_llm.understand("qualquer consulta")
    assert plan == query_llm.QueryPlan()
    assert plan.ok is False
    assert plan.tipo == "generico"


def test_understand_empty_query_short_circuits(monkeypatch):
    from core import query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "fake-key")
    assert query_llm.understand("   ").ok is False


def _fake_post(status_code=200, content=None, raise_exc=None):
    def _post(*args, **kwargs):
        if raise_exc:
            raise raise_exc

        class _Resp:
            def __init__(self):
                self.status_code = status_code

            def json(self):
                return {"choices": [{"message": {"content": content}}]}

        return _Resp()

    return _post


def test_understand_parses_object_query(monkeypatch):
    from core import query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "fake-key")
    payload = json.dumps(
        {
            "tipo": "objeto",
            "consulta_reescrita": "Nissan Skyline azul e prata",
            "pistas_pessoa": [],
            "pistas_objeto": ["Nissan Skyline", "azul e prata"],
        }
    )
    monkeypatch.setattr(query_llm.requests, "post", _fake_post(content=payload))

    plan = query_llm.understand("Arrancada com skyline azul e prata")
    assert plan.ok is True
    assert plan.tipo == "objeto"
    assert plan.pistas_objeto == ["Nissan Skyline", "azul e prata"]


def test_understand_caches_successful_result(monkeypatch):
    from core import query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "fake-key")
    calls = {"n": 0}

    def _post(*args, **kwargs):
        calls["n"] += 1
        return _fake_post(content=json.dumps({"tipo": "generico"}))()

    monkeypatch.setattr(query_llm.requests, "post", _post)

    query_llm.understand("mesma consulta")
    query_llm.understand("MESMA CONSULTA")  # normaliza por lower()
    assert calls["n"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"raise_exc": TimeoutError("timeout")},
        {"status_code": 500, "content": "{}"},
        {"content": "isso nao e json"},
        {"content": json.dumps({"tipo": "generico"})[:-1]},  # JSON truncado
    ],
)
def test_understand_never_raises_on_failure(monkeypatch, kwargs):
    from core import query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(query_llm.requests, "post", _fake_post(**kwargs))

    plan = query_llm.understand("consulta qualquer " + str(kwargs))
    assert plan.ok is False
    assert plan.tipo == "generico"


def test_failed_call_is_not_cached(monkeypatch):
    """Falha transiente não deve virar 'sem plano' permanente pra essa consulta."""
    from core import query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(query_llm.requests, "post", _fake_post(raise_exc=RuntimeError("down")))
    query_llm.understand("consulta instavel")
    assert "consulta instavel" not in query_llm._get_cache()


def test_rewrite_query_passthrough_without_llm(monkeypatch):
    from core import inference_client, query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "")
    q = "carro azul e prata dando arrancada numa corrida de rua a noite"
    assert inference_client._understand_and_rewrite(q) == (q, [], None, None, "generico")
    assert inference_client._understand_and_rewrite("") == ("", [], None, None, "generico")


def test_rewrite_query_appends_pistas_objeto_never_replaces(monkeypatch):
    """Achado 2026-09-08: SUBSTITUIR a consulta por só as pistas encolhe o
    texto o bastante pra disparar por engano a heurística de 'consulta curta
    = título' (uma consulta de musical virou só 'bar' e passou a casar com
    qualquer título contendo essa substring). Corrigido pra só ACRESCENTAR.
    tipo="objeto" também devolve o peso maior do canal léxico de enredo e
    desliga o de personagem, só pra essa busca (achado: os termos
    acrescentados tipo "Dodge Charger" davam falso-positivo no canal de
    personagem, afinado pra consulta curta de nome de gente)."""
    from core import inference_client, query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "fake-key")
    payload = json.dumps(
        {
            "tipo": "objeto",
            "consulta_reescrita": "nao devia aparecer no resultado",
            "pistas_objeto": ["Nissan Skyline GT-R"],
        }
    )
    monkeypatch.setattr(query_llm.requests, "post", _fake_post(content=payload))

    q = "carro azul e prata dando arrancada numa corrida de rua a noite"
    out, pistas_pessoa, plot_lexical_weight, entity_weight, tipo = inference_client._understand_and_rewrite(q)
    assert out.startswith(q)  # original PRESERVADO, nunca substituído
    assert "Nissan Skyline GT-R" in out
    assert "nao devia aparecer" not in out  # consulta_reescrita não é usada
    assert pistas_pessoa == []
    assert plot_lexical_weight == inference_client.OBJECT_PLOT_LEXICAL_WEIGHT
    assert entity_weight == 0.0
    assert tipo == "objeto"


def test_rewrite_query_short_query_still_calls_groq_but_stays_safe(monkeypatch):
    """Achado 2026-09-08: o limiar de palavras só existia pra evitar a
    substituição (já removida, ver teste acima) — sem ela, consulta curta de
    NOME (typo tolerado pelo canal entity) classifica como "pessoa"/
    "generico", nunca "objeto", então nunca dispara acréscimo nem peso maior.
    Isso libera consulta curta de OBJETO real ("Dogde charger preto", 3
    palavras) a se beneficiar — hoje nem chegava a passar pelo Groq."""
    from core import inference_client, query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "fake-key")
    payload = json.dumps({"tipo": "pessoa", "pistas_pessoa": ["plays a character named McQueen"]})
    monkeypatch.setattr(query_llm.requests, "post", _fake_post(content=payload))

    out, pistas_pessoa, plot_lexical_weight, entity_weight, tipo = inference_client._understand_and_rewrite("Mcquen")
    assert out == "Mcquen"  # tipo != objeto -> texto não muda
    assert plot_lexical_weight is None  # e não ganha o peso maior
    assert entity_weight is None  # canal de personagem continua ligado
    assert tipo == "pessoa"


def test_rewrite_query_generico_does_not_change_query(monkeypatch):
    """tipo=generico ainda não tem uma forma segura validada de melhorar a
    consulta (ver docstring de _understand_and_rewrite) — por ora só loga/
    decompõe, não altera o texto buscado nem os pesos."""
    from core import inference_client, query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "fake-key")
    payload = json.dumps({"tipo": "generico", "consulta_reescrita": "consulta limpa e curta"})
    monkeypatch.setattr(query_llm.requests, "post", _fake_post(content=payload))

    q = "consulta bem barroca e cheia de enrolacao mas ainda assim descritiva"
    assert inference_client._understand_and_rewrite(q) == (q, [], None, None, "generico")


def test_rewrite_query_returns_pistas_pessoa_for_tipo_pessoa(monkeypatch):
    """tipo=pessoa não altera o texto buscado nem os pesos (mesma cautela do
    genérico), mas devolve pistas_pessoa pro canal person_match do
    search_engine."""
    from core import inference_client, query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "fake-key")
    payload = json.dumps(
        {
            "tipo": "pessoa",
            "consulta_reescrita": "nao devia importar aqui",
            "pistas_pessoa": ["decorated by the Queen", "had a heavy metal band"],
        }
    )
    monkeypatch.setattr(query_llm.requests, "post", _fake_post(content=payload))

    q = "ator condecorado pela rainha da inglaterra que tinha uma banda de heavy metal"
    out, pistas_pessoa, plot_lexical_weight, entity_weight, tipo = inference_client._understand_and_rewrite(q)
    assert out == q  # texto da busca não muda
    assert plot_lexical_weight is None
    assert entity_weight is None
    assert pistas_pessoa == ["decorated by the Queen", "had a heavy metal band"]
    assert tipo == "pessoa"


# =========================================================== rerank_confirm

_CANDIDATES = [
    {"tmdb_id": 101, "title": "Filme A", "year": 2001, "overview": "sinopse A"},
    {"tmdb_id": 102, "title": "Filme B", "year": 2002, "overview": "sinopse B"},
    {"tmdb_id": 103, "title": "Filme C", "year": 2003, "overview": "sinopse C"},
]


def test_rerank_confirm_promotes_confident_choice(monkeypatch):
    from core import query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "fake-key")
    payload = json.dumps({"confirmados": [{"n": 2, "confianca": "alta"}]})
    monkeypatch.setattr(query_llm.requests, "post", _fake_post(content=payload))

    assert query_llm.rerank_confirm("descrição qualquer", _CANDIDATES) == [102]


def test_rerank_confirm_returns_multiple_in_confidence_order(monkeypatch):
    """Duas respostas plausíveis pra mesma consulta (ex.: "sonho dentro do
    sonho" bate tanto com A Origem quanto com O Discreto Charme da
    Burguesia) devem promover as duas, na ordem de confiança devolvida."""
    from core import query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "fake-key")
    payload = json.dumps({"confirmados": [{"n": 3, "confianca": "alta"}, {"n": 1, "confianca": "media"}]})
    monkeypatch.setattr(query_llm.requests, "post", _fake_post(content=payload))

    assert query_llm.rerank_confirm("descrição qualquer", _CANDIDATES) == [103, 101]


@pytest.mark.parametrize(
    "payload",
    [
        json.dumps({"confirmados": []}),  # não achou nenhum
        json.dumps({"confirmados": [{"n": 2, "confianca": "baixa"}]}),  # confiança baixa nunca promove
        json.dumps({"confirmados": [{"n": 99, "confianca": "alta"}]}),  # índice fora da lista
        json.dumps({"confirmados": "não é uma lista"}),  # formato inesperado
        "isso nao e json",
    ],
)
def test_rerank_confirm_returns_empty_without_confident_valid_choice(monkeypatch, payload):
    from core import query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(query_llm.requests, "post", _fake_post(content=payload))

    assert query_llm.rerank_confirm("descrição qualquer", _CANDIDATES) == []


def test_rerank_confirm_dedupes_repeated_index(monkeypatch):
    from core import query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "fake-key")
    payload = json.dumps({"confirmados": [{"n": 2, "confianca": "alta"}, {"n": 2, "confianca": "media"}]})
    monkeypatch.setattr(query_llm.requests, "post", _fake_post(content=payload))

    assert query_llm.rerank_confirm("descrição qualquer", _CANDIDATES) == [102]


def test_rerank_confirm_never_raises_on_failure(monkeypatch):
    from core import query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(query_llm.requests, "post", _fake_post(raise_exc=TimeoutError("timeout")))

    assert query_llm.rerank_confirm("descrição qualquer", _CANDIDATES) == []


def test_rerank_confirm_empty_candidates_or_no_key(monkeypatch):
    from core import query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "fake-key")
    assert query_llm.rerank_confirm("descrição qualquer", []) == []

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "")
    assert query_llm.rerank_confirm("descrição qualquer", _CANDIDATES) == []


def test_rerank_confirm_caches_result_same_query_and_candidates(monkeypatch):
    """Achado 2026-09-08: sem cache, a MESMA consulta com os MESMOS
    candidatos dava resposta diferente a cada chamada (8 chamadas idênticas
    -> 3 respostas distintas) — a Groq não é perfeitamente determinística
    mesmo com temperature=0. Cache faz a mesma busca sempre devolver a
    mesma resposta a partir da 1ª chamada real."""
    from core import query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "fake-key")
    calls = {"n": 0}

    def _post(*args, **kwargs):
        calls["n"] += 1
        return _fake_post(content=json.dumps({"confirmados": [{"n": 2, "confianca": "alta"}]}))()

    monkeypatch.setattr(query_llm.requests, "post", _post)

    for _ in range(5):
        assert query_llm.rerank_confirm("descrição qualquer", _CANDIDATES) == [102]
    assert calls["n"] == 1  # só a 1ª chamada bateu na rede


def test_rerank_confirm_cache_key_includes_candidate_ids(monkeypatch):
    """Mesma consulta, candidatos DIFERENTES (ex.: índice mudou) não deve
    reusar o cache de outro conjunto de candidatos."""
    from core import query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "fake-key")
    calls = {"n": 0}

    def _post(*args, **kwargs):
        calls["n"] += 1
        return _fake_post(content=json.dumps({"confirmados": [{"n": 1, "confianca": "alta"}]}))()

    monkeypatch.setattr(query_llm.requests, "post", _post)

    outros_candidatos = [{"tmdb_id": 201, "title": "Filme D", "year": 2004, "overview": "sinopse D"}]
    query_llm.rerank_confirm("descrição qualquer", _CANDIDATES)
    query_llm.rerank_confirm("descrição qualquer", outros_candidatos)
    assert calls["n"] == 2


def test_rerank_confirm_failure_is_not_cached(monkeypatch):
    """Falha transiente de rede não deve virar 'nenhum confirmado' permanente."""
    from core import query_llm

    monkeypatch.setattr(query_llm, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(query_llm.requests, "post", _fake_post(raise_exc=RuntimeError("down")))

    query_llm.rerank_confirm("descrição instável", _CANDIDATES)
    key = query_llm._rerank_cache_key("descrição instável", _CANDIDATES)
    assert key not in query_llm._get_rerank_cache()


# ============================================================ _llm_rerank


def _results_from_candidates():
    return [
        {"tmdb_id": 101, "title": "Filme A", "release_year": 2001, "overview": "sinopse A"},
        {"tmdb_id": 102, "title": "Filme B", "release_year": 2002, "overview": "sinopse B"},
        {"tmdb_id": 103, "title": "Filme C", "release_year": 2003, "overview": "sinopse C"},
    ]


def test_llm_rerank_promotes_pick_to_front(monkeypatch):
    from core import inference_client, query_llm

    # _llm_rerank importa `query_llm` localmente a cada chamada — patchar o
    # módulo canônico (não um atributo de inference_client) é o que afeta essa
    # importação local.
    monkeypatch.setattr(query_llm, "rerank_confirm", lambda q, c: [103])
    monkeypatch.setattr("core.catalog.get_movie", lambda tid: {"overview": "sinopse completa"})

    out = inference_client._llm_rerank("descrição", _results_from_candidates())
    assert [r["tmdb_id"] for r in out] == [103, 101, 102]


def test_llm_rerank_promotes_multiple_confirmed_in_order(monkeypatch):
    """Duas respostas plausíveis (ex.: "sonho dentro do sonho") devem as
    duas subir, na ordem de confiança devolvida pela LLM — não só a
    primeira, deixando a outra onde a fusão a tivesse deixado."""
    from core import inference_client, query_llm

    monkeypatch.setattr(query_llm, "rerank_confirm", lambda q, c: [103, 101])
    monkeypatch.setattr("core.catalog.get_movie", lambda tid: {"overview": "sinopse completa"})

    out = inference_client._llm_rerank("descrição", _results_from_candidates())
    assert [r["tmdb_id"] for r in out] == [103, 101, 102]


def test_llm_rerank_leaves_order_when_nothing_confirmed(monkeypatch):
    from core import inference_client, query_llm

    monkeypatch.setattr(query_llm, "rerank_confirm", lambda q, c: [])
    monkeypatch.setattr("core.catalog.get_movie", lambda tid: {"overview": "sinopse completa"})

    original = _results_from_candidates()
    out = inference_client._llm_rerank("descrição", list(original))
    assert [r["tmdb_id"] for r in out] == [r["tmdb_id"] for r in original]


def test_llm_rerank_noop_without_query_or_with_single_result(monkeypatch):
    from core import inference_client, query_llm

    called = {"n": 0}
    monkeypatch.setattr(query_llm, "rerank_confirm", lambda q, c: called.__setitem__("n", called["n"] + 1))

    assert inference_client._llm_rerank("", _results_from_candidates()) == _results_from_candidates()
    assert inference_client._llm_rerank("descrição", _results_from_candidates()[:1]) == _results_from_candidates()[:1]
    assert called["n"] == 0


# =================================================== _pull_franchise_siblings


def test_pull_franchise_siblings_noop_without_collection(monkeypatch):
    from core import inference_client, tmdb

    monkeypatch.setattr(tmdb, "movie_details", lambda tid: {"collection": None})

    original = _results_from_candidates()
    out = inference_client._pull_franchise_siblings(list(original))
    assert [r["tmdb_id"] for r in out] == [r["tmdb_id"] for r in original]


def test_pull_franchise_siblings_noop_without_tmdb(monkeypatch):
    from core import inference_client, tmdb

    monkeypatch.setattr(tmdb, "movie_details", lambda tid: None)

    original = _results_from_candidates()
    out = inference_client._pull_franchise_siblings(list(original))
    assert [r["tmdb_id"] for r in out] == [r["tmdb_id"] for r in original]


def test_pull_franchise_siblings_noop_empty_results(monkeypatch):
    from core import inference_client

    assert inference_client._pull_franchise_siblings([]) == []


def test_pull_franchise_siblings_inserts_missing_sibling_from_catalog(monkeypatch):
    """O #1 é o 1º filme de uma franquia; o 2º (o que o usuário queria) não
    estava em `results` (a fusão nunca o achou), mas existe no catálogo."""
    from core import catalog, inference_client, tmdb

    monkeypatch.setattr(tmdb, "movie_details", lambda tid: {"collection": {"id": 999, "name": "Franquia X"}})
    monkeypatch.setattr(
        tmdb,
        "collection_movies",
        lambda cid: {
            "id": cid,
            "name": "Franquia X",
            "parts": [
                {"tmdb_id": 101, "title": "Filme A", "release_year": 2001},
                {"tmdb_id": 999101, "title": "Filme A 2", "release_year": 2003},
            ],
        },
    )
    monkeypatch.setattr(
        catalog,
        "get_movie",
        lambda tid: (
            {
                "title": "Filme A 2",
                "release_year": 2003,
                "original_language": "en",
                "vote_average": 6.5,
                "overview": "sinopse do segundo filme",
            }
            if tid == 999101
            else None
        ),
    )

    original = _results_from_candidates()  # tmdb_ids 101, 102, 103
    out = inference_client._pull_franchise_siblings(list(original))
    assert [r["tmdb_id"] for r in out] == [101, 999101, 102, 103]
    assert out[1]["why"] == ["Mesma franquia"]


def test_pull_franchise_siblings_moves_existing_sibling_up(monkeypatch):
    """O sibling já estava em `results`, só que mais abaixo — deve subir pra
    logo depois do #1, sem duplicar."""
    from core import inference_client, tmdb

    monkeypatch.setattr(tmdb, "movie_details", lambda tid: {"collection": {"id": 999, "name": "Franquia X"}})
    monkeypatch.setattr(
        tmdb,
        "collection_movies",
        lambda cid: {
            "id": cid,
            "name": "Franquia X",
            "parts": [
                {"tmdb_id": 101, "title": "Filme A", "release_year": 2001},
                {"tmdb_id": 103, "title": "Filme A 3", "release_year": 2003},
            ],
        },
    )

    original = _results_from_candidates()  # 101, 102, 103
    out = inference_client._pull_franchise_siblings(list(original))
    assert [r["tmdb_id"] for r in out] == [101, 103, 102]
    assert "why" not in out[1]  # já vinha de `results`, mantém os campos originais


def test_pull_franchise_siblings_keeps_known_order_over_release_date(monkeypatch):
    """Achado 2026-09-09 (regressão medida em produção): a 1ª versão ordenava
    TUDO por lançamento, então um sibling mais antigo mas irrelevante pra
    aquela consulta passava na frente de um sibling que a fusão/reranking já
    tinham ranqueado bem (ex.: "Hobbs e Toreto" tinha o filme certo em #3 e
    caiu pra #7). Aqui: 103 (2003) já vinha ANTES de 102 (2002) em
    `results` (ou seja, a fusão já preferia 103), mas 102 tem lançamento
    mais antigo. O "known" (103) deve manter sua posição relativa acima do
    "unknown" (999, fora de `results`, sem sinal nenhum), mesmo que a
    ordem de lançamento diga o contrário."""
    from core import catalog, inference_client, tmdb

    monkeypatch.setattr(tmdb, "movie_details", lambda tid: {"collection": {"id": 999, "name": "Franquia X"}})
    monkeypatch.setattr(
        tmdb,
        "collection_movies",
        lambda cid: {
            "id": cid,
            "name": "Franquia X",
            "parts": [
                {"tmdb_id": 999001, "title": "Filme A 0 (mais antigo)", "release_year": 1999},
                {"tmdb_id": 103, "title": "Filme A 3", "release_year": 2003},
            ],
        },
    )
    monkeypatch.setattr(
        catalog,
        "get_movie",
        lambda tid: (
            {
                "title": "Filme A 0 (mais antigo)",
                "release_year": 1999,
                "original_language": "en",
                "vote_average": 5.0,
                "overview": "",
            }
            if tid == 999001
            else None
        ),
    )

    original = [
        {"tmdb_id": 101, "title": "Filme A", "release_year": 2001},
        {"tmdb_id": 103, "title": "Filme A 3", "release_year": 2003},  # já ranqueado acima na fusão
        {"tmdb_id": 102, "title": "Filme A 2", "release_year": 2002},
    ]
    out = inference_client._pull_franchise_siblings(list(original))
    # 103 (known, já vinha bem ranqueado) fica ANTES de 999001 (unknown,
    # mais antigo mas sem sinal de relevância nenhum) mesmo sendo mais novo.
    assert [r["tmdb_id"] for r in out] == [101, 103, 999001, 102]


def test_pull_franchise_siblings_respects_max_cap(monkeypatch):
    from core import catalog, inference_client, tmdb

    monkeypatch.setattr(inference_client, "FRANCHISE_PULLUP_MAX", 1)
    monkeypatch.setattr(tmdb, "movie_details", lambda tid: {"collection": {"id": 999, "name": "Franquia X"}})
    monkeypatch.setattr(
        tmdb,
        "collection_movies",
        lambda cid: {
            "id": cid,
            "name": "Franquia X",
            "parts": [
                {"tmdb_id": 101, "title": "Filme A", "release_year": 2001},
                {"tmdb_id": 201, "title": "Filme A 2", "release_year": 2003},
                {"tmdb_id": 202, "title": "Filme A 3", "release_year": 2005},
            ],
        },
    )
    monkeypatch.setattr(
        catalog,
        "get_movie",
        lambda tid: {
            "title": f"Filme {tid}",
            "release_year": 2000,
            "original_language": "en",
            "vote_average": 6.0,
            "overview": "",
        },
    )

    original = _results_from_candidates()
    out = inference_client._pull_franchise_siblings(list(original))
    # só 1 sibling novo (o cap), + o #1 + o resto original
    assert [r["tmdb_id"] for r in out] == [101, 201, 102, 103]
