# -*- coding: utf-8 -*-
"""Métricas Prometheus do Cinerd.

Expõe, em `/metrics` (tanto na API Flask quanto no serviço de inferência):

  cinerd_requests_total{endpoint,method,status}   contador de requisições
  cinerd_errors_total{endpoint}                   erros não tratados
  cinerd_request_seconds{endpoint}                histograma ponta-a-ponta
  cinerd_stage_seconds{stage}                     histograma por etapa
                                                  (retrieval | rerank | encode | total)
  cinerd_query_cache_events_total{result}         hit | miss do cache de embedding
  cinerd_tmdb_calls_total{result}                 ok | error | capped
  cinerd_process_rss_bytes                        memória residente do processo

Se `prometheus_client` não estiver instalado, tudo vira no-op — o app roda igual.

Além do histograma Prometheus, mantém um **reservatório em memória** dos últimos
`_STAGE_CAP` tempos por etapa (`stage_percentiles()`) — dá p50/p95/p99 exatos do
tráfego real do processo para a página "Engenharia" do site, sem depender de um
scrape do Prometheus. Reinicia a cada boot.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager

# ---- reservatório de amostras por etapa (para percentis exatos ao vivo) ----
_STAGE_CAP = 1024
_STAGE_SAMPLES: "dict[str, deque[float]]" = {}

# ---- captura por requisição (thread-local; isolada por worker gthread) ----
_tl = threading.local()


def _record_stage_sample(stage: str, seconds: float) -> None:
    dq = _STAGE_SAMPLES.get(stage)
    if dq is None:
        dq = _STAGE_SAMPLES[stage] = deque(maxlen=_STAGE_CAP)
    dq.append(seconds * 1000.0)  # guarda em ms
    bucket = getattr(_tl, "stages", None)
    if bucket is not None:
        bucket[stage] = round(bucket.get(stage, 0.0) + seconds * 1000.0, 2)


@contextmanager
def capture_stages():
    """`with capture_stages() as st: ...` → `st` (dict) acumula os ms de cada
    etapa (`retrieval`/`rerank`/`total`…) medida por `stage_timer` dentro do
    escopo. Para logar a latência por etapa de UMA requisição."""
    prev = getattr(_tl, "stages", None)
    bucket: dict[str, float] = {}
    _tl.stages = bucket
    try:
        yield bucket
    finally:
        _tl.stages = prev


def stage_percentiles() -> "dict[str, dict[str, float]]":
    """{stage: {p50, p95, p99, mean, n}} em ms, sobre as amostras recentes."""
    out: dict[str, dict[str, float]] = {}
    for stage, dq in list(_STAGE_SAMPLES.items()):
        xs = sorted(dq)
        n = len(xs)
        if n == 0:
            continue

        def _p(p: float) -> float:
            return xs[min(n - 1, max(0, int(round(p / 100.0 * n)) - 1))]

        out[stage] = {
            "p50": round(_p(50), 2),
            "p95": round(_p(95), 2),
            "p99": round(_p(99), 2),
            "mean": round(sum(xs) / n, 2),
            "n": n,
        }
    return out


try:  # prometheus é opcional
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    _ENABLED = True
except Exception:  # pragma: no cover
    _ENABLED = False
    CONTENT_TYPE_LATEST = "text/plain"


class _Noop:
    def labels(self, *a, **k):
        return self

    def inc(self, *a, **k):
        pass

    def observe(self, *a, **k):
        pass

    def set(self, *a, **k):
        pass

    def set_function(self, *a, **k):
        pass


if _ENABLED:
    _LAT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
    REQUESTS = Counter("cinerd_requests_total", "Requisições HTTP", ["endpoint", "method", "status"])
    ERRORS = Counter("cinerd_errors_total", "Erros não tratados numa requisição", ["endpoint"])
    REQUEST_SECONDS = Histogram("cinerd_request_seconds", "Latência ponta-a-ponta", ["endpoint"], buckets=_LAT_BUCKETS)
    STAGE_SECONDS = Histogram("cinerd_stage_seconds", "Latência por etapa do pipeline", ["stage"], buckets=_LAT_BUCKETS)
    QUERY_CACHE = Counter("cinerd_query_cache_events_total", "Eventos do cache de embedding da consulta", ["result"])
    TMDB_CALLS = Counter("cinerd_tmdb_calls_total", "Chamadas à API da TMDB", ["result"])
    RSS_BYTES = Gauge("cinerd_process_rss_bytes", "Memória residente do processo")
    try:
        import psutil

        _proc = psutil.Process()
        RSS_BYTES.set_function(lambda: float(_proc.memory_info().rss))
    except Exception:  # pragma: no cover
        pass
else:  # pragma: no cover
    REQUESTS = ERRORS = REQUEST_SECONDS = STAGE_SECONDS = QUERY_CACHE = TMDB_CALLS = RSS_BYTES = _Noop()


@contextmanager
def stage_timer(stage: str):
    """`with stage_timer("retrieval"): ...` → observa em cinerd_stage_seconds."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        STAGE_SECONDS.labels(stage=stage).observe(dt)
        _record_stage_sample(stage, dt)


def observe_stage(stage: str, seconds: float) -> None:
    STAGE_SECONDS.labels(stage=stage).observe(seconds)
    _record_stage_sample(stage, seconds)


def query_cache_event(hit: bool) -> None:
    QUERY_CACHE.labels(result="hit" if hit else "miss").inc()


def tmdb_call(result: str) -> None:  # "ok" | "error" | "capped"
    TMDB_CALLS.labels(result=result).inc()


def render_latest() -> tuple[bytes, str]:
    """(corpo, content-type) para a rota /metrics."""
    if not _ENABLED:
        return b"# prometheus_client ausente\n", CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST


# --------------------------------------------------------------------- Flask
def init_flask(app) -> None:
    """Instrumenta uma app Flask: conta requisições/erros/latência e serve /metrics."""
    from flask import g, request

    @app.before_request
    def _m_start():
        g._m_t0 = time.perf_counter()

    @app.after_request
    def _m_end(resp):
        ep = request.endpoint or request.path
        REQUESTS.labels(endpoint=ep, method=request.method, status=resp.status_code).inc()
        t0 = getattr(g, "_m_t0", None)
        if t0 is not None:
            REQUEST_SECONDS.labels(endpoint=ep).observe(time.perf_counter() - t0)
        return resp

    @app.teardown_request
    def _m_err(exc):
        if exc is not None:
            ERRORS.labels(endpoint=(request.endpoint or request.path)).inc()

    @app.route("/metrics")
    def _metrics():
        body, ctype = render_latest()
        return app.response_class(body, mimetype=ctype)
