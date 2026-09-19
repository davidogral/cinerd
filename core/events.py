# -*- coding: utf-8 -*-
"""Registros de uso — buscas, cliques em resultados, filtros, latência
(LGPD: dado pessoal quando `user_id` não é nulo).

Tabela `events` no mesmo `data/user.db`. `log()` é **fire-and-forget**: nunca
levanta e nunca deixa uma falha de registro quebrar a requisição. Retenção
padrão de 90 dias (`RECOMENDAI_EVENT_RETENTION_DAYS`), aplicada de forma
preguiçosa.

Cada evento carrega um `sid` — identificador de sessão (cookie próprio, sem dado
pessoal) — que permite encadear ações de uma visita (funil busca → clique,
reformulação de consulta, sessões novas vs recorrentes) sem identificar a pessoa.

Base legal: legítimo interesse (entender o uso, achar buracos de catálogo,
medir a qualidade da busca em produção) e, quando associado à conta logada, o
consentimento do cadastro. Entra no `GET /auth/export` e é apagado junto com a
conta (`core/users.delete_user`).
"""

from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from core.user_data import _connect

# Nuvem de palavras: tokeniza a consulta e joga fora conectivos PT + muletas.
_WORD_RE = re.compile(r"[a-zà-ú0-9]+", re.I)
_STOP = set(
    "que com sem por para pra dos das num numa uma uns umas não nao mas seu sua seus suas ele ela"
    " eles elas isso esse essa este esta aquele aquela onde quando como qual quais quem filme filmes"
    " cena personagem sobre entre muito mais menos tem ter vai foi era são sao dele dela aos nas nos"
    " ao de do da em no na os as um se ou eu".split()
)

RETENTION_DAYS = int(os.environ.get("RECOMENDAI_EVENT_RETENTION_DAYS", "90"))
_MAX_QUERY = 300

# Só a tabela + índices sobre colunas "base". Os índices de colunas que podem
# não existir num banco antigo (ex.: sid) são criados DEPOIS do ALTER, em
# `_ensure` — senão o `executescript` explode num banco pré-existente e os
# `ALTER TABLE` nem chegam a rodar.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    day        TEXT NOT NULL,
    kind       TEXT NOT NULL,
    query      TEXT NOT NULL DEFAULT '',
    filters    TEXT NOT NULL DEFAULT '',
    n_results  INTEGER,
    found      INTEGER,
    latency_ms REAL,
    stage_ms   TEXT NOT NULL DEFAULT '',
    user_id    INTEGER,
    sid        TEXT,
    ref        TEXT,
    pos        INTEGER,
    item_id    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_day ON events(day);
CREATE INDEX IF NOT EXISTS idx_events_kind_day ON events(kind, day);
CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id);
"""

_UPGRADE_COLS = (("sid", "TEXT"), ("ref", "TEXT"), ("pos", "INTEGER"), ("item_id", "INTEGER"))
_LATE_INDEXES = ("CREATE INDEX IF NOT EXISTS idx_events_sid ON events(sid)",)
_ensured = False


def _ensure(conn) -> None:
    global _ensured
    if _ensured:
        return
    conn.executescript(_SCHEMA)
    have = {r[1] for r in conn.execute("PRAGMA table_info(events)")}
    for col, decl in _UPGRADE_COLS:
        if col not in have:
            conn.execute(f"ALTER TABLE events ADD COLUMN {col} {decl}")
    for stmt in _LATE_INDEXES:  # agora as colunas existem
        conn.execute(stmt)
    _ensured = True


def _now():
    return datetime.now(timezone.utc)


def log(
    kind: str,
    *,
    query: str = "",
    filters: Optional[dict] = None,
    n_results: Optional[int] = None,
    latency_ms: Optional[float] = None,
    stage_ms: Optional[dict] = None,
    user_id: Optional[int] = None,
    sid: Optional[str] = None,
    ref: Optional[str] = None,
    pos: Optional[int] = None,
    item_id: Optional[int] = None,
) -> None:
    """Registra um evento. Silencioso em qualquer falha."""
    try:
        now = _now()
        with _connect() as conn:
            _ensure(conn)
            conn.execute(
                "INSERT INTO events (ts, day, kind, query, filters, n_results, found, latency_ms, stage_ms,"
                " user_id, sid, ref, pos, item_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    now.strftime("%Y-%m-%d"),
                    str(kind)[:40],
                    (query or "").strip()[:_MAX_QUERY],
                    json.dumps(filters, ensure_ascii=False, sort_keys=True) if filters else "",
                    n_results,
                    None if n_results is None else (1 if n_results > 0 else 0),
                    round(latency_ms, 2) if latency_ms is not None else None,
                    json.dumps(stage_ms, sort_keys=True) if stage_ms else "",
                    int(user_id) if user_id else None,
                    (sid or None),
                    (str(ref)[:20] if ref else None),
                    (int(pos) if pos not in (None, "") else None),
                    (int(item_id) if item_id not in (None, "") else None),
                ),
            )
            if random.random() < 0.03:
                cutoff = (now - timedelta(days=RETENTION_DAYS)).strftime("%Y-%m-%d")
                conn.execute("DELETE FROM events WHERE day < ?", (cutoff,))
    except Exception:
        pass


def purge_expired() -> int:
    """Apaga eventos além da retenção; devolve quantos. Uso manual/manutenção."""
    cutoff = (_now() - timedelta(days=RETENTION_DAYS)).strftime("%Y-%m-%d")
    try:
        with _connect() as conn:
            _ensure(conn)
            return conn.execute("DELETE FROM events WHERE day < ?", (cutoff,)).rowcount
    except Exception:
        return 0


def export_for_user(user_id: int) -> list[dict]:
    """Eventos de uma conta — para o `GET /auth/export`."""
    try:
        with _connect() as conn:
            _ensure(conn)
            rows = conn.execute(
                "SELECT ts, kind, query, filters, n_results, latency_ms, ref, pos, item_id FROM events"
                " WHERE user_id = ? ORDER BY ts DESC",
                (int(user_id),),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def delete_for_user(conn, user_id: int) -> None:
    """Apaga os eventos de uma conta — chamado de dentro de `users.delete_user`."""
    conn.execute("DELETE FROM events WHERE user_id = ?", (int(user_id),))


# --------------------------------------------------------------------- analytics
_REC_KINDS = ("recommend", "recommend_history", "recommend_letterboxd", "similar", "essentials")


def _pct(xs: list) -> dict:
    xs = sorted(xs)
    n = len(xs)

    def p(q: float) -> float:
        return round(xs[min(n - 1, max(0, int(round(q / 100.0 * n)) - 1))], 1)

    return {"p50": p(50), "p95": p(95), "p99": p(99), "n": n}


def _pos_bucket(p: int) -> str:
    if p <= 3:
        return str(p)
    if p <= 5:
        return "4–5"
    if p <= 10:
        return "6–10"
    return "11+"


def _delta(cur: float, prev: float):
    if not prev:
        return None
    return round((cur - prev) / prev * 100, 1)


def _window_counts(conn, a: str, b: str) -> dict:
    """Contagens-chave para uma janela [a, b) — usado para o valor atual e o
    período anterior (delta)."""
    row = conn.execute(
        "SELECT"
        "  SUM(kind='search'),"
        "  SUM(kind='search' AND found=0),"
        "  SUM(kind='open' AND ref='search'),"
        "  SUM(kind IN ('open')),"
        "  COUNT(DISTINCT sid),"
        "  COUNT(DISTINCT user_id),"
        f"  SUM(kind IN ({','.join('?' * len(_REC_KINDS))}))"
        " FROM events WHERE day >= ? AND day < ?",
        (*_REC_KINDS, a, b),
    ).fetchone()
    searches, zero, open_from_search, opens, sess, usr, recs = (x or 0 for x in row)
    return {
        "searches": searches,
        "zero": zero,
        "opens": opens,
        "open_from_search": open_from_search,
        "sessions": sess,
        "users": usr,
        "recs": recs,
        "ctr": round(open_from_search / searches, 3) if searches else 0.0,
        "zero_rate": round(zero / searches, 3) if searches else 0.0,
    }


def analytics(days: int = 30) -> dict:
    """Resumo analítico dos últimos `days` dias para o painel admin."""
    days = max(1, min(int(days), 365))
    now = _now()
    d0 = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    d_prev = (now - timedelta(days=2 * days)).strftime("%Y-%m-%d")
    d_end = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    out: dict = {"days": days, "since": d0}
    try:
        with _connect() as conn:
            _ensure(conn)

            cur = _window_counts(conn, d0, d_end)
            prev = _window_counts(conn, d_prev, d0)
            out["kpis"] = {
                "searches": {"v": cur["searches"], "delta": _delta(cur["searches"], prev["searches"])},
                "ctr": {"v": cur["ctr"], "delta": _delta(cur["ctr"], prev["ctr"])},
                "zero_rate": {"v": cur["zero_rate"], "delta": _delta(cur["zero_rate"], prev["zero_rate"])},
                "sessions": {"v": cur["sessions"], "delta": _delta(cur["sessions"], prev["sessions"])},
                "users": {"v": cur["users"], "delta": _delta(cur["users"], prev["users"])},
                "recs": {"v": cur["recs"], "delta": _delta(cur["recs"], prev["recs"])},
            }
            out["by_kind"] = dict(conn.execute("SELECT kind, COUNT(*) FROM events WHERE day >= ? GROUP BY kind", (d0,)))

            # timeseries: busca / clique / rec / sem-resultado por dia
            ts: dict = {}
            for d, k, n, zero in conn.execute(
                "SELECT day, kind, COUNT(*), SUM(found=0) FROM events WHERE day >= ? GROUP BY day, kind", (d0,)
            ):
                row = ts.setdefault(d, {"day": d, "search": 0, "open": 0, "rec": 0, "zero": 0})
                if k == "search":
                    row["search"] += n
                    row["zero"] += zero or 0
                elif k == "open":
                    row["open"] += n
                elif k in _REC_KINDS:
                    row["rec"] += n
            out["by_day"] = [ts[d] for d in sorted(ts)]

            out["top_queries"] = [
                {"query": q, "count": c}
                for q, c in conn.execute(
                    "SELECT query, COUNT(*) c FROM events WHERE kind='search' AND query<>'' AND day>=?"
                    " GROUP BY lower(query) ORDER BY c DESC LIMIT 25",
                    (d0,),
                )
            ]
            out["zero_result"] = [
                {"query": q, "count": c}
                for q, c in conn.execute(
                    "SELECT query, COUNT(*) c FROM events WHERE kind='search' AND found=0 AND query<>'' AND day>=?"
                    " GROUP BY lower(query) ORDER BY c DESC LIMIT 25",
                    (d0,),
                )
            ]

            # histograma de posição do clique (cliques vindos da busca)
            hist: dict = {}
            for p, c in conn.execute(
                "SELECT pos, COUNT(*) FROM events WHERE kind='open' AND ref='search' AND pos IS NOT NULL AND day>=?"
                " GROUP BY pos",
                (d0,),
            ):
                hist[_pos_bucket(int(p))] = hist.get(_pos_bucket(int(p)), 0) + c
            out["click_positions"] = [
                {"bucket": b, "count": hist.get(b, 0)} for b in ("1", "2", "3", "4–5", "6–10", "11+")
            ]

            # funil de sessão
            fn = conn.execute(
                "SELECT"
                "  COUNT(DISTINCT CASE WHEN kind='search' THEN sid END),"
                "  COUNT(DISTINCT CASE WHEN kind='search' AND found=1 THEN sid END),"
                "  COUNT(DISTINCT CASE WHEN kind='open' AND ref='search' THEN sid END)"
                " FROM events WHERE day>=? AND sid IS NOT NULL",
                (d0,),
            ).fetchone()
            out["funnel"] = [
                {"step": "buscou", "sessions": fn[0] or 0},
                {"step": "com resultado", "sessions": fn[1] or 0},
                {"step": "abriu um filme", "sessions": fn[2] or 0},
            ]

            # buscas por sessão + sessões novas vs recorrentes
            sps = conn.execute(
                "SELECT AVG(c), MAX(c) FROM (SELECT sid, COUNT(*) c FROM events"
                " WHERE kind='search' AND sid IS NOT NULL AND day>=? GROUP BY sid)",
                (d0,),
            ).fetchone()
            out["searches_per_session"] = {"avg": round(sps[0] or 0, 1), "max": sps[1] or 0}

            nr: dict = {}
            for d, isnew, n in conn.execute(
                "SELECT e.day,"
                "  (SELECT MIN(day) FROM events e2 WHERE e2.sid=e.sid) >= ? AS is_new,"
                "  COUNT(DISTINCT e.sid)"
                " FROM events e WHERE e.day>=? AND e.sid IS NOT NULL GROUP BY e.day, is_new",
                (d0, d0),
            ):
                row = nr.setdefault(d, {"day": d, "new": 0, "returning": 0})
                row["new" if isnew else "returning"] += n
            out["sessions_by_day"] = [nr[d] for d in sorted(nr)]

            # reformulações de consulta (pares consecutivos dentro da sessão)
            reform: dict = {}
            try:
                for prev_q, q in conn.execute(
                    "SELECT prev, query FROM ("
                    "  SELECT sid, lower(query) query, LAG(lower(query)) OVER (PARTITION BY sid ORDER BY event_id) prev"
                    "  FROM events WHERE kind='search' AND query<>'' AND sid IS NOT NULL AND day>=?"
                    ") WHERE prev IS NOT NULL AND prev<>query",
                    (d0,),
                ):
                    reform[(prev_q, q)] = reform.get((prev_q, q), 0) + 1
            except Exception:
                pass
            out["reformulations"] = [
                {"from": a, "to": b, "count": c} for (a, b), c in sorted(reform.items(), key=lambda kv: -kv[1])[:15]
            ]

            # filmes: mais abertos (qualquer origem), mais clicados a partir da
            # busca, e mais frequentes em #1 do resultado. Títulos anexados na rota.
            def _top_items(where: str, args: tuple, lim: int = 20) -> list:
                return [
                    {"tmdb_id": i, "count": c}
                    for i, c in conn.execute(
                        f"SELECT item_id, COUNT(*) c FROM events WHERE item_id IS NOT NULL AND day>=? AND {where}"
                        f" GROUP BY item_id ORDER BY c DESC LIMIT {lim}",
                        (d0, *args),
                    )
                ]

            out["top_items"] = _top_items("kind='open'", ())
            out["top_clicked"] = _top_items("kind='open' AND ref='search'", ())
            out["top_ranked"] = _top_items("kind='search'", ())  # o filme que ficou em #1

            # nuvem de palavras — termos mais buscados (fora das stopwords)
            wf: dict = {}
            for (q,) in conn.execute("SELECT query FROM events WHERE kind='search' AND query<>'' AND day>=?", (d0,)):
                for w in _WORD_RE.findall(q.lower()):
                    if len(w) >= 3 and w not in _STOP:
                        wf[w] = wf.get(w, 0) + 1
            out["word_freq"] = [{"word": w, "count": c} for w, c in sorted(wf.items(), key=lambda kv: -kv[1])[:45]]

            # heatmap hora × dia-da-semana (0=segunda)
            heat = [[0] * 24 for _ in range(7)]
            for (t,) in conn.execute("SELECT ts FROM events WHERE day>=?", (d0,)):
                try:
                    dt = datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")
                    heat[dt.weekday()][dt.hour] += 1
                except Exception:
                    pass
            out["heatmap"] = heat

            # uso de filtros
            gen: dict = {}
            yr = lang = with_filter = 0
            for (fj,) in conn.execute(
                "SELECT filters FROM events WHERE kind='search' AND filters<>'' AND day>=?", (d0,)
            ):
                try:
                    f = json.loads(fj)
                except Exception:
                    continue
                with_filter += 1
                if f.get("genre"):
                    gen[f["genre"]] = gen.get(f["genre"], 0) + 1
                if f.get("year_min") or f.get("year_max"):
                    yr += 1
                if f.get("language"):
                    lang += 1
            out["filters"] = {
                "of_searches": cur["searches"],
                "with_any_filter": with_filter,
                "year_range": yr,
                "language": lang,
                "genres": sorted(({"genre": g, "count": c} for g, c in gen.items()), key=lambda x: -x["count"])[:12],
            }

            # comprimento da consulta
            qlen = [
                r[0]
                for r in conn.execute(
                    "SELECT length(query) FROM events WHERE kind='search' AND query<>'' AND day>=?", (d0,)
                )
            ]
            if qlen:
                qlen.sort()
                out["query_len"] = {
                    "avg": round(sum(qlen) / len(qlen), 1),
                    "p50": qlen[len(qlen) // 2],
                    "short_share": round(sum(1 for x in qlen if x <= 25) / len(qlen), 2),
                }

            # latência por etapa (amostra recente)
            samples: dict = {}
            for (sj,) in conn.execute(
                "SELECT stage_ms FROM events WHERE stage_ms<>'' AND day>=? ORDER BY event_id DESC LIMIT 5000", (d0,)
            ):
                try:
                    s = json.loads(sj)
                except Exception:
                    continue
                for k, v in s.items():
                    try:
                        samples.setdefault(k, []).append(float(v))
                    except (TypeError, ValueError):
                        pass
            out["latency"] = {k: _pct(v) for k, v in samples.items() if v}
    except Exception as e:  # analytics nunca derruba o painel
        out["error"] = str(e)
    return out


def raw_searches(days: int = 30, limit: int = 500) -> list[dict]:
    """Log cru das últimas buscas (mais recente primeiro) — o `top_queries` de
    `analytics()` agrupa por texto e corta em 25, então uma consulta nova some
    atrás das mais frequentes; esta lista mostra CADA busca, sem agrupar."""
    days = max(1, min(int(days), 365))
    limit = max(1, min(int(limit), 2000))
    d0 = (_now() - timedelta(days=days)).strftime("%Y-%m-%d")
    out = []
    try:
        with _connect() as conn:
            _ensure(conn)
            rows = conn.execute(
                "SELECT ts, query, filters, n_results, found, latency_ms, user_id, item_id"
                " FROM events WHERE kind='search' AND day>=? ORDER BY event_id DESC LIMIT ?",
                (d0, limit),
            )
            for ts, query, filters, n_results, found, latency_ms, user_id, item_id in rows:
                try:
                    f = json.loads(filters) if filters else {}
                except Exception:
                    f = {}
                out.append(
                    {
                        "ts": ts,
                        "query": query,
                        "filters": f,
                        "n_results": n_results,
                        "found": bool(found),
                        "latency_ms": latency_ms,
                        "user_id": user_id,
                        "item_id": item_id,
                    }
                )
    except Exception:
        pass
    return out
