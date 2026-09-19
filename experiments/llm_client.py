# -*- coding: utf-8 -*-
"""Cliente de LLM **instrumentado para medição** — o que `core.query_llm` faz em
produção, mas registrando tudo o que o protocolo exige (§7.1 e §7.3).

Diferenças deliberadas em relação a `core.query_llm`:

| | produção (`core.query_llm`) | medição (aqui) |
|---|---|---|
| cache | permanente em disco, por consulta | opcional (`use_cache=False` nas repetições) |
| falha | silenciosa, vira "sem info" | registrada com status, corpo e tentativa |
| resposta | descartada após o parse | **congelada** com SHA-256 no *ledger* |
| tokens/latência/custo | não medidos | medidos por chamada |

Os *prompts* e os nomes de modelo são **importados** de `core.query_llm`: o
sistema sob teste tem que ser o mesmo que está no ar, e um prompt copiado aqui
divergiria silenciosamente do de produção na primeira edição.

Custo: a cota usada hoje é a gratuita da Groq, então o custo monetário medido é
zero e reportá-lo como "de graça" seria enganoso num artigo. O `PriceTable`
permite declarar preço por milhão de *tokens* e reportar o custo que a mesma
carga teria sob preço de mercado — o número entra no artigo com a tabela de
preços declarada junto.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from core import query_llm

_HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER_DIR = os.path.join(_HERE, "results", "ledger")

MAX_RETRIES = int(os.environ.get("RECOMENDAI_LLM_RETRIES", "6"))
BACKOFF_S = float(os.environ.get("RECOMENDAI_LLM_BACKOFF", "2.0"))
MAX_WAIT_S = float(os.environ.get("RECOMENDAI_LLM_MAX_WAIT", "90"))

# Orçamento de tokens de ENTRADA por minuto, por modelo, na cota gratuita. Vem
# da própria mensagem de recusa do provedor ("ITPM: Limit 7000"). Com ~2000
# tokens por chamada de confirmação, cabem ~3 chamadas por minuto: sem controle
# de vazão, metade da rodada é recusada e o resultado sai silenciosamente
# atenuado — foi o que aconteceu nas rodadas de 18/09 antes desta correção.
ITPM_LIMITS: dict[str, int] = {
    "qwen/qwen3.8-27b": int(os.environ.get("RECOMENDAI_ITPM_CONFIRM", "7000")),
    "openai/gpt-oss-20b": int(os.environ.get("RECOMENDAI_ITPM_UNDERSTAND", "8000")),
}
ITPM_DEFAULT = int(os.environ.get("RECOMENDAI_ITPM_DEFAULT", "7000"))
ITPM_MARGIN = 0.85  # não encosta no teto: a contagem do provedor difere da nossa

# A Groq recusa por tokens-por-MINUTO e diz no corpo quanto esperar. Obedecer a
# dica é a diferença entre 39 chamadas perdidas e zero: um backoff cego de 2s
# não cabe num orçamento de 7000 tokens/min a ~2000 tokens por chamada.
_RETRY_HINT = re.compile(r"try again in ([\d.]+)(ms|s|m)\b", re.I)


def _retry_after(response) -> Optional[float]:
    """Segundos a esperar, segundo o cabeçalho ou o corpo da recusa."""
    try:
        header = response.headers.get("retry-after")
        if header:
            return min(float(header), MAX_WAIT_S)
    except Exception:
        pass
    m = _RETRY_HINT.search(getattr(response, "text", "") or "")
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2).lower()
    seconds = value / 1000.0 if unit == "ms" else (value * 60.0 if unit == "m" else value)
    return min(seconds, MAX_WAIT_S)


@dataclass(frozen=True)
class PriceTable:
    """US$ por milhão de tokens. Zero = cota gratuita (default honesto)."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.input_per_mtok + tokens_out * self.output_per_mtok) / 1_000_000.0


@dataclass
class CallRecord:
    """Uma chamada real ao provedor. É a unidade do *ledger*."""

    stage: str  # "understand" | "confirm"
    qid: str
    repeat: int  # 0 = execução principal; 1..R = repetições de variância
    model: str
    status: int  # HTTP; 0 = exceção de rede/timeout
    ok: bool  # respondeu E fez parse
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    from_cache: bool = False
    attempt: int = 1
    error: Optional[str] = None
    response_sha256: Optional[str] = None
    response_raw: Optional[str] = None
    parsed: Any = None
    prompt_sha256: Optional[str] = None
    throttled_s: float = 0.0  # espera proativa para caber no orçamento/minuto
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class QuotaExhausted(RuntimeError):
    """Cota DIARIA do provedor esgotada - esperar dentro da execucao nao resolve.

    Distinta do limite por minuto, que o controle de vazao absorve. Ao encontrar
    esta, o experimento para num limite de consulta e grava o que ja tem, em vez
    de gerar celulas atenuadas (ver `experiments/factorial.py`)."""


class LLMRunner:
    """Executa e registra as duas etapas de LLM. Um `LLMRunner` por rodada."""

    def __init__(self, run_id: str, prices: Optional[dict[str, PriceTable]] = None, ledger: bool = True):
        self.run_id = run_id
        self.quota_exhausted: Optional[str] = None
        self.prices = prices or {}
        self.records: list[CallRecord] = []
        self._ledger_path: Optional[str] = None
        if ledger:
            os.makedirs(LEDGER_DIR, exist_ok=True)
            self._ledger_path = os.path.join(LEDGER_DIR, f"{run_id}.jsonl")
        # cache em memória por (stage, chave): repete a MESMA rodada sem gastar
        # cota, sem tocar o cache permanente de produção (que contaminaria a
        # medição de variância).
        self._memo: dict[tuple, Any] = {}
        # (modelo) -> [(instante, tokens_de_entrada)] da última janela de 60 s
        self._window: dict[str, list[tuple[float, int]]] = {}

    def _throttle(self, model: str, est_tokens: int) -> float:
        """Espera até caber no orçamento de tokens/minuto. Devolve o tempo esperado.

        Pedir permissão em vez de perdão: contar o que já foi gasto na janela e
        dormir o necessário custa segundos; levar 429 custa a chamada inteira."""
        limit = int(ITPM_LIMITS.get(model, ITPM_DEFAULT) * ITPM_MARGIN)
        hist = self._window.setdefault(model, [])
        waited = 0.0
        while True:
            now = time.monotonic()
            hist[:] = [(t, n) for t, n in hist if now - t < 60.0]
            used = sum(n for _, n in hist)
            if used + est_tokens <= limit or not hist:
                return waited
            sleep_for = min(MAX_WAIT_S, 60.0 - (now - hist[0][0]) + 0.5)
            time.sleep(sleep_for)
            waited += sleep_for

    # ------------------------------------------------------------------ infra
    def _price(self, model: str) -> PriceTable:
        return self.prices.get(model, PriceTable())

    def _log(self, rec: CallRecord) -> None:
        self.records.append(rec)
        if self._ledger_path:
            with open(self._ledger_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def _post(self, model: str, system: str, user: str, max_tokens: int, extra: Optional[dict] = None) -> CallRecord:
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        body.update(extra or {})
        est_tokens = int((len(system) + len(user)) / 3.6) + 64
        throttled_s = self._throttle(model, est_tokens)
        rec = CallRecord(
            stage="",
            qid="",
            repeat=0,
            model=model,
            status=0,
            ok=False,
            latency_ms=0.0,
            prompt_sha256=_sha(system + "\n" + user),
            throttled_s=round(throttled_s, 2),
        )
        for attempt in range(1, MAX_RETRIES + 1):
            rec.attempt = attempt
            t0 = time.perf_counter()
            try:
                r = requests.post(
                    query_llm._URL,
                    headers={"Authorization": f"Bearer {query_llm.GROQ_API_KEY}", "Content-Type": "application/json"},
                    json=body,
                    timeout=max(query_llm.GROQ_TIMEOUT, 30.0),
                )
                rec.latency_ms = (time.perf_counter() - t0) * 1000.0
                rec.status = r.status_code
                if r.status_code == 429 and "per day" in (r.text or "").lower():
                    # TPD: nenhuma espera dentro desta execucao resolve.
                    rec.error = "quota_diaria"
                    self.quota_exhausted = f"{model}: {(r.text or '')[:160]}"
                    return rec
                if r.status_code == 429 and attempt < MAX_RETRIES:
                    rec.error = "rate_limit"
                    hint = _retry_after(r)
                    # margem sobre a dica: o orçamento é por minuto corrido e a
                    # janela do provedor não começa quando a nossa espera começa.
                    time.sleep((hint + 0.5) if hint is not None else BACKOFF_S * attempt)
                    continue
                if r.status_code != 200:
                    rec.error = r.text[:300]
                    return rec
                data = r.json()
                usage = data.get("usage") or {}
                rec.tokens_in = int(usage.get("prompt_tokens") or 0)
                rec.tokens_out = int(usage.get("completion_tokens") or 0)
                rec.cost_usd = self._price(model).cost(rec.tokens_in, rec.tokens_out)
                self._window.setdefault(model, []).append((time.monotonic(), rec.tokens_in))
                content = data["choices"][0]["message"]["content"]
                rec.response_raw = content
                rec.response_sha256 = _sha(content)
                rec.ok = True
                return rec
            except Exception as exc:  # rede, timeout, JSON de transporte
                rec.latency_ms = (time.perf_counter() - t0) * 1000.0
                rec.error = f"{type(exc).__name__}: {exc}"[:300]
                if attempt < MAX_RETRIES:
                    time.sleep(BACKOFF_S * attempt)
                    continue
                return rec
        return rec

    # ------------------------------------------------------------- etapa A
    def understand(self, qid: str, query: str, repeat: int = 0, use_cache: bool = True) -> tuple[dict, CallRecord]:
        """Etapa A (entendimento). Devolve (plano, registro). Falha vira plano vazio
        — igual à produção, que nunca deixa o LLM derrubar a busca."""
        key = ("understand", query.strip().lower())
        if use_cache and key in self._memo:
            plan, base = self._memo[key]
            rec = CallRecord(**{**asdict(base), "qid": qid, "repeat": repeat, "from_cache": True, "cost_usd": 0.0})
            self._log(rec)
            return plan, rec

        rec = self._post(query_llm.GROQ_MODEL, query_llm._SYSTEM, query.strip(), 800, {"reasoning_effort": "low"})
        rec.stage, rec.qid, rec.repeat = "understand", qid, repeat
        plan: dict = {}
        if rec.ok:
            try:
                plan = json.loads(rec.response_raw or "")
                rec.parsed = plan
            except Exception as exc:
                rec.ok = False
                rec.error = f"parse: {exc}"[:200]
        self._log(rec)
        if rec.ok and use_cache:
            self._memo[key] = (plan, rec)
        return plan, rec

    # ------------------------------------------------------------- etapa B
    def confirm(
        self, qid: str, query: str, candidates: list[dict], repeat: int = 0, use_cache: bool = True
    ) -> tuple[list[int], CallRecord]:
        """Etapa B (confirmação). Devolve (tmdb_ids confirmados em ordem de
        confiança, registro). Mesmo prompt, mesmo modelo e mesmo filtro de
        confiança ("alta"/"media") da produção."""
        ids_part = ",".join(str(c.get("tmdb_id")) for c in candidates)
        key = ("confirm", query.strip().lower(), ids_part)
        if use_cache and key in self._memo:
            picks, base = self._memo[key]
            rec = CallRecord(**{**asdict(base), "qid": qid, "repeat": repeat, "from_cache": True, "cost_usd": 0.0})
            self._log(rec)
            return list(picks), rec

        lines = [
            f"{i}. {c.get('title') or '?'} ({c.get('year') or '?'}): {(c.get('overview') or '').strip()}"
            for i, c in enumerate(candidates, 1)
        ]
        user = "Descrição: " + query + "\n\nCandidatos:\n" + "\n".join(lines)
        rec = self._post(query_llm.GROQ_RERANK_MODEL, query_llm._RERANK_SYSTEM, user, 500)
        rec.stage, rec.qid, rec.repeat = "confirm", qid, repeat

        picks: list[int] = []
        if rec.ok:
            try:
                data = json.loads(rec.response_raw or "")
                rec.parsed = data
                seen: set = set()
                for item in data.get("confirmados") or []:
                    if not isinstance(item, dict) or item.get("confianca") not in ("alta", "media"):
                        continue
                    try:
                        idx = int(item.get("n")) - 1
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < len(candidates):
                        tid = candidates[idx].get("tmdb_id")
                        if tid not in seen:
                            seen.add(tid)
                            picks.append(int(tid))
            except Exception as exc:
                rec.ok = False
                rec.error = f"parse: {exc}"[:200]
        self._log(rec)
        if rec.ok and use_cache:
            self._memo[key] = (list(picks), rec)
        return picks, rec

    # ------------------------------------------------------------- resumo
    def summary(self) -> dict:
        real = [r for r in self.records if not r.from_cache]
        # A chamada que DETECTA a cota diária não conta como falha de medição: a
        # consulta dela é descartada pelo chamador, então ela nunca entra num
        # resultado. Contá-la invalidaria justamente a execução que parou certo.
        graded = [r for r in real if r.error != "quota_diaria"]
        by_stage: dict[str, dict] = {}
        for stage in ("understand", "confirm"):
            rs = [r for r in graded if r.stage == stage]
            if not rs:
                continue
            lat = sorted(r.latency_ms for r in rs)
            by_stage[stage] = {
                "calls": len(rs),
                "ok": sum(1 for r in rs if r.ok),
                "failed": sum(1 for r in rs if not r.ok),
                "rate_limited": sum(1 for r in rs if r.error == "rate_limit"),
                "failure_rate": round(sum(1 for r in rs if not r.ok) / len(rs), 4),
                "throttled_s": round(sum(r.throttled_s for r in rs), 1),
                "tokens_in": sum(r.tokens_in for r in rs),
                "tokens_out": sum(r.tokens_out for r in rs),
                "cost_usd": round(sum(r.cost_usd for r in rs), 6),
                "latency_p50_ms": round(lat[len(lat) // 2], 1),
                "latency_p95_ms": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1),
                "model": rs[0].model,
            }
        return {
            "run_id": self.run_id,
            "quota_exhausted": self.quota_exhausted,
            "ledger": os.path.relpath(self._ledger_path, os.path.dirname(_HERE)) if self._ledger_path else None,
            "calls_total": len(self.records),
            "calls_real": len(real),
            "calls_graded": len(graded),
            "calls_quota_stop": len(real) - len(graded),
            "calls_cached": len(self.records) - len(real),
            "by_stage": by_stage,
            "cost_usd_total": round(sum(r.cost_usd for r in real), 6),
            "failure_rate": round(sum(1 for r in graded if not r.ok) / len(graded), 4) if graded else 0.0,
        }


def load_prices(path: Optional[str]) -> dict[str, PriceTable]:
    """JSON `{"modelo": {"input_per_mtok": x, "output_per_mtok": y}}`."""
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: PriceTable(**v) for k, v in raw.items()}
