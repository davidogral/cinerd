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


def _unfence(text: str) -> str:
    """Tira cerca de markdown e prosa ao redor do JSON.

    A Groq com `response_format` implícito devolve JSON limpo; um modelo local
    instruído devolve, com frequência, ```json … ``` ou uma frase antes. Cortar
    isso aqui é a diferença entre "o modelo local não funciona" e "o parser era
    estrito demais" — e as duas conclusões levariam a decisões opostas."""
    t = (text or "").strip()
    if "```" in t:
        parts = t.split("```")
        for part in parts[1:]:
            body = part[4:].lstrip() if part.lower().startswith("json") else part
            if body.strip().startswith("{"):
                t = body
                break
    i, j = t.find("{"), t.rfind("}")
    return t[i : j + 1] if 0 <= i < j else t


class QuotaExhausted(RuntimeError):
    """Cota DIARIA do provedor esgotada - esperar dentro da execucao nao resolve.

    Distinta do limite por minuto, que o controle de vazao absorve. Ao encontrar
    esta, o experimento para num limite de consulta e grava o que ja tem, em vez
    de gerar celulas atenuadas (ver `experiments/factorial.py`)."""


class LLMRunner:
    """Executa e registra as duas etapas de LLM. Um `LLMRunner` por rodada."""

    def __init__(
        self,
        run_id: str,
        prices: Optional[dict[str, PriceTable]] = None,
        ledger: bool = True,
        provider: str = "groq",
        provider_understand: Optional[str] = None,
        provider_confirm: Optional[str] = None,
    ):
        self.run_id = run_id
        # Provedor POR ETAPA. Sem isso, trocar o provedor troca as duas etapas de
        # uma vez, e a comparação deixa de isolar o verificador — foi o que
        # aconteceu na execução de 2026-09-19 e só apareceu porque C10 divergiu
        # numa consulta (o classificador de tipo mudou junto).
        self.providers = {
            "understand": provider_understand or provider,
            "confirm": provider_confirm or provider,
        }
        self.provider = provider
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

    def _post(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        extra: Optional[dict] = None,
        stage: str = "",
    ) -> CallRecord:
        if self.providers.get(stage, self.provider) == "local":
            return self._post_local(system, user, max_tokens)
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

    def _post_local(self, system: str, user: str, max_tokens: int) -> CallRecord:
        """Mesma etapa, modelo local. Mesmo prompt, mesmo formato de registro.

        O `enable_thinking=False` é o ponto que importa: Qwen3 gera um bloco de
        raciocínio por padrão, e o achado de 2026-09-09 mostra que raciocínio
        oculto faz o modelo responder pela memória em vez de examinar a lista —
        exatamente o que a etapa de confirmação NÃO pode fazer."""
        from mlx_lm import generate  # noqa: PLC0415

        model, tokenizer = _local_model()
        rec = CallRecord(
            stage="",
            qid="",
            repeat=0,
            model=f"{LOCAL_MODEL}@{_local_cache.get('revision') or '?'}",
            status=0,
            ok=False,
            latency_ms=0.0,
            prompt_sha256=_sha(system + "\n" + user),
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
        except TypeError:  # tokenizer sem o parâmetro (modelo sem modo thinking)
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        rec.tokens_in = len(prompt) if isinstance(prompt, list) else len(tokenizer.encode(prompt))
        t0 = time.perf_counter()
        try:
            text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        except Exception as exc:
            rec.latency_ms = (time.perf_counter() - t0) * 1000.0
            rec.error = f"{type(exc).__name__}: {exc}"[:300]
            return rec
        rec.latency_ms = (time.perf_counter() - t0) * 1000.0
        rec.status = 200
        rec.tokens_out = len(tokenizer.encode(text))
        # Qwen3 pode emitir <think>…</think> mesmo desligado; o JSON vem depois.
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        rec.response_raw = text.strip()
        rec.response_sha256 = _sha(rec.response_raw)
        rec.cost_usd = self._price(rec.model).cost(rec.tokens_in, rec.tokens_out)
        rec.ok = True
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

        rec = self._post(
            query_llm.GROQ_MODEL,
            query_llm._SYSTEM,
            query.strip(),
            800,
            {"reasoning_effort": "low"},
            stage="understand",
        )
        rec.stage, rec.qid, rec.repeat = "understand", qid, repeat
        plan: dict = {}
        if rec.ok:
            try:
                plan = json.loads(_unfence(rec.response_raw or ""))
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
        rec = self._post(query_llm.GROQ_RERANK_MODEL, query_llm._RERANK_SYSTEM, user, 500, stage="confirm")
        rec.stage, rec.qid, rec.repeat = "confirm", qid, repeat

        picks: list[int] = []
        if rec.ok:
            try:
                data = json.loads(_unfence(rec.response_raw or ""))
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


# ---------------------------------------------------------------------------
# Provedor local (MLX)
#
# Por que Qwen3-8B em 4 bits, e não outro:
#
# 1. **Mesma família e mesma geração que produção** (`qwen/qwen3.8-27b`). O
#    contraste fica sendo escala + hospedagem. Trocar de família (Llama,
#    Mistral) ou de geração (Qwen2.5) somaria um confundimento e tornaria
#    qualquer diferença observada inatribuível.
# 2. **Sem raciocínio oculto.** É restrição dura deste projeto, não preferência:
#    um modelo de raciocínio em esforço baixo responde pela memória paramétrica
#    em vez de examinar os candidatos, e em esforço médio gasta 9–10 mil tokens
#    por chamada (ver `core/query_llm.rerank_confirm`, achado de 2026-09-09).
#    Qwen3 tem modo "thinking" alternável, então ele é **explicitamente
#    desligado** no template. Pela mesma razão, um destilado de R1 está fora.
# 3. **Multilíngue com português** — consultas e sinopses são pt-BR.
# 4. **Apache-2.0**: o protocolo §7.3 exige licença de uso verificável para as
#    respostas entrarem no material replicável.
# 5. **Reprodutível fora do Mac**: os pesos upstream rodam em llama.cpp/vLLM/
#    transformers. MLX é só o runtime local. O ledger grava o *commit hash* do
#    repositório, não só o nome, para a verificação ser exata.
# 6. **Cabe com folga**: ~4,5 GB ao lado do e5-large (~2 GB) e do índice.
#
# Ressalva declarada: 4 bits é quantização, então a comparação com produção
# mistura quantização com escala. Isso é reportado, não escondido.
LOCAL_MODEL = os.environ.get("RECOMENDAI_LOCAL_MODEL", "mlx-community/Qwen3-8B-4bit")
LOCAL_MAX_TOKENS = int(os.environ.get("RECOMENDAI_LOCAL_MAX_TOKENS", "512"))

_local_cache: dict = {}


def _local_model():
    """Carrega o modelo uma vez por processo. Import tardio de propósito: o CI
    instala `requirements-ci.txt`, que não tem mlx, e importar aqui no topo
    quebraria a suíte inteira numa máquina sem Apple Silicon."""
    if "model" not in _local_cache:
        from mlx_lm import load  # noqa: PLC0415

        model, tokenizer = load(LOCAL_MODEL)
        _local_cache["model"] = model
        _local_cache["tokenizer"] = tokenizer
        _local_cache["revision"] = _local_revision()
    return _local_cache["model"], _local_cache["tokenizer"]


def _local_revision() -> Optional[str]:
    """Commit hash do snapshot no cache do HuggingFace — a identidade exata dos
    pesos, que o nome do repositório sozinho não dá."""
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(LOCAL_MODEL, local_files_only=True)
        return os.path.basename(path.rstrip("/"))
    except Exception:
        return None


def local_runtime() -> dict:
    """Identificação completa do runtime local, para o artigo e para replicação.

    Nome de modelo sozinho não reproduz nada: a mesma família em outra
    quantização, outro framework ou outro template de chat é outro sistema."""
    info: dict = {"model": LOCAL_MODEL, "revision": _local_cache.get("revision") or _local_revision()}
    try:
        import glob as _glob
        import json as _json

        cfg = _glob.glob(
            os.path.expanduser(
                f"~/.cache/huggingface/hub/models--{LOCAL_MODEL.replace('/', '--')}/snapshots/*/config.json"
            )
        )
        if cfg:
            with open(cfg[0], encoding="utf-8") as fh:
                d = _json.load(fh)
            info["architecture"] = d.get("model_type")
            info["quantization"] = d.get("quantization")
            info["max_position_embeddings"] = d.get("max_position_embeddings")
    except Exception:
        pass
    try:
        import platform
        import subprocess as _sp

        import mlx.core as _mx
        import mlx_lm as _mlxlm

        info["framework"] = f"mlx-lm {getattr(_mlxlm, '__version__', '?')} / mlx {getattr(_mx, '__version__', '?')}"
        info["hardware"] = _sp.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        mem = _sp.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout.strip()
        info["unified_memory_gib"] = round(int(mem) / 2**30) if mem.isdigit() else None
        info["os"] = f"macOS {platform.mac_ver()[0]}"
    except Exception:
        pass
    info["chat_template"] = "tokenizer.apply_chat_template(add_generation_prompt=True, enable_thinking=False)"
    info["generation"] = {"temperature": 0, "max_tokens_confirm": 500, "max_tokens_understand": 800}
    return info


def load_prices(path: Optional[str]) -> dict[str, PriceTable]:
    """JSON `{"modelo": {"input_per_mtok": x, "output_per_mtok": y}}`."""
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: PriceTable(**v) for k, v in raw.items()}
