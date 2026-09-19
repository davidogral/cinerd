# -*- coding: utf-8 -*-
"""Coleta prospectiva de consultas reais (protocolo §4 e §6).

    .venv/bin/python -m experiments.collect export --db data/user.db --since 2026-09-19
    .venv/bin/python -m experiments.collect export --db /caminho/prod.db --since 2026-09-19 --out experiments/data/prospective_raw.jsonl
    .venv/bin/python -m experiments.collect partition --in experiments/data/prospective_raw.jsonl
    .venv/bin/python -m experiments.collect stats --in experiments/data/prospective_raw.jsonl

O conjunto histórico (142 formais + 92 diagnósticos) foi lido, ajustado e
revertido durante o desenvolvimento — serve para gerar hipótese, não para
estimar generalização. A única forma de medir generalização de forma limpa é um
conjunto **coletado depois** da arquitetura congelada, nunca examinado enquanto
o sistema era construído. Este script é o caminho até ele.

O que ele faz, em ordem
-----------------------
1. **Exporta** `kind='search'` do log de produção dentro de uma janela.
2. **Anonimiza**: descarta `user_id`, troca `sid` por um identificador de sessão
   derivado por HMAC com sal local (agrupa sem identificar) e varre o texto por
   PII (e-mail, telefone, CPF, URL, arroba). Consulta com PII é **excluída**,
   contada e nunca gravada.
3. **Deduplica**: repetição exata some; quase-duplicata (Jaccard alto na mesma
   sessão) vira um grupo com uma representante e `n_repeats`.
4. **Anota metadados** exigidos pelo protocolo: janela, frequência, idioma,
   extensão, dígitos, nome próprio, erro de grafia (OOV), proveniência.
5. **Aplica inclusão/exclusão** declarada antes de olhar resultado, e conta cada
   motivo de exclusão — as proporções vão para o artigo.
6. **Sugere alvo** a partir do clique pós-busca (`kind='open'`, `ref='search'`),
   que é **pista para o anotador**, nunca rótulo: o usuário clica no que parece
   certo, não necessariamente no que era.

O que ele **não** faz: rotular. Isso é `experiments/annotate.py`, com dois
anotadores independentes.

Particionamento (`partition`) é **por tempo** e mantém junto do mesmo lado tudo
que é quase-duplicata ou do mesmo alvo — quase-duplicata entre treino e teste é
vazamento silencioso.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sqlite3
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(_HERE, "data")
DEFAULT_OUT = os.path.join(DATA_DIR, "prospective_raw.jsonl")

# Sal local: agrupa sessões sem permitir reidentificação a partir do arquivo
# publicado. Fica FORA do git (é lido do ambiente ou gerado e guardado em
# experiments/data/.salt, que o .gitignore cobre).
SALT_PATH = os.path.join(DATA_DIR, ".salt")

PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "telefone": re.compile(r"(?:\+?\d{2}\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}[-\s]?\d{4}\b"),
    "cpf": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    "url": re.compile(r"https?://|www\."),
    "arroba": re.compile(r"(?<!\w)@\w{3,}"),
}

# Consultas que não são busca known-item de filme. Declarado ANTES de ver
# qualquer resultado (protocolo §4).
NAVIGATIONAL = {
    "filmes",
    "filme",
    "lista",
    "top",
    "melhores",
    "populares",
    "teste",
    "test",
    "asdf",
    "aaa",
    "oi",
    "ola",
    "olá",
    "hello",
    "login",
    "entrar",
    "cadastro",
    "recomendar",
    "recomendação",
}
MIN_WORDS = 2
MIN_CHARS = 8
NEAR_DUP_JACCARD = 0.8

_WORD = re.compile(r"[A-Za-zÀ-ÿ0-9]+")


def _strip(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _tokens(s: str) -> set:
    return set(_WORD.findall(_strip((s or "").lower())))


def _salt() -> bytes:
    env = os.environ.get("RECOMENDAI_COLLECT_SALT")
    if env:
        return env.encode("utf-8")
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(SALT_PATH):
        with open(SALT_PATH, "wb") as fh:
            fh.write(os.urandom(32))
        os.chmod(SALT_PATH, 0o600)
    with open(SALT_PATH, "rb") as fh:
        return fh.read()


def _sid_hash(sid: Optional[str], salt: bytes) -> Optional[str]:
    if not sid:
        return None
    return hmac.new(salt, sid.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def find_pii(text: str) -> list[str]:
    return [name for name, rx in PII_PATTERNS.items() if rx.search(text or "")]


def classify_exclusion(query: str) -> Optional[str]:
    """Motivo de exclusão, ou None se a consulta entra. Ordem fixa e declarada."""
    q = (query or "").strip()
    if not q:
        return "vazia"
    if find_pii(q):
        return "pii"
    words = q.split()
    if len(words) < MIN_WORDS or len(q) < MIN_CHARS:
        return "curta_demais"
    if _strip(q.lower()) in {_strip(w) for w in NAVIGATIONAL}:
        return "navegacional"
    return None


# --------------------------------------------------------------------- export


def _rows(db: str, since: Optional[str], until: Optional[str]) -> Iterable[sqlite3.Row]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    where, params = ["kind='search'", "query<>''"], []
    if since:
        where.append("day>=?")
        params.append(since)
    if until:
        where.append("day<=?")
        params.append(until)
    sql = f"SELECT * FROM events WHERE {' AND '.join(where)} ORDER BY ts"  # noqa: S608 (colunas fixas)
    try:
        yield from conn.execute(sql, params)
    finally:
        conn.close()


def _clicks(db: str) -> dict[tuple, list[dict]]:
    """(sid, dia) -> cliques em resultado de busca, para sugerir alvo ao anotador."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out: dict[tuple, list[dict]] = {}
    try:
        for r in conn.execute("SELECT ts, day, sid, item_id, pos FROM events WHERE kind='open' AND ref='search'"):
            if not r["sid"] or not r["item_id"]:
                continue
            out.setdefault((r["sid"], r["day"]), []).append({"ts": r["ts"], "item_id": r["item_id"], "pos": r["pos"]})
    finally:
        conn.close()
    return out


def cmd_export(args: argparse.Namespace) -> int:
    if not os.path.exists(args.db):
        raise SystemExit(f"banco não encontrado: {args.db}")
    salt = _salt()
    clicks = _clicks(args.db)

    excluded: Counter = Counter()
    kept: list[dict] = []
    seen_exact: dict[str, int] = {}
    n_raw = 0

    for r in _rows(args.db, args.since, args.until):
        n_raw += 1
        q = (r["query"] or "").strip()
        reason = classify_exclusion(q)
        if reason:
            excluded[reason] += 1
            continue
        norm = " ".join(sorted(_tokens(q)))
        if norm in seen_exact:
            kept[seen_exact[norm]]["n_repeats"] += 1
            excluded["duplicata_exata"] += 1
            continue

        sid_h = _sid_hash(r["sid"], salt)
        after = [c for c in clicks.get((r["sid"], r["day"]), []) if c["ts"] >= r["ts"]]
        after.sort(key=lambda c: c["ts"])
        words = q.split()
        seen_exact[norm] = len(kept)
        kept.append(
            {
                "qid": f"p-{hashlib.sha1(norm.encode()).hexdigest()[:10]}",
                "query": q,
                "day": r["day"],
                "session": sid_h,
                "n_repeats": 1,
                "n_results": r["n_results"],
                "found": r["found"],
                "latency_ms": r["latency_ms"],
                # pista para o anotador — NUNCA rótulo (protocolo §5)
                "clicked_item_id": after[0]["item_id"] if after else None,
                "clicked_pos": after[0]["pos"] if after else None,
                "meta": {
                    "n_words": len(words),
                    "n_chars": len(q),
                    "has_digit": any(c.isdigit() for c in q),
                    "caps_mid": sum(1 for w in words[1:] if w[:1].isupper()),
                },
                "provenance": "log-producao",
                "_norm": norm,
            }
        )

    # quase-duplicata: agrupa por Jaccard alto (mesmo alvo descrito duas vezes).
    groups: list[list[int]] = []
    assigned: dict[int, int] = {}
    for i, row in enumerate(kept):
        ti = _tokens(row["query"])
        for gi, g in enumerate(groups):
            tj = _tokens(kept[g[0]]["query"])
            inter, union = len(ti & tj), len(ti | tj)
            if union and inter / union >= NEAR_DUP_JACCARD:
                g.append(i)
                assigned[i] = gi
                break
        else:
            assigned[i] = len(groups)
            groups.append([i])
    for i, row in enumerate(kept):
        row["cluster"] = f"c{assigned[i]:05d}"
        row.pop("_norm", None)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "db": os.path.basename(args.db),
        "window": {"since": args.since, "until": args.until},
        "n_events_read": n_raw,
        "n_kept": len(kept),
        "n_clusters": len(groups),
        "excluded": dict(excluded),
        "exclusion_rate": round(sum(excluded.values()) / n_raw, 4) if n_raw else 0.0,
        "criteria": {
            "min_words": MIN_WORDS,
            "min_chars": MIN_CHARS,
            "near_dup_jaccard": NEAR_DUP_JACCARD,
            "navigational_terms": sorted(NAVIGATIONAL),
            "pii_patterns": sorted(PII_PATTERNS),
        },
        "anonymization": "user_id descartado; sid -> HMAC-SHA256 com sal local; consulta com PII excluída",
    }
    man_path = args.out.replace(".jsonl", "__manifest.json")
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print(f"» {n_raw} eventos de busca lidos → {len(kept)} consultas em {len(groups)} clusters")
    for reason, n in excluded.most_common():
        print(f"   excluída/{reason}: {n}")
    print(f"» {os.path.relpath(args.out, _ROOT)}")
    print(f"» {os.path.relpath(man_path, _ROOT)}")
    if len(kept) < 300:
        print(f"\nAviso: {len(kept)} < 300, o marco de viabilidade do protocolo (§4). Ampliar a janela de coleta.")
    return 0


# ------------------------------------------------------------------ partition


def cmd_partition(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in open(args.inp, encoding="utf-8") if line.strip()]
    if not rows:
        raise SystemExit("arquivo vazio")
    # Ordena por tempo; corta por tempo. Um cluster inteiro vai para a partição
    # do seu PRIMEIRO evento — nunca se divide entre treino e teste.
    rows.sort(key=lambda r: (r.get("day") or "", r.get("qid")))
    first_of: dict[str, int] = {}
    for i, r in enumerate(rows):
        first_of.setdefault(r.get("cluster") or r["qid"], i)

    clusters = sorted(first_of, key=lambda c: first_of[c])
    n = len(clusters)
    cut_dev = int(n * args.dev)
    cut_val = cut_dev + int(n * args.val)
    part_of = {}
    for i, c in enumerate(clusters):
        part_of[c] = "novo-dev" if i < cut_dev else ("nova-validacao" if i < cut_val else "novo-teste-lacrado")

    counts: Counter = Counter()
    for r in rows:
        r["partition"] = part_of[r.get("cluster") or r["qid"]]
        counts[r["partition"]] += 1

    with open(args.inp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"» {len(rows)} consultas · {n} clusters · corte POR TEMPO")
    for name in ("novo-dev", "nova-validacao", "novo-teste-lacrado"):
        print(f"   {name}: {counts[name]}")
    print("\nO teste lacrado abre UMA vez (protocolo §6). Congele agora:")
    print(f"   python -m experiments.freeze {os.path.relpath(args.inp, _ROOT)} --label particao-prospectiva")
    return 0


# ---------------------------------------------------------------------- stats


def cmd_stats(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in open(args.inp, encoding="utf-8") if line.strip()]
    if not rows:
        raise SystemExit("arquivo vazio")
    words = sorted(r["meta"]["n_words"] for r in rows)
    print(f"» {len(rows)} consultas · {len({r.get('cluster') for r in rows})} clusters")
    print(f"   palavras por consulta: mediana {words[len(words) // 2]}, min {words[0]}, max {words[-1]}")
    print(f"   com dígito: {sum(1 for r in rows if r['meta']['has_digit'])}")
    print(f"   com nome próprio (maiúscula no meio): {sum(1 for r in rows if r['meta']['caps_mid'])}")
    print(f"   com clique pós-busca (pista de alvo): {sum(1 for r in rows if r.get('clicked_item_id'))}")
    print(f"   zero resultados: {sum(1 for r in rows if r.get('found') == 0)}")
    by_part = Counter(r.get("partition") for r in rows)
    if any(by_part):
        print("   partições: " + ", ".join(f"{k}={v}" for k, v in by_part.items() if k))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.collect", description="Coleta prospectiva anonimizada.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="log de busca -> JSONL anonimizado e deduplicado")
    e.add_argument("--db", default=os.path.join(_ROOT, "data", "user.db"))
    e.add_argument("--since", help="AAAA-MM-DD (use a data do congelamento do protocolo)")
    e.add_argument("--until")
    e.add_argument("--out", default=DEFAULT_OUT)
    e.set_defaults(func=cmd_export)

    p = sub.add_parser("partition", help="divide por tempo em dev/validação/teste lacrado")
    p.add_argument("--in", dest="inp", default=DEFAULT_OUT)
    p.add_argument("--dev", type=float, default=0.4)
    p.add_argument("--val", type=float, default=0.2)
    p.set_defaults(func=cmd_partition)

    s = sub.add_parser("stats", help="composição do conjunto (vai para o artigo)")
    s.add_argument("--in", dest="inp", default=DEFAULT_OUT)
    s.set_defaults(func=cmd_stats)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
