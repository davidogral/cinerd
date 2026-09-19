# -*- coding: utf-8 -*-
"""Congelamento com hash — o que torna "não mexemos depois" verificável.

    .venv/bin/python -m experiments.freeze docs/PROTOCOLO-TOIS.md --label protocolo-v1
    .venv/bin/python -m experiments.freeze experiments/data/prospective_raw.jsonl --label particao-prospectiva
    .venv/bin/python -m experiments.freeze --verify experiments/frozen/protocolo-v1.json
    .venv/bin/python -m experiments.freeze --list

O protocolo diz, em vários pontos, "congelado", "aberto uma única vez",
"registrar hash, data, commit e desvios". Sem um registro verificável isso é só
intenção. Cada congelamento grava em `experiments/frozen/<label>.json`:
caminho, SHA-256, tamanho, data UTC, *commit* e estado da árvore de trabalho.

`--verify` recalcula o hash e diz se o arquivo mudou desde então. Um arquivo que
mudou não é fraude — é um **desvio**, que a §10 do protocolo manda registrar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
FROZEN_DIR = os.path.join(_HERE, "frozen")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return None


def freeze(path: str, label: str, note: str = "") -> dict:
    abs_path = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    if not os.path.exists(abs_path):
        raise SystemExit(f"não existe: {path}")
    rel = os.path.relpath(abs_path, _ROOT)
    dirty = bool((_git("status", "--porcelain", "--", rel) or "").strip())
    record = {
        "label": label,
        "path": rel,
        "sha256": sha256_file(abs_path),
        "bytes": os.path.getsize(abs_path),
        "frozen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_commit_short": _git("rev-parse", "--short", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "uncommitted_at_freeze": dirty,
        "note": note,
    }
    os.makedirs(FROZEN_DIR, exist_ok=True)
    out = os.path.join(FROZEN_DIR, f"{label}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    return record


def verify(record_path: str) -> tuple[bool, dict]:
    with open(record_path, encoding="utf-8") as fh:
        rec = json.load(fh)
    abs_path = os.path.join(_ROOT, rec["path"])
    if not os.path.exists(abs_path):
        return False, {**rec, "current_sha256": None, "status": "arquivo sumiu"}
    cur = sha256_file(abs_path)
    return cur == rec["sha256"], {**rec, "current_sha256": cur}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.freeze", description="Congela artefatos com hash verificável.")
    ap.add_argument("path", nargs="?", help="arquivo a congelar")
    ap.add_argument("--label", help="nome do registro (default: nome do arquivo)")
    ap.add_argument("--note", default="")
    ap.add_argument("--verify", help="checa um registro de congelamento")
    ap.add_argument("--list", action="store_true", help="lista os congelamentos e o estado de cada um")
    args = ap.parse_args(argv)

    if args.list:
        if not os.path.isdir(FROZEN_DIR):
            print("nenhum congelamento ainda")
            return 0
        for name in sorted(os.listdir(FROZEN_DIR)):
            if not name.endswith(".json"):
                continue
            ok, rec = verify(os.path.join(FROZEN_DIR, name))
            mark = "intacto" if ok else "MUDOU DESDE O CONGELAMENTO (registrar desvio, protocolo §10)"
            print(f"{rec['label']:<28} {rec['frozen_utc']}  {rec['git_commit_short']}  {rec['path']}")
            print(f"{'':<28} {rec['sha256'][:16]}…  {mark}")
        return 0

    if args.verify:
        ok, rec = verify(args.verify)
        print(f"{rec['path']}\n  congelado: {rec['sha256']}\n  agora:     {rec.get('current_sha256')}")
        print("  intacto" if ok else "  MUDOU — registre o desvio em docs/PROTOCOLO-TOIS.md §10")
        return 0 if ok else 1

    if not args.path:
        ap.error("informe um arquivo, --verify ou --list")
    label = args.label or os.path.splitext(os.path.basename(args.path))[0]
    rec = freeze(args.path, label, args.note)
    print(f"» {rec['path']}")
    print(f"  sha256 {rec['sha256']}")
    print(
        f"  commit {rec['git_commit_short']} ({rec['git_branch']})"
        + ("  [árvore suja]" if rec["uncommitted_at_freeze"] else "")
    )
    print(f"» experiments/frozen/{label}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
