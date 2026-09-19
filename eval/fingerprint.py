# -*- coding: utf-8 -*-
"""Impressão digital do índice de busca, para verificação por terceiros.

    .venv/bin/python -m eval.fingerprint                 # imprime e atualiza o cache
    .venv/bin/python -m eval.fingerprint --dir retrieval/index_e5small

O índice (≈400 MB de *embeddings* e matrizes BM25) é gerenciado por DVC num
*bucket* que exige credencial privada, e o catálogo bruto tem restrição de
licença: um avaliador externo **não** consegue baixá-lo pronto a partir do
repositório público. O que ele pode fazer é reconstruí-lo com
`retrieval/index_builder.py` e **verificar se chegou ao mesmo lugar** — desde que
exista algo com que comparar.

É esse "algo" que este módulo produz: nome, tamanho e SHA-256 de cada artefato do
índice, gravados junto de cada resultado de avaliação (`eval/run.py`). Não
resolve a redistribuição; resolve a verificação, que é a parte que estava
faltando (`docs/PROTOCOLO-TOIS.md` §11.1, item 3).

Hash completo de 400 MB leva alguns segundos, então o resultado é cacheado num
arquivo ao lado do índice, invalidado por tamanho e data de modificação. O cache
é uma otimização: apagá-lo só custa tempo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DEFAULT_DIR = os.path.join(_ROOT, "retrieval", "index")
CACHE_NAME = ".fingerprint.json"

# Extensões que compõem o índice. `.dvc`, `.gitkeep` e o próprio cache ficam fora.
INDEX_SUFFIXES = (".npy", ".npz", ".pkl", ".json")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(index_dir: str = DEFAULT_DIR, use_cache: bool = True) -> dict:
    """{arquivo: {bytes, sha256}} + um hash agregado estável do conjunto."""
    if not os.path.isdir(index_dir):
        return {}
    cache_path = os.path.join(index_dir, CACHE_NAME)
    cache: dict = {}
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as fh:
                cache = json.load(fh)
        except Exception:
            cache = {}

    files: dict[str, dict] = {}
    changed = False
    for name in sorted(os.listdir(index_dir)):
        if name == CACHE_NAME or not name.endswith(INDEX_SUFFIXES):
            continue
        path = os.path.join(index_dir, name)
        if not os.path.isfile(path):
            continue
        st = os.stat(path)
        hit = cache.get(name)
        if hit and hit.get("bytes") == st.st_size and hit.get("mtime_ns") == st.st_mtime_ns:
            files[name] = {"bytes": hit["bytes"], "sha256": hit["sha256"], "mtime_ns": hit["mtime_ns"]}
            continue
        files[name] = {"bytes": st.st_size, "sha256": _sha256(path), "mtime_ns": st.st_mtime_ns}
        changed = True

    if changed and use_cache:
        try:
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(files, fh, indent=2)
        except Exception:
            pass

    # Hash agregado: nome + sha de cada arquivo, em ordem. `mtime` fica DE FORA —
    # dois builds idênticos em máquinas diferentes têm datas diferentes e devem
    # produzir a mesma impressão digital.
    joined = "\n".join(f"{n}:{v['sha256']}" for n, v in sorted(files.items()))
    return {
        "dir": os.path.relpath(index_dir, _ROOT),
        "n_files": len(files),
        "total_bytes": sum(v["bytes"] for v in files.values()),
        "combined_sha256": hashlib.sha256(joined.encode("utf-8")).hexdigest(),
        "files": {n: {"bytes": v["bytes"], "sha256": v["sha256"]} for n, v in files.items()},
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="eval.fingerprint", description="Impressão digital do índice.")
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--json", action="store_true", help="imprime o dicionário completo")
    args = ap.parse_args(argv)

    fp = fingerprint(args.dir, use_cache=not args.no_cache)
    if not fp:
        raise SystemExit(f"diretório de índice ausente: {args.dir}")
    if args.json:
        print(json.dumps(fp, indent=2))
        return 0
    print(f"» {fp['dir']}  ·  {fp['n_files']} arquivos  ·  {fp['total_bytes'] / 1e6:.0f} MB")
    print(f"  impressão digital combinada: {fp['combined_sha256']}")
    print()
    for name, v in fp["files"].items():
        print(f"  {v['sha256'][:16]}…  {v['bytes'] / 1e6:8.1f} MB  {name}")
    print("\nUm terceiro que reconstrua o índice com `python -m retrieval.index_builder`")
    print("compara esta impressão digital com a gravada em eval/results/*.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
