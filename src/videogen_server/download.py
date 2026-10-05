"""`uv run videogen-download --config configs/ltx25.json` - fetch weights into the HF cache."""

from __future__ import annotations

import argparse
import os
import shutil
from fnmatch import fnmatch

from vidgen_core.selfhost import SelfHostConfig, load_selfhost_config


def repo_size_bytes(cfg: SelfHostConfig, token: str | None) -> int:
    from huggingface_hub import HfApi

    info = HfApi().model_info(
        cfg.model.model_id, revision=cfg.model.revision, files_metadata=True, token=token
    )
    total = 0
    for s in info.siblings or []:
        name = s.rfilename
        if cfg.model.ignore_patterns and any(fnmatch(name, p) for p in cfg.model.ignore_patterns):
            continue
        if cfg.model.allow_patterns and not any(fnmatch(name, p) for p in cfg.model.allow_patterns):
            continue
        total += s.size or 0
    return total


def ensure_model(cfg: SelfHostConfig) -> str:
    os.environ.setdefault("HF_HOME", str(cfg.storage.hf_home))
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")  # required: repo is gated
    need = repo_size_bytes(cfg, token)
    cfg.storage.hf_home.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cfg.storage.hf_home).free
    if free - need < cfg.storage.min_free_gb * 1024**3:
        raise RuntimeError(
            f"Thiếu đĩa: cần {need / 1e9:.1f}GB, trống {free / 1e9:.1f}GB, "
            f"phải chừa {cfg.storage.min_free_gb}GB"
        )
    # No local_dir: it would duplicate the data next to the cache.
    return snapshot_download(
        repo_id=cfg.model.model_id,
        revision=cfg.model.revision,
        allow_patterns=cfg.model.allow_patterns,
        ignore_patterns=cfg.model.ignore_patterns,
        token=token,
        max_workers=8,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true", help="only print size + free disk")
    a = ap.parse_args()
    cfg = load_selfhost_config(a.config)
    if a.dry_run:
        need = repo_size_bytes(cfg, os.environ.get("HF_TOKEN"))
        print(
            f"need {need / 1e9:.1f}GB, free {shutil.disk_usage(cfg.storage.hf_home).free / 1e9:.1f}GB"
        )
        return
    print(ensure_model(cfg))
