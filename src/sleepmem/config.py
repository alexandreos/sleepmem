"""Load config/base.yaml, deep-merge a sleep config on top, resolve paths and the model."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(sleep_config: str | Path, base: str | Path | None = None) -> dict:
    base = Path(base) if base else PROJECT_ROOT / "config" / "base.yaml"
    cfg = yaml.safe_load(base.read_text())
    sleep_path = Path(sleep_config)
    cfg = deep_merge(cfg, yaml.safe_load(sleep_path.read_text()) or {})
    cfg["name"] = sleep_path.stem
    cfg["config_path"] = str(sleep_path)
    return cfg


def config_hash(cfg: dict) -> str:
    """Hash of everything that defines the experiment (paths excluded, so sandboxes match)."""
    keep = {k: v for k, v in cfg.items() if k not in ("paths", "root", "config_path")}
    keep = json.loads(json.dumps(keep, default=str))  # int keys -> str, so sorting works
    blob = json.dumps(keep, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def root(cfg: dict) -> Path:
    return Path(cfg.get("root") or PROJECT_ROOT)


def path(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else root(cfg) / p


def project_path(p: str | Path) -> Path:
    """Paths to committed inputs (fixtures, eval sets) resolve against the project, not --root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def model_spec(cfg: dict) -> dict:
    spec = cfg["models"].get(cfg["rung"]) or cfg["models"].get(str(cfg["rung"]))
    if spec is None:
        raise KeyError(f"no model for rung {cfg['rung']!r} in config/base.yaml")
    return spec


def resolve_model(cfg: dict, allow_download: bool = False) -> Path:
    """Local snapshot path of the pinned model; downloads only under the size limit."""
    from huggingface_hub import HfApi, snapshot_download

    spec = model_spec(cfg)
    try:
        return Path(
            snapshot_download(spec["repo"], revision=spec["revision"], local_files_only=True)
        )
    except Exception:
        pass
    info = HfApi().model_info(spec["repo"], revision=spec["revision"], files_metadata=True)
    gb = sum(s.size or 0 for s in info.siblings) / 1e9
    if gb > cfg["max_auto_download_gb"] and not allow_download:
        raise RuntimeError(
            f"{spec['repo']} is {gb:.1f} GB (> {cfg['max_auto_download_gb']} GB): "
            "ask first, then rerun with --allow-download"
        )
    return Path(snapshot_download(spec["repo"], revision=spec["revision"]))
