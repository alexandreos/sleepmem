"""Wrap `mlx_lm.lora`: write its YAML config into the sleep dir and run it as a subprocess."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def lora_config(
    cfg: dict,
    model_path: Path,
    data_dir: Path,
    adapter_dir: Path,
    resume: Path | None,
    iters: int | None = None,
) -> dict:
    lc = dict(cfg["train"]["lora"])
    if iters is not None:
        lc["iters"] = iters
    lc.update(
        model=str(model_path),
        train=True,
        data=str(data_dir),
        adapter_path=str(adapter_dir),
        seed=cfg["seed"],
        resume_adapter_file=str(resume) if resume else None,
    )
    return lc


def resume_from(cfg: dict, adapters_dir: Path) -> Path | None:
    cur = adapters_dir / "current" / "adapters.safetensors"
    if cfg["train"]["adapter_mode"] == "cumulative" and cur.exists():
        return cur.resolve()
    return None


def run_lora(lc: dict, sleep_dir: Path) -> Path:
    cfg_path = sleep_dir / "lora.yaml"
    cfg_path.write_text(yaml.safe_dump(lc, sort_keys=False))
    with open(sleep_dir / "train.log", "w") as log:
        proc = subprocess.run(
            [sys.executable, "-m", "mlx_lm.lora", "--config", str(cfg_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"mlx_lm.lora failed ({proc.returncode}); see {sleep_dir}/train.log")
    out = Path(lc["adapter_path"])
    if not (out / "adapters.safetensors").exists():
        raise RuntimeError(f"no adapters.safetensors in {out}")
    return out
