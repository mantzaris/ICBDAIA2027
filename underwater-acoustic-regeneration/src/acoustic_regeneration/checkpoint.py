"""Atomic trusted-local checkpointing, including optimizer and random state."""
from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG checkpoint cannot resume on a CPU-only host")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    *, update: int, sampler_state: dict, metadata: dict, ema: dict,
                    scheduler=None, scaler=None) -> None:
    required = {"git_commit", "config_hash", "manifest_hash"}
    if not required.issubset(metadata):
        raise ValueError("Checkpoint requires commit, config, and manifest provenance")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "update": update, "sampler": sampler_state, "rng": capture_rng(), "metadata": metadata,
               "ema": ema, "scheduler": scheduler.state_dict() if scheduler is not None else None,
               "scaler": scaler.state_dict() if scaler is not None else None}
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    *, expected_metadata: dict, scheduler=None, scaler=None) -> dict:
    """Only load checkpoints created by this project; torch.load uses pickle.

    Map tensors to CPU first so torch CPU RNG tensors remain valid when restoring
    into a GPU model. Optimizer.load_state_dict handles parameter-device placement.
    """
    import inspect
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    payload = torch.load(Path(path), **kwargs)
    if payload["schema"] != 1 or payload["metadata"] != expected_metadata:
        raise ValueError("Checkpoint metadata differs from this run")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    for name, component in (("scheduler", scheduler), ("scaler", scaler)):
        if (payload[name] is None) != (component is None):
            raise ValueError("Checkpoint " + name + " configuration changed")
        if component is not None:
            component.load_state_dict(payload[name])
    restore_rng(payload["rng"])
    return payload
