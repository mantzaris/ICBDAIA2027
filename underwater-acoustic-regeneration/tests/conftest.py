import torch
from pathlib import Path


def pytest_sessionstart(session):
    torch.set_num_threads(1)
    cache = Path(__file__).resolve().parents[1] / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    session.config.option.basetemp = str(cache / "pytest")
