"""Compatibility check for the published, hash-verified Moliner checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
import typing
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode

ROOT = Path(__file__).resolve().parents[1]
REVISION = "28b0e92683d58d0ab12a73e675a4da387b00caa5"


def load_published_checkpoint(path):
    # The published archive also stores OmegaConf settings. Restrict loading to
    # tensors, primitive containers, and these explicitly inspected data classes.
    allowed = [DictConfig, ListConfig, Metadata, ContainerMetadata, AnyNode,
               defaultdict, typing.Any, dict, list, int]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    path = ROOT / "checkpoints/transfer/22k_8s-850000.pt"
    from acoustic_regeneration.registry import file_hash
    if file_hash(path) != "25436b182f906fbcfe19b5147937f7fb4226542230381fcecb77114bdba60d87":
        raise ValueError("Published checkpoint hash mismatch")
    payload = load_published_checkpoint(path)
    details = {"iteration": payload["it"], "keys": list(payload), "ema_tensor_elements": sum(x.numel() for x in payload["ema"].values()),
               "ema_first_keys": list(payload["ema"])[:10],
               "published_configuration": OmegaConf.to_container(payload["args"], resolve=False)}
    (ROOT / "provenance/transfer_checkpoint_inspection.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps({key: value for key, value in details.items() if key != "published_configuration"}, indent=2))
    if args.inspect_only:
        return
    raise NotImplementedError("Use --inspect-only until the exact model configuration is checked")


if __name__ == "__main__":
    main()
