"""Apply one recorded, value-preserving NumPy fix to the legacy CQT dependency.

Run inside .venv-transfer, after installing cqt-nsgt-pytorch==0.0.10. This does
not change filters, learned weights, or the inpainting algorithm.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SHA256 = "8d0a61d18e59f53dc6876bdc22a7e4dc9bdd10dc24deb880fa2ef09e830c9dca"
BEFORE = "np.clip(M, min_win, np.inf, out=M)"
AFTER = "np.maximum(M, min_win, out=M)"


def main():
    module = importlib.util.find_spec("cqt_nsgt_pytorch")
    path = Path(next(iter(module.submodule_search_locations))) / "nsgfwin.py"
    original = path.read_bytes()
    checksum = hashlib.sha256(original).hexdigest()
    target = ROOT / "provenance/transfer_numpy_compat.json"
    if target.exists() and checksum == json.loads(target.read_text())["patched_sha256"]:
        print("Legacy CQT NumPy compatibility fix already verified")
        return
    if checksum != ORIGINAL_SHA256 or original.count(BEFORE.encode()) != 1:
        raise ValueError("Unexpected dependency source; inspect before changing it")
    patched = original.replace(BEFORE.encode(), AFTER.encode())
    path.write_bytes(patched)
    report = {"package": "cqt-nsgt-pytorch", "version": "0.0.10", "license": "MIT",
              "file": "cqt_nsgt_pytorch/nsgfwin.py", "original_sha256": checksum,
              "patched_sha256": hashlib.sha256(patched).hexdigest(),
              "before": BEFORE, "after": AFTER,
              "reason": "NumPy 1.26 rejects float infinity in an in-place integer clip. M contains finite integer window lengths; lower bounding by integer min_win with maximum is algebraically equivalent to clipping at positive infinity.",
              "scope": "Legacy dependency compatibility only; no signal, model, or inference hyperparameter changes"}
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
