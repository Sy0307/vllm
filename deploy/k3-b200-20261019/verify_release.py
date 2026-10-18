# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Verify the exact scored Python source and native-library bytes before startup."""

import hashlib
import json
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(overlay):
    lock = json.loads((PACKAGE / "source.lock.json").read_text())
    for name, item in lock["runtime_files"].items():
        path = overlay / name
        if not path.is_file() or sha(path) != item["sha256"]:
            raise RuntimeError(f"Release source hash mismatch: {name}")
    for name, item in lock["native_libraries"].items():
        if sha(overlay / name) != item["sha256"]:
            raise RuntimeError(f"Native library hash mismatch: {name}")
    models = json.loads((PACKAGE / "model.lock.json").read_text())["models"]
    for directory, model in models.items():
        root = Path(directory)
        for name, item in model["files"].items():
            if sha(root / name) != item["sha256"]:
                raise RuntimeError(f"Model metadata hash mismatch: {directory}/{name}")
        for item in model["weight_files"]:
            if (root / item["name"]).stat().st_size != item["size"]:
                raise RuntimeError(f"Weight size mismatch: {directory}/{item['name']}")
    print(
        json.dumps(
            {
                "verified": True,
                "runtime_files": len(lock["runtime_files"]),
                "native_libraries": len(lock["native_libraries"]),
                "overlay": str(overlay),
            }
        )
    )


if __name__ == "__main__":
    verify(Path(sys.argv[1]) if len(sys.argv) > 1 else PACKAGE / "overlay")
