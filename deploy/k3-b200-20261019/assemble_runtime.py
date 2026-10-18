# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Assemble a new immutable runtime from locked native binaries and branch sources.

Run from the checkout after provisioning the dependency archive. Existing paths
are never replaced. This does not install packages or start a GPU service.
"""

import json
import os
import shutil
from pathlib import Path

from verify_release import sha, verify

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent.parent
lock = json.loads((PACKAGE / "source.lock.json").read_text())
native = Path(lock["native_overlay"])
destination = PACKAGE / "overlay"
assert not destination.exists(), "Choose a fresh checkout; overlay already exists"
assert native.is_dir(), "Provision the locked dependency archive first"
for name, item in lock["native_libraries"].items():
    assert sha(native / name) == item["sha256"], name
for name, item in lock["runtime_files"].items():
    source = REPO / item["repository_path"]
    assert sha(source) == item["sha256"], str(source)
shutil.copytree(
    native, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
)
for name, item in lock["runtime_files"].items():
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / item["repository_path"], target)
verify(destination)
for root, _, files in os.walk(destination):
    for name in files:
        path = Path(root) / name
        os.chmod(path, path.stat().st_mode & ~0o222)
print("ASSEMBLY_COMPLETE")
