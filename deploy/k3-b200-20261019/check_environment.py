# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check dependency resolution using the exact serving interpreter/environment."""

import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
lock = json.loads((PACKAGE / "environment.lock.json").read_text())
assert sys.version.split()[0] == lock["python"].split()[0], "Python version differs"
for filename, text in lock["pth"].items():
    assert Path(filename).read_text() == text, (
        f"Dependency search path changed: {filename}"
    )
for name, expected in lock["module_origins"].items():
    if expected is None or name in ("vllm", "tokenspeed_mla", "k3memext"):
        continue
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin == expected["origin"], (
        name,
        spec,
        expected,
    )
seen = set()
for package in lock["packages_in_resolution_order"]:
    name = package["name"]
    if name.lower() in seen:
        continue
    seen.add(name.lower())
    assert importlib.metadata.version(name) == package["version"], name
torch = importlib.import_module("torch")

assert torch.__file__ == lock["module_origins"]["torch"]["origin"]
print(
    json.dumps(
        {
            "environment_verified": True,
            "packages": len(seen),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "python": sys.version.split()[0],
        }
    )
)
