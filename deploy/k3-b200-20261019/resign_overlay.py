# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Re-sign source.lock.json for a source tree that differs from the freeze.

assemble_runtime.py refuses to build an overlay unless every runtime file still
matches source.lock.json. After a sync with upstream (or a backported fix) the
Python sources legitimately differ, so recompute those hashes from this
checkout and drop entries the new tree no longer ships. Native library and
model locks are untouched.

usage: resign_overlay.py [--check]
"""

import argparse
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent.parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would change without writing source.lock.json",
    )
    args = parser.parse_args()

    lock_path = PACKAGE / "source.lock.json"
    lock = json.loads(lock_path.read_text())

    changed = 0
    dropped: list[str] = []
    for name, item in lock["runtime_files"].items():
        source = REPO / item["repository_path"]
        if not source.is_file():
            dropped.append(name)
            continue
        digest = sha(source)
        if digest != item["sha256"]:
            changed += 1
            item["sha256"] = digest

    for name in dropped:
        del lock["runtime_files"][name]

    if not args.check:
        lock_path.write_text(json.dumps(lock, indent=1) + "\n")

    print(
        json.dumps(
            {
                "changed": changed,
                "dropped": len(dropped),
                "total": len(lock["runtime_files"]),
                "written": not args.check,
            }
        )
    )


if __name__ == "__main__":
    main()
