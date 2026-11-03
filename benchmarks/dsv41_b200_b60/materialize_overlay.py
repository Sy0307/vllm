"""Combine the archived role files with a matching installed vLLM package."""

import argparse
import shutil
from pathlib import Path


def materialize(base: Path, overrides: Path, output: Path) -> None:
    output.mkdir()
    names = {p.name for p in base.iterdir()} if base.is_dir() else set()
    names.update(p.name for p in overrides.iterdir())
    for name in sorted(names):
        source = overrides / name
        target = output / name
        if source.is_dir():
            materialize(base / name, source, target)
        elif source.is_file():
            shutil.copy2(source, target)
        else:
            target.symlink_to((base / name).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("prefill", "decode"))
    parser.add_argument("--base-vllm", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    base = args.base_vllm.resolve()
    output = args.output.resolve()
    overrides = Path(__file__).parent / "overlays" / args.role / "vllm"
    if not (base / "__init__.py").is_file():
        parser.error("--base-vllm must be the installed vllm package directory")
    if output.exists():
        parser.error("--output must be a new directory")
    if output.is_relative_to(base) or output.is_relative_to(overrides.resolve()):
        parser.error("--output must be outside both source trees")
    output.mkdir(parents=True)
    materialize(base, overrides, output / "vllm")
    print(output)


if __name__ == "__main__":
    main()
