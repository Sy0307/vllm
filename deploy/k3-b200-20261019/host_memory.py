# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reject Mooncake allocations that exceed the host's available memory."""

from pathlib import Path


def check_mooncake_memory(
    segment: str,
    *,
    meminfo: Path = Path("/proc/meminfo"),
) -> dict[str, int | str]:
    units = {"B": 0, "KB": 1, "MB": 2, "GB": 3, "TB": 4}
    unit = next((suffix for suffix in reversed(units) if segment.endswith(suffix)), "")
    amount = segment[: -len(unit)] if unit else ""
    if not amount.isascii() or not amount.isdecimal():
        raise ValueError("mc_segment must use an integer and B/KB/MB/GB/TB")
    size = int(amount) * 1024 ** units[unit]
    fields = dict(line.split(":", 1) for line in meminfo.read_text().splitlines())
    available = int(fields["MemAvailable"].split()[0]) * 1024
    # The TP8/PP2 launcher starts eight local workers. Each also has a 4 GiB
    # Mooncake buffer; reserve 64 GiB for the remaining serving allocations.
    required = 8 * (size + 4 * 1024**3) + 64 * 1024**3
    if required > available:
        raise RuntimeError(
            f"Mooncake mc_segment={segment} across 8 workers needs at least "
            f"{required / 1024**3:.1f} GiB available host memory including "
            f"buffers/headroom; only {available / 1024**3:.1f} GiB is available. "
            "Reduce mc_segment or release host memory before launching."
        )
    return {
        "mc_segment": segment,
        "local_workers": 8,
        "required_bytes": required,
        "available_bytes": available,
    }
