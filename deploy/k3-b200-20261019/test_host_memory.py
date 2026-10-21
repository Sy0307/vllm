# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
import unittest
from pathlib import Path

from host_memory import check_mooncake_memory


class HostMemoryTest(unittest.TestCase):
    def test_rejects_original_segment_on_memory_constrained_host(self):
        with tempfile.TemporaryDirectory() as directory:
            meminfo = Path(directory) / "meminfo"
            meminfo.write_text(f"MemAvailable: {1300 * 1024**2} kB\n")
            with self.assertRaisesRegex(RuntimeError, "Reduce mc_segment"):
                check_mooncake_memory("160GB", meminfo=meminfo)
            result = check_mooncake_memory("64GB", meminfo=meminfo)
            self.assertEqual(result["mc_segment"], "64GB")

    def test_keeps_original_segment_when_memory_is_available(self):
        with tempfile.TemporaryDirectory() as directory:
            meminfo = Path(directory) / "meminfo"
            meminfo.write_text(f"MemAvailable: {1800 * 1024**2} kB\n")
            result = check_mooncake_memory("160GB", meminfo=meminfo)
            self.assertEqual(result["mc_segment"], "160GB")

    def test_invalid_size_fails_before_reading_memory(self):
        with self.assertRaisesRegex(ValueError, "mc_segment"):
            check_mooncake_memory("64", meminfo=Path("/does/not/exist"))


if __name__ == "__main__":
    unittest.main()
