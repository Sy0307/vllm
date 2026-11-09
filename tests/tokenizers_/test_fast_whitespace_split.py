# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fast path for Kimi's whitespace-run split yields exactly what the
per-character split yields."""

import random

import pytest

from vllm.tokenizers.hf import _make_fast_whitespace_split


def _kimi_split(s: str, max_consecutive_slice_len: int):
    # TikTokenTokenizer._split_whitespaces_or_nonwhitespaces (Kimi K3 remote code).
    current_slice_len = 0
    current_slice_is_space = s[0].isspace() if len(s) > 0 else False
    slice_start = 0
    for i in range(len(s)):
        is_now_space = s[i].isspace()
        if current_slice_is_space ^ is_now_space:
            current_slice_len = 1
            current_slice_is_space = is_now_space
        else:
            current_slice_len += 1
            if current_slice_len > max_consecutive_slice_len:
                yield s[slice_start:i]
                slice_start = i
                current_slice_len = 1
    yield s[slice_start:]


_ALPHABET = ["a", " ", "\n", "\t", " ", " ", "　", "\x1c", "\x85", "字"]


def test_matches_reference_on_random_runs():
    fast = _make_fast_whitespace_split(_kimi_split)
    assert fast is not None
    rng = random.Random(0)
    for _ in range(5000):
        max_len = rng.randint(1, 12)
        text = "".join(
            rng.choice(_ALPHABET) * rng.randint(1, 15)
            for _ in range(rng.randint(0, 30))
        )
        assert list(fast(text, max_len)) == list(_kimi_split(text, max_len))


@pytest.mark.parametrize(
    "text",
    [
        "word " * 20_000,  # long, no long run: one piece
        "q" * 30_000 + " tail",  # long non-whitespace run: split
        "a" + " " * 30_000 + "b",  # long whitespace run: split
        "　" * 26_000,  # Unicode whitespace run
    ],
)
def test_long_pieces(text):
    fast = _make_fast_whitespace_split(_kimi_split)
    assert list(fast(text, 25_000)) == list(_kimi_split(text, 25_000))


def test_rejects_a_different_split():
    def split_every_char(s, max_len):
        yield from s

    assert _make_fast_whitespace_split(split_every_char) is None
