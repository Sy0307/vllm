# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The frontend cache of long text-piece encodings returns exactly what the
wrapped encoder returns and stays within its token budget."""

from vllm.tokenizers.hf import _PieceEncodeCache, _make_special_piece_fast_path


class _CountingEncoder:
    def __init__(self):
        self.calls = 0

    def __call__(self, text, allow_special_tokens=True):
        self.calls += 1
        return [ord(c) % 1000 + (1 if allow_special_tokens else 0) for c in text]


def test_short_pieces_are_not_cached():
    enc = _CountingEncoder()
    cache = _PieceEncodeCache(enc, min_chars=8, max_tokens=100)
    assert cache("abc") == enc("abc")
    cache("abc")
    assert enc.calls == 3 and cache.hits == 0


def test_long_piece_hits_return_identical_ids():
    enc = _CountingEncoder()
    cache = _PieceEncodeCache(enc, min_chars=8, max_tokens=100)
    text = "a long repeated message"
    first = cache(text)
    second = cache(text)
    assert first == second == enc(text)
    assert cache.hits == 1 and cache.misses == 1
    # The special-token mode is part of the key.
    assert cache(text, allow_special_tokens=False) == enc(text, False)
    assert cache.misses == 2


def test_lru_eviction_keeps_token_budget():
    enc = _CountingEncoder()
    cache = _PieceEncodeCache(enc, min_chars=4, max_tokens=20)
    a, b, c = "a" * 10, "b" * 10, "c" * 10
    cache(a)
    cache(b)
    cache(a)  # a becomes most recent
    cache(c)  # evicts b, the least recently used
    calls = enc.calls
    cache(a)
    cache(c)
    assert enc.calls == calls  # both still cached
    cache(b)
    assert enc.calls == calls + 1  # b was evicted
    assert cache._num_tokens <= 20


def test_piece_larger_than_budget_is_not_stored():
    enc = _CountingEncoder()
    cache = _PieceEncodeCache(enc, min_chars=4, max_tokens=5)
    text = "x" * 10
    assert cache(text) == enc(text)
    assert cache._num_tokens == 0 and not cache._entries


class _FakeTiktoken:
    def __init__(self, specials):
        self._special_tokens = specials


class _FakeTokenizer:
    def __init__(self, specials):
        self.model = _FakeTiktoken(specials)

    def _encode_text_piece(self, text, allow_special_tokens=True):
        if allow_special_tokens and text in self.model._special_tokens:
            return [self.model._special_tokens[text]]
        return [ord(c) for c in text]


def test_lone_special_tokens_are_looked_up():
    tok = _FakeTokenizer({"<|open|>": 900, "<|sep|>": 901})
    calls = []

    def encode_piece(text, allow_special_tokens=True):
        calls.append((text, allow_special_tokens))
        return tok._encode_text_piece(text, allow_special_tokens)

    fast = _make_special_piece_fast_path(tok, encode_piece)
    assert fast is not None
    calls.clear()
    assert fast("<|open|>") == [900]
    assert not calls  # looked up, not encoded
    # Not a lone special token, or specials not allowed: the wrapped encoder.
    assert fast("<|open|>x") == encode_piece("<|open|>x")
    assert fast("<|sep|>", allow_special_tokens=False) == [ord(c) for c in "<|sep|>"]


def test_special_fast_path_rejects_mismatch():
    tok = _FakeTokenizer({"<|open|>": 900})

    def wrong(text, allow_special_tokens=True):
        return [1]

    assert _make_special_piece_fast_path(tok, wrong) is None
