# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import copy
import queue
import re
import threading
from array import array
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias, TypeVar

from transformers import AutoTokenizer, BatchEncoding, PythonBackend, TokenizersBackend

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_sentence_transformer_tokenizer_config

from .protocol import TokenizerLike

HfTokenizer: TypeAlias = PythonBackend | TokenizersBackend
_T = TypeVar("_T", bound=TokenizerLike)
logger = init_logger(__name__)


class ThreadSafeHFTokenizerMixin:
    """Mixin class for thread-safe HF fast tokenizers."""

    pass


def maybe_make_thread_pool(tokenizer: _T, copies: int = 1):
    """If `tokenizer` is a `TokenizersBackend`, modify the tokenizer
    in-place to make the public interface thread-safe by routing calls
    through a deep-copied tokenizer pool.

    Note that:
    - Only ``TokenizerLike``'s public interface is thread-safe.
      This doesn't include ``_tokenizer`` property nor any mutation
      methods like ``add_special_tokens`` or ``add_tokens``.
    - Adjacent method calls could happen on different deep copies.
    """
    if not isinstance(tokenizer, TokenizersBackend) or isinstance(
        tokenizer, ThreadSafeHFTokenizerMixin
    ):
        return tokenizer

    og_tokenizer = copy.copy(tokenizer)

    tokenizer_pool: queue.Queue[TokenizersBackend] = queue.Queue()
    for _ in range(copies):
        tokenizer_pool.put(copy.deepcopy(og_tokenizer))

    @contextlib.contextmanager
    def _borrow_from_pool():
        try:
            tok = tokenizer_pool.get_nowait()
            yield tok
        except queue.Empty:
            tok = copy.deepcopy(og_tokenizer)
            yield tok
        finally:
            tokenizer_pool.put(tok)

    class TokenizerPool(tokenizer.__class__, ThreadSafeHFTokenizerMixin):  # type: ignore
        def apply_chat_template(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.apply_chat_template(*args, **kwargs)

        def batch_decode(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.batch_decode(*args, **kwargs)

        def batch_encode(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.batch_encode(*args, **kwargs)

        def convert_tokens_to_ids(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.convert_tokens_to_ids(*args, **kwargs)

        def convert_ids_to_tokens(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.convert_ids_to_tokens(*args, **kwargs)

        def convert_tokens_to_string(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.convert_tokens_to_string(*args, **kwargs)

        def decode(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.decode(*args, **kwargs)

        def encode(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.encode(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok(*args, **kwargs)

        def __reduce__(self):
            return maybe_make_thread_pool, (og_tokenizer, copies)

    TokenizerPool.__name__ = f"TokenizerPool{og_tokenizer.__class__.__name__}"

    tokenizer.__class__ = TokenizerPool
    # Return the tokenizer: TokenizerPool.__reduce__ reconstructs through this
    # function, so falling off the end would unpickle to None (issue #45433).
    return tokenizer


_FAST_ENCODE_KWARGS = frozenset({"add_special_tokens", "truncation", "max_length"})


def _fast_encode_probes(tokenizer: HfTokenizer) -> list[str]:
    """Strings on which the direct and the generic encode paths must agree:
    every special token alone, glued to text and wrapped in whitespace, plus
    multilingual and long text."""
    specials = [t for t in tokenizer.get_added_vocab() if t][:512]
    probes = [
        "",
        " ",
        "\n\n  \t",
        "Hello, world! 你好，世界。 Здравствуй, мир. 1234567890",
        "def f(x):\n    return {'a': [1, 2, 3]}  # comment\n",
        "ab" * 3000 + " " * 300 + "\u00e9\u0301" * 50,
    ]
    for tok in specials:
        probes += [
            tok,
            f"x{tok}y",
            f" {tok} ",
            f"\n{tok}\n",
            f"{tok}{tok}",
            f"a {tok}b",
        ]
    probes.append(" text ".join(specials) + " tail")
    return probes


def _make_fast_encode(tokenizer: HfTokenizer) -> Callable[[str], list[int]] | None:
    """A faster ``tokenizer(text, add_special_tokens=False)["input_ids"]``, or
    None if it would not be identical.

    Tiktoken-backed remote tokenizers (Kimi's ``TikTokenTokenizer``) encode a
    plain string natively in ``encode(text)``, but ``__call__`` takes the
    generic slow path: split on added tokens in Python, encode each piece, map
    the ids to token strings and back to ids one by one (~1.7 us/token, i.e.
    0.2 s for a 100K-token agentic prompt). This keeps the same split (so a
    tokenizer that internally chunks long strings sees the same pieces) and
    only drops the id -> string -> id round trip, ~3x faster. Enabled only if
    it matches the generic path on every probe."""
    if not envs.VLLM_TOKENIZER_FAST_ENCODE:
        return None
    if not (hasattr(tokenizer, "_encode_text_piece") and hasattr(tokenizer, "model")):
        return None
    added = dict(tokenizer.get_added_vocab())
    if not added:
        return None
    pattern = re.compile(
        "(" + "|".join(re.escape(t) for t in sorted(added, key=len, reverse=True)) + ")"
    )
    native = tokenizer.encode

    def fast_encode(text: str) -> list[int]:
        ids: list[int] = []
        for i, part in enumerate(pattern.split(text)):
            if part:
                if i % 2:
                    ids.append(added[part])
                else:
                    ids.extend(native(part))
        return ids

    try:
        for text in _fast_encode_probes(tokenizer):
            if fast_encode(text) != list(
                tokenizer(text, add_special_tokens=False)["input_ids"]
            ):
                logger.warning(
                    "Fast tokenizer encode disabled: it differs from the generic "
                    "path on %r.",
                    text[:80],
                )
                return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Fast tokenizer encode disabled: self-check failed: %s", e)
        return None
    return fast_encode


class _PieceEncodeCache:
    """Memoize a pure ``text -> token ids`` encoder for long pieces.

    Tiktoken-backed remote tokenizers (Kimi) encode every chat message through
    ``_encode_text_piece``. In multi-turn agentic traffic each request carries
    the whole conversation again, so the same long message strings are
    re-encoded on every turn (~0.7 us/token: ~70 ms of frontend time per 100K
    tokens, serialized on the renderer thread). Results are cached by the exact
    text (the dict compares it on a hit, so a hash collision cannot return a
    wrong encoding) in compact arrays, LRU-evicted past ``max_tokens``.
    """

    def __init__(self, fn: Callable[..., list[int]], min_chars: int, max_tokens: int):
        self._fn = fn
        self._min_chars = min_chars
        self._max_tokens = max_tokens
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, bool], array] = OrderedDict()
        self._num_tokens = 0
        self.hits = 0
        self.misses = 0

    def __call__(self, text: str, allow_special_tokens: bool = True) -> list[int]:
        if len(text) < self._min_chars:
            return self._fn(text, allow_special_tokens=allow_special_tokens)
        key = (text, allow_special_tokens)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                return cached.tolist()
        ids = self._fn(text, allow_special_tokens=allow_special_tokens)
        packed = array("I", ids)
        with self._lock:
            self.misses += 1
            if key not in self._entries and len(packed) <= self._max_tokens:
                self._entries[key] = packed
                self._num_tokens += len(packed)
                while self._num_tokens > self._max_tokens:
                    _, old = self._entries.popitem(last=False)
                    self._num_tokens -= len(old)
        return ids


def get_cached_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """By default, transformers will recompute multiple tokenizer properties
    each time they are called, leading to a significant slowdown.
    This proxy caches these properties for faster access.
    """
    cached_tokenizer = copy.copy(tokenizer)

    tokenizer_all_special_ids = tokenizer.all_special_ids
    tokenizer_all_special_tokens = tokenizer.all_special_tokens
    tokenizer_vocab = tokenizer.get_vocab()
    tokenizer_len = len(tokenizer)
    # The underlying tokenizer class could be a specific backend,
    # which does not always implement is_fast in Transformers
    tokenizer_is_fast = getattr(tokenizer, "is_fast", True)

    max_token_id = max(tokenizer_vocab.values())
    max_chars_per_token = max(len(tok) for tok in tokenizer_vocab)

    # Some tokenizers (e.g., QwenTokenizer) have special tokens that
    # are added and included in the implementation of the vocab_size
    # property, but not in get_vocab(); if there is an implementation
    # of vocab size, we should take the greater value. vocab_size is a
    # count, so the largest id it implies is vocab_size - 1.
    if hasattr(tokenizer, "vocab_size"):
        with contextlib.suppress(NotImplementedError):
            max_token_id = max(max_token_id, tokenizer.vocab_size - 1)

    piece_cache_tokens = envs.VLLM_TOKENIZER_PIECE_CACHE_TOKENS
    if piece_cache_tokens > 0 and hasattr(tokenizer, "_encode_text_piece"):
        # Instance attribute on the copy: the remote tokenizer's own methods
        # (chat template included) call self._encode_text_piece.
        cached_tokenizer._encode_text_piece = _PieceEncodeCache(
            tokenizer._encode_text_piece, min_chars=1024, max_tokens=piece_cache_tokens
        )
        logger.info(
            "Tokenizer %s: caching encodings of long text pieces (up to %d tokens).",
            tokenizer.__class__.__name__,
            piece_cache_tokens,
        )

    fast_encode = _make_fast_encode(tokenizer)
    if fast_encode is not None:
        logger.info(
            "Tokenizer %s: plain-text encodes skip the id/string round trip.",
            tokenizer.__class__.__name__,
        )

    class CachedTokenizer(tokenizer.__class__):  # type: ignore
        def __call__(self, text=None, *args, **kwargs):
            if (
                fast_encode is not None
                and isinstance(text, str)
                and not args
                and kwargs.get("add_special_tokens") is False
                and kwargs.keys() <= _FAST_ENCODE_KWARGS
            ):
                ids = fast_encode(text)
                max_length = kwargs.get("max_length")
                if (
                    kwargs.get("truncation")
                    and max_length is not None
                    and len(ids) > max_length
                ):
                    ids = (
                        ids[-max_length:]
                        if self.truncation_side == "left"
                        else ids[:max_length]
                    )
                return BatchEncoding(
                    {"input_ids": ids, "attention_mask": [1] * len(ids)}
                )
            return super().__call__(text, *args, **kwargs)

        @property
        def all_special_ids(self) -> list[int]:
            return tokenizer_all_special_ids

        @property
        def all_special_tokens(self) -> list[str]:
            return tokenizer_all_special_tokens

        @property
        def max_token_id(self) -> int:
            return max_token_id

        @property
        def max_chars_per_token(self) -> int:
            return max_chars_per_token

        @property
        def is_fast(self) -> bool:
            return tokenizer_is_fast

        def get_vocab(self) -> dict[str, int]:
            return tokenizer_vocab

        def __len__(self) -> int:
            return tokenizer_len

        def __reduce__(self):
            return get_cached_tokenizer, (tokenizer,)

    CachedTokenizer.__name__ = f"Cached{tokenizer.__class__.__name__}"

    cached_tokenizer.__class__ = CachedTokenizer
    return cached_tokenizer


class CachedHfTokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> HfTokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                path_or_repo_id,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                cache_dir=download_dir,
                **kwargs,
            )
        except ValueError as e:
            # If the error pertains to the tokenizer class not existing or not
            # currently being imported,
            # suggest using the --trust-remote-code flag.
            if not trust_remote_code and (
                "does not exist or is not currently imported." in str(e)
                or "requires you to execute the tokenizer file" in str(e)
            ):
                err_msg = (
                    "Failed to load the tokenizer. If the tokenizer "
                    "is a custom tokenizer not yet available in the "
                    "HuggingFace transformers library, consider "
                    "setting `trust_remote_code=True` in LLM or using "
                    "the `--trust-remote-code` flag in the CLI. If the "
                    "model was created with a newer version of "
                    "transformers, consider upgrading: "
                    "`uv pip install --upgrade transformers`"
                )
                raise RuntimeError(err_msg) from e
            else:
                raise e

        # The special_tokens in tokenizer should also be
        # controlled by do_lower_case in encoder_config
        encoder_config = get_sentence_transformer_tokenizer_config(
            path_or_repo_id, revision
        )
        if isinstance(encoder_config, dict) and encoder_config.get(
            "do_lower_case", False
        ):
            special_tokens_map = {
                k: v.lower() for k, v in tokenizer.special_tokens_map.items()
            }
            tokenizer.add_special_tokens(special_tokens_map)

        return get_cached_tokenizer(tokenizer)
