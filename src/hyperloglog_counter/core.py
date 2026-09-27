"""Core implementation of a HyperLogLog distinct-value estimator.

This module implements the HyperLogLog algorithm as described in
Flajolet, Fusy, Gandouet, Meunier (2007). The design favors simplicity and
determinism over peak accuracy: we use a fixed 64-bit hash via hashlib so the
library has zero third-party dependencies and behaves identically across
platforms.

Interpretation chosen (stated plainly because the brief is ambiguous):
- Elements are accepted as ``bytes``. Callers are responsible for serializing
  richer types. This keeps hashing explicit and avoids hidden surprises from
  Python's ``str.__hash__`` being randomized per-process.
- We expose ``precision`` (the number of register-index bits, ``p``). The
  register count is ``2**p``. ``p`` must satisfy ``4 <= p <= 16``; the lower
  bound keeps bias correction meaningful, the upper bound keeps memory sane.
- Bias correction uses the standard small-range correction when the raw
  estimate is below ``2.5 * m``. We do NOT implement the large-range
  correction for 64-bit hashes because it is only relevant near 2^64, which is
  unreachable with real data; pretending to handle it would be dishonest.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, Iterator, List, Tuple


def _alpha(m: int) -> float:
    """Return the bias-correction constant for a register count ``m``.

    WHY: The constant ``alpha`` scales the raw indicator sum so that the
    estimator is (asymptotically) unbiased. Only two closed forms are needed
    for the practical range of ``m``; we fall back to the general asymptotic
    value for anything outside the documented precision range rather than
    crashing.
    """
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    # For m >= 128 the standard constant 0.7213 / (1 + 1.079 / m) is used.
    return 0.7213 / (1.0 + 1.079 / float(m))


def _hash64(data: bytes) -> int:
    """Return a stable, well-distributed 64-bit hash of ``data``.

    WHY: We use SHA-256 truncated to 64 bits instead of Python's built-in
    ``hash()`` because ``hash()`` is salted per interpreter process for
    strings and is not guaranteed stable across runs or platforms. A
    cryptographic hash is overkill for distribution but is free in the
    standard library and behaves identically everywhere.
    """
    digest = hashlib.sha256(data).digest()
    # Take the first 8 bytes as a little-endian unsigned 64-bit integer.
    return int.from_bytes(digest[:8], "little")


def _rho(rank_bits: int, bit_width: int) -> int:
    """Position of the leftmost 1-bit in ``rank_bits`` plus 1.

    ``bit_width`` is the number of bits available to the left of the
    register-index portion. Returns ``bit_width + 1`` if all those bits are
    zero (meaning the value is effectively zero in the remaining window).
    """
    if rank_bits == 0:
        return bit_width + 1
    # bit_length gives floor(log2(x)) + 1, i.e. 1-based position of the
    # highest set bit counting from the LSB. We want counting from the MSB
    # of the available window.
    return bit_width - rank_bits.bit_length() + 1


class HyperLogLog:
    """A HyperLogLog distinct-element counter.

    Memory is ``2**precision`` 8-byte registers. Adding the same element twice
    (or twenty times) produces the same register state as adding it once.

    Parameters
    ----------
    precision:
        Number of bits used to index registers, ``p``. Must satisfy
        ``4 <= p <= 16``. The standard error is approximately
        ``1.04 / sqrt(2**p)``.
    """

    __slots__ = ("_precision", "_m", "_registers", "_w_bits")

    def __init__(self, precision: int = 12) -> None:
        if not isinstance(precision, int):
            raise TypeError("precision must be an int")
        if precision < 4 or precision > 16:
            raise ValueError("precision must satisfy 4 <= precision <= 16")
        self._precision = precision
        self._m = 1 << precision
        # Registers stored as raw integers (the max leading-zero rank seen).
        self._registers: List[int] = [0] * self._m
        # Width of the remaining hash bits used for rank computation.
        self._w_bits = 64 - precision

    @property
    def precision(self) -> int:
        return self._precision

    @property
    def register_count(self) -> int:
        return self._m

    def add(self, element: bytes) -> None:
        """Add ``element`` to the sketch.

        Raises ``TypeError`` if ``element`` is not ``bytes``. We deliberately
        do not auto-encode ``str`` so that callers make the serialization
        choice explicit (see module docstring).
        """
        if not isinstance(element, (bytes, bytearray)):
            raise TypeError("element must be bytes or bytearray")
        h = _hash64(bytes(element))
        # Top ``precision`` bits select the register.
        index = h >> self._w_bits
        # Remaining bits are used to compute the rank (leading zeros).
        rest = h & ((1 << self._w_bits) - 1)
        rank = _rho(rest, self._w_bits)
        if rank > self._registers[index]:
            self._registers[index] = rank

    def update(self, elements: Iterable[bytes]) -> None:
        """Add multiple elements. Equivalent to calling ``add`` in a loop."""
        for e in elements:
            self.add(e)

    def estimate(self) -> float:
        """Return the current approximate distinct count.

        Applies the standard small-range bias correction when the raw
        estimate is below ``2.5 * m`` and at least one register is zero. The
        large-range correction is intentionally omitted (see module docs).
        """
        m = self._m
        # Sum of 2^{-register[i]} using exp2 for numerical stability;
        # math.exp2 is standard-library and avoids manual bit twiddling.
        indicator = 0.0
        zeros = 0
        for r in self._registers:
            indicator += math.exp2(-float(r))
            if r == 0:
                zeros += 1
        raw = _alpha(m) * (float(m) ** 2) / indicator
        if raw <= 2.5 * float(m) and zeros != 0:
            # Small-range correction: linear counting on the number of
            # empty registers. This is the only correction we implement.
            return float(m) * math.log(float(m) / float(zeros))
        return raw

    def merge(self, other: "HyperLogLog") -> None:
        """Merge another HyperLogLog into this one in place.

        Both sketches must have the same precision; merging across precisions
        is not supported because the register semantics differ.
        """
        if not isinstance(other, HyperLogLog):
            raise TypeError("can only merge with another HyperLogLog")
        if other._precision != self._precision:
            raise ValueError("cannot merge HyperLogLog instances of different precision")
        for i, r in enumerate(other._registers):
            if r > self._registers[i]:
                self._registers[i] = r

    def snapshot(self) -> Tuple[int, Tuple[int, ...]]:
        """Return an immutable copy of the sketch state.

        Useful for checkpointing or for constructing a duplicate via
        ``restore``. Returns ``(precision, registers_tuple)``.
        """
        return (self._precision, tuple(self._registers))

    @classmethod
    def restore(cls, snapshot: Tuple[int, Tuple[int, ...]]) -> "HyperLogLog":
        """Reconstruct a HyperLogLog from a ``snapshot()`` value."""
        precision, registers = snapshot
        obj = cls(precision=precision)
        if len(registers) != obj._m:
            raise ValueError("snapshot register count does not match precision")
        obj._registers = list(registers)
        return obj

    def __len__(self) -> int:
        """Number of registers (not the distinct count)."""
        return self._m

    def __iter__(self) -> Iterator[int]:
        return iter(self._registers)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HyperLogLog):
            return NotImplemented
        return (
            self._precision == other._precision
            and self._registers == other._registers
        )

    def __repr__(self) -> str:
        return f"HyperLogLog(precision={self._precision})"


def estimate_bias_corrected(sketch: HyperLogLog) -> float:
    """Convenience wrapper returning ``sketch.estimate()``.

    WHY a separate function: some callers prefer a free function over a
    method when composing pipelines. The name makes the one correction we
    actually perform (small-range) explicit, rather than implying we apply
    the full battery of Flajolet et al. corrections.
    """
    if not isinstance(sketch, HyperLogLog):
        raise TypeError("estimate_bias_corrected requires a HyperLogLog")
    return sketch.estimate()
