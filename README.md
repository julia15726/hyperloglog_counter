# Hyperloglog Counter

A small, dependency-free Python library that approximates the number of distinct elements in a large data stream using the HyperLogLog algorithm.

```python
from hyperloglog_counter import HyperLogLog

hll = HyperLogLog(precision=12)  # 4096 registers, ~1.6% standard error
for i in range(100_000):
    hll.add(f"user-{i}".encode())

print(hll.estimate())  # ~100000
```

`estimate_bias_corrected(hll)` is a free-function alias for `hll.estimate()` for callers who prefer a functional style.

## Why this exists

Counting distinct values exactly in a streaming setting requires memory proportional to the number of uniques seen. HyperLogLog trades a known, tunable error for constant memory: `2**precision` registers, each an 8-byte integer. At `precision=12` that is 32 KB for roughly 1.6% standard error regardless of how many uniques flow through.

The trade-off chosen here: determinism and portability over peak accuracy. Hashing uses SHA-256 truncated to 64 bits via `hashlib`, not Python's built-in `hash()`, because `hash()` is salted per process for strings and is not stable across runs or platforms. This makes sketches reproducible and mergeable across processes, at the cost of being slower than a native hash.

## The awkward edge

Elements must be `bytes` (or `bytearray`). The library deliberately does not auto-encode `str`, because the encoding choice affects which collisions occur and should be explicit at the call site. If you pass a `str`, you get a `TypeError`, not a silent guess.

Precision must satisfy `4 <= precision <= 16`. Below 4 the bias correction is unreliable; above 16 the register array (`2**p` entries) becomes large enough that the "small memory" premise of the algorithm stops paying off.

Only the small-range bias correction is implemented (linear counting on empty registers when the raw estimate is below `2.5 * m`). The large-range correction for estimates near 2^64 is omitted because it is unreachable with real data and shipping a stub for it would misrepresent the implementation.

## Exported names

- `HyperLogLog` — the sketch class.
  - `HyperLogLog(precision: int = 12)`
  - `add(element: bytes) -> None`
  - `update(elements: Iterable[bytes]) -> None`
  - `estimate() -> float`
  - `merge(other: HyperLogLog) -> None`
  - `snapshot() -> tuple[int, tuple[int, ...]]`
  - `HyperLogLog.restore(snapshot) -> HyperLogLog`
  - properties: `precision`, `register_count`
- `estimate_bias_corrected(sketch: HyperLogLog) -> float`

## Tests

```
PYTHONPATH=src python -m unittest discover -s tests
```
