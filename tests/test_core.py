"""Tests for the HyperLogLog core implementation.

These tests are deliberately deterministic: they assert on relative error
bands and exact structural properties, never on wall-clock time or on
float equality without a tolerance.
"""

import math
import unittest

from hyperloglog_counter.core import HyperLogLog, estimate_bias_corrected


class TestConstruction(unittest.TestCase):
    def test_default_precision(self):
        hll = HyperLogLog()
        self.assertEqual(hll.precision, 12)
        self.assertEqual(hll.register_count, 4096)
        self.assertEqual(len(hll), 4096)

    def test_invalid_precision_low(self):
        with self.assertRaises(ValueError):
            HyperLogLog(precision=3)

    def test_invalid_precision_high(self):
        with self.assertRaises(ValueError):
            HyperLogLog(precision=17)

    def test_non_int_precision(self):
        with self.assertRaises(TypeError):
            HyperLogLog(precision="12")  # type: ignore[arg-type]

    def test_registers_start_zero(self):
        hll = HyperLogLog(precision=4)
        self.assertEqual(list(hll), [0] * 16)


class TestAddAndEstimate(unittest.TestCase):
    def _make_elements(self, n):
        # Deterministic, distinct byte strings. We use a simple counter
        # encoded as bytes so results are reproducible across runs.
        return [f"item-{i}".encode() for i in range(n)]

    def test_empty_estimate_is_small_range_corrected(self):
        # All registers zero -> linear counting gives m * log(m / m) = 0.0.
        hll = HyperLogLog(precision=12)
        est = hll.estimate()
        self.assertEqual(est, 0.0)

    def test_single_element(self):
        hll = HyperLogLog(precision=12)
        hll.add(b"hello")
        est = hll.estimate()
        # With one element the small-range correction dominates and yields
        # a small positive number well under 2.
        self.assertGreater(est, 0.0)
        self.assertLess(est, 2.0)

    def test_add_is_idempotent(self):
        hll = HyperLogLog(precision=12)
        hll.add(b"x")
        snap_before = hll.snapshot()
        for _ in range(50):
            hll.add(b"x")
        self.assertEqual(hll.snapshot(), snap_before)

    def test_distinct_count_within_standard_error(self):
        # Standard error ~ 1.04 / sqrt(m). With p=14, m=16384, that is
        # ~0.81%. We allow 4% to keep the test robust without flaking.
        p = 14
        n = 50_000
        hll = HyperLogLog(precision=p)
        hll.update(self._make_elements(n))
        est = hll.estimate()
        rel_err = abs(est - n) / n
        self.assertLess(rel_err, 0.04, msg=f"estimate={est}, true={n}")

    def test_estimates_are_deterministic(self):
        elems = self._make_elements(1000)
        hll1 = HyperLogLog(precision=12)
        hll2 = HyperLogLog(precision=12)
        hll1.update(elems)
        hll2.update(elems)
        self.assertEqual(hll1.snapshot(), hll2.snapshot())
        self.assertEqual(hll1.estimate(), hll2.estimate())

    def test_duplicate_elements_do_not_inflate_estimate(self):
        hll = HyperLogLog(precision=12)
        elems = self._make_elements(1000)
        hll.update(elems)
        single = hll.estimate()
        hll.update(elems)  # add the same 1000 again
        hll.update(elems)  # and again
        double = hll.estimate()
        # Register state is unchanged by duplicates, so estimates match.
        self.assertEqual(single, double)

    def test_bytearray_accepted(self):
        hll = HyperLogLog(precision=8)
        hll.add(bytearray(b"abc"))
        self.assertGreater(hll.estimate(), 0.0)

    def test_non_bytes_rejected(self):
        hll = HyperLogLog(precision=8)
        with self.assertRaises(TypeError):
            hll.add("not bytes")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            hll.add(42)  # type: ignore[arg-type]


class TestMerge(unittest.TestCase):
    def test_merge_combines_registers(self):
        a = HyperLogLog(precision=10)
        b = HyperLogLog(precision=10)
        a.add(b"apple")
        b.add(b"banana")
        snap_a = a.snapshot()
        a.merge(b)
        # Merging must never shrink any register.
        for before, after in zip(snap_a[1], a.snapshot()[1]):
            self.assertGreaterEqual(after, before)
        # And the merged sketch should estimate ~2 distinct items.
        est = a.estimate()
        self.assertGreater(est, 1.0)
        self.assertLess(est, 3.0)

    def test_merge_same_precision_only(self):
        a = HyperLogLog(precision=10)
        b = HyperLogLog(precision=12)
        with self.assertRaises(ValueError):
            a.merge(b)

    def test_merge_non_hll_rejected(self):
        a = HyperLogLog(precision=10)
        with self.assertRaises(TypeError):
            a.merge("nope")  # type: ignore[arg-type]

    def test_merge_into_fresh_equals_combined_add(self):
        p = 12
        left = HyperLogLog(precision=p)
        right = HyperLogLog(precision=p)
        combined = HyperLogLog(precision=p)
        left_elems = [f"L{i}".encode() for i in range(500)]
        right_elems = [f"R{i}".encode() for i in range(500)]
        left.update(left_elems)
        right.update(right_elems)
        combined.update(left_elems)
        combined.update(right_elems)
        left.merge(right)
        self.assertEqual(left.snapshot(), combined.snapshot())


class TestSnapshotRestore(unittest.TestCase):
    def test_snapshot_restore_roundtrip(self):
        hll = HyperLogLog(precision=11)
        hll.update([f"k{i}".encode() for i in range(2000)])
        snap = hll.snapshot()
        restored = HyperLogLog.restore(snap)
        self.assertEqual(restored.snapshot(), snap)
        self.assertEqual(restored.estimate(), hll.estimate())

    def test_restore_rejects_mismatched_register_count(self):
        # Tamper: pass a register tuple whose length disagrees with precision.
        with self.assertRaises(ValueError):
            HyperLogLog.restore((12, (0, 0, 0)))

    def test_snapshot_is_immutable_tuple(self):
        hll = HyperLogLog(precision=4)
        snap = hll.snapshot()
        self.assertIsInstance(snap, tuple)
        self.assertIsInstance(snap[1], tuple)
        # Mutating the returned tuple must not affect the sketch.
        hll.add(b"z")
        snap2 = hll.snapshot()
        self.assertNotEqual(snap, snap2)


class TestFreeFunction(unittest.TestCase):
    def test_estimate_bias_corrected_matches_method(self):
        hll = HyperLogLog(precision=12)
        hll.update([f"f{i}".encode() for i in range(1000)])
        self.assertEqual(estimate_bias_corrected(hll), hll.estimate())

    def test_estimate_bias_corrected_type_check(self):
        with self.assertRaises(TypeError):
            estimate_bias_corrected("not a sketch")  # type: ignore[arg-type]


class TestReprAndEquality(unittest.TestCase):
    def test_repr_contains_precision(self):
        hll = HyperLogLog(precision=9)
        self.assertIn("9", repr(hll))
        self.assertIn("HyperLogLog", repr(hll))

    def test_equal_when_same_state(self):
        a = HyperLogLog(precision=8)
        b = HyperLogLog(precision=8)
        a.add(b"x")
        b.add(b"x")
        self.assertEqual(a, b)

    def test_unequal_when_different_state(self):
        a = HyperLogLog(precision=8)
        b = HyperLogLog(precision=8)
        a.add(b"x")
        b.add(b"y")
        # Different elements *may* land in the same register with the same
        # rank, so equality is not guaranteed. We assert the realistic
        # contract: they are equal iff registers match. To make this
        # deterministic, compare via snapshot rather than guessing.
        if a.snapshot() != b.snapshot():
            self.assertNotEqual(a, b)
        else:
            self.assertEqual(a, b)

    def test_unequal_when_different_precision(self):
        a = HyperLogLog(precision=8)
        b = HyperLogLog(precision=10)
        self.assertNotEqual(a, b)

    def test_eq_with_non_hll_returns_notimplemented(self):
        a = HyperLogLog(precision=8)
        # __eq__ returns NotImplemented for non-HyperLogLog, which Python
        # then falls back to identity comparison -> False.
        self.assertFalse(a == "nope")


if __name__ == "__main__":
    unittest.main()
