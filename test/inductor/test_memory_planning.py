# Owner(s): ["module: inductor"]
from unittest.mock import MagicMock

import sympy

import torch
from torch._inductor.codegen.memory_planning import (
    Allocation,
    LiveRange,
    LiveRanges,
    MemoryPlanner,
)
from torch._inductor.test_case import run_tests, TestCase


def _mock_buffer(name, device="cuda", dtype=torch.float32, shape=(4, 4)):
    """Create a mock buffer-like object for testing Allocation."""
    buf = MagicMock()
    buf.get_name.return_value = name
    buf.get_device.return_value = torch.device(device)
    buf.get_dtype.return_value = dtype
    buf.get_size.return_value = shape
    buf.get_stride.return_value = tuple(range(1, len(shape) + 1))  # dummy strides
    layout = MagicMock()
    # Use plain integers for size so free_unbacked_symbols returns false
    layout.size = shape
    buf.get_layout.return_value = layout
    return buf


def _make_allocation(name, begin, end, size_hint, device="cuda"):
    """Create an Allocation object with a mock buffer."""
    node = _mock_buffer(name, device=device)
    return Allocation(
        node=node,
        live_range=LiveRange(float(begin), float(end)),
        size_hint=size_hint,
        symbolic_size=sympy.Integer(size_hint),
    )


class TestIntervalGraphColoring(TestCase):
    """Tests for the interval graph coloring algorithm in MemoryPlanner."""

    def _get_coloring(self, allocations):
        """Run coloring and return color groups."""
        planner = MemoryPlanner(wrapper=MagicMock())
        return planner.assign_reuse_candidates(allocations)

    def _group_sizes(self, color_groups):
        """Return the number of allocations in each color group."""
        return [len(g) for g in color_groups]

    def _num_colors(self, color_groups):
        """Return the number of distinct colors used."""
        return len(color_groups)

    def test_single_buffer(self):
        """A single buffer should use exactly one color."""
        allocs = [_make_allocation("a", 0, 1, 100)]
        groups = self._get_coloring(allocs)
        self.assertEqual(self._num_colors(groups), 1)
        self.assertEqual(len(groups[0]), 1)

    def test_no_overlaps(self):
        """Non-overlapping buffers should all get the same color."""
        allocs = [
            _make_allocation("a", 0, 1, 100),
            _make_allocation("b", 1, 2, 100),
            _make_allocation("c", 2, 3, 100),
        ]
        groups = self._get_coloring(allocs)
        # All non-overlapping, so they can share one color
        self.assertEqual(self._num_colors(groups), 1)
        self.assertEqual(len(groups[0]), 3)

    def test_all_overlapping(self):
        """All buffers overlap, so each needs its own color."""
        allocs = [
            _make_allocation("a", 0, 5, 100),
            _make_allocation("b", 1, 6, 200),
            _make_allocation("c", 2, 7, 300),
        ]
        groups = self._get_coloring(allocs)
        self.assertEqual(self._num_colors(groups), 3)
        # Each group should have exactly 1 allocation
        self.assertTrue(all(len(g) == 1 for g in groups))

    def test_overlapping_identical_lifetimes(self):
        """Buffers with identical lifetimes all overlap and need separate colors."""
        allocs = [
            _make_allocation("a", 0, 10, 50),
            _make_allocation("b", 0, 10, 60),
            _make_allocation("c", 0, 10, 70),
        ]
        groups = self._get_coloring(allocs)
        self.assertEqual(self._num_colors(groups), 3)

    def test_partial_overlap_chain(self):
        """Chain of overlapping intervals: [0,2], [1,3], [2,4] -> max overlap is 2."""
        allocs = [
            _make_allocation("a", 0, 2, 100),
            _make_allocation("b", 1, 3, 100),
            _make_allocation("c", 2, 4, 100),
        ]
        groups = self._get_coloring(allocs)
        # Max overlap at time 1-2 is 2 (a and b), so need 2 colors
        self.assertEqual(self._num_colors(groups), 2)
        # a and c don't overlap, should be in same group
        total_allocs = sum(len(g) for g in groups)
        self.assertEqual(total_allocs, 3)

    def test_staircase_overlap(self):
        """Staircase pattern: [0,3], [1,4], [2,5] -> max overlap is 2 at [2,3]."""
        allocs = [
            _make_allocation("a", 0, 3, 100),
            _make_allocation("b", 1, 4, 200),
            _make_allocation("c", 2, 5, 300),
        ]
        groups = self._get_coloring(allocs)
        # max overlap is 2 (a and b at time 2, b and c at time 3)
        # [0,3] and [2,5] don't overlap, so they can share a color
        # With 2 colors: color0={a, c}, color1={b}
        # But greedy by start time gives: a=color0, b=color1, then c: a freed at 3 >= c.start=2? No! a.end=3, c.begin=2, so 3 > 2, meaning a is NOT freed.
        # Actually a.end=3 > c.begin=2, so a is still active when c starts
        # b is active too. Both a and b are still active at time 2.
        # So c needs a new color. That gives 3 colors.
        # Wait, but max overlap is 2. Greedy coloring by start time should give optimal = 2.
        # Let me trace through:
        # Sort: a(0,3), b(1,4), c(2,5)
        # Process a: no active colors, assign color0. active=[(3,0)]
        # Process b: active=[(3,0)], 1 <= 3 so nothing freed. available={}. assign color1. active=[(3,0),(4,1)]
        # Process c: active=[(3,0),(4,1)], 2 <= 3 so nothing freed. available={}. assign color2.
        # Hmm, that gives 3 colors. But the optimal is 2 for interval graphs.
        # Actually, a=[0,3], c=[2,5] DO overlap! They overlap at [2,3].
        # So max overlap is actually 3 at time [2,3]: a(0,3), b(1,4), c(2,5) all overlap there.
        # OK so 3 colors is correct.
        self.assertEqual(self._num_colors(groups), 3)

    def test_nested_intervals(self):
        """Nested intervals: [0,5], [1,4], [2,3] -> all overlap, need 3 colors."""
        allocs = [
            _make_allocation("outer", 0, 5, 100),
            _make_allocation("middle", 1, 4, 200),
            _make_allocation("inner", 2, 3, 300),
        ]
        groups = self._get_coloring(allocs)
        self.assertEqual(self._num_colors(groups), 3)

    def test_zero_sized_buffer(self):
        """Zero-sized buffers should be handled correctly."""
        allocs = [
            _make_allocation("normal", 0, 5, 100),
            _make_allocation("zero", 2, 3, 0),
        ]
        groups = self._get_coloring(allocs)
        # They overlap, so need 2 colors
        self.assertEqual(self._num_colors(groups), 2)

    def test_optimal_coloring_max_overlap(self):
        """
        Verify that the number of colors equals the maximum number of
        simultaneously live intervals (the chromatic number of an interval
        graph equals its maximum clique size).
        """
        # Create intervals where max overlap is 4
        # Pattern: 4 sets of intervals that all overlap at time 50-51
        allocs = []
        for i in range(4):
            allocs.append(_make_allocation(f"overlap_{i}", 0, 100, 100))
        # Add non-overlapping intervals that should reuse colors
        for i in range(4):
            allocs.append(_make_allocation(f"after_{i}", 101, 200, 100))

        groups = self._get_coloring(allocs)
        # Optimal coloring should use exactly 4 colors
        self.assertEqual(self._num_colors(groups), 4)
        # Each color should have exactly 2 allocations (one from overlap set,
        # one from after set)
        for group in groups:
            self.assertEqual(len(group), 2)

    def test_size_based_preference(self):
        """Verify that the algorithm prefers reusing colors with similar sizes."""
        # Scenario: two buffers overlap at time 0-1 (different colors),
        # then a third buffer starts at time 1 when both are freed.
        allocs = [
            _make_allocation("large1", 0, 1, 1000),
            _make_allocation("small1", 0, 1, 10),  # overlaps large1, gets own color
            _make_allocation(
                "small2", 1, 2, 10
            ),  # both colors available, prefer small1's
        ]
        groups = self._get_coloring(allocs)
        # large1 and small1 overlap, so they get different colors.
        # small2 has both colors available; should pick small1's (size 10 matches).
        self.assertEqual(self._num_colors(groups), 2)
        # small1 and small2 should be in the same group (same color)
        small_group = [g for g in groups if len(g) == 2][0]
        sizes = {a.size_hint for a in small_group}
        self.assertEqual(sizes, {10})

    def test_large_number_of_buffers(self):
        """Stress test with many buffers to ensure performance and correctness."""
        n = 200
        allocs = []
        for i in range(n):
            # Create pattern where max overlap grows, then shrinks
            begin = float(i // 10)
            end = float(begin + 5)
            allocs.append(_make_allocation(f"buf_{i}", begin, end, 100 + i % 10))

        groups = self._get_coloring(allocs)
        # Verify no two allocations in the same group have overlapping lifetimes
        for group in groups:
            for i, a in enumerate(group):
                for j, b in enumerate(group):
                    if i < j:
                        self.assertFalse(
                            a.live_range.begin < b.live_range.end
                            and b.live_range.begin < a.live_range.end,
                            f"Overlapping allocations {a} and {b} in same color group",
                        )

    def test_empty_input(self):
        """Empty allocation list should return empty groups."""
        groups = self._get_coloring([])
        self.assertEqual(groups, [])

    def test_multiple_devices(self):
        """Allocations on different devices should be colored independently."""
        # This test verifies that the coloring algorithm itself works with
        # allocations from different devices (though device partitioning
        # happens in _allocate_groups_coloring)
        allocs = [
            _make_allocation("cuda_a", 0, 1, 100, device="cuda"),
            _make_allocation("cpu_a", 0, 1, 100, device="cpu"),
            _make_allocation("cuda_b", 1, 2, 100, device="cuda"),
        ]
        groups = self._get_coloring(allocs)
        # cuda_a and cpu_a overlap, cuda_b is after cuda_a
        # Since device isn't considered in coloring, we just verify
        # no two allocations in same group overlap
        for group in groups:
            for i, a in enumerate(group):
                for j, b in enumerate(group):
                    if i < j:
                        self.assertFalse(
                            a.live_range.begin < b.live_range.end
                            and b.live_range.begin < a.live_range.end,
                        )

    def test_sympy_sizes(self):
        """Test that allocations with symbolic sizes work correctly."""
        node = _mock_buffer("sym_buf")
        alloc = Allocation(
            node=node,
            live_range=LiveRange(0.0, 1.0),
            size_hint=64,
            symbolic_size=sympy.Symbol("s", integer=True),
        )
        allocs = [
            alloc,
            _make_allocation("plain", 1.0, 2.0, 64),
        ]
        groups = self._get_coloring(allocs)
        # Non-overlapping, should share a color
        self.assertEqual(self._num_colors(groups), 1)
        self.assertEqual(len(groups[0]), 2)


class TestLiveRange(TestCase):
    """Tests for LiveRange and LiveRanges utility classes."""

    def test_contains(self):
        outer = LiveRange(0, 10)
        inner = LiveRange(2, 8)
        self.assertTrue(outer.contains(inner))
        self.assertFalse(inner.contains(outer))

    def test_contains_edge(self):
        r = LiveRange(0, 5)
        self.assertTrue(r.contains(LiveRange(0, 5)))
        self.assertTrue(r.contains(LiveRange(1, 4)))
        self.assertFalse(r.contains(LiveRange(0, 6)))
        self.assertFalse(r.contains(LiveRange(-1, 5)))

    def test_join(self):
        a = LiveRange(0, 5)
        b = LiveRange(3, 10)
        joined = a.join(b)
        self.assertEqual(joined.begin, 0)
        self.assertEqual(joined.end, 10)

    def test_len(self):
        r = LiveRange(2, 7)
        self.assertEqual(len(r), 5)

    def test_live_ranges_overlap(self):
        a = LiveRanges([LiveRange(0, 5)])
        b = LiveRanges([LiveRange(3, 10)])
        self.assertTrue(a.overlaps(b))

    def test_live_ranges_no_overlap(self):
        a = LiveRanges([LiveRange(0, 2)])
        b = LiveRanges([LiveRange(3, 10)])
        self.assertFalse(a.overlaps(b))

    def test_live_ranges_adjacent(self):
        """Adjacent ranges (end of one == begin of other) should NOT overlap."""
        a = LiveRanges([LiveRange(0, 3)])
        b = LiveRanges([LiveRange(3, 6)])
        self.assertFalse(a.overlaps(b))

    def test_live_ranges_discontiguous(self):
        """Test LiveRanges with multiple non-contiguous sub-ranges."""
        r = LiveRanges([LiveRange(0, 2), LiveRange(5, 7)])
        self.assertEqual(r.begin, 0)
        self.assertEqual(r.end, 7)
        self.assertTrue(r.overlaps(LiveRanges([LiveRange(1, 3)])))
        self.assertTrue(r.overlaps(LiveRanges([LiveRange(6, 8)])))
        self.assertFalse(r.overlaps(LiveRanges([LiveRange(3, 4)])))


if __name__ == "__main__":
    run_tests()
