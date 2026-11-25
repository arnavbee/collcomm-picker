"""Invariants of the cost model, and the three claims the README makes.

Stdlib only, like picker.py.

    python3 -m unittest -v test_picker
"""
import contextlib
import io
import math
import unittest

import picker
from picker import (P, across_nodes, allreduce, apply_overrides, decode_step,
                    inside_node, offload, parse_override, reduce_scatter, ring,
                    tree)

MSG = 32 * P["hidden"] * P["dtype_bytes"]   # a batch-32 decode activation, 512 KB


class Formulas(unittest.TestCase):
    def test_one_rank_is_free(self):
        self.assertEqual(ring(1, MSG, 1e-6, 1e9), 0.0)
        self.assertEqual(tree(1, MSG, 1e-6, 1e9), 0.0)

    def test_cost_grows_with_message(self):
        for f in (ring, tree, offload):
            small = f(8, 1 << 14, 1e-6, 1e9)
            large = f(8, 1 << 23, 1e-6, 1e9)
            self.assertLess(small, large, f.__name__)

    def test_ring_pays_more_steps_than_tree(self):
        # 2(n-1) latency terms against 2*ceil(log2 n); same alpha, so ring loses
        # on latency and only wins back on bandwidth at large messages.
        a, bw = P["nvlink_alpha"], P["nvlink_bw"]
        self.assertGreater(ring(8, 1 << 14, a, bw), tree(8, 1 << 14, a, bw))

    def test_halving_reduce_scatter_moves_the_same_bytes_as_ring(self):
        # both move m(n-1)/n per rank; halving just pays log2(n) latency terms
        # instead of n-1, so the two differ by exactly the step count
        a, bw = 3e-6, 400e9
        for n in (2, 4, 8, 16):
            band = (n - 1) / n * MSG / bw
            self.assertAlmostEqual(ring(n, MSG, 0, bw) / 2, band)
            name, cost = reduce_scatter(n, MSG, a, bw)
            self.assertAlmostEqual(cost, math.log2(n) * a + band)
            # at 2 ranks the two are the same single exchange, so min() keeps
            # whichever key it saw first and the name says ring
            self.assertEqual(name, "ring" if n == 2 else "halving", n)

    def test_reduce_scatter_is_never_worse_than_ring(self):
        a, bw = 3e-6, 400e9
        for n in (2, 3, 4, 7, 8, 12, 16):
            ring_rs = (n - 1) * a + (n - 1) / n * MSG / bw
            self.assertLessEqual(reduce_scatter(n, MSG, a, bw)[1], ring_rs, n)

    def test_halving_needs_a_power_of_two(self):
        # recursive halving splits the buffer in two every step, so 6 ranks get
        # ring only rather than a cost the algorithm cannot actually reach
        for n in (3, 6, 12):
            self.assertEqual(reduce_scatter(n, MSG, 3e-6, 400e9)[0], "ring")
        self.assertEqual(reduce_scatter(1, MSG, 3e-6, 400e9), ("ring", 0.0))

    def test_offload_ignores_rank_count(self):
        # one reduce up, one broadcast down, inside the fabric
        self.assertEqual(offload(4, MSG, 1e-6, 1e9), offload(64, MSG, 1e-6, 1e9))


class Picking(unittest.TestCase):
    def test_pick_is_the_cheapest_option(self):
        algo, cost = inside_node(MSG, 8)
        both = (ring(8, MSG, P["nvlink_alpha"], P["nvlink_bw"]),
                tree(8, MSG, P["nvlink_alpha"], P["nvlink_bw"]))
        self.assertAlmostEqual(cost, min(both))
        self.assertEqual(algo, "tree")

    def test_offload_is_never_worse_than_software(self):
        for n in (16, 32, 64):
            _, with_off = across_nodes(MSG, n, allow_offload=True)
            _, without = across_nodes(MSG, n, allow_offload=False)
            self.assertLessEqual(with_off, without, n)

    def test_more_gpus_never_gets_cheaper_across_nodes(self):
        costs = [allreduce(MSG, n)[1] for n in (16, 32, 64)]
        self.assertEqual(costs, sorted(costs))


class PartialNodes(unittest.TestCase):
    """The bug this file was written for: a partial node used to be free."""

    def test_partial_node_is_rejected(self):
        for n in (9, 12, 20):
            with self.assertRaises(ValueError):
                allreduce(MSG, n)

    def test_full_nodes_and_sub_node_counts_are_fine(self):
        for n in (1, 2, 4, 8, 16, 32):
            self.assertGreaterEqual(allreduce(MSG, n)[1], 0.0)

    def test_zero_or_negative_is_rejected(self):
        for n in (0, -8):
            with self.assertRaises(ValueError):
                allreduce(MSG, n)


class Overrides(unittest.TestCase):
    """--set lets the README's "change them and re-run" happen without an edit."""

    def setUp(self):
        self.base = dict(P)
        self.addCleanup(lambda: (P.clear(), P.update(self.base)))

    def test_override_changes_the_model(self):
        before = decode_step(32, 32)[2]
        apply_overrides(["nic_alpha=16e-6"])
        self.assertEqual(P["nic_alpha"], 16e-6)
        self.assertGreater(decode_step(32, 32)[2], before)

    def test_counts_stay_integers(self):
        apply_overrides(["layers=32", "gpus_per_node=4"])
        for key in ("layers", "gpus_per_node"):
            self.assertIsInstance(P[key], int, key)

    def test_unknown_key_is_rejected(self):
        # a typo must not sit in the dict while the table prints the default
        with self.assertRaises(ValueError):
            apply_overrides(["nic_latency=4e-6"])
        self.assertNotIn("nic_latency", P)

    def test_bad_values_are_rejected(self):
        for bad in ("nic_alpha", "nic_alpha=fast", "nic_bw=0", "nvlink_alpha=-1e-6"):
            with self.assertRaises(ValueError):
                apply_overrides([bad])
        self.assertEqual(P, self.base)

    def test_a_rejected_set_changes_nothing_at_all(self):
        # the good pair used to land before the bad one raised, so
        # `--set layers=32 --set nic_bw=0` kept layers=32 in PARAMS
        for pairs in (["layers=32", "nic_bw=0"],
                      ["nic_alpha=4e-6", "offload_eff=8"],
                      ["hidden=4096", "nic_latency=1e-6"],
                      ["gpus_per_node=4", "nvlink_bw=oops"]):
            with self.assertRaises(ValueError):
                apply_overrides(pairs)
            self.assertEqual(P, self.base, pairs)

    def test_a_key_given_twice_keeps_the_last_value(self):
        changed = apply_overrides(["nic_bw=25e9", "nic_bw=100e9"])
        self.assertEqual(P["nic_bw"], 100e9)
        self.assertEqual(changed, ["nic_bw"])

    def test_parse_override_does_not_touch_params(self):
        self.assertEqual(parse_override(" layers = 32 "), ("layers", 32))
        self.assertEqual(P, self.base)

    def test_non_finite_is_rejected(self):
        # nan is not caught by "must be positive": nan <= 0 is False, so
        # --set nic_bw=nan reached the model, printed nan ms, and still named
        # "ring" as the pick because min() over nan comparisons keeps the first
        # key it saw. A fabricated pick is worse than a refusal.
        for bad in ("nic_bw=nan", "nic_alpha=inf", "nic_bw=-inf",
                    "offload_eff=nan", "hbm_bw=infinity"):
            with self.assertRaises(ValueError):
                apply_overrides([bad])
        self.assertEqual(P, self.base)

    def test_efficiency_above_line_rate_is_rejected(self):
        # offload_eff=8 (a slipped exponent for 0.8) divided the NIC bandwidth by
        # eight times line rate and printed offload winning everywhere.
        for bad in ("offload_eff=8", "offload_eff=1.01"):
            with self.assertRaises(ValueError):
                apply_overrides([bad])
        self.assertEqual(P, self.base)
        apply_overrides(["offload_eff=1"])
        self.assertEqual(P["offload_eff"], 1.0)

    def test_sweep_follows_gpus_per_node(self):
        # the table used to ask for 8 GPUs whatever the node size, so a node that
        # does not divide 8 died with a traceback mid-table
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            picker.main(["--set", "gpus_per_node=6"])
        text = out.getvalue()
        self.assertIn("one 12-GPU TP group better than two 6-GPU groups", text)
        for n in (6, 12, 24):
            self.assertRegex(text, rf"(?m)^\s*{n}\s+1\s")

    def test_main_accepts_set_and_leaves_defaults_alone(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            picker.main(["--set", "offload_eff=0.3"])
        self.assertIn("overridden: offload_eff=0.3", out.getvalue())
        self.assertEqual(P["offload_eff"], 0.3)
        self.assertEqual(picker.PARAMS["nic_alpha"], self.base["nic_alpha"])


class ReadmeClaims(unittest.TestCase):
    def test_tree_wins_inside_a_node_at_every_decode_size(self):
        for batch in (1, 8, 32, 128, 512):
            _, algo, _, _ = decode_step(batch, 8)
            self.assertEqual(algo, "tree", batch)

    def test_offload_pays_from_four_nodes_not_two(self):
        self.assertNotEqual(decode_step(32, 16)[1], "offload")
        self.assertEqual(decode_step(32, 32)[1], "offload")

    def test_comm_dominates_across_nodes(self):
        # 68-85% on two nodes, 81-92% on four; the two floors are 67.6% and 80.7%
        for n, floor in ((16, 0.67), (32, 0.80)):
            for batch in (1, 32, 512):
                _, _, comm, compute = decode_step(batch, n)
                self.assertGreater(comm / (comm + compute), floor, (n, batch))

    def test_two_tp8_groups_beat_one_tp16_group_from_batch_32_up(self):
        # batch 1 is a tie: TP16 comes out 0.6% ahead, which is inside the
        # model's error. From batch 32 the NIC term on every layer decides it.
        tie = [sum(decode_step(1, n)[2:]) for n in (8, 16)]
        self.assertLess(abs(tie[0] - tie[1]) / tie[0], 0.01, tie)
        for batch in (32, 128):
            tp8 = sum(decode_step(batch, 8)[2:])
            tp16 = sum(decode_step(batch, 16)[2:])
            self.assertLess(tp8, tp16, batch)

    def test_the_placement_rule_is_the_nics_latency_not_a_law(self):
        # README: the rule was checked with NIC latency 4-16 us, NIC bandwidth
        # 25-100 GB/s, NVLink latency 1-10 us, offload efficiency 0.3-0.8.
        # Over the full grid "keep TP inside the node" now holds in 144 of 243
        # cells, and which 99 flip is not noise -- it is the NIC latency alone:
        #
        #   16 us NIC: the rule holds in every cell, at every batch.
        #    8 us NIC: it flips at batch 1 only, by under 1% -- a tie.
        #    4 us NIC: TP16 wins outright, by up to 22%, at batch 1 and 32 and
        #              in two thirds of the batch-128 cells.
        #
        # Before the intra-node phases were priced with recursive halving the
        # grid flipped in 9 cells, all one corner, and the README called the
        # rule general. Eight ring latency terms per collective were paying for
        # that generality, so the rule was really a statement about the NIC.
        base = dict(P)
        flipped = []
        try:
            for nic_alpha in (4e-6, 8e-6, 16e-6):
                for nic_bw in (25e9, 50e9, 100e9):
                    for nvlink_alpha in (1e-6, 3e-6, 10e-6):
                        for eff in (0.3, 0.5, 0.8):
                            P.update(nic_alpha=nic_alpha, nic_bw=nic_bw,
                                     nvlink_alpha=nvlink_alpha, offload_eff=eff)
                            for batch in (1, 32, 128):
                                tp8 = sum(decode_step(batch, 8)[2:])
                                tp16 = sum(decode_step(batch, 16)[2:])
                                if tp16 < tp8:
                                    flipped.append((nic_alpha, batch,
                                                    (tp8 - tp16) / tp8))
        finally:
            P.clear()
            P.update(base)

        self.assertEqual(len(flipped), 99, len(flipped))
        self.assertEqual(sum(1 for f in flipped if f[0] == 16e-6), 0)

        slow = [f for f in flipped if f[0] == 8e-6]
        self.assertEqual(len(slow), 27)
        for _, batch, margin in slow:
            self.assertEqual(batch, 1)
            self.assertLess(margin, 0.01, margin)

        fast = [f for f in flipped if f[0] == 4e-6]
        self.assertEqual(len(fast), 72)
        self.assertEqual(sum(1 for f in fast if f[1] == 1), 27)
        self.assertEqual(sum(1 for f in fast if f[1] == 32), 27)
        self.assertEqual(sum(1 for f in fast if f[1] == 128), 18)
        self.assertGreater(max(f[2] for f in fast), 0.2)


if __name__ == "__main__":
    unittest.main()
