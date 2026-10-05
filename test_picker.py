"""Invariants of the cost model, and the three claims the README makes.

Stdlib only, like picker.py.

    python3 -m unittest -v test_picker
"""
import unittest

import picker
from picker import P, across_nodes, allreduce, decode_step, inside_node, offload, ring, tree

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


class ReadmeClaims(unittest.TestCase):
    def test_tree_wins_inside_a_node_at_every_decode_size(self):
        for batch in (1, 8, 32, 128, 512):
            _, algo, _, _ = decode_step(batch, 8)
            self.assertEqual(algo, "tree", batch)

    def test_offload_pays_from_four_nodes_not_two(self):
        self.assertNotEqual(decode_step(32, 16)[1], "offload")
        self.assertEqual(decode_step(32, 32)[1], "offload")

    def test_comm_dominates_across_nodes(self):
        for batch in (1, 32, 512):
            _, _, comm, compute = decode_step(batch, 16)
            self.assertGreater(comm / (comm + compute), 0.75, batch)

    def test_two_tp8_groups_beat_one_tp16_group(self):
        for batch in (1, 32, 128):
            tp8 = sum(decode_step(batch, 8)[2:])
            tp16 = sum(decode_step(batch, 16)[2:])
            self.assertLess(tp8, tp16, batch)

    def test_placement_rule_survives_the_sweep_except_the_fastest_fabric(self):
        # README: the rule was checked with NIC latency 4-16 us, NIC bandwidth
        # 25-100 GB/s, NVLink latency 1-10 us, offload efficiency 0.3-0.8.
        # Over the full grid it holds in 234 of 243 cells. The nine that flip are
        # all the same corner: the fastest NIC (4 us) with the fastest NVLink
        # (1 us) at batch 1, where TP16 wins by under 1% -- a tie, not a reversal.
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
                                    flipped.append((nic_alpha, nvlink_alpha, batch,
                                                    (tp8 - tp16) / tp8))
        finally:
            P.clear()
            P.update(base)

        self.assertEqual(len(flipped), 9, flipped)
        for nic_alpha, nvlink_alpha, batch, margin in flipped:
            self.assertEqual((nic_alpha, nvlink_alpha, batch), (4e-6, 1e-6, 1))
            self.assertLess(margin, 0.01, margin)


if __name__ == "__main__":
    unittest.main()
