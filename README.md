# Collective picker: which all-reduce, and when does offload pay?

A small cost model for the all-reduce that tensor-parallel LLM decode pays twice per layer.
It picks the algorithm (ring, tree, recursive halving, in-network offload) for each batch size
and GPU count, and says whether a 16-GPU tensor-parallel group is worth it or the group should
stay inside one node. Under the default assumptions that answer turns out to be a statement
about the NIC's latency rather than a rule.

This is a model. It is not a measurement. I had no GPU cluster, so every number comes from
the assumptions at the top of `picker.py` (latency and bandwidth of NVLink and a 400 Gb/s NIC,
70B-class model, memory-bound decode). Change them and re-run. Standard library only.

    python3 picker.py
    python3 picker.py --set nic_alpha=4e-6 --set offload_eff=0.3
    python3 -m unittest test_picker

`--set key=value` overrides one assumption without editing the file, and the header of the table
says which numbers are not the defaults. Only keys that already exist in `PARAMS` are accepted,
so a typo stops the run instead of printing a default table that looks swept. `nan` and `inf`
are refused too: `nan` slipped past the positivity check (`nan <= 0` is False), so `--set
nic_bw=nan` used to print `nan` ms and still name "ring" as the pick, because `min()` over
NaN comparisons keeps whichever option it saw first. `offload_eff` is a
fraction of line rate and is refused above 1, because `offload_eff=8` for `0.8` is a fabric eight
times faster than its own NIC and the table then says offload wins everywhere. The swept GPU
counts follow `gpus_per_node` rather than the literals 8/16/32, so `--set gpus_per_node=6` prints
6/12/24 instead of dying half a table in on a GPU count that is not a whole number of nodes.

`test_picker.py` checks the formulas (one rank is free, cost grows with message size, the pick is
the cheapest option) and re-derives every claim below, so the README cannot drift from the model.

The hierarchical path only prices whole nodes, so `allreduce` now refuses a GPU count that is not
a multiple of `gpus_per_node`. It used to shard by node and floor the node count, which made 20
GPUs cost exactly the same as 16 and gave 12 GPUs no inter-node term at all.

Its two intra-node phases are priced with the cheaper of ring and **recursive halving**, not ring
alone. Both move the same `m(n-1)/n` bytes per rank, so halving pays `log2(n)` latency terms where
ring pays `n-1`: 3 instead of 7 on an 8-GPU node, which is 12 microseconds per phase and 3.8 ms
per token over 160 collectives. Hardcoding ring there was the one place the model still preferred
ring on latency while `inside_node` was picking tree for exactly the opposite reason, and it was
paying for the placement rule below: with the phases priced properly the rule stops being general.
Halving needs a power-of-two rank count, so a 6-GPU node is still offered ring only.

## What it says under the default assumptions

- **Inside one node (8 GPUs) tree beats ring at every decode message size.** Messages are 16 KB
  to 8 MB. At that size the step count matters more than bandwidth.
- **Across nodes, communication is 68-92% of the decode step** (68-85% on two nodes, 81-92% on
  four). The time is latency, not bandwidth. Each collective has a fixed floor of tens of
  microseconds, and there are 160 collectives per token.
- **In-network offload pays from 4 nodes (32 GPUs) up**, saving 25-32% of decode communication.
  At 2 nodes it does not beat ring.
- **"Keep the tensor-parallel group inside a node" is the NIC's latency, not a law.** With the
  same 16 GPUs and the default 8 microsecond NIC, two TP8 groups beat one TP16 group at batch 32
  (by 1.2%) and batch 128 (by 5.9%), while batch 1 is a tie (TP16 comes out 0.6% ahead). Sweeping all four
  fabric assumptions together (NIC latency 4 to 16 microseconds, NIC bandwidth 25 to 100 GB/s,
  NVLink latency 1 to 10 microseconds, offload efficiency 0.3 to 0.8) the rule holds in 144 of
  the 243 cells, and which 99 flip is decided by the NIC latency alone: at 16 microseconds it
  holds everywhere, at 8 it flips only at batch 1 and by under 1%, and at 4 microseconds one
  TP16 group wins outright, by up to 22%. So the rule is worth stating as "keep TP inside the
  node unless the inter-node latency is within a few microseconds of NVLink's", which is the
  regime a tightly coupled fabric is sold on. `test_picker.py` pins all three regimes.
  It agrees in direction with
  my earlier Vidur sweep of TP against PP on 8 GPUs
  ([tp-pp-crossover](https://github.com/arnavbee/tp-pp-crossover)), where changing the
  interconnect moved the crossover by one step.

## What I do not know

- The alpha values are guesses. Real small-message all-reduce inside a node is usually slower
  than 3 microseconds, which would make communication share larger, not smaller.
- The offload model is idealised: each rank sends once and receives once, at 80% of line rate.
  Real switch or NIC offload has limits on aggregation rate, tenant sharing and precision that
  this does not model.
- Pipeline parallelism, overlap of compute with communication, and MoE all-to-all are left out.
  ([moe-straggler](https://github.com/arnavbee/moe-straggler) measures the MoE all-to-all side.)

## What I would do next, and would like to do in a lab

Run `nccl-tests` on a real multi-node GPU cluster with small messages (16 KB to 8 MB) and
replace the assumed alpha and bandwidth values with measured ones. Then check whether the
tree, ring and offload crossovers above still fall where the model puts them. That is the
starter task I would take on.
