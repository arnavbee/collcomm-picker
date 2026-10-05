# Collective picker: which all-reduce, and when does offload pay?

A small cost model for the all-reduce that tensor-parallel LLM decode pays twice per layer.
It picks the algorithm (ring, tree, in-network offload) for each batch size and GPU count,
and says whether a 16-GPU tensor-parallel group is worth it or the group should stay inside
one node.

This is a model. It is not a measurement. I had no GPU cluster, so every number comes from
the assumptions at the top of `picker.py` (latency and bandwidth of NVLink and a 400 Gb/s NIC,
70B-class model, memory-bound decode). Change them and re-run. Standard library only.

    python3 picker.py
    python3 -m unittest test_picker

`test_picker.py` checks the formulas (one rank is free, cost grows with message size, the pick is
the cheapest option) and re-derives every claim below, so the README cannot drift from the model.

The hierarchical path only prices whole nodes, so `allreduce` now refuses a GPU count that is not
a multiple of `gpus_per_node`. It used to shard by node and floor the node count, which made 20
GPUs cost exactly the same as 16 and gave 12 GPUs no inter-node term at all.

## What it says under the default assumptions

- **Inside one node (8 GPUs) tree beats ring at every decode message size.** Messages are 16 KB
  to 8 MB. At that size the step count matters more than bandwidth.
- **Across nodes, communication is 78-94% of the decode step.** The time is latency, not
  bandwidth. Two intra-node phases and the inter-node phase add up to a fixed floor of tens
  of microseconds per collective, and there are 160 collectives per token.
- **In-network offload pays from 4 nodes (32 GPUs) up.** It cuts the number of steps, so it
  saves roughly 10-33% of decode communication depending on the latency assumptions. At 2 nodes
  it does not beat ring.
- **Placement rule: keep the tensor-parallel group inside a node.** With the same 16 GPUs, two
  TP8 groups beat one TP16 group at batch 1, 32 and 128. Sweeping all four fabric
  assumptions together (NIC latency 4 to 16 microseconds, NIC bandwidth 25 to 100 GB/s, NVLink
  latency 1 to 10 microseconds, offload efficiency 0.3 to 0.8) the rule holds in 234 of the 243
  cells. The nine that flip are one corner: the fastest NIC and the fastest NVLink at batch 1,
  where TP16 wins by under 1%, which is a tie inside the model's error rather than a reversal.
  `test_picker.py` pins that corner, so a later change to the assumptions cannot quietly widen
  it. It agrees in direction with
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
