# Collective picker: which all-reduce, and when does offload pay?

A small cost model for the all-reduce that tensor-parallel LLM decode pays twice per layer.
It picks the algorithm (ring, tree, in-network offload) for each batch size and GPU count,
and says whether a 16-GPU tensor-parallel group is worth it or the group should stay inside
one node.

This is a model. It is not a measurement. I had no GPU cluster, so every number comes from
the assumptions at the top of `picker.py` (latency and bandwidth of NVLink and a 400 Gb/s NIC,
70B-class model, memory-bound decode). Change them and re-run. Standard library only.

    python3 picker.py

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
  TP8 groups beat one TP16 group at batch 1, 32 and 128. This held when I moved NIC latency
  between 4 and 16 microseconds, NIC bandwidth between 25 and 100 GB/s, NVLink latency between
  1 and 10 microseconds, and offload efficiency between 0.3 and 0.8. It agrees in direction with
  my earlier Vidur sweep of TP against PP on 8 GPUs, where changing the interconnect moved the
  crossover by one step.

## What I do not know

- The alpha values are guesses. Real small-message all-reduce inside a node is usually slower
  than 3 microseconds, which would make communication share larger, not smaller.
- The offload model is idealised: each rank sends once and receives once, at 80% of line rate.
  Real switch or NIC offload has limits on aggregation rate, tenant sharing and precision that
  this does not model.
- Pipeline parallelism, overlap of compute with communication, and MoE all-to-all are left out.
  (I measured the MoE all-to-all straggler separately.)

## What I would do next, and would like to do in a lab

Run `nccl-tests` on a real multi-node GPU cluster with small messages (16 KB to 8 MB) and
replace the assumed alpha and bandwidth values with measured ones. Then check whether the
tree, ring and offload crossovers above still fall where the model puts them. That is the
starter task I would take on.
