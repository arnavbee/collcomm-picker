"""Which all-reduce should a tensor-parallel decode step use, and when does offload pay?

A cost model, not a measurement. Every number in PARAMS is an assumption you can
change. Standard latency-bandwidth (alpha-beta) model, pure stdlib.

    python3 picker.py
"""
import math

# ---- assumptions (edit these, then re-run) ---------------------------------
PARAMS = dict(
    gpus_per_node=8,
    nvlink_alpha=3e-6,      # s, per step inside a node
    nvlink_bw=400e9,        # B/s usable per GPU
    nic_alpha=8e-6,         # s, per step across nodes
    nic_bw=50e9,            # B/s per GPU rail (one 400 Gb/s NIC each)
    offload_eff=0.8,        # fraction of NIC line rate an in-network reduction sustains
    hbm_bw=3.35e12,         # B/s, decode is memory-bound
    model_bytes=140e9,      # 70B params in fp16
    hidden=8192, layers=80, dtype_bytes=2,
)
P = PARAMS


def ring(n, m, a, bw):
    return 2 * (n - 1) * a + 2 * (n - 1) / n * m / bw


def tree(n, m, a, bw):
    if n <= 1:
        return 0.0          # nothing to reduce; the 2m/bw term is per hop, not per rank
    return 2 * math.ceil(math.log2(n)) * a + 2 * m / bw


def offload(n, m, a, bw):
    # reduce up and broadcast down inside the fabric: each rank sends m once, gets m once
    return 2 * a + m / (bw * P["offload_eff"])


def inside_node(m, n):
    """Best software all-reduce among n GPUs on NVLink."""
    opts = {"ring": ring(n, m, P["nvlink_alpha"], P["nvlink_bw"]),
            "tree": tree(n, m, P["nvlink_alpha"], P["nvlink_bw"])}
    k = min(opts, key=opts.get)
    return k, opts[k]


def across_nodes(m, n_gpus, allow_offload):
    """Hierarchical: reduce-scatter in node, all-reduce across nodes on m/8, all-gather in node."""
    g = P["gpus_per_node"]
    nodes = n_gpus // g
    rs = (g - 1) * P["nvlink_alpha"] + (g - 1) / g * m / P["nvlink_bw"]   # same cost for the all-gather
    shard = m / g
    opts = {"ring": ring(nodes, shard, P["nic_alpha"], P["nic_bw"]),
            "tree": tree(nodes, shard, P["nic_alpha"], P["nic_bw"])}
    if allow_offload:
        opts["offload"] = offload(nodes, shard, P["nic_alpha"], P["nic_bw"])
    k = min(opts, key=opts.get)
    return k, 2 * rs + opts[k]


def allreduce(m, n_gpus, allow_offload=True):
    g = P["gpus_per_node"]
    if n_gpus < 1:
        raise ValueError(f"n_gpus must be at least 1, got {n_gpus}")
    if n_gpus > g and n_gpus % g:
        # across_nodes shards by g and counts nodes as n_gpus // g, so a partial
        # node would be priced as free: 20 GPUs came out identical to 16, and 12
        # came out with no inter-node term at all. Refuse instead of lying.
        raise ValueError(
            f"n_gpus={n_gpus} is not a whole number of {g}-GPU nodes; "
            "the hierarchical model only prices full nodes")
    if n_gpus <= g:
        return inside_node(m, n_gpus)
    return across_nodes(m, n_gpus, allow_offload)


def decode_step(batch, n_gpus, allow_offload=True):
    m = batch * P["hidden"] * P["dtype_bytes"]                 # one activation all-reduce
    algo, t = allreduce(m, n_gpus, allow_offload)
    comm = 2 * P["layers"] * t                                 # two per layer
    compute = P["model_bytes"] / n_gpus / P["hbm_bw"]
    return m, algo, comm, compute


def main():
    print("Tensor-parallel decode, 70B-class model, all numbers from PARAMS (assumed)\n")
    print(f"{'gpus':>4} {'batch':>5} {'msg':>8}  {'pick':<8} {'comm ms':>8} {'compute ms':>10} {'comm share':>10}   offload saves")
    for n in (8, 16, 32):
        for b in (1, 8, 32, 128, 512):
            m, algo, comm, comp = decode_step(b, n)
            _, _, comm_sw, _ = decode_step(b, n, allow_offload=False)
            save = f"{(1 - comm / comm_sw) * 100:4.0f}%" if algo == "offload" else "   -"
            print(f"{n:>4} {b:>5} {m / 1024:>6.0f}KB  {algo:<8} {comm * 1e3:>8.2f} {comp * 1e3:>10.2f} "
                  f"{comm / (comm + comp) * 100:>9.0f}%   {save}")
        print()

    print("Placement rule: is one 16-GPU TP group better than two 8-GPU groups?")
    print("(same GPUs; the 16-way group halves compute per GPU but pays the NIC on every layer)")
    for b in (1, 32, 128):
        _, _, c8, p8 = decode_step(b, 8)
        _, _, c16, p16 = decode_step(b, 16)
        print(f"  batch {b:>3}: TP8 step {(c8 + p8) * 1e3:6.2f} ms | TP16 step {(c16 + p16) * 1e3:6.2f} ms"
              f" -> {'TP16 wins' if c16 + p16 < c8 + p8 else 'keep TP inside the node'}")


if __name__ == "__main__":
    main()
