"""Which all-reduce should a tensor-parallel decode step use, and when does offload pay?

A cost model, not a measurement. Every number in PARAMS is an assumption you can
change. Standard latency-bandwidth (alpha-beta) model, pure stdlib.

    python3 picker.py
    python3 picker.py --set nic_alpha=4e-6 --set offload_eff=0.3
"""
import argparse
import math

# ---- assumptions (edit these, or override with --set key=value) -------------
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

# the counts have to stay whole; the rest are physical quantities
INT_PARAMS = ("gpus_per_node", "hidden", "layers", "dtype_bytes")
# fractions of a rate: a value above 1 would be a fabric beating its own line rate
FRACTION_PARAMS = ("offload_eff",)


def apply_overrides(pairs):
    """Apply "key=value" strings to P. Returns the keys that changed.

    Only keys already in PARAMS are accepted: a typo would otherwise sit in the
    dict doing nothing while the table printed the default and looked swept.
    """
    changed = []
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set wants key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        key, raw = key.strip(), raw.strip()
        if key not in PARAMS:
            raise ValueError(f"unknown assumption {key!r}; pick one of "
                             + ", ".join(sorted(PARAMS)))
        try:
            value = int(raw) if key in INT_PARAMS else float(raw)
        except ValueError:
            raise ValueError(f"{key} wants {'an int' if key in INT_PARAMS else 'a number'}, "
                             f"got {raw!r}") from None
        if value <= 0:
            raise ValueError(f"{key} must be positive, got {value}")
        if key in FRACTION_PARAMS and value > 1:
            # offload_eff divides the NIC bandwidth, so 8 instead of 0.8 buys a
            # fabric that sustains ten times line rate and the table then says
            # offload wins everywhere. A fat-fingered exponent must not read as
            # a result.
            raise ValueError(f"{key} is a fraction of line rate, must be <= 1, got {value}")
        P[key] = value
        changed.append(key)
    return changed


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="override one assumption, e.g. --set nic_alpha=4e-6 (repeatable)")
    args = ap.parse_args(argv)
    try:
        changed = apply_overrides(args.overrides)
    except ValueError as e:
        ap.error(str(e))

    print("Tensor-parallel decode, 70B-class model, all numbers from PARAMS (assumed)")
    if changed:
        # say which numbers are not the defaults, so a pasted table is readable
        print("overridden: " + ", ".join(f"{k}={P[k]:g}" for k in changed))
    print()
    # Sweep whole nodes, not the literals 8/16/32: with --set gpus_per_node=6 the
    # old sweep asked allreduce for 8 GPUs, which is 1.33 nodes, and the run died
    # with a traceback half a table in. Same numbers as before at the default 8.
    g = P["gpus_per_node"]
    print(f"{'gpus':>4} {'batch':>5} {'msg':>8}  {'pick':<8} {'comm ms':>8} {'compute ms':>10} {'comm share':>10}   offload saves")
    for n in (g, 2 * g, 4 * g):
        for b in (1, 8, 32, 128, 512):
            m, algo, comm, comp = decode_step(b, n)
            _, _, comm_sw, _ = decode_step(b, n, allow_offload=False)
            save = f"{(1 - comm / comm_sw) * 100:4.0f}%" if algo == "offload" else "   -"
            print(f"{n:>4} {b:>5} {m / 1024:>6.0f}KB  {algo:<8} {comm * 1e3:>8.2f} {comp * 1e3:>10.2f} "
                  f"{comm / (comm + comp) * 100:>9.0f}%   {save}")
        print()

    print(f"Placement rule: is one {2 * g}-GPU TP group better than two {g}-GPU groups?")
    print(f"(same GPUs; the {2 * g}-way group halves compute per GPU but pays the NIC on every layer)")
    for b in (1, 32, 128):
        _, _, c1, p1 = decode_step(b, g)
        _, _, c2, p2 = decode_step(b, 2 * g)
        print(f"  batch {b:>3}: TP{g} step {(c1 + p1) * 1e3:6.2f} ms | TP{2 * g} step {(c2 + p2) * 1e3:6.2f} ms"
              f" -> {f'TP{2 * g} wins' if c2 + p2 < c1 + p1 else 'keep TP inside the node'}")


if __name__ == "__main__":
    main()
