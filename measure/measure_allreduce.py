"""Time torch.distributed all_reduce over message sizes. Works on CPU (gloo) or GPUs (nccl).

    python3 measure_allreduce.py --backend gloo --world 2 --out cpu2.json
    python3 measure_allreduce.py --backend nccl --world 2 --out t4x2.json     # needs 2 GPUs

Writes {"backend","world","device","rows":[{"bytes","median_us","min_us","p90_us"}]}.
"""
import argparse, json, os, statistics, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES = [2 ** k for k in range(3, 28)]            # 8 B .. 128 MB
WARMUP, ITERS = 20, 100


def run(rank, world, backend, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=os.environ.get("MPORT", "29511"))
    dist.init_process_group(backend, rank=rank, world_size=world)
    cuda = backend == "nccl"
    if cuda:
        torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank) if cuda else torch.device("cpu")
    rows = []
    for nbytes in SIZES:
        x = torch.ones(max(nbytes // 4, 1), dtype=torch.float32, device=dev)
        times = []
        for i in range(WARMUP + ITERS):
            dist.barrier()
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            dist.all_reduce(x)
            if cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if i >= WARMUP:
                times.append(dt * 1e6)
        times.sort()
        rows.append(dict(bytes=x.numel() * 4, median_us=statistics.median(times),
                         min_us=times[0], p90_us=times[int(len(times) * 0.9)]))
    if rank == 0:
        name = torch.cuda.get_device_name(0) if cuda else "cpu"
        json.dump(dict(backend=backend, world=world, device=name, torch=torch.__version__, rows=rows),
                  open(out, "w"), indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--out", default="result.json")
    a = ap.parse_args()
    mp.spawn(run, args=(a.world, a.backend, a.out), nprocs=a.world)
