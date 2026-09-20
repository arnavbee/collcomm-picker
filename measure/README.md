# Measuring the assumed numbers

`../picker.py` runs on assumed latency and bandwidth. These two scripts replace the guesses
with measured values.

    python3 measure_allreduce.py --backend nccl --world 2 --out t4x2.json   # 2 GPUs
    python3 fit_alpha_beta.py t4x2.json

`measure_allreduce.py` times `torch.distributed.all_reduce` from 8 B to 128 MB (20 warm-up,
100 timed iterations per size, median, min and p90 recorded). `fit_alpha_beta.py` reads the
sweep and prints the small-message latency floor, the large-message effective bandwidth and the
message size where latency stops being flat.

## What is in here now

`gloo-cpu-2.json` and `gloo-cpu-4.json` are sanity runs on an 8 GB M1 MacBook Air, CPU
backend, processes on one machine. They show the scripts work. They are NOT GPU numbers and
say nothing about NVLink, InfiniBand or any cluster.

    2 ranks: 154 us small-message floor, 3.5 GB/s
    4 ranks: 543 us small-message floor, 1.4 GB/s

## What is still missing

A run on real GPUs. Free option: a Kaggle notebook with the "GPU T4 x2" accelerator
(2 GPUs over PCIe, about 30 hours a week free). Paste this into one cell:

    !git clone https://github.com/arnavbee/collcomm-picker && cd collcomm-picker/measure && \
     python3 measure_allreduce.py --backend nccl --world 2 --out t4x2.json && \
     python3 fit_alpha_beta.py t4x2.json

Even then it is two GPUs on one box, over PCIe. It says nothing about NVLink or multi-node,
which is where the picker's interesting cases are. A multi-node run needs a cluster.
