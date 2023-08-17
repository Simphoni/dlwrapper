import os
import sys
import time
import torch  # torch must be imported ahead of dlwrapper
import torch.distributed as dist
import dlwrapper
import nico

local_rank, world_size = 0, 0
device = None


WARMUP_ROUND = 20
PERF_ROUND = 20


def broadcast():
    t = torch.randn(1024 * 1024 * 1024, device=device)  # 4G
    print(f"rank {local_rank}: {t.sum().item()}")
    for _ in range(WARMUP_ROUND):
        dlwrapper.nico_native.broadcast(t, False)

    start_tick = time.perf_counter_ns()
    for _ in range(PERF_ROUND):
        dlwrapper.nico_native.broadcast(t, True)
    end_tick = time.perf_counter_ns()

    time_in_s = (end_tick - start_tick) / 1e9 / PERF_ROUND
    print(
        f"rank {local_rank}: {t.sum().item()}, {time_in_s * 1000} ms, {4 / time_in_s}GB/s"
    )


def sendrecv(src: int, dst: int):
    if local_rank != src and local_rank != dst:
        return
    t = torch.randn(1024 * 1024 * 1024, dtype=torch.float32, device=device)

    for _ in range(WARMUP_ROUND):
        dlwrapper.nico_native.sendrecv(t, src, dst, False)

    start_tick = time.perf_counter_ns()
    for _ in range(PERF_ROUND):
        dlwrapper.nico_native.sendrecv(t, src, dst, True)
    end_tick = time.perf_counter_ns()

    time_in_s = (end_tick - start_tick) / 1e9 / PERF_ROUND
    print(
        f"rank {local_rank}: {t.sum().item()}, {time_in_s * 1000} ms, {4 / time_in_s}GB/s"
    )


def allgather_into_tensor():
    G = 4
    bytes = G * 1024 * 1024 * 1024
    ten: torch.Tensor = torch.randn(
        (1, bytes // 32), dtype=torch.float32, device=device
    )
    dst: torch.Tensor = torch.randn(
        (8, bytes // 32), dtype=torch.float32, device=device
    )

    for i in range(WARMUP_ROUND):
        dlwrapper.nico_native.allgather_with_peer_access(dst, ten, 0, False)

    start_tick = time.perf_counter_ns()
    for _ in range(PERF_ROUND):
        dlwrapper.nico_native.allgather_with_peer_access(dst, ten, 0, True)
    end_tick = time.perf_counter_ns()
    time_in_s = (end_tick - start_tick) / 1e9 / PERF_ROUND
    print(f"rank {local_rank}: {dst.sum()}, {time_in_s * 1000} ms, {G / time_in_s}GB/s")


if os.getenv("RANK") != None:
    # if __name__ == "__main__":
    local_rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{local_rank}"
    # torch.cuda.set_device("cuda:0")
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=local_rank
    )
    device = torch.device(f"cuda:{local_rank}")

    nico.init_nico(True)
    if True:
        ROUNDS = 500
        for i in range(ROUNDS):
            dlwrapper.nico_native.testing()
        start_tick = time.perf_counter_ns()
        for i in range(ROUNDS):
            dlwrapper.nico_native.testing()
        end_tick = time.perf_counter_ns()
        time_in_s = (end_tick - start_tick) / 1e9
        print(
            f"rank {local_rank}: operation ave cost {time_in_s * 1000 / ROUNDS} ms",
            flush=True,
        )
    allgather_into_tensor()
else:
    # local work
    a = torch.randn(20, 30, dtype=float)
    dlwrapper.nico_native.testing()

dlwrapper.nico_native.export_summary()
