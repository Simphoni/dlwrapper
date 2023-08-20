import os
import sys
import time
import torch  # torch must be imported ahead of dlwrapper
import torch.distributed as dist
from dlwrapper import nico_native

local_rank, world_size = 0, 0
device = None


WARMUP_ROUND = 50
PERF_ROUND = 50


def broadcast():
    t = torch.randn(1024 * 1024 * 1024, device=device)  # 4G
    print(f"rank {local_rank}: {t.sum().item()}")
    for _ in range(WARMUP_ROUND):
        nico_native.broadcast(t, False)

    start_tick = time.perf_counter_ns()
    for _ in range(PERF_ROUND):
        nico_native.broadcast(t, True)
    end_tick = time.perf_counter_ns()

    time_in_s = (end_tick - start_tick) / 1e9 / PERF_ROUND
    print(
        f"rank {local_rank}: {t.sum().item()}, {time_in_s * 1000} ms, {4 / time_in_s}GB/s"
    )


# deprecate
def sendrecv(src: int, dst: int):
    if local_rank != src and local_rank != dst:
        return
    t = torch.randn(1024 * 1024 * 1024, dtype=torch.float32, device=device)

    for _ in range(WARMUP_ROUND):
        nico_native.sendrecv(t, src, dst, False)

    start_tick = time.perf_counter_ns()
    for _ in range(PERF_ROUND):
        nico_native.sendrecv(t, src, dst, True)
    end_tick = time.perf_counter_ns()

    time_in_s = (end_tick - start_tick) / 1e9 / PERF_ROUND
    print(
        f"rank {local_rank}: {t.sum().item()}, {time_in_s * 1000} ms, {4 / time_in_s}GB/s"
    )


def allgather_into_tensor(G):
    bytes = int(G * 1024 * 1024 * 1024)
    ten: torch.Tensor = torch.randn(
        (1, bytes // 32), dtype=torch.float32, device=device
    )
    dst: torch.Tensor = torch.randn(
        (8, bytes // 32), dtype=torch.float32, device=device
    )

    for i in range(WARMUP_ROUND):
        nico_native.allgather(dst, ten, 0, False)

    start_tick = time.perf_counter_ns()
    for _ in range(PERF_ROUND):
        nico_native.allgather(dst, ten, 0, True)
    end_tick = time.perf_counter_ns()
    time_in_s = (end_tick - start_tick) / 1e9 / PERF_ROUND
    print(
        f"rank {local_rank}: {dst.sum()}, {time_in_s * 1000} ms, {bytes / 1e9 / time_in_s}GB/s"
    )
    del ten
    del dst


def scatter(G):
    bytes = int(G * 1024 * 1024 * 1024)
    ten = None
    if local_rank == 0:
        ten = torch.randn((8, bytes // 32), dtype=torch.float32, device=device)
    else:
        ten = torch.empty((bytes // 32), dtype=torch.float32, device=device)

    for i in range(WARMUP_ROUND):
        nico_native.scatter(ten, 0, False)

    start_tick = time.perf_counter_ns()
    for _ in range(PERF_ROUND):
        nico_native.scatter(ten, 0, True)
    end_tick = time.perf_counter_ns()
    time_in_s = (end_tick - start_tick) / 1e9 / PERF_ROUND
    print(
        f"rank {local_rank}: {ten.sum()}, {time_in_s * 1000} ms, {bytes / 1e9 / time_in_s}GB/s"
    )
    if local_rank == 0:
        for i in range(8):
            print(f"src={i}: {ten[i].sum()}")

    del ten


def init_nico(enable_uva: bool):
    nico_native.init_nico(
        pg=dist.distributed_c10d._get_default_group(),
        enable_uva=enable_uva,
    )


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
    node = dist.new_group(list(range(8)))

    init_nico(True)
    test_func = [scatter]
    for G in [0.5, 1, 2, 4, 8]:
        for f in test_func:
            f(G)
            dist.barrier(node)
            if local_rank == 0:
                print("--------------------")
            dist.barrier(node)
else:
    # local work
    a = torch.randn(20, 30, dtype=float)
    nico_native.testing()

nico_native.export_summary()
