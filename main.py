import os
import time
import torch  # torch must be imported ahead of dlwrapper
import torch.distributed as dist
import dlwrapper

local_rank, world_size = 0, 0


def init_nico():
    dlwrapper.nico.init_nccl(
        dist.distributed_c10d._get_default_group(), torch.device("cuda:0")
    )


WARMUP_ROUND = 50
PERF_ROUND = 50


def broadcast():
    t = torch.randn(1024 * 1024 * 1024, device="cuda:0")  # 4G
    print(f"rank {local_rank}: {t.sum().item()}")
    for _ in range(WARMUP_ROUND):
        dlwrapper.nico.broadcast(t, False)

    start_tick = time.perf_counter_ns()
    for _ in range(PERF_ROUND):
        dlwrapper.nico.broadcast(t, True)
    end_tick = time.perf_counter_ns()

    time_in_s = (end_tick - start_tick) / 1e9 / PERF_ROUND
    print(
        f"rank {local_rank}: {t.sum().item()}, {time_in_s * 1000} ms, {4 / time_in_s}GB/s"
    )


def sendrecv(src: int, dst: int):
    if local_rank != src and local_rank != dst:
        return
    t = torch.randn(1024 * 1024 * 1024, dtype=torch.float32, device="cuda:0")

    for _ in range(WARMUP_ROUND):
        dlwrapper.nico.sendrecv(t, src, dst, False)

    start_tick = time.perf_counter_ns()
    for _ in range(PERF_ROUND):
        dlwrapper.nico.sendrecv(t, src, dst, True)
    end_tick = time.perf_counter_ns()

    time_in_s = (end_tick - start_tick) / 1e9 / PERF_ROUND
    print(
        f"rank {local_rank}: {t.sum().item()}, {time_in_s * 1000} ms, {4 / time_in_s}GB/s"
    )


def allgather_into_tensor():
    G = 4
    bytes = G * 1024 * 1024 * 1024
    ten: torch.Tensor = torch.randn(
        (1, bytes // 32), dtype=torch.float32, device="cuda:0"
    )
    dst: torch.Tensor = torch.randn(
        (8, bytes // 32), dtype=torch.float32, device="cuda:0"
    )

    for _ in range(WARMUP_ROUND):
        dlwrapper.nico.allgather_into_tensor_doubling(dst, ten, False)

    start_tick = time.perf_counter_ns()
    for _ in range(PERF_ROUND):
        dlwrapper.nico.allgather_into_tensor_doubling(dst, ten, True)
    end_tick = time.perf_counter_ns()
    time_in_s = (end_tick - start_tick) / 1e9 / PERF_ROUND
    print(f"rank {local_rank}: {dst.sum()}, {time_in_s * 1000} ms, {G / time_in_s}GB/s")


if os.getenv("RANK") != None:
    # if __name__ == "__main__":
    local_rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{local_rank}"
    torch.cuda.set_device("cuda:0")
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=local_rank
    )

    init_nico()
    allgather_into_tensor()
    if local_rank == 0:
        dlwrapper.nico.export_summary()
else:
    a = torch.randn(20, 30, dtype=float)
    dlwrapper.nico.testing(a)
