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


WARMUP_ROUND = 20
PERF_ROUND = 20


def broadcast():
    t = torch.randn(1024 * 1024 * 1024, device="cuda:0")  # 4G
    if local_rank == 0:
        print(f"rank {local_rank}: {t.sum().item()}")
    for _ in range(WARMUP_ROUND):
        dlwrapper.nico.broadcast(t)

    start_tick = time.perf_counter_ns()
    for _ in range(PERF_ROUND):
        dlwrapper.nico.broadcast(t)
    end_tick = time.perf_counter_ns()

    time_in_s = (end_tick - start_tick) / 1e9 / PERF_ROUND
    print(f"rank {local_rank}: {t.sum().item()}, {4 / time_in_s}GB/s")


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
    broadcast()
else:
    a = torch.randn(20, 30)
    dlwrapper.nico.testing(a)
