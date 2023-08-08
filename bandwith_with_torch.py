import torch
import torch.distributed as dist
import os
import time
import datetime


WARMUP_ROUND = 5
PERF_ROUND = 20
rank, world = 0, 0


def test_broadcast(bytes: int, proc_grp):
    ten: torch.Tensor = torch.randn(bytes // 4, dtype=torch.float32, device="cuda:0")

    torch.distributed.barrier(group=process_group, async_op=False)

    for _ in range(WARMUP_ROUND):
        dist.broadcast(tensor=ten, src=0, group=proc_grp)
    torch.distributed.barrier(group=proc_grp, async_op=False)

    begin_time = time.perf_counter_ns()
    for _ in range(PERF_ROUND):
        dist.broadcast(tensor=ten, src=0, group=proc_grp)
    torch.distributed.barrier(group=proc_grp, async_op=False)
    end_time = time.perf_counter_ns()
    time_in_ms = (end_time - begin_time) / PERF_ROUND / 1e6
    print(
        f"[RANK {proc_grp.rank()}]: broadcast {bytes}B consumed {time_in_ms} ms, checksum={ten.mean()}, bandwidth={bytes / (time_in_ms/1000) / 1024/1024/1024}GBps"
    )

    del ten
    torch.cuda.empty_cache()


def test_sendrecv(bytes: int, proc_grp):
    ten0: torch.Tensor = torch.randn(bytes // 4, dtype=torch.float32, device="cuda:0")
    ten1: torch.Tensor = torch.randn(bytes // 4, dtype=torch.float32, device="cuda:0")

    torch.distributed.barrier(group=process_group, async_op=False)

    for _ in range(WARMUP_ROUND):
        if rank % 2 == 0:
            obj0 = dist.isend(tensor=ten0, dst=(rank + 1) % 8, group=proc_grp)
            obj1 = dist.irecv(tensor=ten1, src=(rank + 7) % 8, group=proc_grp)
        else:
            obj1 = dist.irecv(tensor=ten1, src=(rank + 7) % 8, group=proc_grp)
            obj0 = dist.isend(tensor=ten0, dst=(rank + 1) % 8, group=proc_grp)
        obj0.wait()
        obj1.wait()
        torch.distributed.barrier(group=proc_grp, async_op=False)

    begin_time = time.perf_counter_ns()
    for _ in range(PERF_ROUND):
        if rank % 2 == 0:
            obj0 = dist.isend(tensor=ten0, dst=(rank + 1) % 8, group=proc_grp)
            obj1 = dist.irecv(tensor=ten1, src=(rank + 7) % 8, group=proc_grp)
        else:
            obj1 = dist.irecv(tensor=ten1, src=(rank + 7) % 8, group=proc_grp)
            obj0 = dist.isend(tensor=ten0, dst=(rank + 1) % 8, group=proc_grp)
        obj0.wait()
        obj1.wait()
        torch.distributed.barrier(group=proc_grp, async_op=False)
    end_time = time.perf_counter_ns()
    time_in_ms = (end_time - begin_time) / PERF_ROUND / 1e6
    print(
        f"[RANK {proc_grp.rank()}]: sendrecv {bytes}B consumed {time_in_ms} ms, bandwidth={bytes / (time_in_ms/1000) / 1024/1024/1024}GBps"
    )

    del ten0
    del ten1
    torch.cuda.empty_cache()


def test_allgather(bytes: int, proc_grp):
    ten: torch.Tensor = torch.randn(
        (1, bytes // 32), dtype=torch.float32, device="cuda:0"
    )
    dst: torch.Tensor = torch.randn(
        (8, bytes // 32), dtype=torch.float32, device="cuda:0"
    )
    torch.distributed.barrier(group=process_group, async_op=False)

    for _ in range(WARMUP_ROUND):
        dist.all_gather_into_tensor(output_tensor=dst, input_tensor=ten, group=proc_grp)
    torch.distributed.barrier(group=proc_grp, async_op=False)

    begin_time = time.perf_counter_ns()
    for _ in range(PERF_ROUND):
        dist.all_gather_into_tensor(output_tensor=dst, input_tensor=ten, group=proc_grp)
    torch.distributed.barrier(group=proc_grp, async_op=False)
    end_time = time.perf_counter_ns()
    time_in_ms = (end_time - begin_time) / PERF_ROUND / 1e6
    print(
        f"[RANK {proc_grp.rank()}]: allgather into tensor {bytes}B consumed {time_in_ms} ms, checksum={dst.mean()}, bandwidth={bytes / (time_in_ms/1000) / 1024/1024/1024}GBps"
    )

    del ten
    torch.cuda.empty_cache()


if __name__ == "__main__":
    rank: int = int(os.getenv("RANK"))
    world: int = int(os.getenv("WORLD_SIZE"))
    nccl_timeout = int(os.environ.get("NCCL_TIMEOUT", 1800))
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank%8}"
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl",
        world_size=world,
        rank=rank,
        init_method="env://",
        timeout=datetime.timedelta(0, nccl_timeout),
    )
    process_group = dist.new_group([i for i in range(world)])
    test_sendrecv(4 * 1024 * 1024 * 1024, process_group)
