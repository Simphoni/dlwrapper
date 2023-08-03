import torch
import torch.distributed as dist
import os
import time
import datetime


WARMUP_ROUND=5
PERF_ROUND=20

def test_broadcast(bytes: int, proc_grp):
    ten: torch.Tensor = torch.randn(bytes//4, dtype=float, device="cuda:0")
    
    torch.distributed.barrier(group=process_group, async_op=False)

    for i in range(WARMUP_ROUND):
        print(f"{proc_grp.rank()} broadcasting {i}")
        dist.broadcast(tensor=ten, src=0, group=proc_grp)
    torch.distributed.barrier(group=proc_grp, async_op=False)
    
    begin_time = time.perf_counter_ns()
    for _ in range(PERF_ROUND):
        dist.broadcast(tensor=ten, src=0, group=proc_grp)
    torch.distributed.barrier(group=proc_grp, async_op=False)
    end_time = time.perf_counter_ns()
    time_in_ms = (end_time-begin_time) / PERF_ROUND / 1e6
    print(f"[RANK {proc_grp.rank()}: broadcast {bytes}B consumed {time_in_ms} ms, checksum={ten.mean()}, bandwidth={bytes / (time_in_ms/1000) / 1024/1024/1024}GBps")

    del ten
    torch.cuda.empty_cache()

if __name__ == "__main__":
    rank: int = int(os.getenv("RANK"))
    world: int = int(os.getenv("WORLD_SIZE"))
    nccl_timeout = int(os.environ.get("NCCL_TIMEOUT", 1800))
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank%8}"
    dist.init_process_group(
        backend="nccl",
        world_size=world,
        rank=rank,
        init_method="env://",
        timeout=datetime.timedelta(0, nccl_timeout),
    )
    process_group = dist.new_group([i for i in range(world)])
    with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=False, profile_memory=False) as prof:
        test_broadcast(4*1024*1024*1024, process_group)
    prof.export_chrome_trace('./broadcast_profile.json')