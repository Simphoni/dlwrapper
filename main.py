import os
import torch  # torch must be imported ahead of dlwrapper
import torch.distributed as dist
import dlwrapper


def init_nico():
    dlwrapper.nico.init_nccl(
        dist.distributed_c10d._get_default_group(), torch.device("cuda:0")
    )


if __name__ == "__main__":
    local_rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{local_rank}"
    torch.cuda.set_device("cuda:0")

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=local_rank
    )
    init_nico()
