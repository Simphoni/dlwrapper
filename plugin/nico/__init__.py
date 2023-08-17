import torch
import atexit
import torch.distributed as dist
from dlwrapper import nico_native


def init_nico(enable_uva: bool):
    nico_native.init_nico(
        pg=dist.distributed_c10d._get_default_group(),
        enable_uva=enable_uva,
    )
    atexit.register(nico_native.destroy_nico)


__all__ = ["init_nico"]
