import torch
import dlwrapper
import time
import gc
from model import ckpt

torch.cuda.random.manual_seed(2333)

output_dims = (200, 100, 100)
ckpt_meta = ckpt.load()


def run_custom_sampler(input, grid, layout):
    shape = None
    if layout is False:
        shape = (1, *output_dims, input.shape[-1])
    else:
        shape = (1, input.shape[-1], *output_dims)
    output = torch.empty(shape, dtype=torch.float32, device="cuda")
    dlwrapper.F.grid_sample(input, grid, output, layout)
    return output


plane_origin = ckpt_meta["state_dict"]["density_plane.2"]
print(plane_origin.shape)
plane_origin.segment = [1, 1, 24, 1, 1]
tgrid = plane_origin.create_tensor_grid()
slice = tgrid.get_slice(0)
print(slice)

ret = slice.move_to(dlwrapper.MemoryType.PINNED)
slice.wait()
ret = slice.move_to(dlwrapper.MemoryType.DEVICE)
slice.wait()
input_const = slice.torch_get_contiguous(dlwrapper.MemoryType.DEVICE)

input = input_const.clone().detach()
input_T = input.permute(0, 2, 3, 4, 1)
input_T = input_T.contiguous()
sample_grid = torch.rand((1, *output_dims, 3), dtype=torch.float32, device="cuda")
sample_grid.mul_(2).sub_(1)

ans = torch.nn.functional.grid_sample(
    input, sample_grid, align_corners=True, mode="bilinear"
)
out = run_custom_sampler(input_T, sample_grid, False)
out = out.permute(0, 4, 1, 2, 3).contiguous()
diff = (out - ans).abs()
print(f"max absolute error: {diff.max()}")
print(f"average of diff^2: {diff.square().mean()}")

out = run_custom_sampler(input_T, sample_grid, True)
diff = (out - ans).abs()
safe_ans = ans.abs().clamp(min=1e-5)
print(f"max absolute error: {diff.max()}")
print(f"max relative error: {torch.div(diff, safe_ans).max()}")
print(f"average of diff^2: {diff.square().mean()}")


def perf_custom(input, grid, layout, WARMUP=64, ITER=64):
    for i in range(WARMUP):
        run_custom_sampler(input, grid, layout)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    st = time.time()
    for i in range(WARMUP):
        run_custom_sampler(input, grid, layout)
    ed = time.time()
    print(f"dlwrapper.F(layout={layout}) time: {(ed - st) * 1000 / ITER}ms")


def perf_torch(input, grid, WARMUP=64, ITER=64):
    for i in range(WARMUP):
        torch.nn.functional.grid_sample(
            input, grid, align_corners=True, mode="bilinear"
        )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    st = time.time()
    for i in range(WARMUP):
        torch.nn.functional.grid_sample(
            input, grid, align_corners=True, mode="bilinear"
        )
    ed = time.time()
    print(f"torch time: {(ed - st) * 1000 / ITER}ms")


perf_custom(input_T, sample_grid, False, 128, 128)
perf_custom(input_T, sample_grid, True, 128, 128)
perf_torch(input, sample_grid, 128, 128)
