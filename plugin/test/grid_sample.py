import torch
import dlwrapper
import time
import gc

torch.cuda.random.manual_seed(2333)

sample_grid = torch.load("grid_3d_test_small.pt", map_location="cuda:0")
output_dims = tuple(sample_grid.shape[1:4])


def run_custom_sampler(input, grid, layout):
    shape = None
    if layout is False:
        shape = (1, *output_dims, input.shape[-1])
    else:
        shape = (1, input.shape[-1], *output_dims)
    output = torch.empty(shape, dtype=torch.float32, device="cuda")
    dlwrapper.F.grid_sample(input, grid, output, layout)
    return output


def perf_custom(input, grid, layout, WARMUP=64, ITER=64):
    for i in range(WARMUP):
        run_custom_sampler(input, grid, layout)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    st = time.time()
    for i in range(ITER):
        run_custom_sampler(input, grid, layout)
    torch.cuda.synchronize()
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
    for i in range(ITER):
        torch.nn.functional.grid_sample(
            input, grid, align_corners=True, mode="bilinear"
        )
    torch.cuda.synchronize()
    ed = time.time()
    print(f"torch time: {(ed - st) * 1000 / ITER}ms")


input_const = torch.randn(1, 16, 8, 343, 172, dtype=torch.float32, device="cuda")

input = input_const.clone().detach()
input_T = input.permute(0, 2, 3, 4, 1)
input_T = input_T.contiguous()

print(f"tensor config: {input_T.shape}, {sample_grid.shape}")

ans = torch.nn.functional.grid_sample(
    input, sample_grid, align_corners=True, mode="bilinear"
)

tuner = 384

perf_custom(input_T, sample_grid, False, tuner, tuner)
out = run_custom_sampler(input_T, sample_grid, False)
out = out.permute(0, 4, 1, 2, 3).contiguous()
diff = (out - ans).abs()
print(f"max absolute error: {diff.max()}")

perf_custom(input_T, sample_grid, True, tuner, tuner)
out = run_custom_sampler(input_T, sample_grid, True)
diff = (out - ans).abs()
safe_ans = ans.abs().clamp(min=1e-5)
print(f"max absolute error: {diff.max()}")


perf_torch(input, sample_grid, tuner, tuner)
