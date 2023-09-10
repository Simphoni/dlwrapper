import dlwrapper

ckpt = dlwrapper.ModelManagerTorch(
    "/cpfs01/shared/pjlab-lingjun-landmarks/checkpoint/4k_ckpt/"
    "2_preproc_almostall_1k-2k-4k_encapp_div16x12_upsample_fullrange_ctd_hull256/"
    "2_preproc_almostall_1k-2k-4k_encapp_div16x12_upsample_fullrange_ctd_hull256-"
    "merged-stack.th"
)

result = ckpt.load()["state_dict"]
tensor = result["app_plane.2"]
print(tensor.shape)
tensor.segment = [1, 4, 192, 1, 1]
grid = tensor.create_tensor_grid()
slice = grid.get_slice(0)
print(slice)
