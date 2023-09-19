import dlwrapper

ckpt = dlwrapper.ModelManagerTorch(
    "/cpfs01/shared/pjlab-lingjun-landmarks/checkpoint/4k_ckpt/"
    "2_preproc_almostall_1k-2k-4k_encapp_div16x12_upsample_fullrange_ctd_hull256/"
    "2_preproc_almostall_1k-2k-4k_encapp_div16x12_upsample_fullrange_ctd_hull256-"
    "merged-stack.th"
)

if __name__ == "__main__":
    result = ckpt.load()["state_dict"]
    print(ckpt.load())
    otensor = result["app_plane.0"]
    print(otensor.shape)
    otensor.segment = [1, 48, 192, 1, 1]
    grid = otensor.create_tensor_grid()
    slice = grid.get_slice(0)
    print(slice)
    ret = slice.move_to(dlwrapper.MemoryType.PINNED)
    print(ret)
    slice.wait()
    ret = slice.move_to(dlwrapper.MemoryType.DEVICE)
    print(ret)
    slice.wait()
    a = slice.torch_get_contiguous(dlwrapper.MemoryType.DEVICE)
    print(a)
