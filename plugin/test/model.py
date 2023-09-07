import dlwrapper

a = dlwrapper.ModelManagerTorch(
    "/cpfs01/shared/pjlab-lingjun-landmarks/checkpoint/4k_ckpt/"
    "2_preproc_almostall_1k-2k-4k_encapp_div16x12_upsample_fullrange_ctd_hull256/"
    "2_preproc_almostall_1k-2k-4k_encapp_div16x12_upsample_fullrange_ctd_hull256-"
    "merged-stack.th"
)

result = a.load()

print(result)
