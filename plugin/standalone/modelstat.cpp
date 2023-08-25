#include "lazy_unpickler.h"
#include <cassert>
#include <cstdio>
#include <iostream>
int main(int argc, char **argv) {
  std::string file;
  if (argc == 2) {
    file = std::string(argv[1]);
  } else {
    file =
        std::string("/cpfs01/shared/pjlab-lingjun-landmarks/checkpoint/4k_ckpt/"
                    "2_preproc_almostall_1k-2k-4k_encapp_div16x12_upsample_fullrange_ctd_hull256/"
                    "2_preproc_almostall_1k-2k-4k_encapp_div16x12_upsample_fullrange_ctd_hull256-"
                    "merged-stack.th");
  }
  ZipFileParser zfp{file};
  for (auto &p : zfp.pickleFilesMeta) {
    printf("File %s, %lu\n", p.filename.c_str(), p.size);
  }
  return 0;
}