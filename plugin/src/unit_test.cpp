#include "common.h"
#include "device_manager.h"
#include "nico.h"

void test_ipc_allgather() {
  static int callnum = 0;
  callnum++;
  auto manager = DeviceContextManager::get();
  int data = manager->get_local_rank() + callnum;
  int array[] = {data, data, data, data};
  int *result =
      (int *)manager->ipc_allgather((char *)&array[0], sizeof(int) * 4);
  for (int i = 0, res; i < 8; i++) {
    for (int j = 0; j < 4; j++) {
      res = result[i * 4 + j];
      assert(res == callnum + i);
      if (res != callnum + i) {
        char s[1000];
        char *cur = s;
        for (int i = 0; i < 32; i++) {
          cur += sprintf(cur, "%d ", result[i]);
        }
        DEBUG("rank[%d], it=%d: %s", manager->get_local_rank(), callnum, s);
        throw;
      }
    }
  }
}