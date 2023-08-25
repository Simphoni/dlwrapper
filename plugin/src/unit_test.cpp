#include "device_manager.h"
#include "nico.h"
#include "nv_common.h"

void test_ipc_allgather() {
  static int callnum = 0;
  callnum++;
  auto manager = DeviceContextManager::get();
  auto pg = manager->get_process_group(0);
  int data = manager->get_local_rank() + callnum;
  int array[] = {data, data, data, data};
  auto recv = pg->ipc_allgather((char *)&array[0], sizeof(int) * 4);
  int *result = (int *)recv.data();
  if (false) {
    char s[1000];
    char *cur = s;
    for (int i = 0; i < 32; i++) {
      cur += sprintf(cur, "%d ", result[i]);
    }
    if (manager->get_world_rank() == 1)
      DEBUG("rank[%d], it=%d: %s", manager->get_local_rank(), callnum, s);
  }
  for (int i = 0, res; i < 8; i++) {
    for (int j = 0; j < 4; j++) {
      res = result[i * 4 + j];
      assert(res == callnum + i);
    }
  }
}