#include <Python.h>

class LazyUnpickler {
  // A unpickler that keeps references to large objects via file mapping. It
  // is part of a hierarchical model mananagement module.
private:
  static const int MAX_BUF_SIZE = 1 << 18;
  char buf[MAX_BUF_SIZE];
  void *hooks[256];
}