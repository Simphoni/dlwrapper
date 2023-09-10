#include "model_manager.h"
#include "fast_unpickler.h"

#include <chrono>
namespace ch = std::chrono;

void initalizePyInterp() {
  if (Py_IsInitialized() == 0) {
    py::initialize_interpreter();
  }
}

void PyTorchModelManager::load() {
  if (loaded)
    return;
  initalizePyInterp();
  // py::scoped_interpreter guard{};
  INFO("loading model from %s", filename.c_str());
  fileReader = std::make_shared<ZipFileParser>(filename);
  fileReader->parse();

  // validate model file format
  std::vector<UnzippedFileMeta> &filesMeta = fileReader->filesMeta;
  modelname = std::filesystem::path(filesMeta[0].filename).begin()->string();
  INFO("found model name: %s", modelname.c_str());

  // pass a simplified kv mapping to unpickler
  std::unordered_map<std::string, char *> storageMap;
  for (auto file : filesMeta) {
    assert(file.filename.length() > 0);
    auto pth = std::filesystem::path(file.filename);
    auto it  = pth.begin();
    assert(it->string() == modelname);
    it++;
    if (it->string() == "data") {
      // it is a storage
      it++;
      assert(it != pth.end());
      storageMap[it->string()] = file.buffer;
    }
  }
  INFO("number of storage in checkpoint: %lu", storageMap.size());

  // load data.pkl
  auto it = fileReader->nameIdxMap.find(modelname + "/data.pkl");
  assert(it != fileReader->nameIdxMap.end());
  auto &conf = filesMeta[it->second];
  INFO("found data.pkl, now unpickling. data.pkl size: %.3lf MB", conf.size * 1e-6);
  make_memmove_advise(conf.buffer, conf.size, MADV_SEQUENTIAL | MADV_WILLNEED);
  unpickler        = std::make_shared<FastUnpickler>(conf, std::move(storageMap));
  auto parse_start = ch::high_resolution_clock::now();
  parse_result     = unpickler->unpickle();
  auto parse_end   = ch::high_resolution_clock::now();
  INFO("data.pkl parsed successfully in %.3lf ms",
       ch::duration_cast<ch::microseconds>(parse_end - parse_start).count() * 1e-3);

  // record tensor metadata
  parse_result->read_all_tensors(tensorMap);
  INFO("displaying %ld tensor(s)", tensorMap.size());
  int cnt = 0;
  for (auto &[key, val] : tensorMap) {
    cnt++;
    val->set_name(key);
    INFO("|-- (%d): %s: %s", cnt, key.c_str(), val->to_string().c_str());
  }

  loaded = true;
  return;
}