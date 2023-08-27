#include "lazy_unpickler.h"
#include <unordered_map>

// used by unpickle when processing byte stream
template <typename T> inline uint8_t read_uint8(T *ptr) noexcept { return *(uint8_t *)ptr; }

template <typename T> inline uint16_t read_uint16(T *ptr) noexcept { return *(uint16_t *)ptr; }

template <typename T> uint32_t read_uint32(T *ptr) { return *(uint32_t *)ptr; }

void ZipFileParser::parse() {
  // do some checking
  assert(std::filesystem::exists(filename));
  assert(std::filesystem::is_regular_file(filename));
  filelen = std::filesystem::file_size(filename);
  int fd  = open(filename.c_str(), O_RDONLY);
  if (fd == -1) {
    printf("ZipFileParser cannot open file %s\n", filename.c_str());
    throw;
  }
  gigabuffer = (char *)mmap(NULL, filelen, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);

  // parse end of central directory record
  char *ptr = gigabuffer + filelen - 22;
  // .ZIP file can have a variable comment, so iterate for the signature
  while (true) {
    uint32_t data = read_uint32(ptr);
    if (data == kEndOfCentralDirectorySignature)
      break;
    --ptr;
  }
  uint64_t offset_of_start_of_central_directory = *(uint32_t *)(ptr + 16);
  uint64_t n_entries_in_central_directory       = *(uint16_t *)(ptr + 10);
  if (offset_of_start_of_central_directory == 0xFFFFFFFF) {
    ptr -= 20; // zip64 end of central directory locator
    assert(read_uint32(ptr) == kZip64EndOfCentralDirectoryLocatorSignature);
    int64_t offset_of_start_of_central_directory_record = *(uint64_t *)(ptr + 8);
    ptr = gigabuffer + offset_of_start_of_central_directory_record;
    assert(read_uint32(ptr) == kZip64EndOfCentralDirectoryRecordSignature);
    n_entries_in_central_directory       = *(uint64_t *)(ptr + 32);
    offset_of_start_of_central_directory = *(uint64_t *)(ptr + 48);
    ptr                                  = gigabuffer + offset_of_start_of_central_directory;
  } else {
    ptr = gigabuffer + offset_of_start_of_central_directory;
  }

  // use ptr to iterate over central directory headers
  for (uint64_t i = 0; i < n_entries_in_central_directory; i++) {
    auto *header = reinterpret_cast<CentralDirectoryHeader *>(ptr);
    assert(header->signature == kCentralDirectorySignature);
    ptr += sizeof(CentralDirectoryHeader);
    std::string filename = std::string(ptr, header->file_name_length);
    ptr += header->file_name_length;
    nameIdxMap[filename] = i;
    if (header->uncompressed_size != 0xFFFFFFFF) {
      assert(header->compressed_size == header->uncompressed_size);
      filesMeta.emplace_back(std::move(filename),
                             gigabuffer + header->relative_offset_of_local_header,
                             (uint64_t)header->uncompressed_size);
    } else {
      auto *extra_field = reinterpret_cast<Zip64ExtendedExtraField *>(ptr);
      assert(extra_field->block_type == 0x0001);
      assert(extra_field->compressed_size == extra_field->original_size);
      // according to zip64, extra_field might only contain sizes
      filesMeta.emplace_back(std::move(filename), nullptr, extra_field->original_size);
    }
    ptr += header->extra_field_length + header->file_comment_length;
  }

  // iterate over files
  ptr = gigabuffer;
  for (uint64_t i = 0; i < n_entries_in_central_directory; i++) {
    auto *header = reinterpret_cast<LocalFileHeader *>(ptr);
    assert(header->signature == kLocalFileHeaderSignature);
    assert(header->compression_method == 0);
    ptr += sizeof(LocalFileHeader);
    std::string filename = std::string(ptr, header->file_name_length);
    uint64_t idx         = nameIdxMap[filename];
    ptr += header->file_name_length;
    ptr += header->extra_field_length;
    filesMeta[idx].buffer = ptr;
    ptr += filesMeta[idx].size;
    if (i + 1 < n_entries_in_central_directory) {
      int counter = 0;
      while (read_uint32(ptr) != kLocalFileHeaderSignature) {
        ptr++;
        counter++;
        assert(counter <= 64);
      }
    }
  }

  parsed = true;
}

ZipFileParser::~ZipFileParser() { munmap(gigabuffer, filelen); }

std::string LazyUnpickler::readline() {
  char *tail = iterator;
  while (*tail != '\n')
    tail++;
  std::string line(iterator, tail - iterator);
  iterator = tail + 1;
  return line;
}

uint64_t element_size(std::string dtype) {
  uint64_t size = 0;
  if (dtype == "Byte" || dtype == "Char" || dtype == "Uint8" || dtype == "Bool") {
    size = 1;
  } else if (dtype == "Half" || dtype == "Short") {
    size = 2;
  } else if (dtype == "Float" || dtype == "Int") {
    size = 4;
  } else if (dtype == "Double" || dtype == "Long") {
    size = 8;
  } else {
    fprintf(stderr, "%s\n", dtype.c_str());
    throw std::runtime_error("Data type not recognized.");
  }
  return size;
}

std::pair<std::string, uint64_t> parseStorageType(std::string attr) {
  assert(attr.size() > 7);
  auto len = attr.size();
  assert(attr.substr(len - 7, 7) == "Storage");
  std::string dtype = attr.substr(0, len - 7);
  if (dtype == "Untyped") {
    dtype = "Uint8";
  }
  uint64_t size = element_size(dtype);
  return std::make_pair(std::move(dtype), size);
}

void LazyUnpickler::pytorchPersistentId(std::shared_ptr<object> tuple) {
  assert(tuple->type == object::object_t::TUPLE);
  assert(tuple->children.size() == 5);
  assert(tuple->children[0]->extract_string() == "storage");
  auto storage_type = tuple->children[1];
  assert(storage_type->type == object::object_t::MODULE_ATTR);
  assert(storage_type->children[0]->extract_string() == "torch");
  auto [dtype, dsize]  = parseStorageType(storage_type->children[1]->extract_string());
  std::string key      = tuple->children[2]->extract_string();
  std::string location = tuple->children[3]->extract_string();
  uint64_t numel       = tuple->children[4]->extract_int();
  auto it              = storageMap.find(key);
  assert(it != storageMap.end());
  stack.emplace_back(std::make_shared<object>(
      std::make_shared<Storage>(it->second, dtype, location, numel, dsize)));
}

LazyUnpickler::LazyUnpickler(UnzippedFileMeta _file,
                             std::unordered_map<std::string, char *> storageMap)
    : file(_file), storageMap(storageMap) {
  unpickled     = false;
  actions[MARK] = [this]() {
    iterator++;
    stack.emplace_back(std::make_shared<object>(object::object_t::MARK));
  };
  actions[EMPTY_TUPLE] = [this]() {
    iterator++;
    stack.emplace_back(std::make_shared<object>(object::object_t::TUPLE));
  };
  actions[BININT1] = [this]() {
    iterator++;
    stack.emplace_back(std::make_shared<object>((int64_t)read_uint8(iterator)));
    iterator++;
  };
  actions[BININT2] = [this]() {
    iterator++;
    stack.emplace_back(std::make_shared<object>((int64_t)read_uint16(iterator)));
    iterator += 2;
  };
  actions[REDUCE] = [this]() {
    iterator++;
    auto arg = stack.back();
    stack.pop_back();
    auto func = stack.back();
    stack.pop_back();
    assert(func->type == object::object_t::MODULE_ATTR);
    std::string method_name =
        func->children[0]->extract_string() + "." + func->children[1]->extract_string();
    try {
      if (method_name == "torch._utils._rebuild_tensor_v2") {
        fprintf(stderr, "\n\nrebuild tensor %s\n\n", arg->to_string().c_str());
        stack.emplace_back(std::make_shared<object>(object::object_t::DUMMY, arg->children));
      } else if (method_name == "collections.OrderedDict") {
        stack.emplace_back(std::make_shared<object>(object::object_t::DICT, arg->children));
      } else {
        assert(0);
      }
    } catch (...) {
      assert(0);
    }
  };
  actions[BINPERSID] = [this]() {
    iterator++;
    pytorchPersistentId(stack.back());
  };
  actions[BINUNICODE] = [this] {
    iterator++;
    uint32_t len = read_uint32(iterator);
    iterator += 4;
    stack.emplace_back(std::make_shared<object>(std::string(iterator, len)));
    iterator += len;
  };
  actions[EMPTY_LIST] = [this] {
    iterator++;
    stack.emplace_back(std::make_shared<object>(object::object_t::LIST));
  };
  actions[GLOBAL] = [this]() {
    iterator++;
    auto module = readline();
    auto name   = readline();
    std::vector<std::shared_ptr<object>> vec(
        {std::make_shared<object>(std::move(module)), std::make_shared<object>(std::move(name))});
    stack.emplace_back(std::make_shared<object>(object::object_t::MODULE_ATTR, std::move(vec)));
  };
  actions[BINPUT] = [this]() {
    iterator++;
    uint32_t idx = read_uint8(iterator);
    memo[idx]    = stack.back();
    iterator++;
  };
  actions[LONG_BINPUT] = [this]() {
    iterator++;
    uint32_t idx = read_uint32(iterator);
    memo[idx]    = stack.back();
    iterator += 4;
  };
  actions[TUPLE] = [this]() {
    iterator++;
    assert(stack.size() > 0);
    int mark = stack.size() - 1;
    while (mark >= 0 && stack[mark]->type != object::object_t::MARK) {
      mark--;
    }
    assert(mark >= 0);
    std::vector<std::shared_ptr<object>> vec(stack.begin() + mark + 1, stack.end());
    stack.erase(stack.begin() + mark, stack.end());
    stack.emplace_back(std::make_shared<object>(object::object_t::TUPLE, std::move(vec)));
  };
  actions[EMPTY_DICT] = [this]() {
    iterator++;
    stack.emplace_back(std::make_shared<object>(object::object_t::DICT));
  };
  actions[PROTO] = [this]() {
    iterator++;
    assert(read_uint8(iterator) <= HIGHEST_PROTOCOL);
    iterator++;
  };
  actions[TUPLE2] = [this]() {
    iterator++;
    assert(stack.size() >= 2);
    std::vector<std::shared_ptr<object>> vec(stack.end() - 2, stack.end());
    stack.erase(stack.end() - 2, stack.end());
    stack.emplace_back(std::make_shared<object>(object::object_t::TUPLE, std::move(vec)));
  };
  actions[NEWTRUE] = [this]() {
    iterator++;
    stack.emplace_back(std::make_shared<object>(true));
  };
  actions[NEWFALSE] = [this]() {
    iterator++;
    stack.emplace_back(std::make_shared<object>(false));
  };
}

void LazyUnpickler::unpickle() {
  int numit = 0;

  if (unpickled)
    return;
  stop_flag = false;
  iterator  = file.buffer;
  stack.reserve(256);
  assert(read_uint8(iterator) == 0x80);
  while (!stop_flag) {
    fprintf(stderr, "%02x ", read_uint8(iterator));
    try {
      actions[read_uint8(iterator)]();
    } catch (...) {
      printf("\nerror parsing at ---> %d: %x\n", numit, read_uint8(iterator));
      break;
    }

    if (true) {
      numit++;
      if (numit >= 43)
        break;
    }
  }
  puts("");
  INFO("stack size: %ld", stack.size());
  for (auto i : stack)
    INFO("%s", i->to_string().c_str());
  puts("");
  INFO("memo size: %ld", memo.size());
  for (auto i : memo)
    INFO("%u %s", i.first, i.second->to_string().c_str());
  unpickled = true;
}

void PyTorchModelManager::load() {
  if (loaded)
    return;
  INFO("loading model from %s", filename.c_str());
  fileReader = std::make_shared<ZipFileParser>(filename);
  fileReader->parse();

  // validate file format
  std::vector<UnzippedFileMeta> &filesMeta = fileReader->filesMeta;
  try {
    modelname = std::filesystem::path(filesMeta[0].filename).begin()->string();
  } catch (...) {
    fprintf(stderr, "PyTorchModelManager: file format error\n");
    exit(0);
  }
  INFO("found model name: %s", modelname.c_str());
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

  // load data.pkl
  auto it = fileReader->nameIdxMap.find(modelname + "/data.pkl");
  assert(it != fileReader->nameIdxMap.end());
  INFO("found data.pkl, now unpickling%s", "...");
  UnzippedFileMeta &modelconf = filesMeta[it->second];
  unpickler                   = std::make_shared<LazyUnpickler>(modelconf, std::move(storageMap));
  unpickler->unpickle();

  loaded = true;
}