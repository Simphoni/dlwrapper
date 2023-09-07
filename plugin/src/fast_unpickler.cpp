#include "fast_unpickler.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// used by unpickler when processing byte stream
template <typename T> inline uint8_t read_uint8(T *ptr) noexcept { return *(uint8_t *)ptr; }
template <typename T> inline uint16_t read_uint16(T *ptr) noexcept { return *(uint16_t *)ptr; }
template <typename T> inline uint32_t read_uint32(T *ptr) { return *(uint32_t *)ptr; }
double read_big_endian_double(char *ptr) {
  union u_double {
    uint64_t data;
    double value;
  } u;
  u.data = __builtin_bswap64(*(uint64_t *)ptr);
  return u.value;
}

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
  INFO("zipfile size: %.3lfGB", filelen * 1e-9);

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

// ------------------- FastUnpickler -------------------

std::shared_ptr<internal_obj> internal_obj::query_dict(std::string key) const {
  assert(type == internal_obj::DICT || type == internal_obj::ORDERED_DICT);
  for (size_t i = 0; i < children.size(); i += 2) {
    if (children[i]->check_type<std::string>() &&
        children[i]->extract_basic_type<std::string>() == key)
      return children[i + 1];
  }
  return nullptr;
}

void internal_obj::read_all_tensors(
    std::map<std::string, std::shared_ptr<OriginTensor>> &tensors) const {
  if (type == internal_obj::LEAF)
    return;
  if (type == internal_obj::ORDERED_DICT) {
    for (size_t i = 0; i < children.size(); i += 2) {
      if (children[i]->check_type<std::string>() && children[i + 1]->is_tensor()) {
        auto key = children[i]->extract_basic_type<std::string>();
        auto val = children[i + 1]->tensor;
        tensors.insert(std::make_pair(key, val));
      }
    }
  }
  for (size_t i = 0; i < children.size(); i++)
    if (children[i]->type != internal_obj::LEAF)
      children[i]->read_all_tensors(tensors);
}

std::string internal_obj::get_type_name(object_t value) {
  const char *s = 0;
#define PROCESS_VAL(p)                                                                             \
  case (p):                                                                                        \
    s = #p;                                                                                        \
    break;
  switch (value) {
    PROCESS_VAL(LEAF);
    PROCESS_VAL(MODULE_ATTR);
    PROCESS_VAL(TUPLE);
    PROCESS_VAL(DICT);
    PROCESS_VAL(LIST);
    PROCESS_VAL(MARK);
    PROCESS_VAL(ORDERED_DICT);
    PROCESS_VAL(DUMMY);
  default:
    assert(0);
  }
#undef PROCESS_VAL
  return std::string(s);
}

std::string internal_obj::to_string() {
  if (type == LEAF) {
    if (data.index() == 1) {
      return "\"" + std::get<std::string>(data) + "\"";
    } else if (data.index() == 2) {
      return std::to_string(std::get<int64_t>(data));
    } else if (data.index() == 3) {
      return std::get<bool>(data) ? "True" : "False";
    } else if (data.index() == 4) {
      return std::to_string(std::get<double>(data));
    } else if (storage != nullptr) {
      return "/storage/";
    } else if (tensor != nullptr) {
      return tensor->to_string();
    } else if (pyobj != std::nullopt) {
      return "pyobject(" + py::cast<std::string>(py::str(pyobj.value())) + ")";
    } else {
      assert(0);
      return "";
    }
  } else {
    std::string ret = "(";
    ret += get_type_name(type) + " ";
    for (auto &child : children) {
      ret += child->to_string();
      ret += ", ";
    }
    if (attr != std::nullopt) {
      ret += "attr: {";
      for (auto &kv : attr.value()) {
        ret += kv.first + ": " + kv.second->to_string() + ", ";
      }
      ret += "}";
    }
    ret += ")";
    return ret;
  }
}

py::object internal_obj::to_pyobject() const {
  assert(type != MARK);
  if (type == NONE) {
    return py::none();
  } else if (type == LEAF) {
    if (data.index() == 1) {
      return py::cast(std::get<std::string>(data));
    } else if (data.index() == 2) {
      return py::cast(std::get<int64_t>(data));
    } else if (data.index() == 3) {
      return py::cast(std::get<bool>(data));
    } else if (data.index() == 4) {
      return py::cast(std::get<double>(data));
    } else if (pyobj != std::nullopt) {
      return pyobj.value();
    } else if (tensor != nullptr) {
      return py::cast(tensor);
    } else {
      assert(0);
    }
  } else if (type == MODULE_ATTR) {
    auto module = children[0]->extract_basic_type<std::string>();
    auto attr   = children[1]->extract_basic_type<std::string>();
    return py::module_::import(module.c_str()).attr(attr.c_str());
  } else if (type == TUPLE || type == LIST) {
    std::vector<py::object> vec;
    for (const auto &child : children) {
      vec.push_back(child->to_pyobject());
    }
    auto pylist = py::cast(vec);
    return type == TUPLE ? py::tuple(pylist) : pylist;
  } else if (type == DICT) {
    std::map<std::string, py::object> dict;
    for (size_t i = 0; i < children.size(); i += 2) {
      assert(children[i]->check_type<std::string>());
      dict.insert(std::make_pair(children[i]->extract_basic_type<std::string>(),
                                 children[i + 1]->to_pyobject()));
    }
    return py::cast(dict);
  } else if (type == ORDERED_DICT) {
    py::object pydict = py::module_::import("collections").attr("OrderedDict")();
    std::map<std::string, py::object> dict;
    for (size_t i = 0; i < children.size(); i += 2) {
      assert(children[i]->check_type<std::string>());
      dict.insert(std::make_pair(children[i]->extract_basic_type<std::string>(),
                                 children[i + 1]->to_pyobject()));
    }
    pydict.attr("update")(py::cast(dict));
    if (attr != std::nullopt) {
      std::map<std::string, py::object> pyattr;
      for (auto &[k, v] : attr.value()) {
        pyattr.insert(std::make_pair(k, v->to_pyobject()));
      }
      pydict.attr("__dict__").attr("update")(py::cast(pyattr));
    }
    return pydict;
  } else {
    fprintf(stderr, "not guarded");
    throw(std::runtime_error("Unsupported type"));
  }
}

void internal_obj::extend_attr(std::shared_ptr<internal_obj> dict) {
  assert(dict->type == DICT || dict->type == ORDERED_DICT);
  if (attr == std::nullopt) {
    attr = std::map<std::string, std::shared_ptr<object>>();
  }
  for (size_t i = 0; i < dict->children.size(); i += 2) {
    if (dict->children[i]->type == LEAF || dict->children[i]->data.index() == 1) {
      auto key   = std::get<std::string>(dict->children[i]->data);
      auto value = dict->children[i + 1];
      attr->insert(std::make_pair(key, value));
    }
  }
}

std::string FastUnpickler::readline() {
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

int FastUnpickler::find_mark() {
  assert(stack.size() > 0);
  int mark = stack.size() - 1;
  while (mark >= 0 && stack[mark]->type != object::object_t::MARK) {
    mark--;
  }
  assert(mark >= 0);
  return mark;
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

void FastUnpickler::pytorchPersistentId(std::shared_ptr<object> tuple) {
  assert(tuple->type == object::object_t::TUPLE);
  assert(tuple->children.size() == 5);
  assert(tuple->children[0]->extract_basic_type<std::string>() == "storage");
  auto storage_type = tuple->children[1];
  assert(storage_type->type == object::object_t::MODULE_ATTR);
  assert(storage_type->children[0]->extract_basic_type<std::string>() == "torch");
  auto [dtype, dsize] =
      parseStorageType(storage_type->children[1]->extract_basic_type<std::string>());
  std::string key      = tuple->children[2]->extract_basic_type<std::string>();
  std::string location = tuple->children[3]->extract_basic_type<std::string>();
  uint64_t numel       = tuple->children[4]->extract_basic_type<int64_t>();
  auto it              = storageMap.find(key);
  assert(it != storageMap.end());
  stack.emplace_back(std::make_shared<object>(
      std::make_shared<CachedStorage>(it->second, dtype, location, numel, dsize)));
}

std::shared_ptr<OriginTensor>
FastUnpickler::torch_util_rebuild_tensor_v2(std::shared_ptr<object> tuple) {
  auto storage        = tuple->children[0]->extract_storage();
  auto offset         = tuple->children[1]->extract_basic_type<int64_t>();
  auto size           = tuple->children[2]->extract_int_tuple<int64_t>();
  auto stride         = tuple->children[3]->extract_int_tuple<int64_t>();
  auto requires_grad  = tuple->children[4]->extract_basic_type<bool>();
  auto backward_hooks = tuple->children[5]; // backwards compatibility, expect empty ordered dict
  assert(backward_hooks->type == object::object_t::ORDERED_DICT);
  return std::make_shared<OriginTensor>(storage, offset, size, stride, requires_grad);
}

FastUnpickler::FastUnpickler(UnzippedFileMeta _file,
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
  actions[STOP] = [this]() {
    iterator++;
    stop_flag = true;
  };
  actions[BINFLOAT] = [this]() {
    iterator++;
    double value = read_big_endian_double(iterator);
    stack.emplace_back(std::make_shared<object>(value));
    iterator += 8;
  };
  actions[BININT] = [this]() {
    iterator++;
    stack.emplace_back(std::make_shared<object>((int64_t)(*(int *)iterator)));
    iterator += 4;
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
  actions[NONE] = [this]() {
    iterator++;
    stack.emplace_back(std::make_shared<object>(object::object_t::NONE));
  };
  actions[REDUCE] = [this]() {
    iterator++;
    auto arg = stack.back();
    stack.pop_back();
    auto func = stack.back();
    stack.pop_back();
    assert(func->type == object::object_t::MODULE_ATTR);
    std::string method_name = func->children[0]->extract_basic_type<std::string>() + "." +
                              func->children[1]->extract_basic_type<std::string>();
    if (method_name == "torch._utils._rebuild_tensor_v2") {
      stack.emplace_back(std::make_shared<object>(torch_util_rebuild_tensor_v2(arg)));
    } else if (method_name == "collections.OrderedDict") {
      stack.emplace_back(std::make_shared<object>(object::object_t::ORDERED_DICT, arg->children));
    } else if (method_name == "_codecs.encode") { // ignore encoding operations
      stack.emplace_back(arg->children[0]);
    } else {
      py::object pyfunc = func->to_pyobject();
      py::object pyargs = arg->to_pyobject();
      stack.emplace_back(std::make_shared<object>(pyfunc(*pyargs)));
    }
  };
  actions[BINPERSID] = [this]() {
    iterator++;
    auto ptr = stack.back();
    stack.pop_back();
    pytorchPersistentId(ptr);
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
  actions[APPEND] = [this]() {
    iterator++;
    assert(stack.size() >= 2);
    auto obj = stack.back();
    stack.pop_back();
    assert(stack.back()->type == object::object_t::LIST);
    stack.back()->children.emplace_back(std::move(obj));
  };
  actions[BUILD] = [this]() {
    iterator++;
    auto state = stack.back();
    stack.pop_back();
    auto instance = stack.back();
    if (instance->type == object::object_t::ORDERED_DICT) {
      instance->extend_attr(state);
    }
  };
  actions[GLOBAL] = [this]() {
    iterator++;
    auto module = readline();
    auto name   = readline();
    std::vector<std::shared_ptr<object>> vec(
        {std::make_shared<object>(std::move(module)), std::make_shared<object>(std::move(name))});
    stack.emplace_back(std::make_shared<object>(object::object_t::MODULE_ATTR, std::move(vec)));
  };
  actions[APPENDS] = [this]() {
    iterator++;
    int mark = find_mark();
    assert(mark > 0);
    assert(stack[mark - 1]->type == object::object_t::LIST);
    std::vector<std::shared_ptr<object>> &vec = stack[mark - 1]->children;
    vec.reserve(vec.size() + stack.size() - mark - 1);
    vec.insert(vec.end(), stack.begin() + mark + 1, stack.end());
    stack.erase(stack.begin() + mark, stack.end());
  };
  actions[BINGET] = [this]() {
    iterator++;
    uint32_t idx = read_uint8(iterator);
    iterator++;
    stack.emplace_back(memo[idx]);
  };
  actions[LONG_BINGET] = [this]() {
    iterator++;
    uint32_t idx = read_uint32(iterator);
    iterator += 4;
    stack.emplace_back(memo[idx]);
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
  actions[SETITEM] = [this]() {
    iterator++;
    assert(stack.size() >= 3);
    auto val = stack.back();
    stack.pop_back();
    auto key = stack.back();
    stack.pop_back();
    assert(stack.back()->type == object::object_t::DICT ||
           stack.back()->type == object::object_t::ORDERED_DICT);
    stack.back()->children.emplace_back(std::move(key));
    stack.back()->children.emplace_back(std::move(val));
  };
  actions[TUPLE] = [this]() {
    iterator++;
    int mark = find_mark();
    std::vector<std::shared_ptr<object>> vec(stack.begin() + mark + 1, stack.end());
    stack.erase(stack.begin() + mark, stack.end());
    stack.emplace_back(std::make_shared<object>(object::object_t::TUPLE, std::move(vec)));
  };
  actions[SETITEMS] = [this]() {
    iterator++;
    int mark = find_mark();
    assert(stack[mark - 1]->type == object::object_t::DICT ||
           stack[mark - 1]->type == object::object_t::ORDERED_DICT);
    std::vector<std::shared_ptr<object>> &vec = stack[mark - 1]->children;
    vec.reserve(vec.size() + stack.size() - mark - 1);
    vec.insert(vec.end(), stack.begin() + mark + 1, stack.end());
    stack.erase(stack.begin() + mark, stack.end());
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
  actions[TUPLE1] = [this]() {
    iterator++;
    assert(stack.size() >= 1);
    std::vector<std::shared_ptr<object>> vec(stack.end() - 1, stack.end());
    stack.erase(stack.end() - 1, stack.end());
    stack.emplace_back(std::make_shared<object>(object::object_t::TUPLE, std::move(vec)));
  };
  actions[TUPLE2] = [this]() {
    iterator++;
    assert(stack.size() >= 2);
    std::vector<std::shared_ptr<object>> vec(stack.end() - 2, stack.end());
    stack.erase(stack.end() - 2, stack.end());
    stack.emplace_back(std::make_shared<object>(object::object_t::TUPLE, std::move(vec)));
  };
  actions[TUPLE3] = [this]() {
    iterator++;
    assert(stack.size() >= 3);
    std::vector<std::shared_ptr<object>> vec(stack.end() - 3, stack.end());
    stack.erase(stack.end() - 3, stack.end());
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
  actions[LONG1] = [this]() {
    iterator++;
    uint8_t len = read_uint8(iterator);
    assert(len <= 8);
    iterator++;
    int64_t val = 0;
    for (int i = 0; i < len; i++) {
      val = val << 8 | read_uint8(iterator);
      iterator++;
    }
    stack.emplace_back(std::make_shared<object>(val));
  };
}

std::shared_ptr<internal_obj> FastUnpickler::unpickle() {
  if (unpickled)
    return stack[0];
  stop_flag = false;
  iterator  = file.buffer;
  stack.reserve(256);
  assert(read_uint8(iterator) == 0x80);
  bool print_debug_info = false;
  while (!stop_flag) {
    static int numit = 0;
    if (print_debug_info) {
      fprintf(stderr, "(%d %02x) ", numit, read_uint8(iterator));
    }

    actions[read_uint8(iterator)]();

    if (print_debug_info) {
      numit++;
      if (numit >= 400)
        break;
    }
  }
  unpickled = true;
  if (print_debug_info) {
    puts("");
    INFO("stack size: %ld", stack.size());
    for (auto i : stack)
      INFO("%s", i->to_string().c_str());
  }
  assert(stack.size() == 1);
  return stack[0];
}