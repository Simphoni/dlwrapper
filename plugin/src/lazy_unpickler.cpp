#include "lazy_unpickler.h"
#include <unordered_map>

// used by unpickle when processing byte stream
template <typename T> inline uint8_t read_uint8(T *ptr) noexcept { return *(uint8_t *)ptr; }

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
    name_idx_map[filename] = i;
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
    uint64_t idx         = name_idx_map[filename];
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

LazyUnpickler::LazyUnpickler(UnzippedFileMeta _file) : file(_file) {
  unpickled     = false;
  actions[MARK] = [this]() {
    iterator++;
    stack.emplace_back(std::make_shared<object>(object::object_t::MARK));
  };
  actions[EMPTY_TUPLE] = [this]() {
    iterator++;
    stack.emplace_back(std::make_shared<object>(object::object_t::TUPLE));
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
  actions[EMPTY_DICT] = [this]() {
    iterator++;
    stack.emplace_back(std::make_shared<object>(object::object_t::DICT));
  };
  actions[PROTO] = [this]() {
    iterator++;
    assert(read_uint8(iterator) <= HIGHEST_PROTOCOL);
    iterator++;
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
    printf("%02x\n", read_uint8(iterator));
    actions[read_uint8(iterator)]();

    if (true) {
      numit++;
      if (numit >= 4)
        break;
    }
  }
  INFO("stack size: %ld", stack.size());
  for (auto i : stack)
    INFO("%s", i->to_string().c_str());
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
  for (auto file : filesMeta) {
    assert(file.filename.length() > 0);
    auto pth = std::filesystem::path(file.filename);
    assert(pth.begin()->string() == modelname);
  }

  // load data.pkl
  auto it = fileReader->name_idx_map.find(modelname + "/data.pkl");
  assert(it != fileReader->name_idx_map.end());
  INFO("found data.pkl, now unpickling%s", "...");
  UnzippedFileMeta &modelconf = filesMeta[it->second];
  unpickler                   = std::make_shared<LazyUnpickler>(modelconf);
  unpickler->unpickle();

  loaded = true;
}