#pragma once

#include "misc.h"
#include "tensor.h"
#include <fcntl.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cassert>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace py = pybind11;

void initalizePyInterp();

enum ZipSignatures : uint32_t {
  kLocalFileHeaderSignature                   = 0x04034b50,
  kDataDescriptorSignature                    = 0x08074b50,
  kCentralDirectorySignature                  = 0x02014b50,
  kZip64EndOfCentralDirectoryRecordSignature  = 0x06064b50,
  kZip64EndOfCentralDirectoryLocatorSignature = 0x07064b50,
  kEndOfCentralDirectorySignature             = 0x06054b50,
};

#pragma pack(2)
struct CentralDirectoryHeader {
  uint32_t signature;
  uint16_t version_made_by;
  uint16_t version_needed_to_extract;
  uint16_t general_purpose_bit_flag;
  uint16_t compression_method;
  uint16_t last_mod_file_time;
  uint16_t last_mod_file_date;
  uint32_t crc32;
  uint32_t compressed_size;
  uint32_t uncompressed_size;
  uint16_t file_name_length;
  uint16_t extra_field_length;
  uint16_t file_comment_length;
  uint16_t disk_number_start;
  uint16_t internal_file_attributes;
  uint32_t external_file_attributes;
  uint32_t relative_offset_of_local_header;
};

#pragma pack(2)
struct Zip64ExtendedExtraField {
  uint16_t block_type;
  uint16_t block_size;
  uint64_t original_size;
  uint64_t compressed_size;
  uint64_t relative_offset;
  uint32_t disk_start_number;
};

#pragma pack(2)
struct LocalFileHeader {
  uint32_t signature;
  uint16_t version_needed_to_extract;
  uint16_t general_purpose_bit_flag;
  uint16_t compression_method;
  uint16_t last_mod_file_time;
  uint16_t last_mod_file_date;
  uint32_t crc32;
  uint32_t compressed_size;
  uint32_t uncompressed_size;
  uint16_t file_name_length;
  uint16_t extra_field_length;
};

struct UnzippedFileMeta {
  std::string filename;
  char *buffer;
  uint64_t size;
  UnzippedFileMeta() = default;
  UnzippedFileMeta(std::string _filename, char *_buffer, uint64_t _size)
      : filename(_filename), buffer(_buffer), size(_size) {}
};

class ZipFileParser {
  // PyTorch model saving function uses zip file format to store the model.
  // Assume file is uncompressed, unencrypted, unsplitted, unspanned, and trusted.
  // Check only signatures to ensure we are parsing correctly.
  // The parser shall keep the metadata for extensibility.
  //! Remember we are dealing with a GIGANTIC file
private:
  std::string filename;
  char *gigabuffer;
  uint64_t filelen;
  bool parsed;

public:
  std::vector<UnzippedFileMeta> filesMeta;
  std::unordered_map<std::string, uint64_t> nameIdxMap;

  ZipFileParser() = default;
  ZipFileParser(std::string _filename)
      : filename(_filename), gigabuffer(nullptr), filelen(0), parsed(false) {}
  ~ZipFileParser();
  void parse();
};

class __attribute__((visibility("default"))) FastUnpickler {
  // A unpickler that keeps references to large objects via file mapping. It
  // is part of a hierarchical model mananagement module.
private:
  static const int HIGHEST_PROTOCOL = 2;

  UnzippedFileMeta file;
  std::unordered_map<std::string, char *> storageMap;
  bool unpickled, stop_flag;
  std::function<void()> actions[256];
  char *iterator;

public:
  struct object {
    enum object_t : uint8_t {
      NONE = 1,
      LEAF,
      MODULE_ATTR,
      TUPLE,
      LIST,
      DICT,
      ORDERED_DICT,
      MARK,
      DUMMY,
    };
    object_t type;
    std::variant<std::monostate, std::string, int64_t, bool, double> data; // for leaf nodes
    std::shared_ptr<CachedStorage> storage;
    std::shared_ptr<OriginTensor> tensor;
    std::vector<std::shared_ptr<object>> children;
    std::optional<std::map<std::string, std::shared_ptr<object>>> attr;
    std::optional<py::object> pyobj;

    object(object_t type) : type(type) {}
    object(bool data) : type(LEAF), data(data) {}
    object(int64_t data) : type(LEAF), data(data) {}
    object(std::string data) : type(LEAF), data(data) {}
    object(double data) : type(LEAF), data(data) {}
    object(std::shared_ptr<CachedStorage> storage) : type(LEAF), storage(storage) {}
    object(std::shared_ptr<OriginTensor> tensor) : type(LEAF), tensor(tensor) {}
    object(py::object obj) : type(LEAF), pyobj(obj) {}
    object(object_t type, std::vector<std::shared_ptr<object>> children)
        : type(type), children(children) {}

    void extend_attr(std::shared_ptr<object> dict);
    bool is_storage() const { return type == LEAF && data.index() == 0 && storage != nullptr; }
    bool is_tensor() const { return type == LEAF && data.index() == 0 && tensor != nullptr; }
    bool is_empty() const { return type != LEAF && children.size() == 0; }

    template <typename T> bool check_type() const {
      return type == LEAF && std::holds_alternative<T>(data);
    }
    template <typename T> T extract_basic_type() const {
      assert(type == LEAF && std::holds_alternative<T>(data));
      return std::get<T>(data);
    }

    std::shared_ptr<CachedStorage> extract_storage() const {
      assert(type == LEAF && data.index() == 0);
      return storage;
    }

    template <typename T> std::vector<T> extract_int_tuple() const {
      assert(type == TUPLE);
      std::vector<T> ret;
      for (const auto &child : children) {
        ret.push_back((T)child->extract_basic_type<int64_t>());
      }
      return ret;
    }

    // simulated dict query
    std::shared_ptr<object> query_dict(std::string key) const;
    void read_all_tensors(std::map<std::string, std::shared_ptr<OriginTensor>> &tensors) const;

    // export to human readable / python object
    std::string get_type_name(object_t value);
    std::string to_string();
    py::object to_pyobject() const;
  };

private:
  std::vector<std::shared_ptr<object>> stack;
  std::unordered_map<uint32_t, std::shared_ptr<object>> memo;

  enum opcode : uint8_t {
    MARK        = 0x28,
    EMPTY_TUPLE = 0x29,
    STOP        = 0x2e,
    BINFLOAT    = 0x47,
    BININT      = 0x4a,
    BININT1     = 0x4b,
    BININT2     = 0x4d,
    NONE        = 0x4e,
    BINPERSID   = 0x51,
    REDUCE      = 0x52,
    BINUNICODE  = 0x58,
    EMPTY_LIST  = 0x5d,
    APPEND      = 0x61,
    BUILD       = 0x62,
    GLOBAL      = 0x63,
    APPENDS     = 0x65,
    BINGET      = 0x68,
    LONG_BINGET = 0x6a,
    BINPUT      = 0x71,
    LONG_BINPUT = 0x72,
    SETITEM     = 0x73,
    TUPLE       = 0x74,
    SETITEMS    = 0x75,
    EMPTY_DICT  = 0x7d,
    PROTO       = 0x80,
    TUPLE1      = 0x85, // Protocol 2
    TUPLE2      = 0x86,
    TUPLE3      = 0x87,
    NEWTRUE     = 0x88,
    NEWFALSE    = 0x89,
    LONG1       = 0x8a,
  };

  int find_mark();

  // pytorch rewrites persistent_id method to enable tensor loading during unpickling
  void pytorchPersistentId(std::shared_ptr<object> tuple);
  std::shared_ptr<OriginTensor> torch_util_rebuild_tensor_v2(std::shared_ptr<object> tuple);

  friend class PyTorchModelManager;

public:
  FastUnpickler() = delete;
  FastUnpickler(UnzippedFileMeta _file, std::unordered_map<std::string, char *> _storageMap);

  std::string readline();
  std::shared_ptr<object> unpickle();
};

using internal_obj = FastUnpickler::object;