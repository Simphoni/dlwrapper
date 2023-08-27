#include "misc.h"
#include "tensor.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cassert>
#include <filesystem>
#include <functional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

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

class LazyUnpickler {
  // A unpickler that keeps references to large objects via file mapping. It
  // is part of a hierarchical model mananagement module.
private:
  static const int HIGHEST_PROTOCOL = 2;

  UnzippedFileMeta file;
  std::unordered_map<std::string, char *> storageMap;
  bool unpickled, stop_flag;
  std::function<void()> actions[256];
  char *iterator;

  struct object {
    enum object_t : uint8_t {
      LEAF = 1,
      MODULE_ATTR,
      TUPLE,
      DICT,
      LIST,
      MARK,
      DUMMY,
    };
    object_t type;
    std::variant<std::monostate, std::string, int64_t, bool> data; // for leaf nodes
    std::shared_ptr<Storage> storage;
    std::vector<std::shared_ptr<object>> children;

    object(std::shared_ptr<Storage> storage) : type(LEAF), storage(storage) {}
    object(object_t type) : type(type) {}
    object(bool data) : type(LEAF), data(data) {}
    object(int64_t data) : type(LEAF), data(data) {}
    object(std::string data) : type(LEAF), data(data) {}
    object(object_t type, std::vector<std::shared_ptr<object>> children)
        : type(type), children(children) {}

    bool is_storage() {
      // storage nodes are special, they are leaf nodes but are not basic types.
      return type == LEAF && data.index() == 0 && storage != nullptr;
    }

    bool is_empty() { return type != LEAF && children.size() == 0; }

    std::string extract_string() {
      assert(type == LEAF && data.index() == 1);
      return std::get<std::string>(data);
    }

    int64_t extract_int() {
      assert(type == LEAF && data.index() == 2);
      return std::get<int64_t>(data);
    }

    std::string get_type_name(object_t value) {
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
        PROCESS_VAL(DUMMY);
      default:
        assert(0);
      }
#undef PROCESS_VAL
      return std::string(s);
    }

    std::string to_string() {
      if (type == LEAF) {
        if (data.index() == 1) {
          return std::get<std::string>(data);
        } else if (data.index() == 2) {
          return std::to_string(std::get<int64_t>(data));
        } else if (data.index() == 3) {
          return std::get<bool>(data) ? "True" : "False";
        } else {
          return "/storage/";
        }
      } else {
        std::string ret = "(";
        ret += get_type_name(type) + " ";
        for (auto &child : children) {
          ret += child->to_string();
          ret += ", ";
        }
        ret += ")";
        return ret;
      }
    }
  };

  std::vector<std::shared_ptr<object>> stack;
  std::unordered_map<uint32_t, std::shared_ptr<object>> memo;

  enum opcode : uint8_t {
    MARK        = 0x28,
    EMPTY_TUPLE = 0x29,
    BININT1     = 0x4b,
    BININT2     = 0x4d,
    BINPERSID   = 0x51,
    REDUCE      = 0x52,
    BINUNICODE  = 0x58,
    EMPTY_LIST  = 0x5d,
    GLOBAL      = 0x63,
    BINPUT      = 0x71,
    LONG_BINPUT = 0x72,
    TUPLE       = 0x74,
    EMPTY_DICT  = 0x7d,
    PROTO       = 0x80,
    TUPLE2      = 0x86, // Protocol 2
    NEWTRUE     = 0x88,
    NEWFALSE    = 0x89,
  };

  // pytorch rewrites persistent_id method to enable tensor loading during unpickling
  void pytorchPersistentId(std::shared_ptr<object> tuple);

public:
  LazyUnpickler() = delete;
  LazyUnpickler(UnzippedFileMeta _file, std::unordered_map<std::string, char *> _storageMap);

  std::string readline();
  void unpickle();
};

class PyTorchModelManager {
  // Support for PyTorch models
private:
  std::string filename;
  std::string modelname;
  std::shared_ptr<ZipFileParser> fileReader;
  std::shared_ptr<LazyUnpickler> unpickler;
  bool loaded;

public:
  PyTorchModelManager() = default;
  PyTorchModelManager(std::string filename) : filename(filename), loaded(false) {}

  void load();
};