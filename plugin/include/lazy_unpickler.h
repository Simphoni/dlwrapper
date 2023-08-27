#include "misc.h"
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
  std::unordered_map<std::string, uint64_t> name_idx_map;

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
  bool unpickled, stop_flag;
  std::function<void()> actions[256];
  char *iterator;

  struct object {
    enum object_t : uint8_t {
      LEAF        = 1,
      MODULE_ATTR = 2,
      TUPLE       = 3,
      DICT        = 4,
      MARK        = 5,
    };
    object_t type;
    std::variant<std::string, int64_t> data; // for leaf nodes
    std::vector<std::shared_ptr<object>> children;

    object(object_t _type) : type(_type) {}
    object(int64_t _data) : type(LEAF), data(_data) {}
    object(std::string _data) : type(LEAF), data(_data) {}
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
        PROCESS_VAL(MARK);
      default:
        assert(0);
      }
#undef PROCESS_VAL
      return std::string(s);
    }
    object(object_t _type, std::vector<std::shared_ptr<object>> _children)
        : type(_type), children(_children) {}
    std::string to_string() {
      if (type == LEAF) {
        if (data.index() == 0)
          return std::get<std::string>(data);
        else
          return std::to_string(std::get<int64_t>(data));
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
    REDUCE      = 0x52,
    GLOBAL      = 0x63,
    BINPUT      = 0x71,
    EMPTY_DICT  = 0x7d,
    PROTO       = 0x80,
  };

public:
  LazyUnpickler() = delete;
  LazyUnpickler(UnzippedFileMeta _file);

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
  PyTorchModelManager(std::string _filename) : filename(_filename), loaded(false) {}

  void load();
};