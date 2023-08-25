#include "misc.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cassert>
#include <filesystem>
#include <string>
#include <vector>

enum ZipSignatures : uint32_t {
  kLocalFileHeaderSignature = 0x04034b50,
  kDataDescriptorSignature = 0x08074b50,
  kCentralDirectorySignature = 0x02014b50,
  kZip64EndOfCentralDirectoryRecordSignature = 0x06064b50,
  kZip64EndOfCentralDirectoryLocatorSignature = 0x07064b50,
  kEndOfCentralDirectorySignature = 0x06054b50,
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

struct PickleFileMeta {
  std::string filename;
  char *buffer;
  uint64_t size;
  PickleFileMeta(std::string _filename, char *_buffer, uint64_t _size)
      : filename(_filename), buffer(_buffer), size(_size) {}
};

class ZipFileParser {
  // PyTorch model saving function uses zip file format to store the model.
  // Assume file is uncompressed, unencrypted, unsplitted, unspanned, and trusted.
  // Check only signatures to ensure we are parsing correctly.
  //! Remember we are dealing with a GIGANTIC file
private:
  // file metadata
  std::string filename;
  char *gigabuffer; // buffer file mapping
  uint64_t filelen;

public:
  std::vector<PickleFileMeta> pickleFilesMeta;

  ZipFileParser() = delete;
  ZipFileParser(std::string _filename);
  ~ZipFileParser();
};

class LazyUnpickler {
  // A unpickler that keeps references to large objects via file mapping. It
  // is part of a hierarchical model mananagement module.
private:
  static const int MAX_BUF_SIZE = 1 << 18;
  char buf[MAX_BUF_SIZE];
};