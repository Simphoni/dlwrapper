#include "lazy_unpickler.h"

uint32_t read_uint32(char *ptr) { return *(uint32_t *)ptr; }

ZipFileParser::ZipFileParser(std::string _filename) {
  filename = _filename;
  // do some checking
  assert(std::filesystem::exists(filename));
  assert(std::filesystem::is_regular_file(filename));
  filelen = std::filesystem::file_size(filename);
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1) {
    DEBUG("ZipFileParser cannot open file %s\n", filename.c_str());
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
  uint64_t n_entries_in_central_directory = *(uint16_t *)(ptr + 10);
  uint64_t zip64_n_entries_in_central_directory = 0;
  if (offset_of_start_of_central_directory == 0xFFFFFFFF) {
    ptr -= 20; // zip64 end of central directory locator
    assert(read_uint32(ptr) == kZip64EndOfCentralDirectoryLocatorSignature);
    int64_t offset_of_start_of_central_directory_record = *(uint64_t *)(ptr + 8);
    ptr = gigabuffer + offset_of_start_of_central_directory_record;
    assert(read_uint32(ptr) == kZip64EndOfCentralDirectoryRecordSignature);
    zip64_n_entries_in_central_directory = *(uint64_t *)(ptr + 32);
    offset_of_start_of_central_directory = *(uint64_t *)(ptr + 48);
    ptr = gigabuffer + offset_of_start_of_central_directory;
  } else {
    ptr = gigabuffer + offset_of_start_of_central_directory;
  }

  // use ptr to iterate over central directory headers
  for (uint32_t i = 0; i < n_entries_in_central_directory; i++) {
    auto *header = reinterpret_cast<CentralDirectoryHeader *>(ptr);
    assert(header->signature == kCentralDirectorySignature);
    if (header->compressed_size != 0xFFFFFFFF) {
      assert(header->compressed_size == header->uncompressed_size);
      ptr += sizeof(CentralDirectoryHeader);
      pickleFilesMeta.emplace_back(std::move(std::string(ptr, header->file_name_length)),
                                   gigabuffer + header->relative_offset_of_local_header,
                                   (uint64_t)header->uncompressed_size);
      ptr += header->file_name_length + header->extra_field_length + header->file_comment_length;
    } else {
      ptr += sizeof(CentralDirectoryHeader);
      std::string filename = std::string(ptr, header->file_name_length);
      ptr += header->file_name_length;
      auto *extra_field = reinterpret_cast<Zip64ExtendedExtraField *>(ptr);
      assert(extra_field->block_type == 0x0001);
      pickleFilesMeta.emplace_back(std::move(filename), gigabuffer + extra_field->relative_offset,
                                   extra_field->original_size);
      ptr += header->extra_field_length + header->file_comment_length;
    }
  }
}

ZipFileParser::~ZipFileParser() { munmap(gigabuffer, filelen); }