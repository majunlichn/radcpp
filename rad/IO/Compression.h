#pragma once

#include <rad/IO/File.h>
#include <rad/Container/Span.h>
#include <mz.h>

namespace rad
{

struct ZipOptions {
    bool        include_path = false;   // Include full path of files
    // 1: Compress faster;
    // 9: Compress better;
    int16_t     compress_level = MZ_COMPRESS_LEVEL_DEFAULT;
    // BZIP2; LZMA; ZSTD; XZ; AES;
    uint8_t     compress_method = MZ_COMPRESS_METHOD_DEFLATE;
    bool        overwrite = false;      // Overwrite existing files
    bool        append = false;         // Append to existing zip file
    int64_t     disk_size = false;      // Disk size in bytes
    bool        follow_links = false;   // Follow symbolic links
    bool        store_links = false;    // Store symbolic links
    bool        zip_cd = false;         // Zip central directory
    int32_t     encoding = 65001;       // File names use UTF-8 encoding (or specified codepage)
    bool        verbose = false;        // Verbose info
    uint8_t     aes;                    // AES encryption
};

bool ZipCompress(const char* path, const char* password, ZipOptions* options, Span<std::string> args);
bool ZipDecompress(const char* path, const char* pattern, const char* destination, const char* password, ZipOptions* options);

} // namespace rad
