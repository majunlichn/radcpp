#pragma once

#include <radcpp/Core/Platform.h>
#include <radcpp/Container/Span.h>
#include <radcpp/IO/File.h>
#include <mz.h>
#include <mz_zip.h>

namespace rad
{

// https://github.com/zlib-ng/minizip-ng
class ZipWriter
{
public:
    ZipWriter();
    ~ZipWriter();

    void SetPassword(std::string_view password);
    void SetComment(std::string_view comment);
    void EnableAES(bool aes = true);
    // MZ_COMPRESS_METHOD_STORE;DEFLATE;BZIP2;LZMA;ZSTD;XZ;AES;
    void SetCompressMethod(uint16_t method);
    // 1: Compress faster;
    // 9: Compress better;
    void SetCompressLevel(int16_t level);
    void SetFollowLinks(bool followLinks = true);
    void SetStoreLinks(bool storeLinks = true);
    void SetZipCentralDir(bool zipcd = true);

    bool Open(std::string_view fileName, int64_t diskSize, bool append);
    void Close();
    bool AddToArchive(Span<std::string> paths);

private:
    void* m_handle = nullptr;

}; // class ZipWriter

// https://github.com/zlib-ng/minizip-ng
class ZipReader
{
public:
    ZipReader();
    ~ZipReader();

    void SetPattern(std::string_view pattern, bool ignoreCase = true);
    void SetPassword(std::string_view password);
    void SetEncoding(int32_t encoding);

    bool OpenFile(std::string_view fileName);
    bool OpenFileInMemory(std::string_view fileName);
    bool OpenBuffer(uint8_t* buffer, int32_t sizeInBytes, bool copy);
    void Close();

    bool GotoFirstEntry();
    bool GotoNextEntry();
    bool LocateEntry(std::string_view fileName, bool ignoreCase = true);
    bool OpenEntry();
    bool CloseEntry();
    bool ReadEntry(void* buffer, int32_t sizeInBytes);
    mz_zip_file* GetEntryInfo();
    bool IsEntryDirectory();
    bool SaveEntryToFile(std::string_view fileName);
    int32_t GetEntryBufferSize();
    bool SaveEntryToMemory(void* buffer, int32_t sizeInBytes);
    std::vector<uint8_t> SaveEntryToMemory();

    bool ExtractAll(std::string_view destination);

private:
    void* m_handle = nullptr;

}; // class ZipReader

} // namespace rad
