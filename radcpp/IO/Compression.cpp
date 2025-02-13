#include <radcpp/IO/Compression.h>
#include <radcpp/IO/Logging.h>

#include "mz.h"
#include "mz_os.h"
#include "mz_strm.h"
#include "mz_strm_buf.h"
#include "mz_strm_split.h"
#include "mz_zip.h"
#include "mz_zip_rw.h"

namespace rad
{

std::string mz_get_error_string(int32_t err)
{
    switch (err)
    {
    case MZ_OK: return "MZ_OK";
    case MZ_STREAM_ERROR: return "MZ_STREAM_ERROR";
    case MZ_DATA_ERROR: return "MZ_DATA_ERROR";
    case MZ_MEM_ERROR: return "MZ_MEM_ERROR";
    case MZ_BUF_ERROR: return "MZ_BUF_ERROR";
    case MZ_VERSION_ERROR: return "MZ_VERSION_ERROR";
    case MZ_END_OF_LIST:    return "MZ_END_OF_LIST";
    case MZ_END_OF_STREAM:  return "MZ_END_OF_STREAM";
    case MZ_PARAM_ERROR: return "MZ_PARAM_ERROR";
    case MZ_FORMAT_ERROR: return "MZ_FORMAT_ERROR";
    case MZ_INTERNAL_ERROR: return "MZ_INTERNAL_ERROR";
    case MZ_CRC_ERROR: return "MZ_CRC_ERROR";
    case MZ_CRYPT_ERROR: return "MZ_CRYPT_ERROR";
    case MZ_EXIST_ERROR: return "MZ_EXIST_ERROR";
    case MZ_PASSWORD_ERROR: return "MZ_PASSWORD_ERROR";
    case MZ_SUPPORT_ERROR: return "MZ_SUPPORT_ERROR";
    case MZ_HASH_ERROR: return "MZ_HASH_ERROR";
    case MZ_OPEN_ERROR: return "MZ_OPEN_ERROR";
    case MZ_CLOSE_ERROR: return "MZ_CLOSE_ERROR";
    case MZ_SEEK_ERROR: return "MZ_SEEK_ERROR";
    case MZ_TELL_ERROR: return "MZ_TELL_ERROR";
    case MZ_READ_ERROR: return "MZ_READ_ERROR";
    case MZ_WRITE_ERROR: return "MZ_WRITE_ERROR";
    case MZ_SIGN_ERROR: return "MZ_SIGN_ERROR";
    case MZ_SYMLINK_ERROR: return "MZ_SYMLINK_ERROR";
    }
    return std::to_string(err);
}

ZipWriter::ZipWriter()
{
    m_handle = mz_zip_writer_create();
    SetCompressMethod(MZ_COMPRESS_METHOD_DEFLATE);
    SetCompressLevel(MZ_COMPRESS_LEVEL_DEFAULT);
}

ZipWriter::~ZipWriter()
{
    if (m_handle)
    {
        Close();
        mz_zip_writer_delete(&m_handle);
    }
}

void ZipWriter::SetPassword(std::string_view password)
{
    mz_zip_writer_set_password(m_handle, password.data());
}

void ZipWriter::SetComment(std::string_view comment)
{
    mz_zip_writer_set_comment(m_handle, comment.data());
}

void ZipWriter::EnableAES(bool aes)
{
    mz_zip_writer_set_aes(m_handle, aes ? 1 : 0);
}

void ZipWriter::SetCompressMethod(uint16_t method)
{
    mz_zip_writer_set_compress_method(m_handle, method);
}

void ZipWriter::SetCompressLevel(int16_t level)
{
    mz_zip_writer_set_compress_level(m_handle, level);
}

void ZipWriter::SetFollowLinks(bool followLinks)
{
    mz_zip_writer_set_follow_links(m_handle, followLinks ? 1 : 0);
}

void ZipWriter::SetStoreLinks(bool storeLinks)
{
    mz_zip_writer_set_store_links(m_handle, storeLinks ? 1 : 0);
}

void ZipWriter::SetZipCentralDir(bool zipcd)
{
    mz_zip_writer_set_zip_cd(m_handle, zipcd ? 1 : 0);
}

bool ZipWriter::Open(std::string_view fileName, int64_t diskSize, bool append)
{
    int32_t err = mz_zip_writer_open_file(m_handle, fileName.data(),
        diskSize, append ? 1 : 0);
    if (err == MZ_OK)
    {
        return true;
    }
    else
    {
        LOG_DEFAULT(err, "mz_zip_writer_open_file(path=\"{}\"): {}",
            fileName, mz_get_error_string(err));
        return false;
    }
}

void ZipWriter::Close()
{
    if (mz_zip_writer_is_open(m_handle))
    {
        int32_t err = mz_zip_writer_close(m_handle);
        if (err != MZ_OK) {
            LOG_DEFAULT(err, "mz_zip_writer_close: {}",
                mz_get_error_string(err));
        }
    }
}

bool ZipWriter::AddToArchive(Span<std::string> paths)
{
    bool result = true;
    for (const auto& path : paths)
    {
        int32_t err = mz_zip_writer_add_path(m_handle, path.data(),
            nullptr, 0, 1);
        if (err != MZ_OK)
        {
            LOG_DEFAULT(err, "mz_zip_writer_add_path(path=\"{}\"): {}",
                path, mz_get_error_string(err));
            result = false;
        }
    }
    return result;
}

ZipReader::ZipReader()
{
    m_handle = mz_zip_reader_create();
    SetEncoding(65001);
}

ZipReader::~ZipReader()
{
    if (m_handle)
    {
        Close();
        mz_zip_reader_delete(&m_handle);
    }
}

void ZipReader::SetPattern(std::string_view pattern, bool ignoreCase)
{
    mz_zip_reader_set_pattern(m_handle, pattern.data(), ignoreCase ? 1 : 0);
}

void ZipReader::SetPassword(std::string_view password)
{
    mz_zip_reader_set_password(m_handle, password.data());
}

void ZipReader::SetEncoding(int32_t encoding)
{
    mz_zip_reader_set_encoding(m_handle, encoding);
}

bool ZipReader::OpenFile(std::string_view fileName)
{
    int32_t err = mz_zip_reader_open_file(m_handle, fileName.data());

    if (err == MZ_OK)
    {
        return true;
    }
    else
    {
        LOG_DEFAULT(err, "mz_zip_reader_open_file(path=\"{}\"): {}",
            fileName, mz_get_error_string(err));
        return false;
    }
}

bool ZipReader::OpenFileInMemory(std::string_view fileName)
{
    int32_t err = mz_zip_reader_open_file_in_memory(m_handle, fileName.data());

    if (err == MZ_OK)
    {
        return true;
    }
    else
    {
        LOG_DEFAULT(err, "mz_zip_reader_open_file_in_memory(path=\"{}\"): {}",
            fileName, mz_get_error_string(err));
        return false;
    }
}

bool ZipReader::OpenBuffer(uint8_t* buffer, int32_t sizeInBytes, bool copy)
{
    int32_t err = mz_zip_reader_open_buffer(m_handle, buffer, sizeInBytes, copy ? 1 : 0);
    if (err == MZ_OK)
    {
        return true;
    }
    else
    {
        LOG_DEFAULT(err, "mz_zip_reader_open_buffer: {}",
            mz_get_error_string(err));
        return false;
    }
}

void ZipReader::Close()
{
    if (mz_zip_reader_is_open(m_handle))
    {
        int32_t err = mz_zip_reader_close(m_handle);
        if (err != MZ_OK)
        {
            LOG_DEFAULT(err, "mz_zip_reader_close: {}",
                mz_get_error_string(err));
        }
    }
}

bool ZipReader::GotoFirstEntry()
{
    int32_t err = mz_zip_reader_goto_first_entry(m_handle);
    return (err == MZ_OK);
}

bool ZipReader::GotoNextEntry()
{
    int32_t err = mz_zip_reader_goto_next_entry(m_handle);
    return (err == MZ_OK);
}

bool ZipReader::LocateEntry(std::string_view fileName, bool ignoreCase)
{
    int32_t err = mz_zip_reader_locate_entry(m_handle, fileName.data(), ignoreCase ? 1 : 0);
    return (err == MZ_OK);
}

bool ZipReader::OpenEntry()
{
    int32_t err = mz_zip_reader_entry_open(m_handle);
    return (err == MZ_OK);
}

bool ZipReader::CloseEntry()
{
    int32_t err = mz_zip_reader_entry_close(m_handle);
    return (err == MZ_OK);
}

bool ZipReader::ReadEntry(void* buffer, int32_t sizeInBytes)
{
    int32_t err = mz_zip_reader_entry_read(m_handle, buffer, sizeInBytes);
    return (err == MZ_OK);
}

mz_zip_file* ZipReader::GetEntryInfo()
{
    mz_zip_file* info = nullptr;
    int32_t err = mz_zip_reader_entry_get_info(m_handle, &info);
    if (err == MZ_OK)
    {
        return info;
    }
    else
    {
        return nullptr;
    }
}

bool ZipReader::IsEntryDirectory()
{
    return mz_zip_reader_entry_is_dir(m_handle);
}

bool ZipReader::SaveEntryToFile(std::string_view fileName)
{
    int32_t err = mz_zip_reader_entry_save_file(m_handle, fileName.data());
    return (err == MZ_OK);
}

int32_t ZipReader::GetEntryBufferSize()
{
    return mz_zip_reader_entry_save_buffer_length(m_handle);
}

bool ZipReader::SaveEntryToMemory(void* buffer, int32_t sizeInBytes)
{
    int32_t err = mz_zip_reader_entry_save_buffer(m_handle, buffer, sizeInBytes);
    return (err == MZ_OK);
}

std::vector<uint8_t> ZipReader::SaveEntryToMemory()
{
    std::vector<uint8_t> buffer(GetEntryBufferSize());
    if (SaveEntryToMemory(buffer.data(), static_cast<int32_t>(buffer.size())))
    {
        return buffer;
    }
    else
    {
        return {};
    }
}

bool ZipReader::ExtractAll(std::string_view destination)
{
    int32_t err = mz_zip_reader_save_all(m_handle, destination.data());

    if (err == MZ_OK)
    {
        return true;
    }
    else
    {
        LOG_DEFAULT(err, "mz_zip_reader_save_all: {}", mz_get_error_string(err));
        return false;
    }
}

} // namespace rad
