#include <rad/IO/Compression.h>
#include <rad/IO/Logging.h>

#include "mz.h"
#include "mz_os.h"
#include "mz_strm.h"
#include "mz_strm_buf.h"
#include "mz_strm_split.h"
#include "mz_zip.h"
#include "mz_zip_rw.h"

namespace rad
{

const char* mz_get_error_string(int32_t err)
{
    switch (err)
    {
    case MZ_OK: return "MZ_OK";
    case MZ_STREAM_ERROR: return "MZ_STREAM_ERROR";
    case MZ_DATA_ERROR: return "MZ_DATA_ERROR";
    case MZ_MEM_ERROR: return "MZ_MEM_ERROR";
    case MZ_BUF_ERROR: return "MZ_BUF_ERROR";
    case MZ_VERSION_ERROR: return "MZ_VERSION_ERROR";
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
    return "MZ_UNKNOWN";
}

// https://github.com/zlib-ng/minizip-ng/blob/develop/minizip.c
bool ZipCompress(const char* path, const char* password, ZipOptions* options, Span<std::string> args) {
    void* writer = NULL;
    bool result = true;
    int32_t err = MZ_OK;
    int32_t i = 0;
    const char* filename_in_zip = NULL;

    RAD_LOG_DEFAULT(info, "ZipCompress: {}", path);

    /* Create zip writer */
    writer = mz_zip_writer_create();
    if (!writer)
    {
        return false;
    }

    mz_zip_writer_set_password(writer, password);
    mz_zip_writer_set_aes(writer, options->aes);
    if (options->compress_level == 0)
    {
        mz_zip_writer_set_compress_method(writer, MZ_COMPRESS_METHOD_STORE);
    }
    else
    {
        mz_zip_writer_set_compress_method(writer, options->compress_method);
    }
    mz_zip_writer_set_compress_level(writer, options->compress_level);
    mz_zip_writer_set_follow_links(writer, options->follow_links);
    mz_zip_writer_set_store_links(writer, options->store_links);
    //mz_zip_writer_set_overwrite_cb(writer, options, minizip_add_overwrite_cb);
    //mz_zip_writer_set_progress_cb(writer, options, minizip_add_progress_cb);
    //mz_zip_writer_set_entry_cb(writer, options, minizip_add_entry_cb);
    mz_zip_writer_set_zip_cd(writer, options->zip_cd);

    err = mz_zip_writer_open_file(writer, path, options->disk_size, options->append);

    if (err == MZ_OK) {
        for (i = 0; i < static_cast<int32_t>(args.size()); i += 1) {
            filename_in_zip = args[i].c_str();

            /* Add file system path to archive */
            err = mz_zip_writer_add_path(writer, filename_in_zip, NULL, options->include_path, 1);
            if (err != MZ_OK)
            {
                RAD_LOG_DEFAULT(err, "ZipCompress: Error {} adding path to archive {}",
                    mz_get_error_string(err), filename_in_zip);
                result = false;
            }
        }
    }
    else {
        RAD_LOG_DEFAULT(err, "ZipCompress: Error {} opening archive for writing",
            mz_get_error_string(err));
        result = false;
    }

    err = mz_zip_writer_close(writer);
    if (err != MZ_OK) {
        RAD_LOG_DEFAULT(err, "ZipCompress: Error {} closing archive for writing {}",
            mz_get_error_string(err), path);
        result = false;
    }

    mz_zip_writer_delete(&writer);
    return result;
}

// https://github.com/zlib-ng/minizip-ng/blob/develop/minizip.c
bool ZipDecompress(const char* path, const char* pattern, const char* destination, const char* password, ZipOptions* options) {
    void* reader = NULL;
    bool result = true;
    int32_t err = MZ_OK;

    RAD_LOG_DEFAULT(info, "ZipDecompress: {}", path);

    /* Create zip reader */
    reader = mz_zip_reader_create();
    if (!reader)
    {
        return false;
    }

    mz_zip_reader_set_pattern(reader, pattern, 1);
    mz_zip_reader_set_password(reader, password);
    mz_zip_reader_set_encoding(reader, options->encoding);
    //mz_zip_reader_set_entry_cb(reader, options, minizip_extract_entry_cb);
    //mz_zip_reader_set_progress_cb(reader, options, minizip_extract_progress_cb);
    //mz_zip_reader_set_overwrite_cb(reader, options, minizip_extract_overwrite_cb);

    err = mz_zip_reader_open_file(reader, path);

    if (err != MZ_OK) {
        RAD_LOG_DEFAULT(err, "ZipDecompress: Error {} opening archive {}",
            mz_get_error_string(err), path);
        result = false;
    }
    else {
        /* Save all entries in archive to destination directory */
        err = mz_zip_reader_save_all(reader, destination);

        if (err == MZ_END_OF_LIST) {
            if (pattern) {
                RAD_LOG_DEFAULT(warn,
                    "ZipDecompress: Files matching \"{}\" not found in archive!", pattern);
            }
            else {
                RAD_LOG_DEFAULT(warn, "ZipDecompress: No files in archive!");
                err = MZ_OK;
            }
        }
        else if (err != MZ_OK) {
            RAD_LOG_DEFAULT(err, "ZipDecompress: Error {} saving entries to disk {}",
                mz_get_error_string(err), path);
            result = false;
        }
    }

    err = mz_zip_reader_close(reader);
    if (err != MZ_OK) {
        RAD_LOG_DEFAULT(err, "ZipDecompress: Error {} closing archive for reading!",
            mz_get_error_string(err));
        result = false;
    }

    mz_zip_reader_delete(&reader);
    return result;
}

} // namespace rad
