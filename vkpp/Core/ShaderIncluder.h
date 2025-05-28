#pragma once

#include <string>
#include <vector>

#include <cassert>
#include <fstream>

#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace glslc {

using string_piece = std::string_view;

// https://github.com/google/shaderc/blob/main/libshaderc_util/include/libshaderc_util/file_finder.h
// https://github.com/google/shaderc/blob/main/libshaderc_util/src/file_finder.cc
// Finds files within a search path.
class FileFinder {
public:
    // Returns "" if path is empty or ends in '/'.  Otherwise, returns "/".
    static std::string MaybeSlash(const string_piece& path) {
        return (path.empty() || path.back() == '/') ? "" : "/";
    }

    // Searches for a read-openable file based on filename, which must be
    // non-empty.  The search is attempted on filename prefixed by each element of
    // search_path() in turn.  The first hit is returned, or an empty string if
    // there are no hits.  Search attempts treat their argument the way
    // std::fopen() treats its filename argument, ignoring whether the path is
    // absolute or relative.
    //
    // If a search_path() element is non-empty and not ending in a slash, then a
    // slash is inserted between it and filename before its search attempt. An
    // empty string in search_path() means that the filename is tried as-is.
    std::string FindReadableFilepath(const std::string& filename) const {
        assert(!filename.empty());
        static const auto for_reading = std::ios_base::in;
        std::filebuf opener;
        for (const auto& prefix : search_path_) {
            const std::string prefixed_filename =
                prefix + MaybeSlash(prefix) + filename;
            if (opener.open(prefixed_filename, for_reading)) return prefixed_filename;
        }
        return "";
    }

    // Searches for a read-openable file based on filename, which must be
    // non-empty. The search is first attempted as a path relative to
    // the requesting_file parameter. If no file is found relative to the
    // requesting_file then this acts as FindReadableFilepath does. If
    // requesting_file does not contain a '/' or a '\' character then it is
    // assumed to be a filename and the request will be relative to the
    // current directory.
    std::string FindRelativeReadableFilepath(const std::string& requesting_file,
        const std::string& filename) const {
        assert(!filename.empty());

        string_piece dir_name(requesting_file);

        size_t last_slash = requesting_file.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            dir_name = string_piece(requesting_file.c_str(),
                requesting_file.c_str() + last_slash);
        }

        if (dir_name.size() == requesting_file.size()) {
            dir_name = {};
        }

        static const auto for_reading = std::ios_base::in;
        std::filebuf opener;
        const std::string relative_filename =
            std::string(dir_name) + MaybeSlash(dir_name) + filename;
        if (opener.open(relative_filename, for_reading)) return relative_filename;

        return FindReadableFilepath(filename);
    }

    // Search path for Find().  Users may add/remove elements as desired.
    std::vector<std::string>& search_path() { return search_path_; }

private:
    std::vector<std::string> search_path_;
}; // class FileFinder

// https://github.com/google/shaderc/blob/main/glslc/src/file_includer.h
// An includer for files implementing shaderc's includer interface. It responds
// to the file including query from the compiler with the full path and content
// of the file to be included. In the case that the file is not found or cannot
// be opened, the full path field of in the response will point to an empty
// string, and error message will be passed to the content field.
// This class provides the basic thread-safety guarantee.
class FileIncluder : public shaderc::CompileOptions::IncluderInterface {
public:
    explicit FileIncluder(const FileFinder* file_finder)
        : file_finder_(*file_finder) {
    }

    ~FileIncluder() override = default;

    // https://github.com/google/shaderc/blob/main/libshaderc_util/src/io_shaderc.cc
    static bool ReadFile(const std::string& input_file_name,
        std::vector<char>* input_data) {
        std::ifstream input_file;
        input_file.open(input_file_name, std::ios_base::binary);
        if (input_file.fail()) {
            return false;
        }
        *input_data = std::vector<char>((std::istreambuf_iterator<char>(input_file)),
            std::istreambuf_iterator<char>());
        return true;
    }

    static shaderc_include_result* MakeErrorIncludeResult(const char* message) {
        return RAD_NEW shaderc_include_result{ "", 0, message, strlen(message) };
    }

    // Resolves a requested source file of a given type from a requesting
    // source into a shaderc_include_result whose contents will remain valid
    // until it's released.
    shaderc_include_result* GetInclude(const char* requested_source,
        shaderc_include_type include_type,
        const char* requesting_source,
        size_t include_depth) override
    {
        const std::string full_path =
            (include_type == shaderc_include_type_relative)
            ? file_finder_.FindRelativeReadableFilepath(requesting_source,
                requested_source)
            : file_finder_.FindReadableFilepath(requested_source);

        if (full_path.empty())
            return MakeErrorIncludeResult("Cannot find or open include file.");

        // In principle, several threads could be resolving includes at the same
        // time.  Protect the included_files.

        // Read the file and save its full path and contents into stable addresses.
        FileInfo* new_file_info = RAD_NEW FileInfo{ full_path, {} };
        if (!ReadFile(full_path, &(new_file_info->contents))) {
            return MakeErrorIncludeResult("Cannot read file");
        }

        // Protect the included_files.
        {
            std::lock_guard<std::mutex> lock(include_files_mutex_);
            included_files_.insert(full_path);
        }

        return RAD_NEW shaderc_include_result{
            new_file_info->full_path.data(), new_file_info->full_path.length(),
            new_file_info->contents.data(), new_file_info->contents.size(),
            new_file_info };
    }

    // Releases an include result.
    void ReleaseInclude(shaderc_include_result* include_result) override
    {
        FileInfo* info = static_cast<FileInfo*>(include_result->user_data);
        delete info;
        delete include_result;
    }

    // Returns a reference to the member storing the set of included files.
    const std::unordered_set<std::string>& file_path_trace() const {
        return included_files_;
    }

private:
    // Used by GetInclude() to get the full filepath.
    const FileFinder& file_finder_;
    // The full path and content of a source file.
    struct FileInfo {
        const std::string full_path;
        std::vector<char> contents;
    };

    // The set of full paths of included files.
    std::unordered_set<std::string> included_files_;
    std::mutex include_files_mutex_;

}; // class FileIncluder

}  // namespace glslc
