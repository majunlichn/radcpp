#pragma once

#include <rad/Core/FilePath.h>
#include <rad/Core/Result.h>

#include <cstdint>
#include <string_view>
#include <system_error>
#include <vector>

namespace rad
{

using FileTime = std::filesystem::file_time_type;
using FileType = std::filesystem::file_type;
using FilePermissions = std::filesystem::perms;

// Follow symlinks. Return false if the path is missing or the type does not match.
// The error-reporting overloads clear ec on a valid query, including a negative result.
// On error they return false and set ec; the single-argument overloads discard the error.
[[nodiscard]] bool Exists(const FilePath& path) noexcept;
[[nodiscard]] bool Exists(const FilePath& path, std::error_code& ec) noexcept;
[[nodiscard]] bool IsRegularFile(const FilePath& path) noexcept;
[[nodiscard]] bool IsRegularFile(const FilePath& path, std::error_code& ec) noexcept;

// For these metadata and mutation operations, an empty path or embedded NUL is invalid_argument.
// Size of the regular file currently at path. Follows symlinks and reports query errors.
[[nodiscard]] Result<std::uint64_t> GetSize(const FilePath& path) noexcept;
// Queries the modification time, following symlinks.
[[nodiscard]] Result<FileTime> LastWriteTime(const FilePath& path) noexcept;

// Removes a file, a symlink without following it, or an empty directory.
// If the entry is already absent, returns no_such_file_or_directory.
[[nodiscard]] Result<void> Remove(const FilePath& path) noexcept;
// Copies a regular file, following symlinks. Does not overwrite by default.
[[nodiscard]] Result<void> Copy(const FilePath& source, const FilePath& destination,
                                bool overwrite = false) noexcept;
// Renames using std::filesystem::rename semantics, without a cross-filesystem copy fallback.
[[nodiscard]] Result<void> Rename(const FilePath& source, const FilePath& destination) noexcept;

struct PathGlobOptions
{
    // Let wildcards match names beginning with '.', including directories. Ignores OS attributes.
    bool includeDotFiles = false;
};

// Finds paths under root using a UTF-8 relative pattern with '/' separators.
// Supports a POSIX subset: '*', '?', '[abc]', '[!abc]', ASCII ranges, and '\' escapes.
// These match within a segment; '?' matches one Unicode code point. A whole-segment '**'
// matches zero or more segments, as in Bash globstar. A trailing '/' selects directories.
// Matching is case-sensitive. Unless includeDotFiles is set, a dot name requires the pattern
// segment to start with a literal '.'. Shell expansions and POSIX character classes are not
// supported. Leading './' and repeated separators are allowed, but absolute paths, drive
// prefixes, and other '.' or '..' segments are not. Escape a literal unmatched '['.
//
// The root itself is not returned. Within root, directory symlinks and Windows directory
// reparse points are not descended into and do not match a trailing '/'. Links may otherwise
// appear in results. This is not a filesystem sandbox.
//
// Returns relative paths sorted by their UTF-8 spelling with '/' separators. No matches is
// success with an empty vector. Invalid arguments or syntax return invalid_argument; malformed
// Unicode in the pattern or scanned names returns illegal_byte_sequence. I/O errors propagate
// without partial results.
[[nodiscard]] Result<std::vector<FilePath>> PathGlob(const FilePath& root, std::string_view pattern,
                                                     const PathGlobOptions& options = {});

} // namespace rad
