#pragma once

#include <rad/Core/Result.h>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace rad
{

using FilePath = std::filesystem::path;
using FileTime = std::filesystem::file_time_type;
using FileType = std::filesystem::file_type;
using FilePermissions = std::filesystem::perms;

// std::string paths are UTF-8 at this boundary and are converted to the platform-native encoding.
[[nodiscard]] FilePath MakeFilePath(std::string_view utf8Path);
[[nodiscard]] std::string ToUtf8(const FilePath& path);

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
