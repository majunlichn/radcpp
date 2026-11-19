#pragma once

#include <rad/System/FileSystem.h>

#include <mutex>
#include <set>

namespace rad
{

// Cleans registered paths on destruction, ignoring failures. Only register paths you control:
// deletion acts on the current entry at each path, not on the original file.
// Never removes directories. Removes links without following the last path component;
// on Windows, directory symlinks and junctions are not removed.
// Public operations are serialized. Finish all calls before destruction.
class FileCleaner
{
public:
    FileCleaner() = default;
    ~FileCleaner() noexcept;

    FileCleaner(const FileCleaner&) = delete;
    FileCleaner& operator=(const FileCleaner&) = delete;

    // Repeated registration has no effect. Paths need not exist and are made absolute without
    // resolving aliases. A path naming a directory at cleanup time is left intact.
    // Root paths and paths ending in "." or ".." are invalid.
    // Invalid paths and failures to make a path absolute throw std::system_error.
    void Register(const FilePath& path);

    // Cancels registration without deleting. Does nothing if not registered.
    // Use the same absolute path if the working directory can change.
    void Unregister(const FilePath& path);

    // Checks the registration at the time of the call, not whether the file exists.
    // As with Unregister, relative paths are resolved against the current working directory.
    [[nodiscard]] bool IsRegistered(const FilePath& path) const;

    // Attempts all deletions; true means every path was removed or already absent.
    // Failed paths stay registered for retry. Successful deletions are not rolled back.
    [[nodiscard]] bool Clean();

private:
    mutable std::mutex m_mutex;
    std::set<FilePath> m_paths;
}; // class FileCleaner

} // namespace rad
