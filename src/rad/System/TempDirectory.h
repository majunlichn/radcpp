#pragma once

#include <rad/Core/FilePath.h>
#include <rad/Core/Result.h>

namespace rad
{

// Owns a temporary directory and its contents. Cleanup acts on the current entry at Path();
// use a trusted parent and do not rename or replace the directory while it is owned.
// Destruction and move assignment clean up the previous directory on a best-effort basis.
class TempDirectory
{
public:
    ~TempDirectory() noexcept;

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;
    TempDirectory(TempDirectory&& other) noexcept;
    TempDirectory& operator=(TempDirectory&& other) noexcept;

    // Creates a new directory in an existing parent. An empty parent uses the system temporary
    // directory. Uses at most 0700 on POSIX and inherited permissions on Windows.
    [[nodiscard]] static Result<TempDirectory> Create(const FilePath& parent = {}) noexcept;

    // Absolute path; empty after a move, release, or successful removal.
    [[nodiscard]] const FilePath& Path() const noexcept;
    // Transfers ownership of the directory and its contents to the caller.
    [[nodiscard]] FilePath Release() noexcept;
    // Removes the tree without following symbolic links. A missing directory is already clean.
    // On failure some entries may have been removed; the path remains owned for retry.
    [[nodiscard]] Result<void> Clean() noexcept;

private:
    explicit TempDirectory(FilePath path) noexcept;

    FilePath m_path;
}; // class TempDirectory

} // namespace rad
