#include <rad/System/FileSystem.h>

#include <rad/Core/Platform.h>
#include <rad/Core/Unicode.h>

namespace rad
{

FilePath MakeFilePath(std::string_view utf8Path)
{
#if defined(RAD_OS_WINDOWS)
    return FilePath{Utf8ToWide(utf8Path)};
#else
    return FilePath{utf8Path};
#endif
}

std::string ToUtf8(const FilePath& path)
{
#if defined(RAD_OS_WINDOWS)
    return WideToUtf8(path.native());
#else
    return path.string();
#endif
}

} // namespace rad
