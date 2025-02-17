#pragma once

#include <radcpp/Core/Platform.h>
#include <memory>

namespace rad
{

template< class T >
using Shared = std::shared_ptr<T>;

template<class T, class Deleter = std::default_delete<T>>
using Unique = std::unique_ptr<T, Deleter>;

template<class T, class Deleter>
using UniqueArray = std::unique_ptr<T[], Deleter>;

void* AlignedAlloc(std::size_t size, std::size_t alignment);
void AlignedFree(void* p);

} // namespace rad
