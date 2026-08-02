#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <source_location>
#include <string>
#include <utility>

#include <boost/align/aligned_allocator.hpp>

#ifndef RAD_ENABLE_MEMORY_TRACKING
// Fallback when the build does not define it.
#if defined(NDEBUG)
#define RAD_ENABLE_MEMORY_TRACKING 0
#else
#define RAD_ENABLE_MEMORY_TRACKING 1
#endif
#endif

namespace rad
{

enum class AllocationKind
{
    Unknown,
    Raw,
    RawAligned,
    Object,
    ObjectArray,
};

struct AllocationRecord
{
    const void* address = nullptr;
    std::size_t size = 0;
    AllocationKind kind = AllocationKind::Unknown;
    std::source_location location;
    std::string stackTrace;
};

struct MemoryStatistics
{
    std::size_t activeAllocationCount = 0;
    std::size_t activeBytes = 0;
    std::size_t peakAllocationCount = 0;
    std::size_t peakBytes = 0;
    std::size_t totalAllocationCount = 0;
    // Cumulative bytes requested across allocations and realloc growth (shrinks do not subtract).
    std::size_t totalAllocatedBytes = 0;
};

// Thread-safe registry of live allocations for debugging.
class MemoryTracker
{
public:
    MemoryTracker() = default;
    MemoryTracker(const MemoryTracker&) = delete;
    MemoryTracker& operator=(const MemoryTracker&) = delete;

    // A depth of zero disables stack-trace capture.
    void SetStackTraceDepth(std::size_t depth) noexcept;
    [[nodiscard]] std::size_t StackTraceDepth() const noexcept;

    void RecordAllocation(void* ptr, std::size_t size,
                          AllocationKind kind = AllocationKind::Unknown,
                          std::source_location location = std::source_location::current()) noexcept;
    // Removes the allocation for ptr. For Object/ObjectArray, ptr may be an interior base
    // subobject address; Raw/RawAligned require an exact pointer match.
    // kind must match the recorded AllocationKind.
    void RecordDeallocation(const void* ptr, AllocationKind kind) noexcept;
    // oldPtr must be nullptr or a tracked Raw allocation; kind is stored on the new entry.
    void RecordReallocation(
        void* oldPtr, void* newPtr, std::size_t newSize, AllocationKind kind = AllocationKind::Raw,
        std::source_location location = std::source_location::current()) noexcept;

    // Returns thread-safe snapshots of the current tracker state.
    [[nodiscard]] MemoryStatistics Statistics() const noexcept;
    [[nodiscard]] std::map<const void*, AllocationRecord> ActiveAllocations() const;

private:
    mutable std::mutex m_mutex;
    std::map<const void*, AllocationRecord> m_allocations;
    MemoryStatistics m_statistics;
    std::atomic<std::size_t> m_stackTraceDepth = 32;
}; // class MemoryTracker

[[nodiscard]] MemoryTracker& GetGlobalMemoryTracker() noexcept;

[[nodiscard]] inline bool IsAligned(const void* ptr, std::size_t alignment) noexcept
{
    return (alignment != 0) && (reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0);
}

// Signed byte distance from from to to (via address integers; unrelated pointers are OK).
[[nodiscard]] inline std::ptrdiff_t PointerDiff(const void* from, const void* to) noexcept
{
    return reinterpret_cast<std::intptr_t>(to) - reinterpret_cast<std::intptr_t>(from);
}

template <typename T, std::size_t Alignment>
using AlignedAllocator = boost::alignment::aligned_allocator<T, Alignment>;

// Allocates size bytes; returns nullptr for zero size or allocation failure.
[[nodiscard]] void* Allocate(
    std::size_t size, std::source_location location = std::source_location::current()) noexcept;
// Resizes an allocation, preserving the original on failure; a zero size frees it.
[[nodiscard]] void* Reallocate(
    void* ptr, std::size_t size,
    std::source_location location = std::source_location::current()) noexcept;
// Frees memory returned by Allocate or Reallocate; ptr may be nullptr.
void Free(void* ptr) noexcept;

// Allocates size bytes with at least the requested power-of-two alignment.
[[nodiscard]] void* AllocateAligned(
    std::size_t size, std::size_t alignment,
    std::source_location location = std::source_location::current()) noexcept;
// Frees memory returned by AllocateAligned; ptr may be nullptr.
void FreeAligned(void* ptr) noexcept;

template <typename T, typename... Args>
[[nodiscard]] T* New(std::source_location location, Args&&... args)
{
    T* ptr = new T(std::forward<Args>(args)...);
#if RAD_ENABLE_MEMORY_TRACKING
    GetGlobalMemoryTracker().RecordAllocation(ptr, sizeof(T), AllocationKind::Object, location);
#else
    static_cast<void>(location);
#endif
    return ptr;
}

template <typename T>
void Delete(T* ptr) noexcept(noexcept(delete ptr))
{
    static_assert(requires { sizeof(T); }, "rad::Delete requires a complete type");
#if RAD_ENABLE_MEMORY_TRACKING
    GetGlobalMemoryTracker().RecordDeallocation(ptr, AllocationKind::Object);
#endif
    delete ptr;
}

template <typename T>
[[nodiscard]] T* NewArray(std::source_location location, std::size_t count)
{
    assert(count <= std::numeric_limits<std::size_t>::max() / sizeof(T));
    T* ptr = new T[count];
#if RAD_ENABLE_MEMORY_TRACKING
    GetGlobalMemoryTracker().RecordAllocation(ptr, sizeof(T) * count, AllocationKind::ObjectArray,
                                              location);
#else
    static_cast<void>(location);
#endif
    return ptr;
}

template <typename T>
void DeleteArray(T* ptr) noexcept(noexcept(delete[] ptr))
{
    static_assert(requires { sizeof(T); }, "rad::DeleteArray requires a complete type");
#if RAD_ENABLE_MEMORY_TRACKING
    GetGlobalMemoryTracker().RecordDeallocation(ptr, AllocationKind::ObjectArray);
#endif
    delete[] ptr;
}

} // namespace rad

#define RAD_NEW(type, ...)                                                                         \
    ::rad::New<type>(std::source_location::current() __VA_OPT__(, ) __VA_ARGS__)

#define RAD_NEW_ARRAY(type, count) ::rad::NewArray<type>(std::source_location::current(), count)

#define RAD_DELETE(ptr) ::rad::Delete(ptr)

#define RAD_DELETE_ARRAY(ptr) ::rad::DeleteArray(ptr)
