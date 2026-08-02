#include <rad/Core/Memory.h>

#include <rad/Core/Integer.h>
#include <rad/Core/Platform.h>

#include <boost/stacktrace/stacktrace.hpp>

#include <algorithm>
#include <cassert>
#include <cstdlib>

#if defined(RAD_OS_WINDOWS)
#include <malloc.h>
#endif

namespace
{

#if RAD_ENABLE_MEMORY_TRACKING

[[nodiscard]] rad::AllocationRecord MakeAllocationRecord(void* ptr, std::size_t size,
                                                         rad::AllocationKind kind,
                                                         std::source_location location,
                                                         std::size_t stackTraceDepth)
{
    rad::AllocationRecord allocation;
    allocation.address = ptr;
    allocation.size = size;
    allocation.kind = kind;
    allocation.location = location;
    if (stackTraceDepth != 0)
    {
        try
        {
#if defined(RAD_OS_WINDOWS)
            constexpr std::size_t skipFrames = 4;
#else
            constexpr std::size_t skipFrames = 2;
#endif
            allocation.stackTrace = boost::stacktrace::to_string(
                boost::stacktrace::stacktrace(skipFrames, stackTraceDepth));
        }
        catch (...)
        {
        }
    }
    return allocation;
}

void UpdatePeaks(rad::MemoryStatistics& statistics) noexcept
{
    statistics.peakAllocationCount =
        std::max(statistics.peakAllocationCount, statistics.activeAllocationCount);
    statistics.peakBytes = std::max(statistics.peakBytes, statistics.activeBytes);
}

#endif // RAD_ENABLE_MEMORY_TRACKING

} // namespace

namespace rad
{

MemoryTracker& GetGlobalMemoryTracker() noexcept
{
    static MemoryTracker tracker;
    return tracker;
}

void MemoryTracker::SetStackTraceDepth(std::size_t depth) noexcept
{
    m_stackTraceDepth.store(depth, std::memory_order_relaxed);
}

std::size_t MemoryTracker::StackTraceDepth() const noexcept
{
    return m_stackTraceDepth.load(std::memory_order_relaxed);
}

void MemoryTracker::RecordAllocation(void* ptr, std::size_t size, AllocationKind kind,
                                     std::source_location location) noexcept
{
#if RAD_ENABLE_MEMORY_TRACKING
    if (ptr == nullptr)
    {
        return;
    }

    try
    {
        AllocationRecord allocation =
            MakeAllocationRecord(ptr, size, kind, location, StackTraceDepth());

        std::lock_guard lock(m_mutex);
        const auto [iterator, inserted] = m_allocations.emplace(ptr, std::move(allocation));
        static_cast<void>(iterator);
        assert(inserted && "MemoryTracker failed to record allocation");
        if (!inserted)
        {
            return;
        }

        ++m_statistics.activeAllocationCount;
        m_statistics.activeBytes += size;
        ++m_statistics.totalAllocationCount;
        m_statistics.totalAllocatedBytes += size;
        UpdatePeaks(m_statistics);
    }
    catch (...)
    {
        assert(false && "MemoryTracker failed to record allocation");
    }
#else
    static_cast<void>(ptr);
    static_cast<void>(size);
    static_cast<void>(kind);
    static_cast<void>(location);
#endif
}

void MemoryTracker::RecordDeallocation(const void* ptr, AllocationKind kind) noexcept
{
#if RAD_ENABLE_MEMORY_TRACKING
    if (ptr == nullptr)
    {
        return;
    }
    try
    {
        std::lock_guard lock(m_mutex);

        auto iterator = m_allocations.find(ptr);
        // Object deletes may pass an adjusted base pointer; raw frees must match exactly.
        if (iterator == m_allocations.end() &&
            (kind == AllocationKind::Object || kind == AllocationKind::ObjectArray))
        {
            iterator = m_allocations.upper_bound(ptr);
            if (iterator == m_allocations.begin())
            {
                iterator = m_allocations.end();
            }
            else
            {
                --iterator;
                const void* allocationAddress = iterator->first;
                const std::ptrdiff_t offset = PointerDiff(allocationAddress, ptr);
                if (offset < 0 || static_cast<std::size_t>(offset) >= iterator->second.size)
                {
                    iterator = m_allocations.end();
                }
            }
        }

        assert(iterator != m_allocations.end() &&
               "Deallocation of a pointer not tracked by MemoryTracker");
        if (iterator == m_allocations.end())
        {
            return;
        }

        assert(iterator->second.kind == kind && "AllocationKind mismatch on deallocation");
        if (iterator->second.kind != kind)
        {
            return;
        }

        --m_statistics.activeAllocationCount;
        m_statistics.activeBytes -= iterator->second.size;
        m_allocations.erase(iterator);
    }
    catch (...)
    {
        assert(false && "MemoryTracker failed to record deallocation");
    }
#else
    static_cast<void>(ptr);
    static_cast<void>(kind);
#endif
}

void MemoryTracker::RecordReallocation(void* oldPtr, void* newPtr, std::size_t newSize,
                                       AllocationKind kind, std::source_location location) noexcept
{
#if RAD_ENABLE_MEMORY_TRACKING
    if (newPtr == nullptr)
    {
        return;
    }

    try
    {
        AllocationRecord allocation =
            MakeAllocationRecord(newPtr, newSize, kind, location, StackTraceDepth());
        std::lock_guard lock(m_mutex);

        // oldPtr must be nullptr (malloc-style) or already tracked.
        if (oldPtr == nullptr)
        {
            const auto [iterator, inserted] = m_allocations.emplace(newPtr, std::move(allocation));
            static_cast<void>(iterator);
            assert(inserted && "MemoryTracker failed to record reallocation");
            if (!inserted)
            {
                return;
            }

            ++m_statistics.activeAllocationCount;
            m_statistics.activeBytes += newSize;
            ++m_statistics.totalAllocationCount;
            m_statistics.totalAllocatedBytes += newSize;
            UpdatePeaks(m_statistics);
            return;
        }

        auto oldIterator = m_allocations.find(oldPtr);
        assert(oldIterator != m_allocations.end() &&
               "realloc of a pointer not tracked by MemoryTracker");
        if (oldIterator == m_allocations.end())
        {
            return;
        }

        assert(oldIterator->second.kind == kind && "AllocationKind mismatch on reallocation");
        if (oldIterator->second.kind != kind)
        {
            return;
        }

        const std::size_t oldSize = oldIterator->second.size;
        if (newPtr == oldPtr)
        {
            oldIterator->second = std::move(allocation);
        }
        else
        {
            m_allocations.erase(oldIterator);

            auto newIterator = m_allocations.find(newPtr);
            if (newIterator != m_allocations.end())
            {
                assert(false && "realloc moved onto an address still tracked (missed free?)");
                m_statistics.activeBytes -= newIterator->second.size;
                --m_statistics.activeAllocationCount;
                newIterator->second = std::move(allocation);
            }
            else
            {
                const auto [iterator, inserted] =
                    m_allocations.emplace(newPtr, std::move(allocation));
                static_cast<void>(iterator);
                assert(inserted && "MemoryTracker failed to record reallocation");
                if (!inserted)
                {
                    return;
                }
            }
        }

        m_statistics.activeBytes = m_statistics.activeBytes - oldSize + newSize;
        if (newSize > oldSize)
        {
            m_statistics.totalAllocatedBytes += newSize - oldSize;
        }
        UpdatePeaks(m_statistics);
    }
    catch (...)
    {
        assert(false && "MemoryTracker failed to record reallocation");
    }
#else
    static_cast<void>(oldPtr);
    static_cast<void>(newPtr);
    static_cast<void>(newSize);
    static_cast<void>(kind);
    static_cast<void>(location);
#endif
}

MemoryStatistics MemoryTracker::Statistics() const noexcept
{
#if RAD_ENABLE_MEMORY_TRACKING
    std::lock_guard lock(m_mutex);
    return m_statistics;
#else
    return {};
#endif
}

std::map<const void*, AllocationRecord> MemoryTracker::ActiveAllocations() const
{
#if RAD_ENABLE_MEMORY_TRACKING
    std::lock_guard lock(m_mutex);
    return m_allocations;
#else
    return {};
#endif
}

void* Allocate(std::size_t size, std::source_location location) noexcept
{
    if (size == 0)
    {
        return nullptr;
    }

    void* ptr = std::malloc(size);
#if RAD_ENABLE_MEMORY_TRACKING
    GetGlobalMemoryTracker().RecordAllocation(ptr, size, AllocationKind::Raw, location);
#else
    static_cast<void>(location);
#endif
    return ptr;
}

void* Reallocate(void* ptr, std::size_t size, std::source_location location) noexcept
{
    if (size == 0)
    {
#if RAD_ENABLE_MEMORY_TRACKING
        GetGlobalMemoryTracker().RecordDeallocation(ptr, AllocationKind::Raw);
#else
        static_cast<void>(location);
#endif
        std::free(ptr);
        return nullptr;
    }

    void* newPtr = std::realloc(ptr, size);
#if RAD_ENABLE_MEMORY_TRACKING
    if (newPtr != nullptr)
    {
        GetGlobalMemoryTracker().RecordReallocation(ptr, newPtr, size, AllocationKind::Raw,
                                                    location);
    }
#else
    static_cast<void>(location);
#endif
    return newPtr;
}

void Free(void* ptr) noexcept
{
#if RAD_ENABLE_MEMORY_TRACKING
    GetGlobalMemoryTracker().RecordDeallocation(ptr, AllocationKind::Raw);
#endif
    std::free(ptr);
}

void* AllocateAligned(std::size_t size, std::size_t alignment,
                      std::source_location location) noexcept
{
    if (size == 0 || !IsPowerOfTwo(alignment))
    {
        return nullptr;
    }

    if (alignment < sizeof(void*))
    {
        alignment = sizeof(void*);
    }

    void* ptr = nullptr;
#if defined(RAD_OS_WINDOWS)
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0)
    {
        ptr = nullptr;
    }
#endif

#if RAD_ENABLE_MEMORY_TRACKING
    GetGlobalMemoryTracker().RecordAllocation(ptr, size, AllocationKind::RawAligned, location);
#else
    static_cast<void>(location);
#endif
    return ptr;
}

void FreeAligned(void* ptr) noexcept
{
#if RAD_ENABLE_MEMORY_TRACKING
    GetGlobalMemoryTracker().RecordDeallocation(ptr, AllocationKind::RawAligned);
#endif
#if defined(RAD_OS_WINDOWS)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

} // namespace rad
