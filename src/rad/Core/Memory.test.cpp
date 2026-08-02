#include <rad/Core/Memory.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <cstring>

TEST(Core, NewDelete)
{
    int* number = RAD_NEW(int, 42);
    ASSERT_NE(number, nullptr);
    EXPECT_EQ(*number, 42);
    RAD_DELETE(number);
}

TEST(Core, NewDeleteArray)
{
    int* numbers = RAD_NEW_ARRAY(int, 4);
    ASSERT_NE(numbers, nullptr);
    for (std::size_t i = 0; i < 4; ++i)
    {
        numbers[i] = static_cast<int>(i);
        EXPECT_EQ(numbers[i], i);
    }
    RAD_DELETE_ARRAY(numbers);
}

TEST(Core, MemoryTracker)
{
#if RAD_ENABLE_MEMORY_TRACKING
    rad::MemoryTracker tracker;
    const rad::MemoryStatistics statisticsBefore = tracker.Statistics();

    void* ptr = std::malloc(1024);
    ASSERT_NE(ptr, nullptr);
    tracker.RecordAllocation(ptr, 1024, rad::AllocationKind::Raw);
    {
        const rad::MemoryStatistics statistics = tracker.Statistics();
        EXPECT_EQ(statistics.activeAllocationCount, statisticsBefore.activeAllocationCount + 1);
        EXPECT_EQ(statistics.activeBytes, statisticsBefore.activeBytes + 1024);
        const auto records = tracker.ActiveAllocations();
        const auto record = records.find(ptr);
        ASSERT_NE(record, records.end());
        EXPECT_EQ(record->second.address, ptr);
        EXPECT_EQ(record->second.size, 1024u);
        EXPECT_EQ(record->second.kind, rad::AllocationKind::Raw);
    }

    tracker.RecordDeallocation(ptr, rad::AllocationKind::Raw);
    std::free(ptr);
    {
        const rad::MemoryStatistics statistics = tracker.Statistics();
        EXPECT_EQ(statistics.activeAllocationCount, statisticsBefore.activeAllocationCount);
        EXPECT_EQ(statistics.activeBytes, statisticsBefore.activeBytes);
        const auto records = tracker.ActiveAllocations();
        EXPECT_FALSE(records.contains(ptr));
    }
#endif
}

TEST(Core, AllocateAligned)
{
    // Allocations satisfy the requested alignment, enforce pointer alignment, and remain writable.
    {
        constexpr std::array<std::size_t, 7> alignments = {1, 2, 4, 8, 16, 32, 64};

        for (const std::size_t alignment : alignments)
        {
            void* ptr = rad::AllocateAligned(257, alignment);
            ASSERT_NE(ptr, nullptr);
            EXPECT_TRUE(rad::IsAligned(ptr, alignment));
            EXPECT_TRUE(rad::IsAligned(ptr, sizeof(void*)));

            std::memset(ptr, 0xA5, 257);
            rad::FreeAligned(ptr);
        }
    }

    // Invalid allocation requests fail without returning storage.
    {
        EXPECT_EQ(rad::AllocateAligned(0, 16), nullptr);
        EXPECT_EQ(rad::AllocateAligned(16, 0), nullptr);
        EXPECT_EQ(rad::AllocateAligned(16, 3), nullptr);
    }

    // Freeing a null pointer is a no-op.
    {
        rad::FreeAligned(nullptr);
    }
}
