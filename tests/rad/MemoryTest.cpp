#include <rad/Common/Memory.h>
#include <rad/Common/RefCounted.h>
#include <atomic>

#include <gtest/gtest.h>

struct A : rad::RefCounted<A>
{
    static std::atomic<size_t> InstanceCount;
    A() { InstanceCount++; }
    ~A() { InstanceCount--; }
};

std::atomic<size_t> A::InstanceCount;

void TestRefCounted()
{
    rad::Ref<A> a1 = new A();
    EXPECT_EQ(a1->use_count(), 1);
    rad::Ref<A> a2 = a1;
    EXPECT_EQ(a1->use_count(), 2);
    a1.reset();
    EXPECT_EQ(a2->use_count(), 1);
    a2.reset();
    EXPECT_EQ(A::InstanceCount, 0);
}

TEST(Core, Memory)
{
    TestRefCounted();
}
