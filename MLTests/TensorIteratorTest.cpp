#include <MLCore/CPU/CpuBackend.h>
#include <MLCore/TensorIterator.h>
#include <MLCore/Logging.h>

#include <gtest/gtest.h>

TEST(TensorIterator, NextND)
{
    ML::Backend* backend = ML::GetBackend("CPU");
    ML::Device* device = backend->GetDevice(0);
    auto tensor = device->CreateTensor({ 2, 4, 8, 8 }, ML::DataType::Float16);
    ML::TensorIterator iter(tensor.get());
    iter.Reset();
    bool hasNextND = iter.Next();
    std::vector<size_t> coordRef = { 0, 0, 0, 1 };
    ASSERT_TRUE(hasNextND);
    ASSERT_EQ(iter.m_coord, coordRef);
    hasNextND = iter.Next1D();
    coordRef = { 0, 0, 1, 1 };
    ASSERT_TRUE(hasNextND);
    ASSERT_EQ(iter.m_coord, coordRef);
    hasNextND = iter.Next2D();
    coordRef = { 0, 1, 1, 1 };
    ASSERT_TRUE(hasNextND);
    ASSERT_EQ(iter.m_coord, coordRef);
    hasNextND = iter.Next3D();
    coordRef = { 1, 1, 1, 1 };
    ASSERT_TRUE(hasNextND);
    ASSERT_EQ(iter.m_coord, coordRef);
    hasNextND = iter.Next2D();
    coordRef = { 1, 2, 1, 1 };
    ASSERT_TRUE(hasNextND);
    ASSERT_EQ(iter.m_coord, coordRef);
    hasNextND = iter.Next2D();
    coordRef = { 1, 3, 1, 1 };
    ASSERT_TRUE(hasNextND);
    ASSERT_EQ(iter.m_coord, coordRef);
    hasNextND = iter.Next2D();
    ASSERT_FALSE(hasNextND);
    iter.Reset();
    ML_LOG(info, "Offsets=[{}]; Sizes=[{}]; Strides=[{}]",
        rad::ToString(iter.m_offsets), rad::ToString(iter.m_sizes), rad::ToString(iter.m_strides));
    do {
        ML_LOG(info, "Next2D: [{}]; BufferIndex = {}", rad::ToString(iter.m_coord), iter.m_bufferIndex);
    } while (iter.Next2D());
}

TEST(TensorIterator, ForEachSubrangeND)
{
    ML::Backend* backend = ML::GetBackend("CPU");
    ML::Device* device = backend->GetDevice(0);
    ML::TensorOptions options = {};
    auto tensor = device->CreateTensor({ 2, 4, 8, 8 }, ML::DataType::Float16, options);
    ML::TensorIterator iter(tensor.get(), { 1, 2, 0, 0 });
    iter.Reset();
    ML_LOG(info, "Offsets=[{}]; Sizes=[{}]; Strides=[{}]",
        rad::ToString(iter.m_offsets), rad::ToString(iter.m_sizes), rad::ToString(iter.m_strides));
    ForEachSubrangeND(iter, [&](size_t bufferIndex) {
        ML_LOG(info, "ForEachSubrange2D: [{}]; BufferIndex = {}", rad::ToString(iter.m_coord), bufferIndex);
        }, 2);
}

TEST(TensorIterator, ForEachNHWC)
{
    ML::Backend* backend = ML::GetBackend("CPU");
    ML::Device* device = backend->GetDevice(0);
    ML::TensorOptions options = {};
    std::array<size_t, 4> sizes = { 2, 4, 8, 8 };
    options.m_strides = ML::Tensor::MakeStrides(sizes, { 1, 3, 2, 0 });
    auto tensor = device->CreateTensor(sizes, ML::DataType::Float16, options);
    ML::TensorIterator iter(tensor.get(), { 1, 2, 0, 0 });
    iter.Reset();
    ML_LOG(info, "Offsets=[{}]; Sizes=[{}]; Strides=[{}]",
        rad::ToString(iter.m_offsets), rad::ToString(iter.m_sizes), rad::ToString(iter.m_strides));
    ForEach(iter, [&](size_t bufferIndex) {
        ML_LOG(info, "ForEach: [{}]; BufferIndex = {}", rad::ToString(iter.m_coord), bufferIndex);
        });
}
