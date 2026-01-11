#include <MLCore/MLCore.h>

#include <gtest/gtest.h>

TEST(TensorIterator, NextND)
{
    auto tensor = ML::MakeTensor({ 2, 4, 8, 8 }, ML::DataType::Float16);
    ML::TensorIterator iter(&tensor);
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

TEST(TensorIterator, Next)
{
    ML::TensorOptions options = {};
    auto tensor = ML::MakeTensor({ 2, 2, 4, 4 }, ML::DataType::Float16, options);
    ML::TensorIterator iter(&tensor, { 0, 0, 2, 2 });
    iter.Reset();
    ML_LOG(info, "Offsets=[{}]; Sizes=[{}]; Strides=[{}]",
        rad::ToString(iter.m_offsets), rad::ToString(iter.m_sizes), rad::ToString(iter.m_strides));
    do {
        ML_LOG(info, "Next: [{}]; BufferIndex = {}", rad::ToString(iter.m_coord), iter.m_bufferIndex);
        ASSERT_EQ(iter.CoordToBufferIndex(), iter.m_bufferIndex);
    } while (iter.Next());
}

TEST(TensorIterator, ForEach)
{
    ML::TensorOptions options = {};
    auto tensor = ML::MakeTensor({ 2, 2, 4, 4 }, ML::DataType::Float16, options);
    ML::TensorIterator iter(&tensor, { 0, 0, 2, 2 });
    iter.Reset();
    ML_LOG(info, "Offsets=[{}]; Sizes=[{}]; Strides=[{}]",
        rad::ToString(iter.m_offsets), rad::ToString(iter.m_sizes), rad::ToString(iter.m_strides));
    ForEach(iter, [&](size_t bufferIndex) {
        ML_LOG(info, "ForEach: [{}]; BufferIndex = {}", rad::ToString(iter.m_coord), bufferIndex);
        ASSERT_EQ(iter.CoordToBufferIndex(), bufferIndex);
        });
}

TEST(TensorIterator, ForEachRecursively)
{
    ML::TensorOptions options = {};
    auto tensor = ML::MakeTensor({ 2, 2, 4, 4 }, ML::DataType::Float16, options);
    ML::TensorIterator iter(&tensor, { 0, 0, 2, 2 });
    iter.Reset();
    ML_LOG(info, "Offsets=[{}]; Sizes=[{}]; Strides=[{}]",
        rad::ToString(iter.m_offsets), rad::ToString(iter.m_sizes), rad::ToString(iter.m_strides));
    ForEachRecursively(iter, [&](size_t bufferIndex) {
        ML_LOG(info, "ForEachRecursively: [{}]; BufferIndex = {}", rad::ToString(iter.m_coord), bufferIndex);
        ASSERT_EQ(iter.CoordToBufferIndex(), bufferIndex);
        });
}

TEST(TensorIterator, ForEachSubrangeND)
{
    ML::TensorOptions options = {};
    auto tensor = ML::MakeTensor({ 2, 4, 8, 8 }, ML::DataType::Float16, options);
    ML::TensorIterator iter(&tensor, { 1, 2, 4, 4 });
    iter.Reset();
    ML_LOG(info, "Offsets=[{}]; Sizes=[{}]; Strides=[{}]",
        rad::ToString(iter.m_offsets), rad::ToString(iter.m_sizes), rad::ToString(iter.m_strides));
    ForEachSubrangeND(iter, [&](size_t bufferIndex) {
        ML_LOG(info, "ForEachSubrangeND: [{}]; BufferIndex = {}", rad::ToString(iter.m_coord), bufferIndex);
        ASSERT_EQ(iter.CoordToBufferIndex(), bufferIndex);
        }, 2);
}

TEST(TensorIterator, ForEachNHWC)
{
    ML::TensorOptions options = {};
    std::array<size_t, 4> sizes = { 2, 4, 8, 8 };
    options.m_strides = ML::MakeTensorStrides(sizes, { 1, 3, 2, 0 });
    auto tensor = ML::MakeTensor(sizes, ML::DataType::Float16, options);
    ML::TensorIterator iter(&tensor, { 1, 0, 4, 4 });
    iter.Reset();
    ML_LOG(info, "Offsets=[{}]; Sizes=[{}]; Strides=[{}]; Permutation=[{}]",
        rad::ToString(iter.m_offsets), rad::ToString(iter.m_sizes), rad::ToString(iter.m_strides), rad::ToString(iter.m_permutation));
    iter.SetDimOrder({ 1, 3, 2, 0 }); // NHWC
    ML_LOG(info, "Offsets=[{}]; Sizes=[{}]; Strides=[{}]; Permutation=[{}]",
        rad::ToString(iter.m_offsets), rad::ToString(iter.m_sizes), rad::ToString(iter.m_strides), rad::ToString(iter.m_permutation));
    ForEach(iter, [&](size_t bufferIndex) {
        ML_LOG(info, "ForEachNHWC: [{}]; BufferIndex = {}", rad::ToString(iter.GetCoordUnpermuted()), bufferIndex);
        ASSERT_EQ(iter.CoordToBufferIndex(), bufferIndex);
        });
}
