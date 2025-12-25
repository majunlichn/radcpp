#include <MLCore/CPU/CpuBackend.h>
#include <MLCore/TensorIterator.h>
#include <MLCore/CPU/CpuTensorIterator.h>
#include <MLCore/Logging.h>

#include <gtest/gtest.h>

TEST(TensorIterator, NextND)
{
    ML::Backend* backend = ML::GetBackend("CPU");
    ML::Device* device = backend->GetDevice(0);
    auto tensor = device->CreateTensor({ 2, 4, 8, 8 }, ML::DataType::Float16);
    ML::TensorIterator iter(tensor.get());
    iter.Next2D();
    iter.Next2D();
    iter.Next2D();
    iter.Next2D();
    iter.Next2D();
    iter.Next2D();
    iter.Next2D();
    std::vector<size_t> coordRef = { 1, 3, 0, 0 };
    EXPECT_EQ(iter.m_coord, coordRef);
    iter.Reset();
    do {
        ML_LOG(info, "Next2D: [{}]", rad::ToString(iter.m_coord));
    } while (iter.Next2D());

}
