#include <MLCore/CPU/CpuBackend.h>
#include <MLCore/TensorIterator.h>
#include <MLCore/Logging.h>

#include <gtest/gtest.h>

TEST(TensorIterator, NextND)
{
    ML::Backend* backend = ML::GetBackend("CPU");
    ML::Device* device = backend->GetDevice(0);
    {
        auto tensor = device->CreateTensor({ 2, 4, 8, 8 }, ML::DataType::Float16);
        ML::TensorIterator iter(tensor.get());
        iter.Reset();
        do {
            ML_LOG(info, "Next2D: [{}]; BufferIndex = {}", rad::ToString(iter.m_coord), iter.m_bufferIndex);
        } while (iter.Next2D());
        ForEach(iter, [&](size_t bufferIndex) {
            ML_LOG(info, "ForEach: [{}]; BufferIndex = {}", rad::ToString(iter.m_coord), bufferIndex);
            });
    }
    {
        ML::TensorOptions options = {};
        options.m_strides = ML::Tensor::MakeStrides({ 2, 4, 8, 8 }, { 1, 3, 2, 0 }); // N, H, W, C
        auto tensor = device->CreateTensor({ 2, 4, 8, 8 }, ML::DataType::Float16, options);
        ML::TensorIterator iter(tensor.get());
        iter.Reset();
        do {
            ML_LOG(info, "Next2D: [{}]; BufferIndex = {}", rad::ToString(iter.m_coord), iter.m_bufferIndex);
        } while (iter.Next2D());
        ForEach(iter, [&](size_t bufferIndex) {
            ML_LOG(info, "ForEach: [{}]; BufferIndex = {}", rad::ToString(iter.m_coord), bufferIndex);
            });
    }
}
