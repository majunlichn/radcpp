#include <rad/ML/MLCpuContext.h>
#include <rad/ML/MLCpuTensorOp.h>

namespace rad
{

MLCpuContext::MLCpuContext(Ref<MLCpuDevice> device) :
    m_device(std::move(device))
{
}

MLCpuContext::~MLCpuContext()
{
}

void MLCpuContext::FillConstant(MLTensor* input, float value)
{
    assert(IsFloatingPointType(input->m_dataType));
#define MLCPUCONTEXT_DISPATCH_FILL_CONSTANT(DataType)   \
    MLCpuTensorOpForEach<DataType>()(input, [&]() { return DataType(value); })
    switch (input->m_dataType)
    {
    case MLDataType::Float16:       MLCPUCONTEXT_DISPATCH_FILL_CONSTANT(Float16); return;
    case MLDataType::Float32:       MLCPUCONTEXT_DISPATCH_FILL_CONSTANT(Float32); return;
    case MLDataType::Float64:       MLCPUCONTEXT_DISPATCH_FILL_CONSTANT(Float64); return;
    case MLDataType::BFloat16:      MLCPUCONTEXT_DISPATCH_FILL_CONSTANT(BFloat16); return;
    case MLDataType::Float8E4M3:    MLCPUCONTEXT_DISPATCH_FILL_CONSTANT(Float8E4M3); return;
    case MLDataType::Float8E5M2:    MLCPUCONTEXT_DISPATCH_FILL_CONSTANT(Float8E5M2); return;
    }
#undef MLCPUCONTEXT_DISPATCH_FILL_CONSTANT
    RAD_UNREACHABLE();
}

void MLCpuContext::FillConstant(MLTensor* input, int value)
{
    assert(IsIntegerType(input->m_dataType));
#define MLCPUCONTEXT_DISPATCH_FILL_CONSTANT(DataType)   \
    MLCpuTensorOpForEach<DataType>()(input, [&]() { return DataType(value); })
    switch (input->m_dataType)
    {
    case MLDataType::Sint8: MLCPUCONTEXT_DISPATCH_FILL_CONSTANT(Sint8); return;
    case MLDataType::Sint16: MLCPUCONTEXT_DISPATCH_FILL_CONSTANT(Sint16); return;
    case MLDataType::Sint32: MLCPUCONTEXT_DISPATCH_FILL_CONSTANT(Sint32); return;
    case MLDataType::Sint64: MLCPUCONTEXT_DISPATCH_FILL_CONSTANT(Sint64); return;
    case MLDataType::Uint8: MLCPUCONTEXT_DISPATCH_FILL_CONSTANT(Uint8); return;
    case MLDataType::Uint16: MLCPUCONTEXT_DISPATCH_FILL_CONSTANT(Uint16); return;
    case MLDataType::Uint32: MLCPUCONTEXT_DISPATCH_FILL_CONSTANT(Uint32); return;
    case MLDataType::Uint64: MLCPUCONTEXT_DISPATCH_FILL_CONSTANT(Uint64); return;
    }
#undef MLCPUCONTEXT_DISPATCH_FILL_CONSTANT
    RAD_UNREACHABLE();
}

void MLCpuContext::Add(MLTensor* input, MLTensor* other, float alpha, MLTensor* output)
{
    assert(IsFloatingPointType(input->m_dataType));
#define MLCPUCONTEXT_DISPATCH_ADD(DataType, ComputeType)    \
    MLCpuTensorOpElementWiseBinary<DataType, ComputeType>()(input, other, output, [&](ComputeType a, ComputeType b) { return a + ComputeType(alpha) * b; });
    switch (input->m_dataType)
    {
    case MLDataType::Float16:       MLCPUCONTEXT_DISPATCH_ADD(Float16, Float32); return;
    case MLDataType::Float32:       MLCPUCONTEXT_DISPATCH_ADD(Float32, Float32); return;
    case MLDataType::Float64:       MLCPUCONTEXT_DISPATCH_ADD(Float64, Float64); return;
    case MLDataType::BFloat16:      MLCPUCONTEXT_DISPATCH_ADD(BFloat16, Float32); return;
    case MLDataType::Float8E4M3:    MLCPUCONTEXT_DISPATCH_ADD(Float8E4M3, Float32); return;
    case MLDataType::Float8E5M2:    MLCPUCONTEXT_DISPATCH_ADD(Float8E5M2, Float32); return;
    }
#undef MLCPUCONTEXT_DISPATCH_ADD
    RAD_UNREACHABLE();
}

void MLCpuContext::Add(MLTensor* input, MLTensor* other, int alpha, MLTensor* output)
{
    assert(IsIntegerType(input->m_dataType));
#define MLCPUCONTEXT_DISPATCH_ADD(DataType, ComputeType)    \
    MLCpuTensorOpElementWiseBinary<DataType, ComputeType>()(input, other, output, [&](ComputeType a, ComputeType b) { return a + ComputeType(alpha) * b; });
    switch (input->m_dataType)
    {
    case MLDataType::Sint8:         MLCPUCONTEXT_DISPATCH_ADD(Sint8, Sint8); return;
    case MLDataType::Sint16:        MLCPUCONTEXT_DISPATCH_ADD(Sint16, Sint16); return;
    case MLDataType::Sint32:        MLCPUCONTEXT_DISPATCH_ADD(Sint32, Sint32); return;
    case MLDataType::Sint64:        MLCPUCONTEXT_DISPATCH_ADD(Sint64, Sint64); return;
    case MLDataType::Uint8:         MLCPUCONTEXT_DISPATCH_ADD(Uint8, Uint8); return;
    case MLDataType::Uint16:        MLCPUCONTEXT_DISPATCH_ADD(Uint16, Uint16); return;
    case MLDataType::Uint32:        MLCPUCONTEXT_DISPATCH_ADD(Uint32, Uint32); return;
    case MLDataType::Uint64:        MLCPUCONTEXT_DISPATCH_ADD(Uint64, Uint64); return;
    }
#undef MLCPUCONTEXT_DISPATCH_ADD
    RAD_UNREACHABLE();
}

} // namespace rad
