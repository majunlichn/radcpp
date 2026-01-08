#include <MLCore/CPU/CpuContext.h>
#include <MLCore/CPU/CpuTensorOp.h>

namespace ML
{

CpuContext::CpuContext(rad::Ref<CpuDevice> device) :
    Context(std::move(device))
{
}

CpuContext::~CpuContext()
{
}

void CpuContext::FillConstant(const Tensor& input, Scalar value)
{
#define ML_CPU_DISPATCH_FILL_CONSTANT(DataType)   \
    CpuTensorOpForEach<DataType>()(input, [&]() { return DataType(value); })
    switch (input.m_dataType)
    {
    case DataType::Float16:     ML_CPU_DISPATCH_FILL_CONSTANT(rad::Float16); return;
    case DataType::Float32:     ML_CPU_DISPATCH_FILL_CONSTANT(rad::Float32); return;
    case DataType::Float64:     ML_CPU_DISPATCH_FILL_CONSTANT(rad::Float64); return;
    case DataType::Sint8:       ML_CPU_DISPATCH_FILL_CONSTANT(rad::Sint8); return;
    case DataType::Sint16:      ML_CPU_DISPATCH_FILL_CONSTANT(rad::Sint16); return;
    case DataType::Sint32:      ML_CPU_DISPATCH_FILL_CONSTANT(rad::Sint32); return;
    case DataType::Sint64:      ML_CPU_DISPATCH_FILL_CONSTANT(rad::Sint64); return;
    case DataType::Uint8:       ML_CPU_DISPATCH_FILL_CONSTANT(rad::Uint8); return;
    case DataType::Uint16:      ML_CPU_DISPATCH_FILL_CONSTANT(rad::Uint16); return;
    case DataType::Uint32:      ML_CPU_DISPATCH_FILL_CONSTANT(rad::Uint32); return;
    case DataType::Uint64:      ML_CPU_DISPATCH_FILL_CONSTANT(rad::Uint64); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_FILL_CONSTANT(rad::BFloat16); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_FILL_CONSTANT(rad::Float8E4M3); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_FILL_CONSTANT(rad::Float8E5M2); return;
    }
#undef ML_CPU_DISPATCH_FILL_CONSTANT
    RAD_UNREACHABLE();
}

void CpuContext::Add(const Tensor& input, const Scalar other, const Tensor& output)
{
#define ML_CPU_DISPATCH_ADD_SCALAR(DataType, ComputeType)    \
    CpuTensorOpElementWiseUnary<DataType, ComputeType>()(input, output, [&](ComputeType x) { return x + ComputeType(other); });
    switch (input.m_dataType)
    {
    case DataType::Float16:     ML_CPU_DISPATCH_ADD_SCALAR(rad::Float16, rad::Float32); return;
    case DataType::Float32:     ML_CPU_DISPATCH_ADD_SCALAR(rad::Float32, rad::Float32); return;
    case DataType::Float64:     ML_CPU_DISPATCH_ADD_SCALAR(rad::Float64, rad::Float64); return;
    case DataType::Sint8:       ML_CPU_DISPATCH_ADD_SCALAR(rad::Sint8, rad::Sint8); return;
    case DataType::Sint16:      ML_CPU_DISPATCH_ADD_SCALAR(rad::Sint16, rad::Sint16); return;
    case DataType::Sint32:      ML_CPU_DISPATCH_ADD_SCALAR(rad::Sint32, rad::Sint32); return;
    case DataType::Sint64:      ML_CPU_DISPATCH_ADD_SCALAR(rad::Sint64, rad::Sint64); return;
    case DataType::Uint8:       ML_CPU_DISPATCH_ADD_SCALAR(rad::Uint8, rad::Uint8); return;
    case DataType::Uint16:      ML_CPU_DISPATCH_ADD_SCALAR(rad::Uint16, rad::Uint16); return;
    case DataType::Uint32:      ML_CPU_DISPATCH_ADD_SCALAR(rad::Uint32, rad::Uint32); return;
    case DataType::Uint64:      ML_CPU_DISPATCH_ADD_SCALAR(rad::Uint64, rad::Uint64); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_ADD_SCALAR(rad::BFloat16, rad::Float32); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_ADD_SCALAR(rad::Float8E4M3, rad::Float32); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_ADD_SCALAR(rad::Float8E5M2, rad::Float32); return;
    }
#undef ML_CPU_DISPATCH_ADD_SCALAR
    RAD_UNREACHABLE();
}

void CpuContext::Add(const Tensor& input, const Tensor& other, const Scalar alpha, const Tensor& output)
{
#define ML_CPU_DISPATCH_ADD(DataType, ComputeType)    \
    CpuTensorOpElementWiseBinary<DataType, ComputeType>()(input, other, output, [&](ComputeType a, ComputeType b) { return a + ComputeType(alpha) * b; });
    switch (input.m_dataType)
    {
    case DataType::Float16:     ML_CPU_DISPATCH_ADD(rad::Float16, rad::Float32); return;
    case DataType::Float32:     ML_CPU_DISPATCH_ADD(rad::Float32, rad::Float32); return;
    case DataType::Float64:     ML_CPU_DISPATCH_ADD(rad::Float64, rad::Float64); return;
    case DataType::Sint8:       ML_CPU_DISPATCH_ADD(rad::Sint8, rad::Sint8); return;
    case DataType::Sint16:      ML_CPU_DISPATCH_ADD(rad::Sint16, rad::Sint16); return;
    case DataType::Sint32:      ML_CPU_DISPATCH_ADD(rad::Sint32, rad::Sint32); return;
    case DataType::Sint64:      ML_CPU_DISPATCH_ADD(rad::Sint64, rad::Sint64); return;
    case DataType::Uint8:       ML_CPU_DISPATCH_ADD(rad::Uint8, rad::Uint8); return;
    case DataType::Uint16:      ML_CPU_DISPATCH_ADD(rad::Uint16, rad::Uint16); return;
    case DataType::Uint32:      ML_CPU_DISPATCH_ADD(rad::Uint32, rad::Uint32); return;
    case DataType::Uint64:      ML_CPU_DISPATCH_ADD(rad::Uint64, rad::Uint64); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_ADD(rad::BFloat16, rad::Float32); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_ADD(rad::Float8E4M3, rad::Float32); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_ADD(rad::Float8E5M2, rad::Float32); return;
    }
#undef ML_CPU_DISPATCH_ADD
    RAD_UNREACHABLE();
}

void CpuContext::Subtract(const Tensor& input, const Scalar other, const Tensor& output)
{
#define ML_CPU_DISPATCH_SUBTRACT_SCALAR(DataType, ComputeType)    \
    CpuTensorOpElementWiseUnary<DataType, ComputeType>()(input, output, [&](ComputeType x) { return x - ComputeType(other); });
    switch (input.m_dataType)
    {
    case DataType::Float16:     ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::Float16, rad::Float32); return;
    case DataType::Float32:     ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::Float32, rad::Float32); return;
    case DataType::Float64:     ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::Float64, rad::Float64); return;
    case DataType::Sint8:       ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::Sint8, rad::Sint8); return;
    case DataType::Sint16:      ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::Sint16, rad::Sint16); return;
    case DataType::Sint32:      ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::Sint32, rad::Sint32); return;
    case DataType::Sint64:      ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::Sint64, rad::Sint64); return;
    case DataType::Uint8:       ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::Uint8, rad::Uint8); return;
    case DataType::Uint16:      ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::Uint16, rad::Uint16); return;
    case DataType::Uint32:      ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::Uint32, rad::Uint32); return;
    case DataType::Uint64:      ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::Uint64, rad::Uint64); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::BFloat16, rad::Float32); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::Float8E4M3, rad::Float32); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::Float8E5M2, rad::Float32); return;
    }
#undef ML_CPU_DISPATCH_SUBTRACT_SCALAR
    RAD_UNREACHABLE();
}

void CpuContext::Subtract(const Tensor& input, const Tensor& other, const Scalar alpha, const Tensor& output)
{
#define ML_CPU_DISPATCH_SUBTRACT(DataType, ComputeType)    \
    CpuTensorOpElementWiseBinary<DataType, ComputeType>()(input, other, output, [&](ComputeType a, ComputeType b) { return a - ComputeType(alpha) * b; });
    switch (input.m_dataType)
    {
    case DataType::Float16:     ML_CPU_DISPATCH_SUBTRACT(rad::Float16, rad::Float32); return;
    case DataType::Float32:     ML_CPU_DISPATCH_SUBTRACT(rad::Float32, rad::Float32); return;
    case DataType::Float64:     ML_CPU_DISPATCH_SUBTRACT(rad::Float64, rad::Float64); return;
    case DataType::Sint8:       ML_CPU_DISPATCH_SUBTRACT(rad::Sint8, rad::Sint8); return;
    case DataType::Sint16:      ML_CPU_DISPATCH_SUBTRACT(rad::Sint16, rad::Sint16); return;
    case DataType::Sint32:      ML_CPU_DISPATCH_SUBTRACT(rad::Sint32, rad::Sint32); return;
    case DataType::Sint64:      ML_CPU_DISPATCH_SUBTRACT(rad::Sint64, rad::Sint64); return;
    case DataType::Uint8:       ML_CPU_DISPATCH_SUBTRACT(rad::Uint8, rad::Uint8); return;
    case DataType::Uint16:      ML_CPU_DISPATCH_SUBTRACT(rad::Uint16, rad::Uint16); return;
    case DataType::Uint32:      ML_CPU_DISPATCH_SUBTRACT(rad::Uint32, rad::Uint32); return;
    case DataType::Uint64:      ML_CPU_DISPATCH_SUBTRACT(rad::Uint64, rad::Uint64); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_SUBTRACT(rad::BFloat16, rad::Float32); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_SUBTRACT(rad::Float8E4M3, rad::Float32); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_SUBTRACT(rad::Float8E5M2, rad::Float32); return;
    }
#undef ML_CPU_DISPATCH_SUBTRACT
    RAD_UNREACHABLE();
}

void CpuContext::Multiply(const Tensor& input, const Scalar other, const Tensor& output)
{
#define ML_CPU_DISPATCH_MULTIPLY_SCALAR(DataType, ComputeType)    \
    CpuTensorOpElementWiseUnary<DataType, ComputeType>()(input, output, [&](ComputeType x) { return x * ComputeType(other); });
    switch (input.m_dataType)
    {
    case DataType::Float16:     ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::Float16, rad::Float32); return;
    case DataType::Float32:     ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::Float32, rad::Float32); return;
    case DataType::Float64:     ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::Float64, rad::Float64); return;
    case DataType::Sint8:       ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::Sint8, rad::Sint8); return;
    case DataType::Sint16:      ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::Sint16, rad::Sint16); return;
    case DataType::Sint32:      ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::Sint32, rad::Sint32); return;
    case DataType::Sint64:      ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::Sint64, rad::Sint64); return;
    case DataType::Uint8:       ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::Uint8, rad::Uint8); return;
    case DataType::Uint16:      ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::Uint16, rad::Uint16); return;
    case DataType::Uint32:      ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::Uint32, rad::Uint32); return;
    case DataType::Uint64:      ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::Uint64, rad::Uint64); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::BFloat16, rad::Float32); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::Float8E4M3, rad::Float32); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::Float8E5M2, rad::Float32); return;
    }
#undef ML_CPU_DISPATCH_MULTIPLY_SCALAR
    RAD_UNREACHABLE();
}

void CpuContext::Multiply(const Tensor& input, const Tensor& other, const Tensor& output)
{
#define ML_CPU_DISPATCH_MULTIPLY(DataType, ComputeType)    \
    CpuTensorOpElementWiseBinary<DataType, ComputeType>()(input, other, output, [&](ComputeType a, ComputeType b) { return a * b; });
    switch (input.m_dataType)
    {
    case DataType::Float16:     ML_CPU_DISPATCH_MULTIPLY(rad::Float16, rad::Float32); return;
    case DataType::Float32:     ML_CPU_DISPATCH_MULTIPLY(rad::Float32, rad::Float32); return;
    case DataType::Float64:     ML_CPU_DISPATCH_MULTIPLY(rad::Float64, rad::Float64); return;
    case DataType::Sint8:       ML_CPU_DISPATCH_MULTIPLY(rad::Sint8, rad::Sint8); return;
    case DataType::Sint16:      ML_CPU_DISPATCH_MULTIPLY(rad::Sint16, rad::Sint16); return;
    case DataType::Sint32:      ML_CPU_DISPATCH_MULTIPLY(rad::Sint32, rad::Sint32); return;
    case DataType::Sint64:      ML_CPU_DISPATCH_MULTIPLY(rad::Sint64, rad::Sint64); return;
    case DataType::Uint8:       ML_CPU_DISPATCH_MULTIPLY(rad::Uint8, rad::Uint8); return;
    case DataType::Uint16:      ML_CPU_DISPATCH_MULTIPLY(rad::Uint16, rad::Uint16); return;
    case DataType::Uint32:      ML_CPU_DISPATCH_MULTIPLY(rad::Uint32, rad::Uint32); return;
    case DataType::Uint64:      ML_CPU_DISPATCH_MULTIPLY(rad::Uint64, rad::Uint64); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_MULTIPLY(rad::BFloat16, rad::Float32); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_MULTIPLY(rad::Float8E4M3, rad::Float32); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_MULTIPLY(rad::Float8E5M2, rad::Float32); return;
    }
#undef ML_CPU_DISPATCH_MULTIPLY
    RAD_UNREACHABLE();
}

void CpuContext::Divide(const Tensor& input, const Scalar other, const Tensor& output)
{
#define ML_CPU_DISPATCH_DIVIDE_SCALAR(DataType, ComputeType)    \
    CpuTensorOpElementWiseUnary<DataType, ComputeType>()(input, output, [&](ComputeType x) { return x / ComputeType(other); });
    switch (input.m_dataType)
    {
    case DataType::Float16:     ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::Float16, rad::Float32); return;
    case DataType::Float32:     ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::Float32, rad::Float32); return;
    case DataType::Float64:     ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::Float64, rad::Float64); return;
    case DataType::Sint8:       ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::Sint8, rad::Sint8); return;
    case DataType::Sint16:      ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::Sint16, rad::Sint16); return;
    case DataType::Sint32:      ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::Sint32, rad::Sint32); return;
    case DataType::Sint64:      ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::Sint64, rad::Sint64); return;
    case DataType::Uint8:       ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::Uint8, rad::Uint8); return;
    case DataType::Uint16:      ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::Uint16, rad::Uint16); return;
    case DataType::Uint32:      ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::Uint32, rad::Uint32); return;
    case DataType::Uint64:      ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::Uint64, rad::Uint64); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::BFloat16, rad::Float32); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::Float8E4M3, rad::Float32); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::Float8E5M2, rad::Float32); return;
    }
#undef ML_CPU_DISPATCH_DIVIDE_SCALAR
    RAD_UNREACHABLE();
}

void CpuContext::Divide(const Tensor& input, const Tensor& other, const Tensor& output)
{
#define ML_CPU_DISPATCH_DIVIDE(DataType, ComputeType)    \
    CpuTensorOpElementWiseBinary<DataType, ComputeType>()(input, other, output, [&](ComputeType a, ComputeType b) { return a / b; });
    switch (input.m_dataType)
    {
    case DataType::Float16:     ML_CPU_DISPATCH_DIVIDE(rad::Float16, rad::Float32); return;
    case DataType::Float32:     ML_CPU_DISPATCH_DIVIDE(rad::Float32, rad::Float32); return;
    case DataType::Float64:     ML_CPU_DISPATCH_DIVIDE(rad::Float64, rad::Float64); return;
    case DataType::Sint8:       ML_CPU_DISPATCH_DIVIDE(rad::Sint8, rad::Sint8); return;
    case DataType::Sint16:      ML_CPU_DISPATCH_DIVIDE(rad::Sint16, rad::Sint16); return;
    case DataType::Sint32:      ML_CPU_DISPATCH_DIVIDE(rad::Sint32, rad::Sint32); return;
    case DataType::Sint64:      ML_CPU_DISPATCH_DIVIDE(rad::Sint64, rad::Sint64); return;
    case DataType::Uint8:       ML_CPU_DISPATCH_DIVIDE(rad::Uint8, rad::Uint8); return;
    case DataType::Uint16:      ML_CPU_DISPATCH_DIVIDE(rad::Uint16, rad::Uint16); return;
    case DataType::Uint32:      ML_CPU_DISPATCH_DIVIDE(rad::Uint32, rad::Uint32); return;
    case DataType::Uint64:      ML_CPU_DISPATCH_DIVIDE(rad::Uint64, rad::Uint64); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_DIVIDE(rad::BFloat16, rad::Float32); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_DIVIDE(rad::Float8E4M3, rad::Float32); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_DIVIDE(rad::Float8E5M2, rad::Float32); return;
    }
#undef ML_CPU_DISPATCH_DIVIDE
    RAD_UNREACHABLE();
}

} // namespace ML
