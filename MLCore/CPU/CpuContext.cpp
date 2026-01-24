#include <MLCore/CPU/CpuContext.h>
#include <MLCore/CPU/CpuTensorOp.h>
#include <random>

namespace ML
{

CpuContext::CpuContext(rad::Ref<CpuDevice> device) :
    Context(std::move(device))
{
}

CpuContext::~CpuContext()
{
}

void CpuContext::Fill(Tensor& input, const Scalar& value)
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
    case DataType::Complex32:   ML_CPU_DISPATCH_FILL_CONSTANT(rad::Complex32); return;
    case DataType::Complex64:   ML_CPU_DISPATCH_FILL_CONSTANT(rad::Complex64); return;
    case DataType::Complex128:  ML_CPU_DISPATCH_FILL_CONSTANT(rad::Complex128); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_FILL_CONSTANT(rad::BFloat16); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_FILL_CONSTANT(rad::Float8E4M3); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_FILL_CONSTANT(rad::Float8E5M2); return;
    }
#undef ML_CPU_DISPATCH_FILL_CONSTANT
    RAD_UNREACHABLE();
}

void CpuContext::Random(Tensor& input, const Scalar& from, const Scalar& to)
{
    thread_local std::default_random_engine rng;
#define ML_CPU_DISPATCH_RANDOM(DataType, ComputeType, Distribution, DefaultMaxValue)                \
    {                                                                                               \
        ComputeType min = ComputeType(from);                                                        \
        ComputeType max = ComputeType(to);                                                          \
        if ((min == 0) && to.IsNone())                                                              \
        {                                                                                           \
            max = ComputeType(DefaultMaxValue);                                                     \
        }                                                                                           \
        Distribution dist(min, max);                                                                \
        CpuTensorOpForEach<DataType>()(input, [&]() { return DataType(dist(rng)); });               \
    }

    switch (input.m_dataType)
    {
    case DataType::Float16:
        ML_CPU_DISPATCH_RANDOM(rad::Float16, rad::Float32, std::uniform_real_distribution<rad::Float32>, std::pow(2.0, 11.0));
        return;
    case DataType::Float32:
        ML_CPU_DISPATCH_RANDOM(rad::Float32, rad::Float32, std::uniform_real_distribution<rad::Float32>, std::pow(2.0, 24.0));
        return;
    case DataType::Float64:
        ML_CPU_DISPATCH_RANDOM(rad::Float64, rad::Float64, std::uniform_real_distribution<rad::Float64>, std::pow(2.0, 53.0));
        return;
    case DataType::Sint8:
        ML_CPU_DISPATCH_RANDOM(rad::Sint8, rad::Sint32, std::uniform_int_distribution<int32_t>, INT8_MAX);
        return;
    case DataType::Sint16:
        ML_CPU_DISPATCH_RANDOM(rad::Sint16, rad::Sint32, std::uniform_int_distribution<int32_t>, INT16_MAX);
        return;
    case DataType::Sint32:
        ML_CPU_DISPATCH_RANDOM(rad::Sint32, rad::Sint32, std::uniform_int_distribution<int32_t>, INT32_MAX);
        return;
    case DataType::Sint64:
        ML_CPU_DISPATCH_RANDOM(rad::Sint64, rad::Sint64, std::uniform_int_distribution<int64_t>, INT64_MAX);
        return;
    case DataType::Uint8:
        ML_CPU_DISPATCH_RANDOM(rad::Uint8, rad::Uint32, std::uniform_int_distribution<uint32_t>, UINT8_MAX);
        return;
    case DataType::Uint16:
        ML_CPU_DISPATCH_RANDOM(rad::Uint16, rad::Uint32, std::uniform_int_distribution<uint32_t>, UINT16_MAX);
        return;
    case DataType::Uint32:
        ML_CPU_DISPATCH_RANDOM(rad::Uint32, rad::Uint32, std::uniform_int_distribution<uint32_t>, UINT32_MAX);
        return;
    case DataType::Uint64:
        ML_CPU_DISPATCH_RANDOM(rad::Uint64, rad::Uint64, std::uniform_int_distribution<uint64_t>, UINT64_MAX);
        return;
    case DataType::BFloat16:
        ML_CPU_DISPATCH_RANDOM(rad::BFloat16, rad::Float32, std::uniform_real_distribution<rad::Float32>, std::pow(2.0, 8.0));
        return;
    case DataType::Float8E4M3:
        ML_CPU_DISPATCH_RANDOM(rad::Float8E4M3, rad::Float32, std::uniform_real_distribution<rad::Float32>, std::pow(2.0, 4.0));
        return;
    case DataType::Float8E5M2:
        ML_CPU_DISPATCH_RANDOM(rad::Float8E5M2, rad::Float32, std::uniform_real_distribution<rad::Float32>, std::pow(2.0, 3.0));
        return;
    }
#undef ML_CPU_DISPATCH_RANDOM
    RAD_UNREACHABLE();
}

void CpuContext::Add(const Tensor& input, const Scalar& other, Tensor& output)
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
    case DataType::Complex32:   ML_CPU_DISPATCH_ADD_SCALAR(rad::Complex32, rad::Complex64); return;
    case DataType::Complex64:   ML_CPU_DISPATCH_ADD_SCALAR(rad::Complex64, rad::Complex64); return;
    case DataType::Complex128:  ML_CPU_DISPATCH_ADD_SCALAR(rad::Complex128, rad::Complex128); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_ADD_SCALAR(rad::BFloat16, rad::Float32); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_ADD_SCALAR(rad::Float8E4M3, rad::Float32); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_ADD_SCALAR(rad::Float8E5M2, rad::Float32); return;
    }
#undef ML_CPU_DISPATCH_ADD_SCALAR
    RAD_UNREACHABLE();
}

void CpuContext::Add(const Tensor& input, const Tensor& other, const Scalar& alpha, Tensor& output)
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
    case DataType::Complex32:   ML_CPU_DISPATCH_ADD(rad::Complex32, rad::Complex64); return;
    case DataType::Complex64:   ML_CPU_DISPATCH_ADD(rad::Complex64, rad::Complex64); return;
    case DataType::Complex128:  ML_CPU_DISPATCH_ADD(rad::Complex128, rad::Complex128); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_ADD(rad::BFloat16, rad::Float32); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_ADD(rad::Float8E4M3, rad::Float32); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_ADD(rad::Float8E5M2, rad::Float32); return;
    }
#undef ML_CPU_DISPATCH_ADD
    RAD_UNREACHABLE();
}

void CpuContext::Subtract(const Tensor& input, const Scalar& other, Tensor& output)
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
    case DataType::Complex32:   ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::Complex32, rad::Complex64); return;
    case DataType::Complex64:   ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::Complex64, rad::Complex64); return;
    case DataType::Complex128:  ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::Complex128, rad::Complex128); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::BFloat16, rad::Float32); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::Float8E4M3, rad::Float32); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_SUBTRACT_SCALAR(rad::Float8E5M2, rad::Float32); return;
    }
#undef ML_CPU_DISPATCH_SUBTRACT_SCALAR
    RAD_UNREACHABLE();
}

void CpuContext::Subtract(const Tensor& input, const Tensor& other, const Scalar& alpha, Tensor& output)
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
    case DataType::Complex32:   ML_CPU_DISPATCH_SUBTRACT(rad::Complex32, rad::Complex64); return;
    case DataType::Complex64:   ML_CPU_DISPATCH_SUBTRACT(rad::Complex64, rad::Complex64); return;
    case DataType::Complex128:  ML_CPU_DISPATCH_SUBTRACT(rad::Complex128, rad::Complex128); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_SUBTRACT(rad::BFloat16, rad::Float32); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_SUBTRACT(rad::Float8E4M3, rad::Float32); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_SUBTRACT(rad::Float8E5M2, rad::Float32); return;
    }
#undef ML_CPU_DISPATCH_SUBTRACT
    RAD_UNREACHABLE();
}

void CpuContext::Multiply(const Tensor& input, const Scalar& other, Tensor& output)
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
    case DataType::Complex32:   ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::Complex32, rad::Complex64); return;
    case DataType::Complex64:   ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::Complex64, rad::Complex64); return;
    case DataType::Complex128:  ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::Complex128, rad::Complex128); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::BFloat16, rad::Float32); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::Float8E4M3, rad::Float32); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_MULTIPLY_SCALAR(rad::Float8E5M2, rad::Float32); return;
    }
#undef ML_CPU_DISPATCH_MULTIPLY_SCALAR
    RAD_UNREACHABLE();
}

void CpuContext::Multiply(const Tensor& input, const Tensor& other, Tensor& output)
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
    case DataType::Complex32:   ML_CPU_DISPATCH_MULTIPLY(rad::Complex32, rad::Complex64); return;
    case DataType::Complex64:   ML_CPU_DISPATCH_MULTIPLY(rad::Complex64, rad::Complex64); return;
    case DataType::Complex128:  ML_CPU_DISPATCH_MULTIPLY(rad::Complex128, rad::Complex128); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_MULTIPLY(rad::BFloat16, rad::Float32); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_MULTIPLY(rad::Float8E4M3, rad::Float32); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_MULTIPLY(rad::Float8E5M2, rad::Float32); return;
    }
#undef ML_CPU_DISPATCH_MULTIPLY
    RAD_UNREACHABLE();
}

void CpuContext::Divide(const Tensor& input, const Scalar& other, Tensor& output)
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
    case DataType::Complex32:   ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::Complex32, rad::Complex64); return;
    case DataType::Complex64:   ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::Complex64, rad::Complex64); return;
    case DataType::Complex128:  ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::Complex128, rad::Complex128); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::BFloat16, rad::Float32); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::Float8E4M3, rad::Float32); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_DIVIDE_SCALAR(rad::Float8E5M2, rad::Float32); return;
    }
#undef ML_CPU_DISPATCH_DIVIDE_SCALAR
    RAD_UNREACHABLE();
}

void CpuContext::Divide(const Tensor& input, const Tensor& other, Tensor& output)
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
    case DataType::Complex32:   ML_CPU_DISPATCH_DIVIDE(rad::Complex32, rad::Complex64); return;
    case DataType::Complex64:   ML_CPU_DISPATCH_DIVIDE(rad::Complex64, rad::Complex64); return;
    case DataType::Complex128:  ML_CPU_DISPATCH_DIVIDE(rad::Complex128, rad::Complex128); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_DIVIDE(rad::BFloat16, rad::Float32); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_DIVIDE(rad::Float8E4M3, rad::Float32); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_DIVIDE(rad::Float8E5M2, rad::Float32); return;
    }
#undef ML_CPU_DISPATCH_DIVIDE
    RAD_UNREACHABLE();
}

template <std::floating_point T>
T remainder(T a, T b)
{
    T r = std::fmod(a, b);
    if ((r != 0) && ((a < 0) != (b < 0)))
    {
        r += b;
    }
    return r;
}

template <std::signed_integral T>
T remainder(T a, T b)
{
    T r = a % b;
    if ((r != 0) && ((a < 0) != (b < 0)))
    {
        r += b;
    }
    return r;
}

template <std::unsigned_integral T>
T remainder(T a, T b)
{
    return a % b;
}

void CpuContext::Remainder(const Tensor& input, const Scalar& other, Tensor& output)
{
#define ML_CPU_DISPATCH_REMAINDER_SCALAR(DataType, ComputeType)    \
    CpuTensorOpElementWiseUnary<DataType, ComputeType>()(input, output, [&](ComputeType x) { return remainder(x, ComputeType(other)); });
    switch (input.m_dataType)
    {
    case DataType::Float16:     ML_CPU_DISPATCH_REMAINDER_SCALAR(rad::Float16, rad::Float32); return;
    case DataType::Float32:     ML_CPU_DISPATCH_REMAINDER_SCALAR(rad::Float32, rad::Float32); return;
    case DataType::Float64:     ML_CPU_DISPATCH_REMAINDER_SCALAR(rad::Float64, rad::Float64); return;
    case DataType::Sint8:       ML_CPU_DISPATCH_REMAINDER_SCALAR(rad::Sint8, rad::Sint8); return;
    case DataType::Sint16:      ML_CPU_DISPATCH_REMAINDER_SCALAR(rad::Sint16, rad::Sint16); return;
    case DataType::Sint32:      ML_CPU_DISPATCH_REMAINDER_SCALAR(rad::Sint32, rad::Sint32); return;
    case DataType::Sint64:      ML_CPU_DISPATCH_REMAINDER_SCALAR(rad::Sint64, rad::Sint64); return;
    case DataType::Uint8:       ML_CPU_DISPATCH_REMAINDER_SCALAR(rad::Uint8, rad::Uint8); return;
    case DataType::Uint16:      ML_CPU_DISPATCH_REMAINDER_SCALAR(rad::Uint16, rad::Uint16); return;
    case DataType::Uint32:      ML_CPU_DISPATCH_REMAINDER_SCALAR(rad::Uint32, rad::Uint32); return;
    case DataType::Uint64:      ML_CPU_DISPATCH_REMAINDER_SCALAR(rad::Uint64, rad::Uint64); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_REMAINDER_SCALAR(rad::BFloat16, rad::Float32); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_REMAINDER_SCALAR(rad::Float8E4M3, rad::Float32); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_REMAINDER_SCALAR(rad::Float8E5M2, rad::Float32); return;
    }
#undef ML_CPU_DISPATCH_REMAINDER_SCALAR
    RAD_UNREACHABLE();
}

void CpuContext::Remainder(const Tensor& input, const Tensor& other, Tensor& output)
{
#define ML_CPU_DISPATCH_REMAINDER(DataType, ComputeType)    \
    CpuTensorOpElementWiseBinary<DataType, ComputeType>()(input, other, output, [&](ComputeType a, ComputeType b) { return remainder(a, b); });
    switch (input.m_dataType)
    {
    case DataType::Float16:     ML_CPU_DISPATCH_REMAINDER(rad::Float16, rad::Float32); return;
    case DataType::Float32:     ML_CPU_DISPATCH_REMAINDER(rad::Float32, rad::Float32); return;
    case DataType::Float64:     ML_CPU_DISPATCH_REMAINDER(rad::Float64, rad::Float64); return;
    case DataType::Sint8:       ML_CPU_DISPATCH_REMAINDER(rad::Sint8, rad::Sint8); return;
    case DataType::Sint16:      ML_CPU_DISPATCH_REMAINDER(rad::Sint16, rad::Sint16); return;
    case DataType::Sint32:      ML_CPU_DISPATCH_REMAINDER(rad::Sint32, rad::Sint32); return;
    case DataType::Sint64:      ML_CPU_DISPATCH_REMAINDER(rad::Sint64, rad::Sint64); return;
    case DataType::Uint8:       ML_CPU_DISPATCH_REMAINDER(rad::Uint8, rad::Uint8); return;
    case DataType::Uint16:      ML_CPU_DISPATCH_REMAINDER(rad::Uint16, rad::Uint16); return;
    case DataType::Uint32:      ML_CPU_DISPATCH_REMAINDER(rad::Uint32, rad::Uint32); return;
    case DataType::Uint64:      ML_CPU_DISPATCH_REMAINDER(rad::Uint64, rad::Uint64); return;
    case DataType::BFloat16:    ML_CPU_DISPATCH_REMAINDER(rad::BFloat16, rad::Float32); return;
    case DataType::Float8E4M3:  ML_CPU_DISPATCH_REMAINDER(rad::Float8E4M3, rad::Float32); return;
    case DataType::Float8E5M2:  ML_CPU_DISPATCH_REMAINDER(rad::Float8E5M2, rad::Float32); return;
    }
#undef ML_CPU_DISPATCH_REMAINDER
    RAD_UNREACHABLE();
}

void CpuContext::BitwiseNot(const Tensor& input, Tensor& output)
{
    assert(input.IsInteger() && output.IsInteger());
#define ML_CPU_DISPATCH_BITWISE_NOT(DataType, ComputeType)    \
    CpuTensorOpElementWiseUnary<DataType, ComputeType>()(input, output, [&](ComputeType x) { return (~x); });
    switch (input.m_dataType)
    {
    case DataType::Sint8:       ML_CPU_DISPATCH_BITWISE_NOT(rad::Sint8, rad::Sint8); return;
    case DataType::Sint16:      ML_CPU_DISPATCH_BITWISE_NOT(rad::Sint16, rad::Sint16); return;
    case DataType::Sint32:      ML_CPU_DISPATCH_BITWISE_NOT(rad::Sint32, rad::Sint32); return;
    case DataType::Sint64:      ML_CPU_DISPATCH_BITWISE_NOT(rad::Sint64, rad::Sint64); return;
    case DataType::Uint8:       ML_CPU_DISPATCH_BITWISE_NOT(rad::Uint8, rad::Uint8); return;
    case DataType::Uint16:      ML_CPU_DISPATCH_BITWISE_NOT(rad::Uint16, rad::Uint16); return;
    case DataType::Uint32:      ML_CPU_DISPATCH_BITWISE_NOT(rad::Uint32, rad::Uint32); return;
    case DataType::Uint64:      ML_CPU_DISPATCH_BITWISE_NOT(rad::Uint64, rad::Uint64); return;
    }
#undef ML_CPU_DISPATCH_BITWISE_NOT
    RAD_UNREACHABLE();
}

void CpuContext::BitwiseAnd(const Tensor& input, const Scalar& other, Tensor& output)
{
    assert(input.IsInteger() && other.IsInteger());
#define ML_CPU_DISPATCH_BITWISE_AND(DataType, ComputeType)    \
    CpuTensorOpElementWiseUnary<DataType, ComputeType>()(input, output, [&](ComputeType x) { return (x & ComputeType(other)); });
    switch (input.m_dataType)
    {
    case DataType::Sint8:       ML_CPU_DISPATCH_BITWISE_AND(rad::Sint8, rad::Sint8); return;
    case DataType::Sint16:      ML_CPU_DISPATCH_BITWISE_AND(rad::Sint16, rad::Sint16); return;
    case DataType::Sint32:      ML_CPU_DISPATCH_BITWISE_AND(rad::Sint32, rad::Sint32); return;
    case DataType::Sint64:      ML_CPU_DISPATCH_BITWISE_AND(rad::Sint64, rad::Sint64); return;
    case DataType::Uint8:       ML_CPU_DISPATCH_BITWISE_AND(rad::Uint8, rad::Uint8); return;
    case DataType::Uint16:      ML_CPU_DISPATCH_BITWISE_AND(rad::Uint16, rad::Uint16); return;
    case DataType::Uint32:      ML_CPU_DISPATCH_BITWISE_AND(rad::Uint32, rad::Uint32); return;
    case DataType::Uint64:      ML_CPU_DISPATCH_BITWISE_AND(rad::Uint64, rad::Uint64); return;
    }
#undef ML_CPU_DISPATCH_BITWISE_AND
    RAD_UNREACHABLE();
}

void CpuContext::BitwiseAnd(const Tensor& input, const Tensor& other, Tensor& output)
{
    assert(input.IsInteger() && other.IsInteger());
#define ML_CPU_DISPATCH_BITWISE_AND(DataType, ComputeType)    \
    CpuTensorOpElementWiseBinary<DataType, ComputeType>()(input, other, output, [&](ComputeType a, ComputeType b) { return (a & b); });
    switch (input.m_dataType)
    {
    case DataType::Sint8:       ML_CPU_DISPATCH_BITWISE_AND(rad::Sint8, rad::Sint8); return;
    case DataType::Sint16:      ML_CPU_DISPATCH_BITWISE_AND(rad::Sint16, rad::Sint16); return;
    case DataType::Sint32:      ML_CPU_DISPATCH_BITWISE_AND(rad::Sint32, rad::Sint32); return;
    case DataType::Sint64:      ML_CPU_DISPATCH_BITWISE_AND(rad::Sint64, rad::Sint64); return;
    case DataType::Uint8:       ML_CPU_DISPATCH_BITWISE_AND(rad::Uint8, rad::Uint8); return;
    case DataType::Uint16:      ML_CPU_DISPATCH_BITWISE_AND(rad::Uint16, rad::Uint16); return;
    case DataType::Uint32:      ML_CPU_DISPATCH_BITWISE_AND(rad::Uint32, rad::Uint32); return;
    case DataType::Uint64:      ML_CPU_DISPATCH_BITWISE_AND(rad::Uint64, rad::Uint64); return;
    }
#undef ML_CPU_DISPATCH_BITWISE_AND
    RAD_UNREACHABLE();
}

void CpuContext::BitwiseOr(const Tensor& input, const Scalar& other, Tensor& output)
{
    assert(input.IsInteger() && other.IsInteger());
#define ML_CPU_DISPATCH_BITWISE_OR(DataType, ComputeType)    \
    CpuTensorOpElementWiseUnary<DataType, ComputeType>()(input, output, [&](ComputeType x) { return (x | ComputeType(other)); });
    switch (input.m_dataType)
    {
    case DataType::Sint8:       ML_CPU_DISPATCH_BITWISE_OR(rad::Sint8, rad::Sint8); return;
    case DataType::Sint16:      ML_CPU_DISPATCH_BITWISE_OR(rad::Sint16, rad::Sint16); return;
    case DataType::Sint32:      ML_CPU_DISPATCH_BITWISE_OR(rad::Sint32, rad::Sint32); return;
    case DataType::Sint64:      ML_CPU_DISPATCH_BITWISE_OR(rad::Sint64, rad::Sint64); return;
    case DataType::Uint8:       ML_CPU_DISPATCH_BITWISE_OR(rad::Uint8, rad::Uint8); return;
    case DataType::Uint16:      ML_CPU_DISPATCH_BITWISE_OR(rad::Uint16, rad::Uint16); return;
    case DataType::Uint32:      ML_CPU_DISPATCH_BITWISE_OR(rad::Uint32, rad::Uint32); return;
    case DataType::Uint64:      ML_CPU_DISPATCH_BITWISE_OR(rad::Uint64, rad::Uint64); return;
    }
#undef ML_CPU_DISPATCH_BITWISE_OR
    RAD_UNREACHABLE();
}

void CpuContext::BitwiseOr(const Tensor& input, const Tensor& other, Tensor& output)
{
    assert(input.IsInteger() && other.IsInteger());
#define ML_CPU_DISPATCH_BITWISE_OR(DataType, ComputeType)    \
    CpuTensorOpElementWiseBinary<DataType, ComputeType>()(input, other, output, [&](ComputeType a, ComputeType b) { return (a | b); });
    switch (input.m_dataType)
    {
    case DataType::Sint8:       ML_CPU_DISPATCH_BITWISE_OR(rad::Sint8, rad::Sint8); return;
    case DataType::Sint16:      ML_CPU_DISPATCH_BITWISE_OR(rad::Sint16, rad::Sint16); return;
    case DataType::Sint32:      ML_CPU_DISPATCH_BITWISE_OR(rad::Sint32, rad::Sint32); return;
    case DataType::Sint64:      ML_CPU_DISPATCH_BITWISE_OR(rad::Sint64, rad::Sint64); return;
    case DataType::Uint8:       ML_CPU_DISPATCH_BITWISE_OR(rad::Uint8, rad::Uint8); return;
    case DataType::Uint16:      ML_CPU_DISPATCH_BITWISE_OR(rad::Uint16, rad::Uint16); return;
    case DataType::Uint32:      ML_CPU_DISPATCH_BITWISE_OR(rad::Uint32, rad::Uint32); return;
    case DataType::Uint64:      ML_CPU_DISPATCH_BITWISE_OR(rad::Uint64, rad::Uint64); return;
    }
#undef ML_CPU_DISPATCH_BITWISE_OR
    RAD_UNREACHABLE();
}

void CpuContext::BitwiseXor(const Tensor& input, const Scalar& other, Tensor& output)
{
    assert(input.IsInteger() && other.IsInteger());
#define ML_CPU_DISPATCH_BITWISE_XOR(DataType, ComputeType)    \
    CpuTensorOpElementWiseUnary<DataType, ComputeType>()(input, output, [&](ComputeType x) { return (x ^ ComputeType(other)); });
    switch (input.m_dataType)
    {
    case DataType::Sint8:       ML_CPU_DISPATCH_BITWISE_XOR(rad::Sint8, rad::Sint8); return;
    case DataType::Sint16:      ML_CPU_DISPATCH_BITWISE_XOR(rad::Sint16, rad::Sint16); return;
    case DataType::Sint32:      ML_CPU_DISPATCH_BITWISE_XOR(rad::Sint32, rad::Sint32); return;
    case DataType::Sint64:      ML_CPU_DISPATCH_BITWISE_XOR(rad::Sint64, rad::Sint64); return;
    case DataType::Uint8:       ML_CPU_DISPATCH_BITWISE_XOR(rad::Uint8, rad::Uint8); return;
    case DataType::Uint16:      ML_CPU_DISPATCH_BITWISE_XOR(rad::Uint16, rad::Uint16); return;
    case DataType::Uint32:      ML_CPU_DISPATCH_BITWISE_XOR(rad::Uint32, rad::Uint32); return;
    case DataType::Uint64:      ML_CPU_DISPATCH_BITWISE_XOR(rad::Uint64, rad::Uint64); return;
    }
#undef ML_CPU_DISPATCH_BITWISE_XOR
    RAD_UNREACHABLE();
}

void CpuContext::BitwiseXor(const Tensor& input, const Tensor& other, Tensor& output)
{
    assert(input.IsInteger() && other.IsInteger());
#define ML_CPU_DISPATCH_BITWISE_XOR(DataType, ComputeType)    \
    CpuTensorOpElementWiseBinary<DataType, ComputeType>()(input, other, output, [&](ComputeType a, ComputeType b) { return (a ^ b); });
    switch (input.m_dataType)
    {
    case DataType::Sint8:       ML_CPU_DISPATCH_BITWISE_XOR(rad::Sint8, rad::Sint8); return;
    case DataType::Sint16:      ML_CPU_DISPATCH_BITWISE_XOR(rad::Sint16, rad::Sint16); return;
    case DataType::Sint32:      ML_CPU_DISPATCH_BITWISE_XOR(rad::Sint32, rad::Sint32); return;
    case DataType::Sint64:      ML_CPU_DISPATCH_BITWISE_XOR(rad::Sint64, rad::Sint64); return;
    case DataType::Uint8:       ML_CPU_DISPATCH_BITWISE_XOR(rad::Uint8, rad::Uint8); return;
    case DataType::Uint16:      ML_CPU_DISPATCH_BITWISE_XOR(rad::Uint16, rad::Uint16); return;
    case DataType::Uint32:      ML_CPU_DISPATCH_BITWISE_XOR(rad::Uint32, rad::Uint32); return;
    case DataType::Uint64:      ML_CPU_DISPATCH_BITWISE_XOR(rad::Uint64, rad::Uint64); return;
    }
#undef ML_CPU_DISPATCH_BITWISE_XOR
    RAD_UNREACHABLE();
}

} // namespace ML
