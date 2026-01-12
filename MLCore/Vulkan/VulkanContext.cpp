#include <MLCore/Vulkan/VulkanContext.h>
#include <MLCore/Vulkan/VulkanDevice.h>
#include <MLCore/Vulkan/VulkanTensor.h>

namespace ML
{

VulkanContext::VulkanContext(rad::Ref<VulkanDevice> device) :
    Context(std::move(device))
{
    const DataType computableTypes[] = {
        DataType::Float16, DataType::Float32, DataType::Float64,
        DataType::Sint8, DataType::Sint16, DataType::Sint32, DataType::Sint64,
    };
    const DataType intTypes[] = {
        DataType::Sint8, DataType::Sint16, DataType::Sint32, DataType::Sint64,
        DataType::Uint8, DataType::Uint16, DataType::Uint32, DataType::Uint64,
    };
    const DataType sintTypes[] = {
        DataType::Sint8, DataType::Sint16, DataType::Sint32, DataType::Sint64,
    };
    const DataType uintTypes[] = {
        DataType::Uint8, DataType::Uint16, DataType::Uint32, DataType::Uint64,
    };
    m_opFill = RAD_NEW VulkanTensorOpForEach(this, "Fill");
    m_opAddScalar = RAD_NEW VulkanTensorOpElementWiseUnary(this, "AddScalar", computableTypes);
    m_opAdd = RAD_NEW VulkanTensorOpElementWiseBinary(this, "Add", computableTypes);
    m_opSubtractScalar = RAD_NEW VulkanTensorOpElementWiseUnary(this, "SubtractScalar", computableTypes);
    m_opSubtract = RAD_NEW VulkanTensorOpElementWiseBinary(this, "Subtract", computableTypes);
    m_opMultiplyScalar = RAD_NEW VulkanTensorOpElementWiseUnary(this, "MultiplyScalar", computableTypes);
    m_opMultiply = RAD_NEW VulkanTensorOpElementWiseBinary(this, "Multiply", computableTypes);
    m_opDivideScalar = RAD_NEW VulkanTensorOpElementWiseUnary(this, "DivideScalar", computableTypes);
    m_opDivide = RAD_NEW VulkanTensorOpElementWiseBinary(this, "Divide", computableTypes);
    m_opRemainderScalar = RAD_NEW VulkanTensorOpElementWiseUnary(this, "RemainderScalar", computableTypes);
    m_opRemainder = RAD_NEW VulkanTensorOpElementWiseBinary(this, "Remainder", computableTypes);
    m_opBitwiseAndScalar = RAD_NEW VulkanTensorOpElementWiseUnary(this, "BitwiseAndScalar", intTypes);
    m_opBitwiseAnd = RAD_NEW VulkanTensorOpElementWiseBinary(this, "BitwiseAnd", intTypes);
    m_opBitwiseOrScalar = RAD_NEW VulkanTensorOpElementWiseUnary(this, "BitwiseOrScalar", intTypes);
    m_opBitwiseOr = RAD_NEW VulkanTensorOpElementWiseBinary(this, "BitwiseOr", intTypes);
    m_opBitwiseXorScalar = RAD_NEW VulkanTensorOpElementWiseUnary(this, "BitwiseXorScalar", intTypes);
    m_opBitwiseXor = RAD_NEW VulkanTensorOpElementWiseBinary(this, "BitwiseXor", intTypes);
}

VulkanContext::~VulkanContext()
{
}

VulkanDevice* VulkanContext::GetDevice()
{
    return static_cast<VulkanDevice*>(m_device.get());
}

void VulkanContext::Fill(const Tensor& input, Scalar value)
{
    m_opFill->SetTensor(1, input);
    m_opFill->m_shaderUniforms.params.Set(input.m_dataType, value);
    m_opFill->Execute();
}

void VulkanContext::Add(const Tensor& input, const Scalar other, Tensor& output)
{
    assert(input.IsFloatingPoint() == other.IsFloatingPoint());
    m_opAddScalar->SetTensor(1, input);
    m_opAddScalar->SetTensor(2, output ? output : input);
    m_opAddScalar->m_shaderUniforms.params.Set(input.m_dataType, other);
    m_opAddScalar->Execute();
}

void VulkanContext::Add(const Tensor& input, const Tensor& other, const Scalar alpha, Tensor& output)
{
    assert(input.m_dataType == other.m_dataType);
    assert(input.IsFloatingPoint() == alpha.IsFloatingPoint());
    m_opAdd->SetTensor(1, input);
    m_opAdd->SetTensor(2, other);
    m_opAdd->SetTensor(3, output ? output : input);
    m_opAdd->m_shaderUniforms.params.Set(input.m_dataType, alpha);
    m_opAdd->Execute();
}

void VulkanContext::Subtract(const Tensor& input, const Scalar other, Tensor& output)
{
    assert(input.IsFloatingPoint() == other.IsFloatingPoint());
    m_opSubtractScalar->SetTensor(1, input);
    m_opSubtractScalar->SetTensor(2, output ? output : input);
    m_opSubtractScalar->m_shaderUniforms.params.Set(input.m_dataType, other);
    m_opSubtractScalar->Execute();
}

void VulkanContext::Subtract(const Tensor& input, const Tensor& other, const Scalar alpha, Tensor& output)
{
    assert(input.m_dataType == other.m_dataType);
    assert(input.IsFloatingPoint() == alpha.IsFloatingPoint());
    m_opSubtract->SetTensor(1, input);
    m_opSubtract->SetTensor(2, other);
    m_opSubtract->SetTensor(3, output ? output : input);
    m_opSubtract->m_shaderUniforms.params.Set(input.m_dataType, alpha);
    m_opSubtract->Execute();
}

void VulkanContext::Multiply(const Tensor& input, const Scalar other, Tensor& output)
{
    m_opMultiplyScalar->SetTensor(1, input);
    m_opMultiplyScalar->SetTensor(2, output ? output : input);
    m_opMultiplyScalar->m_shaderUniforms.params.Set(input.m_dataType, other);
    m_opMultiplyScalar->Execute();
}

void VulkanContext::Multiply(const Tensor& input, const Tensor& other, Tensor& output)
{
    m_opMultiply->SetTensor(1, input);
    m_opMultiply->SetTensor(2, other);
    m_opMultiply->SetTensor(3, output ? output : input);
    m_opMultiply->Execute();
}

void VulkanContext::Divide(const Tensor& input, const Scalar other, Tensor& output)
{
    m_opDivideScalar->SetTensor(1, input);
    m_opDivideScalar->SetTensor(2, output ? output : input);
    m_opDivideScalar->m_shaderUniforms.params.Set(input.m_dataType, other);
    m_opDivideScalar->Execute();
}

void VulkanContext::Divide(const Tensor& input, const Tensor& other, Tensor& output)
{
    m_opDivide->SetTensor(1, input);
    m_opDivide->SetTensor(2, other);
    m_opDivide->SetTensor(3, output ? output : input);
    m_opDivide->Execute();
}

void VulkanContext::Remainder(const Tensor& input, const Scalar other, Tensor& output)
{
    m_opRemainderScalar->SetTensor(1, input);
    m_opRemainderScalar->SetTensor(2, output ? output : input);
    m_opRemainderScalar->m_shaderUniforms.params.Set(input.m_dataType, other);
    m_opRemainderScalar->Execute();
}

void VulkanContext::Remainder(const Tensor& input, const Tensor& other, Tensor& output)
{
    m_opRemainder->SetTensor(1, input);
    m_opRemainder->SetTensor(2, other);
    m_opRemainder->SetTensor(3, output ? output : input);
    m_opRemainder->Execute();
}

void VulkanContext::BitwiseAnd(const Tensor& input, const Scalar other, Tensor& output)
{
    assert(input.IsInteger() && other.IsInteger());
    m_opBitwiseAndScalar->SetTensor(1, input);
    m_opBitwiseAndScalar->SetTensor(2, output ? output : input);
    m_opBitwiseAndScalar->m_shaderUniforms.params.Set(input.m_dataType, other);
    m_opBitwiseAndScalar->Execute();
}

void VulkanContext::BitwiseAnd(const Tensor& input, const Tensor& other, Tensor& output)
{
    assert(input.IsInteger() && other.IsInteger());
    m_opBitwiseAnd->SetTensor(1, input);
    m_opBitwiseAnd->SetTensor(2, other);
    m_opBitwiseAnd->SetTensor(3, output ? output : input);
    m_opBitwiseAnd->Execute();
}

void VulkanContext::BitwiseOr(const Tensor& input, const Scalar other, Tensor& output)
{
    assert(input.IsInteger() && other.IsInteger());
    m_opBitwiseOrScalar->SetTensor(1, input);
    m_opBitwiseOrScalar->SetTensor(2, output ? output : input);
    m_opBitwiseOrScalar->m_shaderUniforms.params.Set(input.m_dataType, other);
    m_opBitwiseOrScalar->Execute();
}

void VulkanContext::BitwiseOr(const Tensor& input, const Tensor& other, Tensor& output)
{
    assert(input.IsInteger() && other.IsInteger());
    m_opBitwiseOr->SetTensor(1, input);
    m_opBitwiseOr->SetTensor(2, other);
    m_opBitwiseOr->SetTensor(3, output ? output : input);
    m_opBitwiseOr->Execute();
}

void VulkanContext::BitwiseXor(const Tensor& input, const Scalar other, Tensor& output)
{
    assert(input.IsInteger() && other.IsInteger());
    m_opBitwiseXorScalar->SetTensor(1, input);
    m_opBitwiseXorScalar->SetTensor(2, output ? output : input);
    m_opBitwiseXorScalar->m_shaderUniforms.params.Set(input.m_dataType, other);
    m_opBitwiseXorScalar->Execute();
}

void VulkanContext::BitwiseXor(const Tensor& input, const Tensor& other, Tensor& output)
{
    assert(input.IsInteger() && other.IsInteger());
    m_opBitwiseXor->SetTensor(1, input);
    m_opBitwiseXor->SetTensor(2, other);
    m_opBitwiseXor->SetTensor(3, output ? output : input);
    m_opBitwiseXor->Execute();
}

} // namespace ML
