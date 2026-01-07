#include <MLCore/Vulkan/VulkanContext.h>
#include <MLCore/Vulkan/VulkanDevice.h>
#include <MLCore/Vulkan/VulkanTensor.h>

namespace ML
{

VulkanContext::VulkanContext(rad::Ref<VulkanDevice> device) :
    Context(std::move(device))
{
    m_opFillConstant = RAD_NEW VulkanTensorOpForEach(this, "FillConstant");
    m_opAddScalar = RAD_NEW VulkanTensorOpElementWiseUnary(this, "AddScalar");
    m_opAdd = RAD_NEW VulkanTensorOpElementWiseBinary(this, "Add");
    m_opSubtractScalar = RAD_NEW VulkanTensorOpElementWiseUnary(this, "SubtractScalar");
    m_opSubtract = RAD_NEW VulkanTensorOpElementWiseBinary(this, "Subtract");
    m_opMultiplyScalar = RAD_NEW VulkanTensorOpElementWiseUnary(this, "MultiplyScalar");
    m_opMultiply = RAD_NEW VulkanTensorOpElementWiseBinary(this, "Multiply");
    m_opDivideScalar = RAD_NEW VulkanTensorOpElementWiseUnary(this, "DivideScalar");
    m_opDivide = RAD_NEW VulkanTensorOpElementWiseBinary(this, "Divide");
}

VulkanContext::~VulkanContext()
{
}

VulkanDevice* VulkanContext::GetDevice()
{
    return static_cast<VulkanDevice*>(m_device.get());
}

vkpp::Device* VulkanContext::GetDeviceImpl()
{
    return static_cast<VulkanDevice*>(m_device.get())->m_impl.get();
}

void VulkanContext::FillConstant(Tensor* input, float value)
{
    assert(IsFloatingPointType(input->m_dataType));
    m_opFillConstant->SetTensor(1, input);
    m_opFillConstant->SetParameters(glm::vec4(value));
    m_opFillConstant->Execute();
}

void VulkanContext::FillConstant(Tensor* input, int value)
{
    assert(IsIntegerType(input->m_dataType));
    m_opFillConstant->SetTensor(1, input);
    m_opFillConstant->SetParameters(glm::ivec4(value));
    m_opFillConstant->Execute();
}

void VulkanContext::AddScalar(Tensor* input, float other, Tensor* output)
{
    assert(IsFloatingPointType(input->m_dataType));
    m_opAddScalar->SetTensor(1, input);
    m_opAddScalar->SetTensor(2, output ? output : input);
    m_opAddScalar->SetParameters(glm::vec4(other));
    m_opAddScalar->Execute();
}

void VulkanContext::AddScalar(Tensor* input, int other, Tensor* output)
{
    assert(IsIntegerType(input->m_dataType));
    m_opAddScalar->SetTensor(1, input);
    m_opAddScalar->SetTensor(2, output ? output : input);
    m_opAddScalar->SetParameters(glm::ivec4(other));
    m_opAddScalar->Execute();
}

void VulkanContext::Add(Tensor* input, Tensor* other, float alpha, Tensor* output)
{
    assert(IsFloatingPointType(input->m_dataType));
    m_opAdd->SetTensor(1, input);
    m_opAdd->SetTensor(2, other);
    m_opAdd->SetTensor(3, output ? output : input);
    m_opAdd->SetParameters(glm::vec4(alpha));
    m_opAdd->Execute();
}

void VulkanContext::Add(Tensor* input, Tensor* other, int alpha, Tensor* output)
{
    assert(IsIntegerType(input->m_dataType));
    m_opAdd->SetTensor(1, input);
    m_opAdd->SetTensor(2, other);
    m_opAdd->SetTensor(3, output ? output : input);
    m_opAdd->SetParameters(glm::ivec4(alpha));
    m_opAdd->Execute();
}

void VulkanContext::SubtractScalar(Tensor* input, float other, Tensor* output)
{
    assert(IsFloatingPointType(input->m_dataType));
    m_opSubtractScalar->SetTensor(1, input);
    m_opSubtractScalar->SetTensor(2, output ? output : input);
    m_opSubtractScalar->SetParameters(glm::vec4(other));
    m_opSubtractScalar->Execute();
}

void VulkanContext::SubtractScalar(Tensor* input, int other, Tensor* output)
{
    assert(IsIntegerType(input->m_dataType));
    m_opSubtractScalar->SetTensor(1, input);
    m_opSubtractScalar->SetTensor(2, output ? output : input);
    m_opSubtractScalar->SetParameters(glm::ivec4(other));
    m_opSubtractScalar->Execute();
}

void VulkanContext::Subtract(Tensor* input, Tensor* other, float alpha, Tensor* output)
{
    assert(IsFloatingPointType(input->m_dataType));
    m_opSubtract->SetTensor(1, input);
    m_opSubtract->SetTensor(2, other);
    m_opSubtract->SetTensor(3, output ? output : input);
    m_opSubtract->SetParameters(glm::vec4(alpha));
    m_opSubtract->Execute();
}

void VulkanContext::Subtract(Tensor* input, Tensor* other, int alpha, Tensor* output)
{
    assert(IsIntegerType(input->m_dataType));
    m_opSubtract->SetTensor(1, input);
    m_opSubtract->SetTensor(2, other);
    m_opSubtract->SetTensor(3, output ? output : input);
    m_opSubtract->SetParameters(glm::ivec4(alpha));
    m_opSubtract->Execute();
}

void VulkanContext::MultiplyScalar(Tensor* input, float other, Tensor* output)
{
    assert(IsFloatingPointType(input->m_dataType));
    m_opMultiplyScalar->SetTensor(1, input);
    m_opMultiplyScalar->SetTensor(2, output ? output : input);
    m_opMultiplyScalar->SetParameters(glm::vec4(other));
    m_opMultiplyScalar->Execute();
}

void VulkanContext::MultiplyScalar(Tensor* input, int other, Tensor* output)
{
    assert(IsIntegerType(input->m_dataType));
    m_opMultiplyScalar->SetTensor(1, input);
    m_opMultiplyScalar->SetTensor(2, output ? output : input);
    m_opMultiplyScalar->SetParameters(glm::ivec4(other));
    m_opMultiplyScalar->Execute();
}

void VulkanContext::Multiply(Tensor* input, Tensor* other, Tensor* output)
{
    m_opMultiply->SetTensor(1, input);
    m_opMultiply->SetTensor(2, other);
    m_opMultiply->SetTensor(3, output ? output : input);
    m_opMultiply->Execute();
}

void VulkanContext::DivideScalar(Tensor* input, float other, Tensor* output)
{
    assert(IsFloatingPointType(input->m_dataType));
    m_opDivideScalar->SetTensor(1, input);
    m_opDivideScalar->SetTensor(2, output ? output : input);
    m_opDivideScalar->SetParameters(glm::vec4(other));
    m_opDivideScalar->Execute();
}

void VulkanContext::DivideScalar(Tensor* input, int other, Tensor* output)
{
    assert(IsIntegerType(input->m_dataType));
    m_opDivideScalar->SetTensor(1, input);
    m_opDivideScalar->SetTensor(2, output ? output : input);
    m_opDivideScalar->SetParameters(glm::ivec4(other));
    m_opDivideScalar->Execute();
}

void VulkanContext::Divide(Tensor* input, Tensor* other, Tensor* output)
{
    m_opDivide->SetTensor(1, input);
    m_opDivide->SetTensor(2, other);
    m_opDivide->SetTensor(3, output ? output : input);
    m_opDivide->Execute();
}

} // namespace ML
