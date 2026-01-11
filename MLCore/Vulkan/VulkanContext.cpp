#include <MLCore/Vulkan/VulkanContext.h>
#include <MLCore/Vulkan/VulkanDevice.h>
#include <MLCore/Vulkan/VulkanTensor.h>

namespace ML
{

VulkanContext::VulkanContext(rad::Ref<VulkanDevice> device) :
    Context(std::move(device))
{
    m_opFill = RAD_NEW VulkanTensorOpForEach(this, "Fill");
    m_opAddScalar = RAD_NEW VulkanTensorOpElementWiseUnary(this, "AddScalar");
    m_opAdd = RAD_NEW VulkanTensorOpElementWiseBinary(this, "Add");
    m_opSubtractScalar = RAD_NEW VulkanTensorOpElementWiseUnary(this, "SubtractScalar");
    m_opSubtract = RAD_NEW VulkanTensorOpElementWiseBinary(this, "Subtract");
    m_opMultiplyScalar = RAD_NEW VulkanTensorOpElementWiseUnary(this, "MultiplyScalar");
    m_opMultiply = RAD_NEW VulkanTensorOpElementWiseBinary(this, "Multiply");
    m_opDivideScalar = RAD_NEW VulkanTensorOpElementWiseUnary(this, "DivideScalar");
    m_opDivide = RAD_NEW VulkanTensorOpElementWiseBinary(this, "Divide");
    m_opRemainderScalar = RAD_NEW VulkanTensorOpElementWiseUnary(this, "RemainderScalar");
    m_opRemainder = RAD_NEW VulkanTensorOpElementWiseBinary(this, "Remainder");
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

void VulkanContext::Fill(const Tensor& input, Scalar value)
{
    m_opFill->SetTensor(1, input);
    if (input.IsFloatingPoint())
    {
        m_opFill->SetParameters(glm::vec4(float(value)));
    }
    else
    {
        m_opFill->SetParameters(glm::ivec4(int(value)));
    }
    m_opFill->Execute();
}

void VulkanContext::Add(const Tensor& input, const Scalar other, Tensor& output)
{
    m_opAddScalar->SetTensor(1, input);
    m_opAddScalar->SetTensor(2, output ? output : input);
    if (input.IsFloatingPoint())
    {
        m_opAddScalar->SetParameters(glm::vec4(float(other)));
    }
    else
    {
        m_opAddScalar->SetParameters(glm::ivec4(int(other)));
    }
    m_opAddScalar->Execute();
}

void VulkanContext::Add(const Tensor& input, const Tensor& other, const Scalar alpha, Tensor& output)
{
    m_opAdd->SetTensor(1, input);
    m_opAdd->SetTensor(2, other);
    m_opAdd->SetTensor(3, output ? output : input);
    if (input.IsFloatingPoint())
    {
        m_opAdd->SetParameters(glm::vec4(float(alpha)));
    }
    else
    {
        m_opAdd->SetParameters(glm::ivec4(int(alpha)));
    }
    m_opAdd->Execute();
}

void VulkanContext::Subtract(const Tensor& input, const Scalar other, Tensor& output)
{
    m_opSubtractScalar->SetTensor(1, input);
    m_opSubtractScalar->SetTensor(2, output ? output : input);
    if (input.IsFloatingPoint())
    {
        m_opSubtractScalar->SetParameters(glm::vec4(float(other)));
    }
    else
    {
        m_opSubtractScalar->SetParameters(glm::ivec4(int(other)));
    }
    m_opSubtractScalar->Execute();
}

void VulkanContext::Subtract(const Tensor& input, const Tensor& other, const Scalar alpha, Tensor& output)
{
    m_opSubtract->SetTensor(1, input);
    m_opSubtract->SetTensor(2, other);
    m_opSubtract->SetTensor(3, output ? output : input);
    if (input.IsFloatingPoint())
    {
        m_opSubtract->SetParameters(glm::vec4(float(alpha)));
    }
    else
    {
        m_opSubtract->SetParameters(glm::ivec4(int(alpha)));
    }
    m_opSubtract->Execute();
}

void VulkanContext::Multiply(const Tensor& input, const Scalar other, Tensor& output)
{
    m_opMultiplyScalar->SetTensor(1, input);
    m_opMultiplyScalar->SetTensor(2, output ? output : input);
    if (input.IsFloatingPoint())
    {
        m_opMultiplyScalar->SetParameters(glm::vec4(float(other)));
    }
    else
    {
        m_opMultiplyScalar->SetParameters(glm::ivec4(int(other)));
    }
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
    if (input.IsFloatingPoint())
    {
        m_opDivideScalar->SetParameters(glm::vec4(float(other)));
    }
    else
    {
        m_opDivideScalar->SetParameters(glm::ivec4(int(other)));
    }
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
    if (input.IsFloatingPoint())
    {
        m_opRemainderScalar->SetParameters(glm::vec4(float(other)));
    }
    else
    {
        m_opRemainderScalar->SetParameters(glm::ivec4(int(other)));
    }
    m_opRemainderScalar->Execute();
}

void VulkanContext::Remainder(const Tensor& input, const Tensor& other, Tensor& output)
{
    m_opRemainder->SetTensor(1, input);
    m_opRemainder->SetTensor(2, other);
    m_opRemainder->SetTensor(3, output ? output : input);
    m_opRemainder->Execute();
}

} // namespace ML
