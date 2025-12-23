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
    m_opFillConstant->SetTensor(1, static_cast<VulkanTensor*>(input));
    m_opFillConstant->SetParameters(glm::vec4(value));
    m_opFillConstant->Execute();
}

void VulkanContext::FillConstant(Tensor* input, int value)
{
    assert(IsIntegerType(input->m_dataType));
    m_opFillConstant->SetTensor(1, static_cast<VulkanTensor*>(input));
    m_opFillConstant->SetParameters(glm::ivec4(value));
    m_opFillConstant->Execute();
}

void VulkanContext::AddScalar(Tensor* input, float other, Tensor* output)
{
    assert(IsFloatingPointType(input->m_dataType));
    m_opAddScalar->SetTensor(1, static_cast<VulkanTensor*>(input));
    m_opAddScalar->SetTensor(2, static_cast<VulkanTensor*>(output ? output : input));
    m_opAddScalar->SetParameters(glm::vec4(other));
    m_opAddScalar->Execute();
}

void VulkanContext::AddScalar(Tensor* input, int other, Tensor* output)
{
    assert(IsIntegerType(input->m_dataType));
    m_opAddScalar->SetTensor(1, static_cast<VulkanTensor*>(input));
    m_opAddScalar->SetTensor(2, static_cast<VulkanTensor*>(output ? output : input));
    m_opAddScalar->SetParameters(glm::ivec4(other));
    m_opAddScalar->Execute();
}

void VulkanContext::Add(Tensor* input, Tensor* other, float alpha, Tensor* output)
{
    assert(IsFloatingPointType(input->m_dataType));
    m_opAdd->SetTensor(1, static_cast<VulkanTensor*>(input));
    m_opAdd->SetTensor(2, static_cast<VulkanTensor*>(other));
    m_opAdd->SetTensor(3, static_cast<VulkanTensor*>(output ? output : input));
    m_opAdd->SetParameters(glm::vec4(alpha));
    m_opAdd->Execute();
}

void VulkanContext::Add(Tensor* input, Tensor* other, int alpha, Tensor* output)
{
    assert(IsIntegerType(input->m_dataType));
    m_opAdd->SetTensor(1, static_cast<VulkanTensor*>(input));
    m_opAdd->SetTensor(2, static_cast<VulkanTensor*>(other));
    m_opAdd->SetTensor(3, static_cast<VulkanTensor*>(output ? output : input));
    m_opAdd->SetParameters(glm::ivec4(alpha));
    m_opAdd->Execute();
}

} // namespace ML
