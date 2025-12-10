#include <rad/ML/MLCpuContext.h>
#include <rad/ML/MLCpuDevice.h>
#include <rad/ML/MLCpuTensor.h>
#include <rad/ML/MLTensorIterator.h>

namespace rad
{

MLCpuContext::MLCpuContext(Ref<MLCpuDevice> device) :
    m_device(std::move(device))
{
}

MLCpuContext::~MLCpuContext()
{
}

void MLCpuContext::Add(MLTensor* input, MLTensor* other, float alpha, MLTensor* output)
{
    if (output == nullptr)
    {
        output = input;
    }
    MLCpuTensor* cpuInput = static_cast<MLCpuTensor*>(input);
    MLCpuTensor* cpuOther = static_cast<MLCpuTensor*>(input);
    MLCpuTensor* cpuOutput = static_cast<MLCpuTensor*>(input);
    assert(cpuInput->m_dataType == cpuOther->m_dataType);
    assert(cpuInput->m_dataType == cpuOutput->m_dataType);
    assert(cpuInput->m_device == m_device);
    assert(cpuOther->m_device == m_device);
    assert(cpuOutput->m_device == m_device);
}

} // namespace rad
