#include <rad/ML/MLCpuTensor.h>
#include <rad/ML/MLCpuDevice.h>

namespace rad
{

MLCpuTensor::MLCpuTensor(Ref<MLCpuDevice> device) :
    m_device(std::move(device))
{
}

MLCpuTensor::~MLCpuTensor()
{
}

} // namespace rad
