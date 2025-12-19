#include <MLCore/Device.h>
#include <MLCore/Backend.h>
#include <MLCore/Device.h>
#include <MLCore/Context.h>
#include <MLCore/Tensor.h>

namespace ML
{

rad::Ref<Tensor> CreateTensor(rad::ArrayRef<size_t> sizes, DataType dataType, Device* device, const TensorOptions& options)
{
    return device->CreateTensor(sizes, dataType, options);
}

} // namespace ML
