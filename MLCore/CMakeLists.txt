set(MLCORE_SOURCE_FILES
    MLCore.h
    Global.h
    Global.cpp
    Common.h
    Common.cpp
    Backend.h
    Backend.cpp
    Device.h
    Device.cpp
    Context.h
    Context.cpp
    Tensor.h
    Tensor.cpp
    TensorIterator.h
    TensorIterator.cpp
    Logging.h
    Logging.cpp
    CPU/CpuBackend.h
    CPU/CpuBackend.cpp
    CPU/CpuDevice.h
    CPU/CpuDevice.cpp
    CPU/CpuContext.h
    CPU/CpuContext.cpp
    CPU/CpuTensor.h
    CPU/CpuTensor.cpp
    CPU/CpuTensorOp.h
    CPU/CpuTensorOp.cpp
    Vulkan/VulkanBackend.h
    Vulkan/VulkanBackend.cpp
    Vulkan/VulkanDevice.h
    Vulkan/VulkanDevice.cpp
    Vulkan/VulkanContext.h
    Vulkan/VulkanContext.cpp
    Vulkan/VulkanTensor.h
    Vulkan/VulkanTensor.cpp
    Vulkan/VulkanTensorOp.h
    Vulkan/VulkanTensorOp.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${MLCORE_SOURCE_FILES})

add_library(MLCore STATIC ${MLCORE_SOURCE_FILES})

target_link_libraries(MLCore
    PUBLIC rad
    PUBLIC vkpp
)
