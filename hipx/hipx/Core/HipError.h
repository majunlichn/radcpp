#pragma once

#include <hip/hip_runtime.h>
#include <exception>
#include <source_location>

class HipError : public std::exception
{
public:
    HipError(std::source_location location, hipError_t error) : m_error(error) {}
    ~HipError() {}

    virtual const char* what() const noexcept { return hipGetErrorName(m_error); }

    std::source_location m_location;
    hipError_t m_error;

}; // HipError

#define HIP_CHECK(FunctionCall)                                                 \
    do {                                                                        \
        const hipError_t err = FunctionCall;                                    \
        if (err != hipSuccess)                                                  \
        {                                                                       \
            throw HipError(std::source_location::current(), err);               \
        }                                                                       \
    } while(0)
