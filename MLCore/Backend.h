#pragma once

#include <MLCore/Common.h>

namespace ML
{

class Device;
class Context;

class Backend : public rad::RefCounted<Backend>
{
public:
    std::string m_name;

    Backend();
    virtual ~Backend();

    const std::string& GetName() const { return m_name; }

    virtual size_t GetDeviceCount() const = 0;
    virtual Device* GetDevice(size_t index) = 0;

}; // class Backend

} // namespace ML
