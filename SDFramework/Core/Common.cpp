#include <SDFramework/Core/Common.h>

namespace sdf
{

spdlog::logger* GetLogger()
{
    static std::shared_ptr<spdlog::logger> SDLogger = rad::CreateLogger("SDFramework");
    return SDLogger.get();
}

void SetOutOfMemory()
{
    SDL_OutOfMemory();
}

SDL_Buffer::~SDL_Buffer()
{
    if (m_data)
    {
        SDL_free(m_data);
        m_data = nullptr;
    }
}

} // namespace sdf
