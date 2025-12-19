#include <MLCore/Logging.h>

namespace ML
{

spdlog::logger* GetLogger()
{
    static std::shared_ptr<spdlog::logger> logger = rad::CreateLogger("ML");
    return logger.get();
}

} // namespace ML
