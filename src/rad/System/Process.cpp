#include <rad/System/Process.h>

#include <boost/asio/buffer.hpp>
#include <boost/asio/error.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/read.hpp>
#include <boost/process.hpp>

namespace rad
{

std::string Process::ExecuteAndCaptureOutput(const std::string& executable,
                                             const std::vector<std::string>& args)
{
    boost::asio::io_context context;
    auto executablePath = boost::process::v2::environment::find_executable(executable);
    if (executablePath.empty())
    {
        executablePath = executable;
    }

    boost::process::v2::popen process(context, executablePath, args);
    process.get_stdin().close();

    std::string result;
    boost::system::error_code error;
    boost::asio::read(process, boost::asio::dynamic_buffer(result), error);
    if ((error != boost::asio::error::eof) && (error != boost::asio::error::broken_pipe))
    {
        throw boost::system::system_error(error, "Failed to read process output");
    }
    process.wait();
    return result;
}

} // namespace rad
