#pragma once

#include <rad/Common/Platform.h>
#include <rad/Common/Integer.h>
#include <rad/Common/Memory.h>
#include <rad/Common/RefCounted.h>
#include <rad/IO/File.h>
#include <rad/IO/Logging.h>

class PaintManager : public rad::RefCounted<PaintManager>
{
public:
    PaintManager();
    ~PaintManager();

    std::shared_ptr<spdlog::logger> m_logger;

    bool m_showDemoWindow = false;
    bool m_showPlotDemoWindow = false;
    bool m_showAboutWindow = false;

}; // class PaintManager
