#pragma once

#include <rad/Core/Platform.h>
#include <rad/Core/Integer.h>
#include <rad/Core/Memory.h>
#include <rad/Core/RefCounted.h>
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
