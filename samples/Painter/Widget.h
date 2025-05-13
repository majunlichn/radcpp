#pragma once

#include "PaintManager.h"
#include <SDFramework/Gui/GuiContext.h>

class Widget : public rad::RefCounted<Widget>
{
public:
    Widget(PaintManager* manager, sdf::GuiContext* context);
    virtual ~Widget();

    void SetEnabled(bool enabled) { m_enabled = enabled; }
    bool IsEnabled() const { return m_enabled; }

    virtual void OnIdle() = 0;

    PaintManager* m_manager = nullptr;
    sdf::GuiContext* m_context = nullptr;
    bool m_enabled = true;

}; // class Widget
