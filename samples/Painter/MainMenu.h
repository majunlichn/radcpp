#pragma once

#include "Widget.h"

class MainMenu : public Widget
{
public:
    MainMenu(PaintManager* manager, sdf::GuiContext* context);
    ~MainMenu();

    virtual void OnIdle() override;

}; // class MainMenu
