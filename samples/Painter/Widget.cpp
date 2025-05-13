#include "Widget.h"

Widget::Widget(PaintManager* manager, sdf::GuiContext* context) :
    m_manager(manager),
    m_context(context)
{
}

Widget::~Widget()
{
}
