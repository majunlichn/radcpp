#include "Widget.h"

Widget::Widget(PaintManager* manager, sdf::Frame* context) :
    m_manager(manager),
    m_context(context)
{
}

Widget::~Widget()
{
}
