#include <hipx/Core/HipContext.h>

int main()
{
    HipContext ctx;
    if (!ctx.Init())
    {
        return -1;
    }
    return 0;
}
