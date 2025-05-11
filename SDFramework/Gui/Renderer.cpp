#include <SDFramework/Gui/Renderer.h>
#include <SDFramework/Gui/Texture.h>

namespace sdf
{

Renderer::Renderer(Window* window) :
    m_window(window)
{
}

Renderer::~Renderer()
{
    Destroy();
}

bool Renderer::Init()
{
    const char* driver = nullptr;

#if defined(RAD_OS_WINDOWS)
    for (int i = 0; i < SDL_GetNumRenderDrivers(); ++i)
    {
        const char* name = SDL_GetRenderDriver(i);
        if (name)
        {
            if (rad::StrEqual(name, "direct3d12") ||
                rad::StrEqual(name, "direct3d11"))
            {
                driver = name;
                break;
            }
        }
    }
#endif
    m_handle = SDL_CreateRenderer(m_window->GetHandle(), driver);
    if (m_handle)
    {
        const char* name = SDL_GetRendererName(m_handle);
        if (name)
        {
            m_name = name;
        }
        else
        {
            SDF_LOG(err, "SDL_GetRendererName failed: {}", SDL_GetError());
        }
        SDF_LOG(info, "Renderer created: {}", m_name);
        m_propID = SDL_GetRendererProperties(m_handle);
        if (m_propID == 0)
        {
            SDF_LOG(err, "SDL_GetRendererProperties failed: {}", SDL_GetError());
        }
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_CreateRenderer failed: {}", SDL_GetError());
        return false;
    }
}

void Renderer::Destroy()
{
    if (m_handle)
    {
        SDL_DestroyRenderer(m_handle);
        m_handle = nullptr;
    }
}

bool Renderer::GetOutputSize(int* w, int* h)
{
    bool result = SDL_GetRenderOutputSize(m_handle, w, h);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetRenderOutputSize failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::GetCurrentOutputSize(int* w, int* h)
{
    bool result = SDL_GetCurrentRenderOutputSize(m_handle, w, h);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetCurrentRenderOutputSize failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::SetRenderTarget(Texture* texture)
{
    bool result = SDL_SetRenderTarget(m_handle, texture ? texture->GetHandle() : nullptr);
    if (result)
    {
        m_renderTarget = texture;
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetRenderTarget failed: {}", SDL_GetError());
        return false;
    }
}

Texture* Renderer::GetRenderTarget()
{
    if (m_renderTarget)
    {
#if defined(_DEBUG)
        SDL_Texture* handle = SDL_GetRenderTarget(m_handle);
        assert(m_renderTarget->GetHandle() == handle);
#endif
        return m_renderTarget;
    }
    else
    {
        return nullptr;
    }
}

bool Renderer::SetLogicalPresentation(int w, int h, SDL_RendererLogicalPresentation mode)
{
    bool result = SDL_SetRenderLogicalPresentation(m_handle, w, h, mode);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetRenderLogicalPresentation failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::GetLogicalPresentation(
    int* w, int* h, SDL_RendererLogicalPresentation* mode)
{
    bool result = SDL_GetRenderLogicalPresentation(m_handle, w, h, mode);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetRenderLogicalPresentation failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::TransformWindowCoordToRender(float windowX, float windowY, float* x, float* y)
{
    bool result = SDL_RenderCoordinatesFromWindow(m_handle, windowX, windowY, x, y);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_RenderCoordinatesFromWindow failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::TransformRenderCoordToWindow(float x, float y, float* windowX, float* windowY)
{
    bool result = SDL_RenderCoordinatesToWindow(m_handle, x, y, windowX, windowY);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_RenderCoordinatesToWindow failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::TransformCoordToRender(SDL_Event& event)
{
    bool result = SDL_ConvertEventToRenderCoordinates(m_handle, &event);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_ConvertEventToRenderCoordinates failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::SetViewport(const SDL_Rect* rect)
{
    bool result = SDL_SetRenderViewport(m_handle, rect);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetRenderViewport failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::SetViewportFull()
{
    return SetViewport(nullptr);
}

bool Renderer::GetViewport(SDL_Rect* rect)
{
    bool result = SDL_GetRenderViewport(m_handle, rect);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetRenderViewport failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::HasViewportSet()
{
    return (SDL_RenderViewportSet(m_handle) == true);
}

bool Renderer::SetClipRect(const SDL_Rect* rect)
{
    bool result = SDL_SetRenderClipRect(m_handle, rect);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetRenderClipRect failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::DisableClipping()
{
    return SetClipRect(nullptr);
}

bool Renderer::GetClipRect(SDL_Rect* rect)
{
    bool result = SDL_GetRenderClipRect(m_handle, rect);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetRenderClipRect failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::IsClipEnabled()
{
    return (SDL_RenderClipEnabled(m_handle) == true);
}

bool Renderer::SetRenderScale(float scaleX, float scaleY)
{
    bool result = SDL_SetRenderScale(m_handle, scaleX, scaleY);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetRenderScale failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::GetRenderScale(float* scaleX, float* scaleY)
{
    bool result = SDL_GetRenderScale(m_handle, scaleX, scaleY);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetRenderScale failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::SetRenderDrawColor(Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    bool result = SDL_SetRenderDrawColor(m_handle, r, g, b, a);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetRenderDrawColor failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::SetRenderDrawColor(float r, float g, float b, float a)
{
    bool result = SDL_SetRenderDrawColorFloat(m_handle, r, g, b, a);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetRenderDrawColorFloat failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::GetRenderDrawColor(Uint8* r, Uint8* g, Uint8* b, Uint8* a)
{
    bool result = SDL_GetRenderDrawColor(m_handle, r, g, b, a);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetRenderDrawColor failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::GetRenderDrawColor(float* r, float* g, float* b, float* a)
{
    bool result = SDL_GetRenderDrawColorFloat(m_handle, r, g, b, a);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetRenderDrawColorFloat failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::SetColorScale(float scale)
{
    bool result = SDL_SetRenderColorScale(m_handle, scale);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetRenderColorScale failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::GetColorScale(float* scale)
{
    bool result = SDL_GetRenderColorScale(m_handle, scale);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetRenderColorScale failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::SetBlendMode(SDL_BlendMode blendMode)
{
    bool result = SDL_SetRenderDrawBlendMode(m_handle, blendMode);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetRenderDrawBlendMode failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::GetBlendMode(SDL_BlendMode* blendMode)
{
    bool result = SDL_GetRenderDrawBlendMode(m_handle, blendMode);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetRenderDrawBlendMode failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::Clear()
{
    bool result = SDL_RenderClear(m_handle);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_RenderClear failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::DrawPoint(float x, float y)
{
    bool result = SDL_RenderPoint(m_handle, x, y);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_RenderPoint failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::DrawPoints(const SDL_FPoint* points, int count)
{
    bool result = SDL_RenderPoints(m_handle, points, count);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_RenderPoints failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::DrawLine(float x1, float y1, float x2, float y2)
{
    bool result = SDL_RenderLine(m_handle, x1, y1, x2, y2);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_RenderLine failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::DrawLines(const SDL_FPoint* points, int count)
{
    bool result = SDL_RenderLines(m_handle, points, count);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_RenderLines failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::DrawRect(const SDL_FRect* rect)
{
    bool result = SDL_RenderRect(m_handle, rect);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_RenderRect failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::DrawRects(const SDL_FRect* rect, int count)
{
    bool result = SDL_RenderRects(m_handle, rect, count);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_RenderRects failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::FillRect(const SDL_FRect* rect)
{
    bool result = SDL_RenderFillRect(m_handle, rect);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_RenderFillRect failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::FillRects(const SDL_FRect* rect, int count)
{
    bool result = SDL_RenderFillRects(m_handle, rect, count);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_RenderFillRects failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::DrawTexture(Texture* texture, const SDL_FRect* srcRect, const SDL_FRect* dstRect)
{
    bool result = SDL_RenderTexture(m_handle, texture->GetHandle(), srcRect, dstRect);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_RenderTexture failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::DrawTextureRotated(Texture* texture,
    const SDL_FRect* srcRect, const SDL_FRect* dstRect,
    const double angle, const SDL_FPoint* center,
    const SDL_FlipMode flip)
{
    bool result = SDL_RenderTextureRotated(m_handle, texture->GetHandle(),
        srcRect, dstRect, angle, center, flip);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_RenderTextureRotated failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::RenderGeometry(Texture* texture,
    const SDL_Vertex* vertices, int numVertices,
    const int* indices, int numIndices)
{
    bool result = SDL_RenderGeometry(m_handle, texture->GetHandle(),
        vertices, numVertices, indices, numIndices);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_RenderGeometry failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::RenderGeometryRaw(Texture* texture,
    const float* xy, int xyStride,
    const SDL_FColor* color, int colorStride,
    const float* uv, int uvStride,
    int numVertices,
    const void* indices, int numIndices, int indexType)
{
    bool result = SDL_RenderGeometryRaw(m_handle, texture->GetHandle(),
        xy, xyStride, color, colorStride, uv, uvStride, numVertices, indices, numIndices, indexType);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_RenderGeometryRaw failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::Present()
{
    bool result = SDL_RenderPresent(m_handle);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_RenderPresent failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::Flush()
{
    bool result = SDL_FlushRenderer(m_handle);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_FlushRenderer failed: {}", SDL_GetError());
        return false;
    }
}

bool Renderer::SetVSync(int vsync)
{
    bool result = SDL_SetRenderVSync(m_handle, vsync);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetRenderVSync({}) failed: {}",
            vsync, SDL_GetError());
        return false;
    }
}

bool Renderer::GetVSync(int* vsync)
{
    bool result = SDL_GetRenderVSync(m_handle, vsync);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetRenderVSync failed: {}", SDL_GetError());
        return false;
    }
}

} // namespace sdf
