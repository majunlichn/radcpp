#include <SDFramework/Multimedia/Camera.h>

namespace sdf
{

std::vector<const char*> EnumerateCameraDrivers()
{
    std::vector<const char*> drivers;
    int count = SDL_GetNumCameraDrivers();
    for (int i = 0; i < count; ++i)
    {
        const char* driver = SDL_GetCameraDriver(i);
        drivers.push_back(driver);
    }
    return drivers;
}

const char* GetCurrentCameraDriver()
{
    return SDL_GetCurrentCameraDriver();
}

std::vector<rad::Ref<Camera>> EnumerateCameras()
{
    std::vector<rad::Ref<Camera>> cameras;
    int count = 0;
    SDL_CameraID* ids = SDL_GetCameras(&count);
    if (ids && (count > 0))
    {
        cameras.resize(count);
        for (int i = 0; i < count; ++i)
        {
            cameras[i] = RAD_NEW Camera(ids[i]);
        }
        SDL_free(ids);
    }
    else
    {
        SDF_LOG(err, "SDL_GetCameras failed: {}", SDL_GetError());
    }
    return cameras;
}

Camera::Camera(SDL_CameraID id) :
    m_id(id)
{
    const char* name = SDL_GetCameraName(id);
    if (name)
    {
        m_name = name;
    }
    else
    {
        SDF_LOG(err, "SDL_GetCameraName failed: {}", SDL_GetError());
    }
    m_specs = SDL_GetCameraSupportedFormats(id, &m_specCount);
    m_position = SDL_GetCameraPosition(id);
}

Camera::~Camera()
{
    if (m_specs)
    {
        SDL_free(m_specs);
        m_specs = nullptr;
    }

    if (m_handle)
    {
        Close();
    }
}

bool Camera::Open(const SDL_CameraSpec* spec)
{
    m_handle = SDL_OpenCamera(m_id, spec);
    if (m_handle)
    {
#if defined(_DEBUG)
        SDL_CameraID id = SDL_GetCameraID(m_handle);
        assert(id == m_id);
#endif
        SDF_LOG(info, "Camera {} opened successfully.", m_name);
        m_propID = SDL_GetCameraProperties(m_handle);
        if (m_propID == 0)
        {
            SDF_LOG(err, "SDL_GetCameraProperties failed: {}", SDL_GetError());
        }
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_OpenCameraDevice failed: {}", SDL_GetError());
        return false;
    }
}

void Camera::Close()
{
    SDL_CloseCamera(m_handle);
    m_handle = nullptr;
}

Camera::Permission Camera::GetPermission()
{
    assert(m_handle != nullptr); // must has been opened.
    int permission = SDL_GetCameraPermissionState(m_handle);
    return static_cast<Permission>(permission);
}

bool Camera::GetFormat(SDL_CameraSpec* spec)
{
    bool result = SDL_GetCameraFormat(m_handle, spec);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetCameraFormat failed: {}", SDL_GetError());
        return false;
    }
}

SDL_Surface* Camera::AcquireFrame(Uint64* timestamp)
{
    return SDL_AcquireCameraFrame(m_handle, timestamp);
}

void Camera::ReleaseFrame(SDL_Surface* surface)
{
    SDL_ReleaseCameraFrame(m_handle, surface);
}

} // namespace sdf
