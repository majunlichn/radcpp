#pragma once

#include <SDFramework/Core/Common.h>
#include <SDL3/SDL_camera.h>

namespace sdf
{

std::vector<const char*> EnumerateCameraDrivers();
const char* GetCurrentCameraDriver();

class Camera;
std::vector<rad::Ref<Camera>> EnumerateCameras();

// Simple wrapper for SDL_camera.
class Camera : public rad::RefCounted<Camera>
{
public:
    Camera(SDL_CameraID id);
    ~Camera();

    const std::string& GetName() const { return m_name; }
    rad::ArrayRef<SDL_CameraSpec*> GetSupportedFormats() const
    {
        return { m_specs, static_cast<size_t>(m_specCount) };
    }
    SDL_CameraPosition GetPosition() const { return m_position; }

    bool Open(const SDL_CameraSpec* spec);
    void Close();

    enum class Permission : int
    {
        WaitForResponse = 0,
        Denied = -1,
        Approved = 1,
    };
    Permission GetPermission();
    bool GetFormat(SDL_CameraSpec* spec);

    // Do not call SDL_FreeSurface() on the returned surface!
    SDL_Surface* AcquireFrame(Uint64* timestamp);
    void ReleaseFrame(SDL_Surface* surface);

private:
    SDL_CameraID m_id = 0;
    std::string m_name;
    SDL_CameraSpec** m_specs;
    int m_specCount = 0;
    SDL_CameraPosition m_position;
    // The opaque structure used to identify an opened SDL camera.
    SDL_Camera* m_handle = nullptr;
    SDL_PropertiesID m_propID = 0;

}; // class Camera

} // namespace sdf
