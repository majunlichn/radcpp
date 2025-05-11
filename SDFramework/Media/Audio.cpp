#include <SDFramework/Media/Audio.h>

namespace sdf
{

std::vector<const char*> GetAudioDrivers()
{
    std::vector<const char*> drivers;
    int driverCount = SDL_GetNumAudioDrivers();
    if (driverCount > 0)
    {
        drivers.resize(driverCount);
        for (int i = 0; i < driverCount; ++i)
        {
            drivers[i] = SDL_GetAudioDriver(i);
        }
    }
    return drivers;
}

const char* GetCurrentAudioDriver()
{
    return SDL_GetCurrentAudioDriver();
}

std::vector<rad::Ref<AudioDevice>> EnumerateAudioPlaybackDevices()
{
    std::vector<rad::Ref<AudioDevice>> devices;
    int count = 0;
    SDL_AudioDeviceID* ids = SDL_GetAudioPlaybackDevices(&count);
    if (ids && (count > 0))
    {
        devices.resize(count);
        for (int i = 0; i < count; ++i)
        {
            devices[i] = RAD_NEW AudioDevice(ids[i]);
        }
        SDL_free(ids);
        ids = nullptr;
    }
    else
    {
        SDF_LOG(err, "SDL_GetAudioPlaybackDevices failed: {}", SDL_GetError());
    }
    return devices;
}

std::vector<rad::Ref<AudioDevice>> EnumerateAudioRecordingDevices()
{
    std::vector<rad::Ref<AudioDevice>> devices;
    int count = 0;
    SDL_AudioDeviceID* ids = SDL_GetAudioRecordingDevices(&count);
    if (ids && (count > 0))
    {
        devices.resize(count);
        for (int i = 0; i < count; ++i)
        {
            devices[i] = RAD_NEW AudioDevice(ids[i]);
        }
        SDL_free(ids);
        ids = nullptr;
    }
    else
    {
        SDF_LOG(err, "SDL_GetAudioRecordingDevices failed: {}", SDL_GetError());
    }
    return devices;
}

bool LoadWAV(SDL_IOStream* src, bool close, SDL_AudioSpec* spec,
    SDL_Buffer* buffer, Uint32* sizeInBytes)
{
    bool result = SDL_LoadWAV_IO(src, close, spec, &buffer->m_data, sizeInBytes);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_LoadWAV_IO failed: {}", SDL_GetError());
        return false;
    }
}

bool LoadWAVFromFile(std::string_view path, SDL_AudioSpec* spec,
    SDL_Buffer* buffer, Uint32* sizeInBytes)
{
    bool result = SDL_LoadWAV(path.data(), spec, &buffer->m_data, sizeInBytes);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_LoadWAV failed: {}", SDL_GetError());
        return false;
    }
}

bool MixAudio(Uint8* dst, const Uint8* src, SDL_AudioFormat format, Uint32 sizeInBytes, float volume)
{
    bool result = SDL_MixAudio(dst, src, format, sizeInBytes, volume);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_MixAudio failed: {}", SDL_GetError());
        return false;
    }
}

bool ConvertAudioSamples(const SDL_AudioSpec* srcSpec, const Uint8* srcData, int srcSizeInBytes,
    const SDL_AudioSpec* dstSpec, SDL_Buffer* dstData, int* dstSizeInBytes)
{
    bool result = SDL_ConvertAudioSamples(
        srcSpec, srcData, srcSizeInBytes, dstSpec, &dstData->m_data, dstSizeInBytes);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_ConvertAudioSamples failed: {}", SDL_GetError());
        return false;
    }
}

const char* GetAudioFormatName(SDL_AudioFormat format)
{
    return SDL_GetAudioFormatName(format);
}

int GetSilenceValueForFormat(SDL_AudioFormat format)
{
    return SDL_GetSilenceValueForFormat(format);
}

AudioDevice::AudioDevice(SDL_AudioDeviceID id) :
    m_id(id)
{
    const char* name = SDL_GetAudioDeviceName(id);
    if (name)
    {
        m_name = name;
    }
    else
    {
        SDF_LOG(err, "SDL_GetAudioDeviceName failed: {}", SDL_GetError());
    }
    bool result = SDL_GetAudioDeviceFormat(m_id, &m_spec, &m_sampleFrames);
    if (!result)
    {
        SDF_LOG(err, "SDL_GetAudioDeviceFormat failed: {}", SDL_GetError());
    }
}

AudioDevice::~AudioDevice()
{
}


std::vector<int> AudioDevice::GetChannelMap() const
{
    std::vector<int> buffer;
    int count = 0;
    int* channelMap = SDL_GetAudioDeviceChannelMap(m_id, &count);
    if (channelMap && (count > 0))
    {
        buffer.resize(count);
        std::memcpy(buffer.data(), channelMap, count * sizeof(int));
        SDL_free(channelMap);
        channelMap = nullptr;
    }
    return buffer;
}

bool AudioDevice::Open(const SDL_AudioSpec* spec)
{
    SDL_AudioDeviceID id = SDL_OpenAudioDevice(m_id, spec);
    if ((id != 0) && (id == m_id))
    {
        SDF_LOG(info, "AudioDevice#{} {} is opened.", m_id, m_name);
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_OpenAudioDevice failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioDevice::IsPhysical() const
{
    return SDL_IsAudioDevicePhysical(m_id);
}

bool AudioDevice::IsPlayback() const
{
    return SDL_IsAudioDevicePlayback(m_id);
}

bool AudioDevice::Pause()
{
    bool result = SDL_PauseAudioDevice(m_id);
    if (result)
    {
        SDF_LOG(info, "AudioDevice#{} {} is paused.", m_id, m_name);
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_PauseAudioDevice failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioDevice::Resume()
{
    bool result = SDL_ResumeAudioDevice(m_id);
    if (result)
    {
        SDF_LOG(info, "AudioDevice#{} {} is resumed.", m_id, m_name);
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_ResumeAudioDevice failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioDevice::IsPaused()
{
    return (SDL_AudioDevicePaused(m_id) == true);
}

float AudioDevice::GetGain() const
{
    return SDL_GetAudioDeviceGain(m_id);
}

bool AudioDevice::SetGain(float gain) const
{
    bool result = SDL_SetAudioDeviceGain(m_id, gain);
    if (result)
    {
        SDF_LOG(info, "AudioDevice#{} {} gain is set to {}", m_id, m_name, gain);
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetAudioDeviceGain failed: {}", SDL_GetError());
        return false;
    }
}

void AudioDevice::Close()
{
    SDL_CloseAudioDevice(m_id);
    SDF_LOG(info, "AudioDevice#{} {} is closed.", m_id, m_name);
}

bool AudioDevice::BindStreams(SDL_AudioStream* const* streams, int streamCount)
{
    bool result = SDL_BindAudioStreams(m_id, streams, streamCount);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_BindAudioStreams failed: {}", SDL_GetError());
        return false;
    }
}

void AudioDevice::UnbindAudioStreams(SDL_AudioStream* const* streams, int streamCount)
{
    SDL_UnbindAudioStreams(streams, streamCount);
}

bool AudioDevice::BindStream(SDL_AudioStream* stream)
{
    bool result = SDL_BindAudioStream(m_id, stream);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_BindAudioStream failed: {}", SDL_GetError());
        return false;
    }
}

void AudioDevice::UnbindAudioStream(SDL_AudioStream* stream)
{
    SDL_UnbindAudioStream(stream);
}

bool AudioDevice::SetIterationCallbacks(SDL_AudioIterationCallback start, SDL_AudioIterationCallback end, void* userdata)
{
    bool result = SDL_SetAudioIterationCallbacks(m_id, start, end, userdata);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetAudioIterationCallbacks failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioDevice::SetPostmixCallback(SDL_AudioPostmixCallback callback, void* userData)
{
    bool result = SDL_SetAudioPostmixCallback(m_id, callback, userData);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetAudioPostmixCallback failed: {}", SDL_GetError());
        return false;
    }
}

rad::Ref<AudioStream> AudioStream::Create(const SDL_AudioSpec* srcSpec, const SDL_AudioSpec* dstSpec)
{
    SDL_AudioStream* handle = SDL_CreateAudioStream(srcSpec, dstSpec);
    if (handle)
    {
        return RAD_NEW AudioStream(handle);
    }
    else
    {
        SDF_LOG(err, "SDL_CreateAudioStream failed: {}", SDL_GetError());
        return nullptr;
    }
}

rad::Ref<AudioStream> AudioStream::Create(SDL_AudioDeviceID deviceID, const SDL_AudioSpec* spec,
    SDL_AudioStreamCallback callback, void* userData)
{
    SDL_AudioStream* stream = SDL_OpenAudioDeviceStream(deviceID, spec, callback, userData);
    if (stream)
    {
        return RAD_NEW AudioStream(stream);
    }
    else
    {
        SDF_LOG(err, "SDL_OpenAudioDeviceStream failed: {}", SDL_GetError());
        return nullptr;
    }
}

rad::Ref<AudioStream> AudioStream::CreateDefaultPlayback(const SDL_AudioSpec* spec)
{
    SDL_AudioStream* stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK,
        spec, nullptr, nullptr);
    if (stream)
    {
        return RAD_NEW AudioStream(stream);
    }
    else
    {
        SDF_LOG(err, "SDL_OpenAudioDeviceStream failed: {}", SDL_GetError());
        return nullptr;
    }
}

AudioStream::AudioStream(SDL_AudioStream* handle) :
    m_handle(handle)
{
    m_propID = SDL_GetAudioStreamProperties(m_handle);
    if (m_propID == 0)
    {
        SDF_LOG(err, "SDL_GetAudioStreamProperties failed: {}", SDL_GetError());
    }
}

AudioStream::~AudioStream()
{
    Destroy();
}

void AudioStream::Destroy()
{
    if (m_handle)
    {
        SDL_DestroyAudioStream(m_handle);
        m_handle = nullptr;
    }
}

SDL_AudioDeviceID AudioStream::GetDeviceID() const
{
    return SDL_GetAudioStreamDevice(m_handle);
}

bool AudioStream::GetFormat(SDL_AudioSpec* srcSpec, SDL_AudioSpec* dstSpec)
{
    bool result = SDL_GetAudioStreamFormat(m_handle, srcSpec, dstSpec);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetAudioStreamFormat failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioStream::SetFormat(const SDL_AudioSpec* srcSpec, const SDL_AudioSpec* dstSpec)
{
    bool result = SDL_SetAudioStreamFormat(m_handle, srcSpec, dstSpec);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetAudioStreamFormat failed: {}", SDL_GetError());
        return false;
    }
}

float AudioStream::GetFrequencyRatio()
{
    return SDL_GetAudioStreamFrequencyRatio(m_handle);
}

bool AudioStream::SetFrequencyRatio(float ratio)
{
    bool result = SDL_SetAudioStreamFrequencyRatio(m_handle, ratio);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetAudioStreamFrequencyRatio failed: {}", SDL_GetError());
        return false;
    }
}

float AudioStream::GetGain() const
{
    return SDL_GetAudioStreamGain(m_handle);
}

bool AudioStream::SetGain(float gain) const
{
    bool result = SDL_SetAudioStreamGain(m_handle, gain);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetAudioStreamGain failed: {}", SDL_GetError());
        return false;
    }
}

std::vector<int> AudioStream::GetInputChannelMap() const
{
    std::vector<int> buffer;
    int count = 0;
    int* channelMap = SDL_GetAudioStreamInputChannelMap(m_handle, &count);
    if (channelMap && (count > 0))
    {
        buffer.resize(count);
        std::memcpy(buffer.data(), channelMap, count * sizeof(int));
        SDL_free(channelMap);
        channelMap = nullptr;
    }
    return buffer;
}

std::vector<int> AudioStream::GetOutputChannelMap() const
{
    std::vector<int> buffer;
    int count = 0;
    int* channelMap = SDL_GetAudioStreamOutputChannelMap(m_handle, &count);
    if (channelMap && (count > 0))
    {
        buffer.resize(count);
        std::memcpy(buffer.data(), channelMap, count * sizeof(int));
        SDL_free(channelMap);
        channelMap = nullptr;
    }
    return buffer;
}

bool AudioStream::SetInputChannelMap(rad::ArrayRef<int> map)
{
    bool result = SDL_SetAudioStreamInputChannelMap(m_handle, map.data(), map.size());
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetAudioStreamInputChannelMap failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioStream::SetOutputChannelMap(rad::ArrayRef<int> map)
{
    bool result = SDL_SetAudioStreamOutputChannelMap(m_handle, map.data(), map.size());
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetAudioStreamOutputChannelMap failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioStream::PutData(const void* data, int sizeInBytes)
{
    bool result = SDL_PutAudioStreamData(m_handle, data, sizeInBytes);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_PutAudioStreamData failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioStream::PutPlanarData(const void* const* data, int numChannels, int numSamples)
{
    bool result = SDL_PutAudioStreamPlanarData(m_handle, data, numChannels, numSamples);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_PutAudioStreamPlanarData failed: {}", SDL_GetError());
        return false;
    }
}

int AudioStream::GetData(void* data, int sizeInBytes)
{
    int bytesRead = SDL_GetAudioStreamData(m_handle, data, sizeInBytes);
    if (bytesRead == -1)
    {
        SDF_LOG(err, "SDL_GetAudioStreamData failed: {}", SDL_GetError());
    }
    return bytesRead;
}

int AudioStream::GetDataSizeAvailable()
{
    return SDL_GetAudioStreamAvailable(m_handle);
}

int AudioStream::GetDataSizeQueued()
{
    return SDL_GetAudioStreamQueued(m_handle);
}

bool AudioStream::Flush()
{
    bool result = SDL_FlushAudioStream(m_handle);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_FlushAudioStream failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioStream::Clear()
{
    bool result = SDL_ClearAudioStream(m_handle);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_ClearAudioStream failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioStream::Pause()
{
    bool result = SDL_PauseAudioStreamDevice(m_handle);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_PauseAudioStreamDevice failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioStream::Resume()
{
    bool result = SDL_ResumeAudioStreamDevice(m_handle);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_ResumeAudioStreamDevice failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioStream::IsPaused()
{
    return SDL_AudioStreamDevicePaused(m_handle);
}

bool AudioStream::Lock()
{
    bool result = SDL_LockAudioStream(m_handle);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_LockAudioStream failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioStream::Unlock()
{
    bool result = SDL_UnlockAudioStream(m_handle);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_UnlockAudioStream failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioStream::SetAudioStreamGetCallback(SDL_AudioStreamCallback callback, void* userData)
{
    return (SDL_SetAudioStreamGetCallback(m_handle, callback, userData) == 0);
}

bool AudioStream::SetAudioStreamPutCallback(SDL_AudioStreamCallback callback, void* userData)
{
    return (SDL_SetAudioStreamPutCallback(m_handle, callback, userData) == 0);
}

} // namespace sdf
