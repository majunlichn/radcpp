#pragma once

#include <SDFramework/Core/Common.h>
#include <SDL3/SDL_audio.h>
#include <vector>
#include <string_view>

// https://wiki.libsdl.org/SDL3/CategoryAudio

namespace sdf
{

std::vector<const char*> GetAudioDrivers();
const char* GetCurrentAudioDriver();

class AudioDevice;
std::vector<rad::Ref<AudioDevice>> EnumerateAudioPlaybackDevices();
std::vector<rad::Ref<AudioDevice>> EnumerateAudioRecordingDevices();

class AudioStream;

bool LoadWAV(SDL_IOStream* src, bool close, SDL_AudioSpec* spec,
    SDL_Buffer* buffer, Uint32* sizeInBytes);
bool LoadWAVFromFile(std::string_view path, SDL_AudioSpec* spec,
    SDL_Buffer* buffer, Uint32* sizeInBytes);
bool MixAudio(Uint8* dst, const Uint8* src, SDL_AudioFormat format, Uint32 sizeInBytes, float volume);
bool ConvertAudioSamples(const SDL_AudioSpec* srcSpec, const Uint8* srcData, int srcSizeInBytes,
    const SDL_AudioSpec* dstSpec, SDL_Buffer* dstData, int* dstSizeInBytes);
const char* GetAudioFormatName(SDL_AudioFormat format);
int GetSilenceValueForFormat(SDL_AudioFormat format);

class AudioDevice : public rad::RefCounted<AudioDevice>
{
public:
    AudioDevice(SDL_AudioDeviceID id);
    ~AudioDevice();

    SDL_AudioDeviceID GetID() const { return m_id; }
    const std::string& GetName() const { return m_name; }
    std::vector<int> GetChannelMap() const;

    bool Open(const SDL_AudioSpec* spec);
    bool IsPhysical() const;
    bool IsPlayback() const;
    bool Pause();
    bool Resume();
    bool IsPaused();
    // The gain of a device is its volume; a larger gain means a louder output,
    // with a gain of zero being silence.
    // Audio devices default to a gain of 1.0f (no change in output).
    float GetGain() const;
    bool SetGain(float gain) const;
    void Close();

    bool BindStreams(SDL_AudioStream* const* streams, int streamCount);
    void UnbindAudioStreams(SDL_AudioStream* const* streams, int streamCount);
    bool BindStream(SDL_AudioStream* stream);
    void UnbindAudioStream(SDL_AudioStream* stream);

    bool SetIterationCallbacks(SDL_AudioIterationCallback start, SDL_AudioIterationCallback end, void* userdata);
    bool SetPostmixCallback(SDL_AudioPostmixCallback callback, void* userData);

private:
    SDL_AudioDeviceID m_id;
    std::string m_name;
    SDL_AudioSpec m_spec = {};
    int m_sampleFrames = 0;

}; // class AudioDevice


class AudioStream : public rad::RefCounted<AudioStream>
{
public:
    static rad::Ref<AudioStream> Create(const SDL_AudioSpec* srcSpec, const SDL_AudioSpec* dstSpec);
    // Convenience function if all your app intends to do is provide a single source of PCM audio.
    static rad::Ref<AudioStream> Create(SDL_AudioDeviceID deviceID, const SDL_AudioSpec* spec,
        SDL_AudioStreamCallback callback, void* userData);
    static rad::Ref<AudioStream> CreateDefaultPlayback(const SDL_AudioSpec* spec);

    AudioStream(SDL_AudioStream* handle);
    ~AudioStream();

    void Destroy();

    SDL_AudioDeviceID GetDeviceID() const;
    bool GetFormat(SDL_AudioSpec* srcSpec, SDL_AudioSpec* dstSpec);
    bool SetFormat(const SDL_AudioSpec* srcSpec, const SDL_AudioSpec* dstSpec);
    float GetFrequencyRatio();
    bool SetFrequencyRatio(float ratio);
    float GetGain() const;
    bool SetGain(float gain) const;
    std::vector<int> GetInputChannelMap() const;
    std::vector<int> GetOutputChannelMap() const;
    bool SetInputChannelMap(rad::ArrayRef<int> map);
    bool SetOutputChannelMap(rad::ArrayRef<int> map);
    bool PutData(const void* data, int sizeInBytes);
    bool PutPlanarData(const void* const* data, int numChannels, int numSamples);
    // Returns the number of bytes read from the stream or -1 on failure.
    int GetData(void* data, int sizeInBytes);
    //  Get the number of converted/resampled bytes available.
    int GetDataSizeAvailable();
    // Get the number of bytes that are put into a stream as input currently queued.
    int GetDataSizeQueued();
    bool Flush();
    bool Clear();
    bool Pause();
    bool Resume();
    bool IsPaused();
    bool Lock();
    bool Unlock();

    bool SetAudioStreamGetCallback(SDL_AudioStreamCallback callback, void* userData);
    bool SetAudioStreamPutCallback(SDL_AudioStreamCallback callback, void* userData);

private:
    SDL_AudioStream* m_handle;
    SDL_PropertiesID m_propID = 0;

}; // class AudioStream

} // namespace sdf
