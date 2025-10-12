#pragma once

#include <SDFramework/Multimedia/Audio.h>
#include <SDL3_mixer/SDL_mixer.h>

namespace sdf
{

class AudioMixer;

class Audio : public rad::RefCounted<Audio>
{
public:
    Audio(MIX_Audio* handle);
    ~Audio();

    MIX_Audio* GetHandle() const { return m_handle; }

    // MIX_DURATION_UNKNOWN
    // MIX_DURATION_INFINITE
    Sint64 GetDurationInFrames();
    Sint64 GetDurationInMS() { return FramesToMS(GetDurationInFrames()); }
    bool GetFormat(SDL_AudioSpec* spec);

    const char* GetTitle();
    const char* GetArtist();
    const char* GetAlbum();
    const char* GetCopyright();
    Sint64 GetTrackIndex();
    Sint64 GetTotalTrackCount();
    Sint64 GetYear();
    // Same as GetDurationInFrames.
    Sint64 GetFrameCount();
    bool IsInfinite();

    Sint64 MSToFrames(Sint64 ms);
    Sint64 FramesToMS(Sint64 frames);

private:
    MIX_Audio* m_handle;
    SDL_PropertiesID m_props = 0;
}; // class Audio

class AudioTrack : public rad::RefCounted<AudioTrack>
{
public:
    AudioTrack(rad::Ref<AudioMixer> mixer, MIX_Track* handle);
    ~AudioTrack();

    bool SetAudio(MIX_Audio* audio);
    bool SetAudio(Audio* audio);
    bool SetAudioStream(SDL_AudioStream* stream);
    bool SetIOStream(SDL_IOStream* io, bool closeIO);
    bool SetRawIOStream(SDL_IOStream* io, const SDL_AudioSpec* spec, bool closeIO);

    bool Tag(std::string_view tag);
    void Untag(std::string_view tag);

    bool SetPlaybackPosition(Sint64 frames);
    Sint64 GetPlaybackPosition();

    bool IsLooping();

    MIX_Audio* GetTrackAudio();
    SDL_AudioStream* GetTrackAudioStream();

    Sint64 GetRemainingFrames();
    Sint64 MSToFrames(Sint64 ms);
    Sint64 FramesToMS(Sint64 frames);

    bool Play(SDL_PropertiesID options);
    bool Stop(Sint64 fadeOutFrames);
    bool Pause();
    bool Resume();

    bool IsPlaying();
    bool IsPaused();

    bool SetGain(float gain);
    float GetGain();

    bool SetFrequencyRatio(float ratio);
    float GetFrequencyRatio();

    bool SetOutputChannelMap(const int* chmap, int count);
    bool SetStereo(const MIX_StereoGains* gains);
    bool Set3DPosition(const MIX_Point3D* position);
    bool Get3DPosition(MIX_Point3D* position);

    rad::Ref<AudioMixer> m_mixer;
    MIX_Track* m_handle;
    SDL_PropertiesID m_props = 0;

}; // class AudioTrack

class AudioMixer : public rad::RefCounted<AudioMixer>
{
public:
    AudioMixer(SDL_AudioDeviceID deviceID, const SDL_AudioSpec* spec);
    ~AudioMixer();

    MIX_Mixer* GetHandle() const { return m_handle; }

    bool HasDecoder(std::string_view decoder);

    bool GetFormat(SDL_AudioSpec* spec);

    rad::Ref<Audio> LoadAudioIO(SDL_IOStream* io, bool predecode, bool closeIO);
    rad::Ref<Audio> LoadAudio(std::string_view path, bool predecode);
    rad::Ref<Audio> LoadAudioWithProperties(SDL_PropertiesID props);
    rad::Ref<Audio> LoadRawAudioIO(SDL_IOStream* io, const SDL_AudioSpec* spec, bool closeIO);
    rad::Ref<Audio> LoadRawAudio(const void* data, size_t dataSize, const SDL_AudioSpec* spec);
    rad::Ref<Audio> LoadRawAudioNoCopy(const void* data, size_t dataSize, const SDL_AudioSpec* spec);
    rad::Ref<Audio> CreateSineWaveAudio(int hz, int amplitude);

    rad::Ref<AudioTrack> CreateTrack();

    bool PlayTag(std::string_view tag, SDL_PropertiesID options);
    bool PlayAudio(MIX_Audio* audio);
    bool StopAllTracks(Sint64 fadeOutMS);
    bool StopTag(std::string_view tag, Sint64 fadeOutMS);
    bool PauseAllTracks();
    bool PauseTag(std::string_view tag);
    bool ResumeAllTracks();
    bool ResumeTag(std::string_view tag);

    bool SetMasterGain(float gain);
    float GetMasterGain();

    bool SetTagGain(std::string_view tag, float gain);

    bool SetPostMixCallback(MIX_PostMixCallback cb, void* userData);
    bool Generate(void* buffer, int bufferSize);

private:
    int m_version = 0;
    std::vector<std::string> m_decoders;
    MIX_Mixer* m_handle = nullptr;
    SDL_PropertiesID m_props = 0;

}; // class AudioMixer

} // namespace sdf
