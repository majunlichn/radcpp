#include <SDFramework/Multimedia/AudioMixer.h>

namespace sdf
{

Audio::Audio(MIX_Audio* handle) :
    m_handle(handle)
{
    if (m_handle)
    {
        m_props = MIX_GetAudioProperties(m_handle);
        if (!m_props)
        {
            SDF_LOG(err, "MIX_GetAudioProperties failed: {}", SDL_GetError());
        }
    }
}

Audio::~Audio()
{
    if (m_handle)
    {
        MIX_DestroyAudio(m_handle);
        m_handle = nullptr;
    }
}

Sint64 Audio::GetDurationInFrames()
{
    return MIX_GetAudioDuration(m_handle);
}

bool Audio::GetFormat(SDL_AudioSpec* spec)
{
    return MIX_GetAudioFormat(m_handle, spec);
}

const char* Audio::GetTitle()
{
    return SDL_GetStringProperty(m_props, MIX_PROP_METADATA_TITLE_STRING, "");
}

const char* Audio::GetArtist()
{
    return SDL_GetStringProperty(m_props, MIX_PROP_METADATA_ARTIST_STRING, "");
}

const char* Audio::GetAlbum()
{
    return SDL_GetStringProperty(m_props, MIX_PROP_METADATA_ALBUM_STRING, "");
}

const char* Audio::GetCopyright()
{
    return SDL_GetStringProperty(m_props, MIX_PROP_METADATA_COPYRIGHT_STRING, "");
}

Sint64 Audio::GetTrackIndex()
{
    return SDL_GetNumberProperty(m_props, MIX_PROP_METADATA_TRACK_NUMBER, 0);
}

Sint64 Audio::GetTotalTrackCount()
{
    return SDL_GetNumberProperty(m_props, MIX_PROP_METADATA_TOTAL_TRACKS_NUMBER, 0);
}

Sint64 Audio::GetYear()
{
    return SDL_GetNumberProperty(m_props, MIX_PROP_METADATA_YEAR_NUMBER, 0);
}

Sint64 Audio::GetFrameCount()
{
    return SDL_GetNumberProperty(m_props, MIX_PROP_METADATA_DURATION_FRAMES_NUMBER, 0);
}

bool Audio::IsInfinite()
{
    return SDL_GetBooleanProperty(m_props, MIX_PROP_METADATA_DURATION_INFINITE_BOOLEAN, false);
}

Sint64 Audio::MSToFrames(Sint64 ms)
{
    return MIX_AudioMSToFrames(m_handle, ms);
}

Sint64 Audio::FramesToMS(Sint64 frames)
{
    return MIX_AudioFramesToMS(m_handle, frames);
}


AudioTrack::AudioTrack(rad::Ref<AudioMixer> mixer, MIX_Track* handle) :
    m_mixer(std::move(mixer)),
    m_handle(handle)
{
    if (m_handle)
    {
        MIX_Mixer* mixerHandle = MIX_GetTrackMixer(m_handle);
        assert(mixerHandle == mixer->GetHandle());

        m_props = MIX_GetTrackProperties(m_handle);
        if (!m_props)
        {
            SDF_LOG(err, "MIX_GetTrackProperties failed: {}", SDL_GetError());
        }
    }
}

AudioTrack::~AudioTrack()
{
    if (m_handle)
    {
        MIX_DestroyTrack(m_handle);
    }
}

bool AudioTrack::SetAudio(MIX_Audio* audio)
{
    return MIX_SetTrackAudio(m_handle, audio);
}

bool AudioTrack::SetAudio(Audio* audio)
{
    return SetAudio(audio->GetHandle());
}

bool AudioTrack::SetAudioStream(SDL_AudioStream* stream)
{
    return MIX_SetTrackAudioStream(m_handle, stream);
}

bool AudioTrack::SetIOStream(SDL_IOStream* io, bool closeIO)
{
    return MIX_SetTrackIOStream(m_handle, io, closeIO);
}

bool AudioTrack::SetRawIOStream(SDL_IOStream* io, const SDL_AudioSpec* spec, bool closeIO)
{
    return MIX_SetTrackRawIOStream(m_handle, io, spec, closeIO);
}

bool AudioTrack::Tag(std::string_view tag)
{
    return MIX_TagTrack(m_handle, tag.data());
}

void AudioTrack::Untag(std::string_view tag)
{
    MIX_UntagTrack(m_handle, tag.data());
}

bool AudioTrack::SetPlaybackPosition(Sint64 frames)
{
    return MIX_SetTrackPlaybackPosition(m_handle, frames);
}

Sint64 AudioTrack::GetPlaybackPosition()
{
    return MIX_GetTrackPlaybackPosition(m_handle);
}

bool AudioTrack::IsLooping()
{
    return MIX_TrackLooping(m_handle);
}

MIX_Audio* AudioTrack::GetTrackAudio()
{
    return MIX_GetTrackAudio(m_handle);
}

SDL_AudioStream* AudioTrack::GetTrackAudioStream()
{
    return MIX_GetTrackAudioStream(m_handle);
}

Sint64 AudioTrack::GetRemainingFrames()
{
    return MIX_GetTrackRemaining(m_handle);
}

Sint64 AudioTrack::MSToFrames(Sint64 ms)
{
    return MIX_TrackMSToFrames(m_handle, ms);
}

Sint64 AudioTrack::FramesToMS(Sint64 frames)
{
    return MIX_TrackFramesToMS(m_handle, frames);
}

bool AudioTrack::Play(SDL_PropertiesID options)
{
    return MIX_PlayTrack(m_handle, options);
}

bool AudioTrack::Stop(Sint64 fadeOutFrames)
{
    return MIX_StopTrack(m_handle, fadeOutFrames);
}

bool AudioTrack::Pause()
{
    return MIX_PauseTrack(m_handle);
}

bool AudioTrack::Resume()
{
    return MIX_ResumeTrack(m_handle);
}

bool AudioTrack::IsPlaying()
{
    return MIX_TrackPlaying(m_handle);
}

bool AudioTrack::IsPaused()
{
    return MIX_TrackPaused(m_handle);
}

bool AudioTrack::SetGain(float gain)
{
    return MIX_SetTrackGain(m_handle, gain);
}

float AudioTrack::GetGain()
{
    return MIX_GetTrackGain(m_handle);
}

bool AudioTrack::SetFrequencyRatio(float ratio)
{
    return MIX_SetTrackFrequencyRatio(m_handle, ratio);
}

float AudioTrack::GetFrequencyRatio()
{
    return MIX_GetTrackFrequencyRatio(m_handle);
}

bool AudioTrack::SetOutputChannelMap(const int* chmap, int count)
{
    return MIX_SetTrackOutputChannelMap(m_handle, chmap, count);
}

bool AudioTrack::SetStereo(const MIX_StereoGains* gains)
{
    return MIX_SetTrackStereo(m_handle, gains);
}

bool AudioTrack::Set3DPosition(const MIX_Point3D* position)
{
    return MIX_SetTrack3DPosition(m_handle, position);
}

bool AudioTrack::Get3DPosition(MIX_Point3D* position)
{
    return MIX_GetTrack3DPosition(m_handle, position);
}

AudioMixer::AudioMixer(SDL_AudioDeviceID deviceID, const SDL_AudioSpec* spec)
{
    if (MIX_Init())
    {
        m_version = MIX_Version();
        SDF_LOG(info, "SDL Mixer initialized: {}.{}.{}",
            SDL_VERSIONNUM_MAJOR(m_version),
            SDL_VERSIONNUM_MINOR(m_version),
            SDL_VERSIONNUM_MICRO(m_version));
    }
    else
    {
        SDF_LOG(err, "MIX_Init failed: {}", SDL_GetError());
    }

    int decoderCount = MIX_GetNumAudioDecoders();
    m_decoders.resize(decoderCount);
    for (int i = 0; i < decoderCount; ++i)
    {
        m_decoders[i] = MIX_GetAudioDecoder(i);
        SDF_LOG(info, "Decoder#{}: {}", i, m_decoders[i]);
    }

    m_handle = MIX_CreateMixerDevice(deviceID, spec);
    if (m_handle)
    {
        m_props = MIX_GetMixerProperties(m_handle);
        if (!m_props)
        {
            SDF_LOG(err, "MIX_GetMixerProperties failed: {}", SDL_GetError());
        }
    }
    else
    {
        SDF_LOG(err, "MIX_CreateMixerDevice failed: {}", SDL_GetError());
    }
}

AudioMixer::~AudioMixer()
{
    if (m_handle)
    {
        MIX_DestroyMixer(m_handle);
    }
    MIX_Quit();
}

bool AudioMixer::HasDecoder(std::string_view decoder)
{
    return std::find(m_decoders.begin(), m_decoders.end(), decoder) != m_decoders.end();
}

bool AudioMixer::GetFormat(SDL_AudioSpec* spec)
{
    return MIX_GetMixerFormat(m_handle, spec);;
}

rad::Ref<Audio> AudioMixer::LoadAudioIO(SDL_IOStream* io, bool predecode, bool closeIO)
{
    MIX_Audio* audio = MIX_LoadAudio_IO(m_handle, io, predecode, closeIO);
    if (audio)
    {
        return RAD_NEW Audio(audio);
    }
    else
    {
        SDF_LOG(err, "MIX_LoadAudio_IO failed: {}", SDL_GetError());
        return nullptr;
    }
}

rad::Ref<Audio> AudioMixer::LoadAudio(std::string_view path, bool predecode)
{
    MIX_Audio* audio = MIX_LoadAudio(m_handle, path.data(), predecode);
    if (audio)
    {
        return RAD_NEW Audio(audio);
    }
    else
    {
        SDF_LOG(err, "MIX_LoadAudio failed: {}", SDL_GetError());
        return nullptr;
    }
}

rad::Ref<Audio> AudioMixer::LoadAudioWithProperties(SDL_PropertiesID props)
{
    MIX_Audio* audio = MIX_LoadAudioWithProperties(props);
    if (audio)
    {
        return RAD_NEW Audio(audio);
    }
    else
    {
        SDF_LOG(err, "MIX_LoadAudioWithProperties failed: {}", SDL_GetError());
        return nullptr;
    }
}

rad::Ref<Audio> AudioMixer::LoadRawAudioIO(SDL_IOStream* io, const SDL_AudioSpec* spec, bool closeIO)
{
    MIX_Audio* audio = MIX_LoadRawAudio_IO(m_handle, io, spec, closeIO);
    if (audio)
    {
        return RAD_NEW Audio(audio);
    }
    else
    {
        SDF_LOG(err, "MIX_LoadRawAudio_IO failed: {}", SDL_GetError());
        return nullptr;
    }
}

rad::Ref<Audio> AudioMixer::LoadRawAudio(const void* data, size_t dataSize, const SDL_AudioSpec* spec)
{
    MIX_Audio* audio = MIX_LoadRawAudio(m_handle, data, dataSize, spec);
    if (audio)
    {
        return RAD_NEW Audio(audio);
    }
    else
    {
        SDF_LOG(err, "MIX_LoadRawAudio failed: {}", SDL_GetError());
        return nullptr;
    }
}

rad::Ref<Audio> AudioMixer::LoadRawAudioNoCopy(const void* data, size_t dataSize, const SDL_AudioSpec* spec)
{
    MIX_Audio* audio = MIX_LoadRawAudioNoCopy(m_handle, data, dataSize, spec, false);
    if (audio)
    {
        return RAD_NEW Audio(audio);
    }
    else
    {
        SDF_LOG(err, "MIX_LoadRawAudioNoCopy failed: {}", SDL_GetError());
        return nullptr;
    }
}

rad::Ref<Audio> AudioMixer::CreateSineWaveAudio(int hz, int amplitude)
{
    MIX_Audio* audio = MIX_CreateSineWaveAudio(m_handle, hz, amplitude);
    if (audio)
    {
        return RAD_NEW Audio(audio);
    }
    else
    {
        SDF_LOG(err, "MIX_CreateSineWaveAudio failed: {}", SDL_GetError());
        return nullptr;
    }
}

rad::Ref<AudioTrack> AudioMixer::CreateTrack()
{
    MIX_Track* track = MIX_CreateTrack(m_handle);
    if (track)
    {
        return RAD_NEW AudioTrack(this, track);
    }
    else
    {
        SDF_LOG(err, "MIX_CreateTrack failed: {}", SDL_GetError());
        return nullptr;
    }
}

bool AudioMixer::PlayTag(std::string_view tag, SDL_PropertiesID options)
{
    return MIX_PlayTag(m_handle, tag.data(), options);
}

bool AudioMixer::PlayAudio(MIX_Audio* audio)
{
    return MIX_PlayAudio(m_handle, audio);
}

bool AudioMixer::StopAllTracks(Sint64 fadeOutMS)
{
    return MIX_StopAllTracks(m_handle, fadeOutMS);
}

bool AudioMixer::StopTag(std::string_view tag, Sint64 fadeOutMS)
{
    return MIX_StopTag(m_handle, tag.data(), fadeOutMS);
}

bool AudioMixer::PauseAllTracks()
{
    return MIX_PauseAllTracks(m_handle);
}

bool AudioMixer::PauseTag(std::string_view tag)
{
    return MIX_PauseTag(m_handle, tag.data());
}

bool AudioMixer::ResumeAllTracks()
{
    return MIX_ResumeAllTracks(m_handle);
}

bool AudioMixer::ResumeTag(std::string_view tag)
{
    return MIX_ResumeTag(m_handle, tag.data());
}

bool AudioMixer::SetMasterGain(float gain)
{
    return MIX_SetMasterGain(m_handle, gain);
}

float AudioMixer::GetMasterGain()
{
    return MIX_GetMasterGain(m_handle);
}

bool AudioMixer::SetTagGain(std::string_view tag, float gain)
{
    return MIX_SetTagGain(m_handle, tag.data(), gain);
}

bool AudioMixer::SetPostMixCallback(MIX_PostMixCallback cb, void* userData)
{
    return MIX_SetPostMixCallback(m_handle, cb, userData);
}

bool AudioMixer::Generate(void* buffer, int bufferSize)
{
    return MIX_Generate(m_handle, buffer, bufferSize);
}

} // namespace sdf
