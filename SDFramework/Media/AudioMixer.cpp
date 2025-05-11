#include <SDFramework/Media/AudioMixer.h>

namespace sdf
{

AudioChunk::AudioChunk(Mix_Chunk* chunk) :
    m_chunk(chunk)
{
}

AudioChunk::~AudioChunk()
{
    if (m_chunk)
    {
        Mix_FreeChunk(m_chunk);
        m_chunk = nullptr;
    }
}

int AudioChunk::SetVolume(int volume)
{
    return Mix_VolumeChunk(m_chunk, volume);
}

int AudioChunk::GetVolume()
{
    return Mix_VolumeChunk(m_chunk, -1);
}

Music::Music(Mix_Music* music) :
    m_music(music)
{
}

Music::~Music()
{
    if (m_music)
    {
        Mix_FreeMusic(m_music);
        m_music = nullptr;
    }
}

Mix_MusicType Music::GetType()
{
    return Mix_GetMusicType(m_music);
}

const char* Music::GetTitle()
{
    return Mix_GetMusicTitle(m_music);
}

const char* Music::GetTag()
{
    return Mix_GetMusicTitleTag(m_music);
}

const char* Music::GetArtistTag()
{
    return Mix_GetMusicArtistTag(m_music);
}

const char* Music::GetAlbumTag()
{
    return Mix_GetMusicAlbumTag(m_music);
}

const char* Music::GetCopyrightTag()
{
    return Mix_GetMusicCopyrightTag(m_music);
}

int Music::SetVolume(int volume)
{
    return Mix_VolumeMusic(volume);
}

int Music::GetVolume()
{
    return Mix_GetMusicVolume(m_music);
}

int Music::GetTrackCount()
{
    return Mix_GetNumTracks(m_music);
}

bool Music::StartTrack(int track)
{
    bool result = Mix_StartTrack(m_music, track);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "Mix_StartTrack failed: {}", SDL_GetError());
        return false;
    }
}

double Music::GetPosition()
{
    return Mix_GetMusicPosition(m_music);
}

double Music::GetDuration()
{
    return Mix_MusicDuration(m_music);
}

double Music::GetLoopStartTime()
{
    return Mix_GetMusicLoopStartTime(m_music);
}

double Music::GetLoopEndTime()
{
    return Mix_GetMusicLoopEndTime(m_music);
}

double Music::GetLoopLengthTime()
{
    return Mix_GetMusicLoopLengthTime(m_music);
}

AudioMixer::AudioMixer()
{
    m_modules = Mix_Init(
        MIX_INIT_FLAC | MIX_INIT_MOD |
        MIX_INIT_MP3 | MIX_INIT_OGG |
        MIX_INIT_MID | MIX_INIT_OPUS |
        MIX_INIT_WAVPACK);

    std::string moduleNames;
    if (rad::HasBits<uint32_t>(m_modules, MIX_INIT_FLAC))
    {
        moduleNames += "FLAC,";
    }
    if (rad::HasBits<uint32_t>(m_modules, MIX_INIT_MOD))
    {
        moduleNames += "MOD,";
    }
    if (rad::HasBits<uint32_t>(m_modules, MIX_INIT_MP3))
    {
        moduleNames += "MP3,";
    }
    if (rad::HasBits<uint32_t>(m_modules, MIX_INIT_OGG))
    {
        moduleNames += "OGG,";
    }
    if (rad::HasBits<uint32_t>(m_modules, MIX_INIT_MID))
    {
        moduleNames += "MID,";
    }
    if (rad::HasBits<uint32_t>(m_modules, MIX_INIT_OPUS))
    {
        moduleNames += "OPUS,";
    }
    if (rad::HasBits<uint32_t>(m_modules, MIX_INIT_WAVPACK))
    {
        moduleNames += "WAVEPACK,";
    }

    if ((m_modules != 0) && !moduleNames.empty())
    {
        moduleNames.pop_back();
        SDF_LOG(info, "Initialized audio modules: {}", moduleNames);
    }
}

AudioMixer::~AudioMixer()
{
    if (m_opened)
    {
        Close();
    }
    Mix_Quit();
    m_modules = 0;
}

bool AudioMixer::Open(SDL_AudioDeviceID deviceID, const SDL_AudioSpec* spec)
{
    bool result = Mix_OpenAudio(deviceID, spec);
    if (result)
    {
        m_opened = true;
        return true;
    }
    else
    {
        SDF_LOG(err, "Mix_OpenAudio failed: {}", SDL_GetError());
        return false;
    }
}

void AudioMixer::Close()
{
    assert(m_opened);
    Mix_CloseAudio();
    m_opened = false;
}

bool AudioMixer::IsFormatSupported(int flags) const
{
    return rad::HasBits<uint32_t>(m_modules, flags);
}

void AudioMixer::Pause()
{
    assert(m_opened);
    Mix_PauseAudio(1);
}

void AudioMixer::Resume()
{
    assert(m_opened);
    Mix_PauseAudio(0);
}

bool AudioMixer::QuerySpec(int* frequency, SDL_AudioFormat* format, int* channels)
{
    assert(m_opened);
    return Mix_QuerySpec(frequency, format, channels);
}

int AudioMixer::AllocateChannels(int channelCount)
{
    return Mix_AllocateChannels(channelCount);
}

rad::Ref<AudioChunk> AudioMixer::LoadWAV(SDL_IOStream* src, bool closeio)
{
    Mix_Chunk* chunk = Mix_LoadWAV_IO(src, closeio);
    if (chunk)
    {
        return RAD_NEW AudioChunk(chunk);
    }
    else
    {
        return nullptr;
    }
}

rad::Ref<AudioChunk> AudioMixer::LoadWAVFromFile(std::string_view fileName)
{
    Mix_Chunk* chunk = Mix_LoadWAV(fileName.data());
    if (chunk)
    {
        return RAD_NEW AudioChunk(chunk);
    }
    else
    {
        return nullptr;
    }
}

rad::Ref<AudioChunk> AudioMixer::LoadWAVFromMemory(Uint8* memory)
{
    Mix_Chunk* chunk = Mix_QuickLoad_WAV(memory);
    if (chunk)
    {
        return RAD_NEW AudioChunk(chunk);
    }
    else
    {
        return nullptr;
    }
}

rad::Ref<Music> AudioMixer::LoadMusic(SDL_IOStream* src, bool closeio)
{
    Mix_Music* music = Mix_LoadMUS_IO(src, closeio);
    if (music)
    {
        return RAD_NEW Music(music);
    }
    else
    {
        return nullptr;
    }
}

rad::Ref<Music> AudioMixer::LoadMusic(SDL_IOStream* src, Mix_MusicType type, bool closeio)
{
    Mix_Music* music = Mix_LoadMUSType_IO(src, type, closeio);
    if (music)
    {
        return RAD_NEW Music(music);
    }
    else
    {
        return nullptr;
    }
}

rad::Ref<Music> AudioMixer::LoadMusicFromFile(std::string_view fileName)
{
    Mix_Music* music = Mix_LoadMUS(fileName.data());
    if (music)
    {
        return RAD_NEW Music(music);
    }
    else
    {
        SDF_LOG(err, "Mix_LoadMUS(\"{}\") failed: {}",
            fileName, SDL_GetError());
        return nullptr;
    }
}

rad::Ref<AudioChunk> AudioMixer::LoadRawFromMemory(Uint8* memory, Uint32 sizeInBytes)
{
    Mix_Chunk* chunk = Mix_QuickLoad_RAW(memory, sizeInBytes);
    if (chunk)
    {
        return RAD_NEW AudioChunk(chunk);
    }
    else
    {
        return nullptr;
    }
}

std::vector<const char*> AudioMixer::GetChunkDecoders()
{
    std::vector<const char*> decoders;
    int count = Mix_GetNumChunkDecoders();
    if (count > 0)
    {
        decoders.resize(count);
        for (int i = 0; i < count; ++i)
        {
            decoders[i] = Mix_GetChunkDecoder(i);
        }
    }
    return decoders;
}

bool AudioMixer::HasChunkDecoder(std::string_view name)
{
    return (Mix_HasChunkDecoder(name.data()) == true);
}

std::vector<const char*> AudioMixer::GetMusicDecoders()
{
    std::vector<const char*> decoders;
    int count = Mix_GetNumMusicDecoders();
    if (count > 0)
    {
        decoders.resize(count);
        for (int i = 0; i < count; ++i)
        {
            decoders[i] = Mix_GetMusicDecoder(i);
        }
    }
    return decoders;
}

bool AudioMixer::HasMusicDecoder(std::string_view name)
{
    return (Mix_HasMusicDecoder(name.data()) == true);
}

void AudioMixer::SetPostMixingCallback(Mix_MixCallback callback, void* arg)
{
    Mix_SetPostMix(callback, arg);
}

void AudioMixer::HookMusic(Mix_MixCallback mixFunc, void* arg)
{
    Mix_HookMusic(mixFunc, arg);
}

void* AudioMixer::GetMusicHookData()
{
    return Mix_GetMusicHookData();
}

void AudioMixer::HookMusicFinished(Mix_MusicFinishedCallback musicFinished)
{
    Mix_HookMusicFinished(musicFinished);
}

void AudioMixer::SetChannelFinishedCallback(Mix_ChannelFinishedCallback channelFinished)
{
    Mix_ChannelFinished(channelFinished);
}

bool AudioMixer::RegisterEffect(int channel, Mix_EffectFunc_t f, Mix_EffectDone_t d, void* arg)
{
    bool result = Mix_RegisterEffect(channel, f, d, arg);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "Mix_RegisterEffect failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioMixer::UnregisterEffect(int channel, Mix_EffectFunc_t f)
{
    bool result = Mix_UnregisterEffect(channel, f);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "Mix_UnregisterEffect failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioMixer::UnregisterAllEffects(int channel)
{
    bool result = Mix_UnregisterAllEffects(channel);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "Mix_UnregisterAllEffects failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioMixer::SetPanning(int channel, Uint8 left, Uint8 right)
{
    bool result = Mix_SetPanning(channel, left, right);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "Mix_SetPanning failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioMixer::SetPosition(int channel, Sint16 angle, Uint8 distance)
{
    bool result = Mix_SetPosition(channel, angle, distance);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "Mix_SetPosition failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioMixer::SetDistance(int channel, Uint8 distance)
{
    bool result = Mix_SetDistance(channel, distance);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "Mix_SetDistance failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioMixer::SetReverseStereo(int channel, bool flip)
{
    bool result = Mix_SetReverseStereo(channel, flip);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "Mix_SetReverseStereo failed: {}", SDL_GetError());
        return false;
    }
}

int AudioMixer::ReserveChannels(int num)
{
    return Mix_ReserveChannels(num);
}

bool AudioMixer::GroupChannel(int which, int tag)
{
    bool result = Mix_GroupChannel(which, tag);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "Mix_GroupChannel failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioMixer::GroupChannels(int from, int to, int tag)
{
    bool result = Mix_GroupChannels(from, to, tag);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "Mix_GroupChannels failed: {}", SDL_GetError());
        return false;
    }
}

int AudioMixer::GetGroupChannelAvailable(int tag)
{
    return Mix_GroupAvailable(tag);
}

int AudioMixer::GetGroupChannelCount(int tag)
{
    return Mix_GroupCount(tag);
}

int AudioMixer::GetGroupChannelOldest(int tag)
{
    return Mix_GroupOldest(tag);
}

int AudioMixer::GetGroupChannelNewer(int tag)
{
    return Mix_GroupNewer(tag);
}

int AudioMixer::SetVolume(int channel, int volume)
{
    return Mix_Volume(channel, volume);
}

int AudioMixer::GetVolume(int channel)
{
    return Mix_Volume(channel, -1);
}

int AudioMixer::SetMusicVolume(int volume)
{
    return Mix_VolumeMusic(volume);
}

int AudioMixer::GetMusicVolume()
{
    return Mix_VolumeMusic(-1);
}

int AudioMixer::SetMasterVolume(int volume)
{
    return Mix_MasterVolume(volume);
}

int AudioMixer::GetMasterVolume()
{
    return Mix_MasterVolume(-1);
}

int AudioMixer::PlayChannel(int channel, AudioChunk* chunk, int loops)
{
    return Mix_PlayChannel(channel, chunk->GetChunk(), loops);
}

int AudioMixer::PlayChannelTimed(int channel, AudioChunk* chunk, int loops, int ticks)
{
    return Mix_PlayChannelTimed(channel, chunk->GetChunk(), loops, ticks);
}

int AudioMixer::PlayChannelFadeIn(int channel, AudioChunk* chunk, int loops, int ms)
{
    return Mix_FadeInChannel(channel, chunk->GetChunk(), loops, ms);
}

int AudioMixer::PlayChannelFadeInTimed(int channel, AudioChunk* chunk, int loops, int ms, int ticks)
{
    return Mix_FadeInChannelTimed(channel, chunk->GetChunk(), loops, ms, ticks);
}

Mix_Chunk* AudioMixer::GetChunk(int channel)
{
    return Mix_GetChunk(channel);
}

bool AudioMixer::PlayMusic(Music* music, int loops)
{
    bool result = Mix_PlayMusic(music->GetMusic(), loops);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "Mix_PlayMusic failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioMixer::PlayMusicFadeIn(Music* music, int loops, int ms)
{
    bool result = Mix_FadeInMusic(music->GetMusic(), loops, ms);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "Mix_FadeInMusic failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioMixer::PlayMusicFadeInFromPosition(Music* music, int loops, int ms, double position)
{
    bool result = Mix_FadeInMusicPos(music->GetMusic(), loops, ms, position);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "Mix_FadeInMusicPos failed: {}", SDL_GetError());
        return false;
    }
}

void AudioMixer::HaltChannel(int channel)
{
    Mix_HaltChannel(channel);
}

void AudioMixer::HaltGroup(int tag)
{
    Mix_HaltGroup(tag);
}

void AudioMixer::HaltMusic()
{
    Mix_HaltMusic();
}

int AudioMixer::ExpireChannel(int channel, int ticks)
{
    return Mix_ExpireChannel(channel, ticks);
}

int AudioMixer::ExpireAllChannels(int ticks)
{
    return Mix_ExpireChannel(-1, ticks);
}

int AudioMixer::FadeOutChannel(int which, int ms)
{
    return Mix_FadeOutChannel(which, ms);
}

int AudioMixer::FadeOutGroup(int tag, int ms)
{
    return Mix_FadeOutGroup(tag, ms);
}

bool AudioMixer::FadeOutMusic(int ms)
{
    return Mix_FadeOutMusic(ms);
}

Mix_Fading AudioMixer::GetMusicFading()
{
    return Mix_FadingMusic();
}

Mix_Fading AudioMixer::GetChannelFading(int which)
{
    return Mix_FadingChannel(which);
}

void AudioMixer::Pause(int channel)
{
    Mix_Pause(channel);
}

void AudioMixer::PauseGroup(int tag)
{
    Mix_PauseGroup(tag);
}

void AudioMixer::Resume(int channel)
{
    Mix_Resume(channel);
}

void AudioMixer::ResumeGroup(int tag)
{
    Mix_ResumeGroup(tag);
}

bool AudioMixer::IsChannelPaused(int channel)
{
    int result = Mix_Paused(channel);
    return (result == 1);
}

int AudioMixer::GetPausedChannelCount()
{
    return Mix_Paused(-1);
}

void AudioMixer::PauseMusic()
{
    Mix_PauseMusic();
}

void AudioMixer::ResumeMusic()
{
    Mix_ResumeMusic();
}

void AudioMixer::RewindMusic()
{
    Mix_RewindMusic();
}

bool AudioMixer::IsMusicPaused()
{
    return (Mix_PausedMusic() == 1);
}

bool AudioMixer::ModMusicJumpToOrder(int order)
{
    bool result = Mix_ModMusicJumpToOrder(order);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "Mix_ModMusicJumpToOrder failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioMixer::SetMusicPosition(double position)
{
    bool result = Mix_SetMusicPosition(position);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "Mix_SetMusicPosition failed: {}", SDL_GetError());
        return false;
    }
}

bool AudioMixer::IsChannelPlaying(int channel)
{
    int result = Mix_Playing(channel);
    return (result != 0);
}

int AudioMixer::GetPlayingChannelCount()
{
    return Mix_Playing(-1);
}

bool AudioMixer::IsPlayingMusic()
{
    return Mix_PlayingMusic();
}

bool AudioMixer::SetSoundFonts(std::string_view paths)
{
    bool result = Mix_SetSoundFonts(paths.data());
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "Mix_SetSoundFonts failed: {}", SDL_GetError());
        return false;
    }
}

const char* AudioMixer::GetSoundFonts()
{
    return Mix_GetSoundFonts();
}

bool AudioMixer::IterateSoundFonts(Mix_EachSoundFontCallback function, void* data)
{
    return Mix_EachSoundFont(function, data);
}

bool AudioMixer::SetTimidityCfg(std::string_view path)
{
    bool result = Mix_SetTimidityCfg(path.data());
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "Mix_SetTimidityCfg failed: {}", SDL_GetError());
        return false;
    }
}

const char* AudioMixer::GetTimidityCfg()
{
    return Mix_GetTimidityCfg();
}

} // namespace sdf
