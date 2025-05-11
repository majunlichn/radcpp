#pragma once

#include <SDFramework/Media/Audio.h>
#include <SDL3_mixer/SDL_mixer.h>

namespace sdf
{

class AudioChunk : public rad::RefCounted<AudioChunk>
{
public:
    AudioChunk(Mix_Chunk* chunk);
    ~AudioChunk();

    Mix_Chunk* GetChunk() { return m_chunk; }
    int SetVolume(int volume);
    int GetVolume();

private:
    Mix_Chunk* m_chunk;

}; // class AudioChunk

class Music : public rad::RefCounted<Music>
{
public:
    Music(Mix_Music* music);
    ~Music();

    Mix_Music* GetMusic() { return m_music; }

    Mix_MusicType GetType();
    const char* GetTitle();
    const char* GetTag();
    const char* GetArtistTag();
    const char* GetAlbumTag();
    const char* GetCopyrightTag();

    static int SetVolume(int volume);
    int GetVolume();

    int GetTrackCount();
    bool StartTrack(int track);
    double GetPosition();
    double GetDuration();
    double GetLoopStartTime();
    double GetLoopEndTime();
    double GetLoopLengthTime();

private:
    Mix_Music* m_music;
}; // class Music

class AudioMixer : public rad::RefCounted<AudioMixer>
{
public:
    AudioMixer();
    ~AudioMixer();

    bool Open(SDL_AudioDeviceID deviceID, const SDL_AudioSpec* spec);
    void Close();

    bool IsFormatSupported(int flags) const;

    void Pause();
    void Resume();
    bool QuerySpec(int* frequency, SDL_AudioFormat* format, int* channels);
    int AllocateChannels(int channelCount);
    rad::Ref<AudioChunk> LoadWAV(SDL_IOStream* src, bool closeio);
    rad::Ref<AudioChunk> LoadWAVFromFile(std::string_view fileName);
    rad::Ref<AudioChunk> LoadWAVFromMemory(Uint8* memory);
    rad::Ref<Music> LoadMusic(SDL_IOStream* src, bool closeio);
    rad::Ref<Music> LoadMusic(SDL_IOStream* src, Mix_MusicType type, bool closeio);
    rad::Ref<Music> LoadMusicFromFile(std::string_view fileName);
    rad::Ref<AudioChunk> LoadRawFromMemory(Uint8* memory, Uint32 sizeInBytes);

    std::vector<const char*> GetChunkDecoders();
    bool HasChunkDecoder(std::string_view name);
    std::vector<const char*> GetMusicDecoders();
    bool HasMusicDecoder(std::string_view name);

    void SetPostMixingCallback(Mix_MixCallback callback, void* arg);
    void HookMusic(Mix_MixCallback mixFunc, void* arg);
    void* GetMusicHookData();
    void HookMusicFinished(Mix_MusicFinishedCallback musicFinished);

    void SetChannelFinishedCallback(Mix_ChannelFinishedCallback channelFinished);
    bool RegisterEffect(int channel, Mix_EffectFunc_t f, Mix_EffectDone_t d, void* arg);
    bool UnregisterEffect(int channel, Mix_EffectFunc_t f);
    bool UnregisterAllEffects(int channel);

    bool SetPanning(int channel, Uint8 left, Uint8 right);
    bool SetPosition(int channel, Sint16 angle, Uint8 distance);
    bool SetDistance(int channel, Uint8 distance);
    bool SetReverseStereo(int channel, bool flip);

    int ReserveChannels(int num);
    bool GroupChannel(int which, int tag);
    bool GroupChannels(int from, int to, int tag);
    int GetGroupChannelAvailable(int tag);
    int GetGroupChannelCount(int tag);
    int GetGroupChannelOldest(int tag);
    int GetGroupChannelNewer(int tag);

    int SetVolume(int channel, int volume);
    int GetVolume(int channel);
    int SetMusicVolume(int volume);
    int GetMusicVolume();
    int SetMasterVolume(int volume);
    int GetMasterVolume();

    int PlayChannel(int channel, AudioChunk* chunk, int loops);
    int PlayChannelTimed(int channel, AudioChunk* chunk, int loops, int ticks);
    int PlayChannelFadeIn(int channel, AudioChunk* chunk, int loops, int ms);
    int PlayChannelFadeInTimed(int channel, AudioChunk* chunk, int loops, int ms, int ticks);
    Mix_Chunk* GetChunk(int channel);
    bool PlayMusic(Music* music, int loops);
    bool PlayMusicFadeIn(Music* music, int loops, int ms);
    bool PlayMusicFadeInFromPosition(Music* music, int loops, int ms, double position);
    void HaltChannel(int channel);
    void HaltGroup(int tag);
    void HaltMusic();
    int ExpireChannel(int channel, int ticks);
    int ExpireAllChannels(int ticks);
    int FadeOutChannel(int which, int ms);
    int FadeOutGroup(int tag, int ms);
    bool FadeOutMusic(int ms);
    Mix_Fading GetMusicFading();
    Mix_Fading GetChannelFading(int which);
    void Pause(int channel);
    void PauseGroup(int tag);
    void Resume(int channel);
    void ResumeGroup(int tag);
    bool IsChannelPaused(int channel);
    int GetPausedChannelCount();
    void PauseMusic();
    void ResumeMusic();
    void RewindMusic();
    bool IsMusicPaused();
    bool ModMusicJumpToOrder(int order);
    bool SetMusicPosition(double position);

    bool IsChannelPlaying(int channel);
    int GetPlayingChannelCount();
    bool IsPlayingMusic();

    bool SetSoundFonts(std::string_view paths);
    const char* GetSoundFonts();
    bool IterateSoundFonts(Mix_EachSoundFontCallback function, void* data);

    bool SetTimidityCfg(std::string_view path);
    const char* GetTimidityCfg();

private:
    // Save initialized modules, with bits of MIX_InitFlags.
    MIX_InitFlags m_modules = 0;
    bool m_opened = false;

}; // class AudioMixer

} // namespace sdf
