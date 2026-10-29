// sceAvPlayer without video or audio decoders. A movie plays for its real length, read from the MP4 header, as
// black NV12 frames with the timestamps and state events a real player gives, so games wait out their intro movies
// and cutscenes (and see looping movies wrap around) instead of waiting on frames that never come.

#include "hle.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace KytyWeb {
namespace {

HleLayout L;

uint8_t* Ptr(uint64_t a, uint64_t n) {
    if (a < L.base || n > L.size || a - L.base > L.size - n) return nullptr;
    return L.mem + (a - L.base);
}
template <class T> T Rd(uint64_t a) {
    T v{};
    if (auto* p = Ptr(a, sizeof(T))) memcpy(&v, p, sizeof(T));
    return v;
}
template <class T> void Wr(uint64_t a, T v) {
    if (auto* p = Ptr(a, sizeof(T))) memcpy(p, &v, sizeof(T));
}
std::string Str(uint64_t a) {
    std::string s;
    for (char c; s.size() < 512 && (c = Rd<char>(a + s.size())) != 0;) s += c;
    return s;
}

// Event ids passed to the game's callback.
constexpr int32_t kStateStop = 1, kStateReady = 2, kStatePlay = 3, kStatePause = 4, kStateBuffering = 5;
constexpr int32_t kErrInvalidParams = (int32_t)0x806A0001, kErrOperationFailed = (int32_t)0x806A0002;

// --- MP4 header ---

struct Movie {
    uint64_t duration_ms = 0;
    uint32_t width = 0, height = 0, fps = 0;
};

uint32_t Be32(const uint8_t* p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }
uint64_t Be64(const uint8_t* p) { return (uint64_t)Be32(p) << 32 | Be32(p + 4); }

// Calls fn(type, payload, payload size) for each box in [p, end).
template <class F> void Boxes(const uint8_t* p, const uint8_t* end, F fn) {
    while (end - p >= 8) {
        uint64_t size = Be32(p), header = 8;
        if (size == 1 && end - p >= 16) {
            size = Be64(p + 8);
            header = 16;
        } else if (size == 0) {
            size = (uint64_t)(end - p);
        }
        if (size < header || size > (uint64_t)(end - p)) return;
        fn(std::string((const char*)p + 4, 4), p + header, size - header);
        p += size;
    }
}

// The movie length (mvhd) and the frame size (tkhd) and rate (mdhd time scale over the first stts delta) of its
// first video track.
void ParseBoxes(const uint8_t* p, const uint8_t* end, Movie& m, uint32_t& timescale) {
    Boxes(p, end, [&](const std::string& type, const uint8_t* b, uint64_t n) {
        if (type == "mvhd" && n >= 32) {
            const bool v1 = b[0] == 1;
            const uint32_t scale = Be32(b + (v1 ? 20 : 12));
            const uint64_t length = v1 ? Be64(b + 24) : Be32(b + 16);
            if (scale) m.duration_ms = length * 1000 / scale;
        } else if (type == "trak") {
            Movie t;
            uint32_t ts = 0;
            ParseBoxes(b, b + n, t, ts);
            if (t.width && !m.width) {
                m.width = t.width;
                m.height = t.height;
                m.fps = t.fps;
            }
        } else if (type == "mdia" || type == "minf" || type == "stbl") {
            ParseBoxes(b, b + n, m, timescale);
        } else if (type == "tkhd") {
            const uint64_t at = b[0] == 1 ? 88 : 76;
            if (n >= at + 8) {
                m.width = Be32(b + at) >> 16;
                m.height = Be32(b + at + 4) >> 16;
            }
        } else if (type == "mdhd" && n >= 24) {
            timescale = Be32(b + (b[0] == 1 ? 20 : 12));
        } else if (type == "stts" && n >= 16 && timescale) {
            if (const uint32_t delta = Be32(b + 12)) m.fps = (timescale + delta / 2) / delta;
        }
    });
}

enum class Got { Ok, Pending, Failed };

Got ReadRange(const std::string& path, uint64_t off, uint64_t n, std::vector<uint8_t>& out) {
    const uint64_t buf = HleMalloc(n, 16);
    if (!buf) return Got::Failed;
    bool pending = false;
    const int64_t got = HleReadFileRange(path, off, n, buf, &pending);
    const Got res = pending ? Got::Pending : got == (int64_t)n ? Got::Ok : Got::Failed;
    if (res == Got::Ok) out.assign(Ptr(buf, n), Ptr(buf, n) + n);
    HleFree(buf);
    return res;
}

// --- players ---

struct Player {
    uint64_t object = 0, callback = 0;  // SceAvPlayerEventReplacement
    std::string path;
    bool source = false;  // a source was added
    bool ready = false;   // its length and frame size are known
    bool playing = false, paused = false, looping = false, ended = false;
    uint64_t probe_off = 0;  // the top-level MP4 box being looked at
    Movie movie;
    int64_t start_us = 0;   // the clock reading at movie position 0
    int64_t paused_us = 0;  // the clock reading when paused
    int64_t last_frame = -1;
    uint64_t frame = 0;  // a black NV12 frame, every frame handed out
    uint32_t pitch = 0, rows = 0;
    std::deque<int32_t> events;  // waiting for the next call into the player
};
std::unordered_map<uint64_t, Player> g_players;
int g_firing = 0;

// Events go to the callback from the player's own calls, never from inside a callback (which may call the player).
void Fire(Player& p) {
    if (g_firing || !p.callback) return;
    g_firing++;
    while (!p.events.empty()) {
        const int32_t id = p.events.front();
        p.events.pop_front();
        L.call_guest4(p.callback, p.object, (uint64_t)(int64_t)id, 0, 0);
    }
    g_firing--;
}

void MakeReady(Player& p, bool parsed) {
    Movie& m = p.movie;
    if (!parsed || !m.duration_ms) m.duration_ms = 5000;
    if (!m.width || !m.height) m.width = 1920, m.height = 1080;
    if (m.fps < 1 || m.fps > 120) m.fps = 30;
    p.pitch = (m.width + 15) & ~15u;
    p.rows = (m.height + 15) & ~15u;
    const uint64_t luma = (uint64_t)p.pitch * p.rows;
    p.frame = HleMalloc(luma * 3 / 2, 0x10000);
    if (uint8_t* f = Ptr(p.frame, luma * 3 / 2)) {
        memset(f, 16, luma);
        memset(f + luma, 128, luma / 2);
    }
    p.ready = true;
    p.events.push_back(kStateReady);
    printf("[avplayer] %s: %llu ms, %ux%u, %u fps%s\n", p.path.c_str(), (unsigned long long)m.duration_ms, m.width,
           m.height, m.fps, parsed ? "" : " (header not read; assumed)");
}

// Walks the top-level boxes to moov. A streamed file may not have the data yet; the next call tries again.
void Probe(Player& p) {
    uint64_t size = 0;
    if (!HleFileSize(p.path, &size)) return MakeReady(p, false);
    for (int steps = 0; steps < 64 && p.probe_off + 8 <= size; steps++) {
        std::vector<uint8_t> h;
        const Got got = ReadRange(p.path, p.probe_off, std::min<uint64_t>(16, size - p.probe_off), h);
        if (got == Got::Pending) return;
        if (got == Got::Failed) break;
        uint64_t box = Be32(h.data()), header = 8;
        if (box == 1 && h.size() >= 16) {
            box = Be64(h.data() + 8);
            header = 16;
        } else if (box == 0) {
            box = size - p.probe_off;
        }
        if (box < header || box > size - p.probe_off) break;
        if (memcmp(h.data() + 4, "moov", 4) == 0) {
            if (box > (64u << 20)) break;
            std::vector<uint8_t> moov;
            const Got body = ReadRange(p.path, p.probe_off + header, box - header, moov);
            if (body == Got::Pending) return;
            if (body == Got::Failed) break;
            uint32_t timescale = 0;
            ParseBoxes(moov.data(), moov.data() + moov.size(), p.movie, timescale);
            return MakeReady(p, p.movie.duration_ms > 0);
        }
        p.probe_off += box;
    }
    MakeReady(p, false);
}

int64_t Now() { return HleUsecSinceStart(); }

// Brings the player up to the clock: finishes reading the header, ends or wraps the movie, and sends events.
void Update(Player& p) {
    if (p.source && !p.ready) Probe(p);
    if (p.ready && p.playing && !p.paused) {
        const int64_t length_us = (int64_t)p.movie.duration_ms * 1000, now = Now();
        if (now - p.start_us >= length_us) {
            if (p.looping) {
                p.start_us += (now - p.start_us) / length_us * length_us;
                p.last_frame = -1;
            } else {
                p.playing = false;
                p.ended = true;
                p.events.push_back(kStateStop);
                printf("[avplayer] %s ended\n", p.path.c_str());
            }
        }
    }
    Fire(p);
}

Player* Find(uint64_t handle) {
    auto it = g_players.find(handle);
    if (it == g_players.end()) return nullptr;
    Update(it->second);
    return &it->second;
}

uint64_t PositionMs(const Player& p) {
    if (!p.playing) return 0;
    return (uint64_t)std::max<int64_t>(0, (p.paused ? p.paused_us : Now()) - p.start_us) / 1000;
}

uint64_t Create(uint64_t event_object, uint64_t callback) {
    const uint64_t handle = HleMalloc(64, 16);
    Player& p = g_players[handle];
    p.object = event_object;
    p.callback = callback;
    return handle;
}

// SceAvPlayerInitData: memory replacement (5 pointers), file replacement (5), event replacement (object, callback).
void Init(X86Regs& r) { r.rax = r.rdi ? Create(Rd<uint64_t>(r.rdi + 80), Rd<uint64_t>(r.rdi + 88)) : 0; }
// SceAvPlayerInitDataEx: the same after a leading size.
void InitEx(X86Regs& r) {  // (init data*, handle*)
    if (!r.rdi || !r.rsi) { r.rax = (uint64_t)(int64_t)kErrInvalidParams; return; }
    Wr<uint64_t>(r.rsi, Create(Rd<uint64_t>(r.rdi + 88), Rd<uint64_t>(r.rdi + 96)));
    r.rax = 0;
}

void Open(X86Regs& r, const std::string& path) {
    Player* p = Find(r.rdi);
    if (!p || path.empty()) { r.rax = (uint64_t)(int64_t)kErrInvalidParams; return; }
    if (p->frame) HleFree(p->frame);
    Player fresh;
    fresh.object = p->object;
    fresh.callback = p->callback;
    fresh.looping = p->looping;
    fresh.events = std::move(p->events);
    *p = std::move(fresh);
    p->path = path;
    p->source = true;
    p->events.push_back(kStateBuffering);
    Update(*p);
    r.rax = 0;
}
void AddSource(X86Regs& r) { Open(r, Str(r.rsi)); }  // (handle, path)
// (handle, uri type, SceAvPlayerSourceDetails*): the details start with the URI (name, length).
void AddSourceEx(X86Regs& r) { Open(r, r.rdx ? Str(Rd<uint64_t>(r.rdx)) : std::string()); }

void Start(X86Regs& r) {
    Player* p = Find(r.rdi);
    if (!p) { r.rax = (uint64_t)(int64_t)kErrInvalidParams; return; }
    p->playing = true;
    p->paused = p->ended = false;
    p->start_us = Now();
    p->last_frame = -1;
    p->events.push_back(kStatePlay);
    Update(*p);
    r.rax = 0;
}
void Stop(X86Regs& r) {
    Player* p = Find(r.rdi);
    if (!p) { r.rax = (uint64_t)(int64_t)kErrInvalidParams; return; }
    if (p->playing) p->events.push_back(kStateStop);
    p->playing = p->paused = false;
    p->ended = true;
    Update(*p);
    r.rax = 0;
}
void Pause(X86Regs& r) {
    Player* p = Find(r.rdi);
    if (!p) { r.rax = (uint64_t)(int64_t)kErrInvalidParams; return; }
    if (p->playing && !p->paused) {
        p->paused = true;
        p->paused_us = Now();
        p->events.push_back(kStatePause);
        Update(*p);
    }
    r.rax = 0;
}
void Resume(X86Regs& r) {
    Player* p = Find(r.rdi);
    if (!p) { r.rax = (uint64_t)(int64_t)kErrInvalidParams; return; }
    if (p->playing && p->paused) {
        p->start_us += Now() - p->paused_us;
        p->paused = false;
        p->events.push_back(kStatePlay);
        Update(*p);
    }
    r.rax = 0;
}
void JumpToTime(X86Regs& r) {  // (handle, ms)
    Player* p = Find(r.rdi);
    if (!p) { r.rax = (uint64_t)(int64_t)kErrInvalidParams; return; }
    const int64_t ms = std::min<int64_t>((int64_t)r.rsi, (int64_t)p->movie.duration_ms);
    p->start_us = (p->paused ? p->paused_us : Now()) - ms * 1000;
    p->last_frame = -1;
    r.rax = 0;
}
void SetLooping(X86Regs& r) {  // (handle, bool)
    Player* p = Find(r.rdi);
    if (!p) { r.rax = (uint64_t)(int64_t)kErrInvalidParams; return; }
    p->looping = (r.rsi & 0xff) != 0;
    r.rax = 0;
}
void IsActive(X86Regs& r) {
    Player* p = Find(r.rdi);
    r.rax = p && p->ready && !p->ended ? 1 : 0;
}
void CurrentTime(X86Regs& r) {
    Player* p = Find(r.rdi);
    r.rax = p ? PositionMs(*p) : 0;
}
void Close(X86Regs& r) {
    auto it = g_players.find(r.rdi);
    if (it != g_players.end()) {
        if (it->second.frame) HleFree(it->second.frame);
        g_players.erase(it);
    }
    r.rax = 0;
}

// The next frame once the clock has reached it: its index, or -1 when there is none yet.
int64_t NextFrame(Player& p) {
    if (!p.ready || !p.playing || p.paused) return -1;
    const int64_t index = (Now() - p.start_us) * p.movie.fps / 1000000;
    if (index == p.last_frame) return -1;
    p.last_frame = index;
    return index;
}
void VideoDetails(const Player& p, uint64_t d) {  // SceAvPlayerVideo: width, height, aspect ratio, language
    Wr<uint32_t>(d, p.movie.width);
    Wr<uint32_t>(d + 4, p.movie.height);
    Wr<float>(d + 8, 1.0f);
    Wr<uint32_t>(d + 12, 0);
}
void VideoDetailsEx(const Player& p, uint64_t d) {  // SceAvPlayerVideoEx (80 bytes)
    if (uint8_t* z = Ptr(d, 80)) memset(z, 0, 80);
    VideoDetails(p, d);
    Wr<uint32_t>(d + 16, p.movie.fps);
    Wr<uint32_t>(d + 24, p.pitch - p.movie.width);  // crop right
    Wr<uint32_t>(d + 32, p.rows - p.movie.height);  // crop bottom
    Wr<uint32_t>(d + 36, p.pitch);
    Wr<uint8_t>(d + 40, 8);  // luma and chroma bit depths
    Wr<uint8_t>(d + 41, 8);
}
// SceAvPlayerFrameInfo(Ex): data, timestamp (ms) at +16, details at +24.
void GetVideoData(X86Regs& r, bool ex) {
    Player* p = Find(r.rdi);
    const int64_t index = p && r.rsi ? NextFrame(*p) : -1;
    if (index < 0) { r.rax = 0; return; }
    Wr<uint64_t>(r.rsi, p->frame);
    Wr<uint64_t>(r.rsi + 16, (uint64_t)index * 1000 / p->movie.fps);
    if (ex) VideoDetailsEx(*p, r.rsi + 24);
    else VideoDetails(*p, r.rsi + 24);
    r.rax = 1;
}
void GetVideoDataPlain(X86Regs& r) { GetVideoData(r, false); }
void GetVideoDataEx(X86Regs& r) { GetVideoData(r, true); }
void GetAudioData(X86Regs& r) {  // no audio stream
    Find(r.rdi);
    r.rax = 0;
}

// One stream: the video.
void StreamCount(X86Regs& r) {
    Player* p = Find(r.rdi);
    r.rax = p && p->ready ? 1 : (uint64_t)(int64_t)kErrOperationFailed;
}
// SceAvPlayerStreamInfo(Ex): type, details at +8, then duration and start time.
void GetStreamInfo(X86Regs& r, bool ex) {  // (handle, index, info*)
    Player* p = Find(r.rdi);
    if (!p || !p->ready || r.rsi != 0 || !r.rdx) { r.rax = (uint64_t)(int64_t)kErrInvalidParams; return; }
    Wr<uint32_t>(r.rdx, 0);  // video
    if (ex) VideoDetailsEx(*p, r.rdx + 8);
    else VideoDetails(*p, r.rdx + 8);
    const uint64_t tail = r.rdx + (ex ? 88 : 24);
    Wr<uint64_t>(tail, p->movie.duration_ms);
    Wr<uint64_t>(tail + 8, 0);
    r.rax = 0;
}
void GetStreamInfoPlain(X86Regs& r) { GetStreamInfo(r, false); }
void GetStreamInfoEx(X86Regs& r) { GetStreamInfo(r, true); }
void StreamToggle(X86Regs& r) {  // EnableStream, DisableStream: (handle, index)
    Player* p = Find(r.rdi);
    r.rax = p && r.rsi == 0 ? 0 : (uint64_t)(int64_t)kErrInvalidParams;
}
void NoStream(X86Regs& r) { r.rax = (uint64_t)(int64_t)kErrInvalidParams; }
void Ok(X86Regs& r) { r.rax = 0; }

struct Entry { const char* nid; const char* name; HleFn fn; };
const Entry kEntries[] = {
    {"aS66RI0gGgo", "sceAvPlayerInit", Init},
    {"o9eWRkSL+M4", "sceAvPlayerInitEx", InitEx},
    {"HD1YKVU26-M", "sceAvPlayerPostInit", Ok},
    {"KMcEa+rHsIo", "sceAvPlayerAddSource", AddSource},
    {"x8uvuFOPZhU", "sceAvPlayerAddSourceEx", AddSourceEx},
    {"buMCiJftcfw", "sceAvPlayerChangeStream", NoStream},
    {"NkJwDzKmIlw", "sceAvPlayerClose", Close},
    {"wwM99gjFf1Y", "sceAvPlayerCurrentTime", CurrentTime},
    {"ODJK2sn9w4A", "sceAvPlayerEnableStream", StreamToggle},
    {"BOVKAzRmuTQ", "sceAvPlayerDisableStream", StreamToggle},
    {"Wnp1OVcrZgk", "sceAvPlayerGetAudioData", GetAudioData},
    {"d8FcbzfAdQw", "sceAvPlayerGetStreamInfo", GetStreamInfoPlain},
    {"ctTAcF5DiKQ", "sceAvPlayerGetStreamInfoEx", GetStreamInfoEx},
    {"o3+RWnHViSg", "sceAvPlayerGetVideoData", GetVideoDataPlain},
    {"JdksQu8pNdQ", "sceAvPlayerGetVideoDataEx", GetVideoDataEx},
    {"UbQoYawOsfY", "sceAvPlayerIsActive", IsActive},
    {"XC9wM+xULz8", "sceAvPlayerJumpToTime", JumpToTime},
    {"9y5v+fGN4Wk", "sceAvPlayerPause", Pause},
    {"w5moABNwnRY", "sceAvPlayerResume", Resume},
    {"k-q+xOxdc3E", "sceAvPlayerSetAvSyncMode", Ok},
    {"eBTreZ84JFY", "sceAvPlayerSetLogCallback", Ok},
    {"OVths0xGfho", "sceAvPlayerSetLooping", SetLooping},
    {"av8Z++94rs0", "sceAvPlayerSetTrickSpeed", Ok},
    {"ET4Gr-Uu07s", "sceAvPlayerStart", Start},
    {"ZC17w3vB5Lo", "sceAvPlayerStop", Stop},
    {"hdTyRzCXQeQ", "sceAvPlayerStreamCount", StreamCount},
};

const Entry* Lookup(const std::string& nid) {
    for (const auto& e : kEntries)
        if (nid == e.nid) return &e;
    return nullptr;
}

// The first calls, and then every call whose count is a power of two, are logged with their arguments.
template <size_t I> void Logged(X86Regs& r) {
    static uint64_t calls = 0;
    const Entry& e = kEntries[I];
    const uint64_t a0 = r.rdi, a1 = r.rsi, a2 = r.rdx;
    e.fn(r);
    if (++calls <= 12 || (calls & (calls - 1)) == 0)
        printf("[avplayer] %s(0x%llx, 0x%llx, 0x%llx) = 0x%llx (call %llu)\n", e.name, (unsigned long long)a0,
               (unsigned long long)a1, (unsigned long long)a2, (unsigned long long)r.rax, (unsigned long long)calls);
}
template <size_t... I> constexpr std::array<HleFn, sizeof...(I)> LoggedTable(std::index_sequence<I...>) {
    return {Logged<I>...};
}
const auto kLogged = LoggedTable(std::make_index_sequence<std::size(kEntries)>());

} // namespace

void AvPlayerSetup(const HleLayout& layout) { L = layout; }

HleFn AvPlayerFind(const std::string& nid) {
    const Entry* e = Lookup(nid);
    return e ? kLogged[e - kEntries] : nullptr;
}

const char* AvPlayerName(const std::string& nid) {
    const Entry* e = Lookup(nid);
    return e ? e->name : nullptr;
}

} // namespace KytyWeb
