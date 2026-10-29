// libSceAmpr APR command buffers and the libkernel APR (asynchronous file read) entry points.
// Adapted from KytyPS5 (https://github.com/KytyPS5/KytyPS5, src/libs/libAmpr.cpp), GPL-2.0.
//
// Guest pointers are guest addresses. The command buffer header and record bytes the game can see are
// written as the reference writes them; the decoded commands live in a host shadow keyed by the command
// buffer's guest address. Only read-file records are ported (no events, address/counter writes, waits,
// gather/scatter or AMM). There is no worker thread: submissions run when they are queued and again
// whenever a waiter is polled, so a read of a streamed file that is still loading completes later.

#include "hle.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <limits>
#include <string>
#include <unordered_map>
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
uint64_t StackArg(const X86Regs& r, int i) { return Rd<uint64_t>(r.rsp + 8 + 8 * i); }
void Ret(X86Regs& r, int64_t v) { r.rax = (uint64_t)v; }

int g_logs = 0;
#define AMPR_LOG(...)                       \
    do {                                    \
        if (g_logs++ < 200) {               \
            printf("[ampr] " __VA_ARGS__);  \
        }                                   \
    } while (0)

// SCE kernel error codes are 0x80020000 | FreeBSD errno.
constexpr int OK = 0;
constexpr int KERNEL_ERROR_EPERM = (int)0x80020001;
constexpr int KERNEL_ERROR_ENOENT = (int)0x80020002;
constexpr int KERNEL_ERROR_ESRCH = (int)0x80020003;
constexpr int KERNEL_ERROR_EFAULT = (int)0x8002000e;
constexpr int KERNEL_ERROR_EBUSY = (int)0x80020010;
constexpr int KERNEL_ERROR_EINVAL = (int)0x80020016;
constexpr int KERNEL_ERROR_ENFILE = (int)0x80020017;
constexpr int KERNEL_ERROR_ENAMETOOLONG = (int)0x8002003f;

// The reference only rejects null and wrapping ranges; here the range must also be guest memory.
bool IsValidGuestRange(uint64_t addr, uint64_t size) { return size == 0 || Ptr(addr, size) != nullptr; }

int ReadGuestCString(uint64_t addr, char* out, size_t out_size) {
    if (addr == 0) return KERNEL_ERROR_EFAULT;
    for (size_t pos = 0; pos < out_size; pos++) {
        const uint8_t* p = Ptr(addr + pos, 1);
        if (!p) return KERNEL_ERROR_EFAULT;
        out[pos] = (char)*p;
        if (out[pos] == '\0') return OK;
    }
    return KERNEL_ERROR_ENAMETOOLONG;
}

// libkernel APR calls fail like syscalls: -1 with errno (__error()) set to the POSIX code.
int KernelSyscallResult(int result) {
    if (result >= OK) return result;
    static const HleFn error_fn = HleFind("9BcDykPmo1I");  // __error
    if (error_fn) {
        X86Regs r{};
        error_fn(r);
        Wr<int32_t>(r.rax, (int32_t)((uint32_t)result - 0x80020000u));
    }
    return -1;
}

// ---- shared APR state: resolved files and submissions ----

struct ResultBuffer {
    int32_t result = 0;
    uint32_t errorOffset = 0;
};
static_assert(sizeof(ResultBuffer) == 8, "ResultBuffer layout");

struct ResolvedPathInfo {
    int result = OK;
    uint32_t file_id = 0xffffffffu;
    uint64_t file_size = 0;
};

uint32_t g_next_submission_id = 1;
uint32_t g_next_file_id = 1;
std::unordered_map<uint32_t, bool> g_submissions;
std::unordered_map<uint32_t, std::string> g_files;  // file id -> guest path
std::unordered_map<uint32_t, uint64_t> g_file_sizes;
std::unordered_map<std::string, ResolvedPathInfo> g_resolved_paths;

// The reference reports directories with size 0x10000; HleFileSize doesn't tell them apart from files.
int ResolveOnePath(const char* guest_path, uint64_t id, uint64_t size) {
    if (guest_path[0] == '\0') return KERNEL_ERROR_EINVAL;

    const std::string path(guest_path);
    auto it = g_resolved_paths.find(path);
    if (it == g_resolved_paths.end()) {
        ResolvedPathInfo info{};
        if (!HleFileSize(path, &info.file_size)) {
            info.result = KERNEL_ERROR_ENOENT;
            AMPR_LOG("APR resolve missing path: %s\n", guest_path);
        } else if (g_next_file_id == 0xffffffffu) {
            info.result = KERNEL_ERROR_ENFILE;
        } else {
            info.file_id = g_next_file_id++;
            g_files.emplace(info.file_id, path);
            g_file_sizes.emplace(info.file_id, info.file_size);
        }
        it = g_resolved_paths.emplace(path, info).first;
    }

    const ResolvedPathInfo& info = it->second;
    if (info.result != OK) return info.result;
    if (id != 0) {
        if (!IsValidGuestRange(id, sizeof(uint32_t))) return KERNEL_ERROR_EFAULT;
        Wr<uint32_t>(id, info.file_id);
    }
    if (size != 0) {
        if (!IsValidGuestRange(size, sizeof(uint64_t))) return KERNEL_ERROR_EFAULT;
        Wr<uint64_t>(size, info.file_size);
    }
    return OK;
}

int ResolvePathsCommon(uint64_t path_list, uint32_t count, uint64_t ids, uint64_t sizes, uint64_t error_index,
                       uint64_t prefix, uint64_t results) {
    if (path_list == 0 || count == 0 || count > 1024 || (ids == 0 && sizes == 0 && results == 0))
        return KERNEL_ERROR_EINVAL;

    if (error_index != 0 && !IsValidGuestRange(error_index, sizeof(uint32_t))) return KERNEL_ERROR_EFAULT;
    if (ids != 0 && !IsValidGuestRange(ids, count * sizeof(uint32_t))) return KERNEL_ERROR_EFAULT;
    if (sizes != 0 && !IsValidGuestRange(sizes, count * sizeof(uint64_t))) return KERNEL_ERROR_EFAULT;
    if (results != 0 && !IsValidGuestRange(results, count * sizeof(int32_t))) return KERNEL_ERROR_EFAULT;
    if (!IsValidGuestRange(path_list, count * sizeof(uint64_t))) return KERNEL_ERROR_EFAULT;

    char prefix_buf[1024] = {};
    if (prefix != 0) {
        const int result = ReadGuestCString(prefix, prefix_buf, sizeof(prefix_buf));
        if (result != OK) return result;
    }
    const size_t prefix_size = strlen(prefix_buf);
    uint32_t success_count = 0;
    for (uint32_t i = 0; i < count; i++) {
        char resolved_path[1024] = {};
        // Concatenate the prefix literally, including an empty prefix.
        memcpy(resolved_path, prefix_buf, prefix_size);
        int result = ReadGuestCString(Rd<uint64_t>(path_list + 8ull * i), resolved_path + prefix_size,
                                      sizeof(resolved_path) - prefix_size);
        if (result == OK)
            result = ResolveOnePath(resolved_path, ids != 0 ? ids + 4ull * i : 0, sizes != 0 ? sizes + 8ull * i : 0);

        if (results != 0) Wr<int32_t>(results + 4ull * i, result);
        if (result == OK) {
            success_count++;
        } else {
            if (ids != 0) Wr<uint32_t>(ids + 4ull * i, 0xffffffffu);
            if (sizes != 0) Wr<uint64_t>(sizes + 8ull * i, 0);
            if (error_index != 0) Wr<uint32_t>(error_index, i);
            if (results == 0) return result;
        }
    }
    return results != 0 ? (int)success_count : OK;
}

uint32_t AllocateSubmissionId() {
    uint32_t id = g_next_submission_id++;
    if (id == 0) id = g_next_submission_id++;
    g_submissions.emplace(id, false);
    return id;
}

void SetSubmissionComplete(uint32_t submission_id) {
    if (submission_id != 0) g_submissions[submission_id] = true;
}

int WriteResult(uint64_t result, int32_t execution_result, uint32_t error_offset) {
    if (result == 0) return OK;
    if (!IsValidGuestRange(result, sizeof(ResultBuffer))) return KERNEL_ERROR_EFAULT;
    ResultBuffer res{};
    res.result = execution_result;
    res.errorOffset = error_offset;
    Wr<ResultBuffer>(result, res);
    return OK;
}

// ---- Ampr command buffers ----

constexpr uint64_t COMMAND_BUFFER_TYPE_OFFSET = 0x00;
constexpr uint64_t COMMAND_BUFFER_OFFSET_OFFSET = 0x04;
constexpr uint64_t COMMAND_BUFFER_NUM_OFFSET = 0x08;
constexpr uint64_t COMMAND_BUFFER_SIZE_OFFSET = 0x0c;
constexpr uint64_t COMMAND_BUFFER_DATA_OFFSET = 0x10;
constexpr uint64_t COMMAND_BUFFER_SIZE = 0x18;
constexpr uint64_t APR_COMMAND_BUFFER_MAP_OFFSET = 0x18;
constexpr uint64_t APR_COMMAND_BUFFER_SG_OFFSET = 0x20;
constexpr uint32_t COMMAND_BUFFER_SIZE_MAX = 64 * 1024 * 1024;
constexpr uint64_t READ_FILE_RECORD_SIZE = 0x14;
constexpr uint64_t READ_FILE_RECORD_SIZE_EXT = 0x18;
constexpr uint8_t READ_FILE_OPCODE = 0x17;
constexpr uint64_t APR_MAX_READ_LENGTH = 0x0000000100000000ull;
constexpr uint64_t APR_MAX_FILE_OFFSET = 0x0000010000000000ull;
constexpr uint64_t APR_MAX_APP_ADDRESS = 0x0000f00000000000ull;
constexpr uint32_t APR_TYPE_GATHER_SCATTER_VALID = 0x00010000;
constexpr uint32_t APR_PRIORITIES = 7;

struct ReadFileCommand {
    uint32_t file_id = 0;
    uint64_t destination = 0;
    uint64_t size = 0;
    uint64_t file_offset = 0;
};

struct Command {
    uint64_t record_offset;
    ReadFileCommand read;
};

struct CommandBufferState {
    uint64_t buffer = 0;
    uint64_t size = 0;
    uint64_t write_offset = 0;
    bool header_validated = false;
    bool buffer_validated = false;
    std::vector<Command> commands;
};

std::unordered_map<uint64_t, CommandBufferState> g_command_buffers;
std::unordered_map<uint64_t, uint64_t> g_command_buffer_aliases;  // buffer address -> command buffer

void RegisterCommandBufferAlias(uint64_t command_buffer, uint64_t buffer) {
    for (auto it = g_command_buffer_aliases.begin(); it != g_command_buffer_aliases.end();)
        it = it->second == command_buffer ? g_command_buffer_aliases.erase(it) : std::next(it);
    if (buffer != 0 && buffer != command_buffer) g_command_buffer_aliases[buffer] = command_buffer;
}

void EraseCommandBufferState(uint64_t command_buffer) {
    g_command_buffers.erase(command_buffer);
    for (auto it = g_command_buffer_aliases.begin(); it != g_command_buffer_aliases.end();)
        it = it->first == command_buffer || it->second == command_buffer ? g_command_buffer_aliases.erase(it)
                                                                         : std::next(it);
}

bool IsValidAprFileOffset(uint64_t file_offset) { return file_offset < APR_MAX_FILE_OFFSET; }

bool IsValidAprReadRange(uint64_t destination, uint64_t size) {
    return size != 0 && size <= APR_MAX_READ_LENGTH && destination <= APR_MAX_APP_ADDRESS &&
           APR_MAX_APP_ADDRESS - destination >= size;
}

bool ValidateCommandBufferHeader(uint64_t command_buffer, CommandBufferState* state = nullptr) {
    if (state != nullptr && state->header_validated) return true;
    if (!IsValidGuestRange(command_buffer, COMMAND_BUFFER_SIZE)) return false;
    if (state != nullptr) state->header_validated = true;
    return true;
}

bool ValidateCommandBufferBacking(CommandBufferState* state) {
    if (state->buffer == 0 || state->size == 0) return false;
    if (state->buffer_validated) return true;
    if (!IsValidGuestRange(state->buffer, state->size)) return false;
    state->buffer_validated = true;
    return true;
}

bool LoadCommandBufferStateFromGuest(uint64_t command_buffer, CommandBufferState* state) {
    if (!ValidateCommandBufferHeader(command_buffer)) return false;
    CommandBufferState loaded{};
    loaded.buffer = Rd<uint64_t>(command_buffer + COMMAND_BUFFER_DATA_OFFSET);
    loaded.size = Rd<uint32_t>(command_buffer + COMMAND_BUFFER_SIZE_OFFSET);
    loaded.write_offset = Rd<uint32_t>(command_buffer + COMMAND_BUFFER_OFFSET_OFFSET);
    loaded.header_validated = true;
    *state = loaded;
    return true;
}

CommandBufferState* GetOrCreateCommandBufferState(uint64_t command_buffer) {
    auto it = g_command_buffers.find(command_buffer);
    if (it == g_command_buffers.end()) {
        CommandBufferState state{};
        if (!LoadCommandBufferStateFromGuest(command_buffer, &state)) return nullptr;
        it = g_command_buffers.emplace(command_buffer, state).first;
    }
    return &it->second;
}

bool EnsureCommandBufferRecordSpace(uint64_t command_buffer, CommandBufferState* state, uint64_t record_size) {
    if (record_size == 0 || !ValidateCommandBufferHeader(command_buffer, state) || state->buffer == 0 ||
        record_size > state->size || state->write_offset > state->size - record_size)
        return false;
    return ValidateCommandBufferBacking(state);
}

bool CommitCommandBufferRecord(uint64_t command_buffer, CommandBufferState* state, uint64_t record_size) {
    state->write_offset += record_size;
    if (state->write_offset > std::numeric_limits<uint32_t>::max()) return false;
    Wr<uint32_t>(command_buffer + COMMAND_BUFFER_OFFSET_OFFSET, (uint32_t)state->write_offset);
    Wr<int32_t>(command_buffer + COMMAND_BUFFER_NUM_OFFSET, Rd<int32_t>(command_buffer + COMMAND_BUFFER_NUM_OFFSET) + 1);
    return true;
}

void UpdateCommandBufferTypeFlags(uint64_t command_buffer, uint32_t set_bits, uint32_t clear_bits) {
    uint32_t type = Rd<uint32_t>(command_buffer + COMMAND_BUFFER_TYPE_OFFSET);
    type |= set_bits;
    type &= ~clear_bits;
    Wr<uint32_t>(command_buffer + COMMAND_BUFFER_TYPE_OFFSET, type);
}

uint64_t AprReadFileRecordSize(uint64_t file_offset) {
    return (file_offset >> 32u) != 0 ? READ_FILE_RECORD_SIZE_EXT : READ_FILE_RECORD_SIZE;
}

// Submissions may name the command buffer or (via an alias) its backing buffer.
CommandBufferState* ResolveCommandBufferState(uint64_t command_buffer) {
    auto it = g_command_buffers.find(command_buffer);
    if (it != g_command_buffers.end() && !it->second.commands.empty()) return &it->second;
    const auto alias = g_command_buffer_aliases.find(command_buffer);
    if (alias != g_command_buffer_aliases.end()) {
        const auto alias_it = g_command_buffers.find(alias->second);
        if (alias_it != g_command_buffers.end()) return &alias_it->second;
    }
    return it != g_command_buffers.end() ? &it->second : nullptr;
}

bool WriteCommandBufferPointers(uint64_t command_buffer, uint64_t buffer, uint64_t size, uint64_t write_offset = 0) {
    if (write_offset > std::numeric_limits<uint32_t>::max() || size > std::numeric_limits<uint32_t>::max() ||
        !ValidateCommandBufferHeader(command_buffer))
        return false;

    Wr<uint64_t>(command_buffer + COMMAND_BUFFER_DATA_OFFSET, buffer);
    Wr<uint32_t>(command_buffer + COMMAND_BUFFER_SIZE_OFFSET, (uint32_t)size);
    Wr<uint32_t>(command_buffer + COMMAND_BUFFER_OFFSET_OFFSET, (uint32_t)write_offset);

    CommandBufferState& state = g_command_buffers[command_buffer];
    state.buffer = buffer;
    state.size = size;
    state.write_offset = write_offset;
    state.header_validated = true;
    state.buffer_validated = false;
    state.commands.clear();
    RegisterCommandBufferAlias(command_buffer, buffer);
    return true;
}

bool TryGetCommandBufferState(uint64_t command_buffer, CommandBufferState* out) {
    if (const CommandBufferState* state = ResolveCommandBufferState(command_buffer)) {
        *out = *state;
        return true;
    }
    CommandBufferState state{};
    if (!LoadCommandBufferStateFromGuest(command_buffer, &state)) return false;
    g_command_buffers[command_buffer] = state;
    *out = state;
    return true;
}

bool InitializeBaseCommandBuffer(uint64_t command_buffer) {
    uint8_t* header = Ptr(command_buffer, COMMAND_BUFFER_SIZE);
    if (!header) return false;
    memset(header, 0, COMMAND_BUFFER_SIZE);
    EraseCommandBufferState(command_buffer);
    return true;
}

bool WriteGuestZero64Validated(uint64_t addr) {
    if (!IsValidGuestRange(addr, sizeof(uint64_t))) return false;
    Wr<uint64_t>(addr, 0);
    return true;
}

bool InitializeAprReservedState(uint64_t command_buffer, uint64_t reserved_state0, uint64_t reserved_state1) {
    if (reserved_state0 == 0) reserved_state0 = command_buffer + APR_COMMAND_BUFFER_MAP_OFFSET;
    if (reserved_state1 == 0) reserved_state1 = command_buffer + APR_COMMAND_BUFFER_SG_OFFSET;

    if (reserved_state0 == reserved_state1) return WriteGuestZero64Validated(reserved_state0);

    constexpr uint64_t word_size = sizeof(uint64_t);
    constexpr uint64_t span_size = word_size * 2u;
    constexpr uint64_t max = std::numeric_limits<uint64_t>::max();
    const bool adjacent = (reserved_state0 <= max - word_size && reserved_state0 + word_size == reserved_state1) ||
                          (reserved_state1 <= max - word_size && reserved_state1 + word_size == reserved_state0);
    if (adjacent) {
        const uint64_t first = std::min(reserved_state0, reserved_state1);
        if (first > max - span_size || !IsValidGuestRange(first, span_size)) return false;
        Wr<uint64_t>(reserved_state0, 0);
        Wr<uint64_t>(reserved_state1, 0);
        return true;
    }
    return WriteGuestZero64Validated(reserved_state0) && WriteGuestZero64Validated(reserved_state1);
}

bool AppendReadFileRecord(uint64_t command_buffer, uint8_t opcode, uint32_t file_id, uint64_t destination,
                          uint64_t size, uint64_t file_offset, uint64_t record_size) {
    CommandBufferState* state = GetOrCreateCommandBufferState(command_buffer);
    if (!state || !EnsureCommandBufferRecordSpace(command_buffer, state, record_size)) return false;

    const uint64_t record_offset = state->write_offset;
    uint8_t* record = Ptr(state->buffer + record_offset, record_size);
    memset(record, 0, record_size);
    record[0] = opcode;
    state->commands.push_back({record_offset, ReadFileCommand{file_id, destination, size, file_offset}});
    if (!CommitCommandBufferRecord(command_buffer, state, record_size)) return false;
    UpdateCommandBufferTypeFlags(command_buffer, APR_TYPE_GATHER_SCATTER_VALID, 0);
    return true;
}

// ---- APR engine ----

struct Submission {
    std::vector<Command> commands;
    size_t cursor = 0;
    uint64_t done = 0;  // bytes of commands[cursor] already read
    uint64_t result = 0;
    uint32_t id = 0;
};

std::deque<Submission> g_queues[APR_PRIORITIES];  // priority 0 runs first

constexpr int kPending = 1;
// Streamed files arrive in 1 MB chunks (hle.cpp); a stalled read keeps what has arrived chunk by chunk.
constexpr uint64_t kStreamChunk = 1ull << 20;
constexpr uint64_t kReadWindow = 32ull << 20;

// Returns OK when the record is done, kPending while streamed data is still loading, or an error. Reading
// past the end of the file is not an error; the rest of the destination is left as it was.
int ExecuteReadFile(const ReadFileCommand& command, uint64_t* done) {
    const auto file = g_files.find(command.file_id);
    if (file == g_files.end()) {
        AMPR_LOG("APR submit failed for unknown file id: 0x%08x\n", command.file_id);
        return KERNEL_ERROR_ENOENT;
    }
    if (!IsValidGuestRange(command.destination, command.size)) return KERNEL_ERROR_EFAULT;

    while (*done < command.size) {
        const uint64_t pos = command.file_offset + *done, left = command.size - *done;
        uint64_t want = std::min(left, kReadWindow);
        bool pending = false;
        int64_t got = HleReadFileRange(file->second, pos, want, command.destination + *done, &pending);
        if (pending) {
            // The window's missing chunks are now requested; take what is contiguous from the front.
            want = std::min(left, kStreamChunk - pos % kStreamChunk);
            got = HleReadFileRange(file->second, pos, want, command.destination + *done, &pending);
            if (pending) return kPending;
        }
        if (got < 0) {
            AMPR_LOG("APR submit read failed: id=0x%08x, path=%s\n", command.file_id, file->second.c_str());
            return KERNEL_ERROR_ENOENT;
        }
        *done += (uint64_t)got;
        if ((uint64_t)got < want) break;
    }
    return OK;
}

// One command at a time from the front of the highest-priority non-empty queue, as the reference's
// worker does; a read that is still loading holds back everything queued.
void PumpSubmissions() {
    for (;;) {
        auto queue = std::find_if(std::begin(g_queues), std::end(g_queues), [](const auto& q) { return !q.empty(); });
        if (queue == std::end(g_queues)) return;

        Submission& submission = queue->front();
        int result = OK;
        uint32_t error_offset = 0;
        if (submission.cursor < submission.commands.size()) {
            const Command& command = submission.commands[submission.cursor];
            result = ExecuteReadFile(command.read, &submission.done);
            if (result == kPending) return;
            submission.cursor++;
            submission.done = 0;
            if (result != OK) error_offset = (uint32_t)command.record_offset;
        }
        if (result != OK || submission.cursor == submission.commands.size()) {
            WriteResult(submission.result, result, error_offset);
            SetSubmissionComplete(submission.id);
            queue->pop_front();
        }
    }
}

int QueueCommandBuffer(uint64_t command_buffer, uint32_t priority, uint64_t result, uint64_t out_submission_id) {
    if (command_buffer == 0 || priority >= APR_PRIORITIES) return KERNEL_ERROR_EINVAL;
    if ((result != 0 && !IsValidGuestRange(result, sizeof(ResultBuffer))) ||
        (out_submission_id != 0 && !IsValidGuestRange(out_submission_id, sizeof(uint32_t))))
        return KERNEL_ERROR_EFAULT;

    CommandBufferState state{};
    if (!TryGetCommandBufferState(command_buffer, &state)) return KERNEL_ERROR_EFAULT;
    if (state.write_offset == 0) return KERNEL_ERROR_EINVAL;

    const uint32_t id = out_submission_id != 0 ? AllocateSubmissionId() : 0;
    if (out_submission_id != 0) Wr<uint32_t>(out_submission_id, id);
    Submission submission;
    submission.commands = std::move(state.commands);
    submission.result = result;
    submission.id = id;
    g_queues[priority].push_back(std::move(submission));
    PumpSubmissions();
    return OK;
}

// ---- libSceAmpr ----

void CommandBufferConstructor(X86Regs& r) {
    if (r.rdi == 0) { Ret(r, OK); return; }
    Ret(r, InitializeBaseCommandBuffer(r.rdi) ? OK : KERNEL_ERROR_EFAULT);
}

void AprCommandBufferConstructor(X86Regs& r) {
    if (r.rdi == 0) { Ret(r, OK); return; }
    Ret(r, InitializeAprReservedState(r.rdi, r.rsi, r.rdx) ? OK : KERNEL_ERROR_EFAULT);
}

void AprCommandBufferDestructor(X86Regs& r) { Ret(r, OK); }

void CommandBufferDestructor(X86Regs& r) {
    if (r.rdi != 0) EraseCommandBufferState(r.rdi);
    Ret(r, OK);
}

void CommandBufferSetBuffer(X86Regs& r) {
    const uint64_t command_buffer = r.rdi, buffer = r.rsi;
    const uint32_t size = (uint32_t)r.rdx;
    if (command_buffer == 0) { Ret(r, KERNEL_ERROR_EINVAL); return; }
    if (!ValidateCommandBufferHeader(command_buffer)) { Ret(r, KERNEL_ERROR_EFAULT); return; }
    if (buffer == 0 || (buffer & 0x3u) != 0 || size == 0 || size > COMMAND_BUFFER_SIZE_MAX || (size & 0x3u) != 0) {
        Ret(r, KERNEL_ERROR_EINVAL);
        return;
    }
    if (Rd<uint64_t>(command_buffer + COMMAND_BUFFER_DATA_OFFSET) != 0) { Ret(r, KERNEL_ERROR_EBUSY); return; }
    if (!WriteCommandBufferPointers(command_buffer, buffer, size)) { Ret(r, KERNEL_ERROR_EFAULT); return; }
    Wr<int32_t>(command_buffer + COMMAND_BUFFER_NUM_OFFSET, 0);
    Ret(r, OK);
}

void CommandBufferReset(X86Regs& r) {
    const uint64_t command_buffer = r.rdi;
    if (command_buffer == 0) { Ret(r, KERNEL_ERROR_EPERM); return; }
    CommandBufferState state{};
    if (!TryGetCommandBufferState(command_buffer, &state)) { Ret(r, KERNEL_ERROR_EFAULT); return; }
    if (state.buffer == 0) { Ret(r, KERNEL_ERROR_EPERM); return; }
    if (!WriteCommandBufferPointers(command_buffer, state.buffer, state.size)) { Ret(r, KERNEL_ERROR_EFAULT); return; }
    Wr<int32_t>(command_buffer + COMMAND_BUFFER_NUM_OFFSET, 0);
    Ret(r, OK);
}

// (command_buffer, -, -, file_id, destination, size, file_offset), as the reference declares it.
void AprCommandBufferReadFile(X86Regs& r) {
    const uint64_t command_buffer = r.rdi, destination = r.r8, size = r.r9, file_offset = StackArg(r, 0);
    const uint32_t file_id = (uint32_t)r.rcx;
    if (command_buffer == 0 || !IsValidAprReadRange(destination, size) || !IsValidAprFileOffset(file_offset)) {
        Ret(r, KERNEL_ERROR_EINVAL);
        return;
    }
    const bool ok = AppendReadFileRecord(command_buffer, READ_FILE_OPCODE, file_id, destination, size, file_offset,
                                         AprReadFileRecordSize(file_offset));
    Ret(r, ok ? OK : KERNEL_ERROR_EFAULT);
}

// ---- libkernel APR ----

void AprResolveFilepathsToIds(X86Regs& r) {
    Ret(r, KernelSyscallResult(ResolvePathsCommon(r.rdi, (uint32_t)r.rsi, r.rdx, 0, r.rcx, 0, 0)));
}

void AprResolveFilepathsToIdsAndFileSizes(X86Regs& r) {
    Ret(r, KernelSyscallResult(ResolvePathsCommon(r.rdi, (uint32_t)r.rsi, r.rdx, r.rcx, r.r8, 0, 0)));
}

void AprGetFileSize(X86Regs& r) {
    const uint32_t file_id = (uint32_t)r.rdi;
    const uint64_t size = r.rsi;
    if (size == 0) { Ret(r, KernelSyscallResult(KERNEL_ERROR_EINVAL)); return; }
    if (!IsValidGuestRange(size, sizeof(uint64_t))) { Ret(r, KernelSyscallResult(KERNEL_ERROR_EFAULT)); return; }
    const auto it = g_file_sizes.find(file_id);
    if (it == g_file_sizes.end()) {
        AMPR_LOG("APR size failed for unknown file id: 0x%08x\n", file_id);
        Ret(r, KernelSyscallResult(KERNEL_ERROR_ENOENT));
        return;
    }
    Wr<uint64_t>(size, it->second);
    Ret(r, OK);
}

// FreeBSD struct stat as the PS5 libkernel lays it out (0x78 bytes). Files report mode 0777 like the
// reference; the access/modify times are the current time, not the file's.
void AprGetFileStat(X86Regs& r) {
    const uint32_t file_id = (uint32_t)r.rdi;
    const uint64_t st = r.rsi;
    if (st == 0) { Ret(r, KernelSyscallResult(KERNEL_ERROR_EINVAL)); return; }
    const auto file = g_files.find(file_id);
    if (file == g_files.end()) {
        AMPR_LOG("APR stat failed for unknown file id: 0x%08x\n", file_id);
        Ret(r, KernelSyscallResult(KERNEL_ERROR_ENOENT));
        return;
    }
    constexpr uint64_t kStatSize = 0x78;
    if (!IsValidGuestRange(st, kStatSize)) { Ret(r, KernelSyscallResult(KERNEL_ERROR_EFAULT)); return; }
    uint64_t file_size = 0;
    if (!HleFileSize(file->second, &file_size)) { Ret(r, KernelSyscallResult(KERNEL_ERROR_ENOENT)); return; }

    uint8_t b[kStatSize] = {};
    const uint16_t mode = 0100000u | 0777u;
    const int64_t size = (int64_t)file_size, blocks = (size + 511) / 512, sec = (int64_t)time(nullptr);
    const uint32_t blksize = 512;
    memcpy(b + 0x08, &mode, sizeof(mode));
    for (int off : {0x18, 0x28, 0x38, 0x68}) memcpy(b + off, &sec, sizeof(sec));
    memcpy(b + 0x48, &size, sizeof(size));
    memcpy(b + 0x50, &blocks, sizeof(blocks));
    memcpy(b + 0x58, &blksize, sizeof(blksize));
    memcpy(Ptr(st, kStatSize), b, kStatSize);
    Ret(r, OK);
}

void AprSubmitCommandBuffer(X86Regs& r) {
    Ret(r, KernelSyscallResult(QueueCommandBuffer(r.rdi, (uint32_t)r.rsi, 0, 0)));
}

void AprSubmitCommandBufferAndGetId(X86Regs& r) {
    if (r.rdx == 0) { Ret(r, KernelSyscallResult(KERNEL_ERROR_EINVAL)); return; }
    Ret(r, KernelSyscallResult(QueueCommandBuffer(r.rdi, (uint32_t)r.rsi, 0, r.rdx)));
}

void AprSubmitCommandBufferAndGetResult(X86Regs& r) {
    Ret(r, KernelSyscallResult(QueueCommandBuffer(r.rdi, (uint32_t)r.rsi, r.rdx, r.rcx)));
}

// Blocks until the submission has finished (its ResultBuffer is written at completion), then forgets it.
void AprWaitCommandBuffer(X86Regs& r) {
    const uint32_t id = (uint32_t)r.rdi;
    PumpSubmissions();
    const auto it = g_submissions.find(id);
    if (it == g_submissions.end()) { Ret(r, KernelSyscallResult(KERNEL_ERROR_ESRCH)); return; }
    if (!it->second) {
        HlePark("APR command buffer", [id] {
            PumpSubmissions();
            const auto s = g_submissions.find(id);
            return s == g_submissions.end() || s->second;
        });
        return;
    }
    g_submissions.erase(it);
    Ret(r, OK);
}

struct Entry {
    const char* nid;
    const char* name;
    HleFn fn;
};

const Entry kEntries[] = {
    {"8aI7R7WaOlc", "sceAmprCommandBufferConstructor", CommandBufferConstructor},
    {"a8uLzYY--tM", "sceAmprAprCommandBufferConstructor", AprCommandBufferConstructor},
    {"Qs1xtplKo0U", "sceAmprAprCommandBufferDestructor", AprCommandBufferDestructor},
    {"GuchCTefuZw", "sceAmprCommandBufferDestructor", CommandBufferDestructor},
    {"N-FSPA4S3nI", "sceAmprCommandBufferSetBuffer", CommandBufferSetBuffer},
    {"baQO9ez2gL4", "sceAmprCommandBufferReset", CommandBufferReset},
    {"mQ16-QdKv7k", "sceAmprAprCommandBufferReadFile", AprCommandBufferReadFile},
    {"WT-5NKy42fw", "sceKernelAprResolveFilepathsToIds", AprResolveFilepathsToIds},
    {"gEpBkcwxUjw", "sceKernelAprResolveFilepathsToIdsAndFileSizes", AprResolveFilepathsToIdsAndFileSizes},
    {"WvEu7yl3Ivg", "sceKernelAprGetFileSize", AprGetFileSize},
    {"ApkYaHb8Sek", "sceKernelAprGetFileStat", AprGetFileStat},
    {"eE4Szl8sil8", "sceKernelAprSubmitCommandBuffer", AprSubmitCommandBuffer},
    {"qvMUCyyaCSI", "sceKernelAprSubmitCommandBufferAndGetId", AprSubmitCommandBufferAndGetId},
    {"ASoW5WE-UPo", "sceKernelAprSubmitCommandBufferAndGetResult", AprSubmitCommandBufferAndGetResult},
    {"rqwFKI4PAiM", "sceKernelAprWaitCommandBuffer", AprWaitCommandBuffer},
};

const Entry* Lookup(const std::string& nid) {
    for (const auto& e : kEntries)
        if (nid == e.nid) return &e;
    return nullptr;
}

} // namespace

void AmprSetup(const HleLayout& layout) { L = layout; }

HleFn AmprFind(const std::string& nid) {
    const Entry* e = Lookup(nid);
    return e ? e->fn : nullptr;
}

const char* AmprName(const std::string& nid) {
    const Entry* e = Lookup(nid);
    return e ? e->name : nullptr;
}

} // namespace KytyWeb
