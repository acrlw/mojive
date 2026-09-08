#include <mojive/log.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <spdlog/async_logger.h>
#include <spdlog/details/thread_pool.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <stdexcept>

namespace mojive {
namespace {
std::atomic<uint64_t> nextRuntime{1};
bool valid(LogLevel level) {
    return level >= LogLevel::Trace && level <= LogLevel::Off;
}
std::string bounded(std::string_view value, size_t limit) {
    size_t end = std::min(value.size(), limit);
    // Keep UTF-8 code points intact at the byte limit used by Python and ImGui.
    if (end < value.size())
        while (end && (static_cast<unsigned char>(value[end]) & 0xc0) == 0x80)
            --end;
    return std::string(value.substr(0, end));
}
} // namespace
class Log::Impl {
  public:
    const LogOptions options;
    const uint64_t runtime = nextRuntime.fetch_add(1, std::memory_order_relaxed);
    mutable std::mutex mutex;
    std::mutex closeMutex;
    std::deque<LogRecord> history;
    LogStats counters;
    std::shared_ptr<std::atomic<uint64_t>> outputErrors =
        std::make_shared<std::atomic<uint64_t>>(0);
    std::shared_ptr<spdlog::details::thread_pool> pool;
    std::shared_ptr<spdlog::async_logger> output;
    explicit Impl(const LogOptions &value) : options(value) {
        if (!options.capacity || options.byteCapacity < 256 || !options.messageBytes ||
            options.messageBytes > options.byteCapacity - 128 || !valid(options.level) ||
            !options.outputQueue || !options.fileBytes || !options.fileCount)
            throw std::invalid_argument("Invalid native log limits");
        std::vector<spdlog::sink_ptr> sinks;
        if (!options.file.empty())
            sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                options.file, options.fileBytes, options.fileCount));
        if (options.stderrOutput)
            sinks.push_back(std::make_shared<spdlog::sinks::stderr_color_sink_mt>());
        if (!sinks.empty()) {
            pool = std::make_shared<spdlog::details::thread_pool>(options.outputQueue, 1);
            output = std::make_shared<spdlog::async_logger>(
                "mojive", sinks.begin(), sinks.end(), pool,
                spdlog::async_overflow_policy::overrun_oldest);
            output->set_level(spdlog::level::trace);
            output->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
            output->set_error_handler([errors = outputErrors](const std::string &) {
                errors->fetch_add(1, std::memory_order_relaxed);
            });
        }
    }
};
Log::Log(const LogOptions &options) : mImpl(std::make_unique<Impl>(options)) {}
Log::~Log() {
    close();
}
uint64_t Log::runtimeId() const {
    return mImpl->runtime;
}
bool Log::publish(LogLevel level, std::string_view component, std::string_view message,
                  LogOrigin origin, int64_t timestampNs) {
    if (!valid(level) || (origin != LogOrigin::Native && origin != LogOrigin::Python))
        throw std::invalid_argument("Invalid native log severity or origin");
    auto &state = *mImpl;
    if (level == LogLevel::Off || level < state.options.level)
        return false;
    const auto now = std::chrono::system_clock::now();
    LogRecord record;
    record.runtimeId = state.runtime;
    record.timestampNs =
        timestampNs
            ? timestampNs
            : std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    record.threadId = spdlog::details::os::thread_id();
    record.level = level;
    record.origin = origin;
    record.component = bounded(component, 128);
    record.message = bounded(message, state.options.messageBytes);
    const auto bytes = record.component.size() + record.message.size();
    std::lock_guard guard(state.mutex);
    if (state.counters.closed)
        return false;
    record.sequence = ++state.counters.published;
    if (record.message.size() < message.size() || record.component.size() < component.size())
        ++state.counters.truncated;
    while (!state.history.empty() &&
           (state.history.size() >= state.options.capacity ||
            state.counters.retainedBytes + bytes > state.options.byteCapacity)) {
        state.counters.retainedBytes -=
            state.history.front().component.size() + state.history.front().message.size();
        state.history.pop_front();
        ++state.counters.overwritten;
    }
    state.counters.retainedBytes += bytes;
    state.history.push_back(std::move(record));
    if (state.output) {
        const auto &entry = state.history.back();
        auto time = std::chrono::system_clock::time_point(
            std::chrono::duration_cast<std::chrono::system_clock::duration>(
                std::chrono::nanoseconds(entry.timestampNs)));
        const auto text = spdlog::fmt_lib::format(
            "[runtime={} sequence={} component={} origin={}] {}", entry.runtimeId, entry.sequence,
            entry.component, origin == LogOrigin::Native ? "native" : "python", entry.message);
        state.output->log(time, spdlog::source_loc{}, static_cast<spdlog::level::level_enum>(level),
                          text);
    }
    return true;
}
LogBatch Log::read(uint64_t after, size_t limit) const {
    auto &state = *mImpl;
    std::lock_guard guard(state.mutex);
    if (after > state.counters.published || !limit)
        throw std::invalid_argument("Invalid native log cursor or batch limit");
    LogBatch batch;
    batch.next = after;
    if (state.history.empty())
        return batch;
    batch.missed =
        state.history.front().sequence > after + 1 ? state.history.front().sequence - after - 1 : 0;
    for (const auto &record : state.history) {
        if (record.sequence <= after)
            continue;
        batch.records.push_back(record);
        batch.next = record.sequence;
        if (batch.records.size() >= limit)
            break;
    }
    return batch;
}
LogStats Log::stats() const {
    auto &state = *mImpl;
    std::lock_guard guard(state.mutex);
    auto result = state.counters;
    result.retained = state.history.size();
    result.outputErrors = state.outputErrors->load(std::memory_order_relaxed);
    if (state.pool)
        result.outputDropped = state.pool->overrun_counter();
    return result;
}
void Log::close() {
    auto &state = *mImpl;
    // Serialize closers so every close returns only after native output has joined.
    std::lock_guard closeGuard(state.closeMutex);
    std::shared_ptr<spdlog::async_logger> output;
    std::shared_ptr<spdlog::details::thread_pool> pool;
    {
        std::lock_guard guard(state.mutex);
        if (state.counters.closed)
            return;
        state.counters.closed = true;
        output = std::move(state.output);
        pool = std::move(state.pool);
    }
    if (output)
        output->flush();
    const auto dropped = pool ? pool->overrun_counter() : 0;
    output.reset();
    pool.reset();
    std::lock_guard guard(state.mutex);
    state.counters.outputDropped = dropped;
}
} // namespace mojive
