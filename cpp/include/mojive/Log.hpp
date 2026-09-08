#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mojive {
enum class LogLevel { Trace, Debug, Info, Warning, Error, Critical, Off };
enum class LogOrigin { Native, Python };
struct LogOptions {
    size_t capacity = 1024, byteCapacity = 1024 * 1024, messageBytes = 4096;
    LogLevel level = LogLevel::Info;
    std::string file;
    size_t fileBytes = 10 * 1024 * 1024, fileCount = 3, outputQueue = 1024;
    bool stderrOutput = false;
};
struct LogRecord {
    uint64_t sequence = 0, runtimeId = 0, threadId = 0;
    int64_t timestampNs = 0;
    LogLevel level = LogLevel::Info;
    LogOrigin origin = LogOrigin::Native;
    std::string component, message;
};
struct LogBatch {
    std::vector<LogRecord> records;
    uint64_t next = 0, missed = 0;
};
struct LogStats {
    uint64_t published = 0, overwritten = 0, truncated = 0, outputDropped = 0, outputErrors = 0;
    size_t retained = 0, retainedBytes = 0;
    bool closed = false;
};
// Independent subscriptions use sequence cursors. No Python callbacks or references
// enter native logging; close drains accepted file output after stopping producers.
class Log final {
  public:
    explicit Log(const LogOptions &options = {});
    ~Log();
    Log(const Log &) = delete;
    Log &operator=(const Log &) = delete;
    bool publish(LogLevel level, std::string_view component, std::string_view message,
                 LogOrigin origin = LogOrigin::Native, int64_t timestampNs = 0);
    LogBatch read(uint64_t after = 0, size_t limit = 256) const;
    LogStats stats() const;
    uint64_t runtimeId() const;
    void close();

  private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};
} // namespace mojive
