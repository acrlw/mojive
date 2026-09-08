#include <mojive/log.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace mojive;
static void check(bool value, const char *reason) {
    if (!value)
        throw std::runtime_error(reason);
}
int main(int argc, char **argv) {
    try {
        LogOptions options;
        options.capacity = 8;
        options.byteCapacity = 256;
        options.messageBytes = 128;
        Log log(options), peer;
        check(log.runtimeId() != peer.runtimeId(), "Runtime identities collided");
        check(!log.publish(LogLevel::Debug, "scene", "filtered"), "Level filter ignored");
        std::vector<std::thread> producers;
        for (int i = 0; i < 4; ++i)
            producers.emplace_back([&] {
                for (int j = 0; j < 100; ++j)
                    log.publish(LogLevel::Info, "scene", "concurrent");
            });
        for (auto &thread : producers)
            thread.join();
        auto stats = log.stats();
        auto first = log.read(0, 3);
        auto independent = log.read(0, 3);
        check(stats.published == 400 && stats.retained == 8 && stats.overwritten == 392,
              "Concurrent history accounting failed");
        check(first.missed == 392 && first.records.size() == 3 && first.next == 395,
              "Subscription did not report the gap");
        check(independent.next == first.next, "One subscription consumed another");
        check(log.read(first.next).records.size() == 5, "Cursor continuation lost records");
        check(log.read(400).records.empty(), "Cursor replayed consumed records");
        auto message = std::string(127, 'x') + "中文";
        log.publish(LogLevel::Warning, "bridge", message, LogOrigin::Python, 123456789);
        auto record = log.read(400).records.at(0);
        check(record.message == std::string(127, 'x') && record.timestampNs == 123456789 &&
                  record.origin == LogOrigin::Python && record.runtimeId == log.runtimeId(),
              "UTF-8 boundary or source metadata changed");
        check(log.stats().truncated == 1 && log.stats().retainedBytes <= 256,
              "Log bounds exceeded");
        log.close();
        log.close();
        check(log.stats().closed && !log.publish(LogLevel::Error, "late", "closed"),
              "Closed log accepted new work");
        check(log.read(400).records.size() == 1, "Closing destroyed retained diagnostics");
        check(peer.stats().published == 0 && !peer.stats().closed, "Closing affected peer log");
        check(argc == 2, "Missing test output path");
        std::filesystem::remove(argv[1]);
        LogOptions fileOptions;
        fileOptions.file = argv[1];
        fileOptions.outputQueue = 128;
        Log file(fileOptions);
        for (int i = 0; i < 20; ++i)
            file.publish(LogLevel::Info, "file", "native output");
        // Native output drains without a subscription or a Python consumer.
        file.close();
        std::ifstream stream(argv[1]);
        std::string line;
        size_t lines = 0;
        while (std::getline(stream, line)) {
            check(line.find("native output") != std::string::npos, "File contents changed");
            ++lines;
        }
        check(lines == 20 && file.stats().outputDropped == 0 && file.stats().outputErrors == 0,
              "Close returned before output completed");
        Log closing;
        std::thread active([&] {
            for (int i = 0; i < 1000; ++i)
                closing.publish(LogLevel::Info, "shutdown", "concurrent close");
        });
        std::thread closer([&] { closing.close(); });
        closing.close();
        active.join();
        closer.join();
        check(closing.stats().closed, "Concurrent close failed");
        std::cout << "Native logging, subscriptions, bounds, and output shutdown passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
