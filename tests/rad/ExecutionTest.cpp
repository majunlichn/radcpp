#include <rad/Core/ThreadPool.h>
#include <rad/Core/Algorithm.h>
#include <rad/Core/Numeric.h>

#include <rad/IO/Logging.h>
#include <rad/System/Time.h>

#include <algorithm>
#include <numeric>
#include <random>

#include <gtest/gtest.h>

using namespace stdexec;

// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#example-async-inclusive-scan
auto async_inclusive_scan(
    auto& sched,
    std::span<const double> input,
    std::span<double> output,
    double init,
    std::size_t tile_count)
{
    const std::size_t tile_size = (input.size() + tile_count - 1) / tile_count; // divide and round up

    std::vector<double> partials(tile_count + 1);
    partials[0] = init;

    return just(std::move(partials))
        | continues_on(sched)
        | bulk(tile_count,
            [=](std::size_t i, std::vector<double>& partials) {
                auto start = i * tile_size;
                auto end = std::min(input.size(), (i + 1) * tile_size);
                partials[i + 1] = *--std::inclusive_scan(
                    std::begin(input) + start,
                    std::begin(input) + end,
                    std::begin(output) + start);
            })
        | then(
            [](std::vector<double>&& partials) {
                std::inclusive_scan(
                    std::begin(partials), std::end(partials),
                    std::begin(partials));
                return std::move(partials);
            })
        | bulk(tile_count,
            [=](std::size_t i, std::vector<double>& partials) {
                auto start = i * tile_size;
                auto end = std::min(input.size(), (i + 1) * tile_size);
                std::for_each(
                    std::begin(output) + start,
                    std::begin(output) + end,
                    [&](double& e) { e = partials[i] + e; });
            })
        | then(
            [=](std::vector<double>&& partials) {
                return output;
            });
}

TEST(Execution, async_inclusive_scan)
{
    exec::static_thread_pool pool;
    auto sched = pool.get_scheduler();

    constexpr size_t num_elements = 1024 * 1024;

    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> input(num_elements);
    std::vector<double> output(num_elements);
    std::generate(input.begin(), input.end(), [&]() { return dist(gen); });

    rad::Stopwatch stopwatch;

    const size_t tile_count = std::thread::hardware_concurrency();
    stopwatch.Start();
    sync_wait(async_inclusive_scan(sched, input, output, 0.0, tile_count));
    stopwatch.Stop();
    RAD_LOG(info, "async_inclusive_scan: {} ms", stopwatch.GetElapsedMilliseconds());

    // Verification:
    std::vector<double> ref(num_elements);
    stopwatch.Restart();
    std::inclusive_scan(std::begin(input), std::end(input), std::begin(ref));
    stopwatch.Stop();
    RAD_LOG(info, "inclusive_scan: {} ms", stopwatch.GetElapsedMilliseconds());
    double maxDiff = 0.0;
    constexpr double tolerance = 1e-6;
    for (size_t i = 0; i < num_elements; ++i)
    {
        double diff = std::abs(output[i] - ref[i]);
        if (maxDiff < diff)
        {
            maxDiff = diff;
        }
        if (std::abs(output[i] - ref[i]) > tolerance)
        {
            FAIL() << std::format("Verification failed at#{}: Result={}; Ref={}; Diff={}",
                i, output[i], ref[i], diff) << std::endl;
            break;
        }
    }
    if (maxDiff < tolerance)
    {
        RAD_LOG(info, "Verification passed with MaxDiff={}", maxDiff);
    }
}
