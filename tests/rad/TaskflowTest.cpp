#include <rad/IO/Logging.h>

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/algorithm/sort.hpp>
#include <random>

#include <gtest/gtest.h>

// https://github.com/taskflow/taskflow?tab=readme-ov-file#start-your-first-taskflow-program
TEST(Taskflow, Simple)
{
    tf::Executor executor;
    tf::Taskflow taskflow;

    auto [A, B, C, D] = taskflow.emplace(   // crate four tasks
        []() { RAD_LOG(info, "TaskA"); },
        []() { RAD_LOG(info, "TaskB"); },
        []() { RAD_LOG(info, "TaskC"); },
        []() { RAD_LOG(info, "TaskD"); }
    );

    A.precede(B, C);    // A runs before B and C
    D.succeed(B, C);    // D runs after B and C

    executor.run(taskflow).wait();
}

// https://github.com/taskflow/taskflow?tab=readme-ov-file#create-a-subflow-graph
TEST(Taskflow, Subflow)
{
    tf::Executor executor;
    tf::Taskflow taskflow;

    tf::Task A = taskflow.emplace([]() { RAD_LOG(info, "TaskA"); }).name("A");
    tf::Task C = taskflow.emplace([]() { RAD_LOG(info, "TaskC"); }).name("C");
    tf::Task D = taskflow.emplace([]() { RAD_LOG(info, "TaskD"); }).name("D");

    tf::Task B = taskflow.emplace([](tf::Subflow& subflow) {
        tf::Task B1 = subflow.emplace([]() { RAD_LOG(info, "TaskB1"); }).name("B1");
        tf::Task B2 = subflow.emplace([]() { RAD_LOG(info, "TaskB2"); }).name("B2");
        tf::Task B3 = subflow.emplace([]() { RAD_LOG(info, "TaskB3"); }).name("B3");
        B3.succeed(B1, B2);  // B3 runs after B1 and B2
        }).name("B");

    A.precede(B, C);  // A runs before B and C
    D.succeed(B, C);  // D runs after  B and C

    executor.run(taskflow).wait();
}


// https://github.com/taskflow/taskflow?tab=readme-ov-file#integrate-control-flow-to-a-task-graph
TEST(Taskflow, ControlFlow)
{
    tf::Executor executor;
    tf::Taskflow taskflow;

    tf::Task init = taskflow.emplace([]() { RAD_LOG(info, "init"); }).name("init");
    tf::Task stop = taskflow.emplace([]() { RAD_LOG(info, "stop"); }).name("stop");

    int status = 0;
    // creates a condition task that returns a random binary
    tf::Task cond = taskflow.emplace(
        [&]() { RAD_LOG(info, "cond");  ++status; return (status > 3); }
    ).name("cond");

    init.precede(cond);

    // creates a feedback loop {0: cond, 1: stop}
    cond.precede(cond, stop);

    executor.run(taskflow).wait();
}

// https://github.com/taskflow/taskflow?tab=readme-ov-file#compose-task-graphs
TEST(Taskflow, Compose)
{
    tf::Executor executor;
    tf::Taskflow f1, f2;

    // create taskflow f1 of two tasks
    tf::Task f1A = f1.emplace([]() { std::cout << "Task f1A\n"; })
        .name("f1A");
    tf::Task f1B = f1.emplace([]() { std::cout << "Task f1B\n"; })
        .name("f1B");

    // create taskflow f2 with one module task composed of f1
    tf::Task f2A = f2.emplace([]() { std::cout << "Task f2A\n"; })
        .name("f2A");
    tf::Task f2B = f2.emplace([]() { std::cout << "Task f2B\n"; })
        .name("f2B");
    tf::Task f2C = f2.emplace([]() { std::cout << "Task f2C\n"; })
        .name("f2C");

    tf::Task f1_module_task = f2.composed_of(f1)
        .name("module");

    f1_module_task.succeed(f2A, f2B)
        .precede(f2C);

    executor.run(f2).wait();
}

// https://github.com/taskflow/taskflow?tab=readme-ov-file#launch-asynchronous-tasks
TEST(Taskflow, LaunchAsync)
{
    tf::Executor executor;

    // create asynchronous tasks directly from an executor
    std::future<int> future = executor.async([]() {
        RAD_LOG(info, "async task returns 1");
        return 1;
        });
    executor.silent_async([]() { RAD_LOG(info, "async task does not return"); });

    executor.wait_for_all();

    // create asynchronous tasks with dynamic dependencies
    tf::AsyncTask A = executor.silent_dependent_async([]() { RAD_LOG(info, "A"); });
    tf::AsyncTask B = executor.silent_dependent_async([]() { RAD_LOG(info, "B"); }, A);
    tf::AsyncTask C = executor.silent_dependent_async([]() { RAD_LOG(info, "C"); }, A);
    tf::AsyncTask D = executor.silent_dependent_async([]() { RAD_LOG(info, "D"); }, B, C);

    executor.wait_for_all();
}

// https://github.com/taskflow/taskflow?tab=readme-ov-file#leverage-standard-parallel-algorithms
TEST(Taskflow, Algorithms)
{
    tf::Executor executor;
    tf::Taskflow taskflow;

    std::vector<float> data(1024);
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    tf::Task initTask = taskflow.for_each(
        data.begin(), data.end(), [&](auto& i) { i = dist(gen); }
    );
    tf::Task sortTask = taskflow.sort(
        data.begin(), data.end()
    );

    initTask.precede(sortTask);

    executor.run(taskflow);
    executor.wait_for_all();

    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}
