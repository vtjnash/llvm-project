//===------------ TaskDispatch.cpp - ORC task dispatch utils --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TaskDispatch.h"
#include "llvm/Config/llvm-config.h" // for LLVM_ENABLE_THREADS
#include "llvm/ExecutionEngine/Orc/Core.h"

namespace llvm {
namespace orc {

char Task::ID = 0;
char GenericNamedTask::ID = 0;
char IdleTask::ID = 0;

const char *GenericNamedTask::DefaultDescription = "Generic Task";

void Task::anchor() {}

void IdleTask::anchor() {}

TaskDispatcher::~TaskDispatcher() = default;

// InPlaceTaskDispatcher implementation
thread_local SmallVector<std::unique_ptr<Task>>
    InPlaceTaskDispatcher::TaskQueue;

void InPlaceTaskDispatcher::dispatch(std::unique_ptr<Task> T) {
  TaskQueue.push_back(std::move(T));
}

void InPlaceTaskDispatcher::shutdown() {
  if (!TaskQueue.empty())
    report_fatal_error(
        "InPlaceTaskDispatcher shutdown with tasks still in queue");
}

void InPlaceTaskDispatcher::work_until(future_base &F) {
  while (!F.is_ready()) {
    // First, process any tasks in our local queue
    // Process in LIFO order (most recently added first) to avoid deadlocks
    // when tasks have dependencies on each other
    while (!TaskQueue.empty()) {
      auto T = std::move(TaskQueue.back());
      TaskQueue.pop_back();
      T->run();

      // Notify any threads that might be waiting for work to complete
#if LLVM_ENABLE_THREADS
      {
        std::lock_guard<std::mutex> Lock(DispatchMutex);
        bool ShouldNotify = llvm::any_of(
            WaitingFutures, [](future_base *F) { return F->is_ready(); });
        if (ShouldNotify) {
          WaitingFutures.clear();
          WorkFinishedCV.notify_all();
        }
      }
#endif

      // Check if our future is now ready
      if (F.is_ready())
        return;
    }

    // If we get here, our queue is empty but the future isn't ready
    // We need to wait for other threads to finish work that might complete our
    // future
#if LLVM_ENABLE_THREADS
    {
      std::unique_lock<std::mutex> Lock(DispatchMutex);
      WaitingFutures.push_back(&F);
      WorkFinishedCV.wait(Lock, [&F]() { return F.is_ready(); });
    }
#else
    // Without threading, if our queue is empty and future isn't ready,
    // we can't make progress
    return;
#endif
  }
}

#if LLVM_ENABLE_THREADS
void DynamicThreadPoolTaskDispatcher::dispatch(std::unique_ptr<Task> T) {

  enum { Normal, Materialization, Idle } TaskKind;

  if (isa<MaterializationTask>(*T))
    TaskKind = Materialization;
  else if (isa<IdleTask>(*T))
    TaskKind = Idle;
  else
    TaskKind = Normal;

  {
    std::lock_guard<std::mutex> Lock(DispatchMutex);

    // Reject new tasks if they're dispatched after a call to shutdown.
    if (Shutdown)
      return;

    if (TaskKind == Materialization) {

      // If this is a materialization task and there are too many running
      // already then queue this one up and return early.
      if (!canRunMaterializationTaskNow())
        return MaterializationTaskQueue.push_back(std::move(T));

      // Otherwise record that we have a materialization task running.
      ++NumMaterializationThreads;
    } else if (TaskKind == Idle) {
      if (!canRunIdleTaskNow())
        return IdleTaskQueue.push_back(std::move(T));
    }

    ++Outstanding;
  }

  std::thread([this, T = std::move(T), TaskKind]() mutable {
    while (true) {

      // Run the task.
      T->run();

      // Reset the task to free any resources. We need this to happen *before*
      // we notify anyone (via Outstanding) that this thread is done to ensure
      // that we don't proceed with JIT shutdown while still holding resources.
      // (E.g. this was causing "Dangling SymbolStringPtr" assertions).
      T.reset();

      // Check the work queue state and either proceed with the next task or
      // end this thread.
      std::lock_guard<std::mutex> Lock(DispatchMutex);

      if (TaskKind == Materialization)
        --NumMaterializationThreads;
      --Outstanding;

      if (!MaterializationTaskQueue.empty() && canRunMaterializationTaskNow()) {
        // If there are any materialization tasks running then steal that work.
        T = std::move(MaterializationTaskQueue.front());
        MaterializationTaskQueue.pop_front();
        TaskKind = Materialization;
        ++NumMaterializationThreads;
        ++Outstanding;
      } else if (!IdleTaskQueue.empty() && canRunIdleTaskNow()) {
        T = std::move(IdleTaskQueue.front());
        IdleTaskQueue.pop_front();
        TaskKind = Idle;
        ++Outstanding;
      } else {
        if (Outstanding == 0)
          OutstandingCV.notify_all();
        return;
      }
    }
  }).detach();
}

void DynamicThreadPoolTaskDispatcher::shutdown() {
  std::unique_lock<std::mutex> Lock(DispatchMutex);
  Shutdown = true;
  OutstandingCV.wait(Lock, [this]() { return Outstanding == 0; });
}

bool DynamicThreadPoolTaskDispatcher::canRunMaterializationTaskNow() {
  return !MaxMaterializationThreads ||
         (NumMaterializationThreads < *MaxMaterializationThreads);
}

bool DynamicThreadPoolTaskDispatcher::canRunIdleTaskNow() {
  return !MaxMaterializationThreads ||
         (Outstanding < *MaxMaterializationThreads);
}

void DynamicThreadPoolTaskDispatcher::work_until(future_base &F) {
  // TODO: Implement efficient work_until for DynamicThreadPoolTaskDispatcher
  std::unique_lock<std::mutex> Lock(DispatchMutex);
  OutstandingCV.wait(Lock, [this]() { return Outstanding == 0; });
}

#endif

} // namespace orc
} // namespace llvm
