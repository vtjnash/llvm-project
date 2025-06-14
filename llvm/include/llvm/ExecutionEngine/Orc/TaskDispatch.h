//===--------- TaskDispatch.h - ORC task dispatch utils ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Task and TaskDispatch classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_TASKDISPATCH_H
#define LLVM_EXECUTIONENGINE_ORC_TASKDISPATCH_H

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cassert>
#include <string>

#if LLVM_ENABLE_THREADS
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#endif

namespace llvm {
namespace orc {

/// Forward declarations
class future_base;
template <typename T> class future;
template <typename T> class promise;
class TaskDispatcher;

/// Represents an abstract task for ORC to run.
class LLVM_ABI Task : public RTTIExtends<Task, RTTIRoot> {
public:
  static char ID;

  virtual ~Task() = default;

  /// Description of the task to be performed. Used for logging.
  virtual void printDescription(raw_ostream &OS) = 0;

  /// Run the task.
  virtual void run() = 0;

private:
  void anchor() override;
};

/// Base class for generic tasks.
class GenericNamedTask : public RTTIExtends<GenericNamedTask, Task> {
public:
  LLVM_ABI static char ID;
  LLVM_ABI static const char *DefaultDescription;
};

/// Generic task implementation.
template <typename FnT> class GenericNamedTaskImpl : public GenericNamedTask {
public:
  GenericNamedTaskImpl(FnT &&Fn, std::string DescBuffer)
      : Fn(std::forward<FnT>(Fn)), Desc(DescBuffer.c_str()),
        DescBuffer(std::move(DescBuffer)) {}
  GenericNamedTaskImpl(FnT &&Fn, const char *Desc)
      : Fn(std::forward<FnT>(Fn)), Desc(Desc) {
    assert(Desc && "Description cannot be null");
  }
  void printDescription(raw_ostream &OS) override { OS << Desc; }
  void run() override { Fn(); }

private:
  FnT Fn;
  const char *Desc;
  std::string DescBuffer;
};

/// Create a generic named task from a std::string description.
template <typename FnT>
std::unique_ptr<GenericNamedTask> makeGenericNamedTask(FnT &&Fn,
                                                       std::string Desc) {
  return std::make_unique<GenericNamedTaskImpl<FnT>>(std::forward<FnT>(Fn),
                                                     std::move(Desc));
}

/// Create a generic named task from a const char * description.
template <typename FnT>
std::unique_ptr<GenericNamedTask>
makeGenericNamedTask(FnT &&Fn, const char *Desc = nullptr) {
  if (!Desc)
    Desc = GenericNamedTask::DefaultDescription;
  return std::make_unique<GenericNamedTaskImpl<FnT>>(std::forward<FnT>(Fn),
                                                     Desc);
}

/// IdleTask can be used as the basis for low-priority tasks, e.g. speculative
/// lookup.
class LLVM_ABI IdleTask : public RTTIExtends<IdleTask, Task> {
public:
  static char ID;

private:
  void anchor() override;
};

/// Abstract base for classes that dispatch ORC Tasks.
class LLVM_ABI TaskDispatcher {
public:
  virtual ~TaskDispatcher();

  /// Run the given task.
  virtual void dispatch(std::unique_ptr<Task> T) = 0;

  /// Called by ExecutionSession. Waits until all tasks have completed.
  virtual void shutdown() = 0;

  /// Work on dispatched tasks until the given future is ready.
  virtual void work_until(future_base& F) = 0;
};

/// Runs all tasks on the current thread.
class LLVM_ABI InPlaceTaskDispatcher : public TaskDispatcher {
public:
  void dispatch(std::unique_ptr<Task> T) override;
  void shutdown() override;
  void work_until(future_base& F) override;

private:
  thread_local static SmallVector<std::unique_ptr<Task>> TaskQueue;
  
#if LLVM_ENABLE_THREADS
  std::mutex DispatchMutex;
  std::condition_variable WorkFinishedCV;
  SmallVector<future_base*> WaitingFutures;
#endif
};

#if LLVM_ENABLE_THREADS

class LLVM_ABI DynamicThreadPoolTaskDispatcher : public TaskDispatcher {
public:
  DynamicThreadPoolTaskDispatcher(
      std::optional<size_t> MaxMaterializationThreads)
      : MaxMaterializationThreads(MaxMaterializationThreads) {}

  void dispatch(std::unique_ptr<Task> T) override;
  void shutdown() override;
  void work_until(future_base& F) override;
private:
  bool canRunMaterializationTaskNow();
  bool canRunIdleTaskNow();

  std::mutex DispatchMutex;
  bool Shutdown = false;
  size_t Outstanding = 0;
  std::condition_variable OutstandingCV;

  std::optional<size_t> MaxMaterializationThreads;
  size_t NumMaterializationThreads = 0;
  std::deque<std::unique_ptr<Task>> MaterializationTaskQueue;
  std::deque<std::unique_ptr<Task>> IdleTaskQueue;
};

#endif // LLVM_ENABLE_THREADS

/// Type-erased base class for futures
class future_base {
public:
  virtual ~future_base() = default;

  bool is_ready() const {
    return state_->status_.load(std::memory_order_acquire) != 0;
  }

  /// Wait for the future to be ready, helping with task dispatch
  void wait(TaskDispatcher& D) {
    // Keep helping with task dispatch until our future is ready
    if (!is_ready())
      D.work_until(*this);
    assert(is_ready());
  }

protected:
  struct state_base {
    std::atomic<uint8_t> status_{0};
  };

  future_base(std::shared_ptr<state_base> state) : state_(std::move(state)) {}
  future_base() = default;

  std::shared_ptr<state_base> state_;
};

/// ORC-aware future class that can help with task dispatch while waiting
template <typename T>
class future : public future_base {
public:
  struct state : public future_base::state_base {
    T value_;
  };

  future() = default;
  future(const future&) = delete;
  future& operator=(const future&) = delete;
  future(future&&) = default;
  future& operator=(future&&) = default;


  /// Get the value, helping with task dispatch while waiting.
  /// This will destroy the underlying value, so this must only be called once.
  T get(TaskDispatcher& D) {
    wait(D);
    // optionally: state_->ready_.swap(0, std::memory_order_acquire);
    return std::move(static_cast<typename future<T>::state*>(state_.get())->value_);
  }

  /// Cast a future to a different type using static_pointer_cast
  template <typename U>
  static future<U> static_pointer_cast(future<T>&& f) {
    std::shared_ptr<typename future<U>::state> casted_state = std::static_pointer_cast<typename future<U>::state>(std::move(f.state_));
    return future<U>(casted_state);
  }

private:
  friend class promise<T>;
  
  explicit future(std::shared_ptr<state> state) : future_base(state) {}
};

/// ORC-aware promise class that works with ORC future
template <typename T>
class promise {
  friend class future<T>;
  
public:
  promise() : state_(std::make_shared<typename future<T>::state>()) {}
  promise(const promise&) = delete;
  promise& operator=(const promise&) = delete;
  promise(promise&&) = default;
  promise& operator=(promise&&) = default;

  /// Get the associated future
  future<T> get_future() {
    return future<T>(state_);
  }

  /// Set the value
  void set_value(const T& value) {
    state_->value_ = value;
    state_->status_.store(1, std::memory_order_release);
  }
  
  void set_value(T&& value) {
    state_->value_ = std::move(value);
    state_->status_.store(1, std::memory_order_release);
  }

private:
  std::shared_ptr<typename future<T>::state> state_;
};

/// Specialization of future<void>
template <> class future<void> : public future_base {
public:
  using state = future_base::state_base;

  future() = default;
  future(const future &) = delete;
  future &operator=(const future &) = delete;
  future(future &&) = default;
  future &operator=(future &&) = default;

  /// Get the value (void), helping with task dispatch while waiting.
  void get(TaskDispatcher &D) { wait(D); }

private:
  friend class promise<void>;

  explicit future(std::shared_ptr<state> state) : future_base(state) {}
};

/// Specialization of promise<void>
template <> class promise<void> {
  friend class future<void>;

public:
  promise() : state_(std::make_shared<future<void>::state>()) {}
  promise(const promise &) = delete;
  promise &operator=(const promise &) = delete;
  promise(promise &&) = default;
  promise &operator=(promise &&) = default;

  /// Get the associated future
  future<void> get_future() { return future<void>(state_); }

  /// Set the value (void)
  void set_value() { state_->status_.store(1, std::memory_order_release); }

private:
  std::shared_ptr<future<void>::state> state_;
};

} // End namespace orc
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_TASKDISPATCH_H
