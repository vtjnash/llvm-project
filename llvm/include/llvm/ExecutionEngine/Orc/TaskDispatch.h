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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cassert>
#include <string>
#include <type_traits>

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
  virtual void work_until(future_base &F) = 0;
};

/// Runs all tasks on the current thread.
class LLVM_ABI InPlaceTaskDispatcher : public TaskDispatcher {
public:
  void dispatch(std::unique_ptr<Task> T) override;
  void shutdown() override;
  void work_until(future_base &F) override;

private:
  thread_local static SmallVector<std::unique_ptr<Task>> TaskQueue;

#if LLVM_ENABLE_THREADS
  std::mutex DispatchMutex;
  std::condition_variable WorkFinishedCV;
  SmallVector<future_base *> WaitingFutures;
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
  void work_until(future_base &F) override;

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

/// Status for future/promise state
enum class FutureStatus : uint8_t { NotReady = 0, Ready = 1, NotValid = 2 };

/// Type-erased base class for futures
class future_base {
public:
  bool is_ready() const {
    return state_->status_.load(std::memory_order_acquire) !=
           FutureStatus::NotReady;
  }

  /// Check if the future is in a valid state (not moved-from and not consumed)
  bool valid() const {
    return state_ && state_->status_.load(std::memory_order_acquire) !=
                         FutureStatus::NotValid;
  }

  /// Wait for the future to be ready, helping with task dispatch
  void wait(TaskDispatcher &D) {
    // Keep helping with task dispatch until our future is ready
    if (!is_ready())
      D.work_until(*this);
    assert(is_ready());
  }

protected:
  struct state_base {
    std::atomic<FutureStatus> status_{FutureStatus::NotReady};
  };

  future_base(state_base *state) : state_(state) {}
  future_base() = default;
  ~future_base() {
    if (valid())
      report_fatal_error("get() must be called before future destruction");
    delete state_;
  }

  // Move constructor and assignment
  future_base(future_base &&other) noexcept : state_(other.state_) {
    other.state_ = nullptr;
  }
  future_base &operator=(future_base &&other) noexcept {
    if (this != &other) {
      delete state_;
      state_ = other.state_;
      other.state_ = nullptr;
    }
    return *this;
  }

  state_base *state_;
};

/// ORC-aware future class that can help with task dispatch while waiting

template <typename T> class future;
template <typename T> class promise;
template <typename T> class future : public future_base {
public:
  struct state : public future_base::state_base {
    template <typename U> struct value_storage {
      U value_;
    };

    template <> struct value_storage<void> {
      // No value_ member for void
    };

    value_storage<T> storage;
  };

  future() = delete;
  future(const future &) = delete;
  future &operator=(const future &) = delete;
  future(future &&) = default;
  future &operator=(future &&) = default;

  /// Get the value, helping with task dispatch while waiting.
  /// This will destroy the underlying value, so this must only be called once.
  T get(TaskDispatcher &D) {
    if (!valid())
      report_fatal_error("get() must only be called once");
    wait(D);
    auto old_status = state_->status_.exchange(FutureStatus::NotValid,
                                               std::memory_order_release);
    if (old_status != FutureStatus::Ready)
      report_fatal_error("get() must only be called once");
    return take_value();
  }

private:
  friend class promise<T>;

  template <typename U = T>
  typename std::enable_if<!std::is_void<U>::value, U>::type take_value() {
    return std::move(
        static_cast<typename future<T>::state *>(state_)->storage.value_);
  }

  template <typename U = T>
  typename std::enable_if<std::is_void<U>::value, U>::type take_value() {}

  explicit future(state *state) : future_base(state) {}
};

/// ORC-aware promise class that works with ORC future
template <typename T> class promise {
  friend class future<T>;

public:
  promise() : state_(new typename future<T>::state()), future_created_(false) {}

  ~promise() {
    // Delete state only if get_future() was never called
    if (!future_created_) {
      delete state_;
    }
  }

  promise(const promise &) = delete;
  promise &operator=(const promise &) = delete;

  promise(promise &&other) noexcept
      : state_(other.state_), future_created_(other.future_created_) {
    other.state_ = nullptr;
    other.future_created_ = false;
  }

  promise &operator=(promise &&other) noexcept {
    if (this != &other) {
      if (!future_created_) {
        delete state_;
      }
      state_ = other.state_;
      future_created_ = other.future_created_;
      other.state_ = nullptr;
      other.future_created_ = false;
    }
    return *this;
  }

  /// Get the associated future (must only be called once)
  future<T> get_future() {
    assert(!future_created_ && "get_future() can only be called once");
    future_created_ = true;
    return future<T>(state_);
  }

  /// Set the value (must only be called once)
  // In C++20, this std::conditional weirdness can probably be replaced just
  // with requires. It ensures that we don't try to define a method for `void&`,
  // but that if the user calls set_value(v) for any value v that they get a
  // member function error, instead of no member named 'value_'.
  template <typename U = T>
  void
  set_value(const typename std::conditional<std::is_void<T>::value,
                                            std::nullopt_t, T>::type &value) {
    assert(state_ && "Invalid promise state");
    state_->storage.value_ = value;
    state_->status_.store(FutureStatus::Ready, std::memory_order_release);
  }

  template <typename U = T>
  void set_value(typename std::conditional<std::is_void<T>::value,
                                           std::nullopt_t, T>::type &&value) {
    assert(state_ && "Invalid promise state");
    state_->storage.value_ = std::move(value);
    state_->status_.store(FutureStatus::Ready, std::memory_order_release);
  }

  template <typename U = T>
  typename std::enable_if<std::is_void<U>::value, void>::type
  set_value(const std::nullopt_t &value) = delete;

  template <typename U = T>
  typename std::enable_if<std::is_void<U>::value, void>::type
  set_value(std::nullopt_t &&value) = delete;

  template <typename U = T>
  typename std::enable_if<std::is_void<U>::value, void>::type set_value() {
    assert(state_ && "Invalid promise state");
    state_->status_.store(FutureStatus::Ready, std::memory_order_release);
  }

  /// Swap with another promise
  void swap(promise &other) noexcept {
    using std::swap;
    swap(state_, other.state_);
    swap(future_created_, other.future_created_);
  }

private:
  typename future<T>::state *state_;
  bool future_created_;
};

} // End namespace orc
} // End namespace llvm

namespace std {
template <typename T>
void swap(llvm::orc::promise<T> &lhs, llvm::orc::promise<T> &rhs) noexcept {
  lhs.swap(rhs);
}
} // End namespace std

#endif // LLVM_EXECUTIONENGINE_ORC_TASKDISPATCH_H
