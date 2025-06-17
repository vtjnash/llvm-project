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

#include "llvm/ADT/BitmaskEnum.h"
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

/// Value storage template for future EBCO pattern
/// Moved outside class to avoid GCC nested template specialization issues
template <typename U> struct future_value_storage {
  // Union disables default construction/destruction semantics, allowing us to
  // use placement new/delete for precise control over value lifetime
  union {
    U value_;
  };

  future_value_storage() {}
  ~future_value_storage() {}
};

template <> struct future_value_storage<void> {
  // No value_ member for void
};

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

  /// Schedule the given task to run.
  virtual void dispatch(std::unique_ptr<Task> T) = 0;

  /// Called by ExecutionSession. Waits until all tasks have completed.
  virtual void shutdown() = 0;

  /// Work on dispatched tasks until the given future is ready.
  virtual void work_until(future_base &F) = 0;
};

/// Runs all tasks on the current thread, at the next work_until yield point.
class LLVM_ABI InPlaceTaskDispatcher : public TaskDispatcher {
public:
  void dispatch(std::unique_ptr<Task> T) override;
  void shutdown() override;
  void work_until(future_base &F) override;

private:
  /// C++ does not support non-static thread_local variables, so this needs to
  /// store both the task and the associated dispatcher queue so that shutdown
  /// can wait for the correct tasks to finish.
#if LLVM_ENABLE_THREADS
  thread_local static
#endif
      SmallVector<std::pair<std::unique_ptr<Task>, InPlaceTaskDispatcher *>>
          TaskQueue;

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
  SmallVector<future_base *> WaitingFutures;

  std::optional<size_t> MaxMaterializationThreads;
  size_t NumMaterializationThreads = 0;
  std::deque<std::unique_ptr<Task>> MaterializationTaskQueue;
  std::deque<std::unique_ptr<Task>> IdleTaskQueue;
};

#endif // LLVM_ENABLE_THREADS

/// @name ORC Promise/Future Classes
///
/// ORC-aware promise/future implementation that integrates with the
/// TaskDispatcher system to allow efficient cooperative multitasking while
/// waiting for results (with certain limitations on what can be awaited).
/// Together they provide building blocks for a full async/await-like runtime
/// for llvm that supports multiple threads.
///
/// Unlike std::promise/std::future alone, these classes can help dispatch other
/// tasks while waiting, preventing deadlocks and improving overall system
/// throughput. They have a similar API, though with some important differences
/// and some features simply not currently implemented.
///
/// @{

template <typename T> class promise;

/// Status for future/promise state
enum class FutureStatus : uint8_t { NotReady = 0, Ready = 1 };

/// Status for promise state tracking
enum class PromiseStatus : uint8_t {
  Initial = 0,
  FutureCreated = 1,
  ValueSet = 2,
  LLVM_MARK_AS_BITMASK_ENUM(ValueSet)
};

/// @}

/// Type-erased base class for futures, generally for scheduler use to avoid
/// needing virtual dispatches
class future_base {
public:
  /// Check if the future is now ready with a value (precondition: should be
  /// valid)
  bool ready() const {
    return state_->status_.load(std::memory_order_acquire) !=
           FutureStatus::NotReady;
  }

  /// Check if the future is in a valid state (not moved-from and not consumed)
  bool valid() const { return state_ != nullptr; }

  /// Wait for the future to be ready, helping with task dispatch
  void wait(TaskDispatcher &D) {
    // Keep helping with task dispatch until our future is ready
    if (!ready()) {
      D.work_until(*this);
      if (state_->status_.load(std::memory_order_relaxed) !=
          FutureStatus::Ready)
        report_fatal_error(
            "work_until() returned without this future being ready");
    }
  }

protected:
  struct state_base {
    std::atomic<FutureStatus> status_{FutureStatus::NotReady};
  };

  future_base(state_base *state) : state_(state) {}
  future_base() = default;

  /// Only allow deleting the future once it is invalid
  ~future_base() {
    if (valid())
      report_fatal_error("get() must be called before future destruction (ensuring promise::set_value memory is valid)");
    // state_ is already nullptr if get() was called, otherwise we have an error
    // above
  }

  // Move constructor and assignment
  future_base(future_base &&other) noexcept : state_(other.state_) {
    other.state_ = nullptr;
  }

  future_base &operator=(future_base &&other) noexcept {
    if (this != &other) {
      this->~future_base();
      state_ = other.state_;
      other.state_ = nullptr;
    }
    return *this;
  }

  state_base *state_;
};

/// TaskDispatcher-aware future class for cooperative await.
///
/// @tparam T The type of value this future will provide. Use void for futures
/// that
///           signal completion without providing a value.
///
/// This future implementation is similar to `std::future`, so most code can
/// transition to it easily. However, it differs from `std::future` in a few
/// key ways to be aware of:
/// - No exception support (or the overhead for it).
/// - Waiting operations (get(&D), wait(&D)) help dispatch other tasks while
///   blocked, requiring an additional argument of which TaskDispatcher object
///   of where all associated work will be scheduled.
/// - While `wait` may be called multiple times and on multiple threads, all of
///   them must have returned before calling `get` on exactly one thread.
/// - Must call get() exactly once before destruction (enforced with
///   `report_fatal_error`). Internal state is freed when `get` returns.
///
/// Other notable features, in common with `std::future`:
/// - Supports both value types and void specialization through the same
/// interface.
/// - Thread-safe through atomic operations.
/// - Provides acquire-release ordering with `std::promise::set_value()`.
/// - Concurrent access to any method (including to `ready`) on multiple threads
///   is not allowed.
/// - Holding any locks while calling `get()` is likely to lead to deadlock.

template <typename T> class future : public future_base {
public:
  // Template the state struct with EBCO so that future<void> has no wasted
  // overhead for the value. The declaration of future_value_storage is above
  // since GCC doesn't implement nested specializations properly.
  struct state : public future_base::state_base, public future_value_storage<T> {};

  future() = delete;
  future(const future &) = delete;
  future &operator=(const future &) = delete;
  future(future &&) = default;
  future &operator=(future &&) = default;

  /// Get the value, helping with task dispatch while waiting.
  /// This will destroy the underlying value, so this must be called exactly
  /// once.
  T get(TaskDispatcher &D) {
    if (!valid())
      report_fatal_error("get() must only be called once");
    wait(D);
    return take_value();
  }

private:
  friend class promise<T>;

  template <typename U = T>
  typename std::enable_if<!std::is_void<U>::value, U>::type take_value() {
    auto state = static_cast<typename future<T>::state *>(state_);
    T result = std::move(state->value_);
    state->value_.~T();
    delete state_;
    state_ = nullptr;
    return result;
  }

  template <typename U = T>
  typename std::enable_if<std::is_void<U>::value, U>::type take_value() {
    delete state_;
    state_ = nullptr;
  }

  explicit future(state *state) : future_base(state) {}
};

/// TaskDispatcher-aware promise class that provides values to associated
/// futures.
///
/// @tparam T The type of value this promise will provide. Use void for promises
/// that
///           signal completion without providing a value.
///
/// This promise implementation provides the value-setting side of the
/// promise/future pair and integrates with the ORC TaskDispatcher system. Key
/// characteristics:
/// - Must call get_future() no more than once to create the associated future.
/// - Must call set_value() exactly once to provide the result.
/// - Automatically cleans up state if get_future() is never called.
/// - Thread-safe from set_value to get.
/// - Move-only semantics to prevent accidental copying.
///
/// The `promise` can usually be passed to another thread in one of two ways:
/// - With move semantics:
///     * `[P = std::move(P)] () mutable { P.set_value(); }`
///     * Advantages: clearer where `P` is owned, automatic deadlock detection
///     on destruction,
///       easier memory management if the future is returned from the function.
///     * Disadvantages: more verbose syntax, requires unique_function (not
///     compatible with std::function).
/// - By reference:
///     * `[&P] () { P.set_value(); }`
///     * Advantages: simpler memory management if the future is consumed in the
///     same function.
///     * Disadvantages: more difficult memory management if the future is
///     returned from the function, no deadlock detection.
///
/// @par Error Handling:
/// The promise/future system uses report_fatal_error() for misuse:
/// - Calling get_future() more than once.
/// - Calling set_value() more than once.
/// - Destroying a future without calling get().
/// - Calling get() more than once on a future.
///
/// @par Thread Safety:
/// - Each promise/future must only be accessed by one thread, as concurrent
///   calls to the API functions may result in crashes.
/// - Multiple threads can safely access different promise/future pairs.
/// - set_value() and get() operations are atomic and thread-safe.
/// - Move operations should only be performed by a single thread.
template <typename T> class promise {
  friend class future<T>;

public:
  promise()
      : state_(new typename future<T>::state()),
        status_(PromiseStatus::Initial) {}

  ~promise() {
    // Assert proper promise lifecycle: ensure set_value was called if
    // get_future() was called. This can catch deadlocks where get_future() is
    // called but set_value() is never called, though only if the promise is
    // moved from instead of borrowed from the frame with the future.
    assert(
        (bool)(status_ & PromiseStatus::ValueSet) ==
            (bool)(status_ & PromiseStatus::FutureCreated) &&
        "Promise destroyed with mismatched value_set and get_future call state (possible deadlock)");
    // Delete state only if get_future() was never called
    if (!(status_ & PromiseStatus::FutureCreated)) {
      delete state_;
    } else {
      assert((bool)(status_ & PromiseStatus::ValueSet));
    }
  }

  promise(const promise &) = delete;
  promise &operator=(const promise &) = delete;

  promise(promise &&other) noexcept
      : state_(other.state_), status_(other.status_) {
    other.state_ = nullptr;
    other.status_ = PromiseStatus::Initial;
  }

  promise &operator=(promise &&other) noexcept {
    if (this != &other) {
      this->~promise();
      state_ = other.state_;
      status_ = other.status_;
      other.state_ = nullptr;
      other.status_ = PromiseStatus::Initial;
    }
    return *this;
  }

  /// Get the associated future (must only be called once)
  future<T> get_future() {
    assert(!(status_ & PromiseStatus::FutureCreated) &&
           "get_future() can only be called once");
    status_ |= PromiseStatus::FutureCreated;
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
    assert(!(status_ & PromiseStatus::ValueSet) &&
           "set_value() can only be called once");
    new (&state_->value_) T(value);
    status_ |= PromiseStatus::ValueSet;
    state_->status_.store(FutureStatus::Ready, std::memory_order_release);
  }

  template <typename U = T>
  void set_value(typename std::conditional<std::is_void<T>::value,
                                           std::nullopt_t, T>::type &&value) {
    assert(state_ && "Invalid promise state");
    assert(!(status_ & PromiseStatus::ValueSet) &&
           "set_value() can only be called once");
    new (&state_->value_) T(std::move(value));
    status_ |= PromiseStatus::ValueSet;
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
    assert(!(status_ & PromiseStatus::ValueSet) &&
           "set_value() can only be called once");
    status_ |= PromiseStatus::ValueSet;
    state_->status_.store(FutureStatus::Ready, std::memory_order_release);
  }

  /// Swap with another promise
  void swap(promise &other) noexcept {
    using std::swap;
    swap(state_, other.state_);
    swap(status_, other.status_);
  }

private:
  typename future<T>::state *state_;
  PromiseStatus status_;
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
