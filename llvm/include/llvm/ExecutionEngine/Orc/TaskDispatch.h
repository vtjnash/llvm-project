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
#include "llvm/ADT/FunctionExtras.h"
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

  /// Schedule the given task to run.
  virtual void dispatch(std::unique_ptr<Task> T) = 0;

  /// The difference between `dispatch` and `dispatch_elsewhere` is whether
  /// other threads must be allowed to steal this work:
  ///   for `dispatch` the current thread must eventually call `future::get` to
  ///   observe the result (to make progress, work stealing not required). for
  ///   `dispatch_elsewhere` any thread may eventually observe the result (to
  ///   make progress, work stealing may be required).
  /// This distinction does not matter for most schedulers (e.g. except for
  /// InPlaceTaskDispatcher), so this just directly forwards to the `dispatch`
  /// method by default.
  virtual void dispatch_elsewhere(std::unique_ptr<Task> T) {
    dispatch(std::move(T));
  }

  /// Called by ExecutionSession. Waits until all tasks have completed.
  virtual void shutdown() = 0;

protected:
  friend class future_base;
  template <typename T> friend class promise;

  /// Work on dispatched tasks until the given future is ready.
  virtual void work_until(future_base &F) = 0;

  /// Notify all task dispatchers that a future with have_waiter became ready
  LLVM_ABI static void notifyWaiters();

#if LLVM_ENABLE_THREADS
  /// Shared synchronization primitives for all dispatchers
  static std::mutex DispatchMutex;
  /// FutureReadyCV could be a map from Future to condition_variable for more
  /// targeted notifications, but performance measurements are needed to
  /// determine if the added complexity is worthwhile vs. the current broadcast
  /// approach.
  static std::condition_variable FutureReadyCV;
#endif
};

/// Runs all tasks on the current thread, at the next work_until yield point.
class LLVM_ABI InPlaceTaskDispatcher : public TaskDispatcher {
public:
  void dispatch(std::unique_ptr<Task> T) override;
  void dispatch_elsewhere(std::unique_ptr<Task> T) override;
  void shutdown() override;

private:
  void work_until(future_base &F) override;

  /// C++ does not support non-static thread_local variables, so this needs to
  /// store both the task and the associated dispatcher queue so that shutdown
  /// can wait for the correct tasks to finish.
#if LLVM_ENABLE_THREADS
  thread_local static
#endif
      SmallVector<std::pair<std::unique_ptr<Task>, InPlaceTaskDispatcher *>>
          TaskQueue;
  SmallVector<std::unique_ptr<Task>> ElsewhereQueue;
};

#if LLVM_ENABLE_THREADS

class LLVM_ABI DynamicThreadPoolTaskDispatcher : public TaskDispatcher {
public:
  DynamicThreadPoolTaskDispatcher(
      std::optional<size_t> MaxMaterializationThreads)
      : MaxMaterializationThreads(MaxMaterializationThreads) {}

  void dispatch(std::unique_ptr<Task> T) override;
  void shutdown() override;

private:
  void work_until(future_base &F) override;
  bool canRunMaterializationTaskNow();
  bool canRunIdleTaskNow();

  bool Shutdown = false;
  size_t Outstanding = 0;
  std::condition_variable OutstandingCV;

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

/// Value storage template for future EBCO pattern
/// Moved outside class to avoid GCC nested template specialization issues
template <typename U> struct future_value_storage {
  unique_function<void(U &&)> then;
  // Union disables default construction/destruction semantics, allowing us to
  // use placement new/delete for precise control over value lifetime
  union {
    U value_;
  };

  future_value_storage() : then(nullptr) {}
  ~future_value_storage() {}
};

template <> struct future_value_storage<void> {
  // No value_ member for void
  unique_function<void(void)> then = nullptr;
};

/// Status for future/promise state
enum class FutureStatus : uint8_t {
  NotReady = 0,
  Ready = 1,
  HaveWaiter = 2,
  HaveThen = 4,
  LLVM_MARK_AS_BITMASK_ENUM(HaveThen)
};

/// @}

/// Type-erased base class for futures, generally for scheduler use to avoid
/// needing virtual dispatches
class future_base {
public:
  /// Check if the future is now ready with a value (precondition: get_promise()
  /// must have been called)
  bool ready() const {
    if (!valid())
      report_fatal_error("ready() called before get_promise()");
    return (static_cast<FutureStatus>(
                state_->status_.load(std::memory_order_acquire)) &
            FutureStatus::Ready) != FutureStatus::NotReady;
  }

  /// Check if the future is in a valid state (not moved-from and get_promise()
  /// called)
  bool valid() const { return state_ != nullptr; }

  /// Wait for the future to be ready, helping with task dispatch
  void wait() {
    // Set the have_waiter bit to indicate someone is waiting
    auto old_status = static_cast<FutureStatus>(
        state_->status_.fetch_or(static_cast<uint8_t>(FutureStatus::HaveWaiter),
                                 std::memory_order_release));

    // Check if Ready bit was already set before fetch_or
    if ((old_status & FutureStatus::Ready) != FutureStatus::NotReady)
      return;

    // Keep helping with task dispatch until our future is ready
    state_->D.work_until(*this);
    if ((static_cast<FutureStatus>(
             state_->status_.load(std::memory_order_relaxed)) &
         FutureStatus::Ready) == FutureStatus::NotReady)
      report_fatal_error(
          "work_until() returned without this future being ready");
  }

protected:
  struct state_base {
    TaskDispatcher &D;
    std::atomic<uint8_t> status_;
    state_base(TaskDispatcher &D)
        : D(D), status_(static_cast<uint8_t>(FutureStatus::NotReady)) {}
  };

  future_base(state_base *state) : state_(state) {}
  future_base() = default;

  /// Only allow deleting the future once it is invalid
  ~future_base() {
    if (valid())
      report_fatal_error("get() must be called before future destruction "
                         "(ensuring promise::set_value memory is valid)");
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
/// - The future is created before the promise, then the promise is created
///   from the future.
/// - The future is in an invalid state until `get_promise()` has been called.
/// - Waiting operations (`get(&D)`, `wait(&D)`) help dispatch other tasks while
///   blocked, requiring an additional argument of which TaskDispatcher object
///   of where all associated work will be scheduled.
/// - While `wait` may be called multiple times and on multiple threads, all of
///   them must have returned before calling `get` on exactly one thread.
/// - Must call `get()` or `then(next)` exactly once before destruction
///   (enforced with `report_fatal_error`) after each call to `get_promise`.
///   Internal state is freed when `get` returns or the `next` is called, and
///   allocated when `get_promise` is called.
/// - Subsequent work can be scheduled cheaply with `then` instead of requiring
///   creating a dedicated thread and waiting on the `future`.
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
  struct state : public future_base::state_base,
                 public future_value_storage<T> {
    state(TaskDispatcher &D) : state_base(D){};
  };

  future() : future_base(nullptr) {}
  future(const future &) = delete;
  future &operator=(const future &) = delete;
  future(future &&) = default;
  future &operator=(future &&) = default;

  /// Get the value, helping with task dispatch while waiting.
  /// This will destroy the underlying value, so this must be called exactly
  /// once, which returns the future to the initial state.
  T get() {
    if (!valid())
      report_fatal_error(
          "get() or then() must only be called once, after get_promise()");
    wait();
    auto state = static_cast<typename future<T>::state *>(state_);
    state_ = nullptr;
    return take_value(state);
  }

  /// Get the value and then schedule a Task to call `H` using dispatcher `D`
  // This awkward construction is necessary since `void(T)` is invalid to
  // substitute with `void` even though it is legal to be `void`
  using ThenCall = unique_function<typename std::conditional<
      std::is_void<T>::value, void(void),
      void(typename std::conditional<std::is_void<T>::value, std::nullopt_t,
                                     T>::type &&)>::type>;

public:
  void then(ThenCall H) && {
    if (!valid())
      report_fatal_error(
          "get() or then() must only be called once, after get_promise()");
    auto state = static_cast<typename future<T>::state *>(state_);
    assert(!state->then);

    state->then = std::move(H);
    // Set the have_waiter bit to indicate someone is waiting
    auto old_status = static_cast<FutureStatus>(
        state->status_.fetch_or(static_cast<uint8_t>(FutureStatus::HaveThen),
                                std::memory_order_release));
    // Check if Ready bit was already set before fetch_or
    if ((old_status & FutureStatus::Ready) != FutureStatus::NotReady)
      state->D.dispatch(makeGenericNamedTask(
          [f = std::move(*this)]() mutable { f.then_continue(); }));
    else
      state_ = nullptr; // state owned by promise<T> now
  }

  /// Get the associated promise (must only be called once)
  promise<T> get_promise(TaskDispatcher &D) {
    if (valid())
      report_fatal_error("get_promise() can only be called once");
    auto state = new typename future<T>::state(D);
    state_ = state;
    return promise<T>(state);
  }

private:
  friend class promise<T>;
  future(future<T>::state *state) : future_base(state) {}

  template <typename U = T>
  static typename std::enable_if<!std::is_void<U>::value, U>::type
  take_value(state *state) {
    T result = std::move(state->value_);
    state->value_.~T();
    delete state;
    return result;
  }

  template <typename U = T>
  static typename std::enable_if<std::is_void<U>::value, U>::type
  take_value(state *state) {
    delete state;
  }

  template <typename U = T>
  typename std::enable_if<!std::is_void<U>::value, void>::type then_continue() {
    auto state = static_cast<typename future<T>::state *>(state_);
    state_ = nullptr;
    state->then(std::move(state->value_));
    state->value_.~T();
    delete state;
  }

  template <typename U = T>
  typename std::enable_if<std::is_void<U>::value, void>::type then_continue() {
    auto state = static_cast<typename future<T>::state *>(state_);
    state_ = nullptr;
    state->then();
    delete state;
  }
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
/// - Created from a future via get_promise() rather than creating the future
/// from the promise.
/// - Must call get_promise() on the thread that created it (it can be passed to
/// another thread, but do not borrow a reference and use that to mutate it from
/// another thread).
/// - Must call set_value() exactly once per `get_promise()` call to provide the
/// result.
/// - Thread-safe from set_value to get.
/// - Move-only semantics to prevent accidental copying.
///
/// The `promise` can usually be passed to another thread in one of two ways:
/// - With move semantics:
///     * `[P = F.get_promise()] () { P.set_value(); }`
///     * `[P = std::move(P)] () { P.set_value(); }`
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
/// - Calling get_promise() more than once.
/// - Calling set_value() more than once.
/// - Destroying a future without calling get().
/// - Calling get() more than once on a future.
/// - Destroying a promise without calling set_value().
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
  promise() : state_(nullptr) {}

  ~promise() {
    // Assert proper promise lifecycle: ensure set_value was called if promise
    // was valid. This can catch deadlocks where a promise is created but
    // set_value() is never called, though only if the promise is moved from
    // instead of borrowed from the frame with the future. Empty promises
    // (state_ == nullptr) are allowed to be destroyed without calling
    // set_value.
    assert(state_ == nullptr &&
           "Destroying a promise without calling set_value");
  }

  promise(const promise &) = delete;
  promise &operator=(const promise &) = delete;

  promise(promise &&other) noexcept : state_(other.state_) {
    other.state_ = nullptr;
  }

  promise &operator=(promise &&other) noexcept {
    if (this != &other) {
      this->~promise();
      state_ = other.state_;
      other.state_ = nullptr;
    }
    return *this;
  }

  /// Set the value (must only be called once)
  // In C++20, this std::conditional weirdness can probably be replaced just
  // with requires. It ensures that we don't try to define a method for `void&`,
  // but that if the user calls set_value(v) for any value v that they get a
  // member function error, instead of no member named 'value_'.
  template <typename U = T>
  void set_value(
      const typename std::conditional<std::is_void<T>::value, std::nullopt_t,
                                      T>::type &value) const {
    assert(state_ && "set_value() can only be called once");
    new (&state_->value_) T(value);
    notify_waiters();
  }

  template <typename U = T>
  void
  set_value(typename std::conditional<std::is_void<T>::value, std::nullopt_t,
                                      T>::type &&value) const {
    assert(state_ && "set_value() can only be called once");
    new (&state_->value_) T(std::move(value));
    notify_waiters();
  }

  template <typename U = T>
  typename std::enable_if<std::is_void<U>::value, void>::type
  set_value(const std::nullopt_t &value) = delete;

  template <typename U = T>
  typename std::enable_if<std::is_void<U>::value, void>::type
  set_value(std::nullopt_t &&value) = delete;

  template <typename U = T>
  typename std::enable_if<std::is_void<U>::value, void>::type
  set_value() const {
    assert(state_ && "set_value() can only be called once");
    notify_waiters();
  }

  /// Swap with another promise
  void swap(promise &other) noexcept {
    using std::swap;
    swap(state_, other.state_);
  }

private:
  explicit promise(typename future<T>::state *state) : state_(state) {}

  void notify_waiters() const {
    typename future<T>::state *state = state_;
    state_ = nullptr;
    // Check if have_waiter was set before setting ready, then atomically set
    // ready bit (release Ready & acquire HaveThen together)
    auto old_status = static_cast<FutureStatus>(state->status_.fetch_or(
        static_cast<uint8_t>(FutureStatus::Ready), std::memory_order_acq_rel));
    if ((old_status & FutureStatus::HaveWaiter) == FutureStatus::HaveWaiter)
      TaskDispatcher::notifyWaiters();
    if ((old_status & FutureStatus::HaveThen) == FutureStatus::HaveThen) {
      state->D.dispatch_elsewhere(makeGenericNamedTask(
          [f = future<T>(state)]() mutable { f.then_continue(); }));
    }
  }

  mutable typename future<T>::state *state_;
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
