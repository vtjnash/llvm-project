// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 -Wthread-safety -Wthread-safety-beta %s

// A try-acquire call's resolved result is a try fact of its own, separate
// from the definite hold it proved: what happens to the hold afterwards
// (another call acquires the capability, a release releases the result)
// leaves the record of the call's result standing, and one call's record
// says nothing about another call's.

struct __attribute__((capability("mutex"))) Mutex {
  void Lock() __attribute__((acquire_capability()));
  void Unlock() __attribute__((release_capability()));
  bool TryLock() __attribute__((try_acquire_capability(true)));
};

struct __attribute__((capability("mutex"), reentrant_capability)) RMutex {
  void Lock() __attribute__((acquire_capability()));
  void Unlock() __attribute__((release_capability()));
  bool TryLock() __attribute__((try_acquire_capability(true)));
};

Mutex mu;
RMutex rmu;
int a __attribute__((guarded_by(mu)));
int b __attribute__((guarded_by(rmu)));

// The call's failure record survives a later acquisition by a blocking
// call: the result is still falsy, so a re-check inside the locked region
// finds its success edge infeasible and the dead arm's release resolves
// nothing at the join. (A negative fact consumed by the acquisition could
// not carry the failure, and the dead arm then drew a lost-hold, a
// guarded-write and a stray-release warning.)
void failed_survives_lock() {
  bool ok = mu.TryLock();
  if (!ok) {
    mu.Lock();
    if (ok) {
      mu.Unlock();
    }
    a = 1;
  }
  mu.Unlock();
}

// The call's released record survives a later acquisition by a blocking
// call: the stale result resolves no success edge, so the branch on it
// neither resurrects a hold nor releases the blocking call's.
void spent_survives_lock() {
  bool ok = mu.TryLock();
  if (ok)
    mu.Unlock();
  mu.Lock();
  if (ok) {
    a = 1;
  }
  mu.Unlock();
}

// Two proved holds over one reentrant capability, one level released:
// definite levels are interchangeable, so both try facts still prove the
// remaining level and the second release leaves the capability released
// on that path. The hold is lost once at the three-way join, and the
// second call's result reaches the end of the function unchecked.
void two_proofs_one_release() {
  bool ok1 = rmu.TryLock(); // expected-note {{mutex acquired here}}
  bool ok2 = rmu.TryLock(); // expected-note {{mutex acquired here}}
  if (ok1 && ok2) {
    rmu.Unlock();
    if (ok1) {
      b = 1;
      rmu.Unlock();
    }
  }
} // expected-warning {{mutex 'rmu' is not held on every path through here}} \
  // expected-warning {{unchecked result of try-acquire; mutex 'rmu' may still be held at the end of function}}

// The stale-result veto is per call: another call's release on the other
// side of a join says nothing about this call's result, so the hold this
// call proved is still carried, demoted to its conditional try fact, into
// the rebranch on its stored result. (A single negative fact per
// capability could not tell whose stale truth would do the resurrecting,
// and refused the demotion: the hold was lost at the join, the guarded
// write and the release under the rebranch both warned.)
void per_call_spent_veto(bool c) {
  bool ok = false, ok2 = false;
  if (c) {
    ok = mu.TryLock();
    if (!ok)
      return;
  } else {
    ok2 = mu.TryLock();
    if (ok2)
      mu.Unlock();
  }
  if (ok) {
    a = 1;
    mu.Unlock();
  }
}
