// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 -Wthread-safety %s

// Without -Wthread-safety-beta the unchecked-result diagnostics cannot
// report a leaked try-acquire downstream, so a same-origin branch join
// keeps the eager lost-hold diagnosis instead of silently reconstituting
// the try-held state.

struct __attribute__((capability("mutex"))) Mutex {
  void Lock() __attribute__((acquire_capability()));
  void Unlock() __attribute__((release_capability()));
  bool TryLock() __attribute__((try_acquire_capability(true)));
};

Mutex mu;
int a __attribute__((guarded_by(mu)));
bool cond;

void same_origin_branch_join_warns_without_beta() {
  bool failed = !mu.TryLock(); // expected-note {{mutex acquired here}}
  if (failed)
    cond = true;
  a = 3;       // expected-warning {{mutex 'mu' is not held on every path through here}} \
               // expected-warning {{writing variable 'a' requires holding mutex 'mu' exclusively}}
  mu.Unlock(); // expected-warning {{releasing mutex 'mu' that was not held}}
}

// Loop joins are the deliberate exception: the demotion applies there in
// either mode, including at the continue latches the check-first idioms
// create. Gating these on beta as well would only restore the false
// positives the exemption exists to remove -- the eager diagnosis here is
// not a leak report, it is a complaint about a join that loses nothing.
void same_origin_continue_latch_silent_without_beta() {
  bool b = false;
  while (cond) {
    if (b)
      mu.Unlock();
    b = mu.TryLock();
    if (!b)
      continue;
    a = 1;
  }
  if (b)
    mu.Unlock();
}

// The same shape leaking (nothing releases after the loop) is silent in
// this mode too, and equally silent under beta: the loop-carried result
// leaves no fact for the unchecked-result diagnostics to report. Pinned so
// that a future beta-only leak report is a visible change rather than an
// accident.
void same_origin_continue_latch_leak_is_silent() {
  bool b = false;
  while (cond) {
    if (b)
      mu.Unlock();
    b = mu.TryLock();
    if (!b)
      continue;
    a = 1;
  }
}

// A `?:` arm split reconstitutes silently even without beta: the branch
// was never honored before, so its join never warned.
void cond_split_join_stays_silent_without_beta() {
  int r = mu.TryLock() ? 1 : 2;
  if (r == 1) {
    a = 3;       // expected-warning {{writing variable 'a' requires holding mutex 'mu' exclusively}}
    mu.Unlock(); // expected-warning {{releasing mutex 'mu' that may not be held}}
  }
}
