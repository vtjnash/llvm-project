// RUN: %clang_cc1 -fsyntax-only -Wthread-safety -verify=expected,add %s
// RUN: %clang_cc1 -fsyntax-only -Wthread-safety \
// RUN:   -Wno-thread-safety-conversion-add -verify=expected %s
// RUN: %clang_cc1 -fsyntax-only -verify=silent %s
// silent-no-diagnostics

// The C twin of SemaCXX/thread-safety-type-capability-conversion.cpp. C++
// reports these from Sema::PerformImplicitConversion; C has no such single
// seam, so the check sits in Sema::CheckSingleAssignmentConstraints -- which
// every assignment-like context (assignment, initialization, argument
// passing, 'return') funnels through -- and in the conditional operator's
// composite-pointer computation.

#define LOCKABLE __attribute__((lockable))
#define REQUIRES(...) __attribute__((requires_capability(__VA_ARGS__)))
#define REQUIRES_SHARED(...) \
  __attribute__((requires_shared_capability(__VA_ARGS__)))
#define ELR(...) __attribute__((exclusive_locks_required(__VA_ARGS__)))
#define ACQUIRE(...) __attribute__((acquire_capability(__VA_ARGS__)))
#define RELEASE(...) __attribute__((release_capability(__VA_ARGS__)))
#define TRY_ACQUIRE(...) __attribute__((try_acquire_capability(__VA_ARGS__)))
#define EXCLUDES(...) __attribute__((locks_excluded(__VA_ARGS__)))

struct LOCKABLE Mutex {};
struct Mutex mu1, mu2;

typedef void (*plain)(void);
typedef REQUIRES(mu1) void (*req1)(void);
typedef REQUIRES(mu2) void (*req2)(void);
typedef REQUIRES(mu1) REQUIRES(mu2) void (*req12)(void);

//===----------------------------------------------------------------------===//
// Pointer to pointer, in every assignment-like context.
//===----------------------------------------------------------------------===//

// A conversion is reported only where it could leave the analysis believing a
// capability is held when it may not be. Losing requires_capability or
// locks_excluded stops them being checked; losing release_capability leaves
// the analysis believing the capability is still held after a call that
// released it; and gaining acquire, assert or try_acquire makes it believe one
// was taken that the function never touches. The opposite combinations leave
// the analysis believing less, which is conservative, and are not reported.

void take_plain(plain);
void take_req1(req1);

void init(req1 a, plain p) {
  plain lost = a;  // expected-warning {{implicit conversion from 'req1' (aka 'void (*)(void) __attribute__((requires_capability(mu1)))') to 'plain' (aka 'void (*)(void)') drops the 'requires_capability(mu1)' requirement; calls through the result are not checked}}
  req1 gained = p;
  (void)lost;
  (void)gained;
}

void assign(req1 a, plain p) {
  plain lost;
  req1 gained;
  lost = a;   // expected-warning {{drops the 'requires_capability(mu1)' requirement}}
  gained = p;
}

void argument(req1 a, plain p) {
  take_plain(a); // expected-warning {{drops the 'requires_capability(mu1)' requirement}}
  take_req1(p);
}

plain return_drops(req1 a) {
  return a; // expected-warning {{drops the 'requires_capability(mu1)' requirement}}
}

req1 return_adds(plain p) {
  return p;
}

// Only the requirement that actually differs is reported.
void partial(req12 d, req1 a) {
  req1 narrowed = d; // expected-warning {{drops the 'requires_capability(mu2)' requirement}}
  req12 widened = a;
  (void)narrowed;
  (void)widened;
}

// An initializer inside an aggregate goes through the same seam.
struct Ops {
  plain raw;
  req1 annotated;
};

void aggregate(req1 a, plain p) {
  struct Ops o = {a, p}; // expected-warning {{drops the 'requires_capability(mu1)' requirement}}
  (void)o;
}

// A null pointer constant is not a function pointer and states nothing.
req1 from_null = 0;

//===----------------------------------------------------------------------===//
// Decay (and address-of) from a function whose requirement is on its
// *declaration*.
//===----------------------------------------------------------------------===//

void annotated(void) REQUIRES(mu1); // expected-note 3 {{'requires_capability(mu1)' requirement declared here}}
void unannotated(void);

void decay_matching(void) {
  // The intended pattern: the function states what the type states.
  req1 by_decay = annotated;
  req1 by_addr = &annotated;
  (void)by_decay;
  (void)by_addr;
}

void decay_dropping(void) {
  plain by_decay = annotated;  // expected-warning {{'annotated' drops the 'requires_capability(mu1)' requirement when converted to 'plain' (aka 'void (*)(void)')}}
  plain by_addr = &annotated;  // expected-warning {{'annotated' drops the 'requires_capability(mu1)' requirement when converted to 'plain' (aka 'void (*)(void)')}}
  void (*raw)(void) = annotated; // expected-warning {{'annotated' drops the 'requires_capability(mu1)' requirement when converted to 'void (*)(void)'}}
  (void)by_decay;
  (void)by_addr;
  (void)raw;
}

void decay_adding(void) {
  req1 gained = unannotated;
  req1 gained2 = &unannotated;
  (void)gained;
  (void)gained2;
}

// A parameter's attributes are not folded into its type either, so passing an
// equally annotated function to it is silent whichever side spells it.
void takes_annotated_param(void (*cb)(void) REQUIRES(mu1));
void takes_annotated_type(req1 cb);

void call_annotated_param(void) {
  takes_annotated_param(annotated);
  takes_annotated_param(&annotated);
  takes_annotated_type(annotated);
  // The parameter's requirement is not in its type, so the two types print
  // the same here; the note says where the difference comes from.
  takes_annotated_param(unannotated);
}

//===----------------------------------------------------------------------===//
// A requirement that could never have been part of a type is not "lost".
//===----------------------------------------------------------------------===//

void relative(struct Mutex *m) REQUIRES(*m);

void object_relative_is_not_lost(void) {
  void (*p)(struct Mutex *) = relative;
  void (*q)(struct Mutex *) = &relative;
  (void)p;
  (void)q;
}

struct Holder {
  struct Mutex mu;
  void (*cb)(void) REQUIRES(mu); // requirement stays on the field
};

void member_relative_is_not_lost(struct Holder *h) {
  void (*p)(void) = h->cb;
  (void)p;
}

//===----------------------------------------------------------------------===//
// An explicit cast is the escape hatch.
//===----------------------------------------------------------------------===//

void casts(req1 a, plain p) {
  plain c1 = (plain)a;
  req1 c2 = (req1)p;
  plain c3 = (plain)annotated;
  plain c4 = (void (*)(void))&annotated;
  (void)c1; (void)c2; (void)c3; (void)c4;
}

//===----------------------------------------------------------------------===//
// What counts as the same requirement is what makes two types identical.
//===----------------------------------------------------------------------===//

typedef ELR(mu1) void (*elr1)(void);

void synonyms(req1 a, elr1 e) {
  elr1 x = a;
  req1 y = e;
  (void)x;
  (void)y;
}

typedef REQUIRES_SHARED(mu1) void (*shared1)(void);

void sharedness(req1 a, shared1 s) {
  shared1 x = a; // expected-warning {{drops the 'requires_capability(mu1)' requirement}}
  req1 y = s;    // expected-warning {{drops the 'requires_shared_capability(mu1)' requirement}}
  (void)x;
  (void)y;
}

typedef TRY_ACQUIRE(1, mu1) int (*try_true)(void);
typedef TRY_ACQUIRE(0, mu1) int (*try_false)(void);

void try_success_value(try_true t) {
  // Only the gained postcondition is reported: the lost one leaves the
  // analysis assuming less, which is safe.
  try_false x = t; // add-warning {{adds the 'try_acquire_capability(0, mu1)' requirement}}
  (void)x;
}

typedef ACQUIRE(mu1) void (*acq1)(void);
typedef RELEASE(mu1) void (*rel1)(void);
typedef EXCLUDES(mu1) void (*exc1)(void);

// Losing 'acquire' only makes the analysis believe the capability is *not*
// held when it may be, which costs false positives but never a missed race, so
// it is not reported. Losing 'release' is the opposite: the analysis goes on
// believing the capability is held after a call that released it, and would
// then permit accesses that are really unprotected. 'locks_excluded' is a
// precondition, so losing it stops being checked at all.
void other_kinds(acq1 q, rel1 r, exc1 e) {
  plain x = q;
  plain y = r; // expected-warning {{drops the 'release_capability(mu1)' requirement}}
  plain z = e; // expected-warning {{drops the 'locks_excluded(mu1)' requirement}}
  (void)x; (void)y; (void)z;
}

//===----------------------------------------------------------------------===//
// The conditional operator.
//===----------------------------------------------------------------------===//

// The composite type is the intersection, so the operand that required more
// loses the difference. The intersection never adds anything.
void conditional(int c, plain p, req1 a, req2 b, req12 d) {
  __typeof__(c ? a : p) x = p; // expected-warning {{drops the 'requires_capability(mu1)' requirement}}
  req1 y = c ? a : d;          // expected-warning {{drops the 'requires_capability(mu2)' requirement}}
  __typeof__(c ? a : b) z = p; // expected-warning {{drops the 'requires_capability(mu1)' requirement}} \
                               // expected-warning {{drops the 'requires_capability(mu2)' requirement}}
  req1 w = c ? a : a;          // equal sets: nothing changes
  (void)x; (void)y; (void)z; (void)w;
}

//===----------------------------------------------------------------------===//
// A requirement stated by a declaration, when both types are the same.
//===----------------------------------------------------------------------===//

// When the requirement lives on a declaration rather than in either type, the
// source and destination can be the very same type. Naming both would read as
// a mistake ("conversion from 'T' to 'T'"), so the message leads with the
// function whose requirement is at stake and names the type once. The note
// says where the requirement was written, since neither printed type shows it.
void visit_all(void (*visit)(int), int n);
void visit_cb(int x) REQUIRES(mu1); // expected-note {{'requires_capability(mu1)' requirement declared here}}

void same_type_drop(int n) {
  visit_all(visit_cb, n); // expected-warning {{'visit_cb' drops the 'requires_capability(mu1)' requirement when converted to 'void (*)(int)'; calls through the result are not checked}}
}

// The mirror image -- a parameter that states a precondition the argument does
// not -- is not reported: the callee is simply happy to be called with more
// held than it needs.
void visit_all_req(void (*visit)(int) REQUIRES(mu1), int n);
void visit_plain(int x);

void same_type_add(int n) {
  visit_all_req(visit_plain, n);
}

// A requirement on a parameter may name another parameter of the same
// prototype, which stands for whatever is passed for it at each call. Such a
// requirement can never be part of a type, so matching it against the
// argument's own requirement takes substituting the argument -- the
// kref_put_lock() shape. ('lock' precedes the callback here so that this stays
// independent of -fexperimental-late-parse-attributes; Sema/thread-safety-
// late-parse.c covers naming a parameter declared later.)
struct Mutex mu_a;
struct Mutex mu_b;

void put_lock(int *obj, struct Mutex *lock, void (*release)(int *) RELEASE(lock));
void release_a(int *obj) RELEASE(&mu_a); // expected-note {{'release_capability(&mu_a)' requirement declared here}}

void substituted_match(int *obj) {
  // '&mu_a' is passed for 'lock', which is what release_a releases: silent.
  put_lock(obj, &mu_a, release_a);
}

void substituted_mismatch(int *obj) {
  // '&mu_b' is passed for 'lock', but release_a releases '&mu_a', so the
  // requirement really is dropped.
  put_lock(obj, &mu_b, release_a); // expected-warning {{'release_a' drops the 'release_capability(&mu_a)' requirement when converted to 'void (*)(int *)'; calls through the result are not checked}}
}

// Substitution picks the parameter the requirement names, not the neighbour.
void put_two(int *obj, struct Mutex *other, struct Mutex *lock,
             void (*release)(int *) RELEASE(lock));

void substituted_second(int *obj) {
  put_two(obj, &mu_b, &mu_a, release_a);
}
