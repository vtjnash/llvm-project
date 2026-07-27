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

void take_plain(plain);
void take_req1(req1);

void init(req1 a, plain p) {
  plain lost = a;  // expected-warning {{implicit conversion from 'req1' (aka 'void (*)(void) __attribute__((requires_capability(mu1)))') to 'plain' (aka 'void (*)(void)') drops the 'requires_capability' requirement; calls through the result are not checked}}
  req1 gained = p; // add-warning {{implicit conversion from 'plain' (aka 'void (*)(void)') to 'req1' (aka 'void (*)(void) __attribute__((requires_capability(mu1)))') adds a 'requires_capability' requirement that the source does not state}}
  (void)lost;
  (void)gained;
}

void assign(req1 a, plain p) {
  plain lost;
  req1 gained;
  lost = a;   // expected-warning {{drops the 'requires_capability' requirement}}
  gained = p; // add-warning {{adds a 'requires_capability' requirement}}
}

void argument(req1 a, plain p) {
  take_plain(a); // expected-warning {{drops the 'requires_capability' requirement}}
  take_req1(p);  // add-warning {{adds a 'requires_capability' requirement}}
}

plain return_drops(req1 a) {
  return a; // expected-warning {{drops the 'requires_capability' requirement}}
}

req1 return_adds(plain p) {
  return p; // add-warning {{adds a 'requires_capability' requirement}}
}

// Only the requirement that actually differs is reported.
void partial(req12 d, req1 a) {
  req1 narrowed = d; // expected-warning {{drops the 'requires_capability' requirement}}
  req12 widened = a; // add-warning {{adds a 'requires_capability' requirement}}
  (void)narrowed;
  (void)widened;
}

// An initializer inside an aggregate goes through the same seam.
struct Ops {
  plain raw;
  req1 annotated;
};

void aggregate(req1 a, plain p) {
  struct Ops o = {a, p}; // expected-warning {{drops the 'requires_capability' requirement}} \
                         // add-warning {{adds a 'requires_capability' requirement}}
  (void)o;
}

// A null pointer constant is not a function pointer and states nothing.
req1 from_null = 0;

//===----------------------------------------------------------------------===//
// Decay (and address-of) from a function whose requirement is on its
// *declaration*.
//===----------------------------------------------------------------------===//

void annotated(void) REQUIRES(mu1); // expected-note 3 {{'requires_capability' requirement declared here}}
void unannotated(void);

void decay_matching(void) {
  // The intended pattern: the function states what the type states.
  req1 by_decay = annotated;
  req1 by_addr = &annotated;
  (void)by_decay;
  (void)by_addr;
}

void decay_dropping(void) {
  plain by_decay = annotated;  // expected-warning {{drops the 'requires_capability' requirement}}
  plain by_addr = &annotated;  // expected-warning {{drops the 'requires_capability' requirement}}
  void (*raw)(void) = annotated; // expected-warning {{drops the 'requires_capability' requirement}}
  (void)by_decay;
  (void)by_addr;
  (void)raw;
}

void decay_adding(void) {
  req1 gained = unannotated;   // add-warning {{adds a 'requires_capability' requirement}}
  req1 gained2 = &unannotated; // add-warning {{adds a 'requires_capability' requirement}}
  (void)gained;
  (void)gained2;
}

// A parameter's attributes are not folded into its type either, so passing an
// equally annotated function to it is silent whichever side spells it.
void takes_annotated_param(void (*cb)(void) REQUIRES(mu1)); // add-note {{'requires_capability' requirement declared here}}
void takes_annotated_type(req1 cb);

void call_annotated_param(void) {
  takes_annotated_param(annotated);
  takes_annotated_param(&annotated);
  takes_annotated_type(annotated);
  // The parameter's requirement is not in its type, so the two types print
  // the same here; the note says where the difference comes from.
  takes_annotated_param(unannotated); // add-warning {{adds a 'requires_capability' requirement}}
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
  shared1 x = a; // expected-warning {{drops the 'requires_capability' requirement}} \
                 // add-warning {{adds a 'requires_shared_capability' requirement}}
  req1 y = s;    // expected-warning {{drops the 'requires_shared_capability' requirement}} \
                 // add-warning {{adds a 'requires_capability' requirement}}
  (void)x;
  (void)y;
}

typedef TRY_ACQUIRE(1, mu1) int (*try_true)(void);
typedef TRY_ACQUIRE(0, mu1) int (*try_false)(void);

void try_success_value(try_true t) {
  try_false x = t; // expected-warning {{drops the 'try_acquire_capability' requirement}} \
                   // add-warning {{adds a 'try_acquire_capability' requirement}}
  (void)x;
}

typedef ACQUIRE(mu1) void (*acq1)(void);
typedef RELEASE(mu1) void (*rel1)(void);
typedef EXCLUDES(mu1) void (*exc1)(void);

void other_kinds(acq1 q, rel1 r, exc1 e) {
  plain x = q; // expected-warning {{drops the 'acquire_capability' requirement}}
  plain y = r; // expected-warning {{drops the 'release_capability' requirement}}
  plain z = e; // expected-warning {{drops the 'locks_excluded' requirement}}
  (void)x; (void)y; (void)z;
}

//===----------------------------------------------------------------------===//
// The conditional operator.
//===----------------------------------------------------------------------===//

// The composite type is the intersection, so the operand that required more
// loses the difference. The intersection never adds anything.
void conditional(int c, plain p, req1 a, req2 b, req12 d) {
  __typeof__(c ? a : p) x = p; // expected-warning {{drops the 'requires_capability' requirement}}
  req1 y = c ? a : d;          // expected-warning {{drops the 'requires_capability' requirement}}
  __typeof__(c ? a : b) z = p; // expected-warning 2 {{drops the 'requires_capability' requirement}}
  req1 w = c ? a : a;          // equal sets: nothing changes
  (void)x; (void)y; (void)z; (void)w;
}
