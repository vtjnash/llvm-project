// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wthread-safety -verify=expected,add %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wthread-safety \
// RUN:   -Wno-thread-safety-conversion-add -verify=expected %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify=silent %s
// silent-no-diagnostics

// A capability requirement carried by a function type is transparent to
// conversions -- Sema::IsFunctionConversion adds and drops it the way it adds
// and drops 'noexcept' -- but silently doing so is exactly what an annotation
// must not do: converting to a type without the requirement makes every call
// through the result unchecked, and converting to a type with one promises a
// precondition the source never stated. Both directions are reported, in
// separate subgroups of -Wthread-safety-conversion.

#define LOCKABLE __attribute__((lockable))
#define REQUIRES(...) __attribute__((requires_capability(__VA_ARGS__)))
#define REQUIRES_SHARED(...) \
  __attribute__((requires_shared_capability(__VA_ARGS__)))
#define ELR(...) __attribute__((exclusive_locks_required(__VA_ARGS__)))
#define ACQUIRE(...) __attribute__((acquire_capability(__VA_ARGS__)))
#define RELEASE(...) __attribute__((release_capability(__VA_ARGS__)))
#define RELEASE_GENERIC(...) __attribute__((release_generic_capability(__VA_ARGS__)))
#define TRY_ACQUIRE(...) __attribute__((try_acquire_capability(__VA_ARGS__)))
#define EXCLUDES(...) __attribute__((locks_excluded(__VA_ARGS__)))

class LOCKABLE Mutex {};
Mutex mu1, mu2;

typedef void (*plain)();
typedef REQUIRES(mu1) void (*req1)();
typedef REQUIRES(mu2) void (*req2)();
typedef REQUIRES(mu1) REQUIRES(mu2) void (*req12)();

//===----------------------------------------------------------------------===//
// Pointer to pointer, in every assignment-like context.
//===----------------------------------------------------------------------===//

// Only the 'drop' direction is reported for a precondition. requires_capability
// and locks_excluded constrain the *caller*: giving a pointer a requirement the
// function itself does not state only asks callers for more than the function
// needs, and every call through the pointer is still checked against what the
// type says. Losing one, on the other hand, stops the checking. The
// postcondition attributes -- acquire, release, assert, try_acquire -- are
// reported in both directions, because gaining one makes the analysis believe
// the callee touches a capability it does not (see the sections below).

void take_plain(plain);
void take_req1(req1);

void init(req1 a, plain p) {
  plain lost = a;  // expected-warning {{implicit conversion from 'req1' (aka 'void (*)() __attribute__((requires_capability(mu1)))') to 'plain' (aka 'void (*)()') drops the 'requires_capability(mu1)' requirement; calls through the result are not checked}}
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
  req1 narrowed = d;  // expected-warning {{drops the 'requires_capability(mu2)' requirement}}
  req12 widened = a;
  (void)narrowed;
  (void)widened;
}

// A null pointer constant states nothing and is not a function pointer, so it
// never reports anything.
req1 from_null = nullptr;
req1 from_zero = 0;

//===----------------------------------------------------------------------===//
// Decay (and address-of) from a function whose requirement is on its
// *declaration*: a function declaration's attributes are deliberately not
// folded into its type, so they are read from the declaration here.
//===----------------------------------------------------------------------===//

void annotated() REQUIRES(mu1);   // expected-note 3 {{'requires_capability(mu1)' requirement declared here}}
void unannotated();

void decay_matching() {
  // The feature's intended pattern: the function states what the type states,
  // so nothing changes and nothing is reported.
  req1 by_decay = annotated;
  req1 by_addr = &annotated;
  (void)by_decay;
  (void)by_addr;
}

void decay_dropping() {
  plain by_decay = annotated;      // expected-warning {{'annotated' drops the 'requires_capability(mu1)' requirement when converted to 'plain' (aka 'void (*)()')}}
  plain by_addr = &annotated;      // expected-warning {{'annotated' drops the 'requires_capability(mu1)' requirement when converted to 'plain' (aka 'void (*)()')}}
  void (*raw)() = annotated;       // expected-warning {{'annotated' drops the 'requires_capability(mu1)' requirement when converted to 'void (*)()'}}
  (void)by_decay;
  (void)by_addr;
  (void)raw;
}

void decay_adding() {
  // The pointer type promises a precondition the function does not need --
  // usually annotation drift on one side or the other.
  req1 gained = unannotated;
  req1 gained2 = &unannotated;
  (void)gained;
  (void)gained2;
}

// Passing an annotated function to an equally annotated parameter is silent
// too, whichever side spells the requirement: a parameter's attributes are not
// folded either (its type is part of the enclosing function's type), so they
// are read from the parameter declaration.
void takes_annotated_param(void (*cb)() REQUIRES(mu1));
void takes_annotated_type(req1 cb);

void call_annotated_param() {
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

// A capability argument that is relative to an object or to a parameter
// cannot be expressed by any function pointer type (see
// Sema::foldCapabilityAttrsIntoType), so storing such a function in a plain
// function pointer is not dropping anything the pointer could have kept. This
// is the form that predates type-carried requirements, and it stays silent.
void relative(Mutex *m) REQUIRES(*m);

void object_relative_is_not_lost() {
  void (*p)(Mutex *) = relative;
  void (*q)(Mutex *) = &relative;
  (void)p;
  (void)q;
}

struct Holder {
  Mutex mu;
  void (*cb)() REQUIRES(mu); // requirement stays on the field
};

void member_relative_is_not_lost(Holder *h) {
  void (*p)() = h->cb;
  (void)p;
}

//===----------------------------------------------------------------------===//
// Explicit casts are the escape hatch.
//===----------------------------------------------------------------------===//

void casts(req1 a, plain p) {
  plain c1 = (plain)a;
  req1 c2 = (req1)p;
  plain c3 = static_cast<plain>(a);
  req1 c4 = static_cast<req1>(p);
  plain c5 = reinterpret_cast<plain>(a);
  req1 c6 = reinterpret_cast<req1>(p);
  plain c7 = plain(a);
  plain c8 = (plain)annotated;
  plain c9 = static_cast<plain>(&annotated);
  (void)c1; (void)c2; (void)c3; (void)c4; (void)c5;
  (void)c6; (void)c7; (void)c8; (void)c9;
}

//===----------------------------------------------------------------------===//
// What counts as the same requirement is what makes two types identical.
//===----------------------------------------------------------------------===//

// Differently spelled synonyms state the same requirement.
typedef ELR(mu1) void (*elr1)();

void synonyms(req1 a, elr1 e) {
  elr1 x = a;
  req1 y = e;
  (void)x;
  (void)y;
}

// Sharedness is part of the requirement, so exclusive and shared are two
// different requirements: one is dropped and the other added.
typedef REQUIRES_SHARED(mu1) void (*shared1)();

void sharedness(req1 a, shared1 s) {
  shared1 x = a; // expected-warning {{drops the 'requires_capability(mu1)' requirement}}
  req1 y = s;    // expected-warning {{drops the 'requires_shared_capability(mu1)' requirement}}
  (void)x;
  (void)y;
}

// So is a try-acquire's success value.
typedef TRY_ACQUIRE(1, mu1) bool (*try_true)();
typedef TRY_ACQUIRE(0, mu1) bool (*try_false)();

void try_success_value(try_true t) {
  try_false x = t; // expected-warning {{drops the 'try_acquire_capability(1, mu1)' requirement}} \
                   // add-warning {{adds the 'try_acquire_capability(0, mu1)' requirement}}
  (void)x;
}

// And so is the genericness of a release.
typedef RELEASE(mu1) void (*rel1)();
typedef RELEASE_GENERIC(mu1) void (*relgen1)();

void genericness(rel1 r) {
  relgen1 x = r; // expected-warning {{drops the 'release_capability(mu1)' requirement}} \
                 // add-warning {{adds the 'release_generic_capability(mu1)' requirement}}
  (void)x;
}

// Every kind of capability attribute participates.
typedef ACQUIRE(mu1) void (*acq1)();
typedef EXCLUDES(mu1) void (*exc1)();

void other_kinds(acq1 q, exc1 e) {
  plain x = q; // expected-warning {{drops the 'acquire_capability(mu1)' requirement}}
  plain y = e; // expected-warning {{drops the 'locks_excluded(mu1)' requirement}}
  (void)x;
  (void)y;
}

//===----------------------------------------------------------------------===//
// The conditional operator.
//===----------------------------------------------------------------------===//

// The composite type is the intersection, so an operand that required more
// loses the difference -- a real loss, reported like any other. The
// intersection never adds anything, so the 'add' direction cannot fire.
void conditional(int c, plain p, req1 a, req2 b, req12 d) {
  auto x = c ? a : p; // expected-warning {{drops the 'requires_capability(mu1)' requirement}}
  auto y = c ? a : d; // expected-warning {{drops the 'requires_capability(mu2)' requirement}}
  auto z = c ? a : b; // expected-warning {{drops the 'requires_capability(mu1)' requirement}} \
                      // expected-warning {{drops the 'requires_capability(mu2)' requirement}}
  auto w = c ? a : a; // equal sets: nothing changes
  (void)x; (void)y; (void)z; (void)w;
}

//===----------------------------------------------------------------------===//
// Speculative conversions must stay quiet.
//===----------------------------------------------------------------------===//

// Overload resolution considers a candidate it does not pick; only the
// conversion that is actually committed is reported.
void overloaded(plain);
void overloaded(int);

void pick_overload(req1 a) {
  overloaded(a); // expected-warning {{drops the 'requires_capability(mu1)' requirement}}
}

// Deduction produces the argument's own type, so there is no conversion.
template <class T> void deduce(T) {}
template <class T> void deduce_ptr(void (*)()) {}

void deduction(req1 a) {
  deduce(a);
  auto copy = a;
  (void)copy;
}

// A template instantiated with an annotated type converts inside the
// instantiation, once.
template <class T> plain to_plain(T cb) {
  return cb; // expected-warning {{drops the 'requires_capability(mu1)' requirement}}
}

void instantiate(req1 a) {
  to_plain(a); // expected-note {{in instantiation of function template specialization 'to_plain<void (*)() __attribute__((requires_capability(mu1)))>' requested here}}
}

//===----------------------------------------------------------------------===//
// A reference to a function pointer binds without converting.
//===----------------------------------------------------------------------===//

void by_reference(req1 &r, const req1 &cr) {
  req1 copy = r;
  req1 copy2 = cr;
  (void)copy;
  (void)copy2;
}

//===----------------------------------------------------------------------===//
// A captureless lambda converted to a function pointer.
//===----------------------------------------------------------------------===//

// The requirement is written on the closure's operator(), and the pointer the
// conversion yields cannot carry it, so it is dropped just as it is for a
// named function. The message names the lambda rather than 'operator()'.
// Every initialization form is covered: a user-defined conversion is reported
// where it is performed (SemaInit), not at the PerformImplicitConversion seam
// it never reaches.
void take_plain_fp(plain);
plain return_lambda() {
  return [](void) REQUIRES(mu1) {}; // expected-warning {{lambda drops the 'requires_capability(mu1)' requirement when converted to 'plain' (aka 'void (*)()'); calls through the result are not checked}} \
                                    // expected-note {{'requires_capability(mu1)' requirement declared here}}
}

void lambda_to_fnptr() {
  plain a = [](void) REQUIRES(mu1) {}; // expected-warning {{lambda drops the 'requires_capability(mu1)' requirement}} \
                                       // expected-note {{'requires_capability(mu1)' requirement declared here}}
  plain b;
  b = [](void) REQUIRES(mu1) {};       // expected-warning {{lambda drops the 'requires_capability(mu1)' requirement}} \
                                       // expected-note {{'requires_capability(mu1)' requirement declared here}}
  take_plain_fp([](void) REQUIRES(mu1) {}); // expected-warning {{lambda drops the 'requires_capability(mu1)' requirement}} \
                                            // expected-note {{'requires_capability(mu1)' requirement declared here}}
  (void)a; (void)b;
}

// Nothing is lost when the destination states the requirement too, even though
// the closure's conversion operator itself returns a plain pointer: the
// comparison is against what is ultimately being initialized.
void take_req1_fp(req1);
void lambda_to_annotated_fnptr() {
  req1 a = [](void) REQUIRES(mu1) {};
  take_req1_fp([](void) REQUIRES(mu1) {});
  (void)a;
}

// An explicit cast opts out here as it does at the other seams, and an
// unannotated lambda has nothing to lose.
void lambda_optouts() {
  plain a = (plain)[](void) REQUIRES(mu1) {};
  plain b = [](void) {};
  (void)a; (void)b;
}

// A user-defined conversion that is not a lambda is unaffected.
struct ConvToFp { operator plain(); };
void non_lambda_conversion() {
  plain a = ConvToFp();
  (void)a;
}

// A lambda passed to a by-value template parameter (std::function and
// llvm::unique_function shaped) deduces the closure type, so no function
// pointer conversion happens at all and there is nothing to report. The call
// inside the lambda body is still checked against the lambda's own annotation.
template <class Sig> struct erased { template <class G> erased(G g) {} };
void take_erased(erased<void (*)()>);
void annotated_cb(void) REQUIRES(mu1);
void via_lambda() REQUIRES(mu1) {
  take_erased([](void) REQUIRES(mu1) { annotated_cb(); });
}
