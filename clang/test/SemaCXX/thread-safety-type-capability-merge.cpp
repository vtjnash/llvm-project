// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wthread-safety %s

// The C++ counterpart of Sema/thread-safety-type-capability-merge.c: forming a
// composite pointer type for the conditional operator has to be transparent to
// capability attributes, the same way IsFunctionConversion is, and has to
// intersect them -- a call through the composite may reach either operand, so
// it can only be required to hold what both operands require. Before this,
//'c ? caps_fn_ptr : plain_fn_ptr' was a hard error in C++.

#define LOCKABLE __attribute__((lockable))
#define REQ(...) __attribute__((requires_capability(__VA_ARGS__)))

class LOCKABLE Mutex {};
Mutex mu1, mu2;

typedef void (*plain)();
typedef REQ(mu1) void (*req1)();
typedef REQ(mu2) void (*req2)();
typedef REQ(mu1) REQ(mu2) void (*req12)();

template <typename T, typename U> struct SameType;
template <typename T> struct SameType<T, T> {};

//===----------------------------------------------------------------------===//
// The conditional operator takes the intersection.
//===----------------------------------------------------------------------===//

void composite(int c, plain p, req1 a, req2 b, req12 d) {
  // {mu1} intersect {} == {}, in either operand order.
  SameType<decltype(c ? a : p), plain>();
  SameType<decltype(c ? p : a), plain>();

  // {mu1} intersect {mu1, mu2} == {mu1}, in either operand order.
  SameType<decltype(c ? a : d), req1>();
  SameType<decltype(c ? d : a), req1>();

  // Disjoint requirements intersect to nothing.
  SameType<decltype(c ? a : b), plain>();

  // The composite of two equal sets is that set. (The unary '+' only makes
  // the operands prvalues, so that decltype does not report a reference.)
  SameType<decltype(c ? +a : +a), req1>();
  SameType<decltype(c ? +d : +d), req12>();

  // A null pointer constant leaves the annotated type alone.
  SameType<decltype(c ? a : nullptr), req1>();
}

// A call through the composite is only checked for what both operands
// required. (The analysis needs a declaration to report, so the composite is
// bound to a variable rather than called directly.)
void call_composite(int c, plain p, req1 a, req12 d) {
  auto dropped = c ? a : p;
  auto kept = c ? a : d;
  dropped();
  kept(); // expected-warning {{calling function 'kept' requires holding mutex 'mu1' exclusively}}
}

//===----------------------------------------------------------------------===//
// Redefining a typedef with a different requirement stays an error.
//===----------------------------------------------------------------------===//

// See ThreadSafetyTypeCapabilities-ReviewFindings.md, F16: the underlying types
// really are different types, and clang rejects the same mismatch for every
// other property carried by a canonical function type (noexcept, calling
// convention, cfi_salt, function effects such as [[clang::nonblocking]]).
typedef REQ(mu1) void (*redef)(); // expected-note 2 {{previous definition is here}}
typedef void (*redef)();          // expected-error {{typedef redefinition with different types}}
typedef REQ(mu2) void (*redef)(); // expected-error {{typedef redefinition with different types}}

// Repeating the identical requirement, including through a synonym, is fine.
typedef REQ(mu1) void (*ok_redef)();
typedef REQ(mu1) void (*ok_redef)();
typedef __attribute__((exclusive_locks_required(mu1))) void (*ok_redef)();

// The same holds for a 'using' alias.
using ualias REQ(mu1) = void (*)(); // expected-note {{previous definition is here}}
using ualias = void (*)();          // expected-error {{type alias redefinition with different types}}

//===----------------------------------------------------------------------===//
// Negative guard: distinct requirements stay distinct types.
//===----------------------------------------------------------------------===//

void f(req1);
void f(req2);
void f(plain);
void f(req12);

// Overload resolution sees four different parameter types; redeclaring one of
// them is not a redefinition of another.
void g(req1 a, req2 b, plain p, req12 d) {
  f(a);
  f(b);
  f(p);
  f(d);
}
