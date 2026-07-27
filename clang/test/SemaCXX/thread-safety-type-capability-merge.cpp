// RUN: %clang_cc1 -fsyntax-only -verify=expected -std=c++11 -Wthread-safety %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,merge -std=c++11 \
// RUN:            -Wthread-safety -Wthread-safety-typedef-merge %s
//
// Merging two variable declarations also has to check their exception
// specifications, which are not part of the type here; that check is only run
// when exceptions are enabled, hence the second configuration.
// RUN: %clang_cc1 -fsyntax-only -verify=expected,exc -std=c++11 \
// RUN:     -fcxx-exceptions -fexceptions -Wthread-safety %s

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

// Converting an operand to the composite is a real conversion, so the operand
// that required more is reported by -Wthread-safety-conversion-drop -- the
// requirement it stated really does stop being enforced. The intersection can
// only ever remove requirements, so the 'add' direction never fires here.
void composite(int c, plain p, req1 a, req2 b, req12 d) {
  // {mu1} intersect {} == {}, in either operand order.
  SameType<decltype(c ? a : p), plain>(); // expected-warning {{drops the 'requires_capability' requirement}}
  SameType<decltype(c ? p : a), plain>(); // expected-warning {{drops the 'requires_capability' requirement}}

  // {mu1} intersect {mu1, mu2} == {mu1}, in either operand order.
  SameType<decltype(c ? a : d), req1>(); // expected-warning {{drops the 'requires_capability' requirement}}
  SameType<decltype(c ? d : a), req1>(); // expected-warning {{drops the 'requires_capability' requirement}}

  // Disjoint requirements intersect to nothing.
  SameType<decltype(c ? a : b), plain>(); // expected-warning 2 {{drops the 'requires_capability' requirement}}

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
  auto dropped = c ? a : p; // expected-warning {{drops the 'requires_capability' requirement}}
  auto kept = c ? a : d;    // expected-warning {{drops the 'requires_capability' requirement}}
  dropped();
  kept(); // expected-warning {{calling function 'kept' requires holding mutex 'mu1' exclusively}}
}

//===----------------------------------------------------------------------===//
// A variable's redeclarations merge the same way (Sema::MergeVarDeclTypes).
//===----------------------------------------------------------------------===//

// A capability attribute written on a function-pointer variable is folded into
// the variable's type, so writing it on only one declaration makes the two
// declarations' types differ. C++ compares redeclared variable types with
// hasSameType, which would reject that; capability-only differences are routed
// through the same union rule ASTContext::mergeFunctionTypes uses for C.
extern void (*var_annotated_second)();
void (*var_annotated_second)() REQ(mu1);

void (*var_annotated_first)() REQ(mu1);
extern void (*var_annotated_first)();

// Each declaration may state a different requirement; the variable requires
// both.
extern void (*var_both)() REQ(mu1);
void (*var_both)() REQ(mu2);

// The union is the type, not merely an attribute inherited onto the second
// declaration. (Only the single-requirement case is pinned down this way: the
// order of a type's requirement list is part of its identity, and the union
// lists the new declaration's requirements first.)
SameType<decltype(var_annotated_first), req1> merged_is_req1;
SameType<decltype(var_annotated_second), req1> merged_is_req1_too;

// Two declarations that state the same two requirements in opposite orders.
// Requirement order is part of a type's identity, so these are two different
// types and must go through the union -- which gives both of them the new
// declaration's order -- rather than being waved through as "the same set".
typedef REQ(mu2) REQ(mu1) void (*req21)();
SameType<req12, req12> req12_is_itself;
static_assert(!__is_same(req12, req21),
              "requirement order is part of the type");

extern void (*var_order)() REQ(mu1) REQ(mu2);
void (*var_order)() REQ(mu2) REQ(mu1);
SameType<decltype(var_order), req21> reordered_takes_new_order;

// The same, stated through the two typedefs rather than directly.
extern req12 var_typedef_order;
extern req21 var_typedef_order;
SameType<decltype(var_typedef_order), req21> reordered_typedef_takes_new_order;

void use_reordered_vars() {
  var_order();         // expected-warning {{calling function 'var_order' requires holding mutex 'mu1' exclusively}} expected-warning {{calling function 'var_order' requires holding mutex 'mu2' exclusively}}
  var_typedef_order(); // expected-warning {{calling function 'var_typedef_order' requires holding mutex 'mu1' exclusively}} expected-warning {{calling function 'var_typedef_order' requires holding mutex 'mu2' exclusively}}
}

void use_merged_vars() {
  var_annotated_second(); // expected-warning {{calling function 'var_annotated_second' requires holding mutex 'mu1' exclusively}}
  var_annotated_first();  // expected-warning {{calling function 'var_annotated_first' requires holding mutex 'mu1' exclusively}}
  var_both();             // expected-warning {{calling function 'var_both' requires holding mutex 'mu1' exclusively}} expected-warning {{calling function 'var_both' requires holding mutex 'mu2' exclusively}}

  auto copy = var_annotated_first;
  copy(); // expected-warning {{calling function 'copy' requires holding mutex 'mu1' exclusively}}
}

// A static data member declared in the class and defined out of line, with the
// requirement written only in the class.
struct Holder {
  static void (*fp)() REQ(mu1);
};
void (*Holder::fp)();

void use_static_member() {
  auto copy = Holder::fp;
  copy(); // expected-warning {{calling function 'copy' requires holding mutex 'mu1' exclusively}}
}

//===----------------------------------------------------------------------===//
// Redefining a typedef with a different requirement takes the union.
//===----------------------------------------------------------------------===//

// Unlike every other property carried by a canonical function type (noexcept,
// calling convention, cfi_salt, function effects), a differing requirement is
// not a redefinition conflict: the typedef requires all of what its
// definitions state. This is what lets a callback type declared by an external
// header be annotated -- by repeating its typedef with the attribute -- without
// modifying that header. -Wthread-safety-typedef-merge reports the difference,
// and is deliberately not part of -Wthread-safety.
typedef REQ(mu1) void (*redef)(); // merge-note {{previous definition is here}}
typedef void (*redef)();          // merge-warning {{does not state the same capability requirements}} \
                                  // merge-note {{previous definition is here}}
typedef REQ(mu2) void (*redef)(); // merge-warning {{does not state the same capability requirements}}

void use_redef(redef p) {
  p(); // expected-warning {{calling function 'p' requires holding mutex 'mu1' exclusively}} expected-warning {{calling function 'p' requires holding mutex 'mu2' exclusively}}
}

// Repeating the identical requirement, including through a synonym, is fine
// and is not reported even with -Wthread-safety-typedef-merge.
typedef REQ(mu1) void (*ok_redef)();
typedef REQ(mu1) void (*ok_redef)();
typedef __attribute__((exclusive_locks_required(mu1))) void (*ok_redef)();

// The same holds for a 'using' alias, in either order.
using ualias REQ(mu1) = void (*)(); // merge-note {{previous definition is here}}
using ualias = void (*)();          // merge-warning {{does not state the same capability requirements}}

using ualias2 = void (*)();          // merge-note {{previous definition is here}}
using ualias2 REQ(mu1) = void (*)(); // merge-warning {{does not state the same capability requirements}}

void use_ualias(ualias p, ualias2 q) {
  p(); // expected-warning {{calling function 'p' requires holding mutex 'mu1' exclusively}}
  q(); // expected-warning {{calling function 'q' requires holding mutex 'mu1' exclusively}}
}

// A redefinition that differs in more than its requirements is still an error.
typedef REQ(mu1) void (*bad_redef)(int);  // expected-note {{previous definition is here}}
typedef REQ(mu1) void (*bad_redef)(long); // expected-error {{typedef redefinition with different types}}

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

// A redeclaration that differs in more than its requirements is still an
// error.
extern void (*var_conflict)() REQ(mu1); // expected-note {{previous declaration is here}}
extern int (*var_conflict)() REQ(mu1);  // expected-error {{redeclaration of 'var_conflict' with a different type}}

//===----------------------------------------------------------------------===//
// The capability merge does not skip the exception-specification check.
//
// This section comes last on purpose: it is the only one that reports an
// error before the file's last analysis-based warning, and an error anywhere
// earlier would suppress every warning after it.
//===----------------------------------------------------------------------===//

// The exception specification of a variable's type is not part of that type
// before C++17 (and an unresolved one never is), so Sema::MergeVarDeclTypes
// checks it separately instead of leaving it to hasSameType. Two declarations
// that differ in their requirements as well take the capability-merge path,
// which has to run the same check: writing a capability attribute on one of
// them must not excuse a mismatch that is diagnosed without one.
extern void (*espec_base)() throw(int);   // exc-note {{previous declaration is here}}
extern void (*espec_base)() throw(float); // exc-error {{exception specification in declaration does not match previous declaration}}

extern void (*espec_caps)() throw(int) REQ(mu1); // exc-note {{previous declaration is here}}
extern void (*espec_caps)() throw(float);        // exc-error {{exception specification in declaration does not match previous declaration}}

// Matching exception specifications are no obstacle: the requirements still
// merge to their union.
extern void (*espec_ok)() throw(int) REQ(mu1);
extern void (*espec_ok)() throw(int) REQ(mu2);
SameType<decltype(espec_ok), req21> espec_ok_is_both;
