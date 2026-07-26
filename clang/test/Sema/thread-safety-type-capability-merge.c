// RUN: %clang_cc1 -fsyntax-only -verify -Wthread-safety %s

// Capability attributes are part of the canonical function type, so every
// place that merges two function types has to have a rule for them.
// ASTContext::mergeFunctionTypes takes the union when merging a redeclaration
// (supplemental information is often written on only one declaration) and the
// intersection when forming a composite type for the conditional operator (a
// call through the composite may reach either operand). This mirrors how
// noreturn and function effects are merged.

#define LOCKABLE __attribute__((lockable))
#define REQ(...) __attribute__((requires_capability(__VA_ARGS__)))

struct LOCKABLE Mutex {};
struct Mutex mu1, mu2;

typedef void (*plain)(void);
typedef REQ(mu1) void (*req1)(void);
typedef REQ(mu2) void (*req2)(void);
typedef REQ(mu1) REQ(mu2) void (*req12)(void);

//===----------------------------------------------------------------------===//
// Redeclaration merging takes the union.
//===----------------------------------------------------------------------===//

// The requirement written on the first declaration survives a compatible
// redeclaration that does not repeat it, in either order.
void annotated_first(req1 p);
void annotated_first(plain p);

void annotated_second(plain p);
void annotated_second(req1 p);

// Two declarations that each state a different requirement merge to both.
void both(req1 p);
void both(req12 p);

void use_merged(req1 p, plain q) {
  p(); // expected-warning {{calling function 'p' requires holding mutex 'mu1' exclusively}}
  q();
}

//===----------------------------------------------------------------------===//
// The conditional operator takes the intersection.
//===----------------------------------------------------------------------===//

// __typeof__ pins down the composite type: assigning a 'plain' value to it is
// only valid when the composite dropped the requirement, and assigning it to a
// 'req1' variable is only valid when the composite kept it.
void composite(int c, plain p, req1 a, req2 b, req12 d) {
  // {mu1} intersect {} == {}, in either operand order.
  __typeof__(c ? a : p) drops1 = p;
  __typeof__(c ? p : a) drops2 = p;

  // {mu1} intersect {mu1, mu2} == {mu1}, in either operand order.
  req1 keeps1 = c ? a : d;
  req1 keeps2 = c ? d : a;

  // Disjoint requirements intersect to nothing.
  __typeof__(c ? a : b) empty = p;

  // The composite of two equal sets is that set.
  req1 same = c ? a : a;

  (void)drops1; (void)drops2; (void)keeps1;
  (void)keeps2; (void)empty; (void)same;
}

// A call through the composite is only checked for what both operands
// required. (The analysis needs a declaration to report, so the composite is
// bound to a variable rather than called directly.)
void call_composite(int c, plain p, req1 a, req12 d) {
  __typeof__(c ? a : p) dropped = c ? a : p;
  __typeof__(c ? a : d) kept = c ? a : d;
  dropped();
  kept(); // expected-warning {{calling function 'kept' requires holding mutex 'mu1' exclusively}}
}

//===----------------------------------------------------------------------===//
// A variable's redeclarations merge the same way.
//===----------------------------------------------------------------------===//

// A capability attribute written on a function-pointer variable is folded into
// the variable's type, so an attribute written on only one of its declarations
// makes the two declarations' types differ. mergeFunctionTypes unions them, so
// the variable requires whatever any of its declarations states, in either
// order.
extern void (*var_annotated_second)(void);
void (*var_annotated_second)(void) REQ(mu1);

void (*var_annotated_first)(void) REQ(mu1);
extern void (*var_annotated_first)(void);

// Each declaration may state a different requirement; the variable requires
// both.
extern void (*var_both)(void) REQ(mu1);
void (*var_both)(void) REQ(mu2);

void use_merged_vars(void) {
  var_annotated_second(); // expected-warning {{calling function 'var_annotated_second' requires holding mutex 'mu1' exclusively}}
  var_annotated_first();  // expected-warning {{calling function 'var_annotated_first' requires holding mutex 'mu1' exclusively}}
  var_both();             // expected-warning {{calling function 'var_both' requires holding mutex 'mu1' exclusively}} expected-warning {{calling function 'var_both' requires holding mutex 'mu2' exclusively}}
}

// The merged requirement is in the type, not merely inherited onto the second
// declaration, so it survives a copy of the same type.
void use_merged_var_copy(void) {
  __typeof__(var_annotated_first) copy = var_annotated_first;
  copy(); // expected-warning {{calling function 'copy' requires holding mutex 'mu1' exclusively}}
}

//===----------------------------------------------------------------------===//
// Redefining a typedef with a different requirement stays an error.
//===----------------------------------------------------------------------===//

// The two underlying types really are different types, and clang rejects the
// same mismatch for every other property carried by a canonical function type
// (noexcept, calling convention, cfi_salt, function effects such as
// [[clang::nonblocking]]). Type-carried capabilities are treated no
// differently: see ThreadSafetyTypeCapabilities-ReviewFindings.md, F16.
typedef REQ(mu1) void (*redef)(void); // expected-note 2 {{previous definition is here}}
typedef void (*redef)(void);          // expected-error {{typedef redefinition with different types}}
typedef REQ(mu2) void (*redef)(void); // expected-error {{typedef redefinition with different types}}

// Repeating the identical requirement, including through a synonym, is fine.
typedef REQ(mu1) void (*ok_redef)(void);
typedef REQ(mu1) void (*ok_redef)(void);
typedef __attribute__((exclusive_locks_required(mu1))) void (*ok_redef)(void);

// A redeclaration that differs in more than its requirements is still an
// error.
extern void (*var_conflict)(void) REQ(mu1); // expected-note {{previous declaration is here}}
extern int (*var_conflict)(void) REQ(mu1);  // expected-error {{redeclaration of 'var_conflict' with a different type}}
