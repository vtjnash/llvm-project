// RUN: %clang_cc1 -std=c++11 -ast-print -Wthread-safety %s > %t.cpp
// RUN: FileCheck --input-file=%t.cpp %s
//
// The printed output has to parse back, and to print the same way the second
// time around: every argument is printed (including try-acquire's success
// value), and the GNU spelling is used, which is the one accepted here.
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -Wthread-safety %t.cpp
// RUN: %clang_cc1 -std=c++11 -ast-print -Wthread-safety %t.cpp | FileCheck %s

// Capability attributes folded into a function-pointer typedef's type are
// printed as part of the type.

struct __attribute__((capability("mutex"))) Mutex {};
Mutex mu;
Mutex mu2;

typedef void (*req_cb_t)(void) __attribute__((requires_capability(mu)));
// CHECK: typedef void (*req_cb_t)() __attribute__((requires_capability(mu)));

typedef void (*req_shared_cb_t)(void)
    __attribute__((requires_shared_capability(mu)));
// CHECK: typedef void (*req_shared_cb_t)() __attribute__((requires_shared_capability(mu)));

typedef void (*acq_cb_t)(void) __attribute__((acquire_capability(mu)));
// CHECK: typedef void (*acq_cb_t)() __attribute__((acquire_capability(mu)));

typedef void (*acq_shared_cb_t)(void)
    __attribute__((acquire_shared_capability(mu)));
// CHECK: typedef void (*acq_shared_cb_t)() __attribute__((acquire_shared_capability(mu)));

typedef void (*rel_cb_t)(void) __attribute__((release_capability(mu)));
// CHECK: typedef void (*rel_cb_t)() __attribute__((release_capability(mu)));

typedef void (*rel_shared_cb_t)(void)
    __attribute__((release_shared_capability(mu)));
// CHECK: typedef void (*rel_shared_cb_t)() __attribute__((release_shared_capability(mu)));

typedef void (*rel_generic_cb_t)(void)
    __attribute__((release_generic_capability(mu)));
// CHECK: typedef void (*rel_generic_cb_t)() __attribute__((release_generic_capability(mu)));

// The success value is an argument of the attribute like any other: dropping
// it would print something that does not parse back.
typedef bool (*tryacq_cb_t)(void)
    __attribute__((try_acquire_capability(true, mu)));
// CHECK: typedef bool (*tryacq_cb_t)() __attribute__((try_acquire_capability(true, mu)));

typedef bool (*tryacq_shared_cb_t)(void)
    __attribute__((try_acquire_shared_capability(0, mu)));
// CHECK: typedef bool (*tryacq_shared_cb_t)() __attribute__((try_acquire_shared_capability(0, mu)));

typedef void (*assert_cb_t)(void) __attribute__((assert_capability(mu)));
// CHECK: typedef void (*assert_cb_t)() __attribute__((assert_capability(mu)));

typedef void (*assert_shared_cb_t)(void)
    __attribute__((assert_shared_capability(mu)));
// CHECK: typedef void (*assert_shared_cb_t)() __attribute__((assert_shared_capability(mu)));

typedef void (*excl_cb_t)(void) __attribute__((locks_excluded(mu, mu2)));
// CHECK: typedef void (*excl_cb_t)() __attribute__((locks_excluded(mu, mu2)));

// Several attributes on one type.
typedef void (*multi_cb_t)(void) __attribute__((requires_capability(mu)))
__attribute__((locks_excluded(mu2)));
// CHECK: typedef void (*multi_cb_t)() __attribute__((requires_capability(mu))) __attribute__((locks_excluded(mu2)));

// The name the user spelled the attribute with is kept: the GNU-legacy
// spellings are not reprinted as their clang equivalents. (These have to name
// a capability no equivalent type above already names: two spellings of the
// same attribute describe the same type, and the type that gets uniqued away
// takes the surviving type's spelling with it.)
Mutex gnu_mu;

typedef void (*gnu_req_cb_t)(void)
    __attribute__((exclusive_locks_required(gnu_mu)));
// CHECK: typedef void (*gnu_req_cb_t)() __attribute__((exclusive_locks_required(gnu_mu)));

typedef void (*gnu_req_shared_cb_t)(void)
    __attribute__((shared_locks_required(gnu_mu)));
// CHECK: typedef void (*gnu_req_shared_cb_t)() __attribute__((shared_locks_required(gnu_mu)));

typedef void (*gnu_acq_cb_t)(void)
    __attribute__((exclusive_lock_function(gnu_mu)));
// CHECK: typedef void (*gnu_acq_cb_t)() __attribute__((exclusive_lock_function(gnu_mu)));

typedef void (*gnu_acq_shared_cb_t)(void)
    __attribute__((shared_lock_function(gnu_mu)));
// CHECK: typedef void (*gnu_acq_shared_cb_t)() __attribute__((shared_lock_function(gnu_mu)));

typedef void (*gnu_rel_cb_t)(void) __attribute__((unlock_function(gnu_mu)));
// CHECK: typedef void (*gnu_rel_cb_t)() __attribute__((unlock_function(gnu_mu)));

typedef bool (*gnu_tryacq_cb_t)(void)
    __attribute__((exclusive_trylock_function(true, gnu_mu)));
// CHECK: typedef bool (*gnu_tryacq_cb_t)() __attribute__((exclusive_trylock_function(true, gnu_mu)));

typedef bool (*gnu_tryacq_shared_cb_t)(void)
    __attribute__((shared_trylock_function(true, gnu_mu)));
// CHECK: typedef bool (*gnu_tryacq_shared_cb_t)() __attribute__((shared_trylock_function(true, gnu_mu)));

typedef void (*gnu_assert_cb_t)(void)
    __attribute__((assert_exclusive_lock(gnu_mu)));
// CHECK: typedef void (*gnu_assert_cb_t)() __attribute__((assert_exclusive_lock(gnu_mu)));

typedef void (*gnu_assert_shared_cb_t)(void)
    __attribute__((assert_shared_lock(gnu_mu)));
// CHECK: typedef void (*gnu_assert_shared_cb_t)() __attribute__((assert_shared_lock(gnu_mu)));

// A C++11-spelled attribute is written on the declaration, but still prints as
// part of the type -- with the GNU spelling, which is the one this position
// accepts.
Mutex cxx11_mu;
[[clang::requires_capability(cxx11_mu)]] typedef void (*cxx11_req_cb_t)(void);
// CHECK: typedef void (*cxx11_req_cb_t)() __attribute__((requires_capability(cxx11_mu)));

// Qualifiers and nullability survive the fold, so they still print where they
// were written -- in particular the 'const' stays on the pointer instead of
// disappearing or drifting onto the pointee.
typedef void (*const const_req_cb_t)(void)
    __attribute__((requires_capability(mu)));
// CHECK: typedef void (*const const_req_cb_t)() __attribute__((requires_capability(mu)));

typedef void (*_Nonnull nonnull_req_cb_t)(void)
    __attribute__((requires_capability(mu)));
// CHECK: typedef void (* _Nonnull nonnull_req_cb_t)() __attribute__((requires_capability(mu)));

// FIXME: An alias declaration is deliberately not printed here: the folded
// attribute prints as part of the type, i.e. after the type-id, and that
// position is a parse error in an alias declaration ('using cb = void (*)()
// __attribute__((requires_capability(mu)));'), so the output would not
// re-parse. It has to print after the alias name instead, which needs the
// declaration printer to take the attributes over from the type printer.

// A requirement written on a function-pointer variable or field is folded into
// the declaration's type and also kept on the declaration -- the analysis
// reads it from both. It is one requirement, so it prints once.
//
// A variable prints its type as written (from its TypeSourceInfo), so the
// attribute comes from the declaration; a field prints its (folded) type, so
// the declaration does not repeat what the type already states. Either way the
// output round-trips.
void (*req_var)(void) __attribute__((requires_capability(mu)));
// CHECK: void (*req_var)() __attribute__((requires_capability(mu)));

void (*gnu_req_var)(void) __attribute__((shared_locks_required(gnu_mu)));
// CHECK: void (*gnu_req_var)() __attribute__((shared_locks_required(gnu_mu)));

bool (*tryacq_var)(void) __attribute__((try_acquire_capability(true, mu2)));
// CHECK: bool (*tryacq_var)() __attribute__((try_acquire_capability(true, mu2)));

struct CbOps {
  void (*req_field)(void) __attribute__((requires_capability(mu)));
  void (*multi_field)(void) __attribute__((acquire_capability(mu)))
  __attribute__((locks_excluded(mu2)));
};
// CHECK:      struct CbOps {
// CHECK-NEXT:     void (*req_field)() __attribute__((requires_capability(mu)));
// CHECK-NEXT:     void (*multi_field)() __attribute__((acquire_capability(mu))) __attribute__((locks_excluded(mu2)));
// CHECK-NEXT: };

// A field whose type is a typedef that already carries the requirement, and
// which writes that same requirement itself. The printed type is the typedef's
// *name*, which does not display the requirement, so the declaration is the
// only place left to print it -- suppressing it because the type it names
// happens to carry it would drop it from the output.
struct TypedefCbOps {
  req_cb_t same_req __attribute__((requires_capability(mu)));
  // Writing a requirement the typedef does not state changes the type, and the
  // rebuild cannot put the typedef sugar back, so this one prints both
  // requirements as part of the type instead.
  req_cb_t extra_req __attribute__((requires_capability(mu2)));
};
// CHECK:      struct TypedefCbOps {
// CHECK-NEXT:     req_cb_t same_req __attribute__((requires_capability(mu)));
// CHECK-NEXT:     void (*extra_req)() __attribute__((requires_capability(mu))) __attribute__((requires_capability(mu2)));
// CHECK-NEXT: };

// A requirement that names a sibling member cannot become part of the type, so
// it stays -- and prints -- on the declaration alone. (It prints with the
// implicit 'this->' the parser built, which is what re-parses.)
struct MemberMutexOps {
  Mutex omu;
  void (*req_field)(void) __attribute__((requires_capability(omu)));
};
// CHECK:      struct MemberMutexOps {
// CHECK-NEXT:     Mutex omu;
// CHECK-NEXT:     void (*req_field)() __attribute__((requires_capability(this->omu)));
// CHECK-NEXT: };

// Attributes on a function declaration print on the declaration, unchanged.
void locked_fn(void) __attribute__((exclusive_locks_required(mu)));
// CHECK: void locked_fn() __attribute__((exclusive_locks_required(mu)));

bool trylock_fn(void) __attribute__((exclusive_trylock_function(true, mu)));
// CHECK: bool trylock_fn() __attribute__((exclusive_trylock_function(true, mu)));

// In a template, an attribute whose arguments are still dependent stays on the
// typedef declaration and prints there; the instantiation's copy has been
// substituted and folded into the type, so it prints as part of the type.
template <Mutex *M> struct Tmpl {
  typedef void (*cb)(void) __attribute__((requires_capability(*M)));
};
// CHECK:      template <Mutex *M> struct Tmpl {
// CHECK-NEXT:     typedef void (*cb)() __attribute__((requires_capability(*M)));
// CHECK-NEXT: };
// CHECK:      template<> struct Tmpl<&mu> {
// CHECK-NEXT:     typedef void (*cb)() __attribute__((requires_capability(*&mu)));
// CHECK-NEXT: };
void use_tmpl(Tmpl<&mu>::cb);
// CHECK: void use_tmpl(Tmpl<&mu>::cb);
