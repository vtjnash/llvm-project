// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 -Wthread-safety \
// RUN:   -Wnullable-to-nonnull-conversion %s

// Folding a capability attribute into a function-pointer typedef rebuilds the
// pointer, so everything spelled around it has to be put back: the qualifiers
// on the pointer itself and the sugar that carries a nullability specifier.
// Losing them would silently change what the typedef means -- a 'const'
// function pointer would become assignable, a '_Nonnull' one would stop being
// checked -- while still looking right in the source.
//
// The requirement must of course keep working in every one of those shapes.

#define LOCKABLE __attribute__((lockable))
#define EXCLUSIVE_LOCK_FUNCTION(...) \
  __attribute__((exclusive_lock_function(__VA_ARGS__)))
#define UNLOCK_FUNCTION(...) __attribute__((unlock_function(__VA_ARGS__)))
#define REQUIRES(...) __attribute__((exclusive_locks_required(__VA_ARGS__)))

class LOCKABLE Mutex {
public:
  void Lock() EXCLUSIVE_LOCK_FUNCTION();
  void Unlock() UNLOCK_FUNCTION();
};

Mutex mu1;

typedef void (*plain)(void) REQUIRES(mu1);

//===----------------------------------------------------------------------===//
// Qualifiers on the pointer
//===----------------------------------------------------------------------===//

typedef void (*const cfp)(void) REQUIRES(mu1);
typedef void (*volatile vfp)(void) REQUIRES(mu1);
typedef void (*__attribute__((address_space(1))) asfp)(void) REQUIRES(mu1);

// Qualifiers are part of the canonical type, so a lost qualifier collapses the
// typedef onto the unqualified one.
static_assert(!__is_same(cfp, plain), "'const' must survive the fold");
static_assert(!__is_same(vfp, plain), "'volatile' must survive the fold");
static_assert(!__is_same(asfp, plain), "the address space must survive");
static_assert(!__is_same(cfp, vfp), "distinct qualifiers, distinct types");

// Two spellings of the same qualified requirement still unique together.
typedef void (*const cfp2)(void) REQUIRES(mu1);
static_assert(__is_same(cfp, cfp2), "same qualifiers and requirement");

// A qualifier can also come from a typedef that the fold has to look through
// (the sugar is dropped, the qualifier it contributed is not).
typedef void (*const already_const)(void);
typedef already_const cfp3 REQUIRES(mu1);
static_assert(__is_same(cfp, cfp3), "a qualifier behind sugar must survive");

void call_const(cfp p) {
  p(); // expected-warning {{calling function 'p' requires holding mutex 'mu1' exclusively}}
}

void call_const_locked(cfp p) {
  mu1.Lock();
  p();
  mu1.Unlock();
}

void call_volatile(vfp p) {
  p(); // expected-warning {{calling function 'p' requires holding mutex 'mu1' exclusively}}
}

void call_behind_sugar(cfp3 p) {
  p(); // expected-warning {{calling function 'p' requires holding mutex 'mu1' exclusively}}
}

//===----------------------------------------------------------------------===//
// Nullability
//===----------------------------------------------------------------------===//

// Nullability is type sugar rather than a qualifier, so re-applying the
// qualifiers is not enough: the AttributedType has to be rebuilt around the
// new pointer. Both typedefs carry the same requirement so that the
// conversion below is a single implicit conversion.
typedef void (*_Nonnull nonnull_cb)(void) REQUIRES(mu1);
typedef void (*_Nullable nullable_cb)(void) REQUIRES(mu1);

void nullability_kept(nullable_cb n) {
  nonnull_cb p = n; // expected-warning {{implicit conversion from nullable pointer 'nullable_cb' (aka 'void (*)() __attribute__((exclusive_locks_required(mu1)))') to non-nullable pointer type 'nonnull_cb'}}
  p();              // expected-warning {{calling function 'p' requires holding mutex 'mu1' exclusively}}
}

// Nullability and a qualifier at the same time: the qualifier sits inside the
// AttributedType here, so both rebuilds have to happen in the right order.
typedef void (*_Nonnull const nonnull_const_cb)(void) REQUIRES(mu1);
static_assert(!__is_same(nonnull_const_cb, plain), "'const' must survive");

void call_nonnull_const(nonnull_const_cb p) {
  p(); // expected-warning {{calling function 'p' requires holding mutex 'mu1' exclusively}}
}

//===----------------------------------------------------------------------===//
// The user-visible symptom of a dropped qualifier
//===----------------------------------------------------------------------===//

void some_function(void);

void assign_const(cfp p) { // expected-note {{variable 'p' declared const here}}
  p = &some_function;      // expected-error {{cannot assign to variable 'p' with const-qualified type 'cfp'}}
}

void assign_nonnull_const(nonnull_const_cb p) { // expected-note {{variable 'p' declared const here}}
  p = &some_function; // expected-error {{cannot assign to variable 'p' with const-qualified type 'nonnull_const_cb'}}
}

void assign_plain(plain p) {
  p = &some_function; // no error: nothing here is const
}
