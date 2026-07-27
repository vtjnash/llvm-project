// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wthread-safety %s

// Capability attributes folded into a function type take part in that type's
// uniquing. Two properties have to hold, and this file tests both directions:
//
//  * differently spelled synonyms denote the same requirement and must unify
//    into a single type (so the spelling index must NOT be profiled), and
//  * semantic variants -- sharedness, release genericness and try-acquire's
//    success value -- must stay distinct, even though they share an attribute
//    kind and an identical argument list. Otherwise the folding set hands the
//    second declaration the first one's already-uniqued type, silently giving
//    it the first declaration's semantics.

#define LOCKABLE __attribute__((lockable))

class LOCKABLE Mutex {
public:
  void Lock() __attribute__((acquire_capability()));
  void ReaderLock() __attribute__((acquire_shared_capability()));
  void Unlock() __attribute__((release_generic_capability()));
  void ReaderUnlock() __attribute__((release_shared_capability()));
};

Mutex mu;
int x __attribute__((guarded_by(mu)));

//===----------------------------------------------------------------------===//
// Type identity
//===----------------------------------------------------------------------===//

// requires_capability / exclusive_locks_required
typedef void (*req_t)(void) __attribute__((requires_capability(mu)));
typedef void (*req_syn_t)(void) __attribute__((exclusive_locks_required(mu)));
typedef void (*req_sh_t)(void) __attribute__((requires_shared_capability(mu)));
typedef void (*req_sh_syn_t)(void) __attribute__((shared_locks_required(mu)));
static_assert(__is_same(req_t, req_syn_t), "requires synonyms must unify");
static_assert(__is_same(req_sh_t, req_sh_syn_t),
              "requires_shared synonyms must unify");
static_assert(!__is_same(req_t, req_sh_t),
              "requires and requires_shared must stay distinct");

// acquire_capability / exclusive_lock_function
typedef void (*acq_t)(void) __attribute__((acquire_capability(mu)));
typedef void (*acq_syn_t)(void) __attribute__((exclusive_lock_function(mu)));
typedef void (*acq_sh_t)(void) __attribute__((acquire_shared_capability(mu)));
typedef void (*acq_sh_syn_t)(void) __attribute__((shared_lock_function(mu)));
static_assert(__is_same(acq_t, acq_syn_t), "acquire synonyms must unify");
static_assert(__is_same(acq_sh_t, acq_sh_syn_t),
              "acquire_shared synonyms must unify");
static_assert(!__is_same(acq_t, acq_sh_t),
              "acquire and acquire_shared must stay distinct");

// release_generic_capability / unlock_function -- note that the historical
// spelling is the *generic* one, not the exclusive one.
typedef void (*rel_gen_t)(void)
    __attribute__((release_generic_capability(mu)));
typedef void (*rel_gen_syn_t)(void) __attribute__((unlock_function(mu)));
typedef void (*rel_t)(void) __attribute__((release_capability(mu)));
typedef void (*rel_sh_t)(void) __attribute__((release_shared_capability(mu)));
static_assert(__is_same(rel_gen_t, rel_gen_syn_t),
              "release_generic synonyms must unify");
static_assert(!__is_same(rel_gen_t, rel_t) && !__is_same(rel_gen_t, rel_sh_t) &&
                  !__is_same(rel_t, rel_sh_t),
              "the three release kinds must stay distinct");

// assert_capability / assert_exclusive_lock
typedef void (*assert_t)(void) __attribute__((assert_capability(mu)));
typedef void (*assert_syn_t)(void) __attribute__((assert_exclusive_lock(mu)));
typedef void (*assert_sh_t)(void)
    __attribute__((assert_shared_capability(mu)));
typedef void (*assert_sh_syn_t)(void) __attribute__((assert_shared_lock(mu)));
static_assert(__is_same(assert_t, assert_syn_t), "assert synonyms must unify");
static_assert(__is_same(assert_sh_t, assert_sh_syn_t),
              "assert_shared synonyms must unify");
static_assert(!__is_same(assert_t, assert_sh_t),
              "assert and assert_shared must stay distinct");

// try_acquire_capability / exclusive_trylock_function; the success value is a
// separate argument that getCapabilityAttrArgs deliberately excludes, so it
// needs profiling of its own.
typedef bool (*try_t)(void) __attribute__((try_acquire_capability(true, mu)));
typedef bool (*try_syn_t)(void)
    __attribute__((exclusive_trylock_function(true, mu)));
typedef bool (*try_false_t)(void)
    __attribute__((try_acquire_capability(false, mu)));
typedef bool (*try_sh_t)(void)
    __attribute__((try_acquire_shared_capability(true, mu)));
typedef bool (*try_sh_syn_t)(void)
    __attribute__((shared_trylock_function(true, mu)));
static_assert(__is_same(try_t, try_syn_t), "try_acquire synonyms must unify");
static_assert(__is_same(try_sh_t, try_sh_syn_t),
              "try_acquire_shared synonyms must unify");
static_assert(!__is_same(try_t, try_false_t),
              "try_acquire success values must stay distinct");
static_assert(!__is_same(try_t, try_sh_t),
              "try_acquire and try_acquire_shared must stay distinct");

// The success value is profiled by the value it denotes, not by the way it is
// written: 'true' and '1' are one requirement, so they must be one type, or
// two headers that spell the same trylock differently would give it two
// incompatible types (and a spurious ODR mismatch across modules).
typedef bool (*try_one_t)(void) __attribute__((try_acquire_capability(1, mu)));
typedef bool (*try_zero_t)(void)
    __attribute__((try_acquire_capability(0, mu)));
typedef bool (*try_two_t)(void) __attribute__((try_acquire_capability(2, mu)));
static_assert(__is_same(try_t, try_one_t),
              "'true' and '1' are the same success value");
static_assert(__is_same(try_false_t, try_zero_t),
              "'false' and '0' are the same success value");
static_assert(!__is_same(try_one_t, try_two_t),
              "different success values must stay distinct");

// Being one type, the two spellings redefine the same typedef rather than
// conflicting.
typedef bool (*try_t)(void) __attribute__((try_acquire_capability(1, mu)));
typedef bool (*try_false_t)(void)
    __attribute__((try_acquire_capability(0, mu)));

// locks_excluded carries no extra semantic state, but must still not collide
// with an unrelated kind over the same argument.
typedef void (*excl_t)(void) __attribute__((locks_excluded(mu)));
static_assert(!__is_same(excl_t, req_t),
              "locks_excluded and requires must stay distinct");

// The argument list is what separates one attribute from the next, so a single
// attribute over two capabilities differs from two attributes of one each.
Mutex mu2;
typedef void (*req_two_t)(void) __attribute__((requires_capability(mu, mu2)));
typedef void (*req_one_one_t)(void) __attribute__((requires_capability(mu)))
    __attribute__((requires_capability(mu2)));
static_assert(!__is_same(req_two_t, req_one_one_t),
              "argument grouping must be part of the type");

//===----------------------------------------------------------------------===//
// Behavior
//
// In each pair the exclusive/generic variant is declared first, so a colliding
// profile would give the second typedef the first one's semantics.
//===----------------------------------------------------------------------===//

void testExclusiveAcquire(acq_t acq) {
  acq();
  x = 1;
  mu.Unlock();
}

void testSharedAcquire(acq_sh_t acq) {
  acq();
  (void)x; // ok: reading under a shared lock
  mu.ReaderUnlock();
}

void testSharedAcquireThenWrite(acq_sh_t acq) {
  acq();
  x = 1; // expected-warning {{writing variable 'x' requires holding mutex 'mu' exclusively}}
  mu.ReaderUnlock();
}

void testSharedRequiresUnderReaderLock(req_sh_t req) {
  mu.ReaderLock();
  req(); // ok: a shared requirement is satisfied by a reader lock
  mu.ReaderUnlock();
}

void testExclusiveRequiresUnderReaderLock(req_t req) {
  mu.ReaderLock();
  req(); // expected-warning {{calling function 'req' requires holding mutex 'mu' exclusively}}
  mu.ReaderUnlock();
}

void testGenericReleaseOfExclusive(rel_gen_t rel) {
  mu.Lock();
  rel(); // ok: a generic release matches either access kind
}

void testSharedReleaseOfExclusive(rel_sh_t rel) {
  mu.Lock(); // expected-note {{mutex acquired here}}
  rel(); // expected-warning {{releasing mutex 'mu' using shared access, expected exclusive access}}
}

void testTryAcquireSucceedsOnTrue(try_t tryacq) {
  if (tryacq()) {
    x = 1; // ok: mu is held on the branch matching the success value
    mu.Unlock();
  }
}

void testTryAcquireSucceedsOnFalse(try_false_t tryacq) {
  if (tryacq()) {
    x = 1; // expected-warning {{writing variable 'x' requires holding mutex 'mu' exclusively}}
  } else {
    x = 2; // ok: mu is held on the false branch
    mu.Unlock();
  }
}

void testSharedTryAcquireThenWrite(try_sh_t tryacq) {
  if (tryacq()) {
    x = 1; // expected-warning {{writing variable 'x' requires holding mutex 'mu' exclusively}}
    mu.ReaderUnlock();
  }
}

void testExclusiveAssertThenWrite(assert_t a) {
  a();
  x = 1; // ok
}

void testSharedAssertThenWrite(assert_sh_t a) {
  a();
  (void)x; // ok
  x = 1;   // expected-warning {{writing variable 'x' requires holding mutex 'mu' exclusively}}
}
