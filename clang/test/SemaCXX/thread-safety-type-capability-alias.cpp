// RUN: %clang_cc1 -fsyntax-only -verify=analysis -std=c++17 -Wthread-safety %s
// RUN: %clang_cc1 -fsyntax-only -verify=badpos -std=c++17 -Wthread-safety \
// RUN:   -Wno-thread-safety-analysis -Wno-thread-safety-attributes \
// RUN:   -DTEST_ATTRIBUTE_POSITIONS %s

// A capability attribute on an alias declaration is folded into the alias'
// type exactly like one on a typedef: 'using' is what C++ code actually
// writes, and an attribute that is accepted but ignored is worse than one that
// is rejected.
//
// The attribute has to be written in the same place a declaration attribute
// goes on an alias declaration, i.e. right after the alias name; see the
// TEST_ATTRIBUTE_POSITIONS section at the bottom for what the other positions
// do.

#define LOCKABLE __attribute__((lockable))
#define EXCLUSIVE_LOCK_FUNCTION(...) \
  __attribute__((exclusive_lock_function(__VA_ARGS__)))
#define UNLOCK_FUNCTION(...) __attribute__((unlock_function(__VA_ARGS__)))
#define REQUIRES(...) __attribute__((exclusive_locks_required(__VA_ARGS__)))
#define ACQUIRE(...) __attribute__((acquire_capability(__VA_ARGS__)))
#define RELEASE(...) __attribute__((release_capability(__VA_ARGS__)))

class LOCKABLE Mutex {
public:
  void Lock() EXCLUSIVE_LOCK_FUNCTION();
  void Unlock() UNLOCK_FUNCTION();
};

Mutex mu1;
Mutex mu2;

//===----------------------------------------------------------------------===//
// The requirement becomes part of the type
//===----------------------------------------------------------------------===//

using cb REQUIRES(mu1) = void (*)(void);

static_assert(!__is_same(cb, void (*)(void)),
              "the requirement must be part of the type");

// An alias and a typedef spelling the same requirement are the same type.
typedef void (*cb_typedef)(void) REQUIRES(mu1);
static_assert(__is_same(cb, cb_typedef), "alias and typedef must agree");

// Different mutexes are still different types.
using cb2 REQUIRES(mu2) = void (*)(void);
static_assert(!__is_same(cb, cb2), "different mutexes, different types");

void unlocked(cb f) {
  f(); // analysis-warning {{calling function 'f' requires holding mutex 'mu1' exclusively}}
}

void locked(cb f) {
  mu1.Lock();
  f();
  mu1.Unlock();
}

// The requirement travels with the type, not with the declaration it was
// written on.
void through_auto(cb f) {
  auto g = f;
  g(); // analysis-warning {{calling function 'g' requires holding mutex 'mu1' exclusively}}
}

// The C++11 spelling goes in the same position.
using cb_cxx11 [[clang::requires_capability(mu1)]] = void (*)(void);
static_assert(__is_same(cb, cb_cxx11), "spelling must not change the type");

void unlocked_cxx11(cb_cxx11 f) {
  f(); // analysis-warning {{calling function 'f' requires holding mutex 'mu1' exclusively}}
}

//===----------------------------------------------------------------------===//
// Acquiring and releasing through an alias
//===----------------------------------------------------------------------===//

using acquire_cb ACQUIRE(mu1) = void (*)(void);
using release_cb RELEASE(mu1) = void (*)(void);

void acquire_then_release(acquire_cb acq, release_cb rel) {
  acq();
  rel();
}

void acquire_only(acquire_cb acq) {
  acq(); // analysis-note {{mutex acquired here}}
} // analysis-warning {{mutex 'mu1' is still held at the end of function}}

void release_without_holding(release_cb rel) {
  rel(); // analysis-warning {{releasing mutex 'mu1' that was not held}}
}

//===----------------------------------------------------------------------===//
// Aliases in a class
//===----------------------------------------------------------------------===//

// The attribute is late-parsed here, so the fold runs from
// ActOnFinishDelayedAttribute rather than from the alias declaration itself.
struct Host {
  using member_cb REQUIRES(mu1) = void (*)(void);

  Mutex m;
  // An argument naming a member has to be resolved against an object, which a
  // type cannot carry, so this one is not folded and simply is not checked.
  using self_cb REQUIRES(m) = void (*)(void);

  void use(member_cb f) {
    f(); // analysis-warning {{calling function 'f' requires holding mutex 'mu1' exclusively}}
  }

  void use_self(self_cb f) { f(); }
};

void outside(Host::member_cb f) {
  f(); // analysis-warning {{calling function 'f' requires holding mutex 'mu1' exclusively}}
}

//===----------------------------------------------------------------------===//
// Templates
//===----------------------------------------------------------------------===//

// A member alias of a class template is instantiated as a declaration, so the
// fold is retried per instantiation with the substituted argument.
template <Mutex *M> struct Holder {
  using cb REQUIRES(*M) = void (*)(void);
};

static_assert(!__is_same(Holder<&mu1>::cb, Holder<&mu2>::cb),
              "different template arguments must give different types");
// (Holder<&mu1>::cb's argument is the substituted '*&mu1', which is not the
// same expression as the 'mu1' written above, so the two do not unique
// together; the analysis resolves both to the same capability.)

void instantiated(Holder<&mu1>::cb f) {
  f(); // analysis-warning {{calling function 'f' requires holding mutex 'mu1' exclusively}}
}

// An alias template whose underlying type does not depend on the template
// parameters is folded once, in the pattern.
template <class T> using indirect_cb REQUIRES(mu1) = void (*)(void);

void through_alias_template(indirect_cb<int> f) {
  f(); // analysis-warning {{calling function 'f' requires holding mutex 'mu1' exclusively}}
}

// FIXME: An alias template is not instantiated as a declaration -- using it
// just substitutes into the pattern's underlying type -- so when the fold has
// to be deferred (a dependent underlying type, or a dependent capability
// argument) there is nothing that retries it and the requirement is lost.
// Both cases below should warn.
template <class T> using dependent_cb REQUIRES(mu1) = T;

void dependent_underlying_type(dependent_cb<void (*)(void)> f) {
  f(); // no warning
}

template <Mutex *M> using nttp_cb REQUIRES(*M) = void (*)(void);

void dependent_argument(nttp_cb<&mu1> f) {
  f(); // no warning
}

//===----------------------------------------------------------------------===//
// Subject checking
//===----------------------------------------------------------------------===//

// analysis-warning@+1 {{'exclusive_locks_required' attribute on a typedef requires the typedef to be of function pointer type}}
using not_a_function REQUIRES(mu1) = int;

//===----------------------------------------------------------------------===//
// Other attribute positions
//===----------------------------------------------------------------------===//

// A separate run, with the analysis silenced, so that only what these
// positions themselves produce is checked.
#ifdef TEST_ATTRIBUTE_POSITIONS
// The GNU spelling after the type-id is not part of an alias declaration's
// grammar at all.
using trailing_gnu = void (*)(void) __attribute__((requires_capability(mu1)));
// badpos-error@-1 {{expected ';' after alias declaration}}

// The C++11 spelling there appertains to the type, which a thread-safety
// attribute cannot.
using trailing_cxx11 = void (*)(void) [[clang::requires_capability(mu1)]];
// badpos-error@-1 {{'clang::requires_capability' attribute cannot be applied to types}}

// FIXME: A declaration attribute written inside the declarator slides onto the
// declaration for a typedef, but an alias declaration's type-id is parsed
// without a declaration to slide onto, so this one is dropped silently. It
// should either be honored or diagnosed.
using inside_declarator = void (*__attribute__((requires_capability(mu1))))(void);
typedef void (*__attribute__((requires_capability(mu1))) inside_declarator_td)(void);
static_assert(!__is_same(inside_declarator, inside_declarator_td),
              "FIXME: these should be the same type");
#endif
