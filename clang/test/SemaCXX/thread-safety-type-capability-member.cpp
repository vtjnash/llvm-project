// RUN: %clang_cc1 -fsyntax-only -verify=analysis -std=c++17 -Wthread-safety %s
// RUN: %clang_cc1 -fsyntax-only -verify=late -std=c++17 -Wthread-safety \
// RUN:   -Wno-thread-safety-analysis -DTEST_LATE_ARGUMENTS %s

// A capability attribute on a class-member typedef is folded into the
// typedef's type, which means the fold has to run before anything can name
// that typedef. A TypedefType records the canonical type its declaration had
// when the type was created and only reads the sugar back from the
// declaration, so a fold that ran later would leave every member declared with
// the typedef -- and every type built from those -- naming a different type
// than the same typedef used anywhere else.
//
// The thread-safety attributes are late-parsed, precisely so that a member
// function's requirement may name a member declared further down in the class.
// A typedef's requirement is therefore parsed eagerly instead: it sees only
// what precedes it, exactly as an alias declaration's already does, and in
// exchange the typedef means one thing everywhere. See the TEST_LATE_ARGUMENTS
// section at the bottom for what naming a later member does.

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

Mutex mu;

//===----------------------------------------------------------------------===//
// A use inside the class names the same type as a use outside it
//===----------------------------------------------------------------------===//

struct Host {
  typedef void (*cb)(void) REQUIRES(mu);

  // Every one of these is written before the class is complete, which is where
  // the attribute used to be attached.
  cb direct;
  cb *pointer;
  cb array[2];
  void method(cb);
  cb returns();
  typedef cb through_typedef;
};

Host::cb outside;
Host::cb *outside_pointer;

static_assert(!__is_same(Host::cb, void (*)(void)),
              "the requirement must be part of the type");
static_assert(__is_same(decltype(Host::direct), Host::cb),
              "a member declared with the typedef has the typedef's type");
static_assert(__is_same(decltype(Host::pointer), Host::cb *),
              "and so does a type built from it");
static_assert(__is_same(decltype(Host::array), Host::cb[2]), "array of it");
static_assert(__is_same(decltype(&Host::method), void (Host::*)(Host::cb)),
              "a signature written with it");
static_assert(__is_same(decltype(&Host::returns), Host::cb (Host::*)()),
              "a return type written with it");
static_assert(__is_same(Host::through_typedef, Host::cb),
              "a typedef of it");
static_assert(__is_same(decltype(Host::direct), decltype(outside)),
              "inside and outside the class agree");
static_assert(__is_same(decltype(Host::pointer), decltype(outside_pointer)),
              "and so do types built from them");

// The same identity question the linker asks: these two declare one function,
// not an overload set, so they cannot end up as two entities sharing a mangled
// name. If they were distinct, the call below would be ambiguous.
void mangled(decltype(Host::direct));
void mangled(Host::cb) {}
void call_mangled() { mangled(nullptr); }

//===----------------------------------------------------------------------===//
// The analysis sees the requirement through a member declared before the
// class was complete
//===----------------------------------------------------------------------===//

struct Checked {
  typedef void (*cb)(void) REQUIRES(mu);

  cb field;

  void call_param(cb f) {
    f(); // analysis-warning {{calling function 'f' requires holding mutex 'mu' exclusively}}
  }
  void call_field() {
    field(); // analysis-warning {{calling function 'field' requires holding mutex 'mu' exclusively}}
  }
};

void call_outside(Checked::cb f) {
  f(); // analysis-warning {{calling function 'f' requires holding mutex 'mu' exclusively}}
}

void call_member(Checked *c) {
  c->field(); // analysis-warning {{calling function 'field' requires holding mutex 'mu' exclusively}}
}

//===----------------------------------------------------------------------===//
// An alias declaration behaves the same way
//===----------------------------------------------------------------------===//

struct AliasHost {
  using cb REQUIRES(mu) = void (*)(void);
  cb direct;
  cb *pointer;
};

static_assert(__is_same(decltype(AliasHost::direct), AliasHost::cb), "alias");
static_assert(__is_same(decltype(AliasHost::pointer), AliasHost::cb *),
              "alias, derived type");

//===----------------------------------------------------------------------===//
// Member typedefs of a class template
//===----------------------------------------------------------------------===//

// A member typedef whose requirement does not depend on the template
// parameters is folded in the pattern, before the pattern's other members are
// parsed, so the instantiation inherits an already-consistent type.
template <class T> struct Templated {
  typedef void (*cb)(void) REQUIRES(mu);
  cb direct;
  T other;
};

static_assert(!__is_same(Templated<int>::cb, void (*)(void)),
              "folded in the pattern");
static_assert(__is_same(decltype(Templated<int>::direct), Templated<int>::cb),
              "instantiated member declared with the typedef");

void call_instantiated(Templated<int>::cb f) {
  f(); // analysis-warning {{calling function 'f' requires holding mutex 'mu' exclusively}}
}

// A dependent requirement, by contrast, can only be folded once it has been
// substituted, i.e. into the instantiated typedef -- which is too late for a
// member of the same instantiation, because that member's type was substituted
// from the pattern's (unfolded, and not instantiation-dependent) TypedefType
// and so never reached the instantiated typedef at all. This is the same
// mismatch F4 is about, arriving by a different route; see the FIXME(F3) in
// thread-safety-type-capability-templates.cpp, which is what has to be fixed
// for the two to agree.
template <Mutex *M> struct Dependent {
  typedef void (*cb)(void) REQUIRES(*M);
  cb direct;
};

Mutex mu2;
static_assert(!__is_same(Dependent<&mu>::cb, Dependent<&mu2>::cb),
              "different template arguments give different types");
// FIXME(F3): these should be the same type.
static_assert(__is_same(decltype(Dependent<&mu>::direct), void (*)(void)),
              "FIXME: the member kept the pattern's unfolded typedef");

void call_dependent(Dependent<&mu>::cb f) {
  f(); // analysis-warning {{calling function 'f' requires holding mutex 'mu' exclusively}}
}

//===----------------------------------------------------------------------===//
// An argument that cannot become part of the type
//===----------------------------------------------------------------------===//

struct Unfoldable {
  Mutex m;
  // Resolving this needs an object, which a type cannot carry, so the
  // attribute stays on the declaration, where nothing reads it. The typedef
  // still means one thing everywhere, which is what matters here.
  typedef void (*cb)(void) REQUIRES(m);
  cb direct;

  void use(cb f) { f(); } // no warning
};

static_assert(__is_same(Unfoldable::cb, void (*)(void)),
              "an unfolded typedef is just its underlying type");
static_assert(__is_same(decltype(Unfoldable::direct), Unfoldable::cb),
              "and it is that same type everywhere");

//===----------------------------------------------------------------------===//
// The attribute in the declaration-specifier position
//===----------------------------------------------------------------------===//

// Written before the declarator there is no declarator to tell the parser that
// this declaration is a typedef, so the attribute is still late-parsed and the
// fold is refused rather than performed too late. The requirement is lost --
// but the typedef, again, means one thing everywhere.
struct DeclSpecPosition {
  REQUIRES(mu) typedef void (*cb)(void);
  cb direct;

  void use(cb f) { f(); } // no warning
};

static_assert(__is_same(DeclSpecPosition::cb, void (*)(void)),
              "the declaration-specifier position is not folded");
static_assert(__is_same(decltype(DeclSpecPosition::direct),
                        DeclSpecPosition::cb),
              "but it is consistently not folded");

// With nothing to hand the type out early, the late fold is safe and happens.
struct DeclSpecPositionUnused {
  REQUIRES(mu) typedef void (*cb)(void);
};

static_assert(!__is_same(DeclSpecPositionUnused::cb, void (*)(void)),
              "nothing named it before the end of the class");

//===----------------------------------------------------------------------===//
// Arguments naming a member declared later
//===----------------------------------------------------------------------===//

#ifdef TEST_LATE_ARGUMENTS
struct Later {
  // late-error@+1 {{use of undeclared identifier 'static_mu'}}
  typedef void (*static_cb)(void) REQUIRES(static_mu);
  static Mutex static_mu;

  // late-error@+1 {{use of undeclared identifier 'member_mu'}}
  typedef void (*member_cb)(void) REQUIRES(member_mu);
  Mutex member_mu;

  // An alias declaration has always behaved this way.
  // late-error@+1 {{use of undeclared identifier 'alias_mu'}}
  using alias_cb REQUIRES(alias_mu) = void (*)(void);
  static Mutex alias_mu;

  // A member function's requirement is still late-parsed, so it may name any
  // member of the class.
  void ok() REQUIRES(static_mu) {}
  void ok_member() REQUIRES(member_mu) {}
};
#endif
