// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 -Wthread-safety %s

// A capability attribute on a function-pointer typedef is folded into the
// typedef's type. Inside a template that can only happen once the attribute's
// arguments are known: an argument that is still dependent must NOT be folded,
// because the rebuilt type is uniqued in the ASTContext before substitution
// and every instantiation would then share -- and be checked against -- the
// pattern's un-substituted argument. Such attributes stay on the declaration
// and are folded again on the instantiated typedef, where the arguments have
// been substituted.

#define LOCKABLE __attribute__((lockable))
#define EXCLUSIVE_LOCK_FUNCTION(...) \
  __attribute__((exclusive_lock_function(__VA_ARGS__)))
#define UNLOCK_FUNCTION(...) __attribute__((unlock_function(__VA_ARGS__)))
#define REQUIRES(...) __attribute__((exclusive_locks_required(__VA_ARGS__)))
#define REQUIRES_SHARED(...) __attribute__((shared_locks_required(__VA_ARGS__)))
#define GUARDED_BY(x) __attribute__((guarded_by(x)))

class LOCKABLE Mutex {
public:
  void Lock() EXCLUSIVE_LOCK_FUNCTION();
  void ReaderLock() __attribute__((acquire_shared_capability()));
  void Unlock() UNLOCK_FUNCTION();
};

Mutex mu1;
Mutex mu2;
int x GUARDED_BY(mu1);

//===----------------------------------------------------------------------===//
// A mutex named by a non-type template parameter
//===----------------------------------------------------------------------===//

// '*M' is dependent, so nothing is folded in the pattern -- in particular the
// analysis must not end up checking calls against a capability literally named
// 'M'. The fold happens per instantiation, against the substituted mutex.
template <Mutex *M> struct NTTP {
  typedef void (*cb)(void) REQUIRES(*M);
};

// Each instantiation substitutes a different mutex, so the folded types are
// distinct -- this exercises substitution and the type's Profile end to end.
static_assert(!__is_same(NTTP<&mu1>::cb, NTTP<&mu2>::cb),
              "different template arguments must give different types");
static_assert(!__is_same(NTTP<&mu1>::cb, void (*)(void)),
              "the requirement must be part of the instantiated type");
static_assert(__is_same(NTTP<&mu1>::cb, NTTP<&mu1>::cb),
              "the same template argument must give the same type");

void nttp_unlocked(NTTP<&mu1>::cb f) {
  f(); // expected-warning {{calling function 'f' requires holding mutex 'mu1' exclusively}}
}

void nttp_locked(NTTP<&mu1>::cb f) {
  mu1.Lock();
  f(); // no warning: the substituted mutex is held
  mu1.Unlock();
}

// The substituted mutex is the one that matters, not some other one.
void nttp_wrong_mutex(NTTP<&mu2>::cb f) {
  mu1.Lock();
  f(); // expected-warning {{calling function 'f' requires holding mutex 'mu2' exclusively}}
  mu1.Unlock();
}

// A capability named directly by a non-type template parameter resolves to the
// argument it was substituted with, too.
template <Mutex *M> void nttp_direct() REQUIRES(*M) {
  x = 1;
}
void call_nttp_direct() {
  mu1.Lock();
  nttp_direct<&mu1>();
  mu1.Unlock();
  nttp_direct<&mu1>(); // expected-warning {{calling function 'nttp_direct<&mu1>' requires holding mutex 'mu1' exclusively}}
}

//===----------------------------------------------------------------------===//
// A mutex reached through a dependent base / dependent qualified name
//===----------------------------------------------------------------------===//

struct Base {
  static Mutex smu;
};

// 'T::smu' parses as a DependentScopeDeclRefExpr. Folding it would bake an
// unresolvable expression into the type; the analysis used to report
// 'cannot resolve lock expression' at every call.
template <class T> struct DepBase : T {
  typedef void (*cb)(void) REQUIRES(T::smu);
};

void depbase_unlocked(DepBase<Base>::cb f) {
  f(); // expected-warning {{calling function 'f' requires holding mutex 'smu' exclusively}}
}

void depbase_locked(DepBase<Base>::cb f) {
  Base::smu.Lock();
  f(); // no warning
  Base::smu.Unlock();
}

// Sharedness survives substitution as well.
template <class T> struct DepBaseShared : T {
  typedef void (*cb)(void) REQUIRES_SHARED(T::smu);
};

void depbase_shared(DepBaseShared<Base>::cb f) {
  Base::smu.ReaderLock();
  f(); // no warning
  Base::smu.Unlock();
}

void depbase_shared_unlocked(DepBaseShared<Base>::cb f) {
  f(); // expected-warning {{calling function 'f' requires holding mutex 'smu'}}
}

static_assert(!__is_same(DepBase<Base>::cb, DepBaseShared<Base>::cb),
              "sharedness must survive substitution");

//===----------------------------------------------------------------------===//
// A dependent underlying type
//===----------------------------------------------------------------------===//

// The subject check cannot know yet whether 'T' is a function pointer, so it
// must accept the attribute here and recheck after substitution.
template <class T> struct DepType {
  typedef T cb REQUIRES(mu1); // no warning at parse time: 'T' may well be a function pointer
  static void run(cb f) { f(); } // expected-warning {{calling function 'f' requires holding mutex 'mu1' exclusively}}
};

// Instantiating with a function pointer folds and checks as usual...
template struct DepType<void (*)(void)>; // expected-note {{in instantiation of member function 'DepType<void (*)()>::run' requested here}}

// ... instantiating with anything else produces the deferred subject warning.
template <class T> struct DepTypeDecl {
  typedef T cb REQUIRES(mu1); // expected-warning {{'exclusive_locks_required' attribute on a typedef requires the typedef to be of function pointer type}}
};
DepTypeDecl<int>::cb not_a_function_pointer; // expected-note {{in instantiation of template class 'DepTypeDecl<int>' requested here}}

//===----------------------------------------------------------------------===//
// A non-dependent argument inside a template
//===----------------------------------------------------------------------===//

// Nothing is dependent about 'mu2', so the pattern folds at parse time. The
// folded attribute then rides along through substitution -- including when the
// function type itself has to be rebuilt because its return type is dependent.
template <class T> struct NonDep {
  typedef void (*cb)(void) REQUIRES(mu2);
  typedef T (*dep_cb)(void) REQUIRES(mu2);
  static void run(cb f) { f(); } // expected-warning {{calling function 'f' requires holding mutex 'mu2' exclusively}}
  static void run_locked(cb f) {
    mu2.Lock();
    f(); // no warning
    mu2.Unlock();
  }
};
template struct NonDep<int>; // expected-note {{in instantiation of member function 'NonDep<int>::run' requested here}}

static_assert(!__is_same(NonDep<int>::cb, void (*)(void)),
              "a non-dependent requirement folds in the pattern");
static_assert(__is_same(NonDep<int>::cb, NonDep<char>::cb),
              "instantiations share the pattern's folded type");
static_assert(!__is_same(NonDep<int>::dep_cb, int (*)(void)),
              "the folded attribute survives rebuilding the function type");

void nondep_outside(NonDep<int>::dep_cb f) {
  f(); // expected-warning {{calling function 'f' requires holding mutex 'mu2' exclusively}}
}

//===----------------------------------------------------------------------===//
// A block-scope mutex
//===----------------------------------------------------------------------===//

void local_mutex_typedef() {
  Mutex lmu;
  // 'lmu' does not outlive this function, but a folded function type is
  // uniqued in the ASTContext for the whole translation unit, so the attribute
  // cannot become part of the type -- and a typedef's declaration attributes
  // are read by nothing, so it is reported as ignored.
  // expected-warning@+1 {{'exclusive_locks_required' attribute on 'cb' cannot become part of the type it names because the capability does not have global storage; attribute ignored}}
  typedef void (*cb)(void) REQUIRES(lmu);
  static_assert(__is_same(cb, void (*)(void)),
                "a block-scope capability must not be folded into a type");
  cb f = nullptr;
  f(); // no warning: the attribute above was reported as ignored
}

// A static local, by contrast, has global storage and does fold.
void static_local_mutex_typedef() {
  static Mutex smu;
  typedef void (*cb)(void) REQUIRES(smu);
  static_assert(!__is_same(cb, void (*)(void)),
                "a static-storage capability folds");
  cb f = nullptr;
  f(); // expected-warning {{calling function 'f' requires holding mutex 'smu' exclusively}}
}

//===----------------------------------------------------------------------===//
// Known limitation
//===----------------------------------------------------------------------===//

// FIXME(F3): a use of the typedef from *inside* the template names the
// pattern's TypedefType, which is not instantiation-dependent (only the
// attribute's arguments are), so Sema::SubstType leaves it alone and the
// instantiated -- folded -- typedef is never reached. The requirement is
// therefore invisible to calls written inside the template. Fixing this needs
// the dependent attribute to be folded into the pattern's type and substituted
// by TreeTransform, which also requires the type to become
// instantiation-dependent.
template <Mutex *M> struct InsideTemplate {
  typedef void (*cb)(void) REQUIRES(*M);
  static void run(cb f) { f(); } // no warning today; should warn
};
template struct InsideTemplate<&mu1>;
