// Two function types that differ only in their thread-safety capability
// requirements are distinct types, but the Itanium mangler does not emit the
// requirements, so they collide. That is exactly what 'noreturn' and function
// effects do today; -fduplicate-mangled-name says what to do about it, and
// -fmangle-capability-requirements opts into making them distinct symbols.

// RUN: not %clang_cc1 -emit-llvm -std=c++20 -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ERROR
// RUN: %clang_cc1 -emit-llvm -std=c++20 -fduplicate-mangled-name=warn \
// RUN:     -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: %clang_cc1 -emit-llvm -std=c++20 -fduplicate-mangled-name=ignore \
// RUN:     -o - %s 2>&1 | FileCheck %s --check-prefix=IGNORE
// RUN: %clang_cc1 -emit-llvm -std=c++20 -fmangle-capability-requirements \
// RUN:     -o - %s | FileCheck %s --check-prefix=MANGLE

struct __attribute__((capability("mutex"))) M;
extern M *mu;
#define REQ __attribute__((requires_capability(mu)))

typedef void (*plain_t)(void);
typedef void (*caps_t)(void) REQ;

template <class T> void g(T) {}
template void g<plain_t>(plain_t);
template void g<caps_t>(caps_t);

// ERROR: error: definition with same mangled name '_Z1gIPFvvEEvT_'
// WARN: warning: definition with same mangled name '_Z1gIPFvvEEvT_' as another definition; keeping the first

// IGNORE-NOT: error:
// IGNORE-NOT: warning:
// IGNORE: define {{.*}}@_Z1gIPFvvEEvT_

// With the requirements mangled the two are separate symbols, and the
// requirement is spelled independently of which synonym was written.
// MANGLE-DAG: define {{.*}}@_Z1gIPFvvEEvT_
// MANGLE-DAG: define {{.*}}@_Z1gIPU13__tsa_req0_muFvvEEvT_

// The encoding is spelling-independent: 'exclusive_locks_required' states the
// same requirement as 'requires_capability' and must mangle identically, or
// two translation units that spelled it differently would not link.
typedef void (*caps_syn_t)(void) __attribute__((exclusive_locks_required(mu)));
void h(caps_syn_t) {}
// MANGLE: define {{.*}}@_Z1hPU13__tsa_req0_muFvvE

// Sharedness is part of the requirement, so it is part of the encoding.
typedef void (*caps_sh_t)(void) __attribute__((requires_shared_capability(mu)));
void i(caps_sh_t) {}
// MANGLE: define {{.*}}@_Z1iPU13__tsa_req1_muFvvE

