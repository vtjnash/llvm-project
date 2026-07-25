// RUN: %clang_cc1 -ast-print %s | FileCheck %s

// Capability attributes folded into a function-pointer typedef's type are
// printed as part of the type.

struct __attribute__((capability("mutex"))) Mutex {};
Mutex mu;

typedef void (*req_cb_t)(void) __attribute__((requires_capability(mu)));
// CHECK: typedef void (*req_cb_t)() __attribute__((requires_capability(mu)));

typedef void (*rel_cb_t)(void) __attribute__((release_capability(mu)));
// CHECK: typedef void (*rel_cb_t)() __attribute__((release_capability(mu)));
