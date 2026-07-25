// Thread-safety capability attributes folded into a function type are part of
// the canonical type, so the ASTImporter has to import them and
// ASTStructuralEquivalence has to compare them.
//
// RUN: %clang_cc1 -emit-pch -o %t.1.ast %S/Inputs/caps1.c
// RUN: %clang_cc1 -emit-pch -o %t.2.ast %S/Inputs/caps2.c
// RUN: %clang_cc1 -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only %s 2>&1 \
// RUN:   | FileCheck --check-prefix=MERGE %s
// RUN: %clang_cc1 -ast-merge %t.1.ast -ast-dump %s | FileCheck --check-prefix=IMPORT %s

// The imported typedef and the variables using it keep their requirement
// instead of being silently stripped.
//
// IMPORT: TypedefDecl {{.*}} cb1 'void (*)(void) __attribute__((requires_capability(mu1)))'
// IMPORT: VarDecl {{.*}} same 'cb1':'void (*)(void) __attribute__((requires_capability(mu1)))'

// 'same' carries the same requirement on both sides -- spelled with a synonym
// on one of them -- and merges silently. The other two do not.
//
// MERGE-NOT: 'same'
// MERGE: caps2.c:11:5: warning: external variable 'differ_mutex' declared with incompatible types in different translation units ('cb2' (aka 'void (*)(void) __attribute__((requires_capability(mu2)))') vs. 'cb1' (aka 'void (*)(void) __attribute__((requires_capability(mu1)))'))
// MERGE: caps1.c:7:5: note: declared here with type 'cb1' (aka 'void (*)(void) __attribute__((requires_capability(mu1)))')
// MERGE: caps2.c:12:7: warning: external variable 'differ_presence' declared with incompatible types in different translation units ('plain' (aka 'void (*)(void)') vs. 'cb1' (aka 'void (*)(void) __attribute__((requires_capability(mu1)))'))
// MERGE: caps1.c:8:5: note: declared here with type 'cb1' (aka 'void (*)(void) __attribute__((requires_capability(mu1)))')
// MERGE-NOT: 'same'
// MERGE: 2 warnings generated
