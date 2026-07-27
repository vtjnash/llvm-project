// Thread-safety capability attributes folded into a function type are part of
// the canonical type, so ODRHash has to hash them: two modules that define the
// same class with different requirements must not merge silently.
//
// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps \
// RUN:            -I%S/Inputs/thread-safety-type-capability -std=c++11 \
// RUN:            -Wthread-safety -verify %s

#include "a.h"
#include "b.h"

// Identical requirements, and differently spelled synonyms for the same
// requirement, merge without a diagnostic.
Same same;
Synonym synonym;

DiffMutex diff_mutex;
// expected-error@b.h:* {{'DiffMutex::cb' from module 'b' is not present in definition of 'DiffMutex' in module 'a'}}
// expected-note@a.h:* {{declaration of 'cb' does not match}}
// expected-error@b.h:* {{'DiffMutex::m' from module 'b' is not present in definition of 'DiffMutex' in module 'a'}}
// expected-note@a.h:* {{declaration of 'm' does not match}}

DiffPresence diff_presence;
// expected-error@b.h:* {{'DiffPresence::cb' from module 'b' is not present in definition of 'DiffPresence' in module 'a'}}
// expected-note@a.h:* {{declaration of 'cb' does not match}}
// expected-error@b.h:* {{'DiffPresence::m' from module 'b' is not present in definition of 'DiffPresence' in module 'a'}}
// expected-note@a.h:* {{declaration of 'm' does not match}}

// The success value is compared by the value it denotes, so 'true' and '1' are
// one requirement and merge; '1' and '2' are not and do not.
TryValue try_value;

DiffTryValue diff_try_value;
// expected-error@b.h:* {{'DiffTryValue::cb' from module 'b' is not present in definition of 'DiffTryValue' in module 'a'}}
// expected-note@a.h:* {{declaration of 'cb' does not match}}
// expected-error@b.h:* {{'DiffTryValue::m' from module 'b' is not present in definition of 'DiffTryValue' in module 'a'}}
// expected-note@a.h:* {{declaration of 'm' does not match}}

DiffSharedness diff_sharedness;
// expected-error@b.h:* {{'DiffSharedness::cb' from module 'b' is not present in definition of 'DiffSharedness' in module 'a'}}
// expected-note@a.h:* {{declaration of 'cb' does not match}}
// expected-error@b.h:* {{'DiffSharedness::m' from module 'b' is not present in definition of 'DiffSharedness' in module 'a'}}
// expected-note@a.h:* {{declaration of 'm' does not match}}
