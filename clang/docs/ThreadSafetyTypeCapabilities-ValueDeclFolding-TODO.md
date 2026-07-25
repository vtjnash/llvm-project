# Follow-up: fold capability requirements into the type for value declarations

## Context

The `[TSA][1..8/N]` commit series makes thread-safety capability requirements
(`requires_capability`, `acquire_capability`, `release_capability`,
`try_acquire_capability`, `assert_capability`, `locks_excluded`, and their
shared variants) part of the **function type** when they are written on a
**function-pointer typedef**. The requirement is stored in
`FunctionType::FunctionTypeExtraAttributeInfo::CapabilityAttrs`, is part of the
canonical type (in `FunctionProtoType::Profile`), round-trips through
serialization, and is read back by the analysis from the callee's type. This
lets the requirement survive `auto`, template instantiation, and same-type
assignment.

The fold is deliberately **limited to typedefs** (see
`Sema::foldCapabilityAttrsIntoType` in `clang/lib/Sema/SemaDeclAttr.cpp`).
This document records what would be needed to also fold the requirement into
the type of **value declarations** — function-pointer *variables*, *fields*,
*parameters*, and (separately) *function declarations* — so those too gain the
"part of the type" behavior instead of relying on the declaration-attribute
path introduced by llvm/llvm-project#191187.

## Why it is not done yet

The analysis reads capability attributes from the **callee declaration** in
more than one place. `foldCapabilityAttrsIntoType` moves the attributes onto
the type and then **drops them from the declaration** (so they are not printed
or processed twice). For a *typedef* that is safe, because the callee at a call
site is a *different* declaration (the variable/parameter/field of the typedef
type), which never carried the attribute. For a *value declaration that is
itself the callee*, dropping the attribute hides it from any analysis path that
still reads the declaration directly.

### Audit of call-site attribute reads (`clang/lib/Analysis/ThreadSafety.cpp`)

- `BuildLockset::handleCall` (~line 2197): builds its attribute list from
  `D->attrs()` **and** `getCalleeFunctionProtoType(D)->getCapabilityAttrs()`.
  **Already reads the type.** Covers `requires`/`acquire`/`release`/`assert`/
  `locks_excluded`.
- Try-acquire: `getTerminatorTrylockCall`, `getEdgeLockset`,
  `getTerminatorTrylockCaps` — go through the `getTryAcquireCapabilityAttrs`
  helper, which reads the declaration **and** the function type.
  **Already reads the type** (added in `[TSA][7/N]`).
- `runAnalysis` (~lines 2822 and 2857): reads the **analyzed function's own**
  attributes (`D->attrs()`) and its parameters' attributes (`Param->attrs()`)
  to seed the entry lockset. This concerns the function *being analyzed*, not a
  callee, so it only matters if we fold **function declarations** (see below).
- Scoped-lockable / function-pointer-parameter handling in `handleCall`
  (~lines 2281–2287): reads the *called function's* `Param->attrs()`. Relevant
  only when folding parameters of the called function.

So the two callee-facing paths (`handleCall` and try-acquire) are already
type-aware. That was not true when value-decl folding was first attempted; the
try-acquire fix in `[TSA][7/N]` removed one of the two original blockers.

### Resolved: the shared-variant regression was a folding-set collision

When value-decl folding was briefly enabled, the pre-existing function-pointer
*variable* tests in `clang/test/SemaCXX/warn-thread-safety-analysis.cpp`
(namespace `FunctionPointers`) regressed: the **shared** variants behaved as if
exclusive (`shared_lock_fn()` stopped leaving `mu` held as shared;
`shared_requires_fn()` warned under a reader lock).

Root cause: `FunctionType::FunctionTypeExtraAttributeInfo::Profile` hashed only
each attribute's `attr::Kind` plus its `getCapabilityAttrArgs`. Sharedness and
release genericness are spelling-derived, and try-acquire's success value is a
separate argument, so `ACQUIRE(mu)` and `ACQUIRE_SHARED(mu)` produced identical
`FoldingSetNodeID`s: `ASTContext::getFunctionTypeInternal` handed the second
declaration the first one's uniqued `FunctionProtoType`, carrying the *first*
declaration's `Attr*` objects. **First created wins.** In `FunctionPointers`
every shared variant is declared immediately after its exclusive twin with an
identical signature and the same `mu`, which is exactly why the reduced
single-declaration repro did not reproduce it.

This was a live bug for plain typedefs too, independent of value-decl folding.
Fixed in `[TSA][9/N]` by profiling `isShared()` / `isGeneric()` / the
try-acquire success value (semantically, not via the spelling index, so
synonyms still unify), plus a per-attribute argument-count separator; see
`clang/test/SemaCXX/thread-safety-type-capability-uniquing.cpp`. It fully
explains the observed failures — no unexplained residue remains.

## Work items to complete the follow-up

1. ~~**Root-cause the shared-variant regression**~~ — **done**, see above; the
   fix landed in `[TSA][9/N]`. Value-decl folding still needs a
   `requires_shared_capability` **variable** test when it is re-enabled.

2. **Extend `foldCapabilityAttrsIntoType`** to accept `VarDecl` and `FieldDecl`
   of function-pointer type. A value declaration stores its type separately
   from its `TypeSourceInfo`, so set **both** `DeclaratorDecl::setTypeSourceInfo`
   and `ValueDecl::setType`. Keep the existing `capabilityArgIsContextFree`
   guard (member-relative arguments must stay on the declaration) and the
   late-parsed hook in `ActOnFinishDelayedAttribute`.

3. **Parameters** desync from the enclosing function type, which is built from
   the parameter types *before* `ProcessDeclAttributes` runs on the
   `ParmVarDecl`. Either rebuild the enclosing `FunctionProtoType` after folding
   the parameter, or leave parameters on the declaration path. Simplest first
   step: do **not** fold parameters (the analysis already reads them, and the
   typedef path covers the common case).

4. **Function declarations** are the riskiest. Folding changes the function's
   own type identity, which affects:
   - redeclaration merging — `ASTContext::mergeFunctionTypes` currently ignores
     `CapabilityAttrs` (they don't make types incompatible, but a merged type
     silently keeps one side's set); decide union vs. keep-first and implement.
   - virtual override checking — an override with different requirements.
   - the entry-lockset seeding in `runAnalysis` (lines ~2822/2857) must read the
     requirements from the function's **type** if they no longer live on the
     decl.
   Name **mangling is already safe**: `CapabilityAttrs` live in
   `FunctionTypeExtraAttributeInfo`, which the Itanium mangler does not emit
   (verified: a function taking a caps-carrying vs. plain function-pointer
   parameter mangles identically, `...PFvvE`). So folding does not change ABI.

5. **Dropping attributes from the declaration**: confirm nothing outside the
   analysis depends on these attributes being queryable on the value decl
   (AST matchers, clang-tidy checks, tooling). If something does, consider
   *keeping* the attribute on the decl and instead **de-duplicating by pointer**
   in every reader (fold reuses the same `Attr*` objects, so a `SmallPtrSet`
   dedup across `D->attrs()` + type attrs avoids double-processing without
   dropping).

6. **Tests**: once enabled, add `auto`/propagation coverage for variables and
   fields (mirroring the typedef tests), a `requires_shared` variable test, and
   re-verify the entire `FunctionPointers` namespace plus the `Sema`/`SemaCXX`/
   `AST`/`PCH`/`Modules` sweeps show no regressions.

## Recommended order

Do (1) first — it is diagnostic and may reveal a small bug that unblocks the
rest. Then (2) for variables and fields only, keeping parameters and functions
on the declaration path. Treat (4) as a separate, later effort with its own
design review, since function-type identity changes have the widest blast
radius.
