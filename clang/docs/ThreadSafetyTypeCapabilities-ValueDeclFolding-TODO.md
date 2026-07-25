# Follow-up: fold capability requirements into the type for value declarations

## Context

The `[TSA][1..17/N]` commit series makes thread-safety capability requirements
(`requires_capability`, `acquire_capability`, `release_capability`,
`try_acquire_capability`, `assert_capability`, `locks_excluded`, and their
shared variants) part of the **function type** when they are written on a
**function-pointer typedef or alias declaration**. The requirement is stored in
`FunctionType::FunctionTypeExtraAttributeInfo::CapabilityAttrs`, is part of the
canonical type (in `FunctionProtoType::Profile`), round-trips through
serialization, is hashed by `ODRHash`, compared by structural equivalence,
imported by `ASTImporter`, merged by `ASTContext::mergeFunctionTypes`, and is
read back by the analysis from the callee expression's type. This lets the
requirement survive `auto`, template instantiation, and same-type assignment.

The fold is deliberately **limited to typedefs and alias declarations** (see
`Sema::foldCapabilityAttrsIntoType` in `clang/lib/Sema/SemaDeclAttr.cpp`).
This document records what remains to be done to also fold the requirement into
the type of **value declarations** — function-pointer *variables* and *fields*
— and why *parameters* and *function declarations* stay on the
declaration-attribute path introduced by llvm/llvm-project#191187.

It is a work list, not a design record. The design questions, the branch review,
and the status of every finding live in
`ThreadSafetyTypeCapabilities-ReviewFindings.md`; the open canonical-vs-sugar
decision is its Part IV, and the study this list is derived from is its
Part III.

## Why it was not done with the typedef work

The analysis reads capability attributes from the **callee declaration** in
more than one place. For a *typedef* the callee at a call site is a *different*
declaration (the variable/parameter/field of the typedef type), which never
carried the attribute, so `foldCapabilityAttrsIntoType` can drop the attributes
from the typedef. For a *value declaration that is itself the callee*, dropping
would hide the attribute from any analysis path that still reads the
declaration directly — so value-decl folding must **keep** the attributes on
the declaration and rely on de-duplication instead (see below).

### Audit of call-site attribute reads (`clang/lib/Analysis/ThreadSafety.cpp`)

Re-verified after `[TSA][13/N]`; complete.

- `BuildLockset::handleCall` (`:2239`): builds its attribute list from
  `D->attrs()` (`:2348`) **and** `getTypeCapabilityAttrs(Exp, D)` (`:2346`),
  which reads the *callee expression's* type and falls back to the
  declaration's. **Already reads the type**, and already de-duplicates a
  requirement that appears in both, via `areEquivalentCapabilityAttrs`. Covers
  `requires`/`acquire`/`release`/`assert`/`locks_excluded`.
- Try-acquire: `getTerminatorTrylockCall`, `getEdgeLockset`,
  `getTerminatorTrylockCaps` — go through `getTryAcquireCapabilityAttrs`
  (`:135`), which reads the declaration **and** the type, with the same dedup.
  **Already reads the type** (added in `[TSA][7/N]`).
- `runAnalysis` (`:2839`): reads the **analyzed function's own** attributes
  (`D->attrs()`, `:2903`) and its parameters' attributes (`Param->attrs()`,
  `:2938`) to seed the entry lockset. This concerns the function *being
  analyzed*, not a callee, so it only matters if we fold **function
  declarations** (see item 3 below).
- Scoped-lockable handling in `handleCall` (`:2366`): reads the *called
  function's* `Param->attrs()`. Since `f2fc9cc59cf8` both this loop and the
  `runAnalysis` one skip function-pointer parameters outright
  (`isFunctionPointerParam`, `:72`), so both are already inert for the
  parameters this work would otherwise touch.

**Nothing outside `clang/lib/Analysis` and `clang/lib/Sema` reads these
attributes** — in particular nothing in `clang-tools-extra` does — so folding
them into the type breaks no tooling. Verified as part of the Part III study.

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
explains the observed failures — no unexplained residue remains, and this is no
longer a blocker.

## Remaining work items

1. **Fold `FieldDecl`** (lowest risk, ships first). Timing verified safe:
   late-parsed member attrs attach (`ParseDeclCXX.cpp:3725`) after
   `ActOnFinishCXXMemberSpecification` but *before*
   `ParseLexedMemberInitializers` (`:3731`); triviality and layout are
   unaffected (pointer size and alignment are unchanged, and `TypeInfo` is keyed
   on the canonical `Type*`). In C, thread-safety field attributes are not
   late-parsed at all — the struct-body path asks for
   `LateAttrParseExperimentalExtOnly` (`ParseDecl.cpp:4989`) and these are
   `LateAttrParseStandard` — and the non-late path
   (`Sema::CheckFieldDecl`, `SemaDecl.cpp:19635`) precedes record completion.
   A `FieldDecl` is never
   redeclared, so it is immune to the merge problem in item 2.

2. **Fold `VarDecl`**, excluding `ParmVarDecl`. `ProcessDeclAttributes` at
   `SemaDecl.cpp:8277` precedes merge and initializer checking for namespace and
   block scope.
   - Prerequisite, half-done: redeclaring a folded variable without the
     attribute. In C this is already handled — `ASTContext::mergeFunctionTypes`
     unions the requirement sets as of `[TSA][14/N]` (see
     `clang/test/Sema/thread-safety-type-capability-merge.c`). In C++
     `Sema::MergeVarDeclTypes` still compares with a bare `hasSameType`
     (`SemaDecl.cpp:4604`) and would reject the attr-free redeclaration with
     `redeclaration with a different type`; the fix is to try
     `mergeCapabilityAttrs` (the union helper added in `[TSA][14/N]`, declared
     in `Attr.h`) before diagnosing. It is untestable until there is a folded
     `VarDecl` to redeclare, which is why it lives here rather than having
     landed with the merge work.
   - Residual hazard: an in-class static data member with an initializer — the
     initializer is checked at `ParseDeclCXX.cpp:3183-3192`, before late
     attributes attach at `:3725`. Exclude static data members in the first cut,
     or test the case explicitly.

3. **Function declarations: still a no-go.** Override matching is safe, but
   redeclaration is not (`MergeFunctionDecl` → `err_conflicting_types`), the
   entry-lockset seeding in `runAnalysis` (`ThreadSafety.cpp:2903`/`:2938`) would
   go blind unless it learns to read the function's *own* type, and
   `checkThisInStaticMemberFunctionAttributes` ordering has to be preserved.
   Revisit only after item 2 lands, with a union-merge policy and tests.

   Name mangling is **not** a reason to feel safe here — it is the hazard.
   `CapabilityAttrs` are part of the canonical type but the Itanium mangler
   ignores `FunctionTypeExtraAttributeInfo`, so two functions whose types differ
   only in their requirements mangle identically: two definitions are a
   hard `error: definition with same mangled name`, and declaration-only
   overload sets are a silent link trap. This already applies to a
   caps-carrying *typedef* used as a parameter type, and folding function
   declarations would extend it to the functions themselves. See finding F2 and
   the Option A / Option B decision in Part IV of
   `ThreadSafetyTypeCapabilities-ReviewFindings.md`; that decision should be
   made before this item is attempted.

4. **Parameters: skip.** Not for build-order reasons —
   `ActOnParamDeclarator` runs `ProcessDeclAttributes` (`SemaDecl.cpp:15858`)
   *before* `GetFullTypeForDeclarator` collects the parameter types
   (`SemaType.cpp:5265`), so a folded parameter *would* propagate into the
   enclosing prototype. That is precisely the problem: it would change the
   enclosing function's own type, and hence its overload identity
   (`SemaOverload.cpp:1373`) and its mangling. The declaration path already
   covers parameters, and the typedef path covers the common case of a
   parameter written with an annotated type.

5. **Do not drop the attributes from the declaration.** The typedef fold drops
   them (a typedef is never itself the callee, and `TypePrinter` emits them from
   the type), but a value declaration *is* the callee, and dropping would break
   redeclaration attribute inheritance, `-ast-dump`, and any later
   function-declaration folding. Keep them and rely on the de-duplication that
   `[TSA][13/N]` added — note that it is **not** the pointer-identity dedup an
   earlier draft of this document suggested: the fold and the declaration hold
   distinct `Attr*` objects, so `clang::areEquivalentCapabilityAttrs` compares
   them by the same profile the folding set uses. Two requirements are redundant
   exactly when they would produce the same function type, which makes the dedup
   spelling-insensitive but sharedness-, genericness- and
   success-value–sensitive.

6. **Set only `ValueDecl::setType`**, and leave the `TypeSourceInfo` alone.
   Divergence between a declaration's type and its TSI is an accepted pattern
   (cf. `ParmVarDecl` decay), whereas the typedef path's
   `getTrivialTypeSourceInfo` replacement loses the written sugar. Read the
   declaration's type rather than `OldTSI->getType()`; for value declarations
   the two genuinely differ, e.g. after address-space adjustment.
   Note that `TypedefNameDecl` is **not** a `DeclaratorDecl`, so the obvious
   unified refactor of the two paths null-derefs — keep them separate.

7. **Tests**: `requires_shared_capability` on a *variable* plus `auto`; the
   exclusive variable auto/keep/drop trio mirroring the typedef tests; the five
   explicit Profile-collision pairs and the try-acquire success-value pair as
   variables; redeclaration without the attribute (C++ compiles and keeps the
   requirement, plus the C twin); a dependent-argument class template (no crash,
   sane capability name); the static-data-member initializer hazard; an NSDMI
   field; a field reached through `auto` must warn; a class-template field with
   a global argument and with a sibling-member argument; PCH and `-ast-print`
   coverage for a folded variable and field; the `FPOps`/`BDevOps` C structs
   must stay unfolded; the whole `FunctionPointers` namespace unchanged; and
   `Sema`/`SemaCXX`/`AST`/`PCH`/`Modules` sweeps.

## Recommended order

Item 1 (fields) first, since fields cannot be redeclared and therefore need
none of item 2's merge work. Then item 2 (variables) together with the
`MergeVarDeclTypes` union. Items 5 and 6 are constraints on how 1 and 2 are
implemented, not separate steps. Item 3 stays a separate, later effort with its
own design review — and should wait on the Part IV decision, since
function-type identity changes have the widest blast radius.
