# TSA type-carried capabilities: review findings and fix plan

Working document for the `jn/tsa-typedef-capability` branch (commits
`f2fc9cc59cf8..cd7e942d5879`). Consolidates three investigations (2026-07-25):
a root-cause of the regression recorded in
`ThreadSafetyTypeCapabilities-ValueDeclFolding-TODO.md`, a full code review of
the branch, and an implementation study for value-declaration folding.
Not intended for upstreaming; convert surviving items to GitHub issues.

Status legend: [ ] open, [x] fixed on branch, [~] documented/deferred.

---

## Part I — Root cause of the "shared variants behaved as if exclusive" regression

**RESOLVED (diagnosed): FoldingSet profile collision.** CONFIRMED empirically.

`FunctionTypeExtraAttributeInfo::Profile` (`clang/lib/AST/Type.cpp:4133`)
hashes only `A->getKind()` plus the mutex args from `getCapabilityAttrArgs`
(`Type.cpp:4111`, which returns just `args_begin()/args_size()`). Sharedness
lives in the spelling index, not the kind: `ACQUIRE(mu)` and
`ACQUIRE_SHARED(mu)` are both `attr::AcquireCapability` with identical args,
so they produce identical `FoldingSetNodeID`s and
`ASTContext::getFunctionTypeInternal` (`ASTContext.cpp:5066`) hands the second
typedef the first one's already-uniqued `FunctionProtoType` — carrying the
*first* declaration's `Attr*` objects. First-created wins. This is a live bug
at HEAD for plain typedefs, independent of value-decl folding.

Verified at clean HEAD: all six semantic-variant pairs wrongly unify
(requires/requires_shared, acquire/acquire_shared, release
generic/exclusive/shared, assert/assert_shared, try_acquire success values
true/false, try_acquire exclusive/shared), with real diagnostic fallout
(missed exclusive-write warnings after a shared acquire; spurious
"requires ... exclusively" under a reader lock; wrong trylock branch; wrong
release-kind diagnostics).

The value-decl-folding regression is **fully explained**: in the
`FunctionPointers` namespace of
`clang/test/SemaCXX/warn-thread-safety-analysis.cpp`, every shared variant is
declared after its exclusive twin with an identical signature and the same
`mu` (`:7970/:7971`, `:7972/:7973`, `:7974/:7975`, `:7977/:7978`,
`:7979/:7980`). With an experimental VarDecl/FieldDecl fold applied, exactly
the TODO's two failures reproduce; with the Profile fix added, all
thread-safety tests pass (6/6 files, all RUN lines) and the
`Sema/SemaCXX/AST/PCH/Modules/Analysis` sweep is clean (5743 tests; one
unrelated stale-binary failure in `Analysis/func-mapping-test.cpp`).
**No unexplained residue.** TODO work item 1 is thereby resolved.

Fix (semantic accessors, NOT `getSpellingListIndex()` — verified that
differently-spelled synonyms must still unify):

| attr | add to Profile |
|---|---|
| `RequiresCapability` | `isShared()` |
| `AcquireCapability`  | `isShared()` |
| `AssertCapability`   | `isShared()` |
| `ReleaseCapability`  | `isShared()` **and** `isGeneric()` |
| `TryAcquireCapability` | `isShared()` **and** `getSuccessValue()` (profiled canonically) |
| `LocksExcluded` | nothing extra |

Note: the tablegen `A->Profile(ID, Ctx)` is not a substitute — it profiles
`SuccessValue` but not the spelling index. Also add a per-attribute
`ID.AddInteger(NumArgs)` separator and a sentinel for null args — today the
attrs' arg streams are concatenated with no boundary, so
`{K:(x)}{K:(y)}` vs `{K:(x,y)}`-style layouts are only accidentally distinct.

- [x] **F1. Fix `Profile`** as above; regression test covering both
  directions (synonym spellings unify; semantic variants stay distinct) plus
  behavioral shared/exclusive/trylock tests. Update the TODO doc's
  "Known-unexplained regression" section. *(Reference materials in scratchpad:
  `PROFILE_FIX.patch`, `synonym_unify.cpp`, `repro_typedef_collision.cpp`,
  `repro2_release_and_identity.cpp`, `successvalue_print.cpp`.)*

---

## Part II — Code-review findings (branch bugs beyond F1)

### Design-level (needs an explicit decision — see Part IV)

- [~] **F2. Distinct canonical types mangle identically → CodeGen hard error /
  silent ODR hazard.** `CapabilityAttrs` are part of the canonical type
  (`ASTContext.cpp:5126` keeps them in `CanonicalEPI`), so `void (*)()` and
  the caps-carrying variant are different types for overloading/templates,
  but the Itanium mangler ignores `FunctionTypeExtraAttributeInfo`. Verified:
  two `g(...)` definitions differing only in caps → *error: definition with
  same mangled name `_Z1gPFvvE`*; declaration-only overload sets are a silent
  link trap. The TODO doc's "mangling is already safe" conclusion is inverted
  — identical mangling is the bug, not the safety argument. Resolution
  options in Part IV.

### Concrete bugs

- [ ] **F3. Folded attrs are never substituted during template instantiation.**
  `TreeTransform::TransformFunctionProtoType` (`TreeTransform.h:6681`) copies
  `ExtraAttributeInfo` verbatim; the fold dropped the decl attrs, so
  `instantiateTemplateAttribute` never runs. `capabilityArgIsContextFree`
  (`SemaDeclAttr.cpp:8916`) does not reject dependent exprs, so the fold
  fires inside templates and un-substituted dependent args (e.g. `*M`,
  `T::smu`) survive into every instantiation — unsatisfiable warnings or
  `cannot resolve lock expression`. Fix: reject
  `isInstantiationDependent()` args in the guard (leave on decl), and make
  instantiated typedefs fold correctly (see F10); longer term, teach
  `TreeTransform` to transform the attr array.

- [ ] **F4. Late fold retroactively mutates a member typedef's underlying
  type, leaving stale canonical `TypedefType`s.** For class-member typedefs
  the attrs are late-parsed, so the fold (`SemaDecl.cpp:17252`) runs at
  end-of-class — after members declared with that typedef already interned a
  `TypedefType` whose canonical was computed pre-fold. Verified:
  `__is_same(decltype(Host::early), Host::cb)` is false; feeds the F2
  mangled-name error. Needs either fold-before-first-use, invalidation of
  `TypeForDecl`, or the sugar design (Part IV).

- [ ] **F5. `addCapabilityAttrsToFunctionType` drops qualifiers and
  pointer-level sugar.** (`SemaDeclAttr.cpp:8866-8905`) `T->getAs<PointerType>()`
  discards outer quals and sugar; rebuild loses `const` (verified: `typedef
  void (*const cfp)(void) REQ(mu)` → assignable) and `_Nonnull`. Fix: split
  off `Qualifiers` and re-apply; preserve sugar where feasible.

- [ ] **F6. `-ast-print` drops try-acquire's success value and doesn't
  re-parse.** (`TypePrinter.cpp:1122-1163`) The hand-rolled printer uses
  `getCapabilityAttrArgs`, which excludes `SuccessValue`; printed output is
  not round-trippable. Fix: call the tblgen `A->printPretty(OS, Policy)`
  instead of the spelling switch (the comment's rationale is inaccurate — the
  stored attr keeps its spelling index; what's unstable is *which* attr got
  stored, i.e. F1). Also fixes `unlock_function` → `release_generic_capability`
  re-spelling churn.

- [ ] **F7. Duplicate diagnostics when the same requirement is on both decl
  and type.** `handleCall` (`ThreadSafety.cpp:2197-2200`) concatenates without
  dedup; verified double warning for `void f(req_cb_t cb REQUIRES(mu))`.
  Pointer-identity dedup is insufficient here (distinct `Attr*` objects);
  dedup on kind + resolved capability, or skip decl attrs already present in
  the type set. Also **F7b (perf)**: don't copy `D->attrs()` into a
  `SmallVector` on every call when the type carries nothing.

- [ ] **F8. `using`-alias declarations accept the attribute and silently
  ignore it.** `ActOnAliasDeclaration` → `ProcessDeclAttributeList` never
  reaches the fold (`SemaDeclAttr.cpp:9044` only). C++ users will write
  `using`. Fix: invoke `foldCapabilityAttrsIntoType` on alias declarations.

- [ ] **F9. Typedef subject check diverges from the established helper; dead
  code.** (`SemaDeclAttr.cpp:475-483` vs `:440-444`) Raw
  `isFunctionPointerType()` instead of `isFunctionPointerOrDependent()`:
  dependent typedefs get a spurious `warn_thread_attribute_not_on_fun_ptr`,
  and function-reference typedefs are rejected while
  `addCapabilityAttrsToFunctionType` carries unreachable
  `Block/LValueRef/RValueRef/None` branches. Also
  `checkInstantiatedThreadSafetyAttrs` (`SemaDeclAttr.cpp:499`) has no
  `TypedefNameDecl` arm, so there is no post-instantiation recheck.

- [ ] **F10. Instantiated typedefs never fold.**
  `TemplateDeclInstantiator::VisitTypedefNameDecl` instantiates attrs but
  never calls the fold, so a class-template member typedef keeps its attrs on
  the decl in every instantiation — where the analysis never reads them
  (see F12). Fold after `InstantiateAttrs` when args became non-dependent.
  (Follows from F3/F9; same fix batch.)

- [ ] **F11. Calls with no `NamedDecl` callee are unchecked; docs overstate.**
  (`ThreadSafety.cpp:81-88`; `handleCall` requires `getCalleeDecl()`.)
  Verified unchecked: `tab[0]()`, `(*pp)()`, `get()()` for a caps typedef.
  Fix: read caps from the callee *expression's* type when there is no decl
  (best-effort), and/or soften `ThreadSafetyAnalysis.md`'s "calls through any
  value of that type are checked".

- [ ] **F12. Non-foldable typedef attributes are accepted and silently
  no-op.** When `capabilityArgIsContextFree` bails (object-relative args),
  the attrs stay on the `TypedefNameDecl`, which no analysis path reads —
  users get silence where they expect protection. Fix: diagnose ("attribute
  ignored") when the fold declines for a non-dependent reason.

- [ ] **F13. `capabilityArgIsContextFree` misses expression kinds.**
  (`SemaDeclAttr.cpp:8916-8929`) Doesn't reject dependent exprs
  (`CXXDependentScopeMemberExpr`, `DependentScopeDeclRefExpr`,
  `UnresolvedLookupExpr`, pack expansions, `SubstNonTypeTemplateParmExpr`) —
  feeds F3 — and accepts `DeclRefExpr` to function-local `VarDecl`s, interning
  a canonical type that references a block-scope decl. Fix: reject any
  `isInstantiationDependent()` expr and refs to `VarDecl`s without global
  storage. (Same batch as F3.)

- [ ] **F14. Type-identity machinery is unaware of `CapabilityAttrs`.**
  All pre-existing code that assumes function-type identity is fully captured
  elsewhere: `ODRHash::VisitFunctionProtoType` (`ODRHash.cpp:1053`),
  `ASTStructuralEquivalence` (`ASTStructuralEquivalence.cpp:1146`),
  `ASTNodeImporter::VisitFunctionProtoType` (`ASTImporter.cpp:1612-1627`,
  silently strips the attrs — pre-existing for `CFISalt` too), and
  `ASTContext::mergeFunctionTypes` (`ASTContext.cpp:11711-11714`, checks
  `CFISalt` only → C merge silently keeps the left side). Fix under the
  current design: hash/compare/import/merge them (merge rule: union for
  redeclarations, mirroring function effects). Needed regardless of Part IV,
  and `mergeFunctionTypes` union is a prerequisite for value-decl folding
  (Part III B1).

- [ ] **F15. C++ conditional operator hard-errors between caps and non-caps
  function pointers.** Composite-pointer-type computation was not taught the
  transparency that `IsFunctionConversion` got; `c ? a : b` errors in C++
  (and silently takes the LHS type in C). Fix: strip/intersect capability
  attrs when forming composite pointer types.

- [ ] **F16. Redeclaring an annotated typedef without the annotation is a
  hard error** (`MergeTypedefNameDecl`, both C and C++). Will bite
  annotate-the-system-header patterns. Decide transparency vs. strictness
  (consistent with F15/IsFunctionConversion → transparency) + test either way.

- [ ] **F17. `IsFunctionConversion` rebuilds the type even when the attr sets
  are equal.** (`SemaOverload.cpp:2059-2067`) Neighbouring effects block
  guards on inequality; this fires whenever either side is non-empty. Guard it
  (needs a real attr-set comparison helper).

- [ ] **F18. Style/upstreamability.**
  (a) The new `/// Rebuild \p T ...` block landed between the
  `ProcessDeclAttributes` banner comment and its function
  (`SemaDeclAttr.cpp:8858-8867`); move the three new functions above the
  banner. (b) Mis-indented continuation at `:8867`; run `git clang-format`.
  (c) `isCapabilityAttr`/`getCapabilityAttrArgs` declared in `Attr.h` but
  defined in `Type.cpp`; move to `AttrImpl.cpp`. (d) Null-arg skipping in
  `Profile` (folded into F1's separator fix).

### Test gaps (fold into the fixes above; sweep at the end)

- [ ] **T1.** No Modules test (only PCH) — the case that would catch F14's
  ODR-hash gap.
- [ ] **T2.** `ast-print` covers only `requires`/`release`; add try-acquire
  (with success value), assert, locks_excluded, shared variants, and a
  re-parse round-trip RUN line (exposes F6).
- [ ] **T3.** `warn-thread-safety-parsing.cpp` has no typedef-arm coverage of
  `warn_thread_attribute_not_on_fun_ptr` (C++ side).
- [ ] **T4.** No tests: typedef in a template (F3), `using` alias (F8),
  qualified fn-ptr typedef (F5), decl+type duplicate (F7), mangling/CodeGen
  (F2 — whatever the resolution), explicit Profile-collision pairs and
  try-acquire success-value distinctness (F1).

### Doc gaps

- [ ] **D1.** No `clang/docs/ReleaseNotes.rst` entry — needed for the typedef
  feature AND the behavior change in `f2fc9cc59cf8` (fn-ptr parameter attrs no
  longer seed the caller's entry lockset — a silent semantic change to
  existing annotations).
- [ ] **D2.** `ThreadSafetyAnalysis.md` overstates coverage (F11) and omits
  the two limitations users hit first: `using` aliases (F8, until fixed) and
  object-relative args silently ignored (F12, until diagnosed).
- [ ] **D3.** TODO doc corrections: regression section (resolved by F1),
  "mangling is already safe" inverted (F2), parameters rationale wrong (see
  Part III W5).
- [ ] **D4.** `Attr.td` now advertises `TypedefName` subjects while
  `RequiresCapability`/`LocksExcluded` docs remain `[Undocumented]`.

---

## Part III — Finishing value-decl folding (TODO items 2–5): study results

**Blockers**

- **B1 (= F14 merge half). Redeclaration of a folded variable**: hard
  `redeclaration with a different type` error in C++
  (`Sema::MergeVarDeclTypes` uses bare `hasSameType`, `SemaDecl.cpp:4604`);
  in C, `mergeFunctionTypes` returns the attr-free side → requirement
  silently lost. So the `mergeFunctionTypes` work is a *prerequisite* for
  VarDecl folding, not a later item. `FieldDecl` is immune (never
  redeclared) → **fields can ship before variables**.
- **B2 (= F3/F13). Dependent args fold unsubstituted** — must fix first.

**Key design recommendations (reversing two TODO assumptions)**

1. For value decls, **fold without `dropAttr`** (keep decl attrs; add dedup —
   F7 — in `handleCall` and `getTryAcquireCapabilityAttrs`). Dropping is what
   breaks redecl attr inheritance, `-ast-dump`, and future function-decl
   folding. Keep the drop for typedefs (decl is never the callee; printer
   already emits from the type).
2. Set **only `ValueDecl::setType`**, leave the `TypeSourceInfo` untouched
   (decl/TSI divergence is an accepted pattern, cf. `ParmVarDecl` decay); the
   current typedef fold's `getTrivialTypeSourceInfo` replacement loses
   written sugar. Also read the decl's type, not `OldTSI->getType()` (they
   diverge for value decls, e.g. address-space adjustment).
3. **`TypedefNameDecl` is not a `DeclaratorDecl`** — the obvious unified
   refactor null-derefs; keep the typedef and value-decl paths separate.

**Reader audit (TODO item 5) — complete.** The only decl-attr readers are in
`ThreadSafety.cpp` (call paths already type-aware; `runAnalysis` decl loops
matter only for function-decl folding; the two `Param->attrs()` loops are
already inert for fn-ptr params), `Sema` checks that run pre-fold, and
`ASTImporter`. **Nothing in clang-tools-extra reads these attrs at all.**

**Ordered work items**

- [ ] **W1** = F3/F13 guard hardening (prerequisite).
- [ ] **W2. Fold `FieldDecl`** (lowest risk, ships first). Timing verified
  safe: late-parsed member attrs attach (`ParseDeclCXX.cpp:3725`) after
  `ActOnFinishCXXMemberSpecification` but *before*
  `ParseLexedMemberInitializers` (`:3731`); triviality/layout unaffected
  (pointer size/align unchanged; `TypeInfo` keyed on canonical `Type*`). In C,
  TSA field attrs are not late-parsed (`ParseDecl.cpp:4958`), and the
  non-late path (`SemaDecl.cpp:19630`) precedes record completion.
- [ ] **W3** = F14 `mergeFunctionTypes`/`MergeVarDeclTypes` union (blocker B1).
- [ ] **W4. Fold `VarDecl`** (after W3), excluding `ParmVarDecl`.
  `ProcessDeclAttributes` at `SemaDecl.cpp:8277` precedes merge and
  initializer checking for namespace/block scope. Residual hazard: in-class
  static data member with initializer (init checked at
  `ParseDeclCXX.cpp:3183-3192`, before late attrs at `:3725`) — exclude
  static data members in the first cut or test explicitly.
- [~] **W5. Parameters: skip — but the TODO's rationale is wrong.**
  `ActOnParamDeclarator` runs `ProcessDeclAttributes` (`SemaDecl.cpp:15858`)
  *before* `GetFullTypeForDeclarator` collects param types
  (`SemaType.cpp:5265`), so a folded param *would* propagate into the
  enclosing prototype — changing the enclosing function's overload identity
  (`SemaOverload.cpp:1373`). That, not build order, is why params must stay
  on the decl path. Correct the TODO text (D3).
- [~] **W6. Function declarations: no-go for now.** Override matching and
  mangling are safe (contra the TODO's worry — though mangling "safety" is
  exactly F2's hazard); redeclaration is not (`MergeFunctionDecl` →
  `err_conflicting_types`), `runAnalysis` entry-lockset seeding
  (`ThreadSafety.cpp:2822-2845`) would go blind, and
  `checkThisInStaticMemberFunctionAttributes` ordering must be preserved.
  Revisit only after W3 + a union-merge policy with tests.

**Test plan** (from the study; execute with W2/W4): requires_shared VARIABLE +
`auto`; exclusive variable auto/keep/drop mirroring the typedef trio;
explicit collision pairs (all five) as variables; try-acquire success-value
variable pair; redecl-without-attr (C++ compiles + keeps requirement; C twin);
dependent-arg class template (no crash, sane capability name); static-member
initializer hazard; NSDMI field; field via `auto` must warn; class-template
field with global vs sibling-member arg; PCH/ast-print additions for folded
var + field; `FPOps`/`BDevOps` C structs must stay unfolded; full
`FunctionPointers` namespace unchanged; `Sema/SemaCXX/AST/PCH/Modules` sweeps.

---

## Part IV — Open design decision (user input needed before upstreaming)

**Should capability attrs be part of the canonical function type?**
(Determines F2, and colors F4, F14, F15, F16.)

- **Option A — keep canonical (current design) + make manglers emit them**
  (vendor-extension qualifier, like C++17 `noexcept`-in-type). Pros: full
  propagation through `auto`, templates, deduction — the point of the series;
  honest linkage (annotated/unannotated mismatches become link errors instead
  of silent ODR traps). Cons: mangling arbitrary mutex *expressions* is ugly;
  annotating a typedef changes the mangling of every function that takes it —
  an ABI cliff for adopters.
- **Option B — demote to type sugar** (AttributedType-style node; analysis
  desugars). Pros: F2/F4(partially)/F15/F16 evaporate; no ABI impact; much
  smaller identity blast radius. Cons: sugar does not reliably survive
  template deduction/canonicalization — weakens the feature to roughly
  `auto`-only propagation; large rework of the series.
- **Interim (what the fix plan below does): keep canonical, fix all
  *consistency* holes (F14), make Sema *transparent* at conversion seams
  (F15/F16, matching the existing `IsFunctionConversion` philosophy), add a
  CodeGen test documenting the mangling collision, and leave the A/B choice
  flagged.** Recommendation on record: Option A is truer to the feature's
  goal; prototype expression mangling before committing.

---

## Fix execution plan (sequential, one commit each)

1. **P1**: F1 (+ TODO doc regression note). ✔ gate for everything else.
2. **P2**: F6 printer → `printPretty`; T2 round-trip tests.
3. **P3**: F3 + F13 + F9 + F10 template/dependent correctness; T4 template
   tests.
4. **P4**: F5 qualifier preservation; F8 using-alias support; tests.
5. **P5**: F7 dedup + F7b perf; F11 callee-expression fallback (best-effort);
   tests.
6. **P6**: F14 identity consistency (ODRHash, structural equivalence,
   ASTImporter, mergeFunctionTypes union) + F15 + F16 transparency; T1
   Modules test.
7. **P7**: F4 stale member-typedef canonical — investigate; fix or document
   as known limitation tied to Part IV.
8. **P8**: F12 ignored-attr diagnostic; F17; F18 style sweep.
9. **P9**: D1–D4 docs; T3; final test sweep.
10. **P10**: W2 field folding, then W4 variable folding (W3 landed in P6),
    with the Part III test plan. Closes out the ValueDeclFolding TODO's items
    2, 3, 5, 6; item 4 (functions) stays deferred (W6).
