# TSA type-carried capabilities: review findings and fix plan

Working document for the `jn/tsa-typedef-capability` branch. It was written
against commits `f2fc9cc59cf8..cd7e942d5879`; the branch now extends through
`[TSA][17/N]`, which is the last commit of the P9 docs-and-tests phase below,
and every finding's status line reflects that state.
Consolidates three investigations (2026-07-25):
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

- [x] **F3. Folded attrs are never substituted during template instantiation.**
  `TreeTransform::TransformFunctionProtoType` (`TreeTransform.h:6681`) copies
  `ExtraAttributeInfo` verbatim; the fold dropped the decl attrs, so
  `instantiateTemplateAttribute` never runs. `capabilityArgIsContextFree`
  (`SemaDeclAttr.cpp:8916`) does not reject dependent exprs, so the fold
  fires inside templates and un-substituted dependent args (e.g. `*M`,
  `T::smu`) survive into every instantiation — unsatisfiable warnings or
  `cannot resolve lock expression`. Fixed in P3 by rejecting dependent args
  in the guard (F13) so they stay on the decl, and folding the instantiated
  typedef once its args have been substituted (F10). Both review repros are
  fixed: `NTTP<&mu1>::cb` now checks against `mu1`, and the dependent-base
  case resolves to `Base::smu`. `SExprBuilder::translate` also learned
  `SubstNonTypeTemplateParmExpr` (`ThreadSafetyCommon.cpp`), without which a
  substituted non-type template parameter reaches the analysis wrapped in
  substitution sugar and yields `cannot resolve lock expression`; this also
  fixes plain `REQUIRES(*M)` on a function template.
  - **Partial, by design (deferred):** `TreeTransform` still copies the attr
    array verbatim. That is correct for a *non-dependent* argument folded in
    the pattern (nothing to substitute — tested), and unreachable for a
    dependent one (it is never folded into the pattern's type).
  - **Known limitation (new, tracked here):** a use of the typedef from
    *inside* the template still names the pattern's `TypedefType`. That type
    is not instantiation-dependent — only the attribute's arguments are — so
    `Sema::SubstType` short-circuits and the instantiated (folded) typedef is
    never reached; the requirement is invisible to calls written inside the
    template, though it is honored for every use of the instantiated name.
    Closing this needs the dependent attribute folded into the *pattern's*
    type, the type's dependence bits to account for the attribute arguments,
    and `TreeTransform` to substitute the attr array — and, for member
    typedefs, it collides with F4's late-fold ordering. Documented with a
    FIXME in
    `clang/test/SemaCXX/thread-safety-type-capability-templates.cpp`.

- [x] **F4. Late fold retroactively mutates a member typedef's underlying
  type, leaving stale canonical `TypedefType`s.** For class-member typedefs
  the attrs are late-parsed, so the fold (`SemaDecl.cpp:17252`) runs at
  end-of-class — after members declared with that typedef already interned a
  `TypedefType` whose canonical was computed pre-fold. Verified:
  `__is_same(decltype(Host::early), Host::cb)` is false; feeds the F2
  mangled-name error.
  Fixed in P7 by **fold-before-first-use**: `Parser::ParseSingleGNUAttribute`
  does not late-parse a capability attribute when the declarator it is
  attached to is a typedef, so the attribute is attached — and folded — at the
  declaration itself, before any other member can name the typedef. This is
  exactly what an alias declaration already did (`ParseUsingDeclaration`
  passes no `LateParsedAttrList`), which is why `using cb REQUIRES(mu) = …`
  never had the bug and `typedef … cb REQUIRES(mu);` did; the two now behave
  the same. The price is that a typedef's requirement sees only what precedes
  it, so naming a member declared *later* in the class is now
  `use of undeclared identifier` instead of being accepted — as it already was
  for `using`, and as it is at namespace scope. Nothing outside this branch
  can depend on that: capability attributes on typedefs did not exist before
  the series (they were dropped with `-Wignored-attributes`). Member
  *functions*' attributes are untouched and still late-parsed, which is the
  reason late parsing exists.
  - **Why not the alternatives.** Resetting `TypeForDecl` at fold time only
    fixes future `getTypedefType` calls, so `cb early;` and `Host::cb` would
    still disagree — a split brain, strictly worse than the status quo.
    Updating the interned node's canonical in place heals only *direct* uses:
    derived types bake their canonical at creation (`getPointerType`,
    `getFunctionType`), so `cb *p;` and `void m(cb);` would stay stale while
    `cb early;` healed — an inconsistency no invariant could describe.
    Measured: pre-fix, all of `decltype(Host::early)`, `…::earlyp`,
    `cb[2]`, `void (Host::*)(cb)` and `cb (Host::*)()` differ from the
    corresponding `Host::cb`-spelled type, so the rot is as deep as the
    declarations go.
  - **Safety net (`SemaDeclAttr.cpp`).** The fold now declines outright when
    the typedef's type has already been handed out
    (`ASTContext::hasTypedefTypeBeenCreated`, a new accessor for the
    `TypeDecl::TypeForDecl` cache, which `TypedefNameDecl` deliberately
    `= delete`s). So a fold that would be too late never happens, and a
    typedef always names one type. It also documents the invariant the parser
    change exists to maintain.
  - **Residual, diagnosed by that net:** an attribute written in the
    *declaration-specifier* position of a member typedef
    (`REQUIRES(mu) typedef void (*cb)(void);`) is still late-parsed — at the
    point the parser sees it there is no declarator to say the declaration is
    a typedef, and it may even precede the `typedef` keyword. If the typedef
    was used inside the class the fold is refused and the requirement is
    lost — since P8 with F12's "type already used" diagnostic, rather than
    silently; if it was not used, the late fold is safe and still happens.
    Either way the typedef means one thing everywhere.
    This is the only spelling still affected, and it is not the one the
    thread-safety macros produce.
  - **Related, not fixed here (F3 residual):** a member declared with the
    typedef *inside* a template pattern whose requirement is dependent keeps
    the pattern's unfolded `TypedefType`, because that type is not
    instantiation-dependent and substitution never reaches the instantiated
    (folded) typedef. Same mismatch, different cause; closing it is the F3
    work item. A pattern whose requirement is *not* dependent folds in the
    pattern and was fixed by P7 along with the rest.
  - Tests: `clang/test/SemaCXX/thread-safety-type-capability-member.cpp`
    (identity for direct/pointer/array/method-signature/return-type/typedef-of
    uses, inside vs. outside the class, ambiguity-free overload resolution
    standing in for the mangling collision, the analysis still checking
    early-declared members, alias parity, both template shapes, the
    unfoldable-argument and declaration-specifier-position cases, and a
    second `-verify` run for the later-member arguments that are now
    rejected). Verified to fail 20 ways on the pre-fix compiler.

- [x] **F5. `addCapabilityAttrsToFunctionType` drops qualifiers and
  pointer-level sugar.** (`SemaDeclAttr.cpp:8866-8905`) `T->getAs<PointerType>()`
  discards outer quals and sugar; rebuild loses `const` (verified: `typedef
  void (*const cfp)(void) REQ(mu)` → assignable) and `_Nonnull`. Fix: split
  off `Qualifiers` and re-apply; preserve sugar where feasible.
  Fixed in P4: the rebuild peels the pointer level recursively and puts back
  (a) the local qualifiers, (b) an `AttributedType` — which is how nullability
  is spelled, and it is sugar, not a qualifier, so `getQualifiedType` would
  not have brought it back — rebuilt around both its modified and its
  equivalent type, and (c) the `MacroQualifiedType` that wraps the whole
  declarator type whenever the attribute is spelled through a macro (which is
  the normal case, and which hid the `AttributedType` from the rebuild).
  Qualifiers contributed by sugar that cannot be rebuilt (`typedef cfp c2
  REQ(mu)`) are recovered with `QualType::getQualifiers()`.
  - **Residual, deliberate:** other sugar around the pointer level — a typedef
    naming the function-pointer type, in particular — is still dropped, since
    the requirement has to be added *underneath* it and such sugar cannot be
    rebuilt around a different underlying type. Only `-ast-dump` can see this;
    the written form of the type is unchanged. Documented in the function
    comment.
  - **Observed, not caused by the fix:** converting a plain function pointer
    to one carrying a requirement is now a real conversion (the requirement is
    part of the canonical type), so a `_Nullable` → `_Nonnull` conversion that
    also adds the requirement warns *twice*, once per conversion step.
    Investigated in P5, and it is **not** the same family as F7 and not this
    feature's bug: `Sema::PerformImplicitConversion` calls
    `diagnoseNullableToNonnullConversion` at its tail
    (`SemaExprCXX.cpp:5452`) *and* `Sema::ImpCastExprToType` calls it again
    for the cast it builds (`Sema.cpp:797`), so **any** nullable→non-null
    conversion that also produces a cast warns twice. Reproduces with no
    thread-safety attribute in sight — `void f(D *_Nullable d) { B *_Nonnull
    b = d; }`, `int *_Nullable` → `const int *_Nonnull`, `int *_Nullable` →
    `void *_Nonnull` all emit `-Wnullable-to-nonnull-conversion` twice. All a
    type-carried requirement did was turn a formerly identity conversion into
    a real one, exposing the pre-existing duplicate. Fixing it belongs in
    Sema's conversion diagnostics (it would change expectations across
    unrelated nullability tests) and is out of scope for this series.

- [x] **F6. `-ast-print` drops try-acquire's success value and doesn't
  re-parse.** (`TypePrinter.cpp:1122-1163`) The hand-rolled printer uses
  `getCapabilityAttrArgs`, which excludes `SuccessValue`; printed output is
  not round-trippable. Fixed by printing every argument (success value first)
  and `A->getSpelling()`, so `unlock_function` is no longer re-spelled to
  `release_generic_capability`. *Not* `A->printPretty`, as originally planned:
  it emits `[[clang::…]]` for a C++11-spelled attribute, and that spelling is
  rejected in the trailing position of a function declarator (these are
  declaration attributes; only the GNU spelling slides onto the declaration
  from there), so the output would still not re-parse — see F19. The old
  comment's claim was half right: the attr does keep its spelling index, but
  spelling is deliberately not profiled (F1), so the surviving folding-set
  node's spelling is what gets printed for every synonym.
  - Also fixed the tblgen bug this uncovered: `VariadicExprArgument` printed
    its expressions with `OS << Val`, i.e. as a pointer value
    (`__attribute__((requires_capability(0x3fcc7b58)))`) in every
    `Attr::printPretty` of a thread-safety or `annotate` attribute.

- [x] **F7. Duplicate diagnostics when the same requirement is on both decl
  and type.** `handleCall` (`ThreadSafety.cpp:2197-2200`) concatenates without
  dedup; verified double warning for `void f(req_cb_t cb REQUIRES(mu))`.
  Pointer-identity dedup is insufficient here (distinct `Attr*` objects);
  dedup on kind + resolved capability, or skip decl attrs already present in
  the type set. Also **F7b (perf)**: don't copy `D->attrs()` into a
  `SmallVector` on every call when the type carries nothing.
  Fixed in P5. The double warning came from `requires_capability` and
  `locks_excluded` warning per attribute *before* any set insertion; the
  acquire/release/assert kinds were already silently deduped by
  `CapExprSet::push_back_nodup`, which is why only those two kinds doubled.
  Dedup is done at the attribute level, using the folding-set notion of
  equality: `Type.cpp`'s per-attribute profiling was factored out of
  `FunctionTypeExtraAttributeInfo::Profile` into `profileCapabilityAttr`, and
  `clang::areEquivalentCapabilityAttrs` (declared in `Attr.h`) compares two
  attributes by that profile. So two attributes are redundant exactly when
  they would produce the same function type — spelling-independent
  (`exclusive_locks_required` vs `requires_capability` dedup), but sharedness-,
  genericness- and success-value–sensitive (shared and exclusive requirements
  on the same mutex stay two diagnostics). Applied both in `handleCall` and in
  `getTryAcquireCapabilityAttrs`.
  - **F7b** fixed by the same change: the `SmallVector` copy is gone. The
    per-attribute switch became a `HandleAttr` lambda, `D->attrs()` is looped
    over directly, and the redundancy test (and the `D->getASTContext()` walk
    it needs) is skipped entirely when the callee type carries no capability
    attributes — the common case now costs one `ArrayRef::empty()` check.

- [x] **F8. `using`-alias declarations accept the attribute and silently
  ignore it.** `ActOnAliasDeclaration` → `ProcessDeclAttributeList` never
  reaches the fold (`SemaDeclAttr.cpp:9044` only). C++ users will write
  `using`. Fix: invoke `foldCapabilityAttrsIntoType` on alias declarations.
  Fixed in P4: `Sema::ActOnAliasDeclaration` calls the fold right after
  attribute processing. `TypeAliasDecl` is a `TypedefNameDecl`, so the subject
  check (F9) and the instantiation hook (F10 — `TypeAliasDecl` instantiation
  shares `InstantiateTypedefNameDecl`) already applied. (An earlier draft of
  this entry said a member alias' attribute reached the fold through
  `ActOnFinishDelayedAttribute`; it does not — `ParseUsingDeclaration` passes
  no `LateParsedAttrList`, so an alias' attribute is never late-parsed. That
  is precisely why aliases never had F4, and P7 gave typedefs the same
  timing.) Verified working: requires/acquire/release
  through an alias, `auto` propagation, both attribute spellings, aliases in a
  class, member aliases of a class template with an NTTP mutex (folded per
  instantiation), an alias template whose underlying type is not dependent,
  and an alias and a typedef spelling the same requirement uniquing to the
  same type.
  - **Residual F8a (alias templates).** An alias template is not instantiated
    as a declaration — using it substitutes into the pattern's underlying type
    — so when the fold has to be deferred (dependent underlying type, or
    dependent capability argument such as an NTTP mutex) nothing retries it
    and the requirement is lost. A fix belongs in `CheckTemplateIdType`'s
    alias-template path and needs the pattern's attributes substituted without
    a declaration to hang them on. *P8 did not fix that, but it is no longer
    silent: an alias template is the one shape whose deferred fold can never
    be retried, so F12's diagnostic fires at the pattern (reason 4) and the
    attribute is dropped. The FIXME tests in
    `thread-safety-type-capability-alias.cpp` became expectations, joined by a
    member alias template of a class template — reported once at the pattern,
    not once per enclosing instantiation, because the drop takes the attribute
    out of the instantiation's way.*
  - **Residual F8b (other attribute positions).** The review's "trailing
    position is dropped silently" is only partly right, and neither trailing
    form is silent: `using a = void (*)(void) __attribute__((...));` is a
    parse error (`expected ';' after alias declaration`) and
    `using a = void (*)(void) [[clang::...]];` is an error (`attribute cannot
    be applied to types`). What *is* silent is an attribute written inside the
    type-id's declarator (`using a = void (*__attribute__((...)))(void);`):
    for a typedef it slides onto the declaration and works, but an alias
    declaration's type-id is parsed without a declaration to slide onto.
    FIXME test recorded; fixing it means routing the type-id declarator's
    sliding attributes into `ActOnAliasDeclaration`.
  - **Residual F8c (printing).** `-ast-print` prints an alias' folded
    attribute after the type-id — exactly the position that does not re-parse
    (F8b) — so alias output is not round-trippable. Fixing it means letting
    the declaration printer print the type's capability attributes after the
    alias name and suppressing them in the type printer (a `PrintingPolicy`
    bit); left out of P4 and noted in
    `clang/test/AST/ast-print-thread-safety-attrs.cpp`.

- [x] **F9. Typedef subject check diverges from the established helper; dead
  code.** (`SemaDeclAttr.cpp:475-483` vs `:440-444`) Raw
  `isFunctionPointerType()` instead of `isFunctionPointerOrDependent()`:
  dependent typedefs get a spurious `warn_thread_attribute_not_on_fun_ptr`,
  and function-reference typedefs are rejected while
  `addCapabilityAttrsToFunctionType` carries unreachable
  `Block/LValueRef/RValueRef/None` branches. Also
  `checkInstantiatedThreadSafetyAttrs` (`SemaDeclAttr.cpp:499`) has no
  `TypedefNameDecl` arm, so there is no post-instantiation recheck.
  Fixed in P3: the typedef arm moved into a
  `checkThreadSafetyTypedefIsFunPtr` helper that also accepts a dependent
  underlying type, and `checkInstantiatedThreadSafetyAttrs` grew a
  `TypedefNameDecl` arm that reruns it after substitution — so
  `typedef T cb REQUIRES(mu)` is silent at parse and warns when instantiated
  with a non-function-pointer `T`.
  - **Deliberate divergence:** unlike `isFunctionPointerOrDependent`, the
    typedef helper does *not* look through a reference. A reference to a
    function pointer cannot carry the requirement in its type
    (`addCapabilityAttrsToFunctionType` would decline), so accepting it would
    trade an honest diagnostic for silence (F12).
  - **Dead branches:** the gate stays pointer+dependent-only; the helper keeps
    its general shape with a comment explaining that only the pointer path is
    reachable. Admitting block pointers was rejected: the analysis cannot
    check a call through a block at all (F11 — `handleCall` needs a
    `NamedDecl` callee), so it would only fold into a type nothing reads.

- [x] **F10. Instantiated typedefs never fold.**
  `TemplateDeclInstantiator::VisitTypedefNameDecl` instantiates attrs but
  never calls the fold, so a class-template member typedef keeps its attrs on
  the decl in every instantiation — where the analysis never reads them
  (see F12). Fold after `InstantiateAttrs` when args became non-dependent.
  (Follows from F3/F9; same fix batch.) Fixed in P3:
  `InstantiateTypedefNameDecl` calls `foldCapabilityAttrsIntoType` after
  `InstantiateAttrs`. Two instantiations with different mutex arguments now
  get distinct types, and uses of the instantiated name are checked against
  the substituted mutex. See F3 for the residual in-template limitation.

- [x] **F11. Calls with no `NamedDecl` callee are unchecked; docs overstate.**
  (`ThreadSafety.cpp:81-88`; `handleCall` requires `getCalleeDecl()`.)
  Verified unchecked: `tab[0]()`, `(*pp)()`, `get()()` for a caps typedef.
  Fix: read caps from the callee *expression's* type when there is no decl
  (best-effort), and/or soften `ThreadSafetyAnalysis.md`'s "calls through any
  value of that type are checked".
  Fixed in P5, in two independent halves:
  - **Where the attributes come from.** `getTypeCapabilityAttrs(Exp, D)`
    replaces `getCalleeFunctionProtoType(D)` at every use. It reads the
    *callee expression's* type, which is the type of the value actually being
    called, and consults the declaration only when the callee expression has
    no function type of its own (a member call, whose callee is a bound
    member) or when there is no call expression at all (implicit destructor
    calls). This alone fixes `(*pp)()`: a declaration *was* reachable there
    (`pp`), but its type is a pointer *to* the function pointer, so the
    requirement was invisible.
  - **What the call is attributed to.** `getCalleeDeclForAnalysis` falls back,
    when `getCalleeDecl()` names nothing *and* the callee type carries
    capability attributes, to `getIndirectCalleeDecl` — the declaration the
    function pointer was loaded from, found by walking array subscripts down
    to a `DeclRefExpr` or `MemberExpr`. That covers `tab[0]()` and
    `s->tab[0]()`. Used by `BuildLockset::VisitCallExpr` and by
    `getTerminatorTrylockCall`, so try-acquire on a branch works through an
    array element too.
  - **Residual, deliberate:** a callee that is not loaded from a declaration
    at all — `get()()`, `((req_cb_t)p)()` — is still unchecked. Making it work
    needs a null `NamedDecl` to flow through `handleCall`, and the three
    diagnostics that name the callee (`warn_fun_requires_lock` and its
    `_precise` twin take a `NamedDecl` and stream it as `%1`,
    `warn_fun_excludes_mutex` and `warn_fun_requires_negative_cap` take a
    name string) would each need a nameless variant, plus null-tolerance in
    `SExprBuilder::translateAttrExpr`'s `dyn_cast<CXXMethodDecl>(D)`. That is
    a handler-interface change disproportionate to the case; FIXME tests
    record it in `warn-thread-safety-analysis.cpp`.
  - Doc sentence softened (see D2): `ThreadSafetyAnalysis.md` now states which
    callee forms are checked, shows the unchecked one, and notes the F7 dedup.

- [x] **F12. Non-foldable typedef attributes are accepted and silently
  no-op.** When `capabilityArgIsContextFree` bails (object-relative args),
  the attrs stay on the `TypedefNameDecl`, which no analysis path reads —
  users get silence where they expect protection. Fix: diagnose ("attribute
  ignored") when the fold declines for a non-dependent reason.
  Fixed in P8. `warn_thread_attribute_on_typedef_ignored`
  (`-Wthread-safety-attributes`, so on under `-Wthread-safety`) is emitted
  from `Sema::foldCapabilityAttrsIntoType` at the attribute's location, naming
  the attribute, the typedef, and one of five reasons
  (`CapabilityFoldObstacle`, whose enumerators are the diagnostic's `%select`
  values):
  1. *object-relative argument* — `this`, a `MemberExpr`, a `FieldDecl` or a
     `ParmVarDecl`, i.e. the original F12 case;
  2. *no global storage* — a block-scope mutex (P3's guard, F13);
  3. *type already used* — P7's safety net, reachable through the
     declaration-specifier attribute position (F4's residual);
  4. *alias template* — a dependent obstacle on an alias-template pattern
     (F8a), which is never retried;
  5. *no prototype* — a pointer to a `FunctionNoProtoType`, e.g.
     `typedef void (*cb)() REQ(mu);` before C23. The subject check accepts it
     (it *is* a function pointer type) but there is no `FunctionProtoType` to
     carry the attribute. Not previously known; found while enumerating the
     `NewType.isNull()` paths.
  - **Dependent obstacles do not warn in a class or function template.** The
    fold is retried on the instantiated declaration (F10), so nothing is lost.
    An alias template is the one shape that is never instantiated as a
    declaration, so it warns — `foldCapabilityAttrsIntoType` grew an
    `IsAliasTemplatePattern` parameter because neither `ActOnAliasDeclaration`
    nor `InstantiateTypeAliasTemplateDecl` has called
    `setDescribedAliasTemplate` by the time it runs.
  - **The attributes are dropped when they are diagnosed** (and only then —
    a deferred fold must keep them for `InstantiateAttrs` to substitute).
    `DeclPrinter::VisitTypedefDecl`/`VisitTypeAliasDecl` *do* print a
    typedef's declaration attributes, so leaving them would make `-ast-print`
    show an annotation that means nothing; verified that an ignored attribute
    no longer appears while a folded one still does. Dropping also makes the
    diagnostic fire once for a member alias template rather than once per
    enclosing instantiation.
  - Tests updated to expect the warning, and their FIXMEs removed:
    `warn-thread-safety-analysis.cpp` (`Host::member_req_t`, the
    `[TSA][6/N]`-era "no crash" case), `.../-member.cpp` (`Unfoldable::cb`
    and `DeclSpecPosition::cb`; its two `-verify` runs now share an `attrs`
    prefix for the diagnostics common to both), `.../-templates.cpp`
    (block-scope mutex), `.../-alias.cpp` (`Host::self_cb`, and the two F8a
    alias templates plus a new member-alias-template case). New coverage for
    the no-prototype reason in `clang/test/Sema/warn-thread-safety-analysis.c`.

- [x] **F13. `capabilityArgIsContextFree` misses expression kinds.**
  (`SemaDeclAttr.cpp:8916-8929`) Doesn't reject dependent exprs
  (`CXXDependentScopeMemberExpr`, `DependentScopeDeclRefExpr`,
  `UnresolvedLookupExpr`, pack expansions, `SubstNonTypeTemplateParmExpr`) —
  feeds F3 — and accepts `DeclRefExpr` to function-local `VarDecl`s, interning
  a canonical type that references a block-scope decl. Fix: reject any
  `isInstantiationDependent()` expr and refs to `VarDecl`s without global
  storage. (Same batch as F3.) Both guards added in P3. Note the second one
  makes a function-local `REQUIRES(local_mutex)` typedef stop folding, so it
  is now ignored (a `static` local still folds); tested, and diagnosed since
  P8 under F12's reason 2 rather than silent.

- [x] **F14. Type-identity machinery is unaware of `CapabilityAttrs`.**
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
  (Part III B1). *All four done in P6. Two helpers were extracted from
  `profileCapabilityAttr` so that every consumer agrees on what makes two
  requirements different: `getCapabilityAttrSemantics` (the
  sharedness/genericness encoded in the spelling — but not the spelling, so
  synonyms still unify) and `getCapabilityAttrSuccessValue`. Two more were
  added for the merge rules: `areEquivalentCapabilityAttrSets` (set equality,
  order- and duplicate-insensitive) and `mergeCapabilityAttrs` (union or
  intersection, deduplicating, returning an operand's own array when the merge
  did not change it so callers can keep that operand's type).*
  - [x] **F14a. `ODRHash`.** Hashes kind + semantics + success value +
    arguments (via `ODRHash::AddStmt`), mirroring
    `FunctionTypeExtraAttributeInfo::Profile`. Before this, two modules whose
    class definitions differed only in *which* capability a member typedef
    required merged silently (the annotated-vs-unannotated case happened to be
    caught elsewhere; the different-mutex case was not). T1 covers both.
  - [x] **F14b. `ASTStructuralEquivalence`.** New
    `IsEquivalentCapabilityAttrs`, called from the `FunctionProto` case.
    `areEquivalentCapabilityAttrs` is unusable here because it compares within
    one `ASTContext`; the arguments are compared with `IsStructurallyEquivalent`
    instead, following the noexcept-expression precedent.
  - [x] **F14c. `ASTImporter`.** `ToEPI.ExtraAttributeInfo` is now populated:
    each attribute through `ASTImporter::Import(const Attr *)`, the array
    allocated in the destination context, and the `CFISalt` string copied into
    the destination allocator (that half was a pre-existing bug).
  - [x] **F14d. `mergeFunctionTypes`.** Union for redeclaration merging,
    intersection when `IsConditionalOperator`, `allLTypes`/`allRTypes` cleared
    per side, exactly parallel to the neighbouring `FunctionEffects` block.

- [x] **F15. C++ conditional operator hard-errors between caps and non-caps
  function pointers.** Composite-pointer-type computation was not taught the
  transparency that `IsFunctionConversion` got; `c ? a : b` errors in C++
  (and silently takes the LHS type in C). Fix: strip/intersect capability
  attrs when forming composite pointer types. *Done in P6:
  `Sema::FindCompositePointerType` intersects the two attribute sets in the
  same `Steps.size() == 1` block that already merges noreturn, cfi-unchecked
  and the exception spec. The C side is fixed through F14d's
  `IsConditionalOperator` path — it did indeed take the LHS before, and now
  intersects. Tested in both languages, in both operand orders, including an
  empty intersection.*

- [x] **F16. Redeclaring an annotated typedef without the annotation is a
  hard error** (`MergeTypedefNameDecl`, both C and C++). Will bite
  annotate-the-system-header patterns. Decide transparency vs. strictness
  (consistent with F15/IsFunctionConversion → transparency) + test either way.
  ***Decision (P6): keep it strict; no code change, behavior documented by
  test.*** Rationale:
  1. *The closest in-tree analogue behaves this way.* Function effects
     (`[[clang::nonblocking]]`) are also carried in the canonical function
     type, are also an analysis-only property rather than an ABI contract, are
     also made transparent in `IsFunctionConversion`, and are also
     union/intersect-merged in `mergeFunctionTypes` — and
     `typedef void (*F)() [[clang::nonblocking]]; typedef void (*F)();` is a
     hard error today, in both C and C++. Diverging here would be an
     unexplainable inconsistency in review. The same holds for every other
     type-carried property: noexcept, calling convention, `cfi_salt`.
  2. *A typedef redefinition is not a conversion seam.* The transparency
     argument applies where a value of one type must be usable as another
     (implicit conversion, composite type) — there, F15/`IsFunctionConversion`
     now apply it. Redefinition is a question of type *identity*, and the two
     underlying types genuinely are different types.
  3. *Strictness protects against annotation drift*, which is the conservative
     default for a safety analysis; transparency would let a distant, later
     redefinition silently attach requirements to a name.
  4. *The system-header pattern is served differently.* Redefining a typedef
     is not actually how headers get annotated: it is ill-formed in C++ class
     scope and `-Wtypedef-redefinition` in pre-C11 C. Annotating the header
     itself, or wrapping it, is the supported route.

  Note that this is *not* in tension with F14d: C type *compatibility*
  (`mergeTypes`, used for redeclarations of functions and variables) unions
  the requirements, while typedef *redefinition* (`hasSameType`, C11 6.7p3
  "the same type") stays strict. Function effects draw the line in exactly the
  same place.

- [x] **F17. `IsFunctionConversion` rebuilds the type even when the attr sets
  are equal.** (`SemaOverload.cpp:2059-2067`) Neighbouring effects block
  guards on inequality; this fires whenever either side is non-empty. Guard it
  (needs a real attr-set comparison helper). *Done in P8: the block is now
  guarded by `!areEquivalentCapabilityAttrSets(...)` (P6's helper), matching
  the effects block above it. No behavior change — the rebuild was a no-op
  whenever the sets were equal, it just set `Changed` and forced a
  `getFunctionType` lookup; full `Sema`/`SemaCXX`/`SemaTemplate`/`AST`/`PCH`/
  `Modules`/`Analysis`/`ASTMerge` sweep unchanged.*

- [x] **F18. Style/upstreamability.**
  (a) The new `/// Rebuild \p T ...` block landed between the
  `ProcessDeclAttributes` banner comment and its function
  (`SemaDeclAttr.cpp:8858-8867`); move the three new functions above the
  banner. (b) Mis-indented continuation at `:8867`; run `git clang-format`.
  (c) `isCapabilityAttr`/`getCapabilityAttrArgs` declared in `Attr.h` but
  defined in `Type.cpp`; move to `AttrImpl.cpp`. (d) Null-arg skipping in
  `Profile` (folded into F1's separator fix). *All done in P8.*
  - (a)/(b) `addCapabilityAttrsToFunctionType`, `capabilityArgIsContextFree`
    and `Sema::foldCapabilityAttrsIntoType` (plus P8's two new helpers) now sit
    *above* the banner comment, which is re-attached to `ProcessDeclAttributes`.
  - (c) The whole capability-attribute helper family moved to
    `clang/lib/AST/AttrImpl.cpp` under a section banner: `isCapabilityAttr`,
    `getCapabilityAttrArgs`, `getCapabilityAttrSemantics`,
    `getCapabilityAttrSuccessValue`, `profileCapabilityAttr`,
    `areEquivalentCapabilityAttrs`, `containsEquivalentCapabilityAttr`,
    `areEquivalentCapabilityAttrSets`, `mergeCapabilityAttrs`.
    `profileCapabilityAttr` had to stop being `static` — it is shared between
    `areEquivalentCapabilityAttrs` and `FunctionTypeExtraAttributeInfo::Profile`
    — so it is declared in `Attr.h` with the rest. `Profile` itself stays in
    `Type.cpp`, and no new includes were needed: `AttrImpl.cpp` already pulls
    in `ASTContext.h`, `Attr.h`, `Expr.h` and `Type.h`.
  - (d) `git clang-format f2fc9cc59cf8~1..HEAD` over the files the branch
    touches: `ThreadSafety.cpp` (P5's `HandleAttr` lambda left the whole
    `switch` indented one level too deep — whitespace only, confirmed with
    `git diff -w`), `ParseDecl.cpp`, `SemaDeclAttr.cpp` (F18(b)'s
    continuation), `SemaDeclCXX.cpp`, `SemaOverload.cpp`. Re-run: clean.

- [~] **F19. A C++11-spelled capability attribute is not accepted where the
  type printer has to write it.** Found while fixing F6.
  `[[clang::requires_capability(mu)]] typedef void (*cb)(void);` folds fine,
  but `typedef void (*cb)() [[clang::requires_capability(mu)]];` — the only
  place a *type* can carry the attribute — is rejected with
  `err_attribute_not_type_attr` ("attribute cannot be applied to types"),
  because these are declaration attributes and only GNU-syntax ones slide from
  the declarator onto the declaration. So the printer has to normalize to the
  GNU syntax (see F6), and a user cannot write the C++11 spelling on, say, a
  function parameter's function-pointer type. Now that the attributes really
  are part of the function type, `ProcessTypeAttributeList` arguably should
  accept them there.
  ***Assessed in P8; deliberately not done.*** The mechanical part is small,
  the semantic part is a separate feature. What it would take:
  1. *Attr.td.* The six attributes are `InheritableAttr` with
     `Subjects = [Function, Var, Field, TypedefName]`. They would become
     `DeclOrTypeAttr` (the base `CDecl`, `LifetimeBound`, `CountedBy` … use),
     and their `ParsedAttr::AT_*` kinds would join
     `FUNCTION_TYPE_ATTRS_CASELIST` in `SemaType.cpp:152`.
  2. *SemaType.cpp.* A branch in `handleFunctionTypeAttr` alongside
     `AT_CFISalt` (`:8183`), which is the exact precedent: check the
     arguments, require a `FunctionProtoType`, rebuild via
     `ExtProtoInfo::ExtraAttributeInfo`, `unwrapped.wrap`. P8 already has the
     rebuild logic in `addCapabilityAttrsToFunctionType`.
  3. *The argument checking is where it stops being mechanical.*
     `checkAttrArgsAreCapabilityObjs` (`SemaDeclAttr.cpp:345`) takes a
     `Decl *D` and uses it for the no-argument `this` form (`dyn_cast<
     CXXMethodDecl>(D)`, `warn_thread_attribute_not_on_capability_member`) and
     for the `ParamIdxOk` form (`FD->getParamDecl(...)`). In a type position
     there is no declaration, so both forms need a defined answer — presumably
     "rejected in a type position", which is a new diagnostic and a new rule
     rather than a relaxation.
  4. *Late parsing.* These attributes are `LateParsed` with
     `ParseArgsInFunctionScope = 1` precisely so that a member's requirement
     may name members declared later; P7 turned that *off* for typedef
     declarators to fix F4. A type-position spelling is parsed immediately,
     with no declarator to hang a `LateParsedAttrList` on, so it would be a
     third timing regime in machinery this series just finished stabilizing.
  5. *It is a feature expansion, not a spelling fix.* A genuine type attribute
     can be written on a parameter type, a return type, a cast, a template
     argument — every one of which is a new place a requirement can appear and
     a new question for the analysis to answer. F6 needs none of that: the
     printer normalizes to the GNU spelling, which round-trips.
  Conclusion: out of scope for this series; a good standalone patch once the
  Part IV canonical-vs-sugar decision is made, since a sugar-based design
  (Option B) would answer (3)–(5) differently. No code change in P8; parsing
  untouched.

### Test gaps (fold into the fixes above; sweep at the end)

- [x] **T1.** No Modules test (only PCH) — the case that would catch F14's
  ODR-hash gap. *Done in P6:
  `clang/test/Modules/thread-safety-type-capability.cpp` (plus
  `Inputs/thread-safety-type-capability/`) builds two modules that define the
  same classes with a member typedef carrying a requirement, and asserts that
  identical requirements — and differently spelled synonyms for one — merge
  silently, while a different mutex, a missing annotation, and a different
  sharedness each produce an ODR-mismatch diagnostic. The different-mutex case
  is the one that used to merge silently. The `ASTImporter`/structural
  equivalence half is covered at lit level instead of by a unit test (see the
  deviation note under F14c's tests):
  `clang/test/ASTMerge/thread-safety-type-capability/test.c` asserts through
  `-ast-dump` that an imported caps-carrying typedef keeps its requirement, and
  through `-ast-merge` of two ASTs that the cross-TU
  `-Wodr` check distinguishes different requirements while merging equal ones.*
- [x] **T2.** `ast-print` covers only `requires`/`release`; add try-acquire
  (with success value), assert, locks_excluded, shared variants, and a
  re-parse round-trip RUN line (exposes F6). *Done in P2: all six kinds,
  shared/generic variants, every GNU-legacy spelling, the C++11 spelling, and
  a parse-back + print-again round trip in
  `clang/test/AST/ast-print-thread-safety-attrs.cpp`. Still missing: a printed
  template/dependent case (leave to P3's T4). Done in P3: the pattern prints
  its unfolded declaration attribute, the instantiation prints the folded,
  substituted one, and both round-trip.*
- [x] **T3.** `warn-thread-safety-parsing.cpp` has no typedef-arm coverage of
  `warn_thread_attribute_not_on_fun_ptr` (C++ side). *Done in P9: a new
  `TypedefSubjects` namespace covers all six attribute kinds (both spellings of
  each) on a good function-pointer typedef and on a non-function-pointer one,
  the shapes that are deliberately rejected (function reference,
  reference-to-function-pointer, array of function pointers, pointer to member
  function — mirroring the variable/field cases already there), a member
  typedef, a dependent typedef rechecked after substitution, the alias
  declaration arm of each (guarded on C++11, since the file also runs at
  `-std=c++98`), and the argument-level checks that apply to a typedef the same
  way they do to a function.*
- [~] **T4.** No tests: typedef in a template (F3), `using` alias (F8),
  qualified fn-ptr typedef (F5), decl+type duplicate (F7), mangling/CodeGen
  (F2 — whatever the resolution), explicit Profile-collision pairs and
  try-acquire success-value distinctness (F1). *Profile pairs and success
  values done in P1 (`thread-safety-type-capability-uniquing.cpp`); the
  template part done in P3
  (`clang/test/SemaCXX/thread-safety-type-capability-templates.cpp`: NTTP
  mutex, dependent base / dependent qualified name, dependent underlying
  type, non-dependent argument in a class template including a
  dependent-return-type rebuild, block-scope vs static-local mutex, distinct
  types for distinct template arguments) plus the ast-print case in T2. F5 and
  F8 done in P4
  (`clang/test/SemaCXX/thread-safety-type-capability-qualifiers.cpp`:
  const/volatile/address-space, a qualifier behind a typedef, nullability,
  nullability plus const, and the assignment error a dropped `const` would
  hide; `clang/test/SemaCXX/thread-safety-type-capability-alias.cpp`: the
  alias cases listed under F8, with FIXME tests for F8a and F8b), plus
  const/`_Nonnull` round-trip cases in the ast-print test. F7 done in P5
  (`warn-thread-safety-analysis.cpp`, namespace `FunctionPointers`: the
  decl+type duplicate for requires/excludes/acquire/release/try-acquire, a
  differently spelled duplicate, and the two non-duplicates — a different
  mutex, and shared vs exclusive on the same one — that must still warn
  twice), together with the F11 positive and FIXME cases. Still open: F2's
  CodeGen test, in its own patch.*

### Doc gaps

- [x] **D1.** No release-notes entry — needed for the typedef feature AND the
  behavior change in `f2fc9cc59cf8` (fn-ptr parameter attrs no longer seed the
  caller's entry lockset — a silent semantic change to existing annotations).
  *Done in P9, in `clang/docs/ReleaseNotes.md` (the file is Markdown now, not
  `.rst`). Three entries: the typedef/alias feature under "Attribute Changes in
  Clang"; the new `-Wthread-safety-attributes` ignored-attribute diagnostic
  (F12) under "Improvements to Clang's diagnostics"; and the `f2fc9cc59cf8`
  behavior change under "C/C++ Language Potentially Breaking Changes", spelling
  out both halves of it — the requirement was demanded of the *argument* at
  every call site and seeded into the enclosing function's entry lockset — and
  saying explicitly that it changes the meaning of existing annotations
  silently.*
- [x] **D2.** `ThreadSafetyAnalysis.md` overstates coverage (F11) and omits
  the two limitations users hit first: `using` aliases (fixed in P4, but the
  attribute's position on an alias declaration and the F8a alias-template hole
  need saying) and object-relative args, which are ignored — with a
  diagnostic since P8 (F12), which the doc should point at.
  *The F11 half was done in P5: "calls through any of them are
  checked" no longer claims more than the analysis delivers — the doc now
  lists the callee forms that are checked, shows the unchecked
  not-loaded-from-a-declaration case as a FIXME example, and states that a
  requirement written on both a declaration and its type is reported once.
  The rest is done in P9: the section is retitled "Function pointer typedefs
  and aliases" and split into subsections covering the alias form and where its
  attribute goes (F8b), conversion transparency in **both** directions plus the
  `?:` intersection (F15), redefinition strictness with the pointer to
  `noexcept`/`nonblocking` (F16), templates including the pattern-internal gap
  (F3 residual), the exclusive/shared type distinction (F1), the F12 diagnostic
  with its example and the full list of reasons, and the
  requirement-must-precede-the-typedef ordering rule (F4/P7). Every claim in the
  section was re-verified against the built compiler.*
- [x] **D3.** TODO doc corrections: regression section (resolved by F1),
  "mangling is already safe" inverted (F2), parameters rationale wrong (see
  Part III W5). *Done in P9; the TODO doc was rewritten to be only the remaining
  value-decl work list that P10 will consume. The regression section now says it
  is resolved and no longer a blocker; the mangling paragraph says identical
  mangling of distinct canonical types is the hazard and points at F2 and
  Part IV; the parameter item gives W5's rationale (a folded parameter would
  change the enclosing function's overload identity and mangling —
  `ProcessDeclAttributes` does run before the prototype is built); the reader
  audit was re-verified and its line numbers refreshed; item 5's
  pointer-identity dedup suggestion is replaced by what P5 actually did
  (`areEquivalentCapabilityAttrs`, profile-based); and the C half of the
  `mergeFunctionTypes` prerequisite is marked landed with only
  `MergeVarDeclTypes` left.*
- [x] **D4.** `Attr.td` now advertises `TypedefName` subjects while
  `RequiresCapability`/`LocksExcluded` docs remain `[Undocumented]`. *Done in
  P9: `RequiresCapabilityDocs` and `LocksExcludedDocs` written in `AttrDocs.td`
  and wired up in `Attr.td`, and the four existing capability doc blocks
  (`AssertCapability`, `AcquireCapability`, `TryAcquireCapability`,
  `ReleaseCapability`) extended with the function-pointer and typedef/alias
  forms plus an example each. All six link to `ThreadSafetyAnalysis.html`.*

---

## Part III — Finishing value-decl folding (TODO items 2–5): study results

**Blockers**

- **B1 (= F14 merge half). Redeclaration of a folded variable**: hard
  `redeclaration with a different type` error in C++
  (`Sema::MergeVarDeclTypes` uses bare `hasSameType`, `SemaDecl.cpp:4604`);
  in C, `mergeFunctionTypes` returns the attr-free side → requirement
  silently lost. So the `mergeFunctionTypes` work is a *prerequisite* for
  VarDecl folding, not a later item. `FieldDecl` is immune (never
  redeclared) → **fields can ship before variables**. *The C half is
  **satisfied** by P6 (F14d): `mergeFunctionTypes` now unions the requirement
  sets for redeclaration merging, so the C twin of the redecl-without-attr
  test will keep the requirement (already true today for a function parameter
  of a caps-carrying typedef; see
  `clang/test/Sema/thread-safety-type-capability-merge.c`). The C++ half is
  not: `MergeVarDeclTypes` still compares with `hasSameType` and would reject
  a folded variable's attr-free redeclaration. That is untestable until W4
  folds `VarDecl`s at all, so it stays part of W4 — the fix there is to try
  `mergeCapabilityAttrs`-based merging before diagnosing, using the same union
  rule.*
- **B2 (= F3/F13). Dependent args fold unsubstituted** — must fix first.

**Key design recommendations (reversing two TODO assumptions)**

1. For value decls, **fold without `dropAttr`** (keep decl attrs; add dedup —
   F7 — in `handleCall` and `getTryAcquireCapabilityAttrs`). Dropping is what
   breaks redecl attr inheritance, `-ast-dump`, and future function-decl
   folding. Keep the drop for typedefs (decl is never the callee; printer
   already emits from the type). *The dedup half landed in P5, so this
   prerequisite is met: a value decl that keeps its attributes and also has
   them in its type is checked once.*
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

- [x] **W1** = F3/F13 guard hardening (prerequisite). Landed with P3.
- [ ] **W2. Fold `FieldDecl`** (lowest risk, ships first). Timing verified
  safe: late-parsed member attrs attach (`ParseDeclCXX.cpp:3725`) after
  `ActOnFinishCXXMemberSpecification` but *before*
  `ParseLexedMemberInitializers` (`:3731`); triviality/layout unaffected
  (pointer size/align unchanged; `TypeInfo` keyed on canonical `Type*`). In C,
  TSA field attrs are not late-parsed (`ParseDecl.cpp:4958`), and the
  non-late path (`SemaDecl.cpp:19630`) precedes record completion.
- [~] **W3** = F14 `mergeFunctionTypes`/`MergeVarDeclTypes` union (blocker B1).
  *`mergeFunctionTypes` landed with P6 (F14d), which unblocks the C half of
  B1. `MergeVarDeclTypes` is untestable until there is a folded `VarDecl` to
  redeclare, so it moves into W4; `mergeCapabilityAttrs` is there to be
  reused.*
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
  (`SemaOverload.cpp:1373`) and its mangling. That, not build order, is why
  params must stay on the decl path. *TODO text corrected in P9 (D3).*
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
(Determines F2, and colors F14, F15, F16. It also colored F4, which
P7 has since fixed within the current design.)

- **Option A — keep canonical (current design) + make manglers emit them**
  (vendor-extension qualifier, like C++17 `noexcept`-in-type). Pros: full
  propagation through `auto`, templates, deduction — the point of the series;
  honest linkage (annotated/unannotated mismatches become link errors instead
  of silent ODR traps). Cons: mangling arbitrary mutex *expressions* is ugly;
  annotating a typedef changes the mangling of every function that takes it —
  an ABI cliff for adopters.
- **Option B — demote to type sugar** (AttributedType-style node; analysis
  desugars). Pros: F2/F15/F16 evaporate, and the ordering constraint P7
  imposes (a typedef's requirement must be known before the typedef is used)
  goes away with them; no ABI impact; much
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
   tests. ✔ (F3 partial: see its entry for the in-template-use limitation.)
4. **P4**: F5 qualifier preservation; F8 using-alias support; tests. ✔
   (Residuals recorded: F5's unrebuildable sugar, F8a alias templates, F8b
   attribute-in-declarator, F8c alias printing.)
5. **P5**: F7 dedup + F7b perf; F11 callee-expression fallback (best-effort);
   tests. ✔ (Residual recorded: F11's callees that no declaration is reachable
   from. D2's F11 half done; the F5 two-step-conversion note was investigated
   and reattributed to a pre-existing Sema duplicate.)
6. **P6**: F14 identity consistency (ODRHash, structural equivalence,
   ASTImporter, mergeFunctionTypes union) + F15 + F16 transparency; T1
   Modules test. ✔ (F16 decided the other way: strictness, matching function
   effects — see its entry. `MergeVarDeclTypes`, W3's C++ half, deferred to
   W4 for want of anything to test it with.)
7. **P7**: F4 stale member-typedef canonical — investigate; fix or document
   as known limitation tied to Part IV. ✔ Fixed by not late-parsing a
   capability attribute on a typedef declarator (matching what alias
   declarations already did), plus a fold-time refusal to change a type that
   has already been handed out. Residuals recorded: the declaration-specifier
   attribute position, and F3's template-pattern-internal uses.
8. **P8**: F12 ignored-attr diagnostic; F17; F18 style sweep. ✔ (F12 also
   covers F8a's alias templates, F4's declaration-specifier residual and a
   newly found unprototyped-function-type case, and drops the attribute it
   reports. F19 assessed and deliberately left alone — see its entry.)
9. **P9**: D1–D4 docs; T3; final test sweep. ✔ Docs only — no compiler
   behavior change. Note for the record: the release notes now live in
   `ReleaseNotes.md`, not `.rst`.
10. **P10**: W2 field folding, then W4 variable folding (W3 landed in P6),
    with the Part III test plan. Closes out the ValueDeclFolding TODO's items
    2, 3, 5, 6; item 4 (functions) stays deferred (W6).
