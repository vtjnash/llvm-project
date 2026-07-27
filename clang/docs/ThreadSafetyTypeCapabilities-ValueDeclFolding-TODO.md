# Follow-up: fold capability requirements into the type of function declarations

## Status

The `[TSA][1..18/N]` commit series makes thread-safety capability requirements
(`requires_capability`, `acquire_capability`, `release_capability`,
`try_acquire_capability`, `assert_capability`, `locks_excluded`, and their
shared variants) part of the **function type** when they are written on a
function-pointer **typedef or alias declaration** (`[TSA][2/N]`) or on a
function-pointer **variable or field** (`[TSA][18/N]`). The requirement is
stored in `FunctionType::FunctionTypeExtraAttributeInfo::CapabilityAttrs`, is
part of the canonical type (in `FunctionProtoType::Profile`), round-trips
through serialization, is hashed by `ODRHash`, compared by structural
equivalence, imported by `ASTImporter`, merged by
`ASTContext::mergeFunctionTypes` and `Sema::MergeVarDeclTypes`, and is read
back by the analysis from the callee expression's type. This lets the
requirement survive `auto`, template instantiation, and same-type assignment.

**Everything this document originally listed as value-declaration work is
done.** What remains is one item, below: folding requirements into **function
declarations**. It is not a loose end of the value-decl work but a separate
design step, gated on the canonical-vs-sugar decision in Part IV of
`ThreadSafetyTypeCapabilities-ReviewFindings.md`.

Two subjects are **deliberately** left on the declaration-attribute path
introduced by llvm/llvm-project#191187, and are not future work:

- **Parameters.** `ActOnParamDeclarator` runs `ProcessDeclAttributes`
  (`SemaDecl.cpp:15858`) *before* `GetFullTypeForDeclarator` collects the
  parameter types (`SemaType.cpp:5265`), so a folded parameter *would*
  propagate into the enclosing prototype. That is precisely the problem: it
  would change the enclosing function's own type, and hence its overload
  identity (`SemaOverload.cpp:1373`) and its mangling. The declaration path
  already covers parameters, and the typedef path covers the common case of a
  parameter written with an annotated type.
- **Everything that is not a function (pointer) type.** The subject check
  rejects those before the fold is reached.

## What the value-declaration work settled

Recorded here because the function-declaration item inherits all of it.

- **Do not drop the attributes from a value declaration.** The typedef fold
  drops them (a typedef is never itself the callee, and `TypePrinter` emits
  them from the type), but a value declaration *is* the callee, and dropping
  would break redeclaration attribute inheritance, `-ast-dump`, and this item.
  They are kept, and the resulting duplicate is suppressed by
  `clang::areEquivalentCapabilityAttrs`, which compares two requirements by the
  same profile the folding set uses: spelling-insensitive, but sharedness-,
  genericness- and success-value-sensitive.
- **Set only `ValueDecl::setType`**, leaving the `TypeSourceInfo` alone, and
  read the declaration's own type rather than the TSI's (the two genuinely
  differ for value declarations, e.g. after address-space adjustment).
- **An obstacle to the fold is not an error for a value declaration.** Unlike a
  typedef, whose unfoldable attribute would do nothing at all and is therefore
  reported as ignored, a value declaration's attribute keeps working from the
  declaration. See the behavior matrix in Part III of the review findings.
- **`TypedefNameDecl` is not a `DeclaratorDecl`**, so the two paths stay
  separate inside `Sema::foldCapabilityAttrsIntoType`.
- **Merging must be a union, with one shared order.** The order of a type's
  requirement list is part of the type's identity, so a merge has to give both
  sides the *same* list, not each side the other's requirements.
- **Late-parsed attributes need their own fold hooks**: the parser's
  (`Sema::ActOnFinishDelayedAttribute`), template instantiation's
  (`TemplateDeclInstantiator::VisitFieldDecl`,
  `Sema::BuildVariableInstantiation`), and `Sema::InstantiateClass`'s
  late-attribute loop, which is where a *class template's* members get theirs.

## Remaining item: function declarations

Override matching is safe, but:

- **Redeclaration is not.** `MergeFunctionDecl` → `err_conflicting_types`. This
  needs the same union treatment `Sema::MergeVarDeclTypes` got in
  `[TSA][18/N]`, and `Sema::mergeCapabilityAttrsIntoVarType` is the shape to
  copy.
- **The entry lockset would go blind.** `runAnalysis`
  (`ThreadSafety.cpp:2839`) seeds it from the analyzed function's *own*
  attributes (`D->attrs()`, `:2903`) and its parameters' (`Param->attrs()`,
  `:2938`). With the attributes kept on the declaration that keeps working, but
  it should learn to read the function's type as well, or a requirement that
  arrived through a merge, or through an annotated typedef, will not seed the
  lockset.
- **`checkThisInStaticMemberFunctionAttributes` ordering** has to be preserved.

Name mangling is **not** a reason to feel safe here — it is the hazard.
`CapabilityAttrs` are part of the canonical type but the Itanium mangler
ignores `FunctionTypeExtraAttributeInfo`, so two functions whose types differ
only in their requirements mangle identically: two definitions are a hard
`error: definition with same mangled name`, and declaration-only overload sets
are a silent link trap. This already applies to a caps-carrying *typedef* used
as a parameter type, and folding function declarations would extend it to the
functions themselves. See finding F2 and the Option A / Option B decision in
Part IV of `ThreadSafetyTypeCapabilities-ReviewFindings.md`; **that decision
should be made before this item is attempted.**

## Where the rest lives

This is a work list, not a design record. The design questions, the branch
review, the reader audit, the behavior matrix and the status of every finding
live in `ThreadSafetyTypeCapabilities-ReviewFindings.md`: the open
canonical-vs-sugar decision is its Part IV, and the value-declaration study is
its Part III (items W1–W6). The user-facing description of the feature is in
`ThreadSafetyAnalysis.md`.

## Two further items, from the aotcompile.cpp report

### Lambda-to-function-pointer conversions are only half covered

A captureless lambda whose `operator()` states a requirement loses it when the
closure converts to a plain function pointer, exactly as a named function does.
`getConvertedFunctionDecl` now recognizes that shape -- the conversion is a
`CXXMemberCallExpr` to the closure's implicit `CXXConversionDecl`, and the
declaration the resulting pointer reaches is `getLambdaCallOperator()` -- and
the message names the lambda rather than `operator()`.

**Only the assignment seam is covered.** Initialization, argument passing and
`return` reach the conversion through `InitializationSequence::Perform`, which
does not route a user-defined conversion past
`Sema::PerformImplicitConversion`, the seam this check hangs off
(`SemaExprCXX.cpp`). Covering them needs either a hook in the
`SK_UserConversion` step or a check at the point the conversion operator call
is built (which would also see explicit casts, so the opt-out would have to be
re-established there).

Note this does not affect the common type-erasure case. Passing a lambda to a
by-value template parameter (`std::function`, `llvm::unique_function`) deduces
the *closure* type, so no function pointer conversion happens at all and there
is nothing to report; the call inside the lambda body is still checked against
the lambda's own annotation.

### Injecting a declaration's requirements into the decayed pointer type

Tempting, and it would unify a lot: `auto p = annotated_fn;` would give `p` a
type that carries the requirement, so calls through `p` would be checked, and
the lambda gap above would collapse into an ordinary type-level mismatch that
the existing machinery already reports at every seam.

**It cannot be done under the current mangling design.** Capability
requirements are part of the canonical type but the Itanium mangler does not
emit them, so two types differing only in a requirement mangle identically.
That is already a hard error when both are instantiated:

```c++
template <class T> void g(T) {}
template void g<void (*)(void)>(void (*)(void));
template void g<void (*)(void) REQUIRES(mu)>(void (*)(void) REQUIRES(mu));
// error: definition with same mangled name '_Z1gIPFvvEEvT_' as another definition
```

Today that requires deliberately writing both types. If decay injected the
declaration's requirements, `g(annotated_fn)` and `g(plain_fn)` in one
translation unit would produce it from ordinary code -- passing an annotated
and an unannotated function to the same template would stop compiling.

So this item is gated on Part IV of
`ThreadSafetyTypeCapabilities-ReviewFindings.md`: it is only available under
**Option A** (make the manglers emit the requirements), and it is one more
argument for Option A, since the propagation it would buy is exactly what the
series is for. Under Option B (demote to sugar) it is unavailable by
construction. Prototype the expression mangling first, as that entry already
recommends.

