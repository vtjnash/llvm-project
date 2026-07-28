# TSA type-carried capabilities: design record

Design rationale for the `jn/tsa-typedef-capability` series. This is **not** user
documentation of the feature — that is
[ThreadSafetyAnalysis.md](ThreadSafetyAnalysis.md), which every claim here defers
to. What this file records is *why* the design is shaped the way it is, which
constraints forced each choice, what is deliberately not done, and how the result
relates to earlier approaches, so that none of it has to be reconstructed at
review time.

Commits are referenced by their `[TSA][N/N]` labels rather than by hash, since
the series is rebased.

---

## Purpose

Clang's Thread Safety Analysis could only attach capability requirements
(`REQUIRES`, `ACQUIRE`, `RELEASE`, `TRY_ACQUIRE`, `ASSERT_CAPABILITY`,
`EXCLUDES`) to *declarations*. A call through a function pointer was therefore
unchecked: `void f() REQUIRES(mu)` could be annotated, but the moment `f` became
a `void (*)()` the requirement vanished, silently. Callback-based APIs had no way
to state that every callback passed to them must hold `mu`.

This series makes a capability requirement part of the **function type**, so it
travels with the value.

## Relationship to earlier approaches

Earlier attempts at function-pointer thread-safety attributes modeled the
requirement as type **sugar** — an `AttributedType` wrapping the function type.
Review of that approach concluded it could not work: sugar is looked through, so
the requirement cannot survive `auto`, templates or deduction, and — decisively —
a mismatch between two function-pointer types cannot be diagnosed at all. The
conclusion reached there was that the requirement has to become part of the
`FunctionType` itself, stored in trailing storage on `FunctionProtoType`, so that
type mismatches are diagnosable.

**That is precisely what this series does.** It also ships the conversion
diagnostics that conclusion was reached in service of, extends the treatment to
all six attribute kinds and to typedefs and aliases as well as values, and pays
the costs that making a requirement canonical entails — type identity,
merging, and mangling. The full comparison, including the one capability earlier
approaches have that this series does not, is at the end of this document.

## General shape

Requirements live in `FunctionProtoType`'s `ExtProtoInfo::ExtraAttributeInfo`,
alongside `noreturn`, `cfi_salt` and function effects, and are part of the
**canonical** type. Attributes written on a function-pointer typedef, alias,
variable or field are *folded* out of the declaration and into its type; the
analysis then reads the callee expression's type rather than only the callee's
declaration.

Type identity is the load-bearing part. One profile function,
`profileCapabilityAttr` (`clang/lib/AST/AttrImpl.cpp`), defines when two
requirements are the same, and five consumers must agree with it:

- `FunctionTypeExtraAttributeInfo::Profile` (the type folding set),
- `ODRHash`,
- `ASTStructuralEquivalence`,
- `ASTImporter`,
- the merge and de-duplication helpers.

Any dimension profiled in one place and not another is a silent
wrong-type-selected bug, so the rules below are stated once and shared.

## Features

### Where a requirement can be written and folded

- Function-pointer `typedef`s and C++ `using` aliases, including member and
  template-instantiated ones.
- Function-pointer **variables** and **fields** (`[TSA][18/N]`), including static
  data members and NSDMIs.
- Propagation through `auto`, templates and deduction, which is what the
  canonical choice buys.

### Analysis

- Calls through folded types are checked: direct calls, `(*pp)()`, `tab[0]()`,
  `s->tab[0]()`, member calls, and implicit destructor calls.
- Try-acquire is honored through the type, including on branch terminators.
- A requirement stated on both a declaration and its type is reported once.

### Conversion diagnostics

- `-Wthread-safety-conversion-drop` and `-Wthread-safety-conversion-add`, under
  `-Wthread-safety-conversion`, in the `-Wthread-safety` umbrella.
- Requirements carried by *declarations* are accounted for, since function
  declarations and parameters do not fold (see Limitations). That includes
  captureless lambdas whose `operator()` is annotated.
- Explicit casts opt out, at every seam.
- Messages name the requirement in full (`requires_capability(!mu)`) and lead
  with the function when both types print identically — which happens exactly
  when the requirement comes from a declaration rather than from either type.
  The rendering is shared with the type printer
  (`printCapabilityAttrRequirement`).

### Type-identity plumbing

- Conversion transparency in both directions; `?:` intersects the two operands'
  requirements; C's `mergeFunctionTypes` unions them.
- Typedef redefinition takes the union, reported under the opt-in
  `-Wthread-safety-typedef-merge`.
- ODR hashing, structural equivalence, AST import, and PCH/module serialization
  all carry requirements.
- `-fthread-safety-capability-mangling=error|warn|ignore|mangle`, default
  `error`.

## Design decisions and the constraints that forced them

Each rule below exists because the alternative was tried, or was reachable, and
does not work. They are the substance of the design; the code comments point back
at them.

### Identity encodes semantics, never spelling

Sharedness and genericness are not attribute *kinds* — `ACQUIRE(mu)` and
`ACQUIRE_SHARED(mu)` are both `attr::AcquireCapability`, differing only in the
spelling index — and try-acquire's success value is a separate argument that the
generic argument accessor does not report. So identity must profile the attribute
kind **plus** the sharedness/genericness the spelling encodes
(`getCapabilityAttrSemantics`) **plus** the success value **plus** the capability
arguments.

It must equally *not* profile the spelling itself, because differently spelled
synonyms (`exclusive_locks_required` and `requires_capability`) state the same
requirement and have to unify. Profiling the kind and arguments alone makes
semantically different types collide in the folding set; profiling the spelling
makes identical requirements distinct. The dimension list is exactly the
difference between those two failures.

Each attribute's argument stream is separated by its argument count, with a
sentinel for a null argument, so that adjacent attributes' arguments cannot be
read as one another's.

### Order is part of identity, so merging cannot compare sets

Requirement lists are profiled in order, which means a list is only equal to an
identically ordered one. Any merge path therefore has to hand *both* types one
and the same list and then compare what remains, rather than ask whether the two
sets are equivalent — a set comparison accepts two orderings that the type system
considers different types, and a merge built on it will reject the ordering
difference it was written to absorb. The set-comparison helpers exist, and are
deliberately restricted to unioning and de-duplication, where order genuinely
does not matter.

### Success values compare by value, and without evaluation

`try_acquire_capability(true, mu)` and `try_acquire_capability(1, mu)` are the
same requirement, and the analysis evaluates both identically, so the type system
must not distinguish them. Comparison is therefore by value, normalized to a
narrowest-signed `APSInt`.

It reads literal forms rather than invoking the constant evaluator, for two
reasons that are properties of the consumers rather than preferences: `ODRHash`
holds no `ASTContext` and so could not apply an evaluator-based rule at all,
which would break the "all five agree" invariant; and profiling runs *inside* the
type folding set, including on rehash, where creating types is not acceptable. A
non-literal constant expression consequently still compares syntactically — two
spellings of one value stay distinct, which is conservative and never wrong.

### A typedef must name one type from its first use

If a requirement is folded into a typedef's underlying type after some other
declaration has already named that typedef, the earlier declaration keeps a type
whose canonical was computed pre-fold, and the two disagree. Derived types make
this unrecoverable: a pointer, an array or a function type bakes its canonical in
at creation, so healing the interned typedef node afterwards fixes direct uses
while leaving derived ones stale — an inconsistency no invariant can describe.

The fold therefore happens at the declaration, before anything can name it, which
means a capability attribute on a typedef declarator is not late-parsed. This
matches what alias declarations always did. The price is that a typedef's
requirement sees only what precedes it, so it cannot name a member declared later
in the same class; that is the same rule alias declarations and namespace scope
already had. A fold that would arrive too late is refused outright rather than
performed, so the invariant holds by construction.

### Only requirements that a type can represent are folded

A requirement whose argument is object-relative — `this`, a sibling member, a
parameter — cannot be expressed in a type, since the type has no object to
resolve it against. Such requirements stay on the declaration, where they are
fully checked; this is the original supported form and must keep working
unchanged. Dependent arguments are deferred and the fold retried once
substitution has happened, so a template's requirement is folded per
instantiation with its own substituted capability.

When a fold is impossible for a reason that is *not* going to be retried, the
attribute is diagnosed as ignored rather than left to sit on a declaration that
no analysis path reads. Silence there is the worst outcome available: the user has
written a requirement and would have no way to learn it means nothing.

### Soundness, not symmetry, defines the conversion rule

The two conversion directions are not mirror images of one another by
precondition versus postcondition, which is the classification the design first
reached for. `release_capability` is the one postcondition that *removes* from
the lockset, so losing it is unsound in the way that matters:

```c++
mu.Lock();
void (*p)(void) = unlock_fn;   // RELEASE(mu) dropped, silently
p();                           // really unlocks; analysis: still held
x = 1;                         // x GUARDED_BY(mu), really unprotected
```

Nothing is reported, and the access is genuinely unprotected — a missed race, not
a missed diagnostic. `acquire`, `assert` and `try_acquire` only add to the
lockset, so losing one leaves the analysis believing *less* and yields false
positives at worst.

The rule is therefore stated as the property that matters — report a conversion
that could make the analysis believe a capability is held when it may not be —
as `capabilityAttrLossIsUnsound` and `capabilityAttrGainIsUnsound`. That is
losing `requires`/`excludes`/`release` and gaining `acquire`/`assert`/
`try_acquire`: two mirror-image sets, with `release` and `acquire` swapping
sides.

### A gained caller-constraint is not reported

`requires_capability` and `locks_excluded` constrain the *caller*. A function that
does not state one is simply willing to be called with more held than it needs,
and every call through the pointer is still checked against what the type says.
Reporting the gain makes the ordinary shape of a callback API — passing an
unannotated function to a parameter whose type states a requirement — warn with
nothing wrong, for no safety benefit. The postcondition kinds stay reported in
both directions, because they tell the analysis what the callee *did*, and
claiming one the function does not perform is unsound rather than merely
over-constraining.

### Redefining a typedef takes the union

Because the requirement is part of the canonical type, redefining a typedef with
a requirement the earlier definition did not state is otherwise a hard error —
which makes impossible the one pattern the feature most needs to support:
annotating a callback type declared by a header that cannot be edited, by
repeating its typedef with the attribute. The union is monotonic, so it cannot
weaken an annotation, and either declaration order produces the same result; a
redefinition differing in anything else is still an error. The difference is
reported under an opt-in warning that is deliberately *not* part of
`-Wthread-safety`, since the pattern it flags is the intended one.

This reverses an earlier decision to keep redefinition strict by analogy with
`noexcept` and function effects. The analogy holds formally — all are
type-carried, analysis-only properties — but those properties are not ones a user
retrofits onto a third-party declaration, and the adoption path is what decides
the question.

### The conversion check sits where every initialization form passes

A user-defined conversion — the closure-to-function-pointer case — does not reach
the seam where the other conversions hang, so the check is performed at the step
that initialization, assignment, argument passing and `return` all funnel
through. It compares against the type ultimately being initialized rather than
the conversion operator's own return type: a closure's conversion operator
returns a plain function pointer, so comparing against that reports a drop even
when the destination did state the requirement. Explicit casts are honored at
this seam as at the others.

### Requirements stated twice are reported once

A value declaration keeps its attributes when they are folded, because the
analysis reads declaration attributes on paths of its own and dropping them would
break attribute inheritance across redeclarations and `-ast-dump`. That makes the
same requirement reachable twice, so de-duplication uses the same notion of
equality that type identity does — spelling-independent, but sensitive to
sharedness, genericness and success value, so that a shared and an exclusive
requirement on one capability remain two diagnostics.

Typedefs are the exception: their attributes are dropped once folded, because a
typedef declaration is never itself the callee and the printer emits the
requirement from the type.

### Printed output re-parses

The attributes are real `Attr *`s in the type, not a placeholder, so printing
emits every argument — including the success value — and the written spelling.
Output is normalized to the GNU syntax, because the C++11 spelling is not
accepted in the trailing position of a function declarator, which is the only
place a type can carry the attribute. Diagnostics and the type printer share one
requirement-rendering routine so the two cannot drift.

### The canonical choice creates an ABI hazard, so the policy is explicit

Two types differing only in their requirements are distinct types for overloading
and templates but mangle identically, so two definitions can collide on one
symbol. `-fthread-safety-capability-mangling` makes the response a stated policy
rather than an accident: `error` by default, `mangle` to include requirements in
mangled names so the collision cannot arise, and `warn`/`ignore` to keep the
first definition — which is only safe when the two definitions are the same
template expanded two ways, and so is narrowed to that case. It is one
`LangOption` rather than a `LangOption` and a `CodeGenOpt`, so the mangler and
CodeGen cannot disagree. The encoding is keyed on attribute kind plus semantics,
matching the identity rules above, so synonyms mangle identically and order is
part of the encoding.

`noreturn` and function effects are unmangled for the same reason and produce the
same collision today, so the option covers them too.

## Limitations

- **Function declarations and parameters do not fold**; their requirements stay
  on the declaration. Folding function declarations would blind the analysis's
  entry-lockset seeding, which reads them from the declaration, and breaks
  redeclaration merging. Folding parameters would propagate into the enclosing
  prototype and change that function's overload identity and mangling.
- **Mangling is opt-in**, so by default two types differing only in their
  requirements still share a symbol; that is an error, or first-definition-wins
  under `warn`/`ignore`.
- **Templates**: a use of an annotated typedef from *inside* the pattern does not
  see a dependent requirement, because the pattern's `TypedefType` is not
  instantiation-dependent and substitution never reaches the instantiated, folded
  typedef. Alias templates, which are never instantiated as declarations, can
  never retry a deferred fold — diagnosed, not silent.
- **Spelling**: the C++11 `[[clang::…]]` spelling is not accepted in the trailing
  type position; alias-declaration output is not round-trippable; sugar around
  the pointer level — a typedef naming the function-pointer type — is lost in the
  rebuild, visible only in `-ast-dump`.
- **Unchecked callees**: a callee reachable from no declaration (`get()()`,
  `((cb_t)p)()`) is still unchecked. Covering it needs nameless variants of three
  diagnostics that currently stream a `NamedDecl`.
- **Object-relative requirements** stay declaration-scoped, and are excluded from
  conversion comparison so the long-standing pattern does not warn.
- **Non-literal constant** success values still compare syntactically.

## Future work and open decisions

1. **Mangling default.** The policy flag exists; whether `mangle` should ever
   become the default is an ABI question, and it gates folding function
   declarations.
2. **Folding function declarations.** Needs a union-merge policy for function
   redeclaration and a fix for entry-lockset seeding. It would subsume the
   declaration-attribute handling the conversion check does today, and make the
   identical-types message shape disappear.
3. **Accepting the attribute in type position.** A feature in its own right, not
   a spelling fix: the argument checker takes a `Decl *` and uses it for the
   implicit-`this` and parameter-index forms, which have no answer in a type
   position, and a type-position attribute is parsed immediately with no
   declarator to hang late-parsed arguments on. This is the one capability
   earlier approaches have that this series does not.
4. **Upstreaming shape.** The series is too long to review as-is; it wants
   collapsing into a handful of logical patches, with the conversion diagnostics
   plausibly separate from the type-carrying mechanism.
5. **Smaller items**: alias-declaration printing round-trip; routing a type-id
   declarator's sliding attributes into the alias-declaration path.

---

## Comparison with earlier approaches

The existing upstream attempt at function-pointer thread-safety attributes makes
`RequiresCapability` a declaration-or-type attribute and wraps the type in an
`AttributedType`. It has been open and unmerged since 2023, and follows an
earlier abandoned attempt on the same lines.

### Sugar versus canonical

| | Sugar approach | This series |
|---|---|---|
| Storage | `AttributedType` wrapper | `FunctionProtoType` trailing storage |
| Attribute kinds | `RequiresCapability` only | all six |
| Subjects | function-pointer value declarations | typedefs, aliases, variables, fields |
| Survives `auto`/templates/deduction | no — sugar is looked through | yes |
| Conversion mismatch diagnostics | none | both directions, soundness-based |
| Identity machinery touched | none needed | folding set, ODR hash, structural equivalence, AST import, type merging, mangling policy |

The two are not variations on one design; they are the two ends of a single
decision. Sugar keeps the change small and pays none of the costs this series
absorbs — no mangling collision, no ODR hazard, no typedef-redefinition or `?:`
question, and no possibility of a folding-set collision, since a sugar node does
not participate in type identity at all. That cheapness is real. It is also what
makes the diagnostics impossible: a property that is looked through cannot be
compared between two types, so a conversion that drops a requirement cannot be
detected, and the requirement does not survive the deduction and `auto`
propagation that make the feature useful on generic code.

Two conclusions from the review of that approach shaped this one directly. The
first is that the requirement must be part of the `FunctionType`, with trailing
storage on `FunctionProtoType`, precisely so that mismatches are diagnosable —
which is the design implemented here. The second is that the conversion
diagnostics belong in the same body of work rather than a follow-up, on the
grounds that retrofitting them later is far harder; that is why this series ships
them, and it was in building them that the `release` soundness asymmetry above
came to light.

Two implementation details worth carrying forward. The storage question that
stalled the earlier work — that a pointer in `FunctionType::ExtInfo` would grow
it unacceptably while a single bit would not suffice — has since been answered
upstream by `cfi_salt`, whose trailing-storage slot this series reuses; the
concern is no longer live. And the K&R `FunctionNoProtoType` case flagged as the
downside of trailing storage is reached here and diagnosed explicitly, as one of
the reasons a fold can be refused.

### Where the earlier approach is ahead

It accepts the attribute in **type position**
(`int EXCLUSIVE_LOCKS_REQUIRED(mu1) (*fp)(int);`), which this series does not:
here the attribute is written in declaration position and folded into the type.
The sugar approach has that syntax without the semantics; this series has the
semantics without that syntax. Any upstream comparison should say so plainly, so
that it does not read as a regression.

It also illustrates why the syntax is a separate piece of work: in a type
position there is no declaration, so the forms of the argument checker that need
one — the implicit-`this` form and the parameter-index form — have no defined
answer, and skipping them silently is not a rule. Supplying one is a new
diagnostic and a new question about where a requirement may appear (a parameter
type, a return type, a cast, a template argument), not a relaxation of an
existing check.

### Upstreaming implication

There is no function-pointer thread-safety support upstream today. The existing
pull request is the natural place to land this conversation: its review already
converged on the design implemented here, and it supplies ready-made answers to
the first questions any reviewer will ask — why not sugar, why trailing storage
on `FunctionProtoType`, and why the diagnostics ship in the same body of work.
