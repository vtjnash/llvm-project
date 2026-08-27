//===- ThreadSafety.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A intra-procedural analysis for thread safety (e.g. deadlocks and race
// conditions), based off of an annotation system.
//
// See http://clang.llvm.org/docs/ThreadSafetyAnalysis.html
// for more information.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/ThreadSafety.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/Analyses/ThreadSafetyCommon.h"
#include "clang/Analysis/Analyses/ThreadSafetyTIL.h"
#include "clang/Analysis/Analyses/ThreadSafetyUtil.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TrailingObjects.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace clang;
using namespace threadSafety;

// Key method definition
ThreadSafetyHandler::~ThreadSafetyHandler() = default;

/// Whether \p LEK diagnoses the end of the function rather than an interior
/// join.
static bool isEndOfFunctionLEK(LockErrorKind LEK) {
  return LEK == LEK_LockedAtEndOfFunction ||
         LEK == LEK_NotLockedAtEndOfFunction;
}

/// True if capability attributes on \p Param describe the function reached
/// through it rather than the argument bound to it.
///
/// Sema accepts capability attributes on a parameter for two unrelated
/// purposes: a scoped-lockable parameter, where the attributes describe the
/// locks the passed scope object holds, and a parameter naming a function to
/// call -- a function pointer or a function reference -- where they describe
/// the requirements of the function called through it.
static bool isCallbackParam(const ParmVarDecl *Param) {
  QualType T = Param->getType().getNonReferenceType();
  return T->isFunctionPointerType() || T->isFunctionType();
}

/// Issue a warning about an invalid lock expression
static void warnInvalidLock(ThreadSafetyHandler &Handler,
                            const Expr *MutexExp, const NamedDecl *D,
                            const Expr *DeclExp, StringRef Kind) {
  SourceLocation Loc;
  if (DeclExp)
    Loc = DeclExp->getExprLoc();

  // FIXME: add a note about the attribute location in MutexExp or D
  if (Loc.isValid())
    Handler.handleInvalidLockExp(Loc);
}

namespace {

/// A set of CapabilityExpr objects, which are compiled from thread safety
/// attributes on a function.
class CapExprSet : public SmallVector<CapabilityExpr, 4> {
public:
  bool contains(const CapabilityExpr &CapE) const {
    return llvm::any_of(
        *this, [&](const CapabilityExpr &CapE2) { return CapE.equals(CapE2); });
  }

  /// Push M onto list, but discard duplicates.
  void push_back_nodup(const CapabilityExpr &CapE) {
    if (!contains(CapE))
      push_back(CapE);
  }
};

class FactManager;
class FactSet;

/// This is a helper class that stores a fact that is known at a
/// particular point in program execution. Concretely, a fact is a capability,
/// along with additional information, such as where it was acquired, whether
/// it is exclusive or shared, etc.
///
/// A capability's facts come in two forms, and a FactSet keeps them apart:
///
///  * At most one *definite* fact per capability: the capability is held (or,
///    for a negative capability, provably not held). Its reentrancy depth
///    counts the levels acquired on top of the first; definite levels are
///    interchangeable -- releasing "a" level is releasing "the" level -- so
///    a counter is all they need.
///
///  * Any number of *conditional* facts per capability, one per originating
///    try-acquire call: "held if that call succeeded". Each keeps its own
///    lock kind. A branch on the call's result resolves exactly the facts
///    that name it as their origin, and a scope object releases exactly the
///    facts it created.
///
/// A capability is *held* if a definite fact exists, *may be held* if only
/// conditional facts exist, and *not held* if neither does. Every lookup
/// says which of the two it asks for (FactSet::findDefinite(),
/// FactSet::findConditional(), ...) rather than taking whichever fact the
/// set lists first.
///
/// Per capability that gives a ternary state: not-held, try-held (only
/// conditional facts), or held. Permitted transitions:
///
///   not-held --acquire-----------------------------------------> held
///   not-held --try-acquire (BuildLockset::handleCall)----------> try-held
///   try-held --branch on the try-acquire result: success edge--> held
///   try-held --branch on the try-acquire result: failure edge--> not-held
///   try-held --acquire or assert (addLock)---------------------> held
///   held -----branch on the originating try-acquire's result:
///             success edge (the failure edge is infeasible
///             and skipped at joins)-----------------------------> held
///   held -----join with a failed path of the same try-acquire,
///             when the join re-branches on its result
///             (intersectAndWarn)-------------------------------> try-held
///   held -----release------------------------------------------> not-held
///
/// The forms compose: a try-acquire over a held capability -- whatever its
/// reentrancy, since at runtime such a call fails rather than deadlocks --
/// adds a conditional fact beside the definite one, and a reentrant
/// acquire over a try-held capability adds the definite fact beside the
/// conditional ones. A branch's success edge folds the resolved conditional
/// fact into the definite one (one level deeper, or created), the failure
/// edge drops it, and a release unwinds the definite fact one level.
///
/// Branches are resolved in getEdgeLockset(); facts remember their
/// originating call so that later branches on the same result re-resolve
/// them and the join demotion above can identify them. A join of paths
/// holding the capability via different origins clears the merged fact's
/// origin (it is no longer determined by either result).
///
/// Both the join demotion and a branch's fact resolution rest on the
/// premise that a path not holding the capability carries a falsy stored
/// result. Negative facts police that premise: a join keeps a one-sided
/// negative fact as *weak* evidence (not-held on some path), and a
/// release that spends a stored result -- releasing a hold the call's
/// success had proved -- marks its negative fact with the call. A join
/// refuses to demote-and-carry across a spent result, and a branch's
/// success edge re-materializes a fact the analysis lost at a join (e.g.
/// around a loop) only when no surviving negative fact contradicts it.
///
/// Try-held means "held if the try-acquire succeeded", so it warns
/// wherever a definite state is required: it does not satisfy capability
/// requirements, it violates exclusions and negative requirements,
/// releasing it warns (may not be held), and a blocking acquire of it
/// warns (may already be held). Asserts and same-kind reentrant acquires are
/// exempt from the acquire warning: they legitimately acquire a
/// possibly-held capability. An acquire of the other kind (shared vs.
/// exclusive) warns even for a reentrant capability: reentrancy nests
/// levels of one kind. Two unresolved try-acquires of one capability are
/// tracked as two conditional facts, each resolved by the branch on its
/// own result; a repeat of the same call over its own unresolved fact
/// cannot be, and is diagnosed at the call.
///
/// When the analysis loses track of a try-held fact -- at a join with a
/// path that does not hold it, or at the end of the function -- the
/// try-acquire result was never checked and the capability may be leaked;
/// this is diagnosed in beta mode.
class FactEntry : public CapabilityExpr {
public:
  enum FactEntryKind { Lockable, ScopedLockable };

  /// Where a fact comes from.
  enum SourceKind {
    Acquired, ///< The fact has been directly acquired.
    Asserted, ///< The fact has been asserted to be held.
    Declared, ///< The fact is assumed to be held by callers.
    Managed,  ///< The fact has been acquired through a scoped capability.
  };

private:
  const FactEntryKind Kind : 8;

  /// Exclusive or shared.
  LockKind LKind : 8;

  /// How it was acquired.
  SourceKind Source : 8;

  /// Where it was acquired.
  SourceLocation AcquireLoc;

  /// The try-acquire call this fact originates from (or null), and whether
  /// the fact is conditional: held only on the paths where that call
  /// succeeded. A conditional fact always has an origin. A definite fact
  /// promoted on the call's success edge keeps it, recording the call whose
  /// success proved the hold, so that a later join with a path where the
  /// try-acquire failed can be recognized; a definite fact acquired or
  /// asserted unconditionally has none: its hold is not proved by any
  /// call's result. Merging holds proved by different calls clears it.
  llvm::PointerIntPair<const Expr *, 1, bool> TryLock;

  /// Whether this fact holds on only some, not all, paths into the current
  /// program point. Only negative facts are tracked this way: instead of
  /// leaving the intersection silently, a one-sided negative fact is kept
  /// in a join's merged set as a weak fact (intersectAndWarn()) --
  /// evidence that the capability was provably released, or a try-acquire
  /// of it provably failed, on at least one path. The try-held machinery
  /// consults it to refuse carrying (intersectAndWarn()) or
  /// re-materializing (getEdgeLockset()) a hold whose stored try-acquire
  /// result is stale on such a path. A weak fact proves nothing on all
  /// paths: it does not satisfy negative-capability requirements and
  /// cannot prove a branch edge infeasible.
  bool Weak = false;

  /// For a negative fact recorded by the release of a hold that a
  /// try-acquire call's success had proved (a fact promoted from that
  /// call, released by handleUnlock()): that call. The release spends the
  /// call's stored result -- the result stays truthy while the capability
  /// is no longer held -- so a later branch on it must not resurrect the
  /// hold: a join refuses to carry (intersectAndWarn()) and an edge to
  /// re-materialize (getEdgeLockset()) a fact of this call across this
  /// negative. Null for a negative from a plain release or from the call's
  /// failure edge (there the result is provably falsy, and a branch on it
  /// excludes those paths itself). Merges keep it like \c Weak: spent on
  /// some path is spent.
  const Expr *SpentTryLock = nullptr;

protected:
  ~FactEntry() = default;

public:
  FactEntry(FactEntryKind FK, const CapabilityExpr &CE, LockKind LK,
            SourceLocation Loc, SourceKind Src)
      : CapabilityExpr(CE), Kind(FK), LKind(LK), Source(Src), AcquireLoc(Loc) {}

  LockKind kind() const { return LKind;      }
  SourceLocation loc() const { return AcquireLoc; }
  FactEntryKind getFactEntryKind() const { return Kind; }

  bool asserted() const { return Source == Asserted; }
  bool declared() const { return Source == Declared; }
  bool managed() const { return Source == Managed; }

  /// Whether this is a conditional fact (see the class comment).
  bool tryHeld() const { return TryLock.getInt(); }
  /// The try-acquire call this fact originates from, or null.
  const Expr *tryLockCall() const { return TryLock.getPointer(); }

  /// Record that this fact originates from the try-acquire call \p Call,
  /// as a conditional fact if \p Conditional.
  void setTryLock(const Expr *Call, bool Conditional) {
    assert((Call || !Conditional) && "conditional fact without an origin");
    TryLock.setPointerAndInt(Call, Conditional);
  }

  /// Whether losing track of this fact warrants a diagnostic: an asserted
  /// or universal capability's hold is not something the analyzed code is
  /// expected to release, and a negative fact is not a hold at all.
  bool lossNeedsWarning() const {
    return !asserted() && !negative() && !isUniversal();
  }

  bool weak() const { return Weak; }
  /// Mark this fact as holding on only some paths (see \c Weak).
  void setWeak() { Weak = true; }

  const Expr *spentTryLock() const { return SpentTryLock; }
  /// Record that this negative fact spends \p Call's stored result (see
  /// \c SpentTryLock).
  void setSpentTryLock(const Expr *Call) { SpentTryLock = Call; }

  /// The fact's reentrancy depth; only lockable facts can be reentrant.
  virtual unsigned int getReentrancyDepth() const { return 0; }

  virtual void
  handleRemovalFromIntersection(const FactSet &FSet, FactManager &FactMan,
                                SourceLocation JoinLoc, LockErrorKind LEK,
                                ThreadSafetyHandler &Handler) const = 0;
  virtual void handleLock(FactSet &FSet, FactManager &FactMan,
                          const FactEntry &entry,
                          ThreadSafetyHandler &Handler) const = 0;
  virtual void handleUnlock(FactSet &FSet, FactManager &FactMan,
                            const CapabilityExpr &Cp, SourceLocation UnlockLoc,
                            bool FullyRemove,
                            ThreadSafetyHandler &Handler) const = 0;

  // Return true if LKind >= LK, where exclusive > shared
  bool isAtLeast(LockKind LK) const {
    return  (LKind == LK_Exclusive) || (LK == LK_Shared);
  }
};

using FactID = unsigned short;

/// FactManager manages the memory for all facts that are created during
/// the analysis of a single routine.
class FactManager {
private:
  llvm::BumpPtrAllocator &Alloc;
  std::vector<const FactEntry *> Facts;

public:
  FactManager(llvm::BumpPtrAllocator &Alloc) : Alloc(Alloc) {}

  template <typename T, typename... ArgTypes>
  T *createFact(ArgTypes &&...Args) {
    static_assert(std::is_trivially_destructible_v<T>);
    return T::create(Alloc, std::forward<ArgTypes>(Args)...);
  }

  FactID newFact(const FactEntry *Entry) {
    Facts.push_back(Entry);
    assert(Facts.size() - 1 <= std::numeric_limits<FactID>::max() &&
           "FactID space exhausted");
    return static_cast<unsigned short>(Facts.size() - 1);
  }

  const FactEntry &operator[](FactID F) const { return *Facts[F]; }

  /// Whether the analyzed function records any try-acquire call
  /// (TryAcquireCapsMap): only then can weak or spent negative facts exist,
  /// so their bookkeeping is skipped everywhere else.
  bool tracksTryAcquires() const { return TracksTryAcquires; }
  void setTracksTryAcquires(bool V) { TracksTryAcquires = V; }

private:
  bool TracksTryAcquires = false;
};

/// A FactSet is the set of facts that are known to be true at a
/// particular program point.  FactSets must be small, because they are
/// frequently copied, and are thus implemented as a set of indices into a
/// table maintained by a FactManager.  A typical FactSet only holds 1 or 2
/// locks, so we can get away with doing a linear search for lookup.  Note
/// that a hashtable or map is inappropriate in this case, because lookups
/// may involve partial pattern matches, rather than exact matches.
///
/// A capability may be represented by several facts at once (see FactEntry):
/// at most one definite fact, and any number of conditional facts, unique
/// per originating call. The accessors name which of the two they look for;
/// there is no lookup for "the" fact of a capability.
class FactSet {
private:
  using FactVec = SmallVector<FactID, 4>;

  FactVec FactIDs;

public:
  using iterator = FactVec::iterator;
  using const_iterator = FactVec::const_iterator;

private:
  template <typename Pred> iterator findIf(FactManager &FM, Pred P) {
    return llvm::find_if(*this, [&](FactID ID) { return P(FM[ID]); });
  }
  template <typename Pred>
  const FactEntry *findEntry(FactManager &FM, Pred P) const {
    auto I = llvm::find_if(*this, [&](FactID ID) { return P(FM[ID]); });
    return I != end() ? &FM[*I] : nullptr;
  }

public:
  iterator begin() { return FactIDs.begin(); }
  const_iterator begin() const { return FactIDs.begin(); }

  iterator end() { return FactIDs.end(); }
  const_iterator end() const { return FactIDs.end(); }

  bool isEmpty() const { return FactIDs.size() == 0; }

  // Return true if the set holds no definite positive capability. It may
  // hold negative or conditional facts, unlike isEmpty, which tests the
  // set itself.
  bool holdsNoCapability(FactManager &FactMan) const {
    for (const auto FID : *this) {
      if (!FactMan[FID].negative() && !FactMan[FID].tryHeld())
        return false;
    }
    return true;
  }

  void addLockByID(FactID ID) { FactIDs.push_back(ID); }

  FactID addLock(FactManager &FM, const FactEntry *Entry) {
    FactID F = FM.newFact(Entry);
    FactIDs.push_back(F);
    return F;
  }

  /// Remove the fact at \p It, moving the set's last element into its
  /// slot.
  void erase(iterator It) {
    *It = FactIDs.back();
    FactIDs.pop_back();
  }

  /// \name Lookup by identity
  /// The fact \p F itself, which a caller obtained from a lookup below.
  /// \{
  iterator findFactIter(FactManager &FM, const FactEntry &F) {
    return findIf(FM, [&](const FactEntry &FE) { return &FE == &F; });
  }

  bool removeFact(FactManager &FM, const FactEntry &F) {
    iterator It = findFactIter(FM, F);
    if (It == end())
      return false;
    erase(It);
    return true;
  }

  std::optional<FactID> replaceFact(FactManager &FM, iterator It,
                                    const FactEntry *Entry) {
    if (It == end())
      return std::nullopt;
    FactID F = FM.newFact(Entry);
    *It = F;
    return F;
  }

  std::optional<FactID> replaceFact(FactManager &FM, const FactEntry &Old,
                                    const FactEntry *Entry) {
    return replaceFact(FM, findFactIter(FM, Old), Entry);
  }
  /// \}

  /// \name Definite facts
  /// The one fact stating that \p CapE is held (or, negated, provably not
  /// held).
  /// \{
  iterator findDefiniteIter(FactManager &FM, const CapabilityExpr &CapE) {
    return findIf(FM, [&](const FactEntry &FE) {
      return !FE.tryHeld() && FE.matches(CapE);
    });
  }

  const FactEntry *findDefinite(FactManager &FM,
                                const CapabilityExpr &CapE) const {
    return findEntry(FM, [&](const FactEntry &FE) {
      return !FE.tryHeld() && FE.matches(CapE);
    });
  }

  const FactEntry *findDefiniteUniv(FactManager &FM,
                                    const CapabilityExpr &CapE) const {
    return findEntry(FM, [&](const FactEntry &FE) {
      return !FE.tryHeld() && FE.matchesUniv(CapE);
    });
  }

  const FactEntry *findDefinitePartialMatch(FactManager &FM,
                                            const CapabilityExpr &CapE) const {
    return findEntry(FM, [&](const FactEntry &FE) {
      return !FE.tryHeld() && FE.partiallyMatches(CapE);
    });
  }

  bool removeDefinite(FactManager &FM, const CapabilityExpr &CapE) {
    iterator It = findDefiniteIter(FM, CapE);
    if (It == end())
      return false;
    erase(It);
    return true;
  }
  /// \}

  /// \name Conditional facts
  /// The facts stating that \p CapE is held if a try-acquire call
  /// succeeded, one per originating call.
  /// \{
  iterator findConditionalIter(FactManager &FM, const CapabilityExpr &CapE,
                               const Expr *Origin) {
    return findIf(FM, [&](const FactEntry &FE) {
      return FE.tryHeld() && FE.tryLockCall() == Origin && FE.matches(CapE);
    });
  }

  const FactEntry *findConditional(FactManager &FM, const CapabilityExpr &CapE,
                                   const Expr *Origin) const {
    return findEntry(FM, [&](const FactEntry &FE) {
      return FE.tryHeld() && FE.tryLockCall() == Origin && FE.matches(CapE);
    });
  }

  /// The first conditional fact of \p CapE, whichever its origin.
  const FactEntry *firstConditional(FactManager &FM,
                                    const CapabilityExpr &CapE) const {
    return findEntry(FM, [&](const FactEntry &FE) {
      return FE.tryHeld() && FE.matches(CapE);
    });
  }

  bool anyConditional(FactManager &FM, const CapabilityExpr &CapE) const {
    return firstConditional(FM, CapE) != nullptr;
  }

  /// Whether every conditional fact of \p CapE has the lock kind \p Kind.
  bool conditionalsAllOfKind(FactManager &FM, const CapabilityExpr &CapE,
                             LockKind Kind) const {
    return llvm::all_of(*this, [&](FactID ID) {
      const FactEntry &FE = FM[ID];
      return !FE.tryHeld() || !FE.matches(CapE) || FE.kind() == Kind;
    });
  }

  /// Collect every conditional fact of \p CapE into \p Out, so that a
  /// caller can mutate the set while visiting them.
  void collectConditional(FactManager &FM, const CapabilityExpr &CapE,
                          SmallVectorImpl<const FactEntry *> &Out) const {
    for (FactID ID : *this)
      if (FM[ID].tryHeld() && FM[ID].matches(CapE))
        Out.push_back(&FM[ID]);
  }

  bool removeConditional(FactManager &FM, const CapabilityExpr &CapE,
                         const Expr *Origin) {
    iterator It = findConditionalIter(FM, CapE, Origin);
    if (It == end())
      return false;
    erase(It);
    return true;
  }

  void removeAllConditional(FactManager &FM, const CapabilityExpr &CapE) {
    llvm::erase_if(FactIDs, [&](FactID ID) {
      return FM[ID].tryHeld() && FM[ID].matches(CapE);
    });
  }
  /// \}

  /// The first fact of either form matching \p CapE: whether the
  /// capability is held or may be held.
  const FactEntry *findAny(FactManager &FM, const CapabilityExpr &CapE) const {
    return findEntry(FM, [&](const FactEntry &FE) { return FE.matches(CapE); });
  }

  /// \name Counterparts
  /// The fact of the same form as \p F -- definite, or conditional on the
  /// same call -- for \p F's capability: what a join pairs \p F with.
  /// \{
  iterator findCounterpartIter(FactManager &FM, const FactEntry &F) {
    return F.tryHeld() ? findConditionalIter(FM, F, F.tryLockCall())
                       : findDefiniteIter(FM, F);
  }

  const FactEntry *findCounterpart(FactManager &FM, const FactEntry &F) const {
    return F.tryHeld() ? findConditional(FM, F, F.tryLockCall())
                       : findDefinite(FM, F);
  }
  /// \}

  bool containsMutexDecl(FactManager &FM, const ValueDecl* Vd) const {
    auto I = llvm::find_if(
        *this, [&](FactID ID) -> bool { return FM[ID].valueDecl() == Vd; });
    return I != end();
  }
};

class ThreadSafetyAnalyzer;

} // namespace

namespace clang {
namespace threadSafety {

class BeforeSet {
private:
  using BeforeVect = SmallVector<const ValueDecl *, 4>;

  struct BeforeInfo {
    BeforeVect Vect;
    int Visited = 0;

    BeforeInfo() = default;
    BeforeInfo(BeforeInfo &&) = default;
  };

  using BeforeMap =
      llvm::DenseMap<const ValueDecl *, std::unique_ptr<BeforeInfo>>;
  using CycleMap = llvm::DenseMap<const ValueDecl *, bool>;

public:
  BeforeSet() = default;

  BeforeInfo* insertAttrExprs(const ValueDecl* Vd,
                              ThreadSafetyAnalyzer& Analyzer);

  BeforeInfo *getBeforeInfoForDecl(const ValueDecl *Vd,
                                   ThreadSafetyAnalyzer &Analyzer);

  void checkBeforeAfter(const ValueDecl* Vd,
                        const FactSet& FSet,
                        ThreadSafetyAnalyzer& Analyzer,
                        SourceLocation Loc, StringRef CapKind);

private:
  BeforeMap BMap;
  CycleMap CycMap;
};

} // namespace threadSafety
} // namespace clang

namespace {

class LocalVariableMap;

using LocalVarContext = llvm::ImmutableMap<const NamedDecl *, unsigned>;

/// A side (entry or exit) of a CFG node.
enum CFGBlockSide { CBS_Entry, CBS_Exit };

/// CFGBlockInfo is a struct which contains all the information that is
/// maintained for each block in the CFG.  See LocalVariableMap for more
/// information about the contexts.
struct CFGBlockInfo {
  // Lockset held at entry to block
  FactSet EntrySet;

  // Lockset held at exit from block
  FactSet ExitSet;

  // Context held at entry to block
  LocalVarContext EntryContext;

  // Context held at exit from block
  LocalVarContext ExitContext;

  // Location of first statement in block
  SourceLocation EntryLoc;

  // Location of last statement in block.
  SourceLocation ExitLoc;

  // Used to replay contexts later
  unsigned EntryIndex;

  // Is this block reachable?
  bool Reachable = false;

  // Whether the block is reachable only through infeasible edges (or
  // through other such blocks): analyzed for coverage -- diagnostics
  // inside it are real -- but its exit set carries provably dead state,
  // which downstream joins must not consume as if it were a live path
  // (runAnalysis()).
  bool CoverageOnly = false;

  const FactSet &getSet(CFGBlockSide Side) const {
    return Side == CBS_Entry ? EntrySet : ExitSet;
  }

  SourceLocation getLocation(CFGBlockSide Side) const {
    return Side == CBS_Entry ? EntryLoc : ExitLoc;
  }

private:
  CFGBlockInfo(LocalVarContext EmptyCtx)
      : EntryContext(EmptyCtx), ExitContext(EmptyCtx) {}

public:
  static CFGBlockInfo getEmptyBlockInfo(LocalVariableMap &M);
};

// A LocalVariableMap maintains a map from local variables to their currently
// valid definitions.  It provides SSA-like functionality when traversing the
// CFG.  Like SSA, each definition or assignment to a variable is assigned a
// unique name (an integer), which acts as the SSA name for that definition.
// The total set of names is shared among all CFG basic blocks.
// Unlike SSA, we do not rewrite expressions to replace local variables declrefs
// with their SSA-names.  Instead, we compute a Context for each point in the
// code, which maps local variables to the appropriate SSA-name.  This map
// changes with each assignment.
//
// The map is computed in a single pass over the CFG.  Subsequent analyses can
// then query the map to find the appropriate Context for a statement, and use
// that Context to look up the definitions of variables.
class LocalVariableMap {
public:
  using Context = LocalVarContext;

  /// A VarDefinition consists of an expression, representing the value of the
  /// variable, along with the context in which that expression should be
  /// interpreted.  A reference VarDefinition does not itself contain this
  /// information, but instead contains a pointer to a previous VarDefinition.
  struct VarDefinition {
  public:
    friend class LocalVariableMap;

    // The original declaration for this variable.
    const NamedDecl *Dec;

    // The expression for this variable, OR
    const Expr *Exp = nullptr;

    // Direct reference to another VarDefinition; for a merge ("phi"), the
    // definition on the first joined path.
    unsigned DirectRef = 0;

    // Reference to underlying canonical non-reference VarDefinition.
    unsigned CanonicalRef = 0;

    // For a merge ("phi") of two definitions, the definition on the second
    // joined path (DirectRef holds the first); 0 otherwise. A phi is its own
    // canonical definition and is opaque to lookupExpr(); it exists so that
    // a branch on a try-acquire result merged with a constant initializer
    // can still be resolved (see decodeTrylockCond()).
    unsigned PhiAlt = 0;

    // The map with which Exp should be interpreted.
    Context Ctx;

    // Whether this is the definition created at the variable's declaration
    // when it has no initializer: the variable's birth, holding an
    // indeterminate value. An invalidated reference has the same null
    // shape but stands for an unknown later value (chainAvoids()).
    bool UninitDecl = false;

    bool isPhi() const { return PhiAlt != 0; }
    bool isReference() const { return !Exp && !isPhi(); }

    void invalidateRef() { DirectRef = CanonicalRef = PhiAlt = 0; }

  private:
    // Create ordinary variable definition
    VarDefinition(const NamedDecl *D, const Expr *E, Context C)
        : Dec(D), Exp(E), Ctx(C), UninitDecl(!E) {}

    // Create reference to previous definition
    VarDefinition(const NamedDecl *D, unsigned DirectRef, unsigned CanonicalRef,
                  Context C)
        : Dec(D), DirectRef(DirectRef), CanonicalRef(CanonicalRef), Ctx(C) {}
  };

private:
  Context::Factory ContextFactory;
  std::vector<VarDefinition> VarDefinitions;
  std::vector<std::pair<const Stmt *, Context>> SavedContexts;
  // Variables whose storage is reachable through an escaped reference
  // (address taken, captured or bound by non-const reference): a mutation
  // through the reference is invisible to the map, so a merge of such a
  // variable's definitions must not be resolved (see decodeTrylockCond()).
  llvm::SmallPtrSet<const NamedDecl *, 4> EscapedDecls;
  // Memoized constant values of canonical definitions, keyed by definition
  // ID (std::nullopt: does not constant-evaluate): intersectContexts()
  // consults the same definitions at every join they reach.
  llvm::DenseMap<unsigned, std::optional<llvm::APSInt>> ConstantValues;
  // Definitions whose chain of prior definitions holds constants only, all
  // the way to the variable's declaration (chainNonConstantDefs()). Only
  // intersectBackEdge() ever changes an existing definition, and it clears
  // this set when it does.
  llvm::DenseSet<unsigned> CleanChains;

public:
  LocalVariableMap() {
    // index 0 is a placeholder for undefined variables (aka phi-nodes).
    VarDefinitions.push_back(VarDefinition(nullptr, 0, 0, getEmptyContext()));
  }

  /// Look up a definition, within the given context.
  const VarDefinition* lookup(const NamedDecl *D, Context Ctx) {
    const unsigned *i = Ctx.lookup(D);
    if (!i)
      return nullptr;
    assert(*i < VarDefinitions.size());
    return &VarDefinitions[*i];
  }

  /// Look up the canonical definition for \p D within the given context:
  /// the definition its reference chain resolves to (e.g. a loop head wraps
  /// every incoming definition in a reference).  Returns NULL if the
  /// variable is not in the context or resolves to an unknown definition.
  const VarDefinition *lookupCanonical(const NamedDecl *D, Context Ctx) {
    const unsigned *i = Ctx.lookup(D);
    if (!i)
      return nullptr;
    assert(*i < VarDefinitions.size());
    unsigned ID = getCanonicalDefinitionID(*i);
    return ID ? &VarDefinitions[ID] : nullptr;
  }

  /// Look up the expression for the definition \p i, looking through
  /// references. Returns NULL if the expression is not statically known --
  /// including for a phi, which has no single defining expression. If
  /// successful, also modifies Ctx to hold the context of the returned Expr.
  const Expr *lookupExprByID(unsigned i, Context &Ctx) {
    while (i > 0) {
      const VarDefinition &VD = VarDefinitions[i];
      if (VD.Exp) {
        Ctx = VD.Ctx;
        return VD.Exp;
      }
      if (VD.isPhi())
        return nullptr;
      i = VD.DirectRef;
    }
    return nullptr;
  }

  /// Look up the definition for D within the given context.  Returns
  /// NULL if the expression is not statically known.  If successful, also
  /// modifies Ctx to hold the context of the return Expr.
  const Expr* lookupExpr(const NamedDecl *D, Context &Ctx) {
    const unsigned *P = Ctx.lookup(D);
    if (!P)
      return nullptr;
    return lookupExprByID(*P, Ctx);
  }

  void markEscaped(const NamedDecl *D) { EscapedDecls.insert(D); }
  bool isEscaped(const NamedDecl *D) const { return EscapedDecls.count(D); }

  /// What walkChain() does with a definition it has just visited.
  enum class ChainVisit {
    Fail,   ///< The walk's question is answered: stop and return false.
    Follow, ///< Continue into this definition's own prior definitions.
    Prune,  ///< This definition's chain adds nothing: do not follow it.
  };

  /// Walks the chain of \p D's definitions leading up to definition \p ID:
  /// every path of prior definitions from \p ID back to \p D's declaration,
  /// calling \p Visit on each definition passed through (a merge continues
  /// into both of its operands' chains). Returns false if \p Visit rejects
  /// a definition, or if a path reaches an unknown definition, about which
  /// nothing can be concluded; true if every path ended at the declaration.
  /// A loop back edge passes the loop-head definition as \p StopAt: a chain
  /// that reaches the head has been walked as far as the iteration goes,
  /// and what precedes the head is the merge being tested itself.
  template <typename VisitFn>
  bool walkChain(const NamedDecl *D, unsigned ID, unsigned StopAt,
                 VisitFn Visit) {
    SmallVector<unsigned, 4> Worklist = {ID};
    llvm::SmallDenseSet<unsigned, 8> Visited;
    while (!Worklist.empty()) {
      unsigned ID = Worklist.pop_back_val();
      // Resolve references one step at a time: \p StopAt is usually a
      // loop-head reference, which one-hop canonicalization would skip
      // right past.
      bool PathEnds = false;
      while (ID > 0 && VarDefinitions[ID].isReference()) {
        if (ID == StopAt || VarDefinitions[ID].UninitDecl) {
          // The loop head ends the path (see above); so does the variable's
          // declaration, with or without an initializer.
          PathEnds = true;
          break;
        }
        ID = VarDefinitions[ID].DirectRef;
      }
      if (PathEnds || (StopAt != 0 && ID == StopAt))
        continue; // A phi-converted loop head is its own canonical.
      if (ID == 0)
        return false;
      if (!Visited.insert(ID).second)
        continue; // A phi-converted loop head can make the graph cyclic.
      ChainVisit Step = Visit(ID);
      if (Step == ChainVisit::Fail)
        return false;
      if (Step == ChainVisit::Prune)
        continue;
      if (VarDefinitions[ID].isPhi()) {
        // The merged value is one of the operands': the chain continues
        // into both.
        Worklist.push_back(VarDefinitions[ID].DirectRef);
        Worklist.push_back(VarDefinitions[ID].PhiAlt);
        continue;
      }
      const unsigned *P = VarDefinitions[ID].Ctx.lookup(D);
      if (P)
        Worklist.push_back(*P);
      // Otherwise this path reached the declaration.
    }
    return true;
  }

  /// Returns true if the chain of \p D's definitions leading up to
  /// definition \p ID provably does not contain definition \p Avoid: every
  /// path of prior definitions from \p ID reaches \p D's declaration
  /// without passing \p Avoid or an unknown definition (walkChain()). Used
  /// to establish that the assignment creating \p Avoid was never executed
  /// on the paths where \p ID is the reaching definition. \p StopAt is as
  /// in walkChain(): a chain that reaches the loop head avoided \p Avoid
  /// within the iteration.
  bool chainAvoids(const NamedDecl *D, unsigned ID, unsigned Avoid,
                   unsigned StopAt = 0) {
    Avoid = getCanonicalDefinitionID(Avoid);
    return walkChain(D, ID, StopAt, [Avoid](unsigned Def) {
      return Def == Avoid ? ChainVisit::Fail : ChainVisit::Follow;
    });
  }

  /// Collects into \p Defs every non-constant definition (a merge included)
  /// that the chain of \p D's definitions leading up to \p ID passes
  /// through: the definitions a chainAvoids() query on \p ID answers "no"
  /// for, provided the definition asked about is itself non-constant. That
  /// is what resolution asks about -- decodeTrylockCond() and phiAbsorbs()
  /// both name a merge's non-constant operand -- but not what every caller
  /// asks: the back-edge unwrap in intersectBackEdge() can name a constant
  /// operand, and \p Defs deliberately says nothing about those
  /// (constantToKeep() is not consulted there). Returns false if the chain
  /// reaches an unknown definition, in which case \p Defs says nothing at
  /// all.
  ///
  /// A chain that contributes nothing -- constant definitions all the way
  /// to the declaration -- is memoized (CleanChains) and pruned when a
  /// later walk reaches it, so that a variable assigned constants over and
  /// over does not make every join walk its whole history.
  bool chainNonConstantDefs(const NamedDecl *D, unsigned ID,
                            llvm::SmallDenseSet<unsigned, 8> &Defs) {
    bool Known = walkChain(D, ID, /*StopAt=*/0, [&](unsigned Def) {
      if (CleanChains.contains(Def))
        return ChainVisit::Prune;
      if (!constantValue(Def))
        Defs.insert(Def);
      return ChainVisit::Follow;
    });
    if (Known && Defs.empty())
      if (unsigned Canon = getCanonicalDefinitionID(ID))
        CleanChains.insert(Canon);
    return Known;
  }

  /// The constant integer value of the canonical definition \p Canon,
  /// memoized; std::nullopt if the definition is unknown, a merge, or does
  /// not constant-evaluate. Any expression that constant-evaluates counts,
  /// not just a literal: `bool b = kFalseConstant;` is the constant false.
  std::optional<llvm::APSInt> constantValue(unsigned Canon) {
    if (Canon == 0 || VarDefinitions[Canon].isPhi())
      return std::nullopt;
    auto [It, Inserted] = ConstantValues.try_emplace(Canon);
    if (Inserted) {
      const Expr *E = VarDefinitions[Canon].Exp;
      Expr::EvalResult ER;
      if (E && !E->isValueDependent() &&
          E->EvaluateAsInt(ER, VarDefinitions[Canon].Dec->getASTContext()))
        It->second = ER.Val.getInt();
    }
    return It->second;
  }

  /// Whether the canonical definitions \p Canon1 and \p Canon2 constant-
  /// evaluate to the same integer value: e.g. after
  /// `bool b = false; if (c) b = false;` the variable is still the constant
  /// false, and can later merge with a try-acquire result (a merge of
  /// merges is not resolved). The values must match exactly, not merely in
  /// truthiness, and non-integer constants (e.g. two distinct addresses,
  /// which are both "true") never match. Shared by the branch-join and
  /// back-edge merge engines (intersectContexts() / intersectBackEdge())
  /// so that they cannot drift apart.
  bool valueEqualConstants(unsigned Canon1, unsigned Canon2) {
    std::optional<llvm::APSInt> V1 = constantValue(Canon1);
    if (!V1)
      return false;
    std::optional<llvm::APSInt> V2 = constantValue(Canon2);
    return V2 && llvm::APSInt::isSameValue(*V1, *V2);
  }

  /// Which of two value-equal constant definitions of \p D a branch join
  /// may keep for the other, or 0 if neither will do. Equal values alone
  /// do not make the two interchangeable: resolving a merge asks whether
  /// the constant's chain passes the try-acquire call (chainAvoids()), and
  /// the definition that is kept answers that for both paths. In
  /// `if (c1) { ok = mu.TryLock(); if (c2) ok = false; } else ok = false;`
  /// only the first `ok = false` can follow the call, so keeping the else
  /// arm's in its place would resolve a merge that must not resolve.
  ///
  /// The merged value's chain is really the union of the two, so the one
  /// to keep is the one whose non-constant definitions
  /// (chainNonConstantDefs(), the definitions resolution's queries name)
  /// already cover the other's -- it then answers every such query exactly
  /// as the union would, not merely conservatively. A chain that reaches an
  /// unknown definition covers everything, being answered "no" throughout.
  /// When neither covers the other, the join keeps no constant and merges
  /// them.
  unsigned constantToKeep(const NamedDecl *D, unsigned Canon1,
                          unsigned Canon2) {
    if (!valueEqualConstants(Canon1, Canon2))
      return 0;
    llvm::SmallDenseSet<unsigned, 8> Defs1, Defs2;
    if (!chainNonConstantDefs(D, Canon1, Defs1))
      return Canon1;
    if (!chainNonConstantDefs(D, Canon2, Defs2))
      return Canon2;
    auto Covers = [](const llvm::SmallDenseSet<unsigned, 8> &Defs,
                     const llvm::SmallDenseSet<unsigned, 8> &Other) {
      return Defs.size() >= Other.size() &&
             llvm::all_of(Other,
                          [&Defs](unsigned Def) { return Defs.contains(Def); });
    };
    if (Covers(Defs1, Defs2))
      return Canon1;
    if (Covers(Defs2, Defs1))
      return Canon2;
    return 0;
  }

  /// Whether the merge \p CanonPhi already covers the definition
  /// \p CanonOther of variable \p Dec, so that joining the two keeps the
  /// phi as is: \p CanonOther is one of the phi's own operands, or a
  /// definition that is not an operand but is value-equal to the phi's
  /// constant operand (constants of the same value are interchangeable,
  /// valueEqualConstants()) -- provided its chain avoids the phi's
  /// non-constant operand, exactly as resolving the phi imposes on the
  /// recorded constant (chainAvoids()): e.g. phi(call, false) absorbs
  /// another `= false` assignment that cannot follow the call. \p StopAt
  /// is forwarded to chainAvoids() by loop back edges. Like
  /// valueEqualConstants(), shared by both merge engines.
  bool phiAbsorbs(const NamedDecl *Dec, unsigned CanonPhi, unsigned CanonOther,
                  unsigned StopAt = 0) {
    if (CanonPhi == 0 || CanonOther == 0 || !VarDefinitions[CanonPhi].isPhi())
      return false;
    const VarDefinition &VD = VarDefinitions[CanonPhi];
    unsigned Op1 = getCanonicalDefinitionID(VD.DirectRef);
    unsigned Op2 = getCanonicalDefinitionID(VD.PhiAlt);
    if (Op1 == CanonOther || Op2 == CanonOther)
      return true;
    std::optional<llvm::APSInt> VO = constantValue(CanonOther);
    if (!VO)
      return false;
    std::optional<llvm::APSInt> V1 = constantValue(Op1);
    std::optional<llvm::APSInt> V2 = constantValue(Op2);
    if (V1 && V2)
      // Both operands constant: absorb a matching value. Unlike a join of
      // two constants (constantToKeep()), this keeps the phi without asking
      // whether the absorbed definition's chain is covered by the operands'
      // -- a phi of two constants has no non-constant operand, so it
      // resolves no branch by itself, and a call the absorbed chain passed
      // was overwritten before this join, which leaves its conditional fact
      // unchecked for the lockset join to report.
      return llvm::APSInt::isSameValue(*V1, *VO) ||
             llvm::APSInt::isSameValue(*V2, *VO);
    std::optional<llvm::APSInt> VC = V1 ? V1 : V2;
    unsigned NonConstOp = V1 ? Op2 : Op1;
    return VC && llvm::APSInt::isSameValue(*VC, *VO) &&
           chainAvoids(Dec, CanonOther, NonConstOp, StopAt);
  }

  Context getEmptyContext() { return ContextFactory.getEmptyMap(); }

  /// Return the next context after processing S.  This function is used by
  /// clients of the class to get the appropriate context when traversing the
  /// CFG.  It must be called for every assignment or DeclStmt.
  const Context &getNextContext(unsigned &CtxIndex, const Stmt *S,
                                const Context &C) {
    if (SavedContexts[CtxIndex + 1].first == S) {
      CtxIndex++;
      const Context &Result = SavedContexts[CtxIndex].second;
      return Result;
    }
    return C;
  }

  void dumpVarDefinitionName(unsigned i) {
    if (i == 0) {
      llvm::errs() << "Undefined";
      return;
    }
    const NamedDecl *Dec = VarDefinitions[i].Dec;
    if (!Dec) {
      llvm::errs() << "<<NULL>>";
      return;
    }
    Dec->printName(llvm::errs());
    llvm::errs() << "." << i << " " << ((const void*) Dec);
  }

  /// Dumps an ASCII representation of the variable map to llvm::errs()
  void dump() {
    for (unsigned i = 1, e = VarDefinitions.size(); i < e; ++i) {
      const Expr *Exp = VarDefinitions[i].Exp;
      unsigned Ref = VarDefinitions[i].DirectRef;

      dumpVarDefinitionName(i);
      llvm::errs() << " = ";
      if (Exp) Exp->dump();
      else {
        dumpVarDefinitionName(Ref);
        llvm::errs() << "\n";
      }
    }
  }

  /// Dumps an ASCII representation of a Context to llvm::errs()
  void dumpContext(Context C) {
    for (Context::iterator I = C.begin(), E = C.end(); I != E; ++I) {
      const NamedDecl *D = I.getKey();
      D->printName(llvm::errs());
      llvm::errs() << " -> ";
      dumpVarDefinitionName(I.getData());
      llvm::errs() << "\n";
    }
  }

  /// Builds the variable map.
  void traverseCFG(AnalysisDeclContext &AC, CFG *CFGraph,
                   const PostOrderCFGView *SortedGraph,
                   std::vector<CFGBlockInfo> &BlockInfo);

protected:
  friend class VarMapBuilder;

  // Resolve any definition ID down to its non-reference base ID.
  //
  // This follows the CanonicalRef each reference caches when it is created
  // (addReference()), which intersectBackEdge() can outdate: it converts a
  // loop-head reference into a phi, or invalidates it, in place -- after an
  // inner loop's head has wrapped that reference in one of its own, and
  // cached the base it resolved to back then. Stepping through DirectRef
  // instead reaches the mutated head, so the two walks can disagree for a
  // reference created inside a loop whose head is merged later. Nothing
  // depends on the difference today: the consumers of this cache either run
  // before the mutation or re-check isPhi() on what they get back, and a
  // walk that must see the current state resolves references one at a time
  // (walkChain()). A new consumer must not assume otherwise.
  unsigned getCanonicalDefinitionID(unsigned ID) const {
    while (ID > 0 && VarDefinitions[ID].isReference())
      ID = VarDefinitions[ID].CanonicalRef;
    return ID;
  }

  // Get the current context index
  unsigned getContextIndex() { return SavedContexts.size()-1; }

  // Save the current context for later replay
  void saveContext(const Stmt *S, Context C) {
    SavedContexts.push_back(std::make_pair(S, C));
  }

  // Adds a new definition to the given context, and returns a new context.
  // This method should be called when declaring a new variable.
  Context addDefinition(const NamedDecl *D, const Expr *Exp, Context Ctx) {
    assert(!Ctx.contains(D));
    unsigned newID = VarDefinitions.size();
    Context NewCtx = ContextFactory.add(Ctx, D, newID);
    VarDefinitions.push_back(VarDefinition(D, Exp, Ctx));
    return NewCtx;
  }

  // Add a new reference to an existing definition.
  Context addReference(const NamedDecl *D, unsigned Ref, Context Ctx) {
    unsigned newID = VarDefinitions.size();
    Context NewCtx = ContextFactory.add(Ctx, D, newID);
    VarDefinitions.push_back(
        VarDefinition(D, Ref, getCanonicalDefinitionID(Ref), Ctx));
    return NewCtx;
  }

  // Merge two distinct definitions into a phi definition: the variable's
  // value is that of one of the two. Most consumers treat a phi like a
  // cleared definition; see VarDefinition::PhiAlt for why it exists.
  Context addPhiDefinition(const NamedDecl *D, unsigned Ref1, unsigned Ref2,
                           Context Ctx) {
    assert(Ref1 && Ref2 && "phi operands must be known definitions");
    unsigned newID = VarDefinitions.size();
    Context NewCtx =
        ContextFactory.add(ContextFactory.remove(Ctx, D), D, newID);
    VarDefinition VD(D, Ref1, /*CanonicalRef=*/0, Ctx);
    VD.PhiAlt = Ref2;
    VarDefinitions.push_back(VD);
    return NewCtx;
  }

  // Updates a definition only if that definition is already in the map.
  // This method should be called when assigning to an existing variable.
  Context updateDefinition(const NamedDecl *D, Expr *Exp, Context Ctx) {
    if (Ctx.contains(D)) {
      unsigned newID = VarDefinitions.size();
      Context NewCtx = ContextFactory.remove(Ctx, D);
      NewCtx = ContextFactory.add(NewCtx, D, newID);
      VarDefinitions.push_back(VarDefinition(D, Exp, Ctx));
      return NewCtx;
    }
    return Ctx;
  }

  // Removes a definition from the context, but keeps the variable name
  // as a valid variable.  The index 0 is a placeholder for cleared definitions.
  Context clearDefinition(const NamedDecl *D, Context Ctx) {
    Context NewCtx = Ctx;
    if (NewCtx.contains(D)) {
      NewCtx = ContextFactory.remove(NewCtx, D);
      NewCtx = ContextFactory.add(NewCtx, D, 0);
    }
    return NewCtx;
  }

  // Remove a definition entirely frmo the context.
  Context removeDefinition(const NamedDecl *D, Context Ctx) {
    Context NewCtx = Ctx;
    if (NewCtx.contains(D)) {
      NewCtx = ContextFactory.remove(NewCtx, D);
    }
    return NewCtx;
  }

  Context intersectContexts(Context C1, Context C2);
  Context createReferenceContext(Context C);
  void intersectBackEdge(Context C1, Context C2);
};

} // namespace

// This has to be defined after LocalVariableMap.
CFGBlockInfo CFGBlockInfo::getEmptyBlockInfo(LocalVariableMap &M) {
  return CFGBlockInfo(M.getEmptyContext());
}

namespace {

/// Visitor which builds a LocalVariableMap
class VarMapBuilder : public ConstStmtVisitor<VarMapBuilder> {
public:
  LocalVariableMap* VMap;
  LocalVariableMap::Context Ctx;

  VarMapBuilder(LocalVariableMap *VM, LocalVariableMap::Context C,
                AnalysisDeclContext &AC)
      : VMap(VM), Ctx(C), AC(AC) {}

  void VisitDeclStmt(const DeclStmt *S);
  void VisitBinaryOperator(const BinaryOperator *BO);
  void VisitUnaryOperator(const UnaryOperator *UO);
  void VisitLambdaExpr(const LambdaExpr *LE);
  void VisitCallExpr(const CallExpr *CE);
  void VisitCXXConstructExpr(const CXXConstructExpr *CE);

private:
  // Only used to reach the body's parent map, and only for an address-of
  // expression: the map is built lazily, so functions that take no address
  // never pay for it.
  AnalysisDeclContext &AC;

  void markEscapedIfDeclRef(const Expr *E);
  void markEscapedRefBindings(const InitListExpr *ILE);
};

} // namespace

// The one rule for marking a variable whose storage becomes reachable
// through a reference: it can then be mutated without a visible assignment.
// Shared by every escape site so they cannot drift apart; IgnoreParenCasts,
// because an explicit cast (`(bool &)b`) hides the variable just as well as
// an implicit one.
void VarMapBuilder::markEscapedIfDeclRef(const Expr *E) {
  if (const auto *DRE = dyn_cast<DeclRefExpr>(E->IgnoreParenCasts()))
    VMap->markEscaped(DRE->getDecl());
}

// Marks variables bound to non-const reference members in an aggregate
// initialization (`struct W { bool &b; }; W w{ok};`): the aggregate can
// mutate them without a visible assignment, like any reference binding.
void VarMapBuilder::markEscapedRefBindings(const InitListExpr *ILE) {
  // Descends into an initializer that is itself a list (an array element, a
  // base class, a nested aggregate member).
  auto Descend = [this](const Expr *Init) {
    if (const auto *Nested = dyn_cast_or_null<InitListExpr>(
            Init ? Init->IgnoreParenImpCasts() : nullptr))
      markEscapedRefBindings(Nested);
  };

  // An array of aggregates has no fields of its own: every element carries
  // its own initializer list.
  if (ILE->getType()->isArrayType()) {
    for (const Expr *Init : ILE->inits())
      Descend(Init);
    Descend(ILE->getArrayFiller());
    return;
  }

  const RecordDecl *RD = ILE->getType()->getAsRecordDecl();
  if (!RD || RD->isUnion())
    return; // A union cannot have a reference member.
  unsigned I = 0, N = ILE->getNumInits();
  // C++17 aggregate initialization lists the base classes before the
  // fields, so the field walk starts past them or the two run out of step.
  if (const auto *CXXRD = dyn_cast<CXXRecordDecl>(RD))
    for (unsigned NumBases = CXXRD->getNumBases(); I < N && I < NumBases; ++I)
      Descend(ILE->getInit(I));
  auto FI = RD->field_begin(), FE = RD->field_end();
  for (; I < N && FI != FE; ++I, ++FI) {
    const Expr *Init = ILE->getInit(I);
    if (!Init)
      continue;
    QualType FT = FI->getType();
    if (FT->isReferenceType() && !FT.getNonReferenceType().isConstQualified())
      markEscapedIfDeclRef(Init);
    else
      Descend(Init);
  }
}

// Add new local variables to the variable map
void VarMapBuilder::VisitDeclStmt(const DeclStmt *S) {
  bool modifiedCtx = false;
  const DeclGroupRef DGrp = S->getDeclGroup();
  for (const auto *D : DGrp) {
    if (const auto *VD = dyn_cast_or_null<VarDecl>(D)) {
      const Expr *E = VD->getInit();

      // Add local variables with trivial type to the variable map
      QualType T = VD->getType();
      if (T.isTrivialType(VD->getASTContext())) {
        Ctx = VMap->addDefinition(VD, E, Ctx);
        modifiedCtx = true;
      } else if (T->isReferenceType() && E &&
                 !T.getNonReferenceType().isConstQualified()) {
        // Binding a non-const reference to a variable lets the variable be
        // mutated without a visible assignment.
        markEscapedIfDeclRef(E);
      }
      // Aggregate initialization can bind non-const reference members.
      if (const auto *ILE = dyn_cast_or_null<InitListExpr>(
              E ? E->IgnoreParenImpCasts() : nullptr))
        markEscapedRefBindings(ILE);
    }
  }
  if (modifiedCtx)
    VMap->saveContext(S, Ctx);
}

// Update local variable definitions in variable map
void VarMapBuilder::VisitBinaryOperator(const BinaryOperator *BO) {
  if (!BO->isAssignmentOp())
    return;

  Expr *LHSExp = BO->getLHS()->IgnoreParenCasts();

  // Update the variable map and current context.
  if (const auto *DRE = dyn_cast<DeclRefExpr>(LHSExp)) {
    const ValueDecl *VDec = DRE->getDecl();
    if (Ctx.lookup(VDec)) {
      if (BO->getOpcode() == BO_Assign)
        Ctx = VMap->updateDefinition(VDec, BO->getRHS(), Ctx);
      else
        // FIXME -- handle compound assignment operators
        Ctx = VMap->clearDefinition(VDec, Ctx);
      VMap->saveContext(BO, Ctx);
    }
  }
}

// True if the address the expression produces can only be read through.
// The address is followed out through the conversions it flows into, to
// the last one that is still a pointer, and only that outermost type
// decides: an intermediate `const bool *` proves nothing when a cast
// strips the const again (`const_cast<bool *>(static_cast<const bool *>
// (&b))`). `observe(&b)` with `void observe(const bool *)` is the shape
// this recognizes -- the argument is converted to `const bool *` before
// the call sees it.
static bool addrOfIsReadOnly(const UnaryOperator *UO, ParentMap &PM) {
  const Expr *E = UO;
  while (true) {
    // Only the conversions the address itself flows through are followed;
    // any other parent consumes the pointer as it stands.
    const Stmt *P = PM.getParent(E);
    if (!P || !(isa<ParenExpr>(P) || isa<CastExpr>(P)))
      break;
    // Once it is no longer a pointer (`(void)&b`, a cast to an integer)
    // the address cannot be followed any further; the last pointer type it
    // had is the one that reached that conversion.
    const auto *PE = cast<Expr>(P);
    if (!PE->getType()->isPointerType())
      break;
    E = PE;
  }
  QualType T = E->getType();
  return T->isPointerType() && T->getPointeeType().isConstQualified();
}

// Marks a variable whose address is taken: it can then be mutated without a
// visible assignment. An address that is only readable through is not an
// escape, the same const distinction VisitDeclStmt(), VisitCallExpr() and
// VisitCXXConstructExpr() make for reference and pointer parameters --
// without it, passing `&b` to a const-taking API would lose the plain
// `bool b = mu.TryLock(); if (b) ...` form.
void VarMapBuilder::VisitUnaryOperator(const UnaryOperator *UO) {
  if (UO->getOpcode() != UO_AddrOf)
    return;
  // Checked before the parent map is touched: building it is what makes
  // this more than a type test, and nothing else here needs it.
  if (!isa<DeclRefExpr>(UO->getSubExpr()->IgnoreParenCasts()))
    return;
  if (addrOfIsReadOnly(UO, AC.getParentMap()))
    return;
  markEscapedIfDeclRef(UO->getSubExpr());
}

// Marks variables captured by reference in a lambda: any later call may
// mutate them without a visible assignment.
void VarMapBuilder::VisitLambdaExpr(const LambdaExpr *LE) {
  for (const LambdaCapture &LC : LE->captures()) {
    if (!LC.capturesVariable() || LC.getCaptureKind() != LCK_ByRef)
      continue;
    const ValueDecl *VD = LC.getCapturedVar();
    VMap->markEscaped(VD);
    // A reference init-capture (`[&x = b]`) binds like a reference
    // declaration: the escaped variable is the one in the initializer.
    if (const auto *IC = dyn_cast<VarDecl>(VD); IC && IC->isInitCapture())
      if (const Expr *Init = IC->getInit())
        markEscapedIfDeclRef(Init);
  }
}

// Invalidates local variable definitions if variable escaped.
void VarMapBuilder::VisitCallExpr(const CallExpr *CE) {
  const FunctionDecl *FD = CE->getDirectCallee();
  if (!FD)
    return;

  // Heuristic for likely-benign functions that pass by mutable reference. This
  // is needed to avoid a slew of false positives due to mutable reference
  // passing where the captured reference is usually passed on by-value.
  if (const IdentifierInfo *II = FD->getIdentifier()) {
    // Any kind of std::bind-like functions.
    if (II->isStr("bind") || II->isStr("bind_front"))
      return;
  }

  // Invalidate local variable definitions that are passed by non-const
  // reference or non-const pointer.
  for (unsigned Idx = 0; Idx < CE->getNumArgs(); ++Idx) {
    if (Idx >= FD->getNumParams())
      break;

    const Expr *Arg = CE->getArg(Idx)->IgnoreParenImpCasts();
    const ParmVarDecl *PVD = FD->getParamDecl(Idx);
    QualType ParamType = PVD->getType();

    // Potential reassignment if passed by non-const reference / pointer.
    const ValueDecl *VDec = nullptr;
    if (ParamType->isReferenceType() &&
        !ParamType->getPointeeType().isConstQualified()) {
      if (const auto *DRE = dyn_cast<DeclRefExpr>(Arg))
        VDec = DRE->getDecl();
    } else if (ParamType->isPointerType() &&
               !ParamType->getPointeeType().isConstQualified()) {
      Arg = Arg->IgnoreParenCasts();
      if (const auto *UO = dyn_cast<UnaryOperator>(Arg)) {
        if (UO->getOpcode() == UO_AddrOf) {
          const Expr *SubE = UO->getSubExpr()->IgnoreParenCasts();
          if (const auto *DRE = dyn_cast<DeclRefExpr>(SubE))
            VDec = DRE->getDecl();
        }
      }
    }

    if (VDec)
      Ctx = VMap->clearDefinition(VDec, Ctx);
  }
  // Save the context after the call where escaped variables' definitions (if
  // they exist) are cleared.
  VMap->saveContext(CE, Ctx);
}

// Marks variables bound to a constructor's non-const reference parameters:
// the constructed object can store the reference and mutate them later
// without a visible assignment. (VisitCallExpr() above only clears the
// definition for an ordinary call, which is assumed to mutate during the
// call but not to retain the reference.)
void VarMapBuilder::VisitCXXConstructExpr(const CXXConstructExpr *CE) {
  const CXXConstructorDecl *CD = CE->getConstructor();
  if (!CD)
    return;
  for (unsigned Idx = 0, N = CE->getNumArgs(); Idx < N; ++Idx) {
    if (Idx >= CD->getNumParams())
      break;
    QualType ParamType = CD->getParamDecl(Idx)->getType();
    if (ParamType->isReferenceType() &&
        !ParamType->getPointeeType().isConstQualified())
      markEscapedIfDeclRef(CE->getArg(Idx));
  }
}

// Computes the intersection of two contexts.  The intersection is the
// set of variables which have the same definition in both contexts;
// variables with different definitions are discarded.
LocalVariableMap::Context
LocalVariableMap::intersectContexts(Context C1, Context C2) {
  Context Result = C1;
  for (const auto &P : C1) {
    const NamedDecl *Dec = P.first;
    const unsigned *I2 = C2.lookup(Dec);
    if (!I2) {
      // The variable doesn't exist on second path.
      Result = removeDefinition(Dec, Result);
    } else if (P.second != *I2) {
      unsigned Canon1 = getCanonicalDefinitionID(P.second);
      unsigned Canon2 = getCanonicalDefinitionID(*I2);
      if (Canon1 == Canon2 && Canon1 != 0)
        continue; // Same underlying definition on both paths.
      // Distinct definitions that constant-evaluate to the same integer
      // value are interchangeable for resolution purposes, provided the
      // one kept answers every later chain query for both paths
      // (constantToKeep()).
      if (unsigned Keep = constantToKeep(Dec, Canon1, Canon2)) {
        if (Keep == Canon1)
          continue; // Keep the first path's.
        Result =
            ContextFactory.add(ContextFactory.remove(Result, Dec), Dec, *I2);
        continue;
      }
      // A phi merged with a definition it already covers is just the phi:
      // at a join of three or more predecessors the paths merge pairwise,
      // so e.g. (constant, call) -> phi followed by (phi, constant) must
      // not discard the merge the first pair created; absorbing a
      // value-equal constant that is not an operand keeps the result
      // independent of the order in which the paths merge (phiAbsorbs()).
      if (phiAbsorbs(Dec, Canon1, Canon2))
        continue; // Keep the first path's phi.
      if (phiAbsorbs(Dec, Canon2, Canon1)) {
        // Keep the second path's phi.
        Result =
            ContextFactory.add(ContextFactory.remove(Result, Dec), Dec, *I2);
        continue;
      }
      // The underlying definitions differ. If both are known (and not
      // already merges themselves), remember the pair as a phi definition;
      // otherwise invalidate.
      if (Canon1 != 0 && Canon2 != 0 && !VarDefinitions[Canon1].isPhi() &&
          !VarDefinitions[Canon2].isPhi())
        Result = addPhiDefinition(Dec, P.second, *I2, Result);
      else
        Result = clearDefinition(Dec, Result);
    }
  }
  return Result;
}

// For every variable in C, create a new variable that refers to the
// definition in C.  Return a new context that contains these new variables.
// (We use this for a naive implementation of SSA on loop back-edges.)
LocalVariableMap::Context LocalVariableMap::createReferenceContext(Context C) {
  Context Result = getEmptyContext();
  for (const auto &P : C)
    Result = addReference(P.first, P.second, Result);
  return Result;
}

// This routine also takes the intersection of C1 and C2, but it does so by
// altering the VarDefinitions.  C1 must be the result of an earlier call to
// createReferenceContext.
void LocalVariableMap::intersectBackEdge(Context C1, Context C2) {
  for (const auto &P : C1) {
    const unsigned I1 = P.second;
    VarDefinition *VDef = &VarDefinitions[I1];
    assert(VDef->isReference() || VDef->isPhi());

    const unsigned *I2 = C2.lookup(P.first);
    if (!I2) {
      // Variable does not exist at the end of the loop, invalidate.
      VDef->invalidateRef();
      continue;
    }

    const unsigned Canon2 = getCanonicalDefinitionID(*I2);

    if (VDef->isPhi()) {
      // A previous back edge already merged this variable. Keep the phi only
      // if this back edge carries a definition the merge already covers --
      // one of its operands, or a value-equal constant whose chain avoids
      // the non-constant operand up to the loop head (phiAbsorbs(), as at
      // branch joins) -- or the loop-head reference itself (Canon2 == I1, a
      // phi is its own canonical): a back edge that does not reassign the
      // variable, or reassigns it a value the merge covers, must not
      // discard the merge another back edge created.
      if (Canon2 != I1 && !phiAbsorbs(P.first, I1, Canon2, /*StopAt=*/I1))
        VDef->invalidateRef();
      continue;
    }

    // Compare the canonical IDs. This correctly handles chains of references
    // and determines if the variable is truly loop-invariant.
    if (VDef->CanonicalRef != Canon2) {
      // The variable was reassigned in the loop a value that is a constant
      // value-equal to the loop-head value: interchangeable for resolution
      // purposes (valueEqualConstants(), as at branch joins), so the head
      // reference stands.
      if (valueEqualConstants(VDef->CanonicalRef, Canon2))
        continue;
      // The variable is redefined in the loop. The back-edge value may
      // itself be a merge created at an intra-loop join with the loop-head
      // value as one of its operands (e.g. the path of a `continue` that
      // reassigned the variable joining the loop-end path that did not):
      // the head value then merges with the other operand. An operand that
      // is a constant value-equal to the head's constant stands in for the
      // head value the same way -- provided its chain avoids the other
      // operand up to the loop head, exactly as absorbing it at a branch
      // join would require (phiAbsorbs()).
      auto RefersTo = [this](unsigned ID, unsigned Target) {
        while (ID > 0 && ID != Target && VarDefinitions[ID].isReference())
          ID = VarDefinitions[ID].DirectRef;
        return ID == Target;
      };
      unsigned Alt = *I2;
      unsigned CanonAlt = Canon2;
      if (CanonAlt != 0 && VarDefinitions[CanonAlt].isPhi()) {
        const VarDefinition &P2 = VarDefinitions[CanonAlt];
        const unsigned OpD = getCanonicalDefinitionID(P2.DirectRef);
        const unsigned OpA = getCanonicalDefinitionID(P2.PhiAlt);
        if (RefersTo(P2.DirectRef, I1))
          Alt = P2.PhiAlt;
        else if (RefersTo(P2.PhiAlt, I1))
          Alt = P2.DirectRef;
        else if (valueEqualConstants(VDef->CanonicalRef, OpD) &&
                 chainAvoids(P.first, OpD, OpA, /*StopAt=*/I1))
          Alt = P2.PhiAlt;
        else if (valueEqualConstants(VDef->CanonicalRef, OpA) &&
                 chainAvoids(P.first, OpA, OpD, /*StopAt=*/I1))
          Alt = P2.DirectRef;
        else
          Alt = 0;
        CanonAlt = getCanonicalDefinitionID(Alt);
      }
      // If both the incoming definition and the back edge's are known (and
      // the latter is not itself a merge), remember the pair as a phi
      // definition (in place, like the invalidation below) rather than
      // discarding it, so that a branch on a try-acquire result merged
      // with its pre-loop initializer can still be resolved.
      if (VDef->CanonicalRef != 0 && Alt != 0 && CanonAlt != 0 &&
          !VarDefinitions[CanonAlt].isPhi()) {
        VDef->PhiAlt = Alt;
        VDef->CanonicalRef = 0;
      } else {
        VDef->invalidateRef(); // Mark this variable as undefined
      }
    }
  }
  // A back edge is the only thing that ever changes an existing definition
  // (in place, above), which can make a memoized clean chain stale.
  CleanChains.clear();
}

// Traverse the CFG in topological order, so all predecessors of a block
// (excluding back-edges) are visited before the block itself.  At
// each point in the code, we calculate a Context, which holds the set of
// variable definitions which are visible at that point in execution.
// Visible variables are mapped to their definitions using an array that
// contains all definitions.
//
// At join points in the CFG, the set is computed as the intersection of
// the incoming sets along each edge, E.g.
//
//                       { Context                 | VarDefinitions }
//   int x = 0;          { x -> x1                 | x1 = 0 }
//   int y = 0;          { x -> x1, y -> y1        | y1 = 0, x1 = 0 }
//   if (b) x = 1;       { x -> x2, y -> y1        | x2 = 1, y1 = 0, ... }
//   else   x = 2;       { x -> x3, y -> y1        | x3 = 2, x2 = 1, ... }
//   ...                 { y -> y1  (x is unknown) | x3 = 2, x2 = 1, ... }
//
// This is essentially a simpler and more naive version of the standard SSA
// algorithm.  Those definitions that remain in the intersection are from blocks
// that strictly dominate the current block.  We do not bother to insert proper
// phi nodes, because they are not used in our analysis; instead, wherever
// a phi node would be required, we simply remove that definition from the
// context (E.g. x above).
//
// The initial traversal does not capture back-edges, so those need to be
// handled on a separate pass.  Whenever the first pass encounters an
// incoming back edge, it duplicates the context, creating new definitions
// that refer back to the originals.  (These correspond to places where SSA
// might have to insert a phi node.)  On the second pass, these definitions are
// set to NULL if the variable has changed on the back-edge (i.e. a phi
// node was actually required.)  E.g.
//
//                       { Context           | VarDefinitions }
//   int x = 0, y = 0;   { x -> x1, y -> y1  | y1 = 0, x1 = 0 }
//   while (b)           { x -> x2, y -> y1  | [1st:] x2=x1; [2nd:] x2=NULL; }
//     x = x+1;          { x -> x3, y -> y1  | x3 = x2 + 1, ... }
//   ...                 { y -> y1           | x3 = 2, x2 = 1, ... }
void LocalVariableMap::traverseCFG(AnalysisDeclContext &AC, CFG *CFGraph,
                                   const PostOrderCFGView *SortedGraph,
                                   std::vector<CFGBlockInfo> &BlockInfo) {
  PostOrderCFGView::CFGBlockSet VisitedBlocks(CFGraph);

  for (const auto *CurrBlock : *SortedGraph) {
    unsigned CurrBlockID = CurrBlock->getBlockID();
    CFGBlockInfo *CurrBlockInfo = &BlockInfo[CurrBlockID];

    VisitedBlocks.insert(CurrBlock);

    // Calculate the entry context for the current block
    bool HasBackEdges = false;
    bool CtxInit = true;
    for (CFGBlock::const_pred_iterator PI = CurrBlock->pred_begin(),
         PE  = CurrBlock->pred_end(); PI != PE; ++PI) {
      // if *PI -> CurrBlock is a back edge, so skip it
      if (*PI == nullptr || !VisitedBlocks.alreadySet(*PI)) {
        HasBackEdges = true;
        continue;
      }

      unsigned PrevBlockID = (*PI)->getBlockID();
      CFGBlockInfo *PrevBlockInfo = &BlockInfo[PrevBlockID];

      if (CtxInit) {
        CurrBlockInfo->EntryContext = PrevBlockInfo->ExitContext;
        CtxInit = false;
      }
      else {
        CurrBlockInfo->EntryContext =
          intersectContexts(CurrBlockInfo->EntryContext,
                            PrevBlockInfo->ExitContext);
      }
    }

    // Duplicate the context if we have back-edges, so we can call
    // intersectBackEdges later.
    if (HasBackEdges)
      CurrBlockInfo->EntryContext =
        createReferenceContext(CurrBlockInfo->EntryContext);

    // Create a starting context index for the current block
    saveContext(nullptr, CurrBlockInfo->EntryContext);
    CurrBlockInfo->EntryIndex = getContextIndex();

    // Visit all the statements in the basic block.
    VarMapBuilder VMapBuilder(this, CurrBlockInfo->EntryContext, AC);
    for (const auto &BI : *CurrBlock) {
      switch (BI.getKind()) {
        case CFGElement::Statement: {
          CFGStmt CS = BI.castAs<CFGStmt>();
          VMapBuilder.Visit(CS.getStmt());
          break;
        }
        default:
          break;
      }
    }
    CurrBlockInfo->ExitContext = VMapBuilder.Ctx;

    // Mark variables on back edges as "unknown" if they've been changed.
    for (CFGBlock::const_succ_iterator SI = CurrBlock->succ_begin(),
         SE  = CurrBlock->succ_end(); SI != SE; ++SI) {
      // if CurrBlock -> *SI is *not* a back edge
      if (*SI == nullptr || !VisitedBlocks.alreadySet(*SI))
        continue;

      CFGBlock *FirstLoopBlock = *SI;
      Context LoopBegin = BlockInfo[FirstLoopBlock->getBlockID()].EntryContext;
      Context LoopEnd   = CurrBlockInfo->ExitContext;
      intersectBackEdge(LoopBegin, LoopEnd);
    }
  }

  // Put an extra entry at the end of the indexed context array
  unsigned exitID = CFGraph->getExit().getBlockID();
  saveContext(nullptr, BlockInfo[exitID].ExitContext);
}

/// Find the appropriate source locations to use when producing diagnostics for
/// each block in the CFG.
static void findBlockLocations(CFG *CFGraph,
                               const PostOrderCFGView *SortedGraph,
                               std::vector<CFGBlockInfo> &BlockInfo) {
  for (const auto *CurrBlock : *SortedGraph) {
    CFGBlockInfo *CurrBlockInfo = &BlockInfo[CurrBlock->getBlockID()];

    // Find the source location of the last statement in the block, if the
    // block is not empty.
    if (const Stmt *S = CurrBlock->getTerminatorStmt()) {
      CurrBlockInfo->EntryLoc = CurrBlockInfo->ExitLoc = S->getBeginLoc();
    } else {
      for (CFGBlock::const_reverse_iterator BI = CurrBlock->rbegin(),
           BE = CurrBlock->rend(); BI != BE; ++BI) {
        // FIXME: Handle other CFGElement kinds.
        if (std::optional<CFGStmt> CS = BI->getAs<CFGStmt>()) {
          CurrBlockInfo->ExitLoc = CS->getStmt()->getBeginLoc();
          break;
        }
      }
    }

    if (CurrBlockInfo->ExitLoc.isValid()) {
      // This block contains at least one statement. Find the source location
      // of the first statement in the block.
      for (const auto &BI : *CurrBlock) {
        // FIXME: Handle other CFGElement kinds.
        if (std::optional<CFGStmt> CS = BI.getAs<CFGStmt>()) {
          CurrBlockInfo->EntryLoc = CS->getStmt()->getBeginLoc();
          break;
        }
      }
    } else if (CurrBlock->pred_size() == 1 && *CurrBlock->pred_begin() &&
               CurrBlock != &CFGraph->getExit()) {
      // The block is empty, and has a single predecessor. Use its exit
      // location.
      CurrBlockInfo->EntryLoc = CurrBlockInfo->ExitLoc =
          BlockInfo[(*CurrBlock->pred_begin())->getBlockID()].ExitLoc;
    } else if (CurrBlock->succ_size() == 1 && *CurrBlock->succ_begin()) {
      // The block is empty, and has a single successor. Use its entry
      // location.
      CurrBlockInfo->EntryLoc = CurrBlockInfo->ExitLoc =
          BlockInfo[(*CurrBlock->succ_begin())->getBlockID()].EntryLoc;
    }
  }
}

namespace {

static void installNegativeFact(FactSet &FSet, FactManager &FactMan,
                                const CapabilityExpr &NegCp, SourceLocation Loc,
                                const Expr *OriginCall, const Expr *SpentCall,
                                bool KeepExistingReal);

static bool consumeNegativeFact(FactSet &FSet, FactManager &FactMan,
                                const CapabilityExpr &NegCp,
                                const Expr *AcquiringCall);

class LockableFactEntry final : public FactEntry {
private:
  /// Reentrancy depth: incremented when a capability has been acquired
  /// again after its initial acquisition -- by a reentrant acquire, or by
  /// the resolved success of a try-acquire over a definite hold. Always 0
  /// for a conditional fact, which is a single level.
  unsigned int ReentrancyDepth = 0;

  LockableFactEntry(const CapabilityExpr &CE, LockKind LK, SourceLocation Loc,
                    SourceKind Src)
      : FactEntry(Lockable, CE, LK, Loc, Src) {}

public:
  static LockableFactEntry *create(llvm::BumpPtrAllocator &Alloc,
                                   const LockableFactEntry &Other) {
    return new (Alloc) LockableFactEntry(Other);
  }

  static LockableFactEntry *create(llvm::BumpPtrAllocator &Alloc,
                                   const CapabilityExpr &CE, LockKind LK,
                                   SourceLocation Loc,
                                   SourceKind Src = Acquired) {
    return new (Alloc) LockableFactEntry(CE, LK, Loc, Src);
  }

  unsigned int getReentrancyDepth() const override { return ReentrancyDepth; }

  void
  handleRemovalFromIntersection(const FactSet &FSet, FactManager &FactMan,
                                SourceLocation JoinLoc, LockErrorKind LEK,
                                ThreadSafetyHandler &Handler) const override {
    if (lossNeedsWarning()) {
      Handler.handleMutexHeldEndOfScope(getKind(), toString(), loc(), JoinLoc,
                                        LEK);
    }
  }

  void handleLock(FactSet &FSet, FactManager &FactMan, const FactEntry &entry,
                  ThreadSafetyHandler &Handler) const override {
    if (const FactEntry *RFact = attemptReenter(FactMan, entry.kind())) {
      // This capability has been reentrantly acquired.
      FSet.replaceFact(FactMan, *this, RFact);
    } else {
      Handler.handleDoubleLock(entry.getKind(), entry.toString(), loc(),
                               entry.loc(), false);
    }
  }

  void handleUnlock(FactSet &FSet, FactManager &FactMan,
                    const CapabilityExpr &Cp, SourceLocation UnlockLoc,
                    bool FullyRemove,
                    ThreadSafetyHandler &Handler) const override {
    FSet.removeFact(FactMan, *this);

    if (const FactEntry *RFact = leaveReentrant(FactMan)) {
      // This capability remains reentrantly acquired.
      FSet.addLock(FactMan, RFact);
    } else if (!Cp.negative() && !FSet.anyConditional(FactMan, Cp)) {
      // Provably released -- unless a conditional fact remains, in which
      // case the capability is now merely try-held. Releasing a hold that
      // a try-acquire's success proved spends the call's stored result: it
      // stays truthy, but no longer witnesses a live hold (see
      // SpentTryLock).
      installNegativeFact(FSet, FactMan, !Cp, UnlockLoc,
                          /*OriginCall=*/nullptr, /*SpentCall=*/tryLockCall(),
                          /*KeepExistingReal=*/false);
    }
  }

  // Return an updated FactEntry one level deeper, or nullptr if a blocking
  // acquisition cannot nest in this capability: it must be reentrant, and
  // the kinds must match.
  const FactEntry *attemptReenter(FactManager &FactMan,
                                  LockKind ReenterKind) const {
    if (!reentrant())
      return nullptr;
    if (kind() != ReenterKind)
      return nullptr;
    auto *NewFact = FactMan.createFact<LockableFactEntry>(*this);
    NewFact->ReentrancyDepth++;
    return NewFact;
  }

  // Return an updated FactEntry if we are releasing a capability previously
  // acquired more than once, nullptr otherwise.
  const FactEntry *leaveReentrant(FactManager &FactMan) const {
    if (!ReentrancyDepth)
      return nullptr;
    auto *NewFact = FactMan.createFact<LockableFactEntry>(*this);
    NewFact->ReentrancyDepth--;
    return NewFact;
  }

  /// This definite fact one level deeper, the level proved by the success
  /// of the try-acquire call \p Origin (a conditional fact resolved over
  /// this hold). Whatever the capability's reentrancy: at runtime a
  /// try-acquire over a held capability fails rather than deadlocks, so
  /// its success edge is merely dead code that must still be well-formed.
  const LockableFactEntry *deepen(FactManager &FactMan,
                                  const Expr *Origin) const {
    assert(!tryHeld() && "only a definite fact has levels to deepen");
    auto *NewFact = FactMan.createFact<LockableFactEntry>(*this);
    NewFact->ReentrancyDepth++;
    NewFact->setTryLock(Origin, /*Conditional=*/false);
    return NewFact;
  }

  /// The definite fact this conditional fact is promoted to on its call's
  /// success edge: the same acquisition, now proved.
  const LockableFactEntry *promote(FactManager &FactMan) const {
    assert(tryHeld() && "only a conditional fact is promoted");
    auto *NewFact = FactMan.createFact<LockableFactEntry>(*this);
    NewFact->setTryLock(tryLockCall(), /*Conditional=*/false);
    return NewFact;
  }

  /// A conditional fact of this capability -- kind, source and location
  /// as this fact's -- originating from \p Origin: the form a definite
  /// hold takes when a join can only keep it as "held if \p Origin
  /// succeeded" (intersectAndWarn()).
  const LockableFactEntry *asConditional(FactManager &FactMan,
                                         const Expr *Origin) const {
    auto *NewFact = FactMan.createFact<LockableFactEntry>(*this);
    NewFact->ReentrancyDepth = 0;
    NewFact->setTryLock(Origin, /*Conditional=*/true);
    return NewFact;
  }

  /// This fact with its origin replaced by \p Origin (or cleared).
  const LockableFactEntry *withOrigin(FactManager &FactMan,
                                      const Expr *Origin) const {
    auto *NewFact = FactMan.createFact<LockableFactEntry>(*this);
    NewFact->setTryLock(Origin, tryHeld());
    return NewFact;
  }

  static bool classof(const FactEntry *A) {
    return A->getFactEntryKind() == Lockable;
  }
};

/// Install the negative fact for \p NegCp at \p Loc: the capability is
/// provably not held from here. \p OriginCall records the try-acquire whose
/// failure edge proves it (getEdgeLockset()); \p SpentCall marks the fact as
/// spending that call's stored result (see SpentTryLock). A weak negative
/// already in the set (not-held on only some paths, see intersectAndWarn())
/// is superseded -- the caller proves not-held on every path from here --
/// keeping any spend evidence it carries; a real one is kept when
/// \p KeepExistingReal (it already proves as much) and replaced otherwise.
/// The supersede scan runs only in functions with try-acquires, the only
/// place weak or spent facts exist.
static void installNegativeFact(FactSet &FSet, FactManager &FactMan,
                                const CapabilityExpr &NegCp, SourceLocation Loc,
                                const Expr *OriginCall, const Expr *SpentCall,
                                bool KeepExistingReal) {
  FactSet::iterator Existing = FSet.end();
  if (FactMan.tracksTryAcquires()) {
    Existing = FSet.findDefiniteIter(FactMan, NegCp);
    if (Existing != FSet.end()) {
      const FactEntry &Neg = FactMan[*Existing];
      if (KeepExistingReal && !Neg.weak())
        return;
      if (!SpentCall)
        SpentCall = Neg.spentTryLock();
    }
  }
  auto *NegFact =
      FactMan.createFact<LockableFactEntry>(NegCp, LK_Exclusive, Loc);
  if (OriginCall)
    NegFact->setTryLock(OriginCall, /*Conditional=*/false);
  if (SpentCall)
    NegFact->setSpentTryLock(SpentCall);
  // Replacing in place keeps the superseded fact's slot: removing swaps in
  // the set's last element and would reorder unrelated facts.
  if (Existing != FSet.end())
    FSet.replaceFact(FactMan, Existing, NegFact);
  else
    FSet.addLock(FactMan, NegFact);
}

/// Consume the negative fact for an acquisition of its capability: after
/// the acquisition the capability is possibly held, so the negative no
/// longer describes the state. One that spent another try-acquire's stored
/// result (see SpentTryLock) survives as a weak fact -- the resurrection
/// vetoes still need it -- unless \p AcquiringCall is the spent call
/// itself re-executing, which overwrites the stored result and ends its
/// staleness. Returns whether the consumed fact proved the capability not
/// held on every path (a real negative) -- what an acquisition's negative
/// capability requirement asks for.
static bool consumeNegativeFact(FactSet &FSet, FactManager &FactMan,
                                const CapabilityExpr &NegCp,
                                const Expr *AcquiringCall) {
  FactSet::iterator It = FSet.findDefiniteIter(FactMan, NegCp);
  if (It == FSet.end())
    return false;
  const FactEntry &Neg = FactMan[*It];
  const bool WasProven = !Neg.weak();
  if (Neg.spentTryLock() && Neg.spentTryLock() != AcquiringCall) {
    if (WasProven) {
      auto *NewFact =
          FactMan.createFact<LockableFactEntry>(cast<LockableFactEntry>(Neg));
      NewFact->setWeak();
      FSet.replaceFact(FactMan, It, NewFact);
    }
  } else {
    FSet.erase(It);
  }
  return WasProven;
}

/// The location for an unmatched-unlock "released here" note: the negative
/// fact's location if one exists -- unless it came from a try-acquire's
/// failure edge (getEdgeLockset()), which records where the call failed,
/// not a release, and the note would misread it.
static SourceLocation unmatchedUnlockNoteLoc(const FactSet &FSet,
                                             FactManager &FactMan,
                                             const CapabilityExpr &Cp) {
  if (const FactEntry *Neg = FSet.findDefinite(FactMan, !Cp);
      Neg && !Neg->tryLockCall())
    return Neg->loc();
  return SourceLocation();
}

/// Release the capability \p Cp, which is only try-held (conditional facts
/// but no definite one); returns true if the release was handled here.
/// With a \p Handler, diagnose like an unmatched unlock, drop every
/// conditional fact, and leave the negative fact behind: the release is an
/// unconditional demand, and the thread provably does not hold the
/// capability afterwards, whether the try-acquires succeeded or failed. A
/// null \p Handler (a scoped guard's destructor, FullyRemove=true) is a
/// conditional release -- the destructor releases the capability only if
/// the guard holds it -- so the guard's own conditional fact, the one its
/// construction \p OwnOrigin created, is disarmed silently: the conditional
/// release pairs with it exactly, discharging the obligation to check the
/// result and, with no other fact of the capability left, leaving the
/// negative fact. Another call's fact is kept unchanged: it records an
/// acquisition the guard does not own, which the destructor's conditional
/// release cannot pair with.
static bool handleUncheckedTryHeldUnlock(FactSet &FSet, FactManager &FactMan,
                                         const CapabilityExpr &Cp,
                                         SourceLocation UnlockLoc,
                                         ThreadSafetyHandler *Handler,
                                         const Expr *OwnOrigin = nullptr) {
  if (!FSet.anyConditional(FactMan, Cp))
    return false;
  if (Handler) {
    Handler->handleUnmatchedUnlock(Cp.getKind(), Cp.toString(), UnlockLoc,
                                   SourceLocation(), true);
    FSet.removeAllConditional(FactMan, Cp);
  } else if (!OwnOrigin || !FSet.removeConditional(FactMan, Cp, OwnOrigin) ||
             FSet.anyConditional(FactMan, Cp)) {
    return true;
  }
  // A pre-existing real negative already proves not-held on every path
  // and is kept; a weak one is superseded: the release proves it
  // everywhere from here.
  if (!Cp.negative())
    installNegativeFact(FSet, FactMan, !Cp, UnlockLoc,
                        /*OriginCall=*/nullptr, /*SpentCall=*/nullptr,
                        /*KeepExistingReal=*/true);
  return true;
}

enum UnderlyingCapabilityKind {
  UCK_Acquired,          ///< Any kind of acquired capability.
  UCK_ReleasedShared,    ///< Shared capability that was released.
  UCK_ReleasedExclusive, ///< Exclusive capability that was released.
};

struct UnderlyingCapability {
  CapabilityExpr Cap;
  UnderlyingCapabilityKind Kind;
};

class ScopedLockableFactEntry final
    : public FactEntry,
      private llvm::TrailingObjects<ScopedLockableFactEntry,
                                    UnderlyingCapability> {
  friend TrailingObjects;

private:
  const unsigned ManagedCapacity;
  unsigned ManagedSize = 0;
  /// The construction whose try-acquire attributes acquired the managed
  /// capabilities conditionally, if any: the origin of the conditional
  /// facts this guard created, which are exactly the ones its destructor
  /// releases (see unlock()); null for a definite guard.
  const Expr *CondAcquireExpr = nullptr;

  ScopedLockableFactEntry(const CapabilityExpr &CE, SourceLocation Loc,
                          SourceKind Src, unsigned ManagedCapacity)
      : FactEntry(ScopedLockable, CE, LK_Exclusive, Loc, Src),
        ManagedCapacity(ManagedCapacity) {}

  void addManaged(const CapabilityExpr &M, UnderlyingCapabilityKind UCK) {
    assert(ManagedSize < ManagedCapacity);
    new (getTrailingObjects() + ManagedSize) UnderlyingCapability{M, UCK};
    ++ManagedSize;
  }

  ArrayRef<UnderlyingCapability> getManaged() const {
    return getTrailingObjects(ManagedSize);
  }

public:
  static ScopedLockableFactEntry *create(llvm::BumpPtrAllocator &Alloc,
                                         const CapabilityExpr &CE,
                                         SourceLocation Loc, SourceKind Src,
                                         unsigned ManagedCapacity) {
    void *Storage =
        Alloc.Allocate(totalSizeToAlloc<UnderlyingCapability>(ManagedCapacity),
                       alignof(ScopedLockableFactEntry));
    return new (Storage) ScopedLockableFactEntry(CE, Loc, Src, ManagedCapacity);
  }

  CapExprSet getUnderlyingMutexes() const {
    CapExprSet UnderlyingMutexesSet;
    for (const UnderlyingCapability &UnderlyingMutex : getManaged())
      UnderlyingMutexesSet.push_back(UnderlyingMutex.Cap);
    return UnderlyingMutexesSet;
  }

  /// \name Adding managed locks
  /// Capacity for managed locks must have been allocated via \ref create.
  /// There is no reallocation in case the capacity is exceeded!
  /// \{
  void addLock(const CapabilityExpr &M) { addManaged(M, UCK_Acquired); }

  /// Record that this guard's construction \p Exp acquired its managed
  /// capabilities conditionally.
  void setCondAcquireExpr(const Expr *Exp) { CondAcquireExpr = Exp; }

  void addExclusiveUnlock(const CapabilityExpr &M) {
    addManaged(M, UCK_ReleasedExclusive);
  }

  void addSharedUnlock(const CapabilityExpr &M) {
    addManaged(M, UCK_ReleasedShared);
  }
  /// \}

  void
  handleRemovalFromIntersection(const FactSet &FSet, FactManager &FactMan,
                                SourceLocation JoinLoc, LockErrorKind LEK,
                                ThreadSafetyHandler &Handler) const override {
    if (isEndOfFunctionLEK(LEK))
      return;

    for (const auto &UnderlyingMutex : getManaged()) {
      // Held or possibly held: either way the scope still has it to release.
      const auto *Entry = FSet.findAny(FactMan, UnderlyingMutex.Cap);
      if ((UnderlyingMutex.Kind == UCK_Acquired && Entry) ||
          (UnderlyingMutex.Kind != UCK_Acquired && !Entry)) {
        // If this scoped lock manages another mutex, and if the underlying
        // mutex is still/not held, then warn about the underlying mutex.
        Handler.handleMutexHeldEndOfScope(UnderlyingMutex.Cap.getKind(),
                                          UnderlyingMutex.Cap.toString(), loc(),
                                          JoinLoc, LEK);
      }
    }
  }

  void handleLock(FactSet &FSet, FactManager &FactMan, const FactEntry &entry,
                  ThreadSafetyHandler &Handler) const override {
    for (const auto &UnderlyingMutex : getManaged()) {
      if (UnderlyingMutex.Kind == UCK_Acquired)
        lock(FSet, FactMan, UnderlyingMutex.Cap, entry.kind(), entry.loc(),
             &Handler);
      else
        unlock(FSet, FactMan, UnderlyingMutex.Cap, entry.loc(), &Handler);
    }
  }

  void handleUnlock(FactSet &FSet, FactManager &FactMan,
                    const CapabilityExpr &Cp, SourceLocation UnlockLoc,
                    bool FullyRemove,
                    ThreadSafetyHandler &Handler) const override {
    assert(!Cp.negative() && "Managing object cannot be negative.");
    for (const auto &UnderlyingMutex : getManaged()) {
      // Remove/lock the underlying mutex if it exists/is still unlocked; warn
      // on double unlocking/locking if we're not destroying the scoped object.
      ThreadSafetyHandler *TSHandler = FullyRemove ? nullptr : &Handler;
      if (UnderlyingMutex.Kind == UCK_Acquired) {
        unlock(FSet, FactMan, UnderlyingMutex.Cap, UnlockLoc, TSHandler);
      } else {
        LockKind kind = UnderlyingMutex.Kind == UCK_ReleasedShared
                            ? LK_Shared
                            : LK_Exclusive;
        lock(FSet, FactMan, UnderlyingMutex.Cap, kind, UnlockLoc, TSHandler);
      }
    }
    if (FullyRemove)
      FSet.removeFact(FactMan, *this);
  }

  static bool classof(const FactEntry *A) {
    return A->getFactEntryKind() == ScopedLockable;
  }

private:
  void lock(FactSet &FSet, FactManager &FactMan, const CapabilityExpr &Cp,
            LockKind kind, SourceLocation loc,
            ThreadSafetyHandler *Handler) const {
    if (const auto It = FSet.findDefiniteIter(FactMan, Cp); It != FSet.end()) {
      const auto &Fact = cast<LockableFactEntry>(FactMan[*It]);
      if (const FactEntry *RFact = Fact.attemptReenter(FactMan, kind)) {
        // This capability has been reentrantly acquired.
        FSet.replaceFact(FactMan, It, RFact);
      } else if (Handler) {
        Handler->handleDoubleLock(Cp.getKind(), Cp.toString(), Fact.loc(), loc,
                                  /*MaybeHeld=*/false);
      }
      return;
    }
    if (const FactEntry *Cond = FSet.firstConditional(FactMan, Cp)) {
      // Try-held: a reentrant acquire of the same kind adds the definite
      // level beside the conditional ones; anything else may deadlock.
      if (Cp.reentrant() && FSet.conditionalsAllOfKind(FactMan, Cp, kind))
        FSet.addLock(FactMan, FactMan.createFact<LockableFactEntry>(
                                  Cp, kind, loc, Managed));
      else if (Handler)
        Handler->handleDoubleLock(Cp.getKind(), Cp.toString(), Cond->loc(), loc,
                                  /*MaybeHeld=*/true);
      return;
    }
    consumeNegativeFact(FSet, FactMan, !Cp, /*AcquiringCall=*/nullptr);
    FSet.addLock(FactMan,
                 FactMan.createFact<LockableFactEntry>(Cp, kind, loc, Managed));
  }

  void unlock(FactSet &FSet, FactManager &FactMan, const CapabilityExpr &Cp,
              SourceLocation loc, ThreadSafetyHandler *Handler) const {
    if (const auto It = FSet.findDefiniteIter(FactMan, Cp); It != FSet.end()) {
      const auto &Fact = cast<LockableFactEntry>(FactMan[*It]);
      // The level this guard releases is the one it acquired: a try-guard
      // nesting over an already definite hold releases its own conditional
      // fact, and the outer definite hold survives. A conditional level
      // another call contributed stays: this guard did not acquire it, and
      // its own level was a definite one.
      if (CondAcquireExpr &&
          FSet.removeConditional(FactMan, Cp, CondAcquireExpr))
        return;
      if (const FactEntry *RFact = Fact.leaveReentrant(FactMan)) {
        // This capability remains reentrantly acquired.
        FSet.replaceFact(FactMan, It, RFact);
        return;
      }

      // As in LockableFactEntry::handleUnlock(): released -- unless a
      // conditional fact remains, in which case the capability is now
      // merely try-held -- and releasing a hold proved by a try-acquire's
      // success spends the call's stored result. Only an actual release
      // supersedes a coexisting weak negative: an unmatched unlock must
      // not destroy that evidence.
      const Expr *Spent = Fact.tryLockCall();
      FSet.erase(It);
      if (!FSet.anyConditional(FactMan, Cp))
        installNegativeFact(FSet, FactMan, !Cp, loc, /*OriginCall=*/nullptr,
                            /*SpentCall=*/Spent, /*KeepExistingReal=*/false);
      return;
    }
    if (handleUncheckedTryHeldUnlock(FSet, FactMan, Cp, loc, Handler,
                                     CondAcquireExpr))
      return;
    if (Handler)
      Handler->handleUnmatchedUnlock(Cp.getKind(), Cp.toString(), loc,
                                     unmatchedUnlockNoteLoc(FSet, FactMan, Cp),
                                     false);
  }
};

/// Per-switch facts getSwitchEdgeValue() needs on every outgoing edge,
/// computed once per terminator (getSwitchSummary()): the switch's own
/// case labels, whether the listed cases cover zero and one, and whether
/// the condition is boolean.
struct SwitchSummary {
  /// This switch's own case labels with their evaluated [Lo, Hi] value
  /// range: the fall-out successor can carry a label belonging to an
  /// enclosing switch, which says nothing about this condition.
  llvm::SmallDenseMap<const CaseStmt *, std::pair<llvm::APSInt, llvm::APSInt>,
                      8>
      OwnCases;
  bool ZeroListed = false;
  bool OneListed = false;
  bool IsBool = false;
};

/// Class which implements the core thread safety analysis routines.
class ThreadSafetyAnalyzer {
  friend class BuildLockset;
  friend class threadSafety::BeforeSet;

  llvm::BumpPtrAllocator Bpa;
  threadSafety::til::MemRegionRef Arena;
  threadSafety::SExprBuilder SxBuilder;

  ThreadSafetyHandler &Handler;
  const FunctionDecl *CurrentFunction;
  ASTContext *ASTCtx = nullptr;
  LocalVariableMap LocalVarMap;
  // The beta unchecked-result diagnostics already emitted, keyed by
  // "<join location>:<acquisition location>:<capability>". A join of three
  // or more predecessors is intersected pairwise, and some predecessor
  // orders lose the same fact twice -- e.g. (try-held, no-fact, try-held):
  // the fact-free middle predecessor removes it from the entry set with a
  // diagnostic, then the last predecessor re-supplies it one-sided and
  // would diagnose the same leak again (intersectAndWarn()).
  llvm::StringSet<> NeverCheckedWarned;
  // Maps constructed objects to `this` placeholder prior to initialization.
  llvm::SmallDenseMap<const Expr *, til::LiteralPtr *> ConstructedObjects;
  /// The capabilities named by a try-acquire call's attributes, translated
  /// in the call's own context and grouped by the attribute's lock kind and
  /// success value (Falsy: reported acquired when the call returns false).
  struct TryAcquireCaps {
    CapExprSet TruthyExclusive, TruthyShared;
    CapExprSet FalsyExclusive, FalsyShared;
    /// Capabilities reconcileTryAcquireCaps() moved out of the polarity
    /// groups: acquired regardless of the call's result. handleCall()
    /// turns them into unconditional acquisitions, with the diagnostic.
    /// Exclusive only when both polarities promised an exclusive hold; a
    /// cross-kind pairing guarantees no more than a shared hold either
    /// way.
    CapExprSet UnconditionalExclusive, UnconditionalShared;
    /// Whether try-held facts were created for these capabilities at the
    /// call, i.e. whether the walk has reached it. getEdgeLockset() only
    /// re-materializes a lost fact for a call that tracked one to lose.
    bool TracksFacts = false;

    /// Visit every capability the call's attributes name conditionally --
    /// the groups decodeTrylockBranch() builds its per-edge resolutions
    /// from -- until \p Pred returns true. An unconditional acquisition
    /// resolves on no edge, so a fact of one is never re-resolved.
    bool anyConditionalCap(
        llvm::function_ref<bool(const CapabilityExpr &)> Pred) const {
      for (const CapExprSet *Group :
           {&TruthyExclusive, &TruthyShared, &FalsyExclusive, &FalsyShared})
        for (const CapabilityExpr &Cap : *Group)
          if (Pred(Cap))
            return true;
      return false;
    }

    /// Visit every capability of every group until \p Pred returns true;
    /// a group added to this struct must be added here too.
    bool anyCap(llvm::function_ref<bool(const CapabilityExpr &)> Pred) const {
      for (const CapExprSet *Group :
           {&TruthyExclusive, &TruthyShared, &FalsyExclusive, &FalsyShared,
            &UnconditionalExclusive, &UnconditionalShared})
        for (const CapabilityExpr &Cap : *Group)
          if (Pred(Cap))
            return true;
      return false;
    }
  };
  // Maps each try-acquire call to its attributes' capabilities, recorded
  // before the lockset walk.
  llvm::SmallDenseMap<const Expr *, TryAcquireCaps> TryAcquireCapsMap;
  // Every capability some try-acquire call names, flattened and deduped
  // from the map by recordTryAcquireCalls() (isTryAcquireCapability()).
  CapExprSet AllTryAcquireCaps;
  FactManager FactMan;
  std::vector<CFGBlockInfo> BlockInfo;

  BeforeSet *GlobalBeforeSet;

public:
  ThreadSafetyAnalyzer(ThreadSafetyHandler &H, BeforeSet *Bset)
      : Arena(&Bpa), SxBuilder(Arena), Handler(H), FactMan(Bpa),
        GlobalBeforeSet(Bset) {}

  bool inCurrentScope(const CapabilityExpr &CapE);

  void addLock(FactSet &FSet, const FactEntry *Entry, bool ReqAttr = false);
  void addTryLock(FactSet &FSet, const CapabilityExpr &CE, LockKind LK,
                  SourceLocation Loc, const Expr *Call,
                  FactEntry::SourceKind Src = FactEntry::Acquired);
  void checkAcquiredCapability(FactSet &FSet, const FactEntry &Entry,
                               bool ReqAttr);
  const FactEntry *cloneAsWeak(const FactEntry &FE);
  bool isTryAcquireCapability(const CapabilityExpr &CE);
  // Whether the function constructs a scoped object whose constructor
  // try-acquires: such a call records in-walk (handleCall), so its
  // capabilities are missing from AllTryAcquireCaps while the blocks
  // before it are analyzed. Joins there keep every negative as weak
  // evidence rather than analyze differently by block order.
  bool HasUnrecordedTryAcquire = false;
  bool callNamesCapability(const Expr *Call, const CapabilityExpr &CE);
  void keepAsWeak(FactSet &Set, FactID Fact);
  void injectLoopWeakNegatives(const CFGBlock *Head, const CFGBlock *Latch,
                               PostOrderCFGView::CFGBlockSet &Visited);
  void removeLock(FactSet &FSet, const CapabilityExpr &CapE,
                  SourceLocation UnlockLoc, bool FullyRemove, LockKind Kind);

  template <typename AttrType>
  void getMutexIDs(CapExprSet &Mtxs, AttrType *Attr, const Expr *Exp,
                   const NamedDecl *D, til::SExpr *Self = nullptr);

  void recordTryAcquireCall(const Expr *Exp, const NamedDecl *D,
                            til::SExpr *Self = nullptr,
                            TryAcquireCaps *NoExprCaps = nullptr);
  void recordTryAcquireCalls(const PostOrderCFGView *SortedGraph);
  void reconcileTryAcquireCaps(TryAcquireCaps &Caps);

  /// Intermediate state for decodeTrylockBranch.
  struct TrylockDecode {
    /// The try-acquire call reached in the AST walk.
    const CallExpr *TrylockCall = nullptr;
    /// The condition tests the negated call result.
    bool Negate = false;
    /// Set when the branched-on variable merges the call's result with a
    /// constant: the branch-condition truthiness of the edges where the
    /// value may be the constant rather than the call's result (see
    /// decodeTrylockCond()).
    std::optional<bool> AmbiguousCond;
    /// The second of two structurally identical try-acquire calls whose
    /// merged result the condition branches on; null otherwise.
    const CallExpr *MergedCall = nullptr;
  };

  void decodeTrylockCond(const Stmt *Cond, LocalVarContext C, TrylockDecode &D);

  /// How one edge of a terminator's branch resolves one capability of the
  /// branched-on try-acquire call.
  enum class CapResolution : uint8_t {
    Unknown, ///< The edge does not decide this capability's outcome.
    Success, ///< The call acquired the capability.
    Failure, ///< The call did not acquire the capability.
  };

  /// One capability of the call with the resolution one branch direction
  /// or edge proves for it.
  struct TrylockEdgeCap {
    CapabilityExpr Cap;
    LockKind Kind;
    CapResolution Resolution;
  };

  struct TrylockBranch {
    /// The try-acquire call whose result the terminator
    /// branches on, or null if it does not branch on one.
    const CallExpr *TrylockCall = nullptr;
    /// When the branched-on variable merges the results of two structurally
    /// identical try-acquire calls, the second path's call (TrylockCall
    /// resolves to the first path's); null otherwise. A join over facts of
    /// the two calls keeps the resolved origin (intersectAndWarn()).
    const CallExpr *TrylockCall2 = nullptr;
    /// The call may not have executed on edges of this direction (the
    /// branched-on variable merges its result with a constant): each
    /// capability's resolution holds only if it did (TrylockEdge's
    /// Ambiguous).
    bool AmbiguousTrue = false, AmbiguousFalse = false;
    /// The call's capabilities for each branch direction.
    SmallVector<TrylockEdgeCap, 1> OnTrue, OnFalse;
  };

  // Memoize the decodeTrylockBranch result by BlockID.
  llvm::SmallDenseMap<unsigned, TrylockBranch, 8> TerminatorTrylockCache;

  // Memoize per-switch summaries (resolveTrylockEdge() visits a switch
  // once per successor).
  llvm::SmallDenseMap<const SwitchStmt *, SwitchSummary, 4> SwitchSummaries;
  const SwitchSummary &getSwitchSummary(ASTContext &Ctx, const SwitchStmt *SW);

  const TrylockBranch &decodeTrylockBranch(const CFGBlock *Block);

  /// The try-acquire calls a block's terminator branches on.
  struct TerminatorTrylockCall {
    const CallExpr *TrylockCall = nullptr;
    const CallExpr *TrylockCall2 = nullptr;
  };
  TerminatorTrylockCall getTerminatorTrylockCall(const CFGBlock *Block);
  const CallExpr *
  getConditionTrylockCallExpr(const CFGBlock *Block,
                              bool *ResolvesAllPaths = nullptr,
                              const CallExpr **MergedCall = nullptr);

  /// One edge from a TrylockBranch.
  struct TrylockEdge {
    const CallExpr *TrylockCall = nullptr;
    /// The edge cannot be taken at all (e.g. the implicit default of a
    /// switch that lists every value of a boolean condition).
    bool Infeasible = false;
    /// The call may not have executed on this edge (the branched-on
    /// variable merges its result with a constant): each capability's
    /// resolution holds only if it did.
    bool Ambiguous = false;
    SmallVector<TrylockEdgeCap, 2> Caps;
  };
  TrylockEdge resolveTrylockEdge(const CFGBlock *PredBlock,
                                 const CFGBlock *CurrBlock);

  bool getEdgeLockset(FactSet &Result, const FactSet &ExitSet,
                      const CFGBlock *PredBlock, const CFGBlock *CurrBlock);

  bool join(const FactEntry &A, const FactEntry &B, SourceLocation JoinLoc,
            LockErrorKind EntryLEK);

  void intersectAndWarn(
      FactSet &EntrySet, const FactSet &ExitSet, SourceLocation JoinLoc,
      LockErrorKind EntryLEK, LockErrorKind ExitLEK,
      const Expr *RebranchTryLock = nullptr,
      bool RebranchResolvesAllPaths = true,
      const Expr *RebranchTryLock2 = nullptr,
      const llvm::SmallPtrSetImpl<const Expr *> *CheckedAroundLoop = nullptr,
      bool ForwardJoin = false);

  void intersectAndWarn(FactSet &EntrySet, const FactSet &ExitSet,
                        SourceLocation JoinLoc, LockErrorKind LEK) {
    intersectAndWarn(EntrySet, ExitSet, JoinLoc, LEK, LEK);
  }

  void runAnalysis(AnalysisDeclContext &AC);

  void warnIfMutexNotHeld(const FactSet &FSet, const NamedDecl *D,
                          const Expr *Exp, AccessKind AK, Expr *MutexExp,
                          ProtectedOperationKind POK, til::SExpr *Self,
                          SourceLocation Loc);
  void warnIfAnyMutexNotHeldForRead(const FactSet &FSet, const NamedDecl *D,
                                    const Expr *Exp,
                                    llvm::ArrayRef<Expr *> Args,
                                    ProtectedOperationKind POK,
                                    SourceLocation Loc);
  void warnIfMutexHeld(const FactSet &FSet, const NamedDecl *D, const Expr *Exp,
                       Expr *MutexExp, til::SExpr *Self, SourceLocation Loc);

  void checkAccess(const FactSet &FSet, const Expr *Exp, AccessKind AK,
                   ProtectedOperationKind POK);
  void checkPtAccess(const FactSet &FSet, const Expr *Exp, AccessKind AK,
                     ProtectedOperationKind POK);
};

} // namespace

/// Process acquired_before and acquired_after attributes on Vd.
BeforeSet::BeforeInfo* BeforeSet::insertAttrExprs(const ValueDecl* Vd,
    ThreadSafetyAnalyzer& Analyzer) {
  // Create a new entry for Vd.
  BeforeInfo *Info = nullptr;
  {
    // Keep InfoPtr in its own scope in case BMap is modified later and the
    // reference becomes invalid.
    std::unique_ptr<BeforeInfo> &InfoPtr = BMap[Vd];
    if (!InfoPtr)
      InfoPtr.reset(new BeforeInfo());
    Info = InfoPtr.get();
  }

  for (const auto *At : Vd->attrs()) {
    switch (At->getKind()) {
      case attr::AcquiredBefore: {
        const auto *A = cast<AcquiredBeforeAttr>(At);

        // Read exprs from the attribute, and add them to BeforeVect.
        for (const auto *Arg : A->args()) {
          CapabilityExpr Cp =
            Analyzer.SxBuilder.translateAttrExpr(Arg, nullptr);
          if (const ValueDecl *Cpvd = Cp.valueDecl()) {
            Info->Vect.push_back(Cpvd);
            const auto It = BMap.find(Cpvd);
            if (It == BMap.end())
              insertAttrExprs(Cpvd, Analyzer);
          }
        }
        break;
      }
      case attr::AcquiredAfter: {
        const auto *A = cast<AcquiredAfterAttr>(At);

        // Read exprs from the attribute, and add them to BeforeVect.
        for (const auto *Arg : A->args()) {
          CapabilityExpr Cp =
            Analyzer.SxBuilder.translateAttrExpr(Arg, nullptr);
          if (const ValueDecl *ArgVd = Cp.valueDecl()) {
            // Get entry for mutex listed in attribute
            BeforeInfo *ArgInfo = getBeforeInfoForDecl(ArgVd, Analyzer);
            ArgInfo->Vect.push_back(Vd);
          }
        }
        break;
      }
      default:
        break;
    }
  }

  return Info;
}

BeforeSet::BeforeInfo *
BeforeSet::getBeforeInfoForDecl(const ValueDecl *Vd,
                                ThreadSafetyAnalyzer &Analyzer) {
  auto It = BMap.find(Vd);
  BeforeInfo *Info = nullptr;
  if (It == BMap.end())
    Info = insertAttrExprs(Vd, Analyzer);
  else
    Info = It->second.get();
  assert(Info && "BMap contained nullptr?");
  return Info;
}

/// Return true if any mutexes in FSet are in the acquired_before set of Vd.
void BeforeSet::checkBeforeAfter(const ValueDecl* StartVd,
                                 const FactSet& FSet,
                                 ThreadSafetyAnalyzer& Analyzer,
                                 SourceLocation Loc, StringRef CapKind) {
  SmallVector<BeforeInfo*, 8> InfoVect;

  // Do a depth-first traversal of Vd.
  // Return true if there are cycles.
  std::function<bool (const ValueDecl*)> traverse = [&](const ValueDecl* Vd) {
    if (!Vd)
      return false;

    BeforeSet::BeforeInfo *Info = getBeforeInfoForDecl(Vd, Analyzer);

    if (Info->Visited == 1)
      return true;

    if (Info->Visited == 2)
      return false;

    if (Info->Vect.empty())
      return false;

    InfoVect.push_back(Info);
    Info->Visited = 1;
    for (const auto *Vdb : Info->Vect) {
      // Exclude mutexes in our immediate before set.
      if (FSet.containsMutexDecl(Analyzer.FactMan, Vdb)) {
        StringRef L1 = StartVd->getName();
        StringRef L2 = Vdb->getName();
        Analyzer.Handler.handleLockAcquiredBefore(CapKind, L1, L2, Loc);
      }
      // Transitively search other before sets, and warn on cycles.
      if (traverse(Vdb)) {
        if (CycMap.try_emplace(Vd, true).second) {
          StringRef L1 = Vd->getName();
          Analyzer.Handler.handleBeforeAfterCycle(L1, Vd->getLocation());
        }
      }
    }
    Info->Visited = 2;
    return false;
  };

  traverse(StartVd);

  for (auto *Info : InfoVect)
    Info->Visited = 0;
}

/// Gets the value decl pointer from DeclRefExprs or MemberExprs.
static const ValueDecl *getValueDecl(const Expr *Exp) {
  if (const auto *CE = dyn_cast<ImplicitCastExpr>(Exp))
    return getValueDecl(CE->getSubExpr());

  if (const auto *DR = dyn_cast<DeclRefExpr>(Exp))
    return DR->getDecl();

  if (const auto *ME = dyn_cast<MemberExpr>(Exp))
    return ME->getMemberDecl();

  return nullptr;
}

bool ThreadSafetyAnalyzer::inCurrentScope(const CapabilityExpr &CapE) {
  const threadSafety::til::SExpr *SExp = CapE.sexpr();
  assert(SExp && "Null expressions should be ignored");

  if (const auto *LP = dyn_cast<til::LiteralPtr>(SExp)) {
    const ValueDecl *VD = LP->clangDecl();
    // Variables defined in a function are always inaccessible.
    if (!VD || !VD->isDefinedOutsideFunctionOrMethod())
      return false;
    // For now we consider static class members to be inaccessible.
    if (isa<CXXRecordDecl>(VD->getDeclContext()))
      return false;
    // Global variables are always in scope.
    return true;
  }

  // Members are in scope from methods of the same class.
  if (const auto *P = dyn_cast<til::Project>(SExp)) {
    if (!isa_and_nonnull<CXXMethodDecl>(CurrentFunction))
      return false;
    const ValueDecl *VD = P->clangDecl();
    return VD->getDeclContext() == CurrentFunction->getDeclContext();
  }

  return false;
}

/// Add a new lock to the lockset, warning if the lock is already there.
/// \param ReqAttr -- true if this is part of an initial Requires attribute.
void ThreadSafetyAnalyzer::addLock(FactSet &FSet, const FactEntry *Entry,
                                   bool ReqAttr) {
  if (Entry->shouldIgnore())
    return;
  assert(!Entry->tryHeld() && "conditional facts are added by addTryLock");

  checkAcquiredCapability(FSet, *Entry, ReqAttr);

  if (const FactEntry *Cp = FSet.findDefinite(FactMan, *Entry)) {
    // Held already: reacquire reentrantly or diagnose (handleLock()).
    // Conditional facts beside the definite one are unaffected -- each is
    // resolved by the branch on its own result.
    if (!Entry->asserted())
      Cp->handleLock(FSet, FactMan, *Entry, Handler);
    return;
  }
  if (const FactEntry *Cond = FSet.firstConditional(FactMan, *Entry)) {
    if (Entry->asserted()) {
      // An assert directly upgrades the capability to held, without a
      // diagnostic: it claims exactly that knowledge, and the conditional
      // acquisitions are subsumed by it -- their results, once checked,
      // must not add levels the assert already accounts for.
      FSet.removeAllConditional(FactMan, *Entry);
      FSet.addLock(FactMan, Entry);
      return;
    }
    // A reentrant acquire of the same kind adds the definite level beside
    // the conditional ones.
    if (Entry->reentrant() && isa<LockableFactEntry>(Entry) &&
        FSet.conditionalsAllOfKind(FactMan, *Entry, Entry->kind())) {
      FSet.addLock(FactMan, Entry);
      return;
    }
    // A blocking acquire over a try-held capability may deadlock: diagnose
    // it.
    Handler.handleDoubleLock(Entry->getKind(), Entry->toString(), Cond->loc(),
                             Entry->loc(), /*MaybeHeld=*/true);
    // If the program did not deadlock, the capability is now held.
    FSet.removeAllConditional(FactMan, *Entry);
    FSet.addLock(FactMan, Entry);
    return;
  }
  FSet.addLock(FactMan, Entry);
}

/// The checks an acquisition performs: consume (or require) the negative
/// capability, and check acquired_before/acquired_after ordering. A
/// try-acquire attempts the acquisition, so a conditional \p Entry is
/// checked the same way -- once, at the call. The negative capability is
/// consumed either way: after the call the capability is possibly held, so
/// a negative fact that predates it no longer describes the state (and must
/// not later testify that this call failed); on the call's failure edge
/// getEdgeLockset() re-establishes the negative fact, carrying the call as
/// its origin.
void ThreadSafetyAnalyzer::checkAcquiredCapability(FactSet &FSet,
                                                   const FactEntry &Entry,
                                                   bool ReqAttr) {
  if (!ReqAttr && !Entry.negative()) {
    // look for the negative capability, and remove it from the fact set.
    // A weak negative fact (not-held on only some paths, see
    // intersectAndWarn()) is likewise consumed -- after the call the
    // capability is possibly held everywhere -- but does not satisfy the
    // requirement: it proves nothing on the other paths.
    CapabilityExpr NegC = !Entry;
    if (!consumeNegativeFact(FSet, FactMan, NegC, Entry.tryLockCall()) &&
        inCurrentScope(Entry) && !Entry.asserted() && !Entry.reentrant())
      Handler.handleNegativeNotHeld(Entry.getKind(), Entry.toString(),
                                    NegC.toString(), Entry.loc());
  }

  // Check before/after constraints
  if (!Entry.asserted() && !Entry.declared()) {
    GlobalBeforeSet->checkBeforeAfter(Entry.valueDecl(), FSet, *this,
                                      Entry.loc(), Entry.getKind());
  }
}

/// Clone the negative fact \p FE marked weak: known to hold on only some
/// paths into the current program point.
const FactEntry *ThreadSafetyAnalyzer::cloneAsWeak(const FactEntry &FE) {
  assert(FE.negative() && "only negative facts are tracked as weak");
  auto *NewFact =
      FactMan.createFact<LockableFactEntry>(cast<LockableFactEntry>(FE));
  NewFact->setWeak();
  return NewFact;
}

/// Whether some try-acquire call in the function names a capability
/// matching \p CE in either polarity. Only such capabilities' negatives
/// serve the try-held machinery as weak or spent evidence, so joins keep
/// no others.
bool ThreadSafetyAnalyzer::isTryAcquireCapability(const CapabilityExpr &CE) {
  if (HasUnrecordedTryAcquire)
    return true;
  CapabilityExpr Inverse = !CE;
  return llvm::any_of(AllTryAcquireCaps, [&](const CapabilityExpr &Cap) {
    return CE.matches(Cap) || Inverse.matches(Cap);
  });
}

/// Whether \p Call's try-acquire attributes name a capability matching
/// \p CE itself (not its inverse): a resolved negative fact for such a
/// capability is the call's own promoted acquisition, not a failure-edge
/// record.
bool ThreadSafetyAnalyzer::callNamesCapability(const Expr *Call,
                                               const CapabilityExpr &CE) {
  auto It = TryAcquireCapsMap.find(Call);
  return It != TryAcquireCapsMap.end() &&
         It->second.anyConditionalCap(
             [&](const CapabilityExpr &Cap) { return CE.matches(Cap); });
}

/// Keep the one-sided negative fact \p Fact of a join in \p Set as weak
/// evidence: not-held on some path into the current program point (see
/// intersectAndWarn()).
void ThreadSafetyAnalyzer::keepAsWeak(FactSet &Set, FactID Fact) {
  const FactEntry &FE = FactMan[Fact];
  if (FE.weak())
    Set.addLockByID(Fact);
  else
    Set.addLock(FactMan, cloneAsWeak(FE));
}

/// Add a conditional fact for the capability \p CE acquired by the
/// try-acquire call \p Call at \p Loc; the fact remembers its originating
/// call. It joins the capability's other facts: a definite hold, which the
/// success edge deepens (at runtime a try-acquire over a held capability
/// fails rather than deadlocks, and even a reentrant one may fail), and
/// other calls' conditional facts, each resolved by its own branch.
/// What cannot be tracked is diagnosed and left untracked: a hold of the
/// other kind (shared vs. exclusive), one this acquisition can neither
/// nest in nor coexist with, and a repeat of this call over its own fact
/// (one fact per origin). \p Src is Managed for a scoped lockable's
/// construction, whose destructor conditionally releases (disarms) the
/// fact.
void ThreadSafetyAnalyzer::addTryLock(FactSet &FSet, const CapabilityExpr &CE,
                                      LockKind LK, SourceLocation Loc,
                                      const Expr *Call,
                                      FactEntry::SourceKind Src) {
  auto *Fact = FactMan.createFact<LockableFactEntry>(CE, LK, Loc, Src);
  Fact->setTryLock(Call, /*Conditional=*/true);
  if (Fact->shouldIgnore())
    return;

  checkAcquiredCapability(FSet, *Fact, /*ReqAttr=*/false);

  if (const FactEntry *Cp = FSet.findDefinite(FactMan, CE)) {
    if (Cp->kind() != LK || !isa<LockableFactEntry>(Cp)) {
      Handler.handleDoubleLock(CE.getKind(), CE.toString(), Cp->loc(), Loc,
                               /*MaybeHeld=*/false);
      return;
    }
    // A fresh execution of the call overwrites its stored result: a hold an
    // earlier execution proved is no longer determined by it.
    if (Cp->tryLockCall() == Call)
      FSet.replaceFact(
          FactMan, *Cp,
          cast<LockableFactEntry>(Cp)->withOrigin(FactMan, nullptr));
  }
  SmallVector<const FactEntry *, 2> Conds;
  FSet.collectConditional(FactMan, CE, Conds);
  for (const FactEntry *Cond : Conds) {
    if (Cond->kind() != LK || Cond->tryLockCall() == Call) {
      Handler.handleDoubleLock(CE.getKind(), CE.toString(), Cond->loc(), Loc,
                               /*MaybeHeld=*/true);
      return;
    }
  }
  FSet.addLock(FactMan, Fact);
}

/// Remove a lock from the lockset, warning if the lock is not there.
/// \param UnlockLoc The source location of the unlock (only used in error msg)
void ThreadSafetyAnalyzer::removeLock(FactSet &FSet, const CapabilityExpr &Cp,
                                      SourceLocation UnlockLoc,
                                      bool FullyRemove, LockKind ReceivedKind) {
  if (Cp.shouldIgnore())
    return;

  const FactEntry *LDat = FSet.findDefinite(FactMan, Cp);
  if (!LDat) {
    if (handleUncheckedTryHeldUnlock(FSet, FactMan, Cp, UnlockLoc, &Handler))
      return;
    Handler.handleUnmatchedUnlock(Cp.getKind(), Cp.toString(), UnlockLoc,
                                  unmatchedUnlockNoteLoc(FSet, FactMan, Cp),
                                  false);
    return;
  }

  // Generic lock removal doesn't care about lock kind mismatches, but
  // otherwise diagnose when the lock kinds are mismatched.
  if (ReceivedKind != LK_Generic && LDat->kind() != ReceivedKind) {
    Handler.handleIncorrectUnlockKind(Cp.getKind(), Cp.toString(), LDat->kind(),
                                      ReceivedKind, LDat->loc(), UnlockLoc);
  }

  LDat->handleUnlock(FSet, FactMan, Cp, UnlockLoc, FullyRemove, Handler);
}

/// Extract the list of mutexIDs from the attribute on an expression,
/// and push them onto Mtxs, discarding any duplicates.
template <typename AttrType>
void ThreadSafetyAnalyzer::getMutexIDs(CapExprSet &Mtxs, AttrType *Attr,
                                       const Expr *Exp, const NamedDecl *D,
                                       til::SExpr *Self) {
  if (Attr->args_size() == 0) {
    // The mutex held is the "this" object.
    CapabilityExpr Cp = SxBuilder.translateAttrExpr(nullptr, D, Exp, Self);
    if (Cp.isInvalid()) {
      warnInvalidLock(Handler, nullptr, D, Exp, Cp.getKind());
      return;
    }
    //else
    if (!Cp.shouldIgnore())
      Mtxs.push_back_nodup(Cp);
    return;
  }

  for (const auto *Arg : Attr->args()) {
    CapabilityExpr Cp = SxBuilder.translateAttrExpr(Arg, D, Exp, Self);
    if (Cp.isInvalid()) {
      warnInvalidLock(Handler, nullptr, D, Exp, Cp.getKind());
      continue;
    }
    //else
    if (!Cp.shouldIgnore())
      Mtxs.push_back_nodup(Cp);
  }
}

// Returns whether E is a compile-time constant, setting TCond to its boolean
// value. Looks through parentheses and evaluates constant expressions
// (constexpr values, enumerators), not just literals.
static bool getStaticBooleanValue(const Expr *E, bool &TCond,
                                  const ASTContext &Ctx) {
  return !E->isValueDependent() && E->EvaluateAsBooleanCondition(TCond, Ctx);
}

// If Cond can be traced back to a try-acquire function call, the `D` variable
// will be populated with the call and with how the branched-on value relates
// to its result -- negation (e.g. `if (!mu.tryLock(...))`), a merge with a
// constant, or a merge of two structurally identical calls.
void ThreadSafetyAnalyzer::decodeTrylockCond(const Stmt *Cond,
                                             LocalVarContext C,
                                             TrylockDecode &D) {
  if (!Cond)
    return;

  if (const auto *CallExp = dyn_cast<CallExpr>(Cond)) {
    if (CallExp->getBuiltinCallee() == Builtin::BI__builtin_expect)
      return decodeTrylockCond(CallExp->getArg(0), C, D);
    const auto *FD = dyn_cast_or_null<NamedDecl>(CallExp->getCalleeDecl());
    if (FD && FD->hasAttr<TryAcquireCapabilityAttr>())
      D.TrylockCall = CallExp;
    return;
  }
  else if (const auto *PE = dyn_cast<ParenExpr>(Cond))
    return decodeTrylockCond(PE->getSubExpr(), C, D);
  else if (const auto *CE = dyn_cast<ImplicitCastExpr>(Cond))
    return decodeTrylockCond(CE->getSubExpr(), C, D);
  else if (const auto *FE = dyn_cast<FullExpr>(Cond))
    return decodeTrylockCond(FE->getSubExpr(), C, D);
  else if (const auto *DRE = dyn_cast<DeclRefExpr>(Cond)) {
    // The reasoning below assumes every assignment to the variable is
    // visible in the map. A variable whose reference has escaped (captured
    // or bound by reference, address taken) can be mutated by any call in
    // between, so neither its direct definitions nor its merges identify
    // the branched-on value.
    if (LocalVarMap.isEscaped(DRE->getDecl()))
      return;
    LocalVarContext DefCtx = C;
    if (const Expr *E = LocalVarMap.lookupExpr(DRE->getDecl(), DefCtx))
      return decodeTrylockCond(E, DefCtx, D);
    // A merged ("phi") definition: if the variable merges one non-constant
    // definition with a constant of truthiness K (e.g. a try-acquire result
    // stored over a constant initializer), a branch on the variable still
    // identifies the non-constant definition -- on an edge where the
    // variable's truthiness is !K the value can only be that definition's
    // result. Record in AmbiguousCond the branch-condition truthiness of
    // the edges where the value may instead be the constant;
    // getEdgeLockset() refuses to treat those edges as proof that the call
    // executed. Only one merge can be resolved per condition.
    // The merge may sit behind a chain of references (a loop head wraps
    // every variable in a reference definition), so test the canonical
    // definition, not the immediate one.
    const auto *VDef = LocalVarMap.lookupCanonical(DRE->getDecl(), C);
    if (!VDef || !VDef->isPhi() || D.AmbiguousCond)
      return;
    ASTContext &ACtx = DRE->getDecl()->getASTContext();
    const Expr *NonConst = nullptr, *NonConst2 = nullptr;
    LocalVarContext NonConstCtx = C, NonConstCtx2 = C;
    unsigned NonConstID = 0, ConstID = 0;
    std::optional<bool> K;
    for (unsigned Op : {VDef->DirectRef, VDef->PhiAlt}) {
      LocalVarContext OpCtx = C;
      const Expr *E = LocalVarMap.lookupExprByID(Op, OpCtx);
      if (!E)
        return;
      // Any expression that constant-evaluates counts as the constant, not
      // just a literal: `bool b = kFalseConstant;` merges the same way as
      // `bool b = false;`.
      bool B;
      if (getStaticBooleanValue(E, B, ACtx)) {
        if (K && *K != B)
          return; // Constants of both truthinesses determine nothing.
        K = B;
        ConstID = Op;
      } else if (NonConst) {
        NonConst2 = E;
        NonConstCtx2 = OpCtx;
      } else {
        NonConst = E;
        NonConstCtx = OpCtx;
        NonConstID = Op;
      }
    }
    if (NonConst2) {
      // Two non-constant definitions: a branch still resolves the merge if
      // both are the same branch-relevant expression -- in practice two
      // structurally identical try-acquire calls, as in the retry idiom
      // `ok = mu.TryLock(); while (!ok) ok = mu.TryLock();` -- since either
      // way the variable holds "the result of that call". Resolve to the
      // first path's call: its fact is the one in the entry set wherever
      // this merge is branched on, and joins have verified the two paths'
      // states agree.
      llvm::FoldingSetNodeID ID1, ID2;
      NonConst->IgnoreParens()->Profile(ID1, ACtx, /*Canonical=*/true);
      NonConst2->IgnoreParens()->Profile(ID2, ACtx, /*Canonical=*/true);
      if (ID1 != ID2)
        return;
      // (In the unsound retry-without-checking variant
      // `b = mu.TryLock(); while (work()) b = mu.TryLock();` the second
      // call executes while the first result may still be pending: its
      // fact is a second conditional fact of the capability, and the
      // first's goes unchecked, which the loop join reports under
      // -Wthread-safety-beta.)
      // The second path's call resolves the same way; report it through
      // MergedCall so a join over the two calls' facts can keep the
      // resolved origin (intersectAndWarn()).
      //
      // Whenever the second path does not agree, the whole merge is
      // refused, not just the companion: the branch is decoded as a branch
      // on the first path's call, which only holds if the second path
      // reaches the same call in the same sense. Restoring the decode to
      // what it was on entry (BeforeD) leaves the merge unresolved, the
      // conservative answer -- dropping the companion alone would keep an
      // unsound resolution whose outcome depends on predecessor order.
      // Note the expression comparison above does not establish agreement:
      // the same expression (a reference to one variable) can resolve
      // differently per path (`if (c) { t = !t; ok = t; } else ok = t;`).
      const TrylockDecode BeforeD = D;
      D.MergedCall = nullptr;
      decodeTrylockCond(NonConst, NonConstCtx, D);
      const CallExpr *First = D.TrylockCall;
      // Either the first path resolves to no call at all, or its
      // resolution nests a two-call merge of its own, which a single
      // companion call cannot represent.
      if (!First || D.MergedCall) {
        D = BeforeD;
        return;
      }
      TrylockDecode D2 = BeforeD;
      D2.MergedCall = nullptr;
      decodeTrylockCond(NonConst2, NonConstCtx2, D2);
      const CallExpr *Second = D2.TrylockCall;
      if (!Second || D2.MergedCall || D2.Negate != D.Negate ||
          D2.AmbiguousCond != D.AmbiguousCond) {
        D = BeforeD;
        return;
      }
      // Both paths reaching the very same call needs no companion: the
      // merged value is that one call's result either way.
      if (Second == First)
        return;
      // The stored expressions were compared above, but they may be hops
      // (a copy through another variable) that resolved to calls of their
      // own: the identical-resolution premise holds for the calls
      // themselves, so compare those.
      llvm::FoldingSetNodeID CID1, CID2;
      First->Profile(CID1, ACtx, /*Canonical=*/true);
      Second->Profile(CID2, ACtx, /*Canonical=*/true);
      if (CID1 != CID2) {
        D = BeforeD;
        return;
      }
      D.MergedCall = Second;
      return;
    }
    if (!NonConst || !K)
      return;
    // The reasoning below is only sound if the constant is not a later
    // overwrite of the non-constant definition (`b = try_lock(); b = false;`
    // -- the capability may be held although the variable is false again):
    // the constant's definition chain must show the non-constant assignment
    // never executed on its paths.
    if (!LocalVarMap.chainAvoids(DRE->getDecl(), ConstID, NonConstID))
      return;
    // On the ambiguous edges the variable's truthiness is K; the
    // condition's is K adjusted by the negations applied so far.
    D.AmbiguousCond = *K != D.Negate;
    return decodeTrylockCond(NonConst, NonConstCtx, D);
  }
  else if (const auto *UOP = dyn_cast<UnaryOperator>(Cond)) {
    if (UOP->getOpcode() == UO_LNot) {
      D.Negate = !D.Negate;
      return decodeTrylockCond(UOP->getSubExpr(), C, D);
    }
    return;
  }
  else if (const auto *BOP = dyn_cast<BinaryOperator>(Cond)) {
    if (BOP->getOpcode() == BO_EQ || BOP->getOpcode() == BO_NE) {
      if (BOP->getOpcode() == BO_NE)
        D.Negate = !D.Negate;

      bool TCond = false;
      if (getStaticBooleanValue(BOP->getRHS(), TCond, *ASTCtx)) {
        if (!TCond)
          D.Negate = !D.Negate;
        return decodeTrylockCond(BOP->getLHS(), C, D);
      }
      TCond = false;
      if (getStaticBooleanValue(BOP->getLHS(), TCond, *ASTCtx)) {
        if (!TCond)
          D.Negate = !D.Negate;
        return decodeTrylockCond(BOP->getRHS(), C, D);
      }
      return;
    }
    if (BOP->getOpcode() == BO_LAnd) {
      // LHS must have been evaluated in a different block.
      return decodeTrylockCond(BOP->getRHS(), C, D);
    }
    if (BOP->getOpcode() == BO_LOr)
      return decodeTrylockCond(BOP->getRHS(), C, D);
    // An assignment used as a condition (`if ((b = mu.TryLock()))`)
    // evaluates to its right-hand side.
    if (BOP->getOpcode() == BO_Assign)
      return decodeTrylockCond(BOP->getRHS(), C, D);
    return;
  } else if (const auto *COP = dyn_cast<ConditionalOperator>(Cond)) {
    bool TCond, FCond;
    if (getStaticBooleanValue(COP->getTrueExpr(), TCond, *ASTCtx) &&
        getStaticBooleanValue(COP->getFalseExpr(), FCond, *ASTCtx)) {
      if (TCond && !FCond)
        return decodeTrylockCond(COP->getCond(), C, D);
      if (!TCond && FCond) {
        D.Negate = !D.Negate;
        return decodeTrylockCond(COP->getCond(), C, D);
      }
      return;
    }
    // One arm is a constant of truthiness K, the other is not: like the
    // merged variable above, a branch on the value still identifies the
    // non-constant arm -- on an edge where the value's truthiness is !K it
    // can only be that arm's result. Edges matching K are recorded as
    // ambiguous; only one merge can be resolved per condition.
    bool ArmCond;
    const Expr *NonConstArm = nullptr;
    std::optional<bool> K;
    if (getStaticBooleanValue(COP->getTrueExpr(), ArmCond, *ASTCtx)) {
      K = ArmCond;
      NonConstArm = COP->getFalseExpr();
    } else if (getStaticBooleanValue(COP->getFalseExpr(), ArmCond, *ASTCtx)) {
      K = ArmCond;
      NonConstArm = COP->getTrueExpr();
    }
    if (K && !D.AmbiguousCond) {
      D.AmbiguousCond = *K != D.Negate;
      return decodeTrylockCond(NonConstArm, C, D);
    }
  } else if (const auto *SE = dyn_cast<StmtExpr>(Cond)) {
    if (const auto *CS = SE->getSubStmt(); CS && !CS->body_empty()) {
      if (const auto *E = dyn_cast<Expr>(CS->body_back()))
        return decodeTrylockCond(E, C, D);
    }
  }
}

ThreadSafetyAnalyzer::TerminatorTrylockCall
ThreadSafetyAnalyzer::getTerminatorTrylockCall(const CFGBlock *Block) {
  const TrylockBranch &B = decodeTrylockBranch(Block);
  return {B.TrylockCall, B.TrylockCall2};
}

/// Find the try-acquire call whose result the condition starting at
/// \p Block branches on. Unlike getTerminatorTrylockCall(), this looks
/// through short-circuit evaluation: in a compound condition such as
/// `while (i < n && !ok)`, \p Block tests only `i < n` and the branch on the
/// try-acquire result sits in a successor block of the condition.
///
/// With \p ResolvesAllPaths, also reports whether every outgoing path of
/// \p Block reaches a branch on that same call's result: a short-circuit
/// edge escapes its condition without evaluating the rest, but may itself
/// lead to another branch on the result (`if (c && b) ...; else if (b)`),
/// which is verified by walking each escape edge the same way. A caller
/// weakening a definitely-held fact on the strength of the re-branch needs
/// this: on an escaping path that never re-branches, the weakened fact
/// leaks unresolved (intersectAndWarn()).
const CallExpr *
ThreadSafetyAnalyzer::getConditionTrylockCallExpr(const CFGBlock *Block,
                                                  bool *ResolvesAllPaths,
                                                  const CallExpr **MergedCall) {
  // The walk follows the successor edges of logical-operator terminators,
  // which stay within one condition expression, and the fall-through edge
  // of transition blocks (single successor, no terminator) -- e.g. where a
  // branch join meets a loop back edge, one hop before the loop condition
  // that re-branches on the merged variable. A transition block need not be
  // empty: its statements cannot invalidate the decode, which uses the
  // condition block's own ExitContext, and a write to the branched-on
  // variable in it makes the resolution itself refuse (decodeTrylockCond).
  // A fall-through edge can reach an earlier block (the transition block's
  // successor is the back edge's target), so the visited set keeps the walk
  // finite; it is shared with the escape walks below (any walk that fails
  // ends the search, and the all-paths check below describes how a merge
  // into a visited block resolves).
  llvm::SmallPtrSet<const CFGBlock *, 8> Visited;
  SmallVector<const CFGBlock *, 4> Escapes;
  const CallExpr *WalkMerged = nullptr;
  // The blocks whose terminator decodes to the call: their outgoing edges
  // are what must actually resolve the demoted fact.
  SmallVector<const CFGBlock *, 2> Deciders;
  auto Walk = [&](const CFGBlock *Block) -> const CallExpr * {
    while (Block) {
      if (!Visited.insert(Block).second) {
        TerminatorTrylockCall T = getTerminatorTrylockCall(Block);
        WalkMerged = T.TrylockCall2;
        if (T.TrylockCall)
          Deciders.push_back(Block);
        return T.TrylockCall;
      }
      if (TerminatorTrylockCall T = getTerminatorTrylockCall(Block);
          T.TrylockCall) {
        WalkMerged = T.TrylockCall2;
        Deciders.push_back(Block);
        return T.TrylockCall;
      }
      if (const auto *BOP =
              dyn_cast_or_null<BinaryOperator>(Block->getTerminatorStmt());
          BOP && BOP->isLogicalOp()) {
        // Evaluation of the condition continues on the not-short-circuiting
        // edge: the true edge for &&, the false edge for ||. The other edge
        // escapes the condition; remember it for the all-paths check.
        auto SI = Block->succ_begin();
        auto EscapeSI = SI;
        auto &Advance = BOP->getOpcode() == BO_LOr ? SI : EscapeSI;
        if (Advance == Block->succ_end())
          return nullptr; // A logical operator always has both successors.
        ++Advance;
        if (EscapeSI != Block->succ_end())
          if (const CFGBlock *Escape = EscapeSI->getReachableBlock())
            Escapes.push_back(Escape);
        Block = SI == Block->succ_end() ? nullptr : SI->getReachableBlock();
        continue;
      }
      if (!Block->getTerminatorStmt() && Block->succ_size() == 1) {
        Block = Block->succ_begin()->getReachableBlock();
        continue;
      }
      return nullptr;
    }
    return nullptr;
  };

  const CallExpr *Exp = Walk(Block);
  if (MergedCall)
    *MergedCall = Exp ? WalkMerged : nullptr;
  if (ResolvesAllPaths) {
    *ResolvesAllPaths = Exp != nullptr;
    // Each escape edge must itself lead to a branch on the same call (its
    // own escapes accumulate and are checked in turn). An escape landing
    // directly on an already-visited block has merged into a path already
    // verified to reach the call. A walk that reaches a visited block only
    // deeper in stops with that block's terminator decode (the
    // shared-visited early return above), so it succeeds only when that
    // block itself branches on the call, and otherwise fails the all-paths
    // check conservatively.
    while (Exp && !Escapes.empty()) {
      const CFGBlock *Escape = Escapes.pop_back_val();
      if (Visited.count(Escape))
        continue; // Merged into an already-verified path.
      if (Walk(Escape) != Exp) {
        *ResolvesAllPaths = false;
        break;
      }
    }
    // Finding the branch is not enough: it has to decide the result on at
    // least one edge. A switch on the result whose labels pin nothing
    // (`switch (ok) { default: }`) leaves every edge Unknown, so
    // getEdgeLockset() resolves nothing and a silently demoted hold would
    // leak unreported. A branch that resolves some of its edges keeps the
    // promise where it can; what escapes the others is the ordinary
    // conservatism the beta diagnostics report.
    for (const CFGBlock *D : Deciders) {
      if (!*ResolvesAllPaths)
        break;
      bool AnyEdgeResolves = false;
      for (CFGBlock::const_succ_iterator SI = D->succ_begin(),
                                         SE = D->succ_end();
           SI != SE && !AnyEdgeResolves; ++SI)
        if (const CFGBlock *Succ = SI->getReachableBlock()) {
          TrylockEdge Edge = resolveTrylockEdge(D, Succ);
          AnyEdgeResolves = Edge.Infeasible || Edge.TrylockCall;
        }
      if (!AnyEdgeResolves)
        *ResolvesAllPaths = false;
    }
  }
  return Exp;
}

/// Decode a try-acquire attribute's success value. An expression that does
/// not constant-evaluate reads as false.
static bool getTrySuccessValue(ASTContext &Ctx, const Expr *BrE) {
  bool Result;
  return BrE && getStaticBooleanValue(BrE, Result, Ctx) && Result;
}

/// If the terminator of \p Block branches on the result of a call to a
/// function annotated with try_acquire_capability (possibly negated or stored
/// in a local variable), return the capabilities recorded for the call, each
/// with the resolution every branch direction proves for it.
const ThreadSafetyAnalyzer::TrylockBranch &
ThreadSafetyAnalyzer::decodeTrylockBranch(const CFGBlock *Block) {
  const unsigned BlockID = Block->getBlockID();

  if (auto It = TerminatorTrylockCache.find(BlockID);
      It != TerminatorTrylockCache.end())
    return It->second;
  auto CacheMiss = [&]() -> const TrylockBranch & {
    return TerminatorTrylockCache[BlockID] = TrylockBranch{};
  };

  const Stmt *Cond = Block->getTerminatorCondition();
  if (!Cond)
    return CacheMiss();

  // We don't acquire try-locks on ?: branches, except when its result is used.
  if (const auto *COp =
          dyn_cast_if_present<ConditionalOperator>(Block->getTerminatorStmt()))
    if (!COp->getType()->isVoidType())
      return CacheMiss();

  TrylockDecode D;
  decodeTrylockCond(Cond, BlockInfo[BlockID].ExitContext, D);
  if (!D.TrylockCall)
    return CacheMiss();

  // Translate call truthiness to branch truthiness.
  TrylockBranch Result;
  Result.TrylockCall = D.TrylockCall;
  Result.TrylockCall2 = D.MergedCall;
  if (D.AmbiguousCond)
    (*D.AmbiguousCond ? Result.AmbiguousTrue : Result.AmbiguousFalse) = true;
  if (auto MapIt = TryAcquireCapsMap.find(D.TrylockCall);
      MapIt != TryAcquireCapsMap.end()) {
    const TryAcquireCaps &Caps = MapIt->second;
    auto AddCaps = [&](const CapExprSet &CapSet, LockKind LK, bool Success) {
      for (const CapabilityExpr &CE : CapSet) {
        (Success != D.Negate ? Result.OnTrue : Result.OnFalse)
            .push_back({CE, LK, CapResolution::Success});
        (Success != D.Negate ? Result.OnFalse : Result.OnTrue)
            .push_back({CE, LK, CapResolution::Failure});
      }
    };
    AddCaps(Caps.TruthyExclusive, LK_Exclusive, /*Success=*/true);
    AddCaps(Caps.TruthyShared, LK_Shared, /*Success=*/true);
    AddCaps(Caps.FalsyExclusive, LK_Exclusive, /*Success=*/false);
    AddCaps(Caps.FalsyShared, LK_Shared, /*Success=*/false);
  }
  // A fully-reconciled call (every capability moved to the unconditional
  // groups) records nothing here: it creates no try-held facts, and a
  // branch on its result proves nothing.
  if (Result.OnTrue.empty() && Result.OnFalse.empty())
    return CacheMiss();
  return TerminatorTrylockCache[BlockID] = std::move(Result);
}

/// What an edge out of a terminator implies about the branched-on value.
enum class EdgeValue {
  False,      ///< The value is zero on this edge.
  True,       ///< The value is nonzero on this edge.
  Unknown,    ///< The edge does not determine the value.
  Infeasible, ///< The edge cannot be taken (e.g. the implicit default of a
              ///< switch that lists every value of a boolean condition).
};

/// The value range [Lo, Hi] a case label covers (a single value unless it
/// is a GNU case range).
static std::pair<llvm::APSInt, llvm::APSInt> getCaseRange(ASTContext &Ctx,
                                                          const CaseStmt *CS) {
  llvm::APSInt Lo = CS->getLHS()->EvaluateKnownConstInt(Ctx);
  llvm::APSInt Hi =
      CS->getRHS() ? CS->getRHS()->EvaluateKnownConstInt(Ctx) : Lo;
  return std::make_pair(Lo, Hi);
}

const SwitchSummary &
ThreadSafetyAnalyzer::getSwitchSummary(ASTContext &Ctx, const SwitchStmt *SW) {
  auto [It, Inserted] = SwitchSummaries.try_emplace(SW);
  SwitchSummary &Sum = It->second;
  if (!Inserted)
    return Sum;
  for (const SwitchCase *SC = SW->getSwitchCaseList(); SC;
       SC = SC->getNextSwitchCase()) {
    const auto *CS = dyn_cast<CaseStmt>(SC);
    if (!CS)
      continue;
    auto [Lo, Hi] = getCaseRange(Ctx, CS);
    Sum.OwnCases.try_emplace(CS, Lo, Hi);
    Sum.ZeroListed |= Lo <= 0 && Hi >= 0;
    Sum.OneListed |= Lo <= 1 && Hi >= 1;
  }
  // Not just bool-typed conditions: an int-typed condition provably 0/1
  // (e.g. a comparison in C) derives the same way.
  Sum.IsBool = SW->getCond()->isKnownToHaveBooleanValue();
  return Sum;
}

/// Determine the truthiness of a switch condition along the edge to
/// \p CaseBlock.
static EdgeValue getSwitchEdgeValue(const SwitchSummary &Sum,
                                    const CFGBlock *CaseBlock) {
  // A case label pins the value -- but only a label belonging to this
  // switch: the implicit fall-out successor can itself be a labeled
  // statement, e.g. a case of an enclosing switch that the fall-out edge
  // falls through into, which says nothing about this switch's condition
  // beyond matching none of its cases (the derivation below).
  auto OwnCase = Sum.OwnCases.end();
  if (const auto *CS = dyn_cast_if_present<CaseStmt>(CaseBlock->getLabel()))
    OwnCase = Sum.OwnCases.find(CS);
  if (OwnCase != Sum.OwnCases.end()) {
    // The range was evaluated once for the summary; do not re-run the
    // constant evaluator per edge.
    auto [Lo, Hi] = OwnCase->second;
    // A label a boolean condition can never match is a dead edge.
    if (Sum.IsBool && (Lo > 1 || Hi < 0))
      return EdgeValue::Infeasible;
    if (Lo == 0 && Hi == 0)
      return EdgeValue::False;
    if (Lo <= 0 && Hi >= 0)
      return EdgeValue::Unknown; // A GNU case range spanning zero and nonzero.
    return EdgeValue::True;
  }

  // The default edge (explicit, or the implicit fall-out successor): the
  // value matches none of the case labels. If zero is listed the value must
  // be nonzero; for a boolean condition with one listed it must be zero --
  // and with both listed this edge cannot be taken at all.
  if (Sum.ZeroListed)
    return Sum.IsBool && Sum.OneListed ? EdgeValue::Infeasible
                                       : EdgeValue::True;
  if (Sum.IsBool && Sum.OneListed)
    return EdgeValue::False;
  return EdgeValue::Unknown;
}

/// Decode what the edge from \p PredBlock to \p CurrBlock proves about
/// conditional capabilities, selected by the truthiness the edge assigns
/// to the branched-on value. An edge that does not determine the value
/// reports no branch at all: the facts stay untouched either way.
ThreadSafetyAnalyzer::TrylockEdge
ThreadSafetyAnalyzer::resolveTrylockEdge(const CFGBlock *PredBlock,
                                         const CFGBlock *CurrBlock) {
  const TrylockBranch &B = decodeTrylockBranch(PredBlock);
  TrylockEdge Edge;
  if (!B.TrylockCall)
    return Edge;

  // Determine the truthiness of the branched-on value along this edge.
  EdgeValue CondVal = EdgeValue::Unknown;
  if (const auto *SW =
          dyn_cast_if_present<SwitchStmt>(PredBlock->getTerminatorStmt())) {
    ASTContext &Ctx = B.TrylockCall->getCalleeDecl()->getASTContext();
    CondVal = getSwitchEdgeValue(getSwitchSummary(Ctx, SW), CurrBlock);
  } else {
    bool TrueEdge = false, FalseEdge = false;
    int i = 0;
    for (CFGBlock::const_succ_iterator SI = PredBlock->succ_begin(),
                                       SE = PredBlock->succ_end();
         SI != SE && i < 2; ++SI, ++i)
      if (*SI == CurrBlock)
        (i == 0 ? TrueEdge : FalseEdge) = true;
    if (TrueEdge != FalseEdge)
      CondVal = TrueEdge ? EdgeValue::True : EdgeValue::False;
  }
  if (CondVal == EdgeValue::Infeasible) {
    // TrylockCall stays null: an edge that cannot be taken resolves
    // nothing, and Infeasible is meaningful on its own (getEdgeLockset()
    // tests it before the no-branch bail).
    Edge.Infeasible = true;
    return Edge;
  }
  if (CondVal == EdgeValue::Unknown)
    return Edge;

  Edge.TrylockCall = B.TrylockCall;
  // If the branched-on variable merges the call's result with a constant,
  // an edge matching the constant's truthiness does not prove the call
  // executed.
  Edge.Ambiguous =
      CondVal == EdgeValue::True ? B.AmbiguousTrue : B.AmbiguousFalse;
  const SmallVectorImpl<TrylockEdgeCap> &Dir =
      CondVal == EdgeValue::True ? B.OnTrue : B.OnFalse;
  Edge.Caps.assign(Dir.begin(), Dir.end());
  return Edge;
}

/// Find the lockset that holds on the edge between PredBlock
/// and CurrBlock.  The edge set is the exit set of PredBlock (passed
/// as the ExitSet parameter) plus any trylocks, which are conditionally held.
///
/// Returns true if the edge is infeasible: a fact already promoted to held
/// proves the branched-on try-acquire succeeded on every path into
/// PredBlock, so the failure edge cannot be taken. The caller skips such
/// edges at joins, like unreachable predecessors.
bool ThreadSafetyAnalyzer::getEdgeLockset(FactSet &Result,
                                          const FactSet &ExitSet,
                                          const CFGBlock *PredBlock,
                                          const CFGBlock *CurrBlock) {
  Result = ExitSet;

  TrylockEdge Edge = resolveTrylockEdge(PredBlock, CurrBlock);
  const CallExpr *Exp = Edge.TrylockCall;
  // Infeasibility stands on its own, before the call check: the flag must
  // not depend on TrylockCall also being set.
  if (Edge.Infeasible)
    return true;
  if (!Exp)
    return false;

  // If the branched-on variable merges the call's result with a constant,
  // an edge matching the constant's truthiness does not prove the call
  // executed. Each fact decides for itself what such an edge still proves:
  // it resolves as a failure edge for a fact whose own attribute reports no
  // success here (even the call executing would mean failure for that
  // capability, and the call not executing means it was never acquired),
  // while a fact whose attribute reports success is left untouched, like an
  // unresolved condition -- attributes carry their own success values, so
  // one call's capabilities can split both ways across the same edge. A
  // negative fact likewise concludes no infeasibility on such an edge: the
  // edge may be taken with the constant's value, the call never executed.
  // (A fact already promoted to held proves the call executed and succeeded
  // on every path into PredBlock -- the branch that promoted it overwrote
  // the constant -- so the promoted-fact infeasibility check below remains
  // correct even on an ambiguous edge.)
  const bool Ambiguous = Edge.Ambiguous;

  // Whether a capability is acquired on this edge: it is re-identified by
  // matching against the capabilities recorded at the call, with the
  // resolution this edge proves for each (resolveTrylockEdge()).
  auto FactSucceedsHere = [&](const CapabilityExpr &FE) {
    assert(!Edge.Caps.empty() &&
           "try-acquire fact without capabilities recorded at its call");
    if (llvm::any_of(Edge.Caps, [&](const TrylockEdgeCap &EC) {
          return EC.Resolution == CapResolution::Success && FE.matches(EC.Cap);
        }))
      return true;
    if (llvm::any_of(Edge.Caps, [&](const TrylockEdgeCap &EC) {
          return FE.matches(EC.Cap);
        }))
      return false;
    // A hold of the capability a release-style try-acquire gives up
    // (try_acquire_capability(true, !mu) named !mu, this fact is mu): the
    // call's outcome resolves it inverted -- acquiring !mu releases mu,
    // failing to acquire it leaves the hold standing. A join demotes such
    // a hold with the call as its origin (intersectAndWarn()).
    CapabilityExpr Inverse = !FE;
    const auto *InvEC = llvm::find_if(Edge.Caps, [&](const TrylockEdgeCap &EC) {
      return Inverse.matches(EC.Cap);
    });
    assert(InvEC != Edge.Caps.end() &&
           "try-acquire fact matches neither polarity's capabilities");
    return InvEC != Edge.Caps.end() &&
           InvEC->Resolution == CapResolution::Failure;
  };

  // This edge resolves every fact originating from this call, each with its
  // own attribute's polarity. A conditional fact is folded into the
  // capability's definite fact on the branch on which its attribute reports
  // success -- one level deeper, or newly created, either way proved by
  // this call -- and dropped on the other branch; it is resolved with the
  // capability recorded at the call, never a re-translation at this edge,
  // which could name a different capability (e.g. through a pointer
  // reassigned since the call).
  //
  // A definite fact already promoted by an earlier branch on the same
  // result proves its attribute reported success on every path into
  // PredBlock: an edge implying the opposite result cannot be taken, so the
  // caller skips it at joins like an unreachable predecessor (but still
  // analyzes a block this leaves without feasible predecessors, see
  // runAnalysis()); on other edges the promoted fact is kept unchanged --
  // re-resolving is not a new acquisition, so it keeps its reentrancy depth
  // and source, and the acquisition checks do not run again.
  SmallVector<const FactEntry *> ResolvedTryFacts;
  bool Infeasible = false;
  for (const auto &Fact : Result) {
    const FactEntry &FE = FactMan[Fact];
    if (FE.tryLockCall() != Exp)
      continue;
    if (FE.tryHeld()) {
      ResolvedTryFacts.push_back(&FE);
      continue;
    }
    // A resolved fact of the call: promoted to held on a success edge, or
    // recorded negative on a failure edge. A negative fact is a failure
    // record only when the call's attributes do not name it itself -- a
    // try-acquire can name a negative capability
    // (try_acquire_capability(true, !mu)), and such a fact resolved to
    // held is handled like any promoted one.
    if (FE.negative() && !callNamesCapability(Exp, FE)) {
      // A negative fact recorded on the call's failure edge (below): the
      // call provably failed to acquire this fact's capability on every
      // path here, so an edge on which the capability's own attribute
      // reports success cannot be taken; any other edge is simply
      // consistent with it (attributes carry their own success values, so
      // the test is per fact, not per edge). A weak negative proves the
      // failure on only some paths and cannot rule the edge out; nor can
      // one that merged with a spent-result negative (see SpentTryLock),
      // whose paths carry a truthy result; nor can an ambiguous edge be
      // ruled out at all, since it does not prove the call executed.
      if (!Ambiguous && !FE.weak() && !FE.spentTryLock() &&
          FactSucceedsHere(!FE))
        Infeasible = true;
      continue;
    }
    // A promoted fact; kept weak or spent-merged by a join it proves
    // nothing on every path, as above.
    if (!FE.weak() && !FE.spentTryLock() && !FactSucceedsHere(FE))
      Infeasible = true;
  }
  // An infeasible edge's lockset still seeds the analysis of a block left
  // without feasible predecessors (runAnalysis()). It is deliberately left
  // UNRESOLVED, contradictions and all: the infeasibility proof rests on
  // the local-variable map, which can be stale (a captured mutation), and
  // then the "dead" block is live with exactly this state
  // (tryheld_recheck_after_captured_mutation). Genuinely dead arms pay
  // with residual over-reports against the contradiction-preserving state.
  if (Infeasible)
    return true;
  for (const FactEntry *FE : ResolvedTryFacts) {
    const auto *Cond = cast<LockableFactEntry>(FE);
    const bool Succeeds = FactSucceedsHere(*FE);
    // An ambiguous edge does not prove the call executed, so it cannot
    // promote the fact; it stays try-held, like an unresolved condition.
    if (Succeeds && Ambiguous)
      continue;
    Result.removeFact(FactMan, *FE);
    if (Succeeds) {
      const CapabilityExpr NegC = !*FE;
      const FactEntry *Neg = Result.findDefinite(FactMan, NegC);
      // A negative that spent this call's stored result (a release of the
      // hold its success proved, see SpentTryLock) refutes the promotion:
      // the result stays truthy while the hold is gone, so re-resolving
      // must not resurrect it -- the fact resolves to released instead.
      if (Neg && Neg->spentTryLock() == Exp)
        continue;
      // The successful release of a negative capability discharges one
      // level of the positive hold: with levels remaining the capability
      // is still held, so the conditional negative resolves away instead
      // of promoting.
      if (FE->negative() && Neg && isa<LockableFactEntry>(Neg)) {
        if (const FactEntry *ShallowerPos =
                cast<LockableFactEntry>(Neg)->leaveReentrant(FactMan)) {
          Result.replaceFact(FactMan, *Neg, ShallowerPos);
          continue;
        }
      }
      // The promoted fact keeps its origin: this promotion is proved by the
      // branch, so joins and later branches on the call's result can
      // recognize it (see intersectAndWarn()). The acquisition checks ran
      // at the call (checkAcquiredCapability()); the proved acquisition now
      // consumes the negative capability the call could only require --
      // except a weak negative that spent another call's result, which
      // stays as evidence for that call's staleness.
      if (const FactEntry *Def = Result.findDefinite(FactMan, *FE))
        // A negative fact has no levels: the proved release supersedes an
        // older negative for the capability rather than deepening it.
        Result.replaceFact(
            FactMan, *Def,
            FE->negative()
                ? Cond->promote(FactMan)
                : cast<LockableFactEntry>(Def)->deepen(FactMan, Exp));
      else
        Result.addLock(FactMan, Cond->promote(FactMan));
      if (Neg && !(Neg->weak() && Neg->spentTryLock()))
        Result.removeFact(FactMan, *Neg);
      else if (FE->negative() && !Neg)
        // The proved release of a merely try-held capability consumes its
        // conditional facts: whichever call may have acquired it, it is
        // released now.
        Result.removeAllConditional(FactMan, NegC);
    } else if (!FE->negative() && !Result.findAny(FactMan, *FE) &&
               !Result.anyConditional(FactMan, !*FE)) {
      // Failure edge: the conditional fact is dropped, and when no fact of
      // the capability remains, this edge proves the call did not acquire
      // it: record that as a negative fact carrying the call as its
      // origin, so a later branch on the same result stays consistent (an
      // edge implying success is infeasible, above). A weak negative
      // (not-held on only some paths) is upgraded -- this edge proves it
      // on all -- keeping any spend evidence it carries; a real one is
      // kept, and so is a conditional fact of the negative capability (a
      // release the call may have proved, resolved in its own right). The
      // failure of a try-acquire of a negative capability itself proves
      // nothing about the positive capability, so nothing is recorded.
      installNegativeFact(Result, FactMan, !*FE, Exp->getExprLoc(),
                          /*OriginCall=*/Exp, /*SpentCall=*/nullptr,
                          /*KeepExistingReal=*/true);
    }
  }

  // Re-materialize a fact of the call that the analysis lost track of --
  // e.g. dropped at a join whose paths a loop separates -- as held on the
  // edge where its attribute reports success: the branch proves the call
  // acquired the capability. Refused when a negative fact for the
  // capability survives from anywhere but this call: it proves the hold
  // was since released, or another call failed to acquire it, on all
  // paths (a real negative) or on some path (a weak one, see
  // intersectAndWarn()) -- either way the stored result is stale there
  // and the hold must not be resurrected. This call's own failure-edge
  // negative kept weak by a join is consistent with the model: this edge
  // excludes the paths it holds on, so resolve over it. (Its real form
  // already proved this edge infeasible above.) An ambiguous edge proves
  // no acquisition either way -- the call may never have executed -- so it
  // re-materializes nothing.
  if (auto MapIt = TryAcquireCapsMap.find(Exp);
      !Ambiguous && MapIt != TryAcquireCapsMap.end() &&
      MapIt->second.TracksFacts) {
    for (const TrylockEdgeCap &EC : Edge.Caps) {
      if (EC.Resolution != CapResolution::Success)
        continue;
      const CapabilityExpr &CE = EC.Cap;
      // Any surviving fact of the capability -- a definite hold, or a
      // conditional fact of some call -- means it was not lost.
      if (Result.findAny(FactMan, CE))
        continue;
      if (const FactEntry *Neg = Result.findDefinite(FactMan, !CE)) {
        if (Neg->tryLockCall() != Exp || Neg->spentTryLock())
          continue;
        Result.removeFact(FactMan, *Neg);
      }
      auto *Fact =
          FactMan.createFact<LockableFactEntry>(CE, EC.Kind, Exp->getExprLoc());
      Fact->setTryLock(Exp, /*Conditional=*/false);
      Result.addLock(FactMan, Fact);
    }
  }
  return false;
}

namespace {

/// We use this class to visit different types of expressions in
/// CFGBlocks, and build up the lockset.
/// An expression may cause us to add or remove locks from the lockset, or else
/// output error messages related to missing locks.
/// FIXME: In future, we may be able to not inherit from a visitor.
class BuildLockset : public ConstStmtVisitor<BuildLockset> {
  friend class ThreadSafetyAnalyzer;

  ThreadSafetyAnalyzer *Analyzer;
  FactSet FSet;
  // The fact set for the function on exit.
  const FactSet &FunctionExitFSet;

  /// A `LocalVariableMap::Context` wrapper that groups a context 'Q' with its
  /// immediate predecessor 'P' for a program point.  If the program point is
  /// right after a Stmt 'S', 'P' is the pre-context of 'S' and 'Q' is the
  /// post-context of 'S'.  Otherwise, 'P' == 'Q'.
  ///
  /// A DualLocalVarContext sets the global context for VarDefinition lookup to
  /// the post-context 'Q',  once CREATED or UPDATED to the next program
  /// point.  One can temporarily switch the global context to either 'P' or 'Q'
  /// using `switchToContextForScope`. The lifetime of the global context
  /// switching is bound to the enclosing scope. The global context will be set
  /// back to the prior state by the end of the scope.  This is done by the
  /// returned ContextSwitchScope object.
  ///
  /// Note: The pre- and post-context of a Stmt are distinct only in Beta mode
  /// (i.e., `Analyzer.Handler.issueBetaWarnings()`) because of the
  /// out-parameter validation.  If not in Beta mode, the global context for
  /// VarDefinition lookup is invisible, thus this wrapper has no impact on the
  /// analysis.
  class DualLocalVarContext {
  public:
    enum Point : char { Pre = 0, Post = 1 };

    class ContextSwitchScope {
      DualLocalVarContext &DC;
      Point LastPoint;

    public:
      ContextSwitchScope(DualLocalVarContext &DC, Point LastPoint)
          : DC(DC), LastPoint(LastPoint) {}
      ContextSwitchScope(const ContextSwitchScope &) = delete;
      ContextSwitchScope &operator=(const ContextSwitchScope &) = delete;
      ~ContextSwitchScope() { DC.switchContextTo(LastPoint); }
    };

    /// Temporarily switch context to \p P as long as the returned object lives.
    [[nodiscard]] ContextSwitchScope switchToContextForScope(Point P) {
      Point PriorPoint = CurrPoint;
      switchContextTo(P);
      return ContextSwitchScope(*this, PriorPoint);
    }

    /// Update the pre- and post-contexts to be associated with the next Stmt \p
    /// S. Set the global context to the post-context of \p S upon returning.
    ///
    /// If \p S is null, the behavior is as if the Stmt is a no-op--the
    /// post-context will shift to be the pre-context and the new post-context
    /// is the same as the old one, resulting in identical pre- and
    /// post-contexts.
    void moveToNextContext(const Stmt *S) {
      PrePost[Pre] = PrePost[Post];

      const LocalVariableMap::Context &NewPostCtx =
          S ? Analyzer.LocalVarMap.getNextContext(CtxIndex, S, *PrePost[Post])
            : *PrePost[Pre];

      PrePost[Post] = &NewPostCtx;
      switchContextTo(Post);
    }

    /// Constructs a DualLocalVarContext for the entry program point, where pre-
    /// and post-contexts are both equal to the \p EntryContext.
    DualLocalVarContext(ThreadSafetyAnalyzer &Analyzer, unsigned EntryIdx,
                        const LocalVariableMap::Context *EntryContext)
        : Analyzer(Analyzer), PrePost{EntryContext, EntryContext},
          CurrPoint(Post), CtxIndex(EntryIdx) {
      assert(EntryContext);
      switchContextTo(Post);
    }

  private:
    ThreadSafetyAnalyzer &Analyzer;
    // PrePost[0] points to the pre-context and
    // PrePost[1] points to the post-context:
    std::array<const LocalVariableMap::Context *, 2> PrePost;
    Point CurrPoint;
    unsigned CtxIndex;

    void switchContextTo(Point P) {
      if (!Analyzer.Handler.issueBetaWarnings())
        return;
      Analyzer.SxBuilder.setLookupLocalVarExpr(
          [Ctx = *PrePost[P],
           Analyzer = &Analyzer](const NamedDecl *D) mutable -> const Expr * {
            return Analyzer->LocalVarMap.lookupExpr(D, Ctx);
          });
      CurrPoint = P;
    }
  };

  DualLocalVarContext LVarCtx;

  // To update the context used in attr-expr translation.  If `S` is non-null,
  // the context is updated to the program point right after 'S'.
  void updateLocalVarMapCtx(const Stmt *S) { LVarCtx.moveToNextContext(S); }

  // helper functions

  void checkAccess(const Expr *Exp, AccessKind AK,
                   ProtectedOperationKind POK = POK_VarAccess) {
    Analyzer->checkAccess(FSet, Exp, AK, POK);
  }
  void checkPtAccess(const Expr *Exp, AccessKind AK,
                     ProtectedOperationKind POK = POK_VarAccess) {
    Analyzer->checkPtAccess(FSet, Exp, AK, POK);
  }

  void handleCall(const Expr *Exp, const NamedDecl *D,
                  til::SExpr *Self = nullptr,
                  SourceLocation Loc = SourceLocation());
  void examineArguments(const FunctionDecl *FD,
                        CallExpr::const_arg_iterator ArgBegin,
                        CallExpr::const_arg_iterator ArgEnd,
                        bool SkipFirstParam = false);

public:
  BuildLockset(ThreadSafetyAnalyzer *Anlzr, CFGBlockInfo &Info,
               const FactSet &FunctionExitFSet)
      : ConstStmtVisitor<BuildLockset>(), Analyzer(Anlzr), FSet(Info.EntrySet),
        FunctionExitFSet(FunctionExitFSet),
        LVarCtx(*Analyzer, Info.EntryIndex, &Info.EntryContext) {
    updateLocalVarMapCtx(nullptr);
  }

  ~BuildLockset() { Analyzer->SxBuilder.setLookupLocalVarExpr(nullptr); }
  BuildLockset(const BuildLockset &) = delete;
  BuildLockset &operator=(const BuildLockset &) = delete;

  void VisitUnaryOperator(const UnaryOperator *UO);
  void VisitBinaryOperator(const BinaryOperator *BO);
  void VisitCastExpr(const CastExpr *CE);
  void VisitCallExpr(const CallExpr *Exp);
  void VisitCXXConstructExpr(const CXXConstructExpr *Exp);
  void VisitDeclStmt(const DeclStmt *S);
  void VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *Exp);
  void VisitReturnStmt(const ReturnStmt *S);
};

} // namespace

/// Warn if the LSet does not contain a lock sufficient to protect access
/// of at least the passed in AccessKind.
void ThreadSafetyAnalyzer::warnIfMutexNotHeld(
    const FactSet &FSet, const NamedDecl *D, const Expr *Exp, AccessKind AK,
    Expr *MutexExp, ProtectedOperationKind POK, til::SExpr *Self,
    SourceLocation Loc) {
  LockKind LK = getLockKindFromAccessKind(AK);
  CapabilityExpr Cp = SxBuilder.translateAttrExpr(MutexExp, D, Exp, Self);
  if (Cp.isInvalid()) {
    warnInvalidLock(Handler, MutexExp, D, Exp, Cp.getKind());
    return;
  } else if (Cp.shouldIgnore()) {
    return;
  }

  if (Cp.negative()) {
    // Negative capabilities act like locks excluded. A try-held capability
    // may be held, which violates the exclusion just the same.
    if (FSet.findDefinite(FactMan, !Cp) || FSet.anyConditional(FactMan, !Cp)) {
      Handler.handleFunExcludesLock(
          Cp.getKind(), D->getNameAsString(), (!Cp).toString(), Loc,
          /*MaybeHeld=*/!FSet.findDefinite(FactMan, !Cp));
      return;
    }

    // If this does not refer to a negative capability in the same class,
    // then stop here.
    if (!inCurrentScope(Cp))
      return;

    // Otherwise the negative requirement must be propagated to the caller.
    // A weak negative fact (not-held on only some paths) does not satisfy
    // it.
    if (const FactEntry *Neg = FSet.findDefinite(FactMan, Cp);
        !Neg || Neg->weak())
      Handler.handleNegativeNotHeld(D, Cp.toString(), Loc);
    return;
  }

  const FactEntry *LDat = FSet.findDefiniteUniv(FactMan, Cp);
  bool NoError = true;
  if (!LDat) {
    // No exact match found.  Look for a partial match.
    LDat = FSet.findDefinitePartialMatch(FactMan, Cp);
    if (LDat) {
      // Warn that there's no precise match.
      std::string PartMatchStr = LDat->toString();
      StringRef   PartMatchName(PartMatchStr);
      Handler.handleMutexNotHeld(Cp.getKind(), D, POK, Cp.toString(), LK, Loc,
                                 &PartMatchName);
    } else {
      // Warn that there's no match at all.
      Handler.handleMutexNotHeld(Cp.getKind(), D, POK, Cp.toString(), LK, Loc);
    }
    NoError = false;
  }
  // Make sure the mutex we found is the right kind.
  if (NoError && LDat && !LDat->isAtLeast(LK)) {
    Handler.handleMutexNotHeld(Cp.getKind(), D, POK, Cp.toString(), LK, Loc);
  }
}

void ThreadSafetyAnalyzer::warnIfAnyMutexNotHeldForRead(
    const FactSet &FSet, const NamedDecl *D, const Expr *Exp,
    llvm::ArrayRef<Expr *> Args, ProtectedOperationKind POK,
    SourceLocation Loc) {
  SmallVector<CapabilityExpr, 2> Caps;
  for (auto *Arg : Args) {
    CapabilityExpr Cp = SxBuilder.translateAttrExpr(Arg, D, Exp, nullptr);
    if (Cp.isInvalid()) {
      warnInvalidLock(Handler, Arg, D, Exp, Cp.getKind());
      continue;
    }
    if (Cp.shouldIgnore())
      continue;
    const FactEntry *LDat = FSet.findDefiniteUniv(FactMan, Cp);
    if (LDat && LDat->isAtLeast(LK_Shared))
      return; // At least one held — read access is safe.
    // FIXME: try findDefinitePartialMatch as a fallback to support
    //        -Wno-thread-safety-precise, as warnIfMutexNotHeld does.
    Caps.push_back(Cp);
  }
  if (Caps.empty())
    return;
  // Materialize names only now that we know we are going to warn.
  SmallVector<std::string, 2> NameStorage;
  SmallVector<StringRef, 2> Names;
  for (const auto &Cp : Caps) {
    NameStorage.push_back(Cp.toString());
    Names.push_back(NameStorage.back());
  }
  Handler.handleGuardedByAnyReadNotHeld(D, POK, Names, Loc);
}

/// Warn if the LSet contains the given lock.
void ThreadSafetyAnalyzer::warnIfMutexHeld(const FactSet &FSet,
                                           const NamedDecl *D, const Expr *Exp,
                                           Expr *MutexExp, til::SExpr *Self,
                                           SourceLocation Loc) {
  CapabilityExpr Cp = SxBuilder.translateAttrExpr(MutexExp, D, Exp, Self);
  if (Cp.isInvalid()) {
    warnInvalidLock(Handler, MutexExp, D, Exp, Cp.getKind());
    return;
  } else if (Cp.shouldIgnore()) {
    return;
  }

  // A try-held capability may be held, which violates the exclusion just
  // the same.
  if (FSet.findDefinite(FactMan, Cp))
    Handler.handleFunExcludesLock(Cp.getKind(), D->getNameAsString(),
                                  Cp.toString(), Loc, /*MaybeHeld=*/false);
  else if (FSet.anyConditional(FactMan, Cp))
    Handler.handleFunExcludesLock(Cp.getKind(), D->getNameAsString(),
                                  Cp.toString(), Loc, /*MaybeHeld=*/true);
}

/// Checks guarded_by and pt_guarded_by attributes.
/// Whenever we identify an access (read or write) to a DeclRefExpr that is
/// marked with guarded_by, we must ensure the appropriate mutexes are held.
/// Similarly, we check if the access is to an expression that dereferences
/// a pointer marked with pt_guarded_by.
void ThreadSafetyAnalyzer::checkAccess(const FactSet &FSet, const Expr *Exp,
                                       AccessKind AK,
                                       ProtectedOperationKind POK) {
  Exp = Exp->IgnoreImplicit()->IgnoreParenCasts();

  SourceLocation Loc = Exp->getExprLoc();

  // Local variables of reference type cannot be re-assigned;
  // map them to their initializer.
  while (const auto *DRE = dyn_cast<DeclRefExpr>(Exp)) {
    const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()->getCanonicalDecl());
    if (VD && VD->isLocalVarDecl() && VD->getType()->isReferenceType()) {
      if (const auto *E = VD->getInit()) {
        // Guard against self-initialization. e.g., int &i = i;
        if (E == Exp)
          break;
        Exp = E->IgnoreImplicit()->IgnoreParenCasts();
        continue;
      }
    }
    break;
  }

  if (const auto *UO = dyn_cast<UnaryOperator>(Exp)) {
    // For dereferences
    if (UO->getOpcode() == UO_Deref)
      checkPtAccess(FSet, UO->getSubExpr(), AK, POK);
    return;
  }

  if (const auto *BO = dyn_cast<BinaryOperator>(Exp)) {
    switch (BO->getOpcode()) {
    case BO_PtrMemD: // .*
      return checkAccess(FSet, BO->getLHS(), AK, POK);
    case BO_PtrMemI: // ->*
      return checkPtAccess(FSet, BO->getLHS(), AK, POK);
    default:
      return;
    }
  }

  if (const auto *AE = dyn_cast<ArraySubscriptExpr>(Exp)) {
    checkPtAccess(FSet, AE->getLHS(), AK, POK);
    return;
  }

  if (const auto *ME = dyn_cast<MemberExpr>(Exp)) {
    if (ME->isArrow())
      checkPtAccess(FSet, ME->getBase(), AK, POK);
    else
      checkAccess(FSet, ME->getBase(), AK, POK);
  }

  const ValueDecl *D = getValueDecl(Exp);
  if (!D || !D->hasAttrs())
    return;

  if (D->hasAttr<GuardedVarAttr>() && FSet.holdsNoCapability(FactMan)) {
    Handler.handleNoMutexHeld(D, POK, AK, Loc);
  }

  for (const auto *I : D->specific_attrs<GuardedByAttr>()) {
    if (AK == AK_Written || I->args_size() == 1) {
      // Write requires all capabilities; single-arg read uses the normal
      // per-lock warning path.
      for (auto *Arg : I->args())
        warnIfMutexNotHeld(FSet, D, Exp, AK, Arg, POK, nullptr, Loc);
    } else {
      // Multi-arg read: holding any one of the listed capabilities is
      // sufficient (a writer must hold all, so any one prevents writes).
      warnIfAnyMutexNotHeldForRead(FSet, D, Exp, I->args(), POK, Loc);
    }
  }
}

/// Checks pt_guarded_by and pt_guarded_var attributes.
/// POK is the same  operationKind that was passed to checkAccess.
void ThreadSafetyAnalyzer::checkPtAccess(const FactSet &FSet, const Expr *Exp,
                                         AccessKind AK,
                                         ProtectedOperationKind POK) {
  // Strip off paren- and cast-expressions, checking if we encounter any other
  // operator that should be delegated to checkAccess() instead.
  while (true) {
    if (const auto *PE = dyn_cast<ParenExpr>(Exp)) {
      Exp = PE->getSubExpr();
      continue;
    }
    if (const auto *CE = dyn_cast<CastExpr>(Exp)) {
      if (CE->getCastKind() == CK_ArrayToPointerDecay) {
        // If it's an actual array, and not a pointer, then it's elements
        // are protected by GUARDED_BY, not PT_GUARDED_BY;
        checkAccess(FSet, CE->getSubExpr(), AK, POK);
        return;
      }
      Exp = CE->getSubExpr();
      continue;
    }
    break;
  }

  if (const auto *UO = dyn_cast<UnaryOperator>(Exp)) {
    if (UO->getOpcode() == UO_AddrOf) {
      // Pointer access via pointer taken of variable, so the dereferenced
      // variable is not actually a pointer.
      checkAccess(FSet, UO->getSubExpr(), AK, POK);
      return;
    }
  }

  // Pass by reference/pointer warnings are under a different flag.
  ProtectedOperationKind PtPOK = POK_VarDereference;
  switch (POK) {
  case POK_PassByRef:
    PtPOK = POK_PtPassByRef;
    break;
  case POK_ReturnByRef:
    PtPOK = POK_PtReturnByRef;
    break;
  case POK_PassPointer:
    PtPOK = POK_PtPassPointer;
    break;
  case POK_ReturnPointer:
    PtPOK = POK_PtReturnPointer;
    break;
  default:
    break;
  }

  const ValueDecl *D = getValueDecl(Exp);
  if (!D || !D->hasAttrs())
    return;

  if (D->hasAttr<PtGuardedVarAttr>() && FSet.holdsNoCapability(FactMan))
    Handler.handleNoMutexHeld(D, PtPOK, AK, Exp->getExprLoc());

  for (auto const *I : D->specific_attrs<PtGuardedByAttr>()) {
    if (AK == AK_Written || I->args_size() == 1) {
      // Write requires all capabilities; single-arg read uses the normal
      // per-lock warning path.
      for (auto *Arg : I->args())
        warnIfMutexNotHeld(FSet, D, Exp, AK, Arg, PtPOK, nullptr,
                           Exp->getExprLoc());
    } else {
      // Multi-arg read: holding any one of the listed capabilities is
      // sufficient (a writer must hold all, so any one prevents writes).
      warnIfAnyMutexNotHeldForRead(FSet, D, Exp, I->args(), PtPOK,
                                   Exp->getExprLoc());
    }
  }
}

/// Process a function call, method call, constructor call,
/// or destructor call.  This involves looking at the attributes on the
/// corresponding function/method/constructor/destructor, issuing warnings,
/// and updating the locksets accordingly.
///
/// FIXME: For classes annotated with one of the guarded annotations, we need
/// to treat const method calls as reads and non-const method calls as writes,
/// and check that the appropriate locks are held. Non-const method calls with
/// the same signature as const method calls can be also treated as reads.
///
/// \param Exp   The call expression.
/// \param D     The callee declaration.
/// \param Self  If \p Exp = nullptr, the implicit this argument or the argument
///              of an implicitly called cleanup function.
/// \param Loc   If \p Exp = nullptr, the location.
void BuildLockset::handleCall(const Expr *Exp, const NamedDecl *D,
                              til::SExpr *Self, SourceLocation Loc) {
  // Move to the call Stmt so that both pre- and post-context are available.
  updateLocalVarMapCtx(Exp);

  // Most function attributes are associated with the pre-context. Exceptions
  // are AcquireCapability and AssertCapability, which ensure some locks are
  // held after the call, and thus are associated with the post-context. They
  // will require a temporary switch to the post-context during handling.
  //
  // Parameter attributes are restricted to scoped objects, and thus are NOT
  // context-sensitive.
  auto PreContextForThisScope =
      LVarCtx.switchToContextForScope(DualLocalVarContext::Pre);
  CapExprSet ExclusiveLocksToAdd, SharedLocksToAdd;
  CapExprSet ExclusiveLocksToRemove, SharedLocksToRemove, GenericLocksToRemove;
  CapExprSet ScopedReqsAndExcludes;
  // Try-acquire capabilities of a call without an expression (a destructor
  // or cleanup function): there is no result to branch on, but a
  // reconciled unconditional acquisition still applies.
  ThreadSafetyAnalyzer::TryAcquireCaps NoExprTryCaps;
  bool NoExprTryCapsRecorded = false;

  // Figure out if we're constructing an object of scoped lockable class
  CapabilityExpr Scp;
  if (Exp) {
    assert(!Self);
    const auto *TagT = Exp->getType()->getAs<TagType>();
    if (D->hasAttrs() && TagT && Exp->isPRValue()) {
      til::LiteralPtr *Placeholder =
          Analyzer->SxBuilder.createThisPlaceholder();
      [[maybe_unused]] auto inserted =
          Analyzer->ConstructedObjects.insert({Exp, Placeholder});
      assert(inserted.second && "Are we visiting the same expression again?");
      if (isa<CXXConstructExpr>(Exp))
        Self = Placeholder;
      if (TagT->getDecl()->getMostRecentDecl()->hasAttr<ScopedLockableAttr>())
        Scp = CapabilityExpr(Placeholder, Exp->getType(), /*Neg=*/false);
    }

    assert(Loc.isInvalid());
    Loc = Exp->getExprLoc();
  }

  for(const Attr *At : D->attrs()) {
    switch (At->getKind()) {
      // When we encounter a lock function, we need to add the lock to our
      // lockset.
      case attr::AcquireCapability: {
        auto PostContextForThisScope =
            LVarCtx.switchToContextForScope(DualLocalVarContext::Post);
        const auto *A = cast<AcquireCapabilityAttr>(At);
        Analyzer->getMutexIDs(A->isShared() ? SharedLocksToAdd
                                            : ExclusiveLocksToAdd,
                              A, Exp, D, Self);
        break;
      }

      // Try-acquired capabilities were already recorded for CallExprs, so
      // only a constructor or an expression-less call (a destructor or
      // cleanup function) is recorded here, on its first try-acquire
      // attribute, where its object placeholder is available.
      // The conditional locks are added to our lockset below, from the
      // recorded capabilities in TryAcquireCapsMap.
      case attr::TryAcquireCapability: {
        if (Exp ? (!isa<CXXConstructExpr>(Exp) ||
                   Analyzer->TryAcquireCapsMap.contains(Exp))
                : NoExprTryCapsRecorded)
          break;
        NoExprTryCapsRecorded = true;
        auto PostContextForThisScope =
            LVarCtx.switchToContextForScope(DualLocalVarContext::Post);
        Analyzer->recordTryAcquireCall(Exp, D, Self, &NoExprTryCaps);
        break;
      }

      // An assert will add a lock to the lockset, but will not generate
      // a warning if it is already there, and will not generate a warning
      // if it is not removed.
      case attr::AssertCapability: {
        auto PostContextForThisScope =
            LVarCtx.switchToContextForScope(DualLocalVarContext::Post);
        const auto *A = cast<AssertCapabilityAttr>(At);
        CapExprSet AssertLocks;
        Analyzer->getMutexIDs(AssertLocks, A, Exp, D, Self);
        for (const auto &AssertLock : AssertLocks)
          Analyzer->addLock(
              FSet, Analyzer->FactMan.createFact<LockableFactEntry>(
                        AssertLock, A->isShared() ? LK_Shared : LK_Exclusive,
                        Loc, FactEntry::Asserted));
        break;
      }

      // When we encounter an unlock function, we need to remove unlocked
      // mutexes from the lockset, and flag a warning if they are not there.
      case attr::ReleaseCapability: {
        const auto *A = cast<ReleaseCapabilityAttr>(At);
        if (A->isGeneric())
          Analyzer->getMutexIDs(GenericLocksToRemove, A, Exp, D, Self);
        else if (A->isShared())
          Analyzer->getMutexIDs(SharedLocksToRemove, A, Exp, D, Self);
        else
          Analyzer->getMutexIDs(ExclusiveLocksToRemove, A, Exp, D, Self);
        break;
      }

      case attr::RequiresCapability: {
        const auto *A = cast<RequiresCapabilityAttr>(At);
        for (auto *Arg : A->args()) {
          Analyzer->warnIfMutexNotHeld(FSet, D, Exp,
                                       A->isShared() ? AK_Read : AK_Written,
                                       Arg, POK_FunctionCall, Self, Loc);
          // use for adopting a lock
          if (!Scp.shouldIgnore())
            Analyzer->getMutexIDs(ScopedReqsAndExcludes, A, Exp, D, Self);
        }
        break;
      }

      case attr::LocksExcluded: {
        const auto *A = cast<LocksExcludedAttr>(At);
        for (auto *Arg : A->args()) {
          Analyzer->warnIfMutexHeld(FSet, D, Exp, Arg, Self, Loc);
          // use for deferring a lock
          if (!Scp.shouldIgnore())
            Analyzer->getMutexIDs(ScopedReqsAndExcludes, A, Exp, D, Self);
        }
        break;
      }

      // Ignore attributes unrelated to thread-safety
      default:
        break;
    }
  }

  // Recording reconciled the polarity groups (recordTryAcquireCall); the
  // capabilities the reconciliation moved out of them are acquired
  // regardless of the call's result: diagnose and add them
  // unconditionally. The diagnostic is emitted here in the walk, not at
  // recording, so that unreachable code stays silent as for every other
  // diagnostic.
  ThreadSafetyAnalyzer::TryAcquireCaps *TryCaps = nullptr;
  if (Exp) {
    if (auto It = Analyzer->TryAcquireCapsMap.find(Exp);
        It != Analyzer->TryAcquireCapsMap.end())
      TryCaps = &It->second;
  } else {
    TryCaps = &NoExprTryCaps;
  }
  if (TryCaps) {
    auto AddRegardless = [&](const CapExprSet &Unconditional,
                             CapExprSet &LocksToAdd) {
      for (const auto &M : Unconditional) {
        Analyzer->Handler.handleTryLockRegardlessOfResult(M.getKind(),
                                                          M.toString(), Loc);
        LocksToAdd.push_back_nodup(M);
      }
    };
    AddRegardless(TryCaps->UnconditionalExclusive, ExclusiveLocksToAdd);
    AddRegardless(TryCaps->UnconditionalShared, SharedLocksToAdd);
  }

  std::optional<CallExpr::const_arg_range> Args;
  if (Exp) {
    if (const auto *CE = dyn_cast<CallExpr>(Exp))
      Args = CE->arguments();
    else if (const auto *CE = dyn_cast<CXXConstructExpr>(Exp))
      Args = CE->arguments();
    else
      llvm_unreachable("Unknown call kind");
  }
  const auto *CalledFunction = dyn_cast<FunctionDecl>(D);
  if (CalledFunction && Args.has_value()) {
    for (auto [Param, Arg] : zip(CalledFunction->parameters(), *Args)) {
      if (isCallbackParam(Param))
        continue;
      CapExprSet DeclaredLocks;
      for (const Attr *At : Param->attrs()) {
        switch (At->getKind()) {
        case attr::AcquireCapability: {
          const auto *A = cast<AcquireCapabilityAttr>(At);
          Analyzer->getMutexIDs(A->isShared() ? SharedLocksToAdd
                                              : ExclusiveLocksToAdd,
                                A, Exp, D, Self);
          Analyzer->getMutexIDs(DeclaredLocks, A, Exp, D, Self);
          break;
        }

        case attr::ReleaseCapability: {
          const auto *A = cast<ReleaseCapabilityAttr>(At);
          if (A->isGeneric())
            Analyzer->getMutexIDs(GenericLocksToRemove, A, Exp, D, Self);
          else if (A->isShared())
            Analyzer->getMutexIDs(SharedLocksToRemove, A, Exp, D, Self);
          else
            Analyzer->getMutexIDs(ExclusiveLocksToRemove, A, Exp, D, Self);
          Analyzer->getMutexIDs(DeclaredLocks, A, Exp, D, Self);
          break;
        }

        case attr::RequiresCapability: {
          const auto *A = cast<RequiresCapabilityAttr>(At);
          for (auto *Arg : A->args())
            Analyzer->warnIfMutexNotHeld(FSet, D, Exp,
                                         A->isShared() ? AK_Read : AK_Written,
                                         Arg, POK_FunctionCall, Self, Loc);
          Analyzer->getMutexIDs(DeclaredLocks, A, Exp, D, Self);
          break;
        }

        case attr::LocksExcluded: {
          const auto *A = cast<LocksExcludedAttr>(At);
          for (auto *Arg : A->args())
            Analyzer->warnIfMutexHeld(FSet, D, Exp, Arg, Self, Loc);
          Analyzer->getMutexIDs(DeclaredLocks, A, Exp, D, Self);
          break;
        }

        default:
          break;
        }
      }
      if (DeclaredLocks.empty())
        continue;
      CapabilityExpr Cp(Analyzer->SxBuilder.translate(Arg, nullptr),
                        StringRef("mutex"), /*Neg=*/false, /*Reentrant=*/false);
      if (const auto *CBTE = dyn_cast<CXXBindTemporaryExpr>(Arg->IgnoreCasts());
          Cp.isInvalid() && CBTE) {
        if (auto Object = Analyzer->ConstructedObjects.find(CBTE->getSubExpr());
            Object != Analyzer->ConstructedObjects.end())
          Cp = CapabilityExpr(Object->second, StringRef("mutex"), /*Neg=*/false,
                              /*Reentrant=*/false);
      }
      const FactEntry *Fact = FSet.findDefinite(Analyzer->FactMan, Cp);
      if (!Fact) {
        Analyzer->Handler.handleMutexNotHeld(Cp.getKind(), D, POK_FunctionCall,
                                             Cp.toString(), LK_Exclusive,
                                             Exp->getExprLoc());
        continue;
      }
      const auto *Scope = cast<ScopedLockableFactEntry>(Fact);
      for (const auto &[a, b] :
           zip_longest(DeclaredLocks, Scope->getUnderlyingMutexes())) {
        if (!a.has_value()) {
          Analyzer->Handler.handleExpectFewerUnderlyingMutexes(
              Exp->getExprLoc(), D->getLocation(), Scope->toString(),
              b.value().getKind(), b.value().toString());
        } else if (!b.has_value()) {
          Analyzer->Handler.handleExpectMoreUnderlyingMutexes(
              Exp->getExprLoc(), D->getLocation(), Scope->toString(),
              a.value().getKind(), a.value().toString());
        } else if (!a.value().equals(b.value())) {
          Analyzer->Handler.handleUnmatchedUnderlyingMutexes(
              Exp->getExprLoc(), D->getLocation(), Scope->toString(),
              a.value().getKind(), a.value().toString(), b.value().toString());
          break;
        }
      }
    }
  }
  // Remove locks first to allow lock upgrading/downgrading.
  // FIXME -- should only fully remove if the attribute refers to 'this'.
  bool Dtor = isa<CXXDestructorDecl>(D);
  for (const auto &M : ExclusiveLocksToRemove)
    Analyzer->removeLock(FSet, M, Loc, Dtor, LK_Exclusive);
  for (const auto &M : SharedLocksToRemove)
    Analyzer->removeLock(FSet, M, Loc, Dtor, LK_Shared);
  for (const auto &M : GenericLocksToRemove)
    Analyzer->removeLock(FSet, M, Loc, Dtor, LK_Generic);

  // Add locks.
  FactEntry::SourceKind Source =
      !Scp.shouldIgnore() ? FactEntry::Managed : FactEntry::Acquired;
  for (const auto &M : ExclusiveLocksToAdd)
    Analyzer->addLock(FSet, Analyzer->FactMan.createFact<LockableFactEntry>(
                                M, LK_Exclusive, Loc, Source));
  for (const auto &M : SharedLocksToAdd)
    Analyzer->addLock(FSet, Analyzer->FactMan.createFact<LockableFactEntry>(
                                M, LK_Shared, Loc, Source));

  // Add conditional locks. A scoped lockable's construction acquires its
  // underlying capabilities conditionally too, as managed try-held facts:
  // a constructor has no result to branch on, but the guard's destructor
  // pairs exactly with the conditional acquisition -- it releases each
  // capability only if the guard holds it -- so it disarms the fact
  // silently (handleUncheckedTryHeldUnlock()).
  CapExprSet TryLocksExclusive, TryLocksShared, TryLocksManaged;
  if (Exp && TryCaps) {
    // Set as the walk reaches the call, not in the pre-walk: a branch
    // decoded before the call is walked -- a loop-top `if (ok)` above
    // `ok = mu.TryLock()` -- must not re-materialize a hold for an
    // acquisition that has not happened on the first iteration
    // (tryheld_retry_with_continue).
    TryCaps->TracksFacts = true;
    for (const auto &M : TryCaps->TruthyExclusive)
      TryLocksExclusive.push_back(M);
    for (const auto &M : TryCaps->FalsyExclusive)
      TryLocksExclusive.push_back(M);
    for (const auto &M : TryCaps->TruthyShared)
      TryLocksShared.push_back(M);
    for (const auto &M : TryCaps->FalsyShared)
      TryLocksShared.push_back(M);
    for (const auto &M : TryLocksExclusive)
      Analyzer->addTryLock(FSet, M, LK_Exclusive, Loc, Exp, Source);
    for (const auto &M : TryLocksShared)
      Analyzer->addTryLock(FSet, M, LK_Shared, Loc, Exp, Source);
    // The scoped object manages each capability once, whatever its kinds.
    for (const auto &M : TryLocksExclusive)
      TryLocksManaged.push_back_nodup(M);
    for (const auto &M : TryLocksShared)
      TryLocksManaged.push_back_nodup(M);
  }

  if (!Scp.shouldIgnore()) {
    // Add the managing object as a dummy mutex, mapped to the underlying mutex.
    auto *ScopedEntry = Analyzer->FactMan.createFact<ScopedLockableFactEntry>(
        Scp, Loc, FactEntry::Acquired,
        ExclusiveLocksToAdd.size() + SharedLocksToAdd.size() +
            TryLocksManaged.size() + ScopedReqsAndExcludes.size() +
            ExclusiveLocksToRemove.size() + SharedLocksToRemove.size());
    for (const auto &M : ExclusiveLocksToAdd)
      ScopedEntry->addLock(M);
    for (const auto &M : SharedLocksToAdd)
      ScopedEntry->addLock(M);
    for (const auto &M : TryLocksManaged)
      ScopedEntry->addLock(M);
    // The destructor releases the conditional facts this construction
    // created, which it finds by their origin (unlock()).
    if (!TryLocksManaged.empty())
      ScopedEntry->setCondAcquireExpr(Exp);
    for (const auto &M : ScopedReqsAndExcludes)
      ScopedEntry->addLock(M);
    for (const auto &M : ExclusiveLocksToRemove)
      ScopedEntry->addExclusiveUnlock(M);
    for (const auto &M : SharedLocksToRemove)
      ScopedEntry->addSharedUnlock(M);
    Analyzer->addLock(FSet, ScopedEntry);
  }
}

/// For unary operations which read and write a variable, we need to
/// check whether we hold any required mutexes. Reads are checked in
/// VisitCastExpr.
void BuildLockset::VisitUnaryOperator(const UnaryOperator *UO) {
  switch (UO->getOpcode()) {
    case UO_PostDec:
    case UO_PostInc:
    case UO_PreDec:
    case UO_PreInc:
      checkAccess(UO->getSubExpr(), AK_Written);
      break;
    default:
      break;
  }
}

/// For binary operations which assign to a variable (writes), we need to check
/// whether we hold any required mutexes.
/// FIXME: Deal with non-primitive types.
void BuildLockset::VisitBinaryOperator(const BinaryOperator *BO) {
  if (!BO->isAssignmentOp())
    return;
  checkAccess(BO->getLHS(), AK_Written);
  updateLocalVarMapCtx(BO);
}

/// Whenever we do an LValue to Rvalue cast, we are reading a variable and
/// need to ensure we hold any required mutexes.
/// FIXME: Deal with non-primitive types.
void BuildLockset::VisitCastExpr(const CastExpr *CE) {
  if (CE->getCastKind() != CK_LValueToRValue)
    return;
  checkAccess(CE->getSubExpr(), AK_Read);
}

void BuildLockset::examineArguments(const FunctionDecl *FD,
                                    CallExpr::const_arg_iterator ArgBegin,
                                    CallExpr::const_arg_iterator ArgEnd,
                                    bool SkipFirstParam) {
  // Currently we can't do anything if we don't know the function declaration.
  if (!FD)
    return;

  // NO_THREAD_SAFETY_ANALYSIS does double duty here.  Normally it
  // only turns off checking within the body of a function, but we also
  // use it to turn off checking in arguments to the function.  This
  // could result in some false negatives, but the alternative is to
  // create yet another attribute.
  if (FD->hasAttr<NoThreadSafetyAnalysisAttr>())
    return;

  const ArrayRef<ParmVarDecl *> Params = FD->parameters();
  auto Param = Params.begin();
  if (SkipFirstParam)
    ++Param;

  // There can be default arguments, so we stop when one iterator is at end().
  for (auto Arg = ArgBegin; Param != Params.end() && Arg != ArgEnd;
       ++Param, ++Arg) {
    QualType Qt = (*Param)->getType();
    if (Qt->isReferenceType())
      checkAccess(*Arg, AK_Read, POK_PassByRef);
    else if (Qt->isPointerType())
      checkPtAccess(*Arg, AK_Read, POK_PassPointer);
  }
}

void BuildLockset::VisitCallExpr(const CallExpr *Exp) {
  if (const auto *CE = dyn_cast<CXXMemberCallExpr>(Exp)) {
    const auto *ME = dyn_cast<MemberExpr>(CE->getCallee());
    // ME can be null when calling a method pointer
    const CXXMethodDecl *MD = CE->getMethodDecl();

    if (ME && MD) {
      if (ME->isArrow()) {
        // Should perhaps be AK_Written if !MD->isConst().
        checkPtAccess(CE->getImplicitObjectArgument(), AK_Read);
      } else {
        // Should perhaps be AK_Written if !MD->isConst().
        checkAccess(CE->getImplicitObjectArgument(), AK_Read);
      }
    }

    examineArguments(CE->getDirectCallee(), CE->arg_begin(), CE->arg_end());
  } else if (const auto *OE = dyn_cast<CXXOperatorCallExpr>(Exp)) {
    OverloadedOperatorKind OEop = OE->getOperator();
    switch (OEop) {
      case OO_Equal:
      case OO_PlusEqual:
      case OO_MinusEqual:
      case OO_StarEqual:
      case OO_SlashEqual:
      case OO_PercentEqual:
      case OO_CaretEqual:
      case OO_AmpEqual:
      case OO_PipeEqual:
      case OO_LessLessEqual:
      case OO_GreaterGreaterEqual:
        checkAccess(OE->getArg(1), AK_Read);
        [[fallthrough]];
      case OO_PlusPlus:
      case OO_MinusMinus:
        checkAccess(OE->getArg(0), AK_Written);
        break;
      case OO_Star:
      case OO_ArrowStar:
      case OO_Arrow:
      case OO_Subscript:
        if (!(OEop == OO_Star && OE->getNumArgs() > 1)) {
          // Grrr.  operator* can be multiplication...
          checkPtAccess(OE->getArg(0), AK_Read);
        }
        [[fallthrough]];
      default: {
        // TODO: get rid of this, and rely on pass-by-ref instead.
        const Expr *Obj = OE->getArg(0);
        checkAccess(Obj, AK_Read);
        // Check the remaining arguments. For method operators, the first
        // argument is the implicit self argument, and doesn't appear in the
        // FunctionDecl, but for non-methods it does.
        const FunctionDecl *FD = OE->getDirectCallee();
        examineArguments(FD, std::next(OE->arg_begin()), OE->arg_end(),
                         /*SkipFirstParam*/ !isa<CXXMethodDecl>(FD));
        break;
      }
    }
  } else {
    examineArguments(Exp->getDirectCallee(), Exp->arg_begin(), Exp->arg_end());
  }

  auto *D = dyn_cast_or_null<NamedDecl>(Exp->getCalleeDecl());

  if (D)
    handleCall(Exp, D);
  else
    // Even if we cannot handle the call, we need to update the context for the
    // Stmt:
    updateLocalVarMapCtx(Exp);
}

void BuildLockset::VisitCXXConstructExpr(const CXXConstructExpr *Exp) {
  const CXXConstructorDecl *D = Exp->getConstructor();
  if (D && D->isCopyConstructor()) {
    const Expr* Source = Exp->getArg(0);
    checkAccess(Source, AK_Read);
  } else {
    examineArguments(D, Exp->arg_begin(), Exp->arg_end());
  }
  if (D && D->hasAttrs())
    handleCall(Exp, D);
}

static const Expr *UnpackConstruction(const Expr *E) {
  if (auto *CE = dyn_cast<CastExpr>(E))
    if (CE->getCastKind() == CK_NoOp)
      E = CE->getSubExpr()->IgnoreParens();
  if (auto *CE = dyn_cast<CastExpr>(E))
    if (CE->getCastKind() == CK_ConstructorConversion ||
        CE->getCastKind() == CK_UserDefinedConversion)
      E = CE->getSubExpr();
  if (auto *BTE = dyn_cast<CXXBindTemporaryExpr>(E))
    E = BTE->getSubExpr();
  return E;
}

void BuildLockset::VisitDeclStmt(const DeclStmt *S) {
  for (auto *D : S->getDeclGroup()) {
    if (auto *VD = dyn_cast_or_null<VarDecl>(D)) {
      const Expr *E = VD->getInit();
      if (!E)
        continue;
      E = E->IgnoreParens();

      // handle constructors that involve temporaries
      if (auto *EWC = dyn_cast<ExprWithCleanups>(E))
        E = EWC->getSubExpr()->IgnoreParens();
      E = UnpackConstruction(E);

      if (auto Object = Analyzer->ConstructedObjects.find(E);
          Object != Analyzer->ConstructedObjects.end()) {
        Object->second->setClangDecl(VD);
        Analyzer->ConstructedObjects.erase(Object);
      }
    }
  }
  updateLocalVarMapCtx(S);
}

void BuildLockset::VisitMaterializeTemporaryExpr(
    const MaterializeTemporaryExpr *Exp) {
  if (const ValueDecl *ExtD = Exp->getExtendingDecl()) {
    if (auto Object = Analyzer->ConstructedObjects.find(
            UnpackConstruction(Exp->getSubExpr()));
        Object != Analyzer->ConstructedObjects.end()) {
      Object->second->setClangDecl(ExtD);
      Analyzer->ConstructedObjects.erase(Object);
    }
  }
}

void BuildLockset::VisitReturnStmt(const ReturnStmt *S) {
  if (Analyzer->CurrentFunction == nullptr)
    return;
  const Expr *RetVal = S->getRetValue();
  if (!RetVal)
    return;

  // If returning by reference or pointer, check that the function requires the
  // appropriate capabilities.
  const QualType ReturnType =
      Analyzer->CurrentFunction->getReturnType().getCanonicalType();
  if (ReturnType->isLValueReferenceType()) {
    Analyzer->checkAccess(
        FunctionExitFSet, RetVal,
        ReturnType->getPointeeType().isConstQualified() ? AK_Read : AK_Written,
        POK_ReturnByRef);
  } else if (ReturnType->isPointerType()) {
    Analyzer->checkPtAccess(
        FunctionExitFSet, RetVal,
        ReturnType->getPointeeType().isConstQualified() ? AK_Read : AK_Written,
        POK_ReturnPointer);
  }
}

/// Given two facts merging on a join point, possibly warn and decide whether to
/// keep or replace.
///
/// \return  false if we should keep \p A, true if we should take \p B.
bool ThreadSafetyAnalyzer::join(const FactEntry &A, const FactEntry &B,
                                SourceLocation JoinLoc,
                                LockErrorKind EntryLEK) {
  // Whether we can replace \p A by \p B.
  const bool CanModify = EntryLEK != LEK_LockedSomeLoopIterations;
  assert(!A.tryHeld() && !B.tryHeld() &&
         "conditional facts of one origin are identical; nothing to join");
  const unsigned int ReentrancyDepthA = A.getReentrancyDepth();
  const unsigned int ReentrancyDepthB = B.getReentrancyDepth();

  if (ReentrancyDepthA != ReentrancyDepthB) {
    Handler.handleMutexHeldEndOfScope(B.getKind(), B.toString(), B.loc(),
                                      JoinLoc, EntryLEK,
                                      /*ReentrancyMismatch=*/true);
    // Pick the FactEntry with the greater reentrancy depth as the "good"
    // fact to reduce potential later warnings.
    return CanModify && ReentrancyDepthA < ReentrancyDepthB;
  } else if (A.kind() != B.kind()) {
    // For managed capabilities, the destructor should unlock in the right mode
    // anyway. For asserted capabilities no unlocking is needed.
    if ((A.managed() || A.asserted()) && (B.managed() || B.asserted())) {
      // The shared capability subsumes the exclusive capability, if possible.
      bool ShouldTakeB = B.kind() == LK_Shared;
      if (CanModify || !ShouldTakeB)
        return ShouldTakeB;
    }
    Handler.handleExclusiveAndShared(B.getKind(), B.toString(), B.loc(),
                                     A.loc());
    // Take the exclusive capability to reduce further warnings.
    return CanModify && B.kind() == LK_Exclusive;
  } else {
    // The non-asserted capability is the one we want to track.
    return CanModify && A.asserted() && !B.asserted();
  }
}

/// Compute the intersection of two locksets and issue warnings for any
/// locks in the symmetric difference.
///
/// This function is used at a merge point in the CFG when comparing the lockset
/// of each branch being merged. For example, given the following sequence:
/// A; if () then B; else C; D; we need to check that the lockset after B and C
/// are the same. In the event of a difference, we use the intersection of these
/// two locksets at the start of D.
///
/// \param EntrySet A lockset for entry into a (possibly new) block.
/// \param ExitSet The lockset on exiting a preceding block.
/// \param JoinLoc The location of the join point for error reporting
/// \param EntryLEK The warning if a mutex is missing from \p EntrySet.
/// \param ExitLEK The warning if a mutex is missing from \p ExitSet.
/// \param RebranchTryLock The try-acquire call whose result the joining
/// block's terminator branches on, if any. A held/try-held difference
/// between facts that both originate from that call is not diagnosed as a
/// lost hold: the paths re-diverge at the terminator, so the merged state
/// keeps the call's conditional fact (any reentrancy depth is diagnosed but
/// kept) and re-resolves it on the outgoing edges (getEdgeLockset()). A
/// difference against a fact not created by that call is diagnosed normally.
/// \param RebranchResolvesAllPaths Whether every outgoing path of the
/// joining block reaches the branch on \p RebranchTryLock's result (false
/// when the branch was found behind a short-circuit, whose other edge
/// escapes unresolved). When false, weakening a definitely-held fact is
/// diagnosed at the join after all -- the exemption's promise of
/// re-resolution does not hold on the escaping paths -- though the fact is
/// still demoted so the paths that do re-branch resolve it.
/// \param RebranchTryLock2 When the branched-on variable merges the
/// results of two structurally identical try-acquire calls (the merge
/// resolves to \p RebranchTryLock, the first path's call), the second
/// path's call. A join of two try-held facts whose origins are exactly
/// these two calls keeps \p RebranchTryLock as the merged origin instead
/// of clearing it: the variable holds either call's result and both
/// resolve the capability identically, so the outgoing edges resolve the
/// merged fact like a single call's.
/// \param ForwardJoin Whether the join accumulates a block entry set still
/// under construction (branch and continue joins): only such joins may keep
/// one-sided negatives as weak facts. The sealed post-hoc comparisons (back
/// edges, the function exit) must not grow their sets.
///
/// Facts are paired by form (FactSet::findCounterpart()): a capability's
/// definite facts join through join(), and its conditional facts join per
/// origin -- two facts of one call are identical, and facts of different
/// calls coexist, each still resolved by the branch on its own result. The
/// interplay is where one side holds the capability definitely and the
/// other only conditionally (a "mixed" join): the merged state keeps the
/// conditional side's facts, and the definite side's extra level is
/// diagnosed as lost, or -- when both sides also hold definite levels --
/// as a reentrancy-depth mismatch.
void ThreadSafetyAnalyzer::intersectAndWarn(
    FactSet &EntrySet, const FactSet &ExitSet, SourceLocation JoinLoc,
    LockErrorKind EntryLEK, LockErrorKind ExitLEK, const Expr *RebranchTryLock,
    bool RebranchResolvesAllPaths, const Expr *RebranchTryLock2,
    const llvm::SmallPtrSetImpl<const Expr *> *CheckedAroundLoop,
    bool ForwardJoin) {
  FactSet EntrySetOrig = EntrySet;
  // A loop join compares against an entry set that was analyzed long ago:
  // it diagnoses, but must not rewrite that set.
  const bool CanModify = EntryLEK != LEK_LockedSomeLoopIterations;

  auto IsTrylockRebranched = [RebranchTryLock](const FactEntry &FE) {
    return RebranchTryLock && FE.tryLockCall() == RebranchTryLock;
  };
  // A definite hold of the capability a release-style try-acquire gives up
  // (try_acquire_capability(true, !mu), this fact is mu) is one-sided at
  // the join when the call's success released it on the other side -- and
  // only then: the hold carries no origin of its own, so the evidence must
  // be the call's own resolved fact for the inverse capability in
  // \p OtherSet. A hold that is one-sided for an unrelated reason (never
  // acquired on that path) is diagnosed normally.
  auto IsRebranchedInverseHold = [&, this](const FactSet &OtherSet,
                                           const FactEntry &FE) {
    if (!RebranchTryLock || FE.negative() || FE.tryLockCall())
      return false;
    const FactEntry *Released = OtherSet.findDefinite(FactMan, !FE);
    return Released && Released->tryLockCall() == RebranchTryLock;
  };
  // A one-sided fact under the re-branch exemption is carried (demoted to
  // try-held) on the premise that the fact's stored result is falsy on the
  // side missing it: lost to the call's failure edge, or never acquired.
  // A negative fact on the other side that spent a call's result (a
  // release of a hold a try-acquire's success had proved, see
  // SpentTryLock) refutes that premise: some path there keeps a truthy
  // stored result for a hold that is gone, so re-resolving the carried
  // fact could resurrect the dead hold -- e.g. the release in
  // `if (c) { if (ok) mu.Unlock(); }` followed by another `if (ok)`. A
  // weak such negative (on only some of that side's paths) refutes it the
  // same way, and so does a different call's spend: with one negative fact
  // per capability, the analysis cannot tell whose stale truth would do
  // the resurrecting.
  // A negative fact whose capability the re-branched call itself names is
  // that call's own resolved acquisition
  // (try_acquire_capability(true, !mu)), not not-held evidence: it takes
  // the re-branch demotion like any promoted fact instead of being kept
  // weak.
  auto IsCallsOwnNegative = [&, this](const FactEntry &FE) {
    return IsTrylockRebranched(FE) && callNamesCapability(RebranchTryLock, FE);
  };
  // Either form of the exemption, against the other side of the join.
  auto RebranchExemptAgainst = [&](const FactSet &OtherSet,
                                   const FactEntry &FE) {
    return IsTrylockRebranched(FE) || IsRebranchedInverseHold(OtherSet, FE);
  };
  // Whether a join may keep a one-sided negative as weak evidence: only
  // the forward joins that are still accumulating a block's entry set.
  auto MayKeepWeakNegative = [&](LockErrorKind LEK) {
    return LEK == LEK_LockedSomePredecessors || ForwardJoin;
  };
  auto RebranchVetoedByNegative = [&, this](const FactSet &OtherSet,
                                            const FactEntry &FE) {
    const FactEntry *Neg = OtherSet.findDefinite(FactMan, !FE);
    return Neg && Neg->spentTryLock();
  };
  // Warn about a fact the intersection removes (or weakens to try-held).
  // However, a capability managed by a scoped object is exempt -- the
  // scoped fact still knows to release it -- except where the scope itself
  // ends or repeats.
  auto WarnRemovedEntryFact = [&](const FactEntry &EntryFact) {
    if (!EntryFact.managed() || ExitLEK == LEK_LockedSomeLoopIterations ||
        ExitLEK == LEK_NotLockedAtEndOfFunction)
      EntryFact.handleRemovalFromIntersection(EntrySetOrig, FactMan, JoinLoc,
                                              ExitLEK, Handler);
  };
  auto WarnRemovedExitFact = [&](const FactEntry &ExitFact) {
    if (!ExitFact.managed() || EntryLEK == LEK_LockedAtEndOfFunction)
      ExitFact.handleRemovalFromIntersection(ExitSet, FactMan, JoinLoc,
                                             EntryLEK, Handler);
  };
  // Likewise for the beta diagnostic that a try-acquire's possible success
  // is carried into the join (or out of the function) unchecked: for a
  // conditional fact the analysis loses track of. Emitted once per (join,
  // acquisition, capability): the pairwise intersection of a
  // many-predecessor join can lose the same fact twice, and the leak it
  // reports is one (see NeverCheckedWarned).
  // \p LEK is the error kind governing the branch that fires the warning
  // (EntryLEK for a fact from the exit set, ExitLEK for one from the entry
  // set); it decides the at-end-of-function wording.
  auto WarnNeverChecked = [&](const FactEntry &FE, LockErrorKind LEK) {
    assert(FE.tryHeld() &&
           "only a conditional fact carries an unchecked result");
    if (!Handler.issueBetaWarnings() || !FE.lossNeedsWarning())
      return;
    // A capability managed by a scoped object is exempt at an interior
    // join, as for the lost-hold diagnostics above: the scoped fact is
    // still live and its destructor discharges the conditional
    // acquisition. Only where the scope itself ends or repeats can the
    // guard fail to check it.
    if (FE.managed() && LEK != LEK_LockedSomeLoopIterations &&
        !isEndOfFunctionLEK(LEK))
      return;
    SourceLocation LocAcquired = FE.tryLockCall()->getExprLoc();
    std::string Name = FE.toString();
    SmallString<64> Key;
    llvm::raw_svector_ostream(Key)
        << JoinLoc.getRawEncoding() << ':' << LocAcquired.getRawEncoding()
        << ':' << Name;
    if (!NeverCheckedWarned.insert(Key).second)
      return;
    Handler.handleTryAcquireNeverChecked(FE.getKind(), Name, LocAcquired,
                                         JoinLoc,
                                         /*AtEndOfFunction=*/
                                         isEndOfFunctionLEK(LEK));
  };
  // Diagnose a join where only the guaranteed depth of the hold differs.
  auto WarnReentrancyMismatch = [&](const FactEntry &FE, LockErrorKind LEK) {
    Handler.handleMutexHeldEndOfScope(FE.getKind(), FE.toString(), FE.loc(),
                                      JoinLoc, LEK,
                                      /*ReentrancyMismatch=*/true);
  };
  // The total number of levels a side holds, definite or conditional: what
  // a reentrancy-depth comparison of the two sides sees.
  auto Depth = [](const FactEntry *Def, bool HasCond) {
    return (Def ? Def->getReentrancyDepth() + 1 : 0) + (HasCond ? 1 : 0);
  };
  // Demote the definite hold \p Def, proved by the re-branched call (or
  // given up by it: a hold the call releases on success carries no origin
  // of its own, and the demotion adopts the call, whose outcome resolves
  // the hold inverted, getEdgeLockset()), to that call's conditional fact
  // in \p Into, which getEdgeLockset() re-resolves on the outgoing edges.
  // A mismatched reentrancy depth is diagnosed here but kept -- after the
  // warning, the deeper fact guards more of the releases downstream than
  // a stripped one would -- as the levels below the demoted one, no
  // longer determined by the call: returned for the caller to place, since
  // \p Def itself may belong to the other side.
  auto DemoteToTryHeld = [&, this](FactSet &Into, const FactEntry &Def,
                                   LockErrorKind LEK) -> const FactEntry * {
    const auto &LDef = cast<LockableFactEntry>(Def);
    if (Def.getReentrancyDepth() != 0)
      WarnReentrancyMismatch(Def, LEK);
    if (!Into.findConditional(FactMan, Def, RebranchTryLock))
      Into.addLock(FactMan, LDef.asConditional(FactMan, RebranchTryLock));
    if (const FactEntry *Shallower = LDef.leaveReentrant(FactMan))
      return cast<LockableFactEntry>(Shallower)->withOrigin(FactMan, nullptr);
    return nullptr;
  };
  // A negative fact weak or spent on either side of a pair is weak or
  // spent in the merged set: proven, or spending a result, on some of that
  // side's paths.
  auto MergeNegativeEvidence = [&, this](FactSet::iterator EntryIt,
                                         const FactEntry &EntryFact,
                                         const FactEntry &ExitFact) {
    const bool EitherWeak = EntryFact.weak() || ExitFact.weak();
    const Expr *EitherSpent = EntryFact.spentTryLock()
                                  ? EntryFact.spentTryLock()
                                  : ExitFact.spentTryLock();
    const FactEntry &Merged = FactMan[*EntryIt];
    if (!(EntryLEK == LEK_LockedSomePredecessors || ForwardJoin) ||
        !((EitherWeak && !Merged.weak()) ||
          (EitherSpent && !Merged.spentTryLock())))
      return;
    auto *NewFact =
        FactMan.createFact<LockableFactEntry>(cast<LockableFactEntry>(Merged));
    if (EitherWeak)
      NewFact->setWeak();
    if (EitherSpent && !NewFact->spentTryLock())
      NewFact->setSpentTryLock(EitherSpent);
    EntrySet.replaceFact(FactMan, EntryIt, NewFact);
  };

  // Find locks in ExitSet that conflict or are not in EntrySet, and warn.
  for (const auto &Fact : ExitSet) {
    const FactEntry &ExitFact = FactMan[Fact];
    FactSet::iterator EntryIt = EntrySet.findCounterpartIter(FactMan, ExitFact);
    const bool ExitHasCond = ExitSet.anyConditional(FactMan, ExitFact);
    const bool EntryHasCond = EntrySetOrig.anyConditional(FactMan, ExitFact);

    if (ExitFact.tryHeld()) {
      // A conditional fact. The same call's fact on the other side is
      // identical; otherwise it is carried into the merged state whenever
      // the other side holds the capability in any form (the other side's
      // extra is what gets diagnosed), or the terminator re-branches on its
      // call (unless a spent negative on the other side vetoes that, see
      // RebranchVetoedByNegative above). Missing from a path that does not
      // hold the capability at all, the analysis loses track of it: this
      // predecessor carries a try-acquire result into the join without its
      // result having been checked -- the capability may be leaked, the
      // beta diagnostic. So does one reaching the end of the function,
      // however the expected set holds the capability: nothing after can
      // check it. Loop joins are exempt for now -- a result the code
      // checks on the paths around the loop is not a leak, and telling
      // those apart from a result never checked anywhere takes resolution
      // of stored results, introduced separately.
      if (EntryIt != EntrySet.end())
        continue;
      if (!CanModify) {
        // At a loop join it warns only when the result is not branched on
        // anywhere inside the loop (CheckedAroundLoop): then the next
        // iteration re-executes the call (or the loop discards the result)
        // while this iteration's possible success was never checked -- a
        // check after the loop sees only the last result and cannot make
        // this sound. Joins without that information (continue joins, see
        // runAnalysis()) stay exempt, and so does a fact the pre-loop
        // state holds definitely (the depth mismatch is diagnosed instead);
        // another call's conditional fact there is no check of this one.
        if (!EntrySetOrig.findDefinite(FactMan, ExitFact) &&
            !IsTrylockRebranched(ExitFact) && CheckedAroundLoop &&
            !CheckedAroundLoop->count(ExitFact.tryLockCall()))
          WarnNeverChecked(ExitFact, EntryLEK);
        continue;
      }
      if (EntryLEK == LEK_LockedAtEndOfFunction) {
        WarnNeverChecked(ExitFact, EntryLEK);
        continue;
      }
      // Two structurally identical calls whose merged result the joining
      // block's terminator branches on (RebranchTryLock2): a side holding
      // one of them alone holds "the result of that call" either way, so
      // its fact folds into the fact of the call the merge resolves to
      // (the first path's), which the outgoing edges resolve like a single
      // call's fact. A side holding both executed the second call over the
      // first's unresolved fact: the variable then holds only the second
      // call's result, and the first's fact is left alone, to be reported
      // unchecked where it is lost.
      if (RebranchTryLock2 && ExitFact.tryLockCall() == RebranchTryLock2 &&
          !ExitSet.findConditional(FactMan, ExitFact, RebranchTryLock) &&
          EntrySetOrig.findConditional(FactMan, ExitFact, RebranchTryLock))
        continue;
      if (RebranchTryLock2 && ExitFact.tryLockCall() == RebranchTryLock &&
          !ExitSet.findConditional(FactMan, ExitFact, RebranchTryLock2) &&
          !EntrySetOrig.findConditional(FactMan, ExitFact, RebranchTryLock))
        if (const FactEntry *Twin = EntrySetOrig.findConditional(
                FactMan, ExitFact, RebranchTryLock2))
          EntrySet.removeFact(FactMan, *Twin);
      if (EntrySetOrig.findDefinite(FactMan, ExitFact) || EntryHasCond ||
          (IsTrylockRebranched(ExitFact) &&
           !RebranchVetoedByNegative(EntrySetOrig, ExitFact)))
        EntrySet.addLockByID(Fact);
      else
        WarnNeverChecked(ExitFact, EntryLEK);
      continue;
    }

    if (EntryIt != EntrySet.end()) {
      const FactEntry &EntryFact = FactMan[*EntryIt];
      if (ExitFact.negative()) {
        // Two negative facts: joined as ever, keeping either side's weak
        // or spent evidence.
        if (join(EntryFact, ExitFact, JoinLoc, EntryLEK))
          *EntryIt = Fact;
        MergeNegativeEvidence(EntryIt, EntryFact, ExitFact);
      } else if (EntryHasCond != ExitHasCond) {
        // Mixed: one side's hold is one conditional level deeper. Forgiven
        // when the definite side's extra level was proved by the call the
        // terminator re-branches on and the other side holds that call's
        // fact: the merged state re-resolves it -- unless the re-branch
        // sits behind a short-circuit whose other edge escapes without
        // resolving the result. Otherwise the capability is held on every
        // path and only the guaranteed depth differs: the
        // reentrancy-mismatch wording, like a join of unequal definite
        // depths, under the exemptions of the lost-hold path it replaces
        // (a scoped object still knows to release the levels it manages
        // at an interior join).
        const FactEntry &DefSide = ExitHasCond ? EntryFact : ExitFact;
        const FactEntry &CondSide = ExitHasCond ? ExitFact : EntryFact;
        const FactSet &CondSet = ExitHasCond ? ExitSet : EntrySetOrig;
        const bool Exempt =
            IsTrylockRebranched(DefSide) && RebranchResolvesAllPaths &&
            CondSet.findConditional(FactMan, DefSide, RebranchTryLock);
        if (!Exempt) {
          if (CondSide.lossNeedsWarning() &&
              !(CondSide.managed() && EntryLEK == LEK_LockedSomePredecessors))
            WarnReentrancyMismatch(CondSide, EntryLEK);
        } else if (CanModify && Depth(&EntryFact, EntryHasCond) !=
                                    Depth(&ExitFact, ExitHasCond)) {
          WarnReentrancyMismatch(ExitFact, EntryLEK);
        }
        // The merged state is the conditional side's.
        if (CanModify && ExitHasCond)
          *EntryIt = Fact;
      } else if (join(EntryFact, ExitFact, JoinLoc, EntryLEK)) {
        *EntryIt = Fact;
      }
      // If the two paths hold the capability via different origins, the
      // merged fact is not determined by either try-acquire's result.
      if (const FactEntry &Merged = FactMan[*EntryIt];
          EntryLEK == LEK_LockedSomePredecessors && Merged.tryLockCall() &&
          EntryFact.tryLockCall() != ExitFact.tryLockCall())
        EntrySet.replaceFact(
            FactMan, EntryIt,
            cast<LockableFactEntry>(Merged).withOrigin(FactMan, nullptr));
      continue;
    }

    // A definite fact of this predecessor only.
    if (ExitFact.negative() && !IsCallsOwnNegative(ExitFact)) {
      // A negative fact on this predecessor only: keep it in the merged
      // set as a weak fact at branch and continue joins -- evidence for
      // the try-held machinery that the capability was released, or a
      // try-acquire of it failed, on some path (see
      // RebranchVetoedByNegative above and getEdgeLockset()'s
      // re-materialization veto). Under a re-branch this loses nothing:
      // the failure edge re-derives the real negative over it. Skipped for
      // capabilities no try-acquire in the function names: nothing
      // consults weak facts for them. A promoted fact of a negative
      // capability the re-branched call itself names is excluded: it is
      // the call's own resolved acquisition, not not-held evidence, and
      // takes the re-branch demotion below like any promoted fact.
      if (MayKeepWeakNegative(EntryLEK) && isTryAcquireCapability(ExitFact))
        keepAsWeak(EntrySet, Fact);
    } else if (EntryHasCond) {
      if (!ExitHasCond) {
        // Mixed, with the other side holding only conditionally: forgiven
        // under the re-branch, else diagnosed as not held on the other
        // path; the merged state is the other side's.
        const bool Exempt =
            IsTrylockRebranched(ExitFact) && RebranchResolvesAllPaths &&
            EntrySetOrig.findConditional(FactMan, ExitFact, RebranchTryLock);
        if (!Exempt)
          WarnRemovedExitFact(ExitFact);
        else if (CanModify && ExitFact.getReentrancyDepth() != 0)
          WarnReentrancyMismatch(ExitFact, EntryLEK);
      } else {
        // Both sides hold conditionally, this one definitely as well: a
        // reentrancy-depth mismatch. Keep the deeper state, to minimize
        // follow-on warnings.
        WarnReentrancyMismatch(ExitFact, EntryLEK);
        if (CanModify)
          EntrySet.addLockByID(Fact);
      }
    } else if (RebranchExemptAgainst(EntrySetOrig, ExitFact) &&
               !RebranchVetoedByNegative(EntrySetOrig, ExitFact)) {
      // Held on this predecessor only, but the terminator re-branches on
      // the try-acquire that proved it (or gives it up): demote it to
      // try-held without warning, as getEdgeLockset will re-resolve it on
      // the outgoing edges.
      if (CanModify) {
        // A re-branch behind a short-circuit does not resolve the result on
        // the escaping edge: a definite hold weakened here can leak there,
        // so it is diagnosed at this join after all (the demotion stands,
        // for the paths that do re-branch).
        if (!RebranchResolvesAllPaths)
          WarnRemovedExitFact(ExitFact);
        if (const FactEntry *Rest =
                DemoteToTryHeld(EntrySet, ExitFact, EntryLEK))
          EntrySet.addLock(FactMan, Rest);
      }
    } else {
      // The analysis loses track of the fact here: the default
      // lost-capability diagnostic. (Its conditional facts, if any, are
      // lost above.)
      WarnRemovedExitFact(ExitFact);
    }
  }

  // Find locks in EntrySet that are not in ExitSet, and remove them.
  for (const auto &Fact : EntrySetOrig) {
    const FactEntry *EntryFact = &FactMan[Fact];
    const FactEntry *ExitFact = ExitSet.findCounterpart(FactMan, *EntryFact);
    if (ExitFact)
      continue;
    const bool ExitHasCond = ExitSet.anyConditional(FactMan, *EntryFact);
    const bool EntryHasCond = EntrySetOrig.anyConditional(FactMan, *EntryFact);

    if (EntryFact->tryHeld()) {
      // As above: a conditional fact is kept wherever the other side holds
      // the capability in any form or the terminator re-branches on its
      // call, and lost otherwise -- with the unchecked try-acquire on an
      // earlier predecessor; the beta warning again not at a loop join (a
      // try-held fact missing from a loop's back edge was checked inside
      // the loop, which is not a leak).
      // A fact of the second of two identical calls the terminator's merged
      // variable resolves (RebranchTryLock2), held alone on this side,
      // folds into the first call's fact, kept from the exit side above.
      if (RebranchTryLock2 && ExitLEK == LEK_LockedSomePredecessors &&
          EntryFact->tryLockCall() == RebranchTryLock2 &&
          !EntrySetOrig.findConditional(FactMan, *EntryFact, RebranchTryLock) &&
          ExitSet.findConditional(FactMan, *EntryFact, RebranchTryLock)) {
        EntrySet.removeFact(FactMan, *EntryFact);
        continue;
      }
      if (ExitSet.findDefinite(FactMan, *EntryFact) || ExitHasCond ||
          (IsTrylockRebranched(*EntryFact) &&
           !RebranchVetoedByNegative(ExitSet, *EntryFact)))
        continue;
      // (The CheckedAroundLoop narrowing the exit-set side applies is
      // deliberately not mirrored here: a fact reaching this arm was
      // released inside the loop, which is diagnosed at the release itself
      // unless an assert claimed the hold -- and a loop join leaves the
      // entry set unmodified, so the fact is diagnosed again wherever it
      // is finally lost.)
      if (ExitLEK != LEK_LockedSomeLoopIterations)
        WarnNeverChecked(*EntryFact, ExitLEK);
      if (ExitLEK == LEK_LockedSomePredecessors)
        EntrySet.removeFact(FactMan, *EntryFact);
      continue;
    }

    if (EntryFact->negative() && !IsCallsOwnNegative(*EntryFact)) {
      // As above: a one-sided negative is kept in the merged set as a
      // weak fact at branch and continue joins (or dropped, for a
      // capability no try-acquire names), and a promoted negative
      // capability of the re-branched call is excluded, taking the
      // demotion below; other joins leave the entry set unmodified.
      if (!EntryFact->weak()) {
        if (isTryAcquireCapability(*EntryFact)) {
          if (MayKeepWeakNegative(ExitLEK))
            EntrySet.replaceFact(FactMan, *EntryFact, cloneAsWeak(*EntryFact));
        } else if (ExitLEK == LEK_LockedSomePredecessors) {
          // Only a branch join rewrites the entry set; a loop join
          // leaves it as it is.
          EntrySet.removeFact(FactMan, *EntryFact);
        }
      }
      continue;
    }

    if (ExitHasCond) {
      if (!EntryHasCond) {
        // Mixed (see above): the exit side's conditional facts were kept
        // by the first loop; this definite fact is diagnosed unless
        // forgiven, and gives way to them.
        const bool Exempt =
            IsTrylockRebranched(*EntryFact) && RebranchResolvesAllPaths &&
            ExitSet.findConditional(FactMan, *EntryFact, RebranchTryLock);
        if (!Exempt)
          WarnRemovedEntryFact(*EntryFact);
        else if (CanModify && EntryFact->getReentrancyDepth() != 0)
          WarnReentrancyMismatch(*EntryFact, ExitLEK);
        if (ExitLEK == LEK_LockedSomePredecessors)
          EntrySet.removeFact(FactMan, *EntryFact);
      } else {
        // Both sides conditional, this one definitely deeper: diagnosed
        // above from the exit side's view; the deeper state is kept.
        WarnReentrancyMismatch(*EntryFact, ExitLEK);
      }
      continue;
    }
    if (RebranchExemptAgainst(ExitSet, *EntryFact) &&
        !RebranchVetoedByNegative(ExitSet, *EntryFact)) {
      // As above, but here the fact is kept in the intersection in its
      // demoted try-held form (except at a loop join, where the entry set
      // is left unmodified).
      if (CanModify) {
        // As above: an escaping short-circuit edge means the weakened
        // definite hold is diagnosed at the join after all.
        if (!RebranchResolvesAllPaths)
          WarnRemovedEntryFact(*EntryFact);
        const FactEntry *Rest = DemoteToTryHeld(EntrySet, *EntryFact, ExitLEK);
        if (Rest)
          EntrySet.replaceFact(FactMan, *EntryFact, Rest);
        else
          EntrySet.removeFact(FactMan, *EntryFact);
      }
      continue;
    }
    WarnRemovedEntryFact(*EntryFact);
    if (ExitLEK == LEK_LockedSomePredecessors)
      EntrySet.removeFact(FactMan, *EntryFact);
  }
}

// Return true if block B never continues to its successors.
static bool neverReturns(const CFGBlock *B) {
  if (B->hasNoReturnElement())
    return true;
  if (B->empty())
    return false;

  CFGElement Last = B->back();
  if (std::optional<CFGStmt> S = Last.getAs<CFGStmt>()) {
    if (isa<CXXThrowExpr>(S->getStmt()))
      return true;
  }

  // If B constructed a temporary whose destructor is noreturn, control entering
  // the decision block will always branch to the non-returning destructor.
  if (B->succ_size() == 1) {
    if (const CFGBlock *Succ = *B->succ_begin()) {
      if (Succ->getTerminator().isTemporaryDtorsBranch() &&
          Succ->succ_size() == 2) {
        // The decision block's terminator is the CXXBindTemporaryExpr; if B
        // bound this temporary, entering Succ from B takes the true (dtor)
        // edge; otherwise it takes the false (alternative dtor / continuation)
        // edge.
        const Stmt *Term = Succ->getTerminatorStmt();
        bool Bound = llvm::any_of(*B, [Term](const CFGElement &CE) {
          auto CS = CE.getAs<CFGStmt>();
          return CS && CS->getStmt() == Term;
        });
        if (const auto *Next =
                (Bound ? *Succ->succ_begin() : *(Succ->succ_begin() + 1))
                    .getReachableBlock())
          return neverReturns(Next);
      }
    }
  }

  return false;
}

/// The same capability listed under opposite success values -- of either
/// lock kind -- is acquired regardless of the call's result: move it out
/// of the polarity groups into an unconditional group, leaving every
/// remaining capability recorded under exactly one polarity and kind.
/// Exclusive under both polarities stays exclusive. A cross-kind pairing
/// (e.g. exclusive on success, shared on failure) may be a deliberate
/// API, but a single fact cannot represent a hold whose kind varies with
/// the result, so it keeps only the guarantee that holds either way: an
/// unconditional shared hold. handleCall() adds the unconditional groups
/// to the lockset, with the diagnostic.
void ThreadSafetyAnalyzer::reconcileTryAcquireCaps(TryAcquireCaps &Caps) {
  CapExprSet Regardless;
  auto CollectAcquiredOnFailureToo = [&](const CapExprSet &Truthy) {
    for (const auto &M : Truthy)
      if (Caps.FalsyExclusive.contains(M) || Caps.FalsyShared.contains(M))
        Regardless.push_back_nodup(M);
  };
  CollectAcquiredOnFailureToo(Caps.TruthyExclusive);
  CollectAcquiredOnFailureToo(Caps.TruthyShared);
  if (Regardless.empty())
    return;
  for (const auto &M : Regardless)
    (Caps.TruthyExclusive.contains(M) && Caps.FalsyExclusive.contains(M)
         ? Caps.UnconditionalExclusive
         : Caps.UnconditionalShared)
        .push_back_nodup(M);
  auto DropRegardless = [&](CapExprSet &Set) {
    llvm::erase_if(
        Set, [&](const CapabilityExpr &M) { return Regardless.contains(M); });
  };
  DropRegardless(Caps.TruthyExclusive);
  DropRegardless(Caps.TruthyShared);
  DropRegardless(Caps.FalsyExclusive);
  DropRegardless(Caps.FalsyShared);
}

/// Record the capabilities named by the try-acquire attributes of the call
/// or construction \p Exp to \p D into TryAcquireCapsMap, translated in the
/// currently installed context, and reconcile degenerate annotations. A
/// call without an expression (a destructor or cleanup function) records
/// into \p NoExprCaps instead: there is no result to branch on, but a
/// reconciled unconditional acquisition still applies.
void ThreadSafetyAnalyzer::recordTryAcquireCall(const Expr *Exp,
                                                const NamedDecl *D,
                                                til::SExpr *Self,
                                                TryAcquireCaps *NoExprCaps) {
  assert((Exp || NoExprCaps) && "expression-less call without a caps store");
  TryAcquireCaps &Caps = Exp ? TryAcquireCapsMap[Exp] : *NoExprCaps;
  for (const Attr *At : D->attrs()) {
    const auto *A = dyn_cast<TryAcquireCapabilityAttr>(At);
    if (!A)
      continue;
    bool Success = getTrySuccessValue(D->getASTContext(), A->getSuccessValue());
    CapExprSet &Group =
        Success ? (A->isShared() ? Caps.TruthyShared : Caps.TruthyExclusive)
                : (A->isShared() ? Caps.FalsyShared : Caps.FalsyExclusive);
    CapExprSet AttrCaps;
    getMutexIDs(AttrCaps, A, Exp, D, Self);
    for (const auto &M : AttrCaps)
      Group.push_back_nodup(M);
  }
  reconcileTryAcquireCaps(Caps);
  if (Exp) {
    // Keep the per-function summaries in step wherever a call records --
    // constructions insert in-walk, after the pre-walk pass
    // (recordTryAcquireCalls()): whether any try-acquire exists at all
    // (weak/spent bookkeeping, FactManager::tracksTryAcquires()), and
    // every capability one names, flattened (isTryAcquireCapability()).
    FactMan.setTracksTryAcquires(true);
    Caps.anyCap([&](const CapabilityExpr &Cap) {
      AllTryAcquireCaps.push_back_nodup(Cap);
      return false;
    });
  }
}

/// Populate TryAcquireCapsMap for every try-acquire CallExpr in the
/// function, before the lockset walk: a branch on a stored result can
/// precede the call in block order (a loop-top check `if (ok)` above
/// `ok = mu.TryLock()`), and the terminator decode (decodeTrylockBranch)
/// folds the record into its memoized per-capability resolutions. The
/// variable map's per-statement contexts are complete by now
/// (from traverseCFG), so each call's attributes translate in the call's own
/// post-context by replaying the saved contexts block by block.
/// Constructors are excluded: they record in handleCall, where the
/// constructed-object placeholder is available.
void ThreadSafetyAnalyzer::recordTryAcquireCalls(
    const PostOrderCFGView *SortedGraph) {
  for (const CFGBlock *B : *SortedGraph) {
    const CFGBlockInfo &Info = BlockInfo[B->getBlockID()];
    unsigned CtxIndex = Info.EntryIndex;
    LocalVariableMap::Context Ctx = Info.EntryContext;
    for (const auto &BI : *B) {
      std::optional<CFGStmt> CS = BI.getAs<CFGStmt>();
      if (!CS)
        continue;
      const Stmt *S = CS->getStmt();
      // Advance to the post-context of S; a no-op for statements the
      // variable map saved no context for.
      Ctx = LocalVarMap.getNextContext(CtxIndex, S, Ctx);
      if (const auto *CtorE = dyn_cast<CXXConstructExpr>(S)) {
        // A construction records in handleCall, where the constructed
        // object's placeholder exists; note it here so the per-function
        // summaries are conservative from the first block on.
        if (const CXXConstructorDecl *Ctor = CtorE->getConstructor();
            Ctor && Ctor->hasAttr<TryAcquireCapabilityAttr>()) {
          HasUnrecordedTryAcquire = true;
          FactMan.setTracksTryAcquires(true);
        }
        continue;
      }
      const auto *CE = dyn_cast<CallExpr>(S);
      if (!CE)
        continue;
      const auto *D = dyn_cast_or_null<NamedDecl>(CE->getCalleeDecl());
      if (!D || !D->hasAttr<TryAcquireCapabilityAttr>())
        continue;
      // Mirror BuildLockset's post-context attribute translation.
      if (Handler.issueBetaWarnings())
        SxBuilder.setLookupLocalVarExpr(
            [Ctx, this](const NamedDecl *VD) mutable -> const Expr * {
              return LocalVarMap.lookupExpr(VD, Ctx);
            });
      recordTryAcquireCall(CE, D);
    }
  }
  if (Handler.issueBetaWarnings())
    SxBuilder.setLookupLocalVarExpr(nullptr);
}

/// Record the negative facts reaching a loop's latch as weak evidence in
/// the sealed exit sets its exit edges are computed from: an iteration may
/// have released the capability (or failed to re-acquire it), and the
/// blocks were analyzed before this back edge was seen, so the exit edges
/// would otherwise re-materialize a hold the loop may have released
/// (getEdgeLockset()).
void ThreadSafetyAnalyzer::injectLoopWeakNegatives(
    const CFGBlock *Head, const CFGBlock *Latch,
    PostOrderCFGView::CFGBlockSet &Visited) {
  if (!FactMan.tracksTryAcquires())
    return;
  // A weak negative may coexist with the positive fact it weakens (see
  // intersectAndWarn()), so a positive still in an exit set does not block
  // the evidence -- a spend inside the loop body must reach the loop's
  // exit edges. Only capabilities some try-acquire names can matter to
  // weak facts' consumers.
  SmallVector<FactID, 2> Injectable;
  for (const auto &Fact : BlockInfo[Latch->getBlockID()].ExitSet) {
    const FactEntry &FE = FactMan[Fact];
    if (FE.negative() && isTryAcquireCapability(FE))
      Injectable.push_back(Fact);
  }
  if (Injectable.empty())
    return;

  // The loop's blocks: those reaching the latch backwards without passing
  // the head, bounded by what the head reaches forwards. Without that
  // bound a second entry into the body (a goto into the loop, or an
  // irreducible loop) would let the backward walk escape the loop and
  // absorb arbitrary blocks up to the function's entry.
  llvm::SmallPtrSet<const CFGBlock *, 16> FromHead;
  SmallVector<const CFGBlock *, 16> Work{Head};
  FromHead.insert(Head);
  while (!Work.empty()) {
    const CFGBlock *B = Work.pop_back_val();
    for (CFGBlock::const_succ_iterator SI = B->succ_begin(), SE = B->succ_end();
         SI != SE; ++SI)
      if (*SI && FromHead.insert(*SI).second)
        Work.push_back(*SI);
  }
  llvm::SmallPtrSet<const CFGBlock *, 8> LoopBlocks;
  LoopBlocks.insert(Head);
  LoopBlocks.insert(Latch);
  Work.assign({Latch});
  while (!Work.empty()) {
    const CFGBlock *B = Work.pop_back_val();
    for (CFGBlock::const_pred_iterator PI = B->pred_begin(), PE = B->pred_end();
         PI != PE; ++PI)
      if (*PI && FromHead.contains(*PI) && LoopBlocks.insert(*PI).second)
        Work.push_back(*PI);
  }

  // Any member owning an exit edge needs the evidence (the head, a break,
  // a goto out), and so does a sealed block outside the loop that an exit
  // edge passes through: a break statement's own block does not reach the
  // latch, yet its exit set feeds the post-loop join. An unvisited
  // successor needs nothing -- its entry derives from a patched set later.
  SmallVector<const CFGBlock *, 8> PatchBlocks;
  llvm::SmallPtrSet<const CFGBlock *, 8> SealedOutside;
  SmallVector<const CFGBlock *, 4> OutsideWork;
  auto NoteOutsideSucc = [&](const CFGBlock *S) {
    if (S && !LoopBlocks.contains(S) && Visited.alreadySet(S) &&
        SealedOutside.insert(S).second)
      OutsideWork.push_back(S);
  };
  for (const CFGBlock *B : LoopBlocks) {
    bool HasExitEdge = false;
    for (CFGBlock::const_succ_iterator SI = B->succ_begin(), SE = B->succ_end();
         SI != SE; ++SI) {
      if (*SI && !LoopBlocks.contains(*SI))
        HasExitEdge = true;
      NoteOutsideSucc(*SI);
    }
    if (HasExitEdge)
      PatchBlocks.push_back(B);
  }
  while (!OutsideWork.empty()) {
    const CFGBlock *B = OutsideWork.pop_back_val();
    PatchBlocks.push_back(B);
    for (CFGBlock::const_succ_iterator SI = B->succ_begin(), SE = B->succ_end();
         SI != SE; ++SI)
      NoteOutsideSucc(*SI);
  }

  for (const CFGBlock *B : PatchBlocks) {
    FactSet &Target = BlockInfo[B->getBlockID()].ExitSet;
    for (FactID Fact : Injectable) {
      const FactEntry &FE = FactMan[Fact];
      // A weak negative already in the set absorbs the back edge's spend
      // instead of blocking it: the loop's exit edges resolve results the
      // body may have spent, so the body's spend is the one the promotion
      // veto needs (getEdgeLockset()). A real negative stays as it is --
      // it blocks resurrection on its own.
      if (const FactEntry *Existing = Target.findDefinite(FactMan, FE)) {
        if (Existing->weak() && FE.spentTryLock() &&
            FE.spentTryLock() != Existing->spentTryLock())
          Target.replaceFact(FactMan, *Existing, cloneAsWeak(FE));
        continue;
      }
      keepAsWeak(Target, Fact);
    }
  }
}

/// Check a function's CFG for thread-safety violations.
///
/// We traverse the blocks in the CFG, compute the set of mutexes that are held
/// at the end of each block, and issue warnings for thread safety violations.
/// Each block in the CFG is traversed exactly once.
void ThreadSafetyAnalyzer::runAnalysis(AnalysisDeclContext &AC) {
  // TODO: this whole function needs be rewritten as a visitor for CFGWalker.
  // For now, we just use the walker to set things up.
  threadSafety::CFGWalker walker;
  if (!walker.init(AC))
    return;

  // AC.dumpCFG(true);
  // threadSafety::printSCFG(walker);

  CFG *CFGraph = walker.getGraph();
  const NamedDecl *D = walker.getDecl();
  CurrentFunction = dyn_cast<FunctionDecl>(D);
  ASTCtx = &D->getASTContext();

  if (D->hasAttr<NoThreadSafetyAnalysisAttr>())
    return;

  // FIXME: Do something a bit more intelligent inside constructor and
  // destructor code.  Constructors and destructors must assume unique access
  // to 'this', so checks on member variable access is disabled, but we should
  // still enable checks on other objects.
  if (isa<CXXConstructorDecl>(D))
    return;  // Don't check inside constructors.
  if (isa<CXXDestructorDecl>(D))
    return;  // Don't check inside destructors.

  Handler.enterFunction(CurrentFunction);

  BlockInfo.resize(CFGraph->getNumBlockIDs(),
    CFGBlockInfo::getEmptyBlockInfo(LocalVarMap));

  // We need to explore the CFG via a "topological" ordering.
  // That way, we will be guaranteed to have information about required
  // predecessor locksets when exploring a new block.
  const PostOrderCFGView *SortedGraph = walker.getSortedGraph();
  PostOrderCFGView::CFGBlockSet VisitedBlocks(CFGraph);

  CFGBlockInfo &Initial = BlockInfo[CFGraph->getEntry().getBlockID()];
  CFGBlockInfo &Final   = BlockInfo[CFGraph->getExit().getBlockID()];

  // Mark entry block as reachable
  Initial.Reachable = true;

  // Compute SSA names for local variables
  LocalVarMap.traverseCFG(AC, CFGraph, SortedGraph, BlockInfo);

  // Fill in source locations for all CFGBlocks.
  findBlockLocations(CFGraph, SortedGraph, BlockInfo);

  CapExprSet ExclusiveLocksAcquired;
  CapExprSet SharedLocksAcquired;
  CapExprSet LocksReleased;

  // Add locks from exclusive_locks_required and shared_locks_required
  // to initial lockset. Also turn off checking for lock and unlock functions.
  // FIXME: is there a more intelligent way to check lock/unlock functions?
  if (!SortedGraph->empty()) {
    assert(*SortedGraph->begin() == &CFGraph->getEntry());
    FactSet &InitialLockset = Initial.EntrySet;

    CapExprSet ExclusiveLocksToAdd;
    CapExprSet SharedLocksToAdd;

    SourceLocation Loc = D->getLocation();
    for (const auto *Attr : D->attrs()) {
      Loc = Attr->getLocation();
      if (const auto *A = dyn_cast<RequiresCapabilityAttr>(Attr)) {
        getMutexIDs(A->isShared() ? SharedLocksToAdd : ExclusiveLocksToAdd, A,
                    nullptr, D);
      } else if (const auto *A = dyn_cast<ReleaseCapabilityAttr>(Attr)) {
        // UNLOCK_FUNCTION() is used to hide the underlying lock implementation.
        // We must ignore such methods.
        if (A->args_size() == 0)
          return;
        getMutexIDs(A->isShared() ? SharedLocksToAdd : ExclusiveLocksToAdd, A,
                    nullptr, D);
        getMutexIDs(LocksReleased, A, nullptr, D);
      } else if (const auto *A = dyn_cast<AcquireCapabilityAttr>(Attr)) {
        if (A->args_size() == 0)
          return;
        getMutexIDs(A->isShared() ? SharedLocksAcquired
                                  : ExclusiveLocksAcquired,
                    A, nullptr, D);
      } else if (isa<TryAcquireCapabilityAttr>(Attr)) {
        // Don't try to check trylock functions for now.
        return;
      }
    }
    ArrayRef<ParmVarDecl *> Params;
    if (CurrentFunction)
      Params = CurrentFunction->getCanonicalDecl()->parameters();
    else if (auto CurrentMethod = dyn_cast<ObjCMethodDecl>(D))
      Params = CurrentMethod->getCanonicalDecl()->parameters();
    else
      llvm_unreachable("Unknown function kind");
    for (const ParmVarDecl *Param : Params) {
      if (isCallbackParam(Param))
        continue;
      CapExprSet UnderlyingLocks;
      for (const auto *Attr : Param->attrs()) {
        Loc = Attr->getLocation();
        if (const auto *A = dyn_cast<ReleaseCapabilityAttr>(Attr)) {
          getMutexIDs(A->isShared() ? SharedLocksToAdd : ExclusiveLocksToAdd, A,
                      nullptr, Param);
          getMutexIDs(LocksReleased, A, nullptr, Param);
          getMutexIDs(UnderlyingLocks, A, nullptr, Param);
        } else if (const auto *A = dyn_cast<RequiresCapabilityAttr>(Attr)) {
          getMutexIDs(A->isShared() ? SharedLocksToAdd : ExclusiveLocksToAdd, A,
                      nullptr, Param);
          getMutexIDs(UnderlyingLocks, A, nullptr, Param);
        } else if (const auto *A = dyn_cast<AcquireCapabilityAttr>(Attr)) {
          getMutexIDs(A->isShared() ? SharedLocksAcquired
                                    : ExclusiveLocksAcquired,
                      A, nullptr, Param);
          getMutexIDs(UnderlyingLocks, A, nullptr, Param);
        } else if (const auto *A = dyn_cast<LocksExcludedAttr>(Attr)) {
          getMutexIDs(UnderlyingLocks, A, nullptr, Param);
        }
      }
      if (UnderlyingLocks.empty())
        continue;
      CapabilityExpr Cp(SxBuilder.translateVariable(Param, nullptr),
                        StringRef(),
                        /*Neg=*/false, /*Reentrant=*/false);
      auto *ScopedEntry = FactMan.createFact<ScopedLockableFactEntry>(
          Cp, Param->getLocation(), FactEntry::Declared,
          UnderlyingLocks.size());
      for (const CapabilityExpr &M : UnderlyingLocks)
        ScopedEntry->addLock(M);
      addLock(InitialLockset, ScopedEntry, true);
    }

    // FIXME -- Loc can be wrong here.
    for (const auto &Mu : ExclusiveLocksToAdd) {
      const auto *Entry = FactMan.createFact<LockableFactEntry>(
          Mu, LK_Exclusive, Loc, FactEntry::Declared);
      addLock(InitialLockset, Entry, true);
    }
    for (const auto &Mu : SharedLocksToAdd) {
      const auto *Entry = FactMan.createFact<LockableFactEntry>(
          Mu, LK_Shared, Loc, FactEntry::Declared);
      addLock(InitialLockset, Entry, true);
    }
  }

  // Record the capabilities of every try-acquire call, recorded in the exact
  // context of that call.
  recordTryAcquireCalls(SortedGraph);

  // Compute the expected exit set.
  // By default, we expect all locks held on entry to be held on exit.
  FactSet ExpectedFunctionExitSet = Initial.EntrySet;

  // Adjust the expected exit set by adding or removing locks, as declared
  // by *-LOCK_FUNCTION and UNLOCK_FUNCTION.  The intersect below will then
  // issue the appropriate warning.
  // FIXME: the location here is not quite right.
  for (const auto &Lock : ExclusiveLocksAcquired)
    ExpectedFunctionExitSet.addLock(
        FactMan, FactMan.createFact<LockableFactEntry>(Lock, LK_Exclusive,
                                                       D->getLocation()));
  for (const auto &Lock : SharedLocksAcquired)
    ExpectedFunctionExitSet.addLock(
        FactMan, FactMan.createFact<LockableFactEntry>(Lock, LK_Shared,
                                                       D->getLocation()));
  for (const auto &Lock : LocksReleased)
    ExpectedFunctionExitSet.removeDefinite(FactMan, Lock);

  for (const auto *CurrBlock : *SortedGraph) {
    unsigned CurrBlockID = CurrBlock->getBlockID();
    CFGBlockInfo *CurrBlockInfo = &BlockInfo[CurrBlockID];

    // Use the default initial lockset in case there are no predecessors.
    VisitedBlocks.insert(CurrBlock);

    // Iterate through the predecessor blocks and warn if the lockset for all
    // predecessors is not the same. We take the entry lockset of the current
    // block to be the intersection of all previous locksets.
    // FIXME: By keeping the intersection, we may output more errors in future
    // for a lock which is not in the intersection, but was in the union. We
    // may want to also keep the union in future. As an example, let's say
    // the intersection contains Mutex L, and the union contains L and M.
    // Later we unlock M. At this point, we would output an error because we
    // never locked M; although the real error is probably that we forgot to
    // lock M on all code paths. Conversely, let's say that later we lock M.
    // In this case, we should compare against the intersection instead of the
    // union because the real error is probably that we forgot to unlock M on
    // all code paths.
    bool LocksetInitialized = false;
    // The try-acquire call whose result the condition starting at this
    // block branches on, if any. Computed lazily on the first join where a
    // set carries a try-acquire fact at all. Each incoming set is scanned
    // once as it arrives (JoinHasTryLockFact accumulates); the entry set
    // itself never gains try-acquire facts from anywhere else.
    const CallExpr *RebranchTryLock = nullptr;
    const CallExpr *RebranchTryLock2 = nullptr;
    bool RebranchTryLockComputed = false;
    bool RebranchResolvesAllPaths = true;
    bool JoinHasTryLockFact = false;
    auto HasTryLockFact = [this](const FactSet &FS) {
      // TryAcquireCapsMap is empty in functions without try-acquires (the
      // common case): skip scanning the fact sets entirely.
      return FactMan.tracksTryAcquires() && llvm::any_of(FS, [this](FactID ID) {
               return FactMan[ID].tryLockCall();
             });
    };
    // The lockset of the first infeasible incoming edge, if any (see below).
    std::optional<FactSet> InfeasibleEdgeSet;
    for (CFGBlock::const_pred_iterator PI = CurrBlock->pred_begin(),
         PE  = CurrBlock->pred_end(); PI != PE; ++PI) {
      // if *PI -> CurrBlock is a back edge
      if (*PI == nullptr || !VisitedBlocks.alreadySet(*PI))
        continue;

      unsigned PrevBlockID = (*PI)->getBlockID();
      CFGBlockInfo *PrevBlockInfo = &BlockInfo[PrevBlockID];

      // Ignore edges from blocks that can't return.
      if (neverReturns(*PI) || !PrevBlockInfo->Reachable)
        continue;

      FactSet PrevLockset;
      if (getEdgeLockset(PrevLockset, PrevBlockInfo->ExitSet, *PI, CurrBlock) ||
          PrevBlockInfo->CoverageOnly) {
        // The edge cannot be taken (a promoted fact proves the branched-on
        // try-acquire succeeded), or the predecessor itself was analyzed
        // only for coverage and its exit set is dead state either way: skip
        // the edge at the join like an unreachable predecessor. Remember
        // the lockset in case no live predecessor remains: infeasibility
        // only prunes joins, never analysis coverage (see below).
        if (!InfeasibleEdgeSet)
          InfeasibleEdgeSet = std::move(PrevLockset);
        continue;
      }

      // Okay, we can reach this block from the entry.
      CurrBlockInfo->Reachable = true;

      if (!LocksetInitialized) {
        CurrBlockInfo->EntrySet = PrevLockset;
        JoinHasTryLockFact = HasTryLockFact(PrevLockset);
        LocksetInitialized = true;
      } else {
        // Surprisingly 'continue' doesn't always produce back edges, because
        // the CFG has empty "transition" blocks where they meet with the end
        // of the regular loop body. We still want to diagnose them as loop.
        if (isa_and_nonnull<ContinueStmt>((*PI)->getTerminatorStmt())) {
          // Loop join: warn on locks held for only some iterations.
          intersectAndWarn(CurrBlockInfo->EntrySet, PrevLockset,
                           CurrBlockInfo->EntryLoc,
                           LEK_LockedSomeLoopIterations,
                           LEK_LockedSomeLoopIterations, nullptr,
                           /*RebranchResolvesAllPaths=*/true,
                           /*RebranchTryLock2=*/nullptr,
                           /*CheckedAroundLoop=*/nullptr,
                           /*ForwardJoin=*/true);
        } else {
          // Branch join: a difference in the facts created by a try-acquire
          // is demoted to try-held and re-resolved on the outgoing edges if
          // the condition branches on that call's result -- possibly behind
          // short-circuit blocks of a compound condition like `c && ok`.
          if (!RebranchTryLockComputed && !JoinHasTryLockFact)
            JoinHasTryLockFact = HasTryLockFact(PrevLockset);
          if (!RebranchTryLockComputed && JoinHasTryLockFact) {
            // Compute once; the result depends only on CurrBlock, not on
            // *PI. Skipped entirely (the common case) until some fact at
            // this join originates from a try-acquire.
            RebranchTryLock = getConditionTrylockCallExpr(
                CurrBlock, &RebranchResolvesAllPaths, &RebranchTryLock2);
            RebranchTryLockComputed = true;
          }
          intersectAndWarn(CurrBlockInfo->EntrySet, PrevLockset,
                           CurrBlockInfo->EntryLoc, LEK_LockedSomePredecessors,
                           LEK_LockedSomePredecessors, RebranchTryLock,
                           RebranchResolvesAllPaths, RebranchTryLock2,
                           /*CheckedAroundLoop=*/nullptr,
                           /*ForwardJoin=*/true);
        }
      }
    }

    // A block reached only through infeasible edges is dynamically dead if
    // the infeasibility proofs are right -- but the proof rests on the
    // local-variable map, which can be stale (e.g. a result variable
    // mutated through an escaped reference), and even genuinely dead code
    // gets its diagnostics. So analyze the block anyway, with one of the
    // infeasible edges' locksets: infeasibility prunes joins, never
    // analysis coverage. The block is marked coverage-only, which
    // quarantines its exit set from downstream joins (above) and
    // propagates through blocks reachable only from it.
    if (!CurrBlockInfo->Reachable && InfeasibleEdgeSet) {
      CurrBlockInfo->Reachable = true;
      CurrBlockInfo->CoverageOnly = true;
      CurrBlockInfo->EntrySet = std::move(*InfeasibleEdgeSet);
    }

    // Skip rest of block if it's not reachable.
    if (!CurrBlockInfo->Reachable)
      continue;

    BuildLockset LocksetBuilder(this, *CurrBlockInfo, ExpectedFunctionExitSet);

    // Visit all the statements in the basic block.
    for (const auto &BI : *CurrBlock) {
      switch (BI.getKind()) {
        case CFGElement::Statement: {
          CFGStmt CS = BI.castAs<CFGStmt>();
          LocksetBuilder.Visit(CS.getStmt());
          break;
        }
        // Ignore BaseDtor and MemberDtor for now.
        case CFGElement::AutomaticObjectDtor: {
          CFGAutomaticObjDtor AD = BI.castAs<CFGAutomaticObjDtor>();
          const auto *DD = AD.getDestructorDecl(AC.getASTContext());
          // Function parameters as they are constructed in caller's context and
          // the CFG does not contain the ctors. Ignore them as their
          // capabilities cannot be analysed because of this missing
          // information.
          if (isa_and_nonnull<ParmVarDecl>(AD.getVarDecl()))
            break;
          if (!DD || !DD->hasAttrs())
            break;

          LocksetBuilder.handleCall(
              nullptr, DD,
              SxBuilder.translateVariable(AD.getVarDecl(), nullptr),
              AD.getTriggerStmt()->getEndLoc());
          break;
        }

        case CFGElement::CleanupFunction: {
          const CFGCleanupFunction &CF = BI.castAs<CFGCleanupFunction>();
          LocksetBuilder.handleCall(
              /*Exp=*/nullptr, CF.getFunctionDecl(),
              SxBuilder.translateVariable(CF.getVarDecl(), nullptr),
              CF.getVarDecl()->getLocation());
          break;
        }

        case CFGElement::TemporaryDtor: {
          auto TD = BI.castAs<CFGTemporaryDtor>();

          // Clean up constructed object even if there are no attributes to
          // keep the number of objects in limbo as small as possible.
          if (auto Object = ConstructedObjects.find(
                  TD.getBindTemporaryExpr()->getSubExpr());
              Object != ConstructedObjects.end()) {
            const auto *DD = TD.getDestructorDecl(AC.getASTContext());
            if (DD->hasAttrs())
              // TODO: the location here isn't quite correct.
              LocksetBuilder.handleCall(nullptr, DD, Object->second,
                                        TD.getBindTemporaryExpr()->getEndLoc());
            ConstructedObjects.erase(Object);
          }
          break;
        }
        default:
          break;
      }
    }
    CurrBlockInfo->ExitSet = LocksetBuilder.FSet;

    // A block analyzed only for coverage stops here: its exit set is
    // provably dead state, so back-edge comparisons must not consume it
    // either (the predecessor loop above keeps it out of forward joins).
    if (CurrBlockInfo->CoverageOnly)
      continue;

    // For every back edge from CurrBlock (the end of the loop) to another block
    // (FirstLoopBlock) we need to check that the Lockset of Block is equal to
    // the one held at the beginning of FirstLoopBlock. We can look up the
    // Lockset held at the beginning of FirstLoopBlock in the EntryLockSets map.
    for (CFGBlock::const_succ_iterator SI = CurrBlock->succ_begin(),
         SE  = CurrBlock->succ_end(); SI != SE; ++SI) {
      // if CurrBlock -> *SI is *not* a back edge
      if (*SI == nullptr || !VisitedBlocks.alreadySet(*SI))
        continue;

      CFGBlock *FirstLoopBlock = *SI;
      CFGBlockInfo *PreLoop = &BlockInfo[FirstLoopBlock->getBlockID()];
      CFGBlockInfo *LoopEnd = &BlockInfo[CurrBlockID];
      // A back-edge difference in the facts created by a try-acquire is
      // forgiven when the loop condition branches on that call's result
      // (e.g. a spin loop storing the result), possibly behind
      // short-circuit blocks of a compound condition: the condition's
      // outgoing edges re-resolve the fact, so it does not leak around the
      // loop. Skipped entirely while the function has no try-acquire facts.
      const Expr *RebranchTryLock =
          !TryAcquireCapsMap.empty()
              ? getConditionTrylockCallExpr(FirstLoopBlock)
              : nullptr;
      // For the unchecked-result warning: the try-acquire results branched
      // on inside this back edge's natural loop are (or will be, on the
      // next iteration) checked around the loop. Results checked only
      // outside the loop are not: the loop re-executes the call (or
      // discards the result) unchecked.
      llvm::SmallPtrSet<const Expr *, 4> CheckedInLoop;
      const llvm::SmallPtrSetImpl<const Expr *> *CheckedInLoopPtr = nullptr;
      if (Handler.issueBetaWarnings() && HasTryLockFact(LoopEnd->ExitSet)) {
        // The natural loop of this back edge: the head, plus every block
        // reaching this latch without passing through the head. (All these
        // blocks precede the latch in the traversal, so their exit contexts
        // are available for the decode below; on an irreducible CFG the
        // walk may escape the loop, erring toward suppression.)
        llvm::SmallPtrSet<const CFGBlock *, 8> LoopBlocks;
        SmallVector<const CFGBlock *, 8> Worklist;
        LoopBlocks.insert(FirstLoopBlock);
        if (LoopBlocks.insert(CurrBlock).second)
          Worklist.push_back(CurrBlock);
        while (!Worklist.empty()) {
          const CFGBlock *B = Worklist.pop_back_val();
          for (CFGBlock::const_pred_iterator BPI = B->pred_begin(),
                                             BPE = B->pred_end();
               BPI != BPE; ++BPI)
            if (*BPI && LoopBlocks.insert(*BPI).second)
              Worklist.push_back(*BPI);
        }
        // Decode each loop block's terminator now, rather than consulting
        // what happened to be decoded already: a goto-rotated loop's latch
        // terminator has not had its forward edges processed yet, and its
        // check must still count. (The decode is memoized, so blocks whose
        // edges were already processed cost a cache hit.)
        for (const CFGBlock *B : LoopBlocks) {
          TerminatorTrylockCall Checked = getTerminatorTrylockCall(B);
          if (Checked.TrylockCall)
            CheckedInLoop.insert(Checked.TrylockCall);
          // A branch on a merge of two identical calls checks both results.
          if (Checked.TrylockCall2)
            CheckedInLoop.insert(Checked.TrylockCall2);
        }
        CheckedInLoopPtr = &CheckedInLoop;
      }
      // At a loop join the entry set keeps the (weaker) pre-loop facts and
      // the loop condition re-resolves the result each iteration, so the
      // exemption stands even behind a short-circuit.
      intersectAndWarn(PreLoop->EntrySet, LoopEnd->ExitSet, PreLoop->EntryLoc,
                       LEK_LockedSomeLoopIterations,
                       LEK_LockedSomeLoopIterations, RebranchTryLock,
                       /*RebranchResolvesAllPaths=*/true,
                       /*RebranchTryLock2=*/nullptr, CheckedInLoopPtr);
      // A negative fact reaching the loop head on its back edge is
      // evidence that an iteration may have released the capability (or
      // failed to re-acquire it): patch it into the sealed exit sets the
      // loop's exit edges are computed from.
      injectLoopWeakNegatives(FirstLoopBlock, CurrBlock, VisitedBlocks);
    }
  }

  // Skip the final check if the exit block is unreachable, or reachable
  // only through infeasible edges: its exit set is dead state (the
  // coverage diagnostics inside the dead blocks have already run).
  if (!Final.Reachable || Final.CoverageOnly)
    return;

  // FIXME: Should we call this function for all blocks which exit the function?
  intersectAndWarn(ExpectedFunctionExitSet, Final.ExitSet, Final.ExitLoc,
                   LEK_LockedAtEndOfFunction, LEK_NotLockedAtEndOfFunction);

  Handler.leaveFunction(CurrentFunction);
}

/// Check a function's CFG for thread-safety violations.
///
/// We traverse the blocks in the CFG, compute the set of mutexes that are held
/// at the end of each block, and issue warnings for thread safety violations.
/// Each block in the CFG is traversed exactly once.
void threadSafety::runThreadSafetyAnalysis(AnalysisDeclContext &AC,
                                           ThreadSafetyHandler &Handler,
                                           BeforeSet **BSet) {
  if (!*BSet)
    *BSet = new BeforeSet;
  ThreadSafetyAnalyzer Analyzer(Handler, *BSet);
  Analyzer.runAnalysis(AC);
}

void threadSafety::threadSafetyCleanup(BeforeSet *Cache) { delete Cache; }

/// Helper function that returns a LockKind required for the given level
/// of access.
LockKind threadSafety::getLockKindFromAccessKind(AccessKind AK) {
  switch (AK) {
    case AK_Read :
      return LK_Shared;
    case AK_Written :
      return LK_Exclusive;
  }
  llvm_unreachable("Unknown AccessKind");
}
