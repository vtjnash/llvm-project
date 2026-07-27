//===--- AttrImpl.cpp - Classes for representing attributes -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file contains out-of-line methods for Attr classes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTStructuralEquivalence.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include <optional>
#include <type_traits>
using namespace clang;

void LoopHintAttr::printPrettyPragma(raw_ostream &OS,
                                     const PrintingPolicy &Policy) const {
  unsigned SpellingIndex = getAttributeSpellingListIndex();
  // For "#pragma unroll" and "#pragma nounroll" the string "unroll" or
  // "nounroll" is already emitted as the pragma name.
  if (SpellingIndex == Pragma_nounroll ||
      SpellingIndex == Pragma_nounroll_and_jam)
    return;
  else if (SpellingIndex == Pragma_unroll ||
           SpellingIndex == Pragma_unroll_and_jam) {
    OS << ' ' << getValueString(Policy);
    return;
  }

  assert(SpellingIndex == Pragma_clang_loop && "Unexpected spelling");
  OS << ' ' << getOptionName(option) << getValueString(Policy);
}

// Return a string containing the loop hint argument including the
// enclosing parentheses.
std::string LoopHintAttr::getValueString(const PrintingPolicy &Policy) const {
  std::string ValueName;
  llvm::raw_string_ostream OS(ValueName);
  OS << "(";
  if (state == Numeric)
    value->printPretty(OS, nullptr, Policy);
  else if (state == FixedWidth || state == ScalableWidth) {
    if (value) {
      value->printPretty(OS, nullptr, Policy);
      if (state == ScalableWidth)
        OS << ", scalable";
    } else if (state == ScalableWidth)
      OS << "scalable";
    else
      OS << "fixed";
  } else if (state == Enable)
    OS << "enable";
  else if (state == Full)
    OS << "full";
  else if (state == AssumeSafety)
    OS << "assume_safety";
  else
    OS << "disable";
  OS << ")";
  return ValueName;
}

// Return a string suitable for identifying this attribute in diagnostics.
std::string
LoopHintAttr::getDiagnosticName(const PrintingPolicy &Policy) const {
  unsigned SpellingIndex = getAttributeSpellingListIndex();
  if (SpellingIndex == Pragma_nounroll)
    return "#pragma nounroll";
  else if (SpellingIndex == Pragma_unroll)
    return "#pragma unroll" +
           (option == UnrollCount ? getValueString(Policy) : "");
  else if (SpellingIndex == Pragma_nounroll_and_jam)
    return "#pragma nounroll_and_jam";
  else if (SpellingIndex == Pragma_unroll_and_jam)
    return "#pragma unroll_and_jam" +
           (option == UnrollAndJamCount ? getValueString(Policy) : "");

  assert(SpellingIndex == Pragma_clang_loop && "Unexpected spelling");
  return getOptionName(option) + getValueString(Policy);
}

void OMPDeclareSimdDeclAttr::printPrettyPragma(
    raw_ostream &OS, const PrintingPolicy &Policy) const {
  if (getBranchState() != BS_Undefined)
    OS << ' ' << ConvertBranchStateTyToStr(getBranchState());
  if (auto *E = getSimdlen()) {
    OS << " simdlen(";
    E->printPretty(OS, nullptr, Policy);
    OS << ")";
  }
  if (uniforms_size() > 0) {
    OS << " uniform";
    StringRef Sep = "(";
    for (auto *E : uniforms()) {
      OS << Sep;
      E->printPretty(OS, nullptr, Policy);
      Sep = ", ";
    }
    OS << ")";
  }
  alignments_iterator NI = alignments_begin();
  for (auto *E : aligneds()) {
    OS << " aligned(";
    E->printPretty(OS, nullptr, Policy);
    if (*NI) {
      OS << ": ";
      (*NI)->printPretty(OS, nullptr, Policy);
    }
    OS << ")";
    ++NI;
  }
  steps_iterator I = steps_begin();
  modifiers_iterator MI = modifiers_begin();
  for (auto *E : linears()) {
    OS << " linear(";
    if (*MI != OMPC_LINEAR_unknown)
      OS << getOpenMPSimpleClauseTypeName(llvm::omp::Clause::OMPC_linear, *MI)
         << "(";
    E->printPretty(OS, nullptr, Policy);
    if (*MI != OMPC_LINEAR_unknown)
      OS << ")";
    if (*I) {
      OS << ": ";
      (*I)->printPretty(OS, nullptr, Policy);
    }
    OS << ")";
    ++I;
    ++MI;
  }
}

void OMPDeclareTargetDeclAttr::printPrettyPragma(
    raw_ostream &OS, const PrintingPolicy &Policy) const {
  // Use fake syntax because it is for testing and debugging purpose only.
  if (getDevType() != DT_Any)
    OS << " device_type(" << ConvertDevTypeTyToStr(getDevType()) << ")";
  if (getMapType() != MT_To && getMapType() != MT_Enter)
    OS << ' ' << ConvertMapTypeTyToStr(getMapType());
  if (Expr *E = getIndirectExpr()) {
    OS << " indirect(";
    E->printPretty(OS, nullptr, Policy);
    OS << ")";
  } else if (getIndirect()) {
    OS << " indirect";
  }
}

std::optional<OMPDeclareTargetDeclAttr *>
OMPDeclareTargetDeclAttr::getActiveAttr(const ValueDecl *VD) {
  if (llvm::all_of(VD->redecls(), [](const Decl *D) { return !D->hasAttrs(); }))
    return std::nullopt;
  unsigned Level = 0;
  OMPDeclareTargetDeclAttr *FoundAttr = nullptr;
  for (const Decl *D : VD->redecls()) {
    for (auto *Attr : D->specific_attrs<OMPDeclareTargetDeclAttr>()) {
      if (Level <= Attr->getLevel()) {
        Level = Attr->getLevel();
        FoundAttr = Attr;
      }
    }
  }
  if (FoundAttr)
    return FoundAttr;
  return std::nullopt;
}

std::optional<OMPDeclareTargetDeclAttr::MapTypeTy>
OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(const ValueDecl *VD) {
  std::optional<OMPDeclareTargetDeclAttr *> ActiveAttr = getActiveAttr(VD);
  if (ActiveAttr)
    return (*ActiveAttr)->getMapType();
  return std::nullopt;
}

std::optional<OMPDeclareTargetDeclAttr::DevTypeTy>
OMPDeclareTargetDeclAttr::getDeviceType(const ValueDecl *VD) {
  std::optional<OMPDeclareTargetDeclAttr *> ActiveAttr = getActiveAttr(VD);
  if (ActiveAttr)
    return (*ActiveAttr)->getDevType();
  return std::nullopt;
}

std::optional<SourceLocation>
OMPDeclareTargetDeclAttr::getLocation(const ValueDecl *VD) {
  std::optional<OMPDeclareTargetDeclAttr *> ActiveAttr = getActiveAttr(VD);
  if (ActiveAttr)
    return (*ActiveAttr)->getRange().getBegin();
  return std::nullopt;
}

namespace clang {
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const OMPTraitInfo &TI);
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const OMPTraitInfo *TI);
}

void OMPDeclareVariantAttr::printPrettyPragma(
    raw_ostream &OS, const PrintingPolicy &Policy) const {
  if (const Expr *E = getVariantFuncRef()) {
    OS << "(";
    E->printPretty(OS, nullptr, Policy);
    OS << ")";
  }
  OS << " match(" << traitInfos << ")";

  auto PrintExprs = [&OS, &Policy](Expr **Begin, Expr **End) {
    for (Expr **I = Begin; I != End; ++I) {
      assert(*I && "Expected non-null Stmt");
      if (I != Begin)
        OS << ",";
      (*I)->printPretty(OS, nullptr, Policy);
    }
  };
  if (adjustArgsNothing_size()) {
    OS << " adjust_args(nothing:";
    PrintExprs(adjustArgsNothing_begin(), adjustArgsNothing_end());
    OS << ")";
  }
  if (adjustArgsNeedDevicePtr_size()) {
    OS << " adjust_args(need_device_ptr:";
    PrintExprs(adjustArgsNeedDevicePtr_begin(), adjustArgsNeedDevicePtr_end());
    OS << ")";
  }
  if (adjustArgsNeedDeviceAddr_size()) {
    OS << " adjust_args(need_device_addr:";
    PrintExprs(adjustArgsNeedDeviceAddr_begin(),
               adjustArgsNeedDeviceAddr_end());
    OS << ")";
  }

  auto PrintInteropInfo = [&OS](OMPInteropInfo *Begin, OMPInteropInfo *End) {
    for (OMPInteropInfo *I = Begin; I != End; ++I) {
      if (I != Begin)
        OS << ", ";
      OS << "interop(";
      OS << getInteropTypeString(I);
      OS << ")";
    }
  };
  if (appendArgs_size()) {
    OS << " append_args(";
    PrintInteropInfo(appendArgs_begin(), appendArgs_end());
    OS << ")";
  }
}

unsigned AlignedAttr::getAlignment(ASTContext &Ctx) const {
  assert(!isAlignmentDependent());
  if (getCachedAlignmentValue())
    return *getCachedAlignmentValue();

  // Handle alignmentType case.
  if (!isAlignmentExpr()) {
    QualType T = getAlignmentType()->getType();

    // C++ [expr.alignof]p3:
    //     When alignof is applied to a reference type, the result is the
    //     alignment of the referenced type.
    T = T.getNonReferenceType();

    if (T.getQualifiers().hasUnaligned())
      return Ctx.getCharWidth();

    return Ctx.getTypeAlignInChars(T.getTypePtr()).getQuantity() *
           Ctx.getCharWidth();
  }

  // Handle alignmentExpr case.
  if (alignmentExpr)
    return alignmentExpr->EvaluateKnownConstInt(Ctx).getZExtValue() *
           Ctx.getCharWidth();

  return Ctx.getTargetDefaultAlignForAttributeAligned();
}

StringLiteral *FormatMatchesAttr::getFormatString() const {
  return cast<StringLiteral>(getExpectedFormat());
}

//===----------------------------------------------------------------------===//
// Thread-safety capability attributes
//
// These attributes can be carried by a function type as well as by a
// declaration (see FunctionType::FunctionTypeExtraAttributeInfo), so the
// notions of "which capability" and "same requirement" they need are shared
// between the type and the declaration machinery and live here.
//===----------------------------------------------------------------------===//

bool clang::isCapabilityAttr(const Attr *A) {
  switch (A->getKind()) {
  case attr::RequiresCapability:
  case attr::AcquireCapability:
  case attr::ReleaseCapability:
  case attr::TryAcquireCapability:
  case attr::AssertCapability:
  case attr::LocksExcluded:
    return true;
  default:
    return false;
  }
}

/// The mutex-expression arguments of a thread-safety capability attribute.
/// These identify which capabilities the attribute refers to and are what
/// distinguishes two otherwise-identical function types.
ArrayRef<const Expr *> clang::getCapabilityAttrArgs(const Attr *A) {
  auto Args = [](const auto *CA) -> ArrayRef<Expr *> {
    return ArrayRef<Expr *>(CA->args_begin(), CA->args_size());
  };
  switch (A->getKind()) {
  case attr::RequiresCapability:
    return Args(cast<RequiresCapabilityAttr>(A));
  case attr::AcquireCapability:
    return Args(cast<AcquireCapabilityAttr>(A));
  case attr::ReleaseCapability:
    return Args(cast<ReleaseCapabilityAttr>(A));
  case attr::TryAcquireCapability:
    return Args(cast<TryAcquireCapabilityAttr>(A));
  case attr::AssertCapability:
    return Args(cast<AssertCapabilityAttr>(A));
  case attr::LocksExcluded:
    return Args(cast<LocksExcludedAttr>(A));
  default:
    return {};
  }
}

/// Sharedness and genericness are encoded in the attribute's spelling rather
/// than in its arguments, so every consumer that distinguishes capability
/// requirements has to account for them separately. Do not use the spelling
/// index itself: differently spelled synonyms (e.g. exclusive_locks_required
/// and requires_capability) state the same requirement.
unsigned clang::getCapabilityAttrSemantics(const Attr *A) {
  switch (A->getKind()) {
  case attr::RequiresCapability:
    return cast<RequiresCapabilityAttr>(A)->isShared();
  case attr::AcquireCapability:
    return cast<AcquireCapabilityAttr>(A)->isShared();
  case attr::AssertCapability:
    return cast<AssertCapabilityAttr>(A)->isShared();
  case attr::TryAcquireCapability:
    return cast<TryAcquireCapabilityAttr>(A)->isShared();
  case attr::ReleaseCapability: {
    const auto *RA = cast<ReleaseCapabilityAttr>(A);
    return unsigned(RA->isShared()) | (unsigned(RA->isGeneric()) << 1);
  }
  default:
    // LocksExcluded, and any future kind, carry no extra semantic state.
    return 0;
  }
}

const Expr *clang::getCapabilityAttrSuccessValue(const Attr *A) {
  if (const auto *TA = dyn_cast<TryAcquireCapabilityAttr>(A))
    return TA->getSuccessValue();
  return nullptr;
}

/// The narrowest signed representation of \p V. Normalizing this way is what
/// lets two success values that denote the same value compare equal even
/// though their expressions have different types and widths: 'true' evaluates
/// to a one-bit unsigned 1, '1' to a 32-bit signed 1, and both come out of
/// here as a two-bit signed 1.
static llvm::APSInt normalizeSuccessValue(const llvm::APSInt &V) {
  llvm::APSInt Signed =
      V.isSigned() ? V
                   : llvm::APSInt(V.zext(V.getBitWidth() + 1),
                                  /*isUnsigned=*/false);
  return Signed.trunc(std::max(Signed.getSignificantBits(), 1u));
}

std::optional<llvm::APSInt>
clang::getCapabilityAttrSuccessValueAsInt(const Expr *E) {
  if (!E)
    return std::nullopt;
  E = E->IgnoreParenImpCasts();
  if (const auto *CE = dyn_cast<ConstantExpr>(E);
      CE && CE->hasAPValueResult() && CE->getAPValueResult().isInt())
    return normalizeSuccessValue(CE->getAPValueResult().getInt());
  if (const auto *BL = dyn_cast<CXXBoolLiteralExpr>(E))
    return normalizeSuccessValue(llvm::APSInt::get(BL->getValue()));
  if (const auto *IL = dyn_cast<IntegerLiteral>(E))
    return normalizeSuccessValue(llvm::APSInt(
        IL->getValue(), IL->getType()->isUnsignedIntegerOrEnumerationType()));
  return std::nullopt;
}

/// Add the try-acquire success value \p E to \p ID by the value it denotes,
/// so that two spellings of one value profile the same. An expression whose
/// value cannot be told (see getCapabilityAttrSuccessValueAsInt) falls back to
/// its syntactic form; \p ProfileExpr contributes a leading 0 or 1 for that
/// case, and the 2 here keeps an evaluated profile from ever colliding with a
/// syntactic one.
static void profileCapabilityAttrSuccessValue(
    llvm::FoldingSetNodeID &ID, const Expr *E,
    llvm::function_ref<void(const Expr *)> ProfileExpr) {
  if (std::optional<llvm::APSInt> Val = getCapabilityAttrSuccessValueAsInt(E)) {
    ID.AddInteger(2);
    Val->Profile(ID);
    return;
  }
  ProfileExpr(E);
}

/// Add everything that makes \p A a distinct capability requirement -- its
/// kind, the sharedness and genericness encoded in its spelling, try-acquire's
/// success value, and its capability arguments -- to \p ID. Two attributes
/// with the same profile state the same requirement.
void clang::profileCapabilityAttr(llvm::FoldingSetNodeID &ID, const Attr *A,
                                  const ASTContext &Context) {
  // A capability attribute argument may be null (e.g. after an error), so a
  // sentinel keeps a missing argument distinguishable from a present one.
  auto ProfileExpr = [&](const Expr *E) {
    ID.AddInteger(E != nullptr);
    if (E)
      E->Profile(ID, Context, /*Canonical=*/true);
  };

  ID.AddInteger(A->getKind());

  // The spelling-encoded semantics and try-acquire's success value are not
  // reported by getCapabilityAttrArgs, so they have to be profiled here or
  // semantically different function types collide in the folding set.
  ID.AddInteger(getCapabilityAttrSemantics(A));
  profileCapabilityAttrSuccessValue(ID, getCapabilityAttrSuccessValue(A),
                                    ProfileExpr);

  // The argument count separates one attribute's argument stream from the
  // next one's.
  ArrayRef<const Expr *> Args = getCapabilityAttrArgs(A);
  ID.AddInteger(Args.size());
  for (const Expr *E : Args)
    ProfileExpr(E);
}

bool clang::areEquivalentCapabilityAttrs(const Attr *A, const Attr *B,
                                         const ASTContext &Context) {
  if (A == B)
    return true;
  if (A->getKind() != B->getKind())
    return false;
  llvm::FoldingSetNodeID IDA, IDB;
  profileCapabilityAttr(IDA, A, Context);
  profileCapabilityAttr(IDB, B, Context);
  return IDA == IDB;
}

ArrayRef<const Attr *> clang::getCapabilityAttrsOfFunctionType(QualType T) {
  if (T.isNull())
    return {};
  QualType Fn = T;
  if (const auto *PT = T->getAs<PointerType>())
    Fn = PT->getPointeeType();
  else if (const auto *BT = T->getAs<BlockPointerType>())
    Fn = BT->getPointeeType();
  else if (const auto *RT = T->getAs<ReferenceType>())
    Fn = RT->getPointeeType();
  if (const auto *FPT = Fn->getAs<FunctionProtoType>())
    return FPT->getCapabilityAttrs();
  return {};
}

/// Whether \p Set already states the requirement \p A states. The sets are
/// tiny (one attribute per written annotation), so a linear scan is fine.
static bool containsEquivalentCapabilityAttr(ArrayRef<const Attr *> Set,
                                             const Attr *A,
                                             const ASTContext &Context) {
  return llvm::any_of(Set, [&](const Attr *B) {
    return areEquivalentCapabilityAttrs(A, B, Context);
  });
}

bool clang::areEquivalentCapabilityAttrSets(ArrayRef<const Attr *> LHS,
                                            ArrayRef<const Attr *> RHS,
                                            const ASTContext &Context) {
  if (LHS.data() == RHS.data() && LHS.size() == RHS.size())
    return true;
  auto IsSubset = [&](ArrayRef<const Attr *> A, ArrayRef<const Attr *> B) {
    return llvm::all_of(A, [&](const Attr *X) {
      return containsEquivalentCapabilityAttr(B, X, Context);
    });
  };
  return IsSubset(LHS, RHS) && IsSubset(RHS, LHS);
}

ArrayRef<const Attr *> clang::mergeCapabilityAttrs(ArrayRef<const Attr *> LHS,
                                                   ArrayRef<const Attr *> RHS,
                                                   const ASTContext &Context,
                                                   bool IsIntersection) {
  // The overwhelmingly common cases: nothing to merge, or one side says
  // everything the other does. Returning an operand's own array keeps the
  // merged type identical to that operand's, so callers can reuse it.
  if (LHS.empty() || RHS.empty())
    return IsIntersection ? ArrayRef<const Attr *>()
                          : (LHS.empty() ? RHS : LHS);
  if (areEquivalentCapabilityAttrSets(LHS, RHS, Context))
    return LHS;

  SmallVector<const Attr *, 4> Merged;
  auto Add = [&](const Attr *A) {
    if (!containsEquivalentCapabilityAttr(Merged, A, Context))
      Merged.push_back(A);
  };
  for (const Attr *A : LHS)
    if (!IsIntersection || containsEquivalentCapabilityAttr(RHS, A, Context))
      Add(A);
  if (!IsIntersection)
    for (const Attr *A : RHS)
      Add(A);

  if (Merged.empty())
    return {};
  if (areEquivalentCapabilityAttrSets(Merged, LHS, Context))
    return LHS;
  if (areEquivalentCapabilityAttrSets(Merged, RHS, Context))
    return RHS;

  const Attr **Storage = Context.Allocate<const Attr *>(Merged.size());
  llvm::copy(Merged, Storage);
  return ArrayRef<const Attr *>(Storage, Merged.size());
}

namespace {
// Arguments whose types fail this test never compare equal unless there's a
// specialization of equalAttrArgs for the type. Specilization for the following
// arguments haven't been implemented yet:
//  - DeclArgument
//  - OMPTraitInfoArgument
//  - VariadicOMPInteropInfoArgument
#define USE_DEFAULT_EQUALITY                                                   \
  (std::is_same_v<T, StringRef> || std::is_same_v<T, VersionTuple> ||          \
   std::is_same_v<T, IdentifierInfo *> || std::is_same_v<T, char *> ||         \
   std::is_enum_v<T> || std::is_integral_v<T>)

template <class T>
typename std::enable_if_t<!USE_DEFAULT_EQUALITY, bool>
equalAttrArgs(T A, T B, StructuralEquivalenceContext &Context) {
  return false;
}

template <class T>
typename std::enable_if_t<USE_DEFAULT_EQUALITY, bool>
equalAttrArgs(T A1, T A2, StructuralEquivalenceContext &Context) {
  return A1 == A2;
}

template <>
bool equalAttrArgs<ParamIdx>(ParamIdx P1, ParamIdx P2,
                             StructuralEquivalenceContext &) {
  // ParamIdx can be invalid when representing an optional parameter that was
  // not specified (e.g. the second argument of alloc_size(N)).
  // ParamIdx::operator== asserts both sides are valid, so guard against the
  // invalid case before delegating to it.
  if (P1.isValid() != P2.isValid())
    return false;
  if (!P1.isValid())
    return true;
  return P1 == P2;
}

template <class T>
bool equalAttrArgs(T *A1_B, T *A1_E, T *A2_B, T *A2_E,
                   StructuralEquivalenceContext &Context) {
  if (A1_E - A1_B != A2_E - A2_B)
    return false;

  for (; A1_B != A1_E; ++A1_B, ++A2_B)
    if (!equalAttrArgs(*A1_B, *A2_B, Context))
      return false;

  return true;
}

template <>
bool equalAttrArgs<Attr *>(Attr *A1, Attr *A2,
                           StructuralEquivalenceContext &Context) {
  if (!A1 || !A2)
    return A1 == A2;
  return A1->isEquivalent(*A2, Context);
}

template <>
bool equalAttrArgs<Expr *>(Expr *A1, Expr *A2,
                           StructuralEquivalenceContext &Context) {
  return ASTStructuralEquivalence::isEquivalent(Context, A1, A2);
}

template <>
bool equalAttrArgs<QualType>(QualType T1, QualType T2,
                             StructuralEquivalenceContext &Context) {
  return ASTStructuralEquivalence::isEquivalent(Context, T1, T2);
}

template <>
bool equalAttrArgs<const IdentifierInfo *>(
    const IdentifierInfo *Name1, const IdentifierInfo *Name2,
    StructuralEquivalenceContext &Context) {
  return ASTStructuralEquivalence::isEquivalent(Name1, Name2);
}

bool areAlignedAttrsEqual(const AlignedAttr &A1, const AlignedAttr &A2,
                          StructuralEquivalenceContext &Context) {
  if (A1.getSpelling() != A2.getSpelling())
    return false;

  if (A1.isAlignmentExpr() != A2.isAlignmentExpr())
    return false;

  if (A1.isAlignmentExpr())
    return equalAttrArgs(A1.getAlignmentExpr(), A2.getAlignmentExpr(), Context);

  return equalAttrArgs(A1.getAlignmentType()->getType(),
                       A2.getAlignmentType()->getType(), Context);
}
} // namespace

namespace {
// Machinery to unique attributes based on the arguments.
// The construction mirrors the equivalent testing code above.
// The content of the arguments are added to the FoldingSetNodeID instance,
// which allows the AttributedTypes to unique the attributes based on
// the value of the arguments.

#define USE_DEFAULT_PROFILE                                                    \
  (std::is_same_v<T, StringRef> || std::is_same_v<T, VersionTuple> ||          \
   std::is_same_v<T, IdentifierInfo *> ||                                      \
   std::is_same_v<T, const IdentifierInfo *> || std::is_enum_v<T> ||           \
   std::is_integral_v<T>)

template <class T>
typename std::enable_if_t<!USE_DEFAULT_PROFILE>
profileAttrArg(llvm::FoldingSetNodeID &, const ASTContext &, T) {
  llvm_unreachable("profile not implemented for this type");
}

template <class T>
typename std::enable_if_t<USE_DEFAULT_PROFILE>
profileAttrArg(llvm::FoldingSetNodeID &ID, const ASTContext &, T V) {
  if constexpr (std::is_same_v<T, StringRef>)
    ID.AddString(V);
  else if constexpr (std::is_same_v<T, VersionTuple>) {
    ID.AddInteger(V.getMajor());
    ID.AddInteger(V.getMinor().value_or(0));
    ID.AddInteger(V.getSubminor().value_or(0));
    ID.AddInteger(V.getBuild().value_or(0));
  } else if constexpr (std::is_same_v<T, IdentifierInfo *> ||
                       std::is_same_v<T, const IdentifierInfo *>)
    ID.AddPointer(V);
  else
    ID.AddInteger(static_cast<long long>(V));
}

template <>
inline void profileAttrArg<ParamIdx>(llvm::FoldingSetNodeID &ID,
                                     const ASTContext &, ParamIdx P) {
  ID.AddBoolean(P.isValid());
  if (P.isValid())
    ID.AddInteger(P.getASTIndex());
}

template <class T>
inline void profileAttrArg(llvm::FoldingSetNodeID &ID, const ASTContext &Ctx,
                           T *Begin, T *End) {
  ID.AddInteger(End - Begin);
  for (; Begin != End; ++Begin)
    profileAttrArg(ID, Ctx, *Begin);
}

template <>
inline void profileAttrArg<Attr *>(llvm::FoldingSetNodeID &ID,
                                   const ASTContext &Ctx, Attr *A) {
  if (!A) {
    ID.AddPointer(nullptr);
    return;
  }
  ID.AddInteger(A->getKind());
  A->Profile(ID, Ctx);
}

template <>
inline void profileAttrArg<Expr *>(llvm::FoldingSetNodeID &ID,
                                   const ASTContext &Ctx, Expr *E) {
  E->Profile(ID, Ctx, /*Canonical=*/true);
}

template <>
inline void profileAttrArg<QualType>(llvm::FoldingSetNodeID &ID,
                                     const ASTContext &, QualType T) {
  ID.AddPointer(T.getCanonicalType().getAsOpaquePtr());
}

void profileAlignedAttr(const AlignedAttr &A, llvm::FoldingSetNodeID &ID,
                        const ASTContext &Ctx) {
  ID.AddInteger(A.getSpellingListIndex());
  ID.AddBoolean(A.isAlignmentExpr());
  if (A.isAlignmentExpr())
    profileAttrArg(ID, Ctx, A.getAlignmentExpr());
  else
    profileAttrArg(ID, Ctx, A.getAlignmentType()->getType());
}
} // namespace

#include "clang/AST/AttrImpl.inc"
