//===- MemSliceAnalysis.cpp - Analyze pointer uses --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the MemSliceAnalysis infrastructure for analyzing
/// how a pointer is accessed by visiting and collecting uses.
///
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MemSliceAnalysis.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/PtrUseVisitor.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "ptr-use-collector"

namespace llvm {

/// Builder for collecting pointer uses.
///
/// This class collects uses of a base ptr value by recursively visiting uses
/// and recording each use along with its computed offset.
class PtrUseCollector::Builder
    : public PtrUseVisitor<PtrUseCollector::Builder> {
  friend class PtrUseVisitor<PtrUseCollector::Builder>;
  friend class InstVisitor<PtrUseCollector::Builder>;

  using Base = PtrUseVisitor<PtrUseCollector::Builder>;

  PtrUseCollector &Collector;

public:
  Builder(const DataLayout &DL, AllocaInst &AI, PtrUseCollector &Collector)
      : PtrUseVisitor<PtrUseCollector::Builder>(DL), Collector(Collector) {}

private:
  void recordUse() { Collector.Uses.push_back({U, Offset, IsOffsetKnown}); }

  void visitPHINode(PHINode &PN) {
    enqueueUsers(PN);
    Base::visitPHINode(PN);
  }

  void visitSelectInst(SelectInst &SI) {
    // If the condition being selected on is a constant, fold the select. Yes
    // this does (rarely) happen early on.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(SI.getCondition()))
      if (SI.getOperandUse(1 + CI->isZero()) != *U)
        return;
    enqueueUsers(SI);
    Base::visitSelectInst(SI);
  }

  void visitStoreInst(StoreInst &SI) {
    // Only record if this is the pointer operand, not the value operand
    if (SI.getValueOperand() == *U)
      return; // Pointer is being stored, not stored to
    recordUse();
  }

  void visitAtomicCmpXchgInst(AtomicCmpXchgInst &AI) {
    // Only record if this is the pointer operand, not the value operand
    if (AI.getNewValOperand() == *U || AI.getCompareOperand() == *U)
      return; // Pointer is being stored, not stored to
    recordUse();
  }

  void visitAtomicRMWInst(AtomicRMWInst &AI) {
    // Only record if this is the pointer operand, not the value operand
    if (AI.getValOperand() == *U)
      return; // Pointer is being stored, not stored to
    recordUse();
  }

  void visitInstruction(Instruction &I) { recordUse(); }

  void visitCallBase(CallBase &CB) { Base::visitCallBase(CB); }
};

PtrUseCollector::PtrUseCollector(const DataLayout &DL, AllocaInst &AI) {
  Builder PB(DL, AI, *this);
  Builder::PtrInfo PtrI = PB.visitPtr(AI);
  if (PtrI.isAborted()) {
    Uses.clear();
    return;
  }

  // Sort the uses: unknown offsets first, then by offset.
  llvm::stable_sort(Uses, [](const PtrUse &LHS, const PtrUse &RHS) {
    if (LHS.IsOffsetKnown != RHS.IsOffsetKnown)
      return !LHS.IsOffsetKnown; // Unknown offsets first
    if (!LHS.IsOffsetKnown)
      return false; // Both unknown, maintain stable order
    return LHS.Offset.ult(RHS.Offset);
  });
}

// A very basic merging algorithm to pick a common access type, which just takes
// the larger input, or returns [i8 x n].
// This could be smarter about splitting up Base or Add, particularly if they
// are integer types (or other easy-to-bitcast type).
Type *mergeTypes(Type *Base, uint64_t Size, Type *Add, uint64_t AddSize,
                 uint64_t AddOffset) {
  if (Size >= AddSize + AddOffset)
    return Base;
  else if (AddOffset == 0 && AddSize > Size)
    return Add;
  else
    return ArrayType::get(Type::getInt8Ty(Base->getContext()),
                          std::max(Size, AddSize + AddOffset));
}

// Callback to add target-specific hooks for deriving information from
// particular call arguments.
Type *noImpliedCallTypes(CallInst *CI, const Use &Use) { return nullptr; }

Type *
processPtrType(LLVMContext &C, ArrayRef<PtrUse> Uses, uint64_t Size,
               const DataLayout &DL,
               function_ref<Type *(Type *Base, uint64_t Size, Type *Add,
                                   uint64_t AddSize, uint64_t AddOffset)>
                   MergeTypes,
               function_ref<Type *(CallInst *, const Use &)> GetImpliedType) {
  SmallVector<Type *> ElemTypes;
  Type *Slice = nullptr;
  uint64_t LastEnd = 0;
  uint64_t SliceOffset = 0;
  uint64_t SliceSize = 0;
  for (const PtrUse &Use : Uses) {
    Instruction *UserInst = cast<Instruction>(Use.U->getUser());
    // Get the implied element type for this offset
    Type *ET = nullptr;
    if (auto *LI = dyn_cast<LoadInst>(UserInst)) {
      ET = LI->getType();
    } else if (auto *SI = dyn_cast<StoreInst>(UserInst)) {
      ET = SI->getValueOperand()->getType();
    } else if (auto *AI = dyn_cast<AtomicCmpXchgInst>(UserInst)) {
      ET = AI->getNewValOperand()->getType();
    } else if (auto *AI = dyn_cast<AtomicRMWInst>(UserInst)) {
      ET = AI->getValOperand()->getType();
    } else if (auto *CI = dyn_cast<CallInst>(UserInst)) {
      ET = GetImpliedType(CI, *Use.U);
    }
    if (!ET)
      continue;
    if (Use.IsOffsetKnown) {
      uint64_t Offset = Use.Offset.getLimitedValue();
      // Check if we're starting a new slice
      if (Offset >= SliceOffset + SliceSize) {
        // Finalize the previous slice, adding padding if needed
        if (Slice) {
          if (SliceOffset != LastEnd)
            ElemTypes.push_back(
                ArrayType::get(Type::getInt8Ty(C), SliceOffset - LastEnd));
          ElemTypes.push_back(Slice);
          LastEnd = SliceOffset + SliceSize;
        }
        SliceOffset = Offset;
        SliceSize = 0;
        Slice = nullptr;
      }

      if (!Slice)
        Slice = ET;
      else if (ET != Slice)
        Slice = MergeTypes(Slice, SliceSize, ET,
                        DL.getTypeAllocSize(ET).getKnownMinValue(),
                        Offset - SliceOffset);
      SliceSize =
          std::max(SliceSize, DL.getTypeAllocSize(Slice).getKnownMinValue());
    } else {
      // TODO: we should actually mergeTypes all of these too.
      // Then if any region is unknown (padding) after merging,
      // this is probably what it should be assigned (making an aligned array).
    }
  }
  if (Slice) {
    // Finalize the last slice, adding padding if needed
    if (SliceOffset != LastEnd)
      ElemTypes.push_back(
          ArrayType::get(Type::getInt8Ty(C), SliceOffset - LastEnd));
    ElemTypes.push_back(Slice);
    LastEnd = SliceOffset + SliceSize;
  }
  // Add trailing padding
  if (Size != LastEnd)
    ElemTypes.push_back(ArrayType::get(Type::getInt8Ty(C), Size - LastEnd));
  if (ElemTypes.empty())
    return ArrayType::get(Type::getInt8Ty(C), Size);
  if (ElemTypes.size() == 1)
    return ElemTypes[0];
  // TODO: compute if all ElemTypes ==
  bool AllSame = false;
  if (AllSame)
    return ArrayType::get(ElemTypes[0], ElemTypes.size());
  // TODO: compute if all ElemTypes are at their expected offsets, and if so,
  // erase_if there is any unnecessary padding
  bool IsPacked = true;
  return StructType::get(C, ElemTypes, IsPacked);
}

} // end namespace llvm
