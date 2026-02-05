//===- MemSliceAnalysis.h - Analyze pointer uses -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines the MemSliceAnalysis infrastructure for analyzing
/// how memory is accessed by visiting uses and collecting them with their
/// offsets.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MEMSLICEANALYSIS_H
#define LLVM_ANALYSIS_MEMSLICEANALYSIS_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Use.h"

namespace llvm {

class AllocaInst;
class AtomicCmpXchgInst;
class AtomicRMWInst;
class CallInst;
class DataLayout;
class IntrinsicInst;
class LoadInst;
class StoreInst;
class Type;
class Value;

/// A pointer use with its offset.
struct PtrUse {
  Use *U;
  APInt Offset;
  bool IsOffsetKnown;
};

/// Visitor for collecting pointer uses.
///
/// This class visits the uses of a base ptr value and collects each use
/// along with its computed offset.
class PtrUseCollector {
public:
  /// Construct the use collector for a particular alloca.
  PtrUseCollector(const DataLayout &DL, AllocaInst &AI);

  /// Access the collected uses.
  ArrayRef<PtrUse> getUses() const { return Uses; }

private:
  class Builder;

  /// The collected uses with their offsets.
  SmallVector<PtrUse, 16> Uses;
};

Type *mergeTypes(Type *Base, uint64_t Size, Type *Add, uint64_t AddSize,
                 uint64_t AddOffset);
Type *noImpliedCallTypes(CallInst *CI, const Use &Use);

/// Process pointer uses, checking for loads and stores.
///
/// This utility function iterates through collected pointer uses and
/// identifies load and store instructions with associated type information.
///
/// Precondition: all Uses are sorted by offset.
Type *
processPtrType(LLVMContext &C, ArrayRef<PtrUse> Uses, uint64_t Size,
               const DataLayout &DL,
               function_ref<Type *(Type *Base, uint64_t Size, Type *Add,
                                   uint64_t AddSize, uint64_t AddOffset)>
                   MergeTypes,
               function_ref<Type *(CallInst *, const Use &)> GetImpliedType);

} // end namespace llvm

#endif // LLVM_ANALYSIS_MEMSLICEANALYSIS_H
