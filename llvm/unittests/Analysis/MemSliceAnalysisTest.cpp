//===--- MemSliceAnalysisTest.cpp - MemSliceAnalysis unit tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MemSliceAnalysis.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class ProcessPtrType : public testing::Test {
protected:
  LLVMContext Context;
  std::unique_ptr<Module> M;
  Function *F;
  BasicBlock *BB;

  void SetUp() override {
    M = std::make_unique<Module>("test", Context);
    FunctionType *FTy = FunctionType::get(Type::getVoidTy(Context), false);
    F = Function::Create(FTy, Function::ExternalLinkage, "test", M.get());
    BB = BasicBlock::Create(Context, "entry", F);
  }

  Type *getInt8Ty() { return Type::getInt8Ty(Context); }
  Type *getInt32Ty() { return Type::getInt32Ty(Context); }
  Type *getInt64Ty() { return Type::getInt64Ty(Context); }
  PointerType *getPtrTy() { return PointerType::getUnqual(Context); }
};

TEST_F(ProcessPtrType, SimpleLoadStore) {
  IRBuilder<> Builder(BB);
  Value *Ptr = Builder.CreateAlloca(getInt64Ty());
  LoadInst *LI = new LoadInst(getInt32Ty(), Ptr, "load", BB);
  StoreInst *SI = new StoreInst(ConstantInt::get(getInt32Ty(), 42), Ptr, BB);
  SmallVector<PtrUse> Uses;
  Uses.push_back({&LI->getOperandUse(0), APInt(64, 0), true});
  Uses.push_back({&SI->getOperandUse(1), APInt(64, 0), true});
  const DataLayout &DL = M->getDataLayout();
  Type *ResultTy =
      processPtrType(Context, Uses, 8, DL, mergeTypes, noImpliedCallTypes);
  ASSERT_EQ(ResultTy, getInt32Ty());
}

TEST_F(ProcessPtrType, MultipleOffsetsCreateStruct) {
  IRBuilder<> Builder(BB);
  Value *Ptr = Builder.CreateAlloca(ArrayType::get(getInt8Ty(), 16));
  LoadInst *L0 = new LoadInst(getInt32Ty(), Ptr, "l0", BB);
  Value *GEP4 = Builder.CreateConstGEP1_64(getInt8Ty(), Ptr, 4);
  LoadInst *L4 = new LoadInst(getInt32Ty(), GEP4, "l4", BB);
  Value *GEP8 = Builder.CreateConstGEP1_64(getInt8Ty(), Ptr, 8);
  LoadInst *L8 = new LoadInst(getInt32Ty(), GEP8, "l8", BB);
  SmallVector<PtrUse> Uses;
  Uses.push_back({&L0->getOperandUse(0), APInt(64, 0), true});
  Uses.push_back({&L4->getOperandUse(0), APInt(64, 4), true});
  Uses.push_back({&L8->getOperandUse(0), APInt(64, 8), true});
  const DataLayout &DL = M->getDataLayout();
  Type *ResultTy =
      processPtrType(Context, Uses, 12, DL, mergeTypes, noImpliedCallTypes);
  EXPECT_EQ(ResultTy, ArrayType::get(getInt32Ty(), 3));
  Type *ResultTy2 =
      processPtrType(Context, Uses, 16, DL, mergeTypes, noImpliedCallTypes);
  EXPECT_EQ(ResultTy, StructType::get(getInt32Ty(), getInt32Ty(), getInt32Ty(),
                                      ArrayType::get(getInt8Ty(), 4)));
}

TEST_F(ProcessPtrType, AtomicInstructions) {
  IRBuilder<> Builder(BB);
  Value *Ptr = Builder.CreateAlloca(getInt64Ty());
  AtomicRMWInst *RMW = new AtomicRMWInst(
      AtomicRMWInst::Add, Ptr, ConstantInt::get(getInt64Ty(), 1), Align(1),
      AtomicOrdering::SequentiallyConsistent, SyncScope::System, BB);
  AtomicCmpXchgInst *CmpXchg = new AtomicCmpXchgInst(
      Ptr, ConstantInt::get(getInt64Ty(), 0), ConstantInt::get(getInt64Ty(), 1),
      Align(1), AtomicOrdering::SequentiallyConsistent,
      AtomicOrdering::SequentiallyConsistent, SyncScope::System, BB);
  SmallVector<PtrUse> Uses;
  Uses.push_back({&RMW->getOperandUse(0), APInt(64, 0), true});
  Uses.push_back({&CmpXchg->getOperandUse(0), APInt(64, 0), true});
  const DataLayout &DL = M->getDataLayout();
  Type *ResultTy =
      processPtrType(Context, Uses, 8, DL, mergeTypes, noImpliedCallTypes);
  ASSERT_EQ(ResultTy, getInt64Ty());
}

TEST_F(ProcessPtrType, GetImpliedTypeCallback) {
  IRBuilder<> Builder(BB);
  Value *Ptr = Builder.CreateAlloca(getInt64Ty());
  FunctionType *CalleeTy =
      FunctionType::get(Type::getVoidTy(Context), {getPtrTy()}, false);
  Function *Callee =
      Function::Create(CalleeTy, Function::ExternalLinkage, "callee", M.get());
  CallInst *CI = CallInst::Create(CalleeTy, Callee, {Ptr}, "", BB);
  SmallVector<PtrUse> Uses;
  Uses.push_back({&CI->getOperandUse(0), APInt(64, 0), true});
  const DataLayout &DL = M->getDataLayout();
  bool ImpliedTypeCalled = false;
  auto GetImpliedType = [&](CallInst *Call, const Use &U) -> Type * {
    ImpliedTypeCalled = true;
    EXPECT_EQ(Call, CI);
    return getInt32Ty();
  };
  Type *ResultTy =
      processPtrType(Context, Uses, 4, DL, mergeTypes, GetImpliedType);
  ASSERT_EQ(ResultTy, getInt32Ty());
  EXPECT_TRUE(ImpliedTypeCalled);
}

} // anonymous namespace
