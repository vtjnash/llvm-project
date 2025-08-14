//===----------- TaskDispatchTest.cpp - Test TaskDispatch APIs ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TaskDispatch.h"
#include "llvm/Config/llvm-config.h" // for LLVM_ENABLE_THREADS
#include "gtest/gtest.h"

#include <future>

using namespace llvm;
using namespace llvm::orc;

TEST(InPlaceTaskDispatchTest, GenericNamedTask) {
  auto D = std::make_unique<InPlaceTaskDispatcher>();
  orc::future<void> F;
  D->dispatch(
      makeGenericNamedTask([B = F.get_promise(*D)]() { B.set_value(); }));
  EXPECT_TRUE(F.valid());
  EXPECT_FALSE(F.ready());
  F.get();
  EXPECT_FALSE(F.valid());
  EXPECT_TRUE(F.ready());
  D->shutdown();
}

#if LLVM_ENABLE_THREADS
TEST(DynamicThreadPoolDispatchTest, GenericNamedTask) {
  auto D = std::make_unique<DynamicThreadPoolTaskDispatcher>(std::nullopt);
  orc::future<void> F;
  D->dispatch(
      makeGenericNamedTask([B = F.get_promise(*D)]() { B.set_value(); }));
  EXPECT_TRUE(F.valid());
  EXPECT_FALSE(F.ready());
  F.get();
  EXPECT_FALSE(F.valid());
  EXPECT_TRUE(F.ready());
  D->shutdown();
}
#endif
