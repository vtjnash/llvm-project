//===------ DylibManager.h - Manage dylibs in the executor ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// APIs for managing real (non-JIT) dylibs in the executing process.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_DYLIBMANAGER_H
#define LLVM_EXECUTIONENGINE_ORC_DYLIBMANAGER_H

#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/ExecutionEngine/Orc/TaskDispatch.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"
#include <mutex>
#include <vector>

namespace llvm::orc {

class SymbolLookupSet;

class LLVM_ABI DylibManager : public ExecutorProcessControl {
public:
  DylibManager(std::shared_ptr<SymbolStringPool> SSP,
               std::unique_ptr<TaskDispatcher> D)
      : ExecutorProcessControl(std::move(SSP), std::move(D)) {}

  /// A pair of a dylib and a set of symbols to be looked up.
  struct LookupRequest {
    LookupRequest(tpctypes::DylibHandle Handle, const SymbolLookupSet &Symbols)
        : Handle(Handle), Symbols(Symbols) {}
    tpctypes::DylibHandle Handle;
    const SymbolLookupSet &Symbols;
  };

  virtual ~DylibManager();

  /// Load the dynamic library at the given path and return a handle to it.
  /// If LibraryPath is null this function will return the global handle for
  /// the target process.
  virtual Expected<tpctypes::DylibHandle> loadDylib(const char *DylibPath) = 0;

  /// Search for symbols in the target process.
  ///
  /// The result of the lookup is a 2-dimensional array of target addresses
  /// that correspond to the lookup order. If a required symbol is not
  /// found then this method will return an error. If a weakly referenced
  /// symbol is not found then it be assigned a '0' value.
  Expected<std::vector<tpctypes::LookupResult>>
  lookupSymbols(ArrayRef<LookupRequest> Request) {
    orc::promise<MSVCPExpected<std::vector<tpctypes::LookupResult>>> RP;
    auto RF = RP.get_future();
    lookupSymbolsAsync(Request,
                       [&RP](auto Result) { RP.set_value(std::move(Result)); });
    return RF.get(getDispatcher());
  }

  using SymbolLookupCompleteFn =
      unique_function<void(Expected<std::vector<tpctypes::LookupResult>>)>;

  /// Search for symbols in the target process.
  ///
  /// The result of the lookup is a 2-dimensional array of target addresses
  /// that correspond to the lookup order. If a required symbol is not
  /// found then this method will return an error. If a weakly referenced
  /// symbol is not found then it be assigned a '0' value.
  virtual void lookupSymbolsAsync(ArrayRef<LookupRequest> Request,
                                  SymbolLookupCompleteFn F) = 0;
};

/// A ExecutorProcessControl instance that asserts if any of its methods are
/// used. Suitable for use is unit tests, and by ORC clients who haven't moved
/// to ExecutorProcessControl-based APIs yet.
class UnsupportedExecutorProcessControl : public DylibManager,
                                          private InProcessMemoryAccess {
public:
  UnsupportedExecutorProcessControl(
      std::shared_ptr<SymbolStringPool> SSP = nullptr,
      std::unique_ptr<TaskDispatcher> D = nullptr, const std::string &TT = "",
      unsigned PageSize = 0)
      : DylibManager(
            SSP ? std::move(SSP) : std::make_shared<SymbolStringPool>(),
            D ? std::move(D) : std::make_unique<InPlaceTaskDispatcher>()),
        InProcessMemoryAccess(*this, Triple(TT).isArch64Bit()) {
    this->TargetTriple = Triple(TT);
    this->PageSize = PageSize;
    this->MemAccess = this;
  }

  Expected<int32_t> runAsMain(ExecutorAddr MainFnAddr,
                              ArrayRef<std::string> Args) override {
    llvm_unreachable("Unsupported");
  }

  Expected<int32_t> runAsVoidFunction(ExecutorAddr VoidFnAddr) override {
    llvm_unreachable("Unsupported");
  }

  Expected<int32_t> runAsIntFunction(ExecutorAddr IntFnAddr, int Arg) override {
    llvm_unreachable("Unsupported");
  }

  void callWrapperAsync(ExecutorAddr WrapperFnAddr,
                        IncomingWFRHandler OnComplete,
                        ArrayRef<char> ArgBuffer) override {
    llvm_unreachable("Unsupported");
  }

  Error disconnect() override { return Error::success(); }

  Expected<tpctypes::DylibHandle> loadDylib(const char *DylibPath) override {
    return make_error<StringError>("Unsupported", inconvertibleErrorCode());
  }

  void lookupSymbolsAsync(ArrayRef<LookupRequest> Request,
                          SymbolLookupCompleteFn F) override {
    F(make_error<StringError>("Unsupported", inconvertibleErrorCode()));
  }
};


/// A ExecutorProcessControl implementation targeting the current process.
class LLVM_ABI SelfExecutorProcessControl : public DylibManager,
                                            private InProcessMemoryAccess {
public:
  SelfExecutorProcessControl(
      std::shared_ptr<SymbolStringPool> SSP, std::unique_ptr<TaskDispatcher> D,
      Triple TargetTriple, unsigned PageSize,
      std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr);

  /// Create a SelfExecutorProcessControl with the given symbol string pool and
  /// memory manager.
  /// If no symbol string pool is given then one will be created.
  /// If no memory manager is given a jitlink::InProcessMemoryManager will
  /// be created and used by default.
  static Expected<std::unique_ptr<SelfExecutorProcessControl>>
  Create(std::shared_ptr<SymbolStringPool> SSP = nullptr,
         std::unique_ptr<TaskDispatcher> D = nullptr,
         std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr = nullptr);

  Expected<int32_t> runAsMain(ExecutorAddr MainFnAddr,
                              ArrayRef<std::string> Args) override;

  Expected<int32_t> runAsVoidFunction(ExecutorAddr VoidFnAddr) override;

  Expected<int32_t> runAsIntFunction(ExecutorAddr IntFnAddr, int Arg) override;

  void callWrapperAsync(ExecutorAddr WrapperFnAddr,
                        IncomingWFRHandler OnComplete,
                        ArrayRef<char> ArgBuffer) override;

  Error disconnect() override;

private:
  static shared::CWrapperFunctionResult
  jitDispatchViaWrapperFunctionManager(void *Ctx, const void *FnTag,
                                       const char *Data, size_t Size);

  Expected<tpctypes::DylibHandle> loadDylib(const char *DylibPath) override;

  void lookupSymbolsAsync(ArrayRef<LookupRequest> Request,
                          SymbolLookupCompleteFn F) override;

  std::unique_ptr<jitlink::JITLinkMemoryManager> OwnedMemMgr;
#ifdef __APPLE__
  std::unique_ptr<UnwindInfoManager> UnwindInfoMgr;
#endif // __APPLE__
  char GlobalManglingPrefix = 0;
};

} // end namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_DYLIBMANAGER_H
