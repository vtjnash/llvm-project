//===------- MemoryAccess.h - Executor memory access APIs -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for accessing memory in the executor processes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_MEMORYACCESS_H
#define LLVM_EXECUTIONENGINE_ORC_MEMORYACCESS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/TaskDispatch.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"

#include <future>

namespace llvm::orc {

class TaskDispatcher;

/// APIs for manipulating memory in the target process.
class LLVM_ABI MemoryAccess {
public:
  /// Callback function for asynchronous writes.
  using WriteResultFn = unique_function<void(Error)>;

  template <typename T> using ReadUIntsResult = std::vector<T>;
  template <typename T>
  using OnReadUIntsCompleteFn =
      unique_function<void(Expected<ReadUIntsResult<T>>)>;

  using ReadPointersResult = std::vector<ExecutorAddr>;
  using OnReadPointersCompleteFn =
      unique_function<void(Expected<ReadPointersResult>)>;

  using ReadBuffersResult = std::vector<std::vector<uint8_t>>;
  using OnReadBuffersCompleteFn =
      unique_function<void(Expected<ReadBuffersResult>)>;

  using ReadStringsResult = std::vector<std::string>;
  using OnReadStringsCompleteFn =
      unique_function<void(Expected<ReadStringsResult>)>;

  MemoryAccess(ExecutorProcessControl &EPC) : EPC(EPC) {}
  virtual ~MemoryAccess();

  virtual void writeUInt8sAsync(ArrayRef<tpctypes::UInt8Write> Ws,
                                WriteResultFn OnWriteComplete) = 0;

  virtual void writeUInt16sAsync(ArrayRef<tpctypes::UInt16Write> Ws,
                                 WriteResultFn OnWriteComplete) = 0;

  virtual void writeUInt32sAsync(ArrayRef<tpctypes::UInt32Write> Ws,
                                 WriteResultFn OnWriteComplete) = 0;

  virtual void writeUInt64sAsync(ArrayRef<tpctypes::UInt64Write> Ws,
                                 WriteResultFn OnWriteComplete) = 0;

  virtual void writePointersAsync(ArrayRef<tpctypes::PointerWrite> Ws,
                                  WriteResultFn OnWriteComplete) = 0;

  virtual void writeBuffersAsync(ArrayRef<tpctypes::BufferWrite> Ws,
                                 WriteResultFn OnWriteComplete) = 0;

  virtual void readUInt8sAsync(ArrayRef<ExecutorAddr> Rs,
                               OnReadUIntsCompleteFn<uint8_t> OnComplete) = 0;

  virtual void readUInt16sAsync(ArrayRef<ExecutorAddr> Rs,
                                OnReadUIntsCompleteFn<uint16_t> OnComplete) = 0;

  virtual void readUInt32sAsync(ArrayRef<ExecutorAddr> Rs,
                                OnReadUIntsCompleteFn<uint32_t> OnComplete) = 0;

  virtual void readUInt64sAsync(ArrayRef<ExecutorAddr> Rs,
                                OnReadUIntsCompleteFn<uint64_t> OnComplete) = 0;

  virtual void readPointersAsync(ArrayRef<ExecutorAddr> Rs,
                                 OnReadPointersCompleteFn OnComplete) = 0;

  virtual void readBuffersAsync(ArrayRef<ExecutorAddrRange> Rs,
                                OnReadBuffersCompleteFn OnComplete) = 0;

  virtual void readStringsAsync(ArrayRef<ExecutorAddr> Rs,
                                OnReadStringsCompleteFn OnComplete) = 0;

  Error writeUInt8s(ArrayRef<tpctypes::UInt8Write> Ws) {
    orc::promise<MSVCPError> ResultP;
    auto ResultF = ResultP.get_future();
    writeUInt8sAsync(Ws, [&](Error Err) { ResultP.set_value(std::move(Err)); });
    return ResultF.get(EPC.getDispatcher());
  }

  Error writeUInt16s(ArrayRef<tpctypes::UInt16Write> Ws) {
    orc::promise<MSVCPError> ResultP;
    auto ResultF = ResultP.get_future();
    writeUInt16sAsync(Ws,
                      [&](Error Err) { ResultP.set_value(std::move(Err)); });
    return ResultF.get(EPC.getDispatcher());
  }

  Error writeUInt32s(ArrayRef<tpctypes::UInt32Write> Ws) {
    orc::promise<MSVCPError> ResultP;
    auto ResultF = ResultP.get_future();
    writeUInt32sAsync(Ws,
                      [&](Error Err) { ResultP.set_value(std::move(Err)); });
    return ResultF.get(EPC.getDispatcher());
  }

  Error writeUInt64s(ArrayRef<tpctypes::UInt64Write> Ws) {
    orc::promise<MSVCPError> ResultP;
    auto ResultF = ResultP.get_future();
    writeUInt64sAsync(Ws,
                      [&](Error Err) { ResultP.set_value(std::move(Err)); });
    return ResultF.get(EPC.getDispatcher());
  }

  Error writePointers(ArrayRef<tpctypes::PointerWrite> Ws) {
    orc::promise<MSVCPError> ResultP;
    auto ResultF = ResultP.get_future();
    writePointersAsync(Ws,
                       [&](Error Err) { ResultP.set_value(std::move(Err)); });
    return ResultF.get(EPC.getDispatcher());
  }

  Error writeBuffers(ArrayRef<tpctypes::BufferWrite> Ws) {
    orc::promise<MSVCPError> ResultP;
    auto ResultF = ResultP.get_future();
    writeBuffersAsync(Ws,
                      [&](Error Err) { ResultP.set_value(std::move(Err)); });
    return ResultF.get(EPC.getDispatcher());
  }

  Expected<ReadUIntsResult<uint8_t>> readUInt8s(ArrayRef<ExecutorAddr> Rs) {
    orc::promise<MSVCPExpected<ReadUIntsResult<uint8_t>>> P;
    auto F = P.get_future();
    readUInt8sAsync(Rs, [&](Expected<ReadUIntsResult<uint8_t>> Result) {
      P.set_value(std::move(Result));
    });
    return F.get(EPC.getDispatcher());
  }

  Expected<ReadUIntsResult<uint16_t>> readUInt16s(ArrayRef<ExecutorAddr> Rs) {
    orc::promise<MSVCPExpected<ReadUIntsResult<uint16_t>>> P;
    auto F = P.get_future();
    readUInt16sAsync(Rs, [&](Expected<ReadUIntsResult<uint16_t>> Result) {
      P.set_value(std::move(Result));
    });
    return F.get(EPC.getDispatcher());
  }

  Expected<ReadUIntsResult<uint32_t>> readUInt32s(ArrayRef<ExecutorAddr> Rs) {
    orc::promise<MSVCPExpected<ReadUIntsResult<uint32_t>>> P;
    auto F = P.get_future();
    readUInt32sAsync(Rs, [&](Expected<ReadUIntsResult<uint32_t>> Result) {
      P.set_value(std::move(Result));
    });
    return F.get(EPC.getDispatcher());
  }

  Expected<ReadUIntsResult<uint64_t>> readUInt64s(ArrayRef<ExecutorAddr> Rs) {
    orc::promise<MSVCPExpected<ReadUIntsResult<uint64_t>>> P;
    auto F = P.get_future();
    readUInt64sAsync(Rs, [&](Expected<ReadUIntsResult<uint64_t>> Result) {
      P.set_value(std::move(Result));
    });
    return F.get(EPC.getDispatcher());
  }

  Expected<ReadPointersResult> readPointers(ArrayRef<ExecutorAddr> Rs) {
    orc::promise<MSVCPExpected<ReadPointersResult>> P;
    auto F = P.get_future();
    readPointersAsync(Rs, [&](Expected<ReadPointersResult> Result) {
      P.set_value(std::move(Result));
    });
    return F.get(EPC.getDispatcher());
  }

  Expected<ReadBuffersResult> readBuffers(ArrayRef<ExecutorAddrRange> Rs) {
    orc::promise<MSVCPExpected<ReadBuffersResult>> P;
    auto F = P.get_future();
    readBuffersAsync(Rs, [&](Expected<ReadBuffersResult> Result) {
      P.set_value(std::move(Result));
    });
    return F.get(EPC.getDispatcher());
  }

  Expected<ReadStringsResult> readStrings(ArrayRef<ExecutorAddr> Rs) {
    orc::promise<MSVCPExpected<ReadStringsResult>> P;
    auto F = P.get_future();
    readStringsAsync(Rs, [&](Expected<ReadStringsResult> Result) {
      P.set_value(std::move(Result));
    });
    return F.get(EPC.getDispatcher());
  }

protected:
  ExecutorProcessControl &EPC;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_MEMORYACCESS_H
