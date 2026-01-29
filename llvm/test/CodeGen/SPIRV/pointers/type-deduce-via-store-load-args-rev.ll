; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - --translator-compatibility-mode | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#Bar:]] "bar"
; CHECK-DAG: OpName %[[#Foo:]] "foo"
; CHECK-DAG: OpName %[[#Test:]] "test"
; CHECK-DAG: %[[#Void:]] = OpTypeVoid
; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#CharGenPtr:]] = OpTypePointer Generic %[[#Char]]
; CHECK-DAG: %[[#CharFunPtr:]] = OpTypePointer Function %[[#Char]]
; CHECK-DAG: %[[#CharGenGenPtr:]] = OpTypePointer Generic %[[#CharGenPtr]]
; CHECK-DAG: %[[#CharFunGenPtr:]] = OpTypePointer Function %[[#CharGenPtr]]

; CHECK: %[[#Bar]] = OpFunction
; CHECK: %[[#BarVar:]] = OpVariable %[[#CharFunPtr]] Function
; CHECK: %[[#BarVarToGen:]] = OpPtrCastToGeneric %[[#CharGenPtr]] %[[#BarVar]]
; CHECK: %[[#]] = OpFunctionCall %[[#Void]] %[[#Foo]] %[[#BarVarToGen]]

; CHECK: %[[#Foo]] = OpFunction
; CHECK: %[[#FooArg1:]] = OpFunctionParameter %[[#CharGenPtr]]
; CHECK: %[[#FooVar:]] = OpVariable %[[#CharFunGenPtr]] Function
; CHECK: %[[#FooVarToGen:]] = OpPtrCastToGeneric %[[#CharGenGenPtr]] %[[#FooVar]]
; CHECK: OpStore %[[#FooVarToGen]] %[[#FooArg1]]
; CHECK: %[[#FooLoad:]] = OpLoad %[[#CharGenPtr]] %[[#FooVarToGen]]
; CHECK: %[[#]] = OpFunctionCall %[[#Void:]] %[[#Test]] %[[#FooLoad:]]

; CHECK: %[[#Test]] = OpFunction
; CHECK: %[[#TestArg1:]] = OpFunctionParameter %[[#CharGenPtr]]
; CHECK: %[[#TestVar:]] = OpVariable %[[#CharFunPtr]] Function
; CHECK: %[[#TestVarToGen:]] = OpPtrCastToGeneric %[[#CharGenPtr]] %[[#TestVar]]
; CHECK: %[[#TestVarCast:]] = OpBitcast %[[#CharGenGenPtr]] %[[#TestVarToGen]]
; CHECK: OpStore %[[#TestVarCast]] %[[#TestArg1]]

%t_range = type { %t_arr }
%t_arr = type { [1 x i64] }

define internal spir_func void @bar() {
  %GlobalOffset = alloca %t_range, align 8
  %GlobalOffset.ascast = addrspacecast ptr %GlobalOffset to ptr addrspace(4)
  call spir_func void @foo(ptr addrspace(4) noundef align 8 dereferenceable(8) %GlobalOffset.ascast)
  ret void
}

define internal spir_func void @foo(ptr addrspace(4) noundef align 8 dereferenceable(8) %Offset) {
entry:
  %Offset.addr = alloca ptr addrspace(4), align 8
  %Offset.addr.ascast = addrspacecast ptr %Offset.addr to ptr addrspace(4)
  store ptr addrspace(4) %Offset, ptr addrspace(4) %Offset.addr.ascast, align 8
  %r2 = load ptr addrspace(4), ptr addrspace(4) %Offset.addr.ascast, align 8
  call spir_func void @test(ptr addrspace(4) noundef align 8 dereferenceable(8) %r2)
  ret void
}

define void @test(ptr addrspace(4) noundef align 8 dereferenceable(8) %offset) {
  %offset.addr = alloca ptr addrspace(4), align 8
  %offset.addr.ascast = addrspacecast ptr %offset.addr to ptr addrspace(4)
  store ptr addrspace(4) %offset, ptr addrspace(4) %offset.addr.ascast, align 8
  ret void
}
