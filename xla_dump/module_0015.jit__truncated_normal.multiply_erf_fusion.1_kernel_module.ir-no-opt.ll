; ModuleID = '__compute_module_multiply_erf_fusion.1_kernel_module'
source_filename = "__compute_module_multiply_erf_fusion.1_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelArg = type { ptr, i64 }
%kernel_dim3 = type { i64, i64, i64 }

; Function Attrs: uwtable
define ptr @multiply_erf_fusion.1(ptr %0) #0 {
  %2 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !4
  %6 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 1, i32 0
  %7 = load ptr, ptr %6, align 8, !invariant.load !3, !dereferenceable !4
  %8 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %9 = load ptr, ptr %8, align 8
  %10 = getelementptr inbounds %kernel_dim3, ptr %9, i32 0, i32 0
  %11 = load i64, ptr %10, align 4, !invariant.load !3
  %12 = getelementptr inbounds %kernel_dim3, ptr %9, i32 0, i32 1
  %13 = load i64, ptr %12, align 4, !invariant.load !3
  %14 = getelementptr inbounds %kernel_dim3, ptr %9, i32 0, i32 2
  %15 = load i64, ptr %14, align 4, !invariant.load !3
  call void @multiply_erf_fusion.1_wrapped(ptr %5, ptr %7, i64 %11, i64 %13, i64 %15)
  ret ptr null
}

; Function Attrs: alwaysinline
define internal void @multiply_erf_fusion.1_wrapped(ptr noalias align 64 dereferenceable(4) %0, ptr noalias align 64 dereferenceable(4) %1, i64 %2, i64 %3, i64 %4) #1 {
  %6 = getelementptr inbounds [1 x i32], ptr %0, i32 0, i32 0
  %7 = load i32, ptr %6, align 4, !invariant.load !3, !noalias !5
  %8 = sitofp i32 %7 to float
  %9 = fmul float %8, 0x3FE6A09E60000000
  %10 = call float @llvm.maximum.f32(float %9, float 0xC00DF38D00000000)
  %11 = call float @llvm.minimum.f32(float %10, float 0x400DF38D00000000)
  %12 = fmul float %11, %11
  %13 = call float @llvm.fma.f32(float 0x3F2E05AA20000000, float %12, float 0x3F6BEBB440000000)
  %14 = call float @llvm.fma.f32(float %13, float %12, float 0x3FAA16DD60000000)
  %15 = call float @llvm.fma.f32(float %14, float %12, float 0x3FC7B4E800000000)
  %16 = call float @llvm.fma.f32(float %15, float %12, float 0x3FF20DD740000000)
  %17 = fmul float %11, %16
  %18 = call float @llvm.fma.f32(float 0xBE7FA720C0000000, float %12, float 0x3EF8B11BE0000000)
  %19 = call float @llvm.fma.f32(float %18, float %12, float 0x3F50ADA500000000)
  %20 = call float @llvm.fma.f32(float %19, float %12, float 0x3F8CD0FA80000000)
  %21 = call float @llvm.fma.f32(float %20, float %12, float 0x3FBC698420000000)
  %22 = call float @llvm.fma.f32(float %21, float %12, float 0x3FDFD68940000000)
  %23 = call float @llvm.fma.f32(float %22, float %12, float 1.000000e+00)
  %24 = fdiv float %17, %23
  %25 = getelementptr inbounds [1 x float], ptr %1, i32 0, i32 0
  store float %24, ptr %25, align 4, !alias.scope !5
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.minimum.f32(float, float) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #2

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 6}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 4}
!5 = !{!6}
!6 = distinct !{!6, !7, !"xla.slice_index=1"}
!7 = distinct !{!7}
