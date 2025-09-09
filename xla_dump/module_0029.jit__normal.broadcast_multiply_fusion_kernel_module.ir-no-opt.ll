; ModuleID = '__compute_module_broadcast_multiply_fusion_kernel_module'
source_filename = "__compute_module_broadcast_multiply_fusion_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelArg = type { ptr, i64 }
%kernel_dim3 = type { i64, i64, i64 }

declare float @xla.log1p.f32(float)

; Function Attrs: uwtable
define ptr @broadcast_multiply_fusion(ptr %0) #0 {
  %2 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !4
  %6 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 1, i32 0
  %7 = load ptr, ptr %6, align 8, !invariant.load !3, !dereferenceable !4
  %8 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 2, i32 0
  %9 = load ptr, ptr %8, align 8, !invariant.load !3, !dereferenceable !4
  %10 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %11 = load ptr, ptr %10, align 8
  %12 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 0
  %13 = load i64, ptr %12, align 4, !invariant.load !3
  %14 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 1
  %15 = load i64, ptr %14, align 4, !invariant.load !3
  %16 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 2
  %17 = load i64, ptr %16, align 4, !invariant.load !3
  call void @broadcast_multiply_fusion_wrapped(ptr %5, ptr %7, ptr %9, i64 %13, i64 %15, i64 %17)
  ret ptr null
}

; Function Attrs: alwaysinline
define internal void @broadcast_multiply_fusion_wrapped(ptr noalias align 64 dereferenceable(384) %0, ptr noalias align 64 dereferenceable(384) %1, ptr noalias align 64 dereferenceable(384) %2, i64 %3, i64 %4, i64 %5) #1 {
  br label %7

7:                                                ; preds = %71, %6
  %8 = phi i64 [ %72, %71 ], [ 0, %6 ]
  %9 = icmp slt i64 %8, 12
  br i1 %9, label %10, label %73

10:                                               ; preds = %7
  %11 = mul nsw i64 %8, 8
  br label %12

12:                                               ; preds = %15, %10
  %13 = phi i64 [ %70, %15 ], [ 0, %10 ]
  %14 = icmp slt i64 %13, 8
  br i1 %14, label %15, label %71

15:                                               ; preds = %12
  %16 = add nsw i64 %11, %13
  %17 = getelementptr inbounds [96 x i32], ptr %0, i32 0, i64 %16
  %18 = load i32, ptr %17, align 4, !invariant.load !3, !noalias !5
  %19 = getelementptr inbounds [96 x i32], ptr %1, i32 0, i64 %16
  %20 = load i32, ptr %19, align 4, !invariant.load !3, !noalias !5
  %21 = xor i32 %18, %20
  %22 = lshr i32 %21, 9
  %23 = or i32 %22, 1065353216
  %24 = bitcast i32 %23 to float
  %25 = fadd float %24, -1.000000e+00
  %26 = fmul float %25, 2.000000e+00
  %27 = fadd float %26, 0xBFEFFFFFE0000000
  %28 = call float @llvm.maximum.f32(float %27, float 0xBFEFFFFFE0000000)
  %29 = fneg float %28
  %30 = fmul float %28, %29
  %31 = call float @xla.log1p.f32(float %30)
  %32 = fneg float %31
  %33 = fcmp olt float %32, 5.000000e+00
  %34 = select i1 %33, float 0x3E5E2CB100000000, float 0xBF2A3E1360000000
  %35 = select i1 %33, float 0x3E970966C0000000, float 0x3F1A76AD60000000
  %36 = call float @llvm.sqrt.f32(float %32)
  %37 = fadd float %32, -2.500000e+00
  %38 = fadd float %36, -3.000000e+00
  %39 = select i1 %33, float %37, float %38
  %40 = fmul float %34, %39
  %41 = fadd float %35, %40
  %42 = select i1 %33, float 0xBECD8E6AE0000000, float 0x3F561B8E40000000
  %43 = fmul float %41, %39
  %44 = fadd float %42, %43
  %45 = select i1 %33, float 0xBED26B5820000000, float 0xBF6E17BCE0000000
  %46 = fmul float %44, %39
  %47 = fadd float %45, %46
  %48 = select i1 %33, float 0x3F2CA65B60000000, float 0x3F77824F60000000
  %49 = fmul float %47, %39
  %50 = fadd float %48, %49
  %51 = select i1 %33, float 0xBF548A8100000000, float 0xBF7F38BAE0000000
  %52 = fmul float %50, %39
  %53 = fadd float %51, %52
  %54 = select i1 %33, float 0xBF711C9DE0000000, float 0x3F8354AFC0000000
  %55 = fmul float %53, %39
  %56 = fadd float %54, %55
  %57 = select i1 %33, float 0x3FCF91EC60000000, float 0x3FF006DB60000000
  %58 = fmul float %56, %39
  %59 = fadd float %57, %58
  %60 = select i1 %33, float 0x3FF805C5E0000000, float 0x4006A9EFC0000000
  %61 = fmul float %59, %39
  %62 = call float @llvm.fabs.f32(float %28)
  %63 = fadd float %60, %61
  %64 = fcmp oeq float %62, 1.000000e+00
  %65 = fmul float %28, 0x7FF0000000000000
  %66 = fmul float %63, %28
  %67 = select i1 %64, float %65, float %66
  %68 = fmul float %67, 0x3FF6A09E60000000
  %69 = getelementptr inbounds [96 x float], ptr %2, i32 0, i64 %16
  store float %68, ptr %69, align 4, !alias.scope !5
  %70 = add i64 %13, 1
  br label %12

71:                                               ; preds = %12
  %72 = add i64 %8, 1
  br label %7

73:                                               ; preds = %7
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #2

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 9}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 384}
!5 = !{!6}
!6 = distinct !{!6, !7, !"xla.slice_index=2"}
!7 = distinct !{!7}
