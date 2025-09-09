; ModuleID = '__compute_module_bitcast_concatenate_fusion_kernel_module'
source_filename = "__compute_module_bitcast_concatenate_fusion_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelArg = type { ptr, i64 }
%kernel_dim3 = type { i64, i64, i64 }

; Function Attrs: uwtable
define ptr @bitcast_concatenate_fusion(ptr %0) #0 {
  %2 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !4
  %6 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 1, i32 0
  %7 = load ptr, ptr %6, align 8, !invariant.load !3, !dereferenceable !4
  %8 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 2, i32 0
  %9 = load ptr, ptr %8, align 8, !invariant.load !3, !dereferenceable !5
  %10 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %11 = load ptr, ptr %10, align 8
  %12 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 0
  %13 = load i64, ptr %12, align 4, !invariant.load !3
  %14 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 1
  %15 = load i64, ptr %14, align 4, !invariant.load !3
  %16 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 2
  %17 = load i64, ptr %16, align 4, !invariant.load !3
  call void @bitcast_concatenate_fusion_wrapped(ptr %5, ptr %7, ptr %9, i64 %13, i64 %15, i64 %17)
  ret ptr null
}

; Function Attrs: alwaysinline
define internal void @bitcast_concatenate_fusion_wrapped(ptr noalias align 64 dereferenceable(8) %0, ptr noalias align 64 dereferenceable(8) %1, ptr noalias align 64 dereferenceable(16) %2, i64 %3, i64 %4, i64 %5) #1 {
  br label %7

7:                                                ; preds = %10, %6
  %8 = phi i64 [ %15, %10 ], [ 0, %6 ]
  %9 = icmp slt i64 %8, 2
  br i1 %9, label %10, label %16

10:                                               ; preds = %7
  %11 = getelementptr inbounds [2 x i32], ptr %1, i32 0, i64 %8
  %12 = load i32, ptr %11, align 4, !invariant.load !3, !noalias !6
  %13 = mul nsw i64 %8, 2
  %14 = getelementptr inbounds [4 x i32], ptr %2, i32 0, i64 %13
  store i32 %12, ptr %14, align 4, !alias.scope !6
  %15 = add i64 %8, 1
  br label %7

16:                                               ; preds = %7
  br label %17

17:                                               ; preds = %20, %16
  %18 = phi i64 [ %26, %20 ], [ 0, %16 ]
  %19 = icmp slt i64 %18, 2
  br i1 %19, label %20, label %27

20:                                               ; preds = %17
  %21 = getelementptr inbounds [2 x i32], ptr %0, i32 0, i64 %18
  %22 = load i32, ptr %21, align 4, !invariant.load !3, !noalias !6
  %23 = mul nsw i64 %18, 2
  %24 = add nsw i64 %23, 1
  %25 = getelementptr inbounds [4 x i32], ptr %2, i32 0, i64 %24
  store i32 %22, ptr %25, align 4
  %26 = add i64 %18, 1
  br label %17

27:                                               ; preds = %17
  ret void
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 5}
!2 = !{!"xla_cpu_emitter__concatenate_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 8}
!5 = !{i64 16}
!6 = !{!7}
!7 = distinct !{!7, !8, !"xla.slice_index=2"}
!8 = distinct !{!8}
