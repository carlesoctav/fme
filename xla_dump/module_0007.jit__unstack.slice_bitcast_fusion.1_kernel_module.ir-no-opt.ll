; ModuleID = '__compute_module_slice_bitcast_fusion.1_kernel_module'
source_filename = "__compute_module_slice_bitcast_fusion.1_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelArg = type { ptr, i64 }
%kernel_dim3 = type { i64, i64, i64 }

; Function Attrs: uwtable
define ptr @slice_bitcast_fusion.1(ptr %0) #0 {
  %2 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !4
  %6 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 1, i32 0
  %7 = load ptr, ptr %6, align 8, !invariant.load !3, !dereferenceable !5
  %8 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %9 = load ptr, ptr %8, align 8
  %10 = getelementptr inbounds %kernel_dim3, ptr %9, i32 0, i32 0
  %11 = load i64, ptr %10, align 4, !invariant.load !3
  %12 = getelementptr inbounds %kernel_dim3, ptr %9, i32 0, i32 1
  %13 = load i64, ptr %12, align 4, !invariant.load !3
  %14 = getelementptr inbounds %kernel_dim3, ptr %9, i32 0, i32 2
  %15 = load i64, ptr %14, align 4, !invariant.load !3
  call void @slice_bitcast_fusion.1_wrapped(ptr %5, ptr %7, i64 %11, i64 %13, i64 %15)
  ret ptr null
}

; Function Attrs: alwaysinline
define internal void @slice_bitcast_fusion.1_wrapped(ptr noalias align 64 dereferenceable(16) %0, ptr noalias align 64 dereferenceable(8) %1, i64 %2, i64 %3, i64 %4) #1 {
  br label %6

6:                                                ; preds = %9, %5
  %7 = phi i64 [ %13, %9 ], [ 0, %5 ]
  %8 = icmp slt i64 %7, 2
  br i1 %8, label %9, label %14

9:                                                ; preds = %6
  %10 = getelementptr inbounds [4 x i32], ptr %0, i32 0, i64 %7
  %11 = load i32, ptr %10, align 4, !invariant.load !3, !noalias !6
  %12 = getelementptr inbounds [2 x i32], ptr %1, i32 0, i64 %7
  store i32 %11, ptr %12, align 4, !alias.scope !6
  %13 = add i64 %7, 1
  br label %6

14:                                               ; preds = %6
  ret void
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 0}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 16}
!5 = !{i64 8}
!6 = !{!7}
!7 = distinct !{!7, !8, !"xla.slice_index=1"}
!8 = distinct !{!8}
