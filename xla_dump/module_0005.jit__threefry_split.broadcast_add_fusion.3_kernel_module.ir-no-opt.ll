; ModuleID = '__compute_module_broadcast_add_fusion.3_kernel_module'
source_filename = "__compute_module_broadcast_add_fusion.3_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelArg = type { ptr, i64 }
%kernel_dim3 = type { i64, i64, i64 }

; Function Attrs: uwtable
define ptr @broadcast_add_fusion.3(ptr %0) #0 {
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
  call void @broadcast_add_fusion.3_wrapped(ptr %5, ptr %7, i64 %11, i64 %13, i64 %15)
  ret ptr null
}

; Function Attrs: alwaysinline
define internal void @broadcast_add_fusion.3_wrapped(ptr noalias align 64 dereferenceable(8) %0, ptr noalias align 64 dereferenceable(8) %1, i64 %2, i64 %3, i64 %4) #1 {
  %6 = getelementptr inbounds [2 x i32], ptr %0, i32 0, i32 0
  %7 = load i32, ptr %6, align 4, !invariant.load !3, !noalias !5
  br label %8

8:                                                ; preds = %11, %5
  %9 = phi i64 [ %16, %11 ], [ 0, %5 ]
  %10 = icmp slt i64 %9, 2
  br i1 %10, label %11, label %17

11:                                               ; preds = %8
  %12 = lshr i64 %9, 32
  %13 = trunc i64 %12 to i32
  %14 = add i32 %13, %7
  %15 = getelementptr inbounds [2 x i32], ptr %1, i32 0, i64 %9
  store i32 %14, ptr %15, align 4, !alias.scope !5
  %16 = add i64 %9, 1
  br label %8

17:                                               ; preds = %8
  ret void
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 6}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 8}
!5 = !{!6}
!6 = distinct !{!6, !7, !"xla.slice_index=1"}
!7 = distinct !{!7}
