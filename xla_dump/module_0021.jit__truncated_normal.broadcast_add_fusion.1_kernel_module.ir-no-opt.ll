; ModuleID = '__compute_module_broadcast_add_fusion.1_kernel_module'
source_filename = "__compute_module_broadcast_add_fusion.1_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelArg = type { ptr, i64 }
%kernel_dim3 = type { i64, i64, i64 }

; Function Attrs: uwtable
define ptr @broadcast_add_fusion.1(ptr %0) #0 {
  %2 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !4
  %6 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 1, i32 0
  %7 = load ptr, ptr %6, align 8, !invariant.load !3, !dereferenceable !5
  %8 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 2, i32 0
  %9 = load ptr, ptr %8, align 8, !invariant.load !3, !dereferenceable !6
  %10 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 3, i32 0
  %11 = load ptr, ptr %10, align 8, !invariant.load !3, !dereferenceable !6
  %12 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 4, i32 0
  %13 = load ptr, ptr %12, align 8, !invariant.load !3, !dereferenceable !6
  %14 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %15 = load ptr, ptr %14, align 8
  %16 = getelementptr inbounds %kernel_dim3, ptr %15, i32 0, i32 0
  %17 = load i64, ptr %16, align 4, !invariant.load !3
  %18 = getelementptr inbounds %kernel_dim3, ptr %15, i32 0, i32 1
  %19 = load i64, ptr %18, align 4, !invariant.load !3
  %20 = getelementptr inbounds %kernel_dim3, ptr %15, i32 0, i32 2
  %21 = load i64, ptr %20, align 4, !invariant.load !3
  call void @broadcast_add_fusion.1_wrapped(ptr %5, ptr %7, ptr %9, ptr %11, ptr %13, i64 %17, i64 %19, i64 %21)
  ret ptr null
}

; Function Attrs: alwaysinline
define internal void @broadcast_add_fusion.1_wrapped(ptr noalias align 64 dereferenceable(4) %0, ptr noalias align 64 dereferenceable(16) %1, ptr noalias align 64 dereferenceable(2048) %2, ptr noalias align 64 dereferenceable(2048) %3, ptr noalias align 64 dereferenceable(2048) %4, i64 %5, i64 %6, i64 %7) #1 {
  %9 = getelementptr inbounds [4 x i32], ptr %1, i32 0, i32 0
  %10 = load i32, ptr %9, align 4, !invariant.load !3, !noalias !7
  %11 = sub i32 32, %10
  %12 = icmp ult i32 %10, 32
  %13 = icmp ult i32 %11, 32
  %14 = getelementptr inbounds [4 x i32], ptr %1, i32 0, i32 1
  %15 = load i32, ptr %14, align 4, !invariant.load !3, !noalias !7
  %16 = sub i32 32, %15
  %17 = icmp ult i32 %15, 32
  %18 = icmp ult i32 %16, 32
  %19 = getelementptr inbounds [4 x i32], ptr %1, i32 0, i32 2
  %20 = load i32, ptr %19, align 4, !invariant.load !3, !noalias !7
  %21 = sub i32 32, %20
  %22 = icmp ult i32 %20, 32
  %23 = icmp ult i32 %21, 32
  %24 = getelementptr inbounds [1 x i32], ptr %0, i32 0, i32 0
  %25 = load i32, ptr %24, align 4, !invariant.load !3, !noalias !7
  br label %26

26:                                               ; preds = %65, %8
  %27 = phi i64 [ %66, %65 ], [ 0, %8 ]
  %28 = icmp slt i64 %27, 32
  br i1 %28, label %29, label %67

29:                                               ; preds = %26
  %30 = mul nsw i64 %27, 16
  br label %31

31:                                               ; preds = %34, %29
  %32 = phi i64 [ %64, %34 ], [ 0, %29 ]
  %33 = icmp slt i64 %32, 16
  br i1 %33, label %34, label %65

34:                                               ; preds = %31
  %35 = add nsw i64 %30, %32
  %36 = getelementptr inbounds [512 x i32], ptr %2, i32 0, i64 %35
  %37 = load i32, ptr %36, align 4, !invariant.load !3, !noalias !7
  %38 = shl i32 %37, %10
  %39 = select i1 %12, i32 %38, i32 0
  %40 = lshr i32 %37, %11
  %41 = select i1 %13, i32 %40, i32 0
  %42 = getelementptr inbounds [512 x i32], ptr %3, i32 0, i64 %35
  %43 = load i32, ptr %42, align 4, !invariant.load !3, !noalias !7
  %44 = or i32 %39, %41
  %45 = add i32 %43, %37
  %46 = xor i32 %45, %44
  %47 = shl i32 %46, %15
  %48 = select i1 %17, i32 %47, i32 0
  %49 = lshr i32 %46, %16
  %50 = select i1 %18, i32 %49, i32 0
  %51 = or i32 %48, %50
  %52 = add i32 %45, %46
  %53 = xor i32 %52, %51
  %54 = shl i32 %53, %20
  %55 = select i1 %22, i32 %54, i32 0
  %56 = lshr i32 %53, %21
  %57 = select i1 %23, i32 %56, i32 0
  %58 = or i32 %55, %57
  %59 = add i32 %52, %53
  %60 = xor i32 %59, %58
  %61 = add i32 %59, %60
  %62 = add i32 %61, %25
  %63 = getelementptr inbounds [512 x i32], ptr %4, i32 0, i64 %35
  store i32 %62, ptr %63, align 4, !alias.scope !7
  %64 = add i64 %32, 1
  br label %31

65:                                               ; preds = %31
  %66 = add i64 %27, 1
  br label %26

67:                                               ; preds = %26
  ret void
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 8}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 4}
!5 = !{i64 16}
!6 = !{i64 2048}
!7 = !{!8}
!8 = distinct !{!8, !9, !"xla.slice_index=4"}
!9 = distinct !{!9}
