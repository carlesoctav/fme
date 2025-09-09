; ModuleID = '__compute_module_broadcast_add_fusion_kernel_module'
source_filename = "__compute_module_broadcast_add_fusion_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelArg = type { ptr, i64 }
%kernel_dim3 = type { i64, i64, i64 }

; Function Attrs: uwtable
define ptr @broadcast_add_fusion(ptr %0) #0 {
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
  %13 = load ptr, ptr %12, align 8, !invariant.load !3, !dereferenceable !4
  %14 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 5, i32 0
  %15 = load ptr, ptr %14, align 8, !invariant.load !3, !dereferenceable !6
  %16 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %17 = load ptr, ptr %16, align 8
  %18 = getelementptr inbounds %kernel_dim3, ptr %17, i32 0, i32 0
  %19 = load i64, ptr %18, align 4, !invariant.load !3
  %20 = getelementptr inbounds %kernel_dim3, ptr %17, i32 0, i32 1
  %21 = load i64, ptr %20, align 4, !invariant.load !3
  %22 = getelementptr inbounds %kernel_dim3, ptr %17, i32 0, i32 2
  %23 = load i64, ptr %22, align 4, !invariant.load !3
  call void @broadcast_add_fusion_wrapped(ptr %5, ptr %7, ptr %9, ptr %11, ptr %13, ptr %15, i64 %19, i64 %21, i64 %23)
  ret ptr null
}

; Function Attrs: alwaysinline
define internal void @broadcast_add_fusion_wrapped(ptr noalias align 64 dereferenceable(4) %0, ptr noalias align 64 dereferenceable(16) %1, ptr noalias align 64 dereferenceable(384) %2, ptr noalias align 64 dereferenceable(384) %3, ptr noalias align 64 dereferenceable(4) %4, ptr noalias align 64 dereferenceable(384) %5, i64 %6, i64 %7, i64 %8) #1 {
  %10 = getelementptr inbounds [4 x i32], ptr %1, i32 0, i32 0
  %11 = load i32, ptr %10, align 4, !invariant.load !3, !noalias !7
  %12 = sub i32 32, %11
  %13 = icmp ult i32 %11, 32
  %14 = icmp ult i32 %12, 32
  %15 = getelementptr inbounds [4 x i32], ptr %1, i32 0, i32 1
  %16 = load i32, ptr %15, align 4, !invariant.load !3, !noalias !7
  %17 = sub i32 32, %16
  %18 = icmp ult i32 %16, 32
  %19 = icmp ult i32 %17, 32
  %20 = getelementptr inbounds [4 x i32], ptr %1, i32 0, i32 2
  %21 = load i32, ptr %20, align 4, !invariant.load !3, !noalias !7
  %22 = sub i32 32, %21
  %23 = icmp ult i32 %21, 32
  %24 = icmp ult i32 %22, 32
  %25 = getelementptr inbounds [4 x i32], ptr %1, i32 0, i32 3
  %26 = load i32, ptr %25, align 4, !invariant.load !3, !noalias !7
  %27 = sub i32 32, %26
  %28 = icmp ult i32 %26, 32
  %29 = icmp ult i32 %27, 32
  %30 = getelementptr inbounds [1 x i32], ptr %0, i32 0, i32 0
  %31 = load i32, ptr %30, align 4, !invariant.load !3, !noalias !7
  %32 = getelementptr inbounds [1 x i32], ptr %4, i32 0, i32 0
  %33 = load i32, ptr %32, align 4, !invariant.load !3, !noalias !7
  %34 = add i32 %33, 1
  br label %35

35:                                               ; preds = %81, %9
  %36 = phi i64 [ %82, %81 ], [ 0, %9 ]
  %37 = icmp slt i64 %36, 12
  br i1 %37, label %38, label %83

38:                                               ; preds = %35
  %39 = mul nsw i64 %36, 8
  br label %40

40:                                               ; preds = %43, %38
  %41 = phi i64 [ %80, %43 ], [ 0, %38 ]
  %42 = icmp slt i64 %41, 8
  br i1 %42, label %43, label %81

43:                                               ; preds = %40
  %44 = add nsw i64 %39, %41
  %45 = getelementptr inbounds [96 x i32], ptr %2, i32 0, i64 %44
  %46 = load i32, ptr %45, align 4, !invariant.load !3, !noalias !7
  %47 = shl i32 %46, %11
  %48 = select i1 %13, i32 %47, i32 0
  %49 = lshr i32 %46, %12
  %50 = select i1 %14, i32 %49, i32 0
  %51 = getelementptr inbounds [96 x i32], ptr %3, i32 0, i64 %44
  %52 = load i32, ptr %51, align 4, !invariant.load !3, !noalias !7
  %53 = or i32 %48, %50
  %54 = add i32 %52, %46
  %55 = xor i32 %54, %53
  %56 = shl i32 %55, %16
  %57 = select i1 %18, i32 %56, i32 0
  %58 = lshr i32 %55, %17
  %59 = select i1 %19, i32 %58, i32 0
  %60 = or i32 %57, %59
  %61 = add i32 %54, %55
  %62 = xor i32 %61, %60
  %63 = shl i32 %62, %21
  %64 = select i1 %23, i32 %63, i32 0
  %65 = lshr i32 %62, %22
  %66 = select i1 %24, i32 %65, i32 0
  %67 = or i32 %64, %66
  %68 = add i32 %61, %62
  %69 = xor i32 %68, %67
  %70 = shl i32 %69, %26
  %71 = select i1 %28, i32 %70, i32 0
  %72 = lshr i32 %69, %27
  %73 = select i1 %29, i32 %72, i32 0
  %74 = add i32 %68, %69
  %75 = or i32 %71, %73
  %76 = xor i32 %74, %75
  %77 = add i32 %76, %31
  %78 = add i32 %77, %34
  %79 = getelementptr inbounds [96 x i32], ptr %5, i32 0, i64 %44
  store i32 %78, ptr %79, align 4, !alias.scope !7
  %80 = add i64 %41, 1
  br label %40

81:                                               ; preds = %40
  %82 = add i64 %36, 1
  br label %35

83:                                               ; preds = %35
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
!6 = !{i64 384}
!7 = !{!8}
!8 = distinct !{!8, !9, !"xla.slice_index=5"}
!9 = distinct !{!9}
