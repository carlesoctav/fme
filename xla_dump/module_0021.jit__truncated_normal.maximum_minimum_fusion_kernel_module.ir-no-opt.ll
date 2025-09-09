; ModuleID = '__compute_module_maximum_minimum_fusion_kernel_module'
source_filename = "__compute_module_maximum_minimum_fusion_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelArg = type { ptr, i64 }
%kernel_dim3 = type { i64, i64, i64 }

declare float @xla.log1p.f32(float)

; Function Attrs: uwtable
define ptr @maximum_minimum_fusion(ptr %0) #0 {
  %2 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !4
  %6 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 1, i32 0
  %7 = load ptr, ptr %6, align 8, !invariant.load !3, !dereferenceable !4
  %8 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 2, i32 0
  %9 = load ptr, ptr %8, align 8, !invariant.load !3, !dereferenceable !5
  %10 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 3, i32 0
  %11 = load ptr, ptr %10, align 8, !invariant.load !3, !dereferenceable !5
  %12 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 4, i32 0
  %13 = load ptr, ptr %12, align 8, !invariant.load !3, !dereferenceable !4
  %14 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 5, i32 0
  %15 = load ptr, ptr %14, align 8, !invariant.load !3, !dereferenceable !4
  %16 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 6, i32 0
  %17 = load ptr, ptr %16, align 8, !invariant.load !3, !dereferenceable !5
  %18 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %19 = load ptr, ptr %18, align 8
  %20 = getelementptr inbounds %kernel_dim3, ptr %19, i32 0, i32 0
  %21 = load i64, ptr %20, align 4, !invariant.load !3
  %22 = getelementptr inbounds %kernel_dim3, ptr %19, i32 0, i32 1
  %23 = load i64, ptr %22, align 4, !invariant.load !3
  %24 = getelementptr inbounds %kernel_dim3, ptr %19, i32 0, i32 2
  %25 = load i64, ptr %24, align 4, !invariant.load !3
  call void @maximum_minimum_fusion_wrapped(ptr %5, ptr %7, ptr %9, ptr %11, ptr %13, ptr %15, ptr %17, i64 %21, i64 %23, i64 %25)
  ret ptr null
}

; Function Attrs: alwaysinline
define internal void @maximum_minimum_fusion_wrapped(ptr noalias align 64 dereferenceable(4) %0, ptr noalias align 64 dereferenceable(4) %1, ptr noalias align 64 dereferenceable(2048) %2, ptr noalias align 64 dereferenceable(2048) %3, ptr noalias align 64 dereferenceable(4) %4, ptr noalias align 64 dereferenceable(4) %5, ptr noalias align 64 dereferenceable(2048) %6, i64 %7, i64 %8, i64 %9) #1 {
  %11 = getelementptr inbounds [1 x float], ptr %0, i32 0, i32 0
  %12 = load float, ptr %11, align 4, !invariant.load !3, !noalias !6
  %13 = getelementptr inbounds [1 x float], ptr %1, i32 0, i32 0
  %14 = load float, ptr %13, align 4, !invariant.load !3, !noalias !6
  %15 = fsub float %12, %14
  %16 = getelementptr inbounds [1 x i32], ptr %4, i32 0, i32 0
  %17 = load i32, ptr %16, align 4, !invariant.load !3, !noalias !6
  %18 = sitofp i32 %17 to float
  %19 = bitcast float %18 to i32
  %20 = and i32 %19, -2147483648
  %21 = and i32 %19, 2147483647
  %22 = icmp sgt i32 %21, 2139095040
  %23 = icmp ne i32 %20, 0
  %24 = or i1 %22, %23
  %25 = select i1 %24, i32 -1, i32 1
  %26 = icmp eq i32 %21, 0
  %27 = add i32 %19, %25
  %28 = select i1 %26, i32 1, i32 %27
  %29 = fcmp une float %18, %18
  %30 = select i1 %29, i32 2143289344, i32 %28
  %31 = bitcast i32 %30 to float
  %32 = getelementptr inbounds [1 x i32], ptr %5, i32 0, i32 0
  %33 = load i32, ptr %32, align 4, !invariant.load !3, !noalias !6
  %34 = sitofp i32 %33 to float
  %35 = bitcast float %34 to i32
  %36 = and i32 %35, -2147483648
  %37 = and i32 %35, 2147483647
  %38 = icmp sgt i32 %37, 2139095040
  %39 = icmp ne i32 %36, -2147483648
  %40 = or i1 %38, %39
  %41 = select i1 %40, i32 -1, i32 1
  %42 = icmp eq i32 %37, 0
  %43 = add i32 %35, %41
  %44 = select i1 %42, i32 -2147483647, i32 %43
  %45 = fcmp une float %34, %34
  %46 = select i1 %45, i32 2143289344, i32 %44
  %47 = bitcast i32 %46 to float
  br label %48

48:                                               ; preds = %114, %10
  %49 = phi i64 [ %115, %114 ], [ 0, %10 ]
  %50 = icmp slt i64 %49, 32
  br i1 %50, label %51, label %116

51:                                               ; preds = %48
  %52 = mul nsw i64 %49, 16
  br label %53

53:                                               ; preds = %56, %51
  %54 = phi i64 [ %113, %56 ], [ 0, %51 ]
  %55 = icmp slt i64 %54, 16
  br i1 %55, label %56, label %114

56:                                               ; preds = %53
  %57 = add nsw i64 %52, %54
  %58 = getelementptr inbounds [512 x i32], ptr %2, i32 0, i64 %57
  %59 = load i32, ptr %58, align 4, !invariant.load !3, !noalias !6
  %60 = getelementptr inbounds [512 x i32], ptr %3, i32 0, i64 %57
  %61 = load i32, ptr %60, align 4, !invariant.load !3, !noalias !6
  %62 = xor i32 %59, %61
  %63 = lshr i32 %62, 9
  %64 = or i32 %63, 1065353216
  %65 = bitcast i32 %64 to float
  %66 = fadd float %65, -1.000000e+00
  %67 = fmul float %66, %15
  %68 = fadd float %67, %14
  %69 = call float @llvm.maximum.f32(float %14, float %68)
  %70 = fneg float %69
  %71 = fmul float %69, %70
  %72 = call float @xla.log1p.f32(float %71)
  %73 = fneg float %72
  %74 = fcmp olt float %73, 5.000000e+00
  %75 = select i1 %74, float 0x3E5E2CB100000000, float 0xBF2A3E1360000000
  %76 = select i1 %74, float 0x3E970966C0000000, float 0x3F1A76AD60000000
  %77 = call float @llvm.sqrt.f32(float %73)
  %78 = fadd float %73, -2.500000e+00
  %79 = fadd float %77, -3.000000e+00
  %80 = select i1 %74, float %78, float %79
  %81 = fmul float %75, %80
  %82 = fadd float %76, %81
  %83 = select i1 %74, float 0xBECD8E6AE0000000, float 0x3F561B8E40000000
  %84 = fmul float %82, %80
  %85 = fadd float %83, %84
  %86 = select i1 %74, float 0xBED26B5820000000, float 0xBF6E17BCE0000000
  %87 = fmul float %85, %80
  %88 = fadd float %86, %87
  %89 = select i1 %74, float 0x3F2CA65B60000000, float 0x3F77824F60000000
  %90 = fmul float %88, %80
  %91 = fadd float %89, %90
  %92 = select i1 %74, float 0xBF548A8100000000, float 0xBF7F38BAE0000000
  %93 = fmul float %91, %80
  %94 = fadd float %92, %93
  %95 = select i1 %74, float 0xBF711C9DE0000000, float 0x3F8354AFC0000000
  %96 = fmul float %94, %80
  %97 = fadd float %95, %96
  %98 = select i1 %74, float 0x3FCF91EC60000000, float 0x3FF006DB60000000
  %99 = fmul float %97, %80
  %100 = fadd float %98, %99
  %101 = select i1 %74, float 0x3FF805C5E0000000, float 0x4006A9EFC0000000
  %102 = fmul float %100, %80
  %103 = call float @llvm.fabs.f32(float %69)
  %104 = fadd float %101, %102
  %105 = fcmp oeq float %103, 1.000000e+00
  %106 = fmul float %69, 0x7FF0000000000000
  %107 = fmul float %104, %69
  %108 = select i1 %105, float %106, float %107
  %109 = fmul float %108, 0x3FF6A09E60000000
  %110 = call float @llvm.maximum.f32(float %31, float %109)
  %111 = call float @llvm.minimum.f32(float %47, float %110)
  %112 = getelementptr inbounds [512 x float], ptr %6, i32 0, i64 %57
  store float %111, ptr %112, align 4, !alias.scope !6
  %113 = add i64 %54, 1
  br label %53

114:                                              ; preds = %53
  %115 = add i64 %49, 1
  br label %48

116:                                              ; preds = %48
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.minimum.f32(float, float) #2

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 10}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 4}
!5 = !{i64 2048}
!6 = !{!7}
!7 = distinct !{!7, !8, !"xla.slice_index=6"}
!8 = distinct !{!8}
