; ModuleID = '__compute_module_broadcast_multiply_fusion_kernel_module'
source_filename = "__compute_module_broadcast_multiply_fusion_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@llvm.compiler.used = appending global [2 x ptr] [ptr @xla.log1p.v2f32, ptr @xla.log1p.v4f32], section "llvm.metadata"

; Function Attrs: uwtable
define noalias noundef ptr @broadcast_multiply_fusion(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = load ptr, ptr %5, align 8, !invariant.load !3, !dereferenceable !4
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %8 = load ptr, ptr %7, align 8, !invariant.load !3, !dereferenceable !4
  tail call void @llvm.experimental.noalias.scope.decl(metadata !5)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !8)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !10)
  br label %vector.ph

vector.ph:                                        ; preds = %1, %vector.ph
  %9 = phi i64 [ 0, %1 ], [ %99, %vector.ph ]
  %10 = shl nuw nsw i64 %9, 3
  %11 = getelementptr inbounds nuw [96 x i32], ptr %4, i64 0, i64 %10
  %wide.load = load <8 x i32>, ptr %11, align 4, !invariant.load !3, !alias.scope !5, !noalias !12
  %12 = getelementptr inbounds nuw [96 x i32], ptr %6, i64 0, i64 %10
  %wide.load3 = load <8 x i32>, ptr %12, align 4, !invariant.load !3, !alias.scope !8, !noalias !15
  %13 = xor <8 x i32> %wide.load3, %wide.load
  %14 = lshr <8 x i32> %13, splat (i32 9)
  %15 = or disjoint <8 x i32> %14, splat (i32 1065353216)
  %16 = bitcast <8 x i32> %15 to <8 x float>
  %17 = fadd <8 x float> %16, splat (float -1.000000e+00)
  %18 = fmul <8 x float> %17, splat (float 2.000000e+00)
  %19 = fadd <8 x float> %18, splat (float 0xBFEFFFFFE0000000)
  %20 = tail call <8 x float> @llvm.maximum.v8f32(<8 x float> %19, <8 x float> splat (float 0xBFEFFFFFE0000000))
  %21 = fneg <8 x float> %20
  %22 = fmul <8 x float> %20, %21
  %23 = fadd <8 x float> %22, splat (float 1.000000e+00)
  %24 = call <8 x float> @llvm.log.v8f32(<8 x float> %23)
  %25 = fmul <8 x float> %22, %22
  %26 = fmul <8 x float> %22, zeroinitializer
  %27 = fadd <8 x float> %26, splat (float 1.000000e+00)
  %28 = fmul <8 x float> %27, %22
  %29 = fadd <8 x float> %28, splat (float 0x402E2035A0000000)
  %30 = fmul <8 x float> %29, %22
  %31 = fadd <8 x float> %30, splat (float 0x4054C30B60000000)
  %32 = fmul <8 x float> %31, %22
  %33 = fadd <8 x float> %32, splat (float 0x406BB865A0000000)
  %34 = fmul <8 x float> %33, %22
  %35 = fadd <8 x float> %34, splat (float 0x4073519460000000)
  %36 = fmul <8 x float> %35, %22
  %37 = fadd <8 x float> %36, splat (float 0x406B0DB140000000)
  %38 = fmul <8 x float> %37, %22
  %39 = fadd <8 x float> %38, splat (float 0x404E0F3040000000)
  %40 = fadd <8 x float> %26, splat (float 0x3F07BC0960000000)
  %41 = fmul <8 x float> %40, %22
  %42 = fadd <8 x float> %41, splat (float 0x3FDFE818A0000000)
  %43 = fmul <8 x float> %42, %22
  %44 = fadd <8 x float> %43, splat (float 0x401A509F40000000)
  %45 = fmul <8 x float> %44, %22
  %46 = fadd <8 x float> %45, splat (float 0x403DE97380000000)
  %47 = fmul <8 x float> %46, %22
  %48 = fadd <8 x float> %47, splat (float 0x404E798EC0000000)
  %49 = fmul <8 x float> %48, %22
  %50 = fadd <8 x float> %49, splat (float 0x404C8E75A0000000)
  %51 = fmul <8 x float> %50, %22
  %52 = fadd <8 x float> %51, splat (float 0x40340A2020000000)
  %53 = fdiv <8 x float> %52, %39
  %54 = fmul <8 x float> %22, %25
  %55 = fmul <8 x float> %54, %53
  %56 = fmul <8 x float> %25, splat (float -5.000000e-01)
  %57 = fadd <8 x float> %56, %55
  %58 = fadd <8 x float> %22, %57
  %59 = call <8 x float> @llvm.fabs.v8f32(<8 x float> %22)
  %60 = fcmp olt <8 x float> %59, splat (float 0x3FDA8279A0000000)
  %61 = select <8 x i1> %60, <8 x float> %58, <8 x float> %24
  %62 = fneg <8 x float> %61
  %63 = fcmp ogt <8 x float> %61, splat (float -5.000000e+00)
  %64 = select <8 x i1> %63, <8 x float> splat (float 0x3E5E2CB100000000), <8 x float> splat (float 0xBF2A3E1360000000)
  %65 = select <8 x i1> %63, <8 x float> splat (float 0x3E970966C0000000), <8 x float> splat (float 0x3F1A76AD60000000)
  %66 = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %62)
  %67 = fsub <8 x float> splat (float -2.500000e+00), %61
  %68 = fadd <8 x float> %66, splat (float -3.000000e+00)
  %69 = select <8 x i1> %63, <8 x float> %67, <8 x float> %68
  %70 = fmul <8 x float> %64, %69
  %71 = fadd <8 x float> %65, %70
  %72 = select <8 x i1> %63, <8 x float> splat (float 0xBECD8E6AE0000000), <8 x float> splat (float 0x3F561B8E40000000)
  %73 = fmul <8 x float> %69, %71
  %74 = fadd <8 x float> %72, %73
  %75 = select <8 x i1> %63, <8 x float> splat (float 0xBED26B5820000000), <8 x float> splat (float 0xBF6E17BCE0000000)
  %76 = fmul <8 x float> %69, %74
  %77 = fadd <8 x float> %75, %76
  %78 = select <8 x i1> %63, <8 x float> splat (float 0x3F2CA65B60000000), <8 x float> splat (float 0x3F77824F60000000)
  %79 = fmul <8 x float> %69, %77
  %80 = fadd <8 x float> %78, %79
  %81 = select <8 x i1> %63, <8 x float> splat (float 0xBF548A8100000000), <8 x float> splat (float 0xBF7F38BAE0000000)
  %82 = fmul <8 x float> %69, %80
  %83 = fadd <8 x float> %81, %82
  %84 = select <8 x i1> %63, <8 x float> splat (float 0xBF711C9DE0000000), <8 x float> splat (float 0x3F8354AFC0000000)
  %85 = fmul <8 x float> %69, %83
  %86 = fadd <8 x float> %84, %85
  %87 = select <8 x i1> %63, <8 x float> splat (float 0x3FCF91EC60000000), <8 x float> splat (float 0x3FF006DB60000000)
  %88 = fmul <8 x float> %69, %86
  %89 = fadd <8 x float> %87, %88
  %90 = select <8 x i1> %63, <8 x float> splat (float 0x3FF805C5E0000000), <8 x float> splat (float 0x4006A9EFC0000000)
  %91 = fmul <8 x float> %69, %89
  %92 = tail call <8 x float> @llvm.fabs.v8f32(<8 x float> %20)
  %93 = fadd <8 x float> %90, %91
  %94 = fcmp oeq <8 x float> %92, splat (float 1.000000e+00)
  %95 = select <8 x i1> %94, <8 x float> splat (float 0x7FF0000000000000), <8 x float> %93
  %96 = fmul <8 x float> %20, %95
  %97 = fmul <8 x float> %96, splat (float 0x3FF6A09E60000000)
  %98 = getelementptr inbounds nuw [96 x float], ptr %8, i64 0, i64 %10
  store <8 x float> %97, ptr %98, align 4, !alias.scope !16, !noalias !17
  %99 = add nuw nsw i64 %9, 1
  %exitcond2.not = icmp eq i64 %99, 12
  br i1 %exitcond2.not, label %broadcast_multiply_fusion_wrapped.exit, label %vector.ph

broadcast_multiply_fusion_wrapped.exit:           ; preds = %vector.ph
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

declare <2 x float> @xla.log1p.v2f32(<2 x float>) local_unnamed_addr

declare <4 x float> @xla.log1p.v4f32(<4 x float>) local_unnamed_addr

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.maximum.v8f32(<8 x float>, <8 x float>) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.fabs.v8f32(<8 x float>) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.log.v8f32(<8 x float>) #2

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 9}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 384}
!5 = !{!6}
!6 = distinct !{!6, !7, !"broadcast_multiply_fusion_wrapped: argument 0"}
!7 = distinct !{!7, !"broadcast_multiply_fusion_wrapped"}
!8 = !{!9}
!9 = distinct !{!9, !7, !"broadcast_multiply_fusion_wrapped: argument 1"}
!10 = !{!11}
!11 = distinct !{!11, !7, !"broadcast_multiply_fusion_wrapped: argument 2"}
!12 = !{!13, !9, !11}
!13 = distinct !{!13, !14, !"xla.slice_index=2"}
!14 = distinct !{!14}
!15 = !{!13, !6, !11}
!16 = !{!13, !11}
!17 = !{!6, !9}
