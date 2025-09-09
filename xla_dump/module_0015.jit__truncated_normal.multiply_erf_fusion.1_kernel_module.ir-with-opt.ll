; ModuleID = '__compute_module_multiply_erf_fusion.1_kernel_module'
source_filename = "__compute_module_multiply_erf_fusion.1_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn uwtable
define noalias noundef ptr @multiply_erf_fusion.1(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = load ptr, ptr %5, align 8, !invariant.load !3, !dereferenceable !4
  tail call void @llvm.experimental.noalias.scope.decl(metadata !5)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !8)
  %7 = load i32, ptr %4, align 4, !invariant.load !3, !alias.scope !5, !noalias !10
  %8 = sitofp i32 %7 to float
  %9 = fmul float %8, 0x3FE6A09E60000000
  %10 = tail call float @llvm.maximum.f32(float %9, float 0xC00DF38D00000000)
  %11 = tail call float @llvm.minimum.f32(float %10, float 0x400DF38D00000000)
  %12 = fmul float %11, %11
  %13 = tail call float @llvm.fma.f32(float %12, float 0x3F2E05AA20000000, float 0x3F6BEBB440000000)
  %14 = tail call float @llvm.fma.f32(float %13, float %12, float 0x3FAA16DD60000000)
  %15 = tail call float @llvm.fma.f32(float %14, float %12, float 0x3FC7B4E800000000)
  %16 = tail call float @llvm.fma.f32(float %15, float %12, float 0x3FF20DD740000000)
  %17 = fmul float %11, %16
  %18 = tail call float @llvm.fma.f32(float %12, float 0xBE7FA720C0000000, float 0x3EF8B11BE0000000)
  %19 = tail call float @llvm.fma.f32(float %18, float %12, float 0x3F50ADA500000000)
  %20 = tail call float @llvm.fma.f32(float %19, float %12, float 0x3F8CD0FA80000000)
  %21 = tail call float @llvm.fma.f32(float %20, float %12, float 0x3FBC698420000000)
  %22 = tail call float @llvm.fma.f32(float %21, float %12, float 0x3FDFD68940000000)
  %23 = tail call float @llvm.fma.f32(float %22, float %12, float 1.000000e+00)
  %24 = fdiv float %17, %23
  store float %24, ptr %6, align 4, !alias.scope !10, !noalias !5
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.minimum.f32(float, float) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 6}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 4}
!5 = !{!6}
!6 = distinct !{!6, !7, !"multiply_erf_fusion.1_wrapped: argument 0"}
!7 = distinct !{!7, !"multiply_erf_fusion.1_wrapped"}
!8 = !{!9}
!9 = distinct !{!9, !7, !"multiply_erf_fusion.1_wrapped: argument 1"}
!10 = !{!11, !9}
!11 = distinct !{!11, !12, !"xla.slice_index=1"}
!12 = distinct !{!12}
