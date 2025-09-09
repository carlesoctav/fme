; ModuleID = '__compute_module_broadcast_multiply_fusion_kernel_module'
source_filename = "__compute_module_broadcast_multiply_fusion_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind uwtable
define noalias noundef ptr @broadcast_multiply_fusion(ptr readonly captures(none) %0) local_unnamed_addr #0 {
vector.ph:
  tail call void @llvm.experimental.noalias.scope.decl(metadata !3)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !6)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !8)
  %1 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %2 = load ptr, ptr %1, align 8, !invariant.load !10
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %4 = load ptr, ptr %3, align 8, !invariant.load !10, !dereferenceable !11
  %5 = load float, ptr %4, align 4, !invariant.load !10, !alias.scope !6, !noalias !12
  %broadcast.splatinsert = insertelement <8 x float> poison, float %5, i64 0
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %7 = load ptr, ptr %6, align 8, !invariant.load !10, !dereferenceable !15
  %8 = load ptr, ptr %2, align 8, !invariant.load !10, !dereferenceable !15
  %wide.vec = load <64 x float>, ptr %8, align 4, !invariant.load !10, !alias.scope !3, !noalias !16
  %9 = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <64 x i32> zeroinitializer
  %interleaved.vec = fmul <64 x float> %9, %wide.vec
  store <64 x float> %interleaved.vec, ptr %7, align 4, !alias.scope !17, !noalias !18
  %10 = getelementptr inbounds nuw i8, ptr %8, i64 256
  %wide.vec.1 = load <64 x float>, ptr %10, align 4, !invariant.load !10, !alias.scope !3, !noalias !16
  %11 = getelementptr inbounds nuw i8, ptr %7, i64 256
  %12 = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <64 x i32> zeroinitializer
  %interleaved.vec.1 = fmul <64 x float> %12, %wide.vec.1
  store <64 x float> %interleaved.vec.1, ptr %11, align 4, !alias.scope !17, !noalias !18
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { nofree norecurse nosync nounwind uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 0}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{!4}
!4 = distinct !{!4, !5, !"broadcast_multiply_fusion_wrapped: argument 0"}
!5 = distinct !{!5, !"broadcast_multiply_fusion_wrapped"}
!6 = !{!7}
!7 = distinct !{!7, !5, !"broadcast_multiply_fusion_wrapped: argument 1"}
!8 = !{!9}
!9 = distinct !{!9, !5, !"broadcast_multiply_fusion_wrapped: argument 2"}
!10 = !{}
!11 = !{i64 4}
!12 = !{!13, !4, !9}
!13 = distinct !{!13, !14, !"xla.slice_index=2"}
!14 = distinct !{!14}
!15 = !{i64 512}
!16 = !{!13, !7, !9}
!17 = !{!13, !9}
!18 = !{!4, !7}
