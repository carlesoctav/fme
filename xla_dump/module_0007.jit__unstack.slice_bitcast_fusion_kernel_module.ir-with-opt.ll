; ModuleID = '__compute_module_slice_bitcast_fusion_kernel_module'
source_filename = "__compute_module_slice_bitcast_fusion_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn uwtable
define noalias noundef ptr @slice_bitcast_fusion(ptr readonly captures(none) %0) local_unnamed_addr #0 {
slice_bitcast_fusion_wrapped.exit:
  %1 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %2 = load ptr, ptr %1, align 8, !invariant.load !3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3, !dereferenceable !4
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !5
  tail call void @llvm.experimental.noalias.scope.decl(metadata !6)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !9)
  %6 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %7 = load i32, ptr %6, align 4, !invariant.load !3, !alias.scope !6, !noalias !11
  store i32 %7, ptr %5, align 4, !alias.scope !11, !noalias !6
  %8 = getelementptr inbounds nuw i8, ptr %3, i64 12
  %9 = load i32, ptr %8, align 4, !invariant.load !3, !alias.scope !6, !noalias !11
  %10 = getelementptr inbounds nuw i8, ptr %5, i64 4
  store i32 %9, ptr %10, align 4, !alias.scope !11, !noalias !6
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 1}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 16}
!5 = !{i64 8}
!6 = !{!7}
!7 = distinct !{!7, !8, !"slice_bitcast_fusion_wrapped: argument 0"}
!8 = distinct !{!8, !"slice_bitcast_fusion_wrapped"}
!9 = !{!10}
!10 = distinct !{!10, !8, !"slice_bitcast_fusion_wrapped: argument 1"}
!11 = !{!12, !10}
!12 = distinct !{!12, !13, !"xla.slice_index=1"}
!13 = distinct !{!13}
