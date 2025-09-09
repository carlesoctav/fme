; ModuleID = '__compute_module_broadcast_add_fusion.2_kernel_module'
source_filename = "__compute_module_broadcast_add_fusion.2_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn uwtable
define noalias noundef ptr @broadcast_add_fusion.2(ptr readonly captures(none) %0) local_unnamed_addr #0 {
broadcast_add_fusion.2_wrapped.exit:
  %1 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %2 = load ptr, ptr %1, align 8, !invariant.load !3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3, !dereferenceable !4
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !4
  tail call void @llvm.experimental.noalias.scope.decl(metadata !5)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !8)
  %6 = getelementptr inbounds nuw i8, ptr %3, i64 4
  %7 = load i32, ptr %6, align 4, !invariant.load !3, !alias.scope !5, !noalias !10
  store i32 %7, ptr %5, align 4, !alias.scope !10, !noalias !5
  %8 = add i32 %7, 1
  %9 = getelementptr inbounds nuw i8, ptr %5, i64 4
  store i32 %8, ptr %9, align 4, !alias.scope !10, !noalias !5
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 4}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 8}
!5 = !{!6}
!6 = distinct !{!6, !7, !"broadcast_add_fusion.2_wrapped: argument 0"}
!7 = distinct !{!7, !"broadcast_add_fusion.2_wrapped"}
!8 = !{!9}
!9 = distinct !{!9, !7, !"broadcast_add_fusion.2_wrapped: argument 1"}
!10 = !{!11, !9}
!11 = distinct !{!11, !12, !"xla.slice_index=1"}
!12 = distinct !{!12}
