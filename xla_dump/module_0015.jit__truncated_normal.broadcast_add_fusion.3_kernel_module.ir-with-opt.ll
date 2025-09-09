; ModuleID = '__compute_module_broadcast_add_fusion.3_kernel_module'
source_filename = "__compute_module_broadcast_add_fusion.3_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind uwtable
define noalias noundef ptr @broadcast_add_fusion.3(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = load ptr, ptr %5, align 8, !invariant.load !3, !dereferenceable !5
  tail call void @llvm.experimental.noalias.scope.decl(metadata !6)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !9)
  %7 = load i32, ptr %4, align 4, !invariant.load !3, !alias.scope !6, !noalias !11
  br label %8

8:                                                ; preds = %1, %8
  %9 = phi i64 [ 0, %1 ], [ %26, %8 ]
  %10 = shl nuw nsw i64 %9, 3
  %11 = getelementptr inbounds nuw [128 x i32], ptr %6, i64 0, i64 %10
  store i32 %7, ptr %11, align 4, !alias.scope !11, !noalias !6
  %12 = or disjoint i64 %10, 1
  %13 = getelementptr inbounds nuw [128 x i32], ptr %6, i64 0, i64 %12
  store i32 %7, ptr %13, align 4, !alias.scope !11, !noalias !6
  %14 = or disjoint i64 %10, 2
  %15 = getelementptr inbounds nuw [128 x i32], ptr %6, i64 0, i64 %14
  store i32 %7, ptr %15, align 4, !alias.scope !11, !noalias !6
  %16 = or disjoint i64 %10, 3
  %17 = getelementptr inbounds nuw [128 x i32], ptr %6, i64 0, i64 %16
  store i32 %7, ptr %17, align 4, !alias.scope !11, !noalias !6
  %18 = or disjoint i64 %10, 4
  %19 = getelementptr inbounds nuw [128 x i32], ptr %6, i64 0, i64 %18
  store i32 %7, ptr %19, align 4, !alias.scope !11, !noalias !6
  %20 = or disjoint i64 %10, 5
  %21 = getelementptr inbounds nuw [128 x i32], ptr %6, i64 0, i64 %20
  store i32 %7, ptr %21, align 4, !alias.scope !11, !noalias !6
  %22 = or disjoint i64 %10, 6
  %23 = getelementptr inbounds nuw [128 x i32], ptr %6, i64 0, i64 %22
  store i32 %7, ptr %23, align 4, !alias.scope !11, !noalias !6
  %24 = or disjoint i64 %10, 7
  %25 = getelementptr inbounds nuw [128 x i32], ptr %6, i64 0, i64 %24
  store i32 %7, ptr %25, align 4, !alias.scope !11, !noalias !6
  %26 = add nuw nsw i64 %9, 1
  %exitcond.not = icmp eq i64 %26, 16
  br i1 %exitcond.not, label %broadcast_add_fusion.3_wrapped.exit, label %8

broadcast_add_fusion.3_wrapped.exit:              ; preds = %8
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { nofree norecurse nosync nounwind uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 7}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 8}
!5 = !{i64 512}
!6 = !{!7}
!7 = distinct !{!7, !8, !"broadcast_add_fusion.3_wrapped: argument 0"}
!8 = distinct !{!8, !"broadcast_add_fusion.3_wrapped"}
!9 = !{!10}
!10 = distinct !{!10, !8, !"broadcast_add_fusion.3_wrapped: argument 1"}
!11 = !{!12, !10}
!12 = distinct !{!12, !13, !"xla.slice_index=1"}
!13 = distinct !{!13}
