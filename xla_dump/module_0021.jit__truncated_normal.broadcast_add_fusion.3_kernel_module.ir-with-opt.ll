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
  %9 = phi i64 [ 0, %1 ], [ %42, %8 ]
  %10 = shl nuw nsw i64 %9, 4
  %11 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %10
  store i32 %7, ptr %11, align 4, !alias.scope !11, !noalias !6
  %12 = or disjoint i64 %10, 1
  %13 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %12
  store i32 %7, ptr %13, align 4, !alias.scope !11, !noalias !6
  %14 = or disjoint i64 %10, 2
  %15 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %14
  store i32 %7, ptr %15, align 4, !alias.scope !11, !noalias !6
  %16 = or disjoint i64 %10, 3
  %17 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %16
  store i32 %7, ptr %17, align 4, !alias.scope !11, !noalias !6
  %18 = or disjoint i64 %10, 4
  %19 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %18
  store i32 %7, ptr %19, align 4, !alias.scope !11, !noalias !6
  %20 = or disjoint i64 %10, 5
  %21 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %20
  store i32 %7, ptr %21, align 4, !alias.scope !11, !noalias !6
  %22 = or disjoint i64 %10, 6
  %23 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %22
  store i32 %7, ptr %23, align 4, !alias.scope !11, !noalias !6
  %24 = or disjoint i64 %10, 7
  %25 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %24
  store i32 %7, ptr %25, align 4, !alias.scope !11, !noalias !6
  %26 = or disjoint i64 %10, 8
  %27 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %26
  store i32 %7, ptr %27, align 4, !alias.scope !11, !noalias !6
  %28 = or disjoint i64 %10, 9
  %29 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %28
  store i32 %7, ptr %29, align 4, !alias.scope !11, !noalias !6
  %30 = or disjoint i64 %10, 10
  %31 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %30
  store i32 %7, ptr %31, align 4, !alias.scope !11, !noalias !6
  %32 = or disjoint i64 %10, 11
  %33 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %32
  store i32 %7, ptr %33, align 4, !alias.scope !11, !noalias !6
  %34 = or disjoint i64 %10, 12
  %35 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %34
  store i32 %7, ptr %35, align 4, !alias.scope !11, !noalias !6
  %36 = or disjoint i64 %10, 13
  %37 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %36
  store i32 %7, ptr %37, align 4, !alias.scope !11, !noalias !6
  %38 = or disjoint i64 %10, 14
  %39 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %38
  store i32 %7, ptr %39, align 4, !alias.scope !11, !noalias !6
  %40 = or disjoint i64 %10, 15
  %41 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %40
  store i32 %7, ptr %41, align 4, !alias.scope !11, !noalias !6
  %42 = add nuw nsw i64 %9, 1
  %exitcond.not = icmp eq i64 %42, 32
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
!1 = !{i32 1, !"xla_dylib_index", i64 6}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 8}
!5 = !{i64 2048}
!6 = !{!7}
!7 = distinct !{!7, !8, !"broadcast_add_fusion.3_wrapped: argument 0"}
!8 = distinct !{!8, !"broadcast_add_fusion.3_wrapped"}
!9 = !{!10}
!10 = distinct !{!10, !8, !"broadcast_add_fusion.3_wrapped: argument 1"}
!11 = !{!12, !10}
!12 = distinct !{!12, !13, !"xla.slice_index=1"}
!13 = distinct !{!13}
