; ModuleID = '__compute_module_broadcast_add_fusion.2_kernel_module'
source_filename = "__compute_module_broadcast_add_fusion.2_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind uwtable
define noalias noundef ptr @broadcast_add_fusion.2(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = load ptr, ptr %5, align 8, !invariant.load !3, !dereferenceable !5
  tail call void @llvm.experimental.noalias.scope.decl(metadata !6)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !9)
  %7 = getelementptr inbounds nuw i8, ptr %4, i64 4
  %8 = load i32, ptr %7, align 4, !invariant.load !3, !alias.scope !6, !noalias !11
  br label %9

9:                                                ; preds = %1, %9
  %10 = phi i64 [ 0, %1 ], [ %75, %9 ]
  %11 = shl nuw nsw i64 %10, 4
  %12 = trunc nuw nsw i64 %11 to i32
  %13 = add i32 %8, %12
  %14 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %11
  store i32 %13, ptr %14, align 4, !alias.scope !11, !noalias !6
  %15 = or disjoint i64 %11, 1
  %16 = trunc nuw nsw i64 %15 to i32
  %17 = add i32 %8, %16
  %18 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %15
  store i32 %17, ptr %18, align 4, !alias.scope !11, !noalias !6
  %19 = or disjoint i64 %11, 2
  %20 = trunc nuw nsw i64 %19 to i32
  %21 = add i32 %8, %20
  %22 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %19
  store i32 %21, ptr %22, align 4, !alias.scope !11, !noalias !6
  %23 = or disjoint i64 %11, 3
  %24 = trunc nuw nsw i64 %23 to i32
  %25 = add i32 %8, %24
  %26 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %23
  store i32 %25, ptr %26, align 4, !alias.scope !11, !noalias !6
  %27 = or disjoint i64 %11, 4
  %28 = trunc nuw nsw i64 %27 to i32
  %29 = add i32 %8, %28
  %30 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %27
  store i32 %29, ptr %30, align 4, !alias.scope !11, !noalias !6
  %31 = or disjoint i64 %11, 5
  %32 = trunc nuw nsw i64 %31 to i32
  %33 = add i32 %8, %32
  %34 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %31
  store i32 %33, ptr %34, align 4, !alias.scope !11, !noalias !6
  %35 = or disjoint i64 %11, 6
  %36 = trunc nuw nsw i64 %35 to i32
  %37 = add i32 %8, %36
  %38 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %35
  store i32 %37, ptr %38, align 4, !alias.scope !11, !noalias !6
  %39 = or disjoint i64 %11, 7
  %40 = trunc nuw nsw i64 %39 to i32
  %41 = add i32 %8, %40
  %42 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %39
  store i32 %41, ptr %42, align 4, !alias.scope !11, !noalias !6
  %43 = or disjoint i64 %11, 8
  %44 = trunc nuw nsw i64 %43 to i32
  %45 = add i32 %8, %44
  %46 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %43
  store i32 %45, ptr %46, align 4, !alias.scope !11, !noalias !6
  %47 = or disjoint i64 %11, 9
  %48 = trunc nuw nsw i64 %47 to i32
  %49 = add i32 %8, %48
  %50 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %47
  store i32 %49, ptr %50, align 4, !alias.scope !11, !noalias !6
  %51 = or disjoint i64 %11, 10
  %52 = trunc nuw nsw i64 %51 to i32
  %53 = add i32 %8, %52
  %54 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %51
  store i32 %53, ptr %54, align 4, !alias.scope !11, !noalias !6
  %55 = or disjoint i64 %11, 11
  %56 = trunc nuw nsw i64 %55 to i32
  %57 = add i32 %8, %56
  %58 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %55
  store i32 %57, ptr %58, align 4, !alias.scope !11, !noalias !6
  %59 = or disjoint i64 %11, 12
  %60 = trunc nuw nsw i64 %59 to i32
  %61 = add i32 %8, %60
  %62 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %59
  store i32 %61, ptr %62, align 4, !alias.scope !11, !noalias !6
  %63 = or disjoint i64 %11, 13
  %64 = trunc nuw nsw i64 %63 to i32
  %65 = add i32 %8, %64
  %66 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %63
  store i32 %65, ptr %66, align 4, !alias.scope !11, !noalias !6
  %67 = or disjoint i64 %11, 14
  %68 = trunc nuw nsw i64 %67 to i32
  %69 = add i32 %8, %68
  %70 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %67
  store i32 %69, ptr %70, align 4, !alias.scope !11, !noalias !6
  %71 = or disjoint i64 %11, 15
  %72 = trunc nuw nsw i64 %71 to i32
  %73 = add i32 %8, %72
  %74 = getelementptr inbounds nuw [512 x i32], ptr %6, i64 0, i64 %71
  store i32 %73, ptr %74, align 4, !alias.scope !11, !noalias !6
  %75 = add nuw nsw i64 %10, 1
  %exitcond.not = icmp eq i64 %75, 32
  br i1 %exitcond.not, label %broadcast_add_fusion.2_wrapped.exit, label %9

broadcast_add_fusion.2_wrapped.exit:              ; preds = %9
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { nofree norecurse nosync nounwind uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 5}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 8}
!5 = !{i64 2048}
!6 = !{!7}
!7 = distinct !{!7, !8, !"broadcast_add_fusion.2_wrapped: argument 0"}
!8 = distinct !{!8, !"broadcast_add_fusion.2_wrapped"}
!9 = !{!10}
!10 = distinct !{!10, !8, !"broadcast_add_fusion.2_wrapped: argument 1"}
!11 = !{!12, !10}
!12 = distinct !{!12, !13, !"xla.slice_index=1"}
!13 = distinct !{!13}
