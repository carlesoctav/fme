; ModuleID = '__compute_module_broadcast_add_fusion.1_kernel_module'
source_filename = "__compute_module_broadcast_add_fusion.1_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind uwtable
define noalias noundef ptr @broadcast_add_fusion.1(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = load ptr, ptr %5, align 8, !invariant.load !3, !dereferenceable !5
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %8 = load ptr, ptr %7, align 8, !invariant.load !3, !dereferenceable !6
  %9 = getelementptr inbounds nuw i8, ptr %3, i64 48
  %10 = load ptr, ptr %9, align 8, !invariant.load !3, !dereferenceable !6
  %11 = getelementptr inbounds nuw i8, ptr %3, i64 64
  %12 = load ptr, ptr %11, align 8, !invariant.load !3, !dereferenceable !6
  tail call void @llvm.experimental.noalias.scope.decl(metadata !7)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !10)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !12)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !14)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !16)
  %13 = load i32, ptr %6, align 4, !invariant.load !3, !alias.scope !10, !noalias !18
  %.fr5 = freeze i32 %13
  %14 = sub i32 32, %.fr5
  %15 = icmp ult i32 %.fr5, 32
  %16 = icmp ult i32 %14, 32
  %17 = getelementptr inbounds nuw i8, ptr %6, i64 4
  %18 = load i32, ptr %17, align 4, !invariant.load !3, !alias.scope !10, !noalias !18
  %19 = sub i32 32, %18
  %20 = icmp ult i32 %18, 32
  %21 = icmp ult i32 %19, 32
  %22 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %23 = load i32, ptr %22, align 4, !invariant.load !3, !alias.scope !10, !noalias !18
  %24 = sub i32 32, %23
  %25 = icmp ult i32 %23, 32
  %26 = icmp ult i32 %24, 32
  %27 = load i32, ptr %4, align 4, !invariant.load !3, !alias.scope !7, !noalias !21
  br i1 %15, label %.split.us.us.preheader, label %.split.preheader

.split.preheader:                                 ; preds = %1
  %broadcast.splatinsert20 = insertelement <8 x i32> poison, i32 %27, i64 0
  %broadcast.splat21 = shufflevector <8 x i32> %broadcast.splatinsert20, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert18 = insertelement <8 x i32> poison, i32 %24, i64 0
  %broadcast.splat19 = shufflevector <8 x i32> %broadcast.splatinsert18, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert16 = insertelement <8 x i32> poison, i32 %23, i64 0
  %broadcast.splat17 = shufflevector <8 x i32> %broadcast.splatinsert16, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert14 = insertelement <8 x i32> poison, i32 %19, i64 0
  %broadcast.splat15 = shufflevector <8 x i32> %broadcast.splatinsert14, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert12 = insertelement <8 x i32> poison, i32 %18, i64 0
  %broadcast.splat13 = shufflevector <8 x i32> %broadcast.splatinsert12, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert = insertelement <8 x i32> poison, i32 %14, i64 0
  %broadcast.splat = shufflevector <8 x i32> %broadcast.splatinsert, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %.split

.split.us.us.preheader:                           ; preds = %1
  %broadcast.splatinsert37 = insertelement <8 x i32> poison, i32 %27, i64 0
  %broadcast.splat38 = shufflevector <8 x i32> %broadcast.splatinsert37, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert35 = insertelement <8 x i32> poison, i32 %24, i64 0
  %broadcast.splat36 = shufflevector <8 x i32> %broadcast.splatinsert35, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert33 = insertelement <8 x i32> poison, i32 %23, i64 0
  %broadcast.splat34 = shufflevector <8 x i32> %broadcast.splatinsert33, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert31 = insertelement <8 x i32> poison, i32 %19, i64 0
  %broadcast.splat32 = shufflevector <8 x i32> %broadcast.splatinsert31, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert29 = insertelement <8 x i32> poison, i32 %18, i64 0
  %broadcast.splat30 = shufflevector <8 x i32> %broadcast.splatinsert29, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert27 = insertelement <8 x i32> poison, i32 %14, i64 0
  %broadcast.splat28 = shufflevector <8 x i32> %broadcast.splatinsert27, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert25 = insertelement <8 x i32> poison, i32 %.fr5, i64 0
  %broadcast.splat26 = shufflevector <8 x i32> %broadcast.splatinsert25, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %.split.us.us

.split.us.us:                                     ; preds = %.split.us.us.preheader, %.split.us.us
  %28 = phi i64 [ %81, %.split.us.us ], [ 0, %.split.us.us.preheader ]
  %29 = shl nuw nsw i64 %28, 4
  %30 = getelementptr inbounds nuw [512 x i32], ptr %8, i64 0, i64 %29
  %wide.load41 = load <8 x i32>, ptr %30, align 4, !invariant.load !3, !alias.scope !12, !noalias !22
  %31 = shl <8 x i32> %wide.load41, %broadcast.splat26
  %32 = lshr <8 x i32> %wide.load41, %broadcast.splat28
  %33 = select i1 %16, <8 x i32> %32, <8 x i32> zeroinitializer
  %34 = getelementptr inbounds nuw [512 x i32], ptr %10, i64 0, i64 %29
  %wide.load42 = load <8 x i32>, ptr %34, align 4, !invariant.load !3, !alias.scope !14, !noalias !23
  %35 = or <8 x i32> %31, %33
  %36 = add <8 x i32> %wide.load42, %wide.load41
  %37 = xor <8 x i32> %35, %36
  %38 = shl <8 x i32> %37, %broadcast.splat30
  %39 = select i1 %20, <8 x i32> %38, <8 x i32> zeroinitializer
  %40 = lshr <8 x i32> %37, %broadcast.splat32
  %41 = select i1 %21, <8 x i32> %40, <8 x i32> zeroinitializer
  %42 = or <8 x i32> %39, %41
  %43 = add <8 x i32> %37, %36
  %44 = xor <8 x i32> %42, %43
  %45 = shl <8 x i32> %44, %broadcast.splat34
  %46 = select i1 %25, <8 x i32> %45, <8 x i32> zeroinitializer
  %47 = lshr <8 x i32> %44, %broadcast.splat36
  %48 = select i1 %26, <8 x i32> %47, <8 x i32> zeroinitializer
  %49 = or <8 x i32> %46, %48
  %50 = add <8 x i32> %44, %43
  %51 = xor <8 x i32> %49, %50
  %52 = add <8 x i32> %50, %broadcast.splat38
  %53 = add <8 x i32> %52, %51
  %54 = getelementptr inbounds nuw [512 x i32], ptr %12, i64 0, i64 %29
  store <8 x i32> %53, ptr %54, align 4, !alias.scope !24, !noalias !25
  %55 = or disjoint i64 %29, 8
  %56 = getelementptr inbounds nuw [512 x i32], ptr %8, i64 0, i64 %55
  %wide.load41.1 = load <8 x i32>, ptr %56, align 4, !invariant.load !3, !alias.scope !12, !noalias !22
  %57 = shl <8 x i32> %wide.load41.1, %broadcast.splat26
  %58 = lshr <8 x i32> %wide.load41.1, %broadcast.splat28
  %59 = select i1 %16, <8 x i32> %58, <8 x i32> zeroinitializer
  %60 = getelementptr inbounds nuw [512 x i32], ptr %10, i64 0, i64 %55
  %wide.load42.1 = load <8 x i32>, ptr %60, align 4, !invariant.load !3, !alias.scope !14, !noalias !23
  %61 = or <8 x i32> %57, %59
  %62 = add <8 x i32> %wide.load42.1, %wide.load41.1
  %63 = xor <8 x i32> %61, %62
  %64 = shl <8 x i32> %63, %broadcast.splat30
  %65 = select i1 %20, <8 x i32> %64, <8 x i32> zeroinitializer
  %66 = lshr <8 x i32> %63, %broadcast.splat32
  %67 = select i1 %21, <8 x i32> %66, <8 x i32> zeroinitializer
  %68 = or <8 x i32> %65, %67
  %69 = add <8 x i32> %63, %62
  %70 = xor <8 x i32> %68, %69
  %71 = shl <8 x i32> %70, %broadcast.splat34
  %72 = select i1 %25, <8 x i32> %71, <8 x i32> zeroinitializer
  %73 = lshr <8 x i32> %70, %broadcast.splat36
  %74 = select i1 %26, <8 x i32> %73, <8 x i32> zeroinitializer
  %75 = or <8 x i32> %72, %74
  %76 = add <8 x i32> %70, %69
  %77 = xor <8 x i32> %75, %76
  %78 = add <8 x i32> %76, %broadcast.splat38
  %79 = add <8 x i32> %78, %77
  %80 = getelementptr inbounds nuw [512 x i32], ptr %12, i64 0, i64 %55
  store <8 x i32> %79, ptr %80, align 4, !alias.scope !24, !noalias !25
  %81 = add nuw nsw i64 %28, 1
  %exitcond9.not = icmp eq i64 %81, 32
  br i1 %exitcond9.not, label %broadcast_add_fusion.1_wrapped.exit, label %.split.us.us, !llvm.loop !26

.split:                                           ; preds = %.split.preheader, %.split
  %82 = phi i64 [ %131, %.split ], [ 0, %.split.preheader ]
  %83 = shl nuw nsw i64 %82, 4
  %84 = getelementptr inbounds nuw [512 x i32], ptr %8, i64 0, i64 %83
  %wide.load = load <8 x i32>, ptr %84, align 4, !invariant.load !3, !alias.scope !12, !noalias !22
  %85 = lshr <8 x i32> %wide.load, %broadcast.splat
  %86 = select i1 %16, <8 x i32> %85, <8 x i32> zeroinitializer
  %87 = getelementptr inbounds nuw [512 x i32], ptr %10, i64 0, i64 %83
  %wide.load22 = load <8 x i32>, ptr %87, align 4, !invariant.load !3, !alias.scope !14, !noalias !23
  %88 = add <8 x i32> %wide.load22, %wide.load
  %89 = xor <8 x i32> %86, %88
  %90 = shl <8 x i32> %89, %broadcast.splat13
  %91 = select i1 %20, <8 x i32> %90, <8 x i32> zeroinitializer
  %92 = lshr <8 x i32> %89, %broadcast.splat15
  %93 = select i1 %21, <8 x i32> %92, <8 x i32> zeroinitializer
  %94 = or <8 x i32> %91, %93
  %95 = add <8 x i32> %89, %88
  %96 = xor <8 x i32> %94, %95
  %97 = shl <8 x i32> %96, %broadcast.splat17
  %98 = select i1 %25, <8 x i32> %97, <8 x i32> zeroinitializer
  %99 = lshr <8 x i32> %96, %broadcast.splat19
  %100 = select i1 %26, <8 x i32> %99, <8 x i32> zeroinitializer
  %101 = or <8 x i32> %98, %100
  %102 = add <8 x i32> %96, %95
  %103 = xor <8 x i32> %101, %102
  %104 = add <8 x i32> %102, %broadcast.splat21
  %105 = add <8 x i32> %104, %103
  %106 = getelementptr inbounds nuw [512 x i32], ptr %12, i64 0, i64 %83
  store <8 x i32> %105, ptr %106, align 4, !alias.scope !24, !noalias !25
  %107 = or disjoint i64 %83, 8
  %108 = getelementptr inbounds nuw [512 x i32], ptr %8, i64 0, i64 %107
  %wide.load.1 = load <8 x i32>, ptr %108, align 4, !invariant.load !3, !alias.scope !12, !noalias !22
  %109 = lshr <8 x i32> %wide.load.1, %broadcast.splat
  %110 = select i1 %16, <8 x i32> %109, <8 x i32> zeroinitializer
  %111 = getelementptr inbounds nuw [512 x i32], ptr %10, i64 0, i64 %107
  %wide.load22.1 = load <8 x i32>, ptr %111, align 4, !invariant.load !3, !alias.scope !14, !noalias !23
  %112 = add <8 x i32> %wide.load22.1, %wide.load.1
  %113 = xor <8 x i32> %110, %112
  %114 = shl <8 x i32> %113, %broadcast.splat13
  %115 = select i1 %20, <8 x i32> %114, <8 x i32> zeroinitializer
  %116 = lshr <8 x i32> %113, %broadcast.splat15
  %117 = select i1 %21, <8 x i32> %116, <8 x i32> zeroinitializer
  %118 = or <8 x i32> %115, %117
  %119 = add <8 x i32> %113, %112
  %120 = xor <8 x i32> %118, %119
  %121 = shl <8 x i32> %120, %broadcast.splat17
  %122 = select i1 %25, <8 x i32> %121, <8 x i32> zeroinitializer
  %123 = lshr <8 x i32> %120, %broadcast.splat19
  %124 = select i1 %26, <8 x i32> %123, <8 x i32> zeroinitializer
  %125 = or <8 x i32> %122, %124
  %126 = add <8 x i32> %120, %119
  %127 = xor <8 x i32> %125, %126
  %128 = add <8 x i32> %126, %broadcast.splat21
  %129 = add <8 x i32> %128, %127
  %130 = getelementptr inbounds nuw [512 x i32], ptr %12, i64 0, i64 %107
  store <8 x i32> %129, ptr %130, align 4, !alias.scope !24, !noalias !25
  %131 = add nuw nsw i64 %82, 1
  %exitcond7.not = icmp eq i64 %131, 32
  br i1 %exitcond7.not, label %broadcast_add_fusion.1_wrapped.exit, label %.split

broadcast_add_fusion.1_wrapped.exit:              ; preds = %.split, %.split.us.us
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { nofree norecurse nosync nounwind uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 8}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 4}
!5 = !{i64 16}
!6 = !{i64 2048}
!7 = !{!8}
!8 = distinct !{!8, !9, !"broadcast_add_fusion.1_wrapped: argument 0"}
!9 = distinct !{!9, !"broadcast_add_fusion.1_wrapped"}
!10 = !{!11}
!11 = distinct !{!11, !9, !"broadcast_add_fusion.1_wrapped: argument 1"}
!12 = !{!13}
!13 = distinct !{!13, !9, !"broadcast_add_fusion.1_wrapped: argument 2"}
!14 = !{!15}
!15 = distinct !{!15, !9, !"broadcast_add_fusion.1_wrapped: argument 3"}
!16 = !{!17}
!17 = distinct !{!17, !9, !"broadcast_add_fusion.1_wrapped: argument 4"}
!18 = !{!19, !8, !13, !15, !17}
!19 = distinct !{!19, !20, !"xla.slice_index=4"}
!20 = distinct !{!20}
!21 = !{!19, !11, !13, !15, !17}
!22 = !{!19, !8, !11, !15, !17}
!23 = !{!19, !8, !11, !13, !17}
!24 = !{!19, !17}
!25 = !{!8, !11, !13, !15}
!26 = distinct !{!26, !27}
!27 = !{!"llvm.loop.unswitch.nontrivial.disable"}
