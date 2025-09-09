; ModuleID = '__compute_module_broadcast_add_fusion_kernel_module'
source_filename = "__compute_module_broadcast_add_fusion_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind uwtable
define noalias noundef ptr @broadcast_add_fusion(ptr readonly captures(none) %0) local_unnamed_addr #0 {
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
  %12 = load ptr, ptr %11, align 8, !invariant.load !3, !dereferenceable !4
  %13 = getelementptr inbounds nuw i8, ptr %3, i64 80
  %14 = load ptr, ptr %13, align 8, !invariant.load !3, !dereferenceable !6
  tail call void @llvm.experimental.noalias.scope.decl(metadata !7)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !10)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !12)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !14)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !16)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !18)
  %15 = load i32, ptr %6, align 4, !invariant.load !3, !alias.scope !10, !noalias !20
  %.fr5 = freeze i32 %15
  %16 = sub i32 32, %.fr5
  %17 = icmp ult i32 %.fr5, 32
  %18 = icmp ult i32 %16, 32
  %19 = getelementptr inbounds nuw i8, ptr %6, i64 4
  %20 = load i32, ptr %19, align 4, !invariant.load !3, !alias.scope !10, !noalias !20
  %21 = sub i32 32, %20
  %22 = icmp ult i32 %20, 32
  %23 = icmp ult i32 %21, 32
  %24 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %25 = load i32, ptr %24, align 4, !invariant.load !3, !alias.scope !10, !noalias !20
  %26 = sub i32 32, %25
  %27 = icmp ult i32 %25, 32
  %28 = icmp ult i32 %26, 32
  %29 = getelementptr inbounds nuw i8, ptr %6, i64 12
  %30 = load i32, ptr %29, align 4, !invariant.load !3, !alias.scope !10, !noalias !20
  %31 = sub i32 32, %30
  %32 = icmp ult i32 %30, 32
  %33 = icmp ult i32 %31, 32
  %34 = load i32, ptr %4, align 4, !invariant.load !3, !alias.scope !7, !noalias !23
  %35 = load i32, ptr %12, align 4, !invariant.load !3, !alias.scope !16, !noalias !24
  %36 = add i32 %34, 1
  %37 = add i32 %36, %35
  br i1 %17, label %.split.us.us.preheader, label %.split.preheader

.split.preheader:                                 ; preds = %1
  %broadcast.splatinsert24 = insertelement <8 x i32> poison, i32 %37, i64 0
  %broadcast.splat25 = shufflevector <8 x i32> %broadcast.splatinsert24, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert22 = insertelement <8 x i32> poison, i32 %31, i64 0
  %broadcast.splat23 = shufflevector <8 x i32> %broadcast.splatinsert22, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert20 = insertelement <8 x i32> poison, i32 %30, i64 0
  %broadcast.splat21 = shufflevector <8 x i32> %broadcast.splatinsert20, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert18 = insertelement <8 x i32> poison, i32 %26, i64 0
  %broadcast.splat19 = shufflevector <8 x i32> %broadcast.splatinsert18, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert16 = insertelement <8 x i32> poison, i32 %25, i64 0
  %broadcast.splat17 = shufflevector <8 x i32> %broadcast.splatinsert16, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert14 = insertelement <8 x i32> poison, i32 %21, i64 0
  %broadcast.splat15 = shufflevector <8 x i32> %broadcast.splatinsert14, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert12 = insertelement <8 x i32> poison, i32 %20, i64 0
  %broadcast.splat13 = shufflevector <8 x i32> %broadcast.splatinsert12, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert = insertelement <8 x i32> poison, i32 %16, i64 0
  %broadcast.splat = shufflevector <8 x i32> %broadcast.splatinsert, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %.split

.split.us.us.preheader:                           ; preds = %1
  %broadcast.splatinsert45 = insertelement <8 x i32> poison, i32 %37, i64 0
  %broadcast.splat46 = shufflevector <8 x i32> %broadcast.splatinsert45, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert43 = insertelement <8 x i32> poison, i32 %31, i64 0
  %broadcast.splat44 = shufflevector <8 x i32> %broadcast.splatinsert43, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert41 = insertelement <8 x i32> poison, i32 %30, i64 0
  %broadcast.splat42 = shufflevector <8 x i32> %broadcast.splatinsert41, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert39 = insertelement <8 x i32> poison, i32 %26, i64 0
  %broadcast.splat40 = shufflevector <8 x i32> %broadcast.splatinsert39, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert37 = insertelement <8 x i32> poison, i32 %25, i64 0
  %broadcast.splat38 = shufflevector <8 x i32> %broadcast.splatinsert37, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert35 = insertelement <8 x i32> poison, i32 %21, i64 0
  %broadcast.splat36 = shufflevector <8 x i32> %broadcast.splatinsert35, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert33 = insertelement <8 x i32> poison, i32 %20, i64 0
  %broadcast.splat34 = shufflevector <8 x i32> %broadcast.splatinsert33, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert31 = insertelement <8 x i32> poison, i32 %16, i64 0
  %broadcast.splat32 = shufflevector <8 x i32> %broadcast.splatinsert31, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert29 = insertelement <8 x i32> poison, i32 %.fr5, i64 0
  %broadcast.splat30 = shufflevector <8 x i32> %broadcast.splatinsert29, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %.split.us.us

.split.us.us:                                     ; preds = %.split.us.us.preheader, %.split.us.us
  %38 = phi i64 [ %103, %.split.us.us ], [ 0, %.split.us.us.preheader ]
  %39 = shl nuw nsw i64 %38, 4
  %40 = getelementptr inbounds nuw [512 x i32], ptr %8, i64 0, i64 %39
  %wide.load49 = load <8 x i32>, ptr %40, align 4, !invariant.load !3, !alias.scope !12, !noalias !25
  %41 = shl <8 x i32> %wide.load49, %broadcast.splat30
  %42 = lshr <8 x i32> %wide.load49, %broadcast.splat32
  %43 = select i1 %18, <8 x i32> %42, <8 x i32> zeroinitializer
  %44 = getelementptr inbounds nuw [512 x i32], ptr %10, i64 0, i64 %39
  %wide.load50 = load <8 x i32>, ptr %44, align 4, !invariant.load !3, !alias.scope !14, !noalias !26
  %45 = or <8 x i32> %41, %43
  %46 = add <8 x i32> %wide.load50, %wide.load49
  %47 = xor <8 x i32> %45, %46
  %48 = shl <8 x i32> %47, %broadcast.splat34
  %49 = select i1 %22, <8 x i32> %48, <8 x i32> zeroinitializer
  %50 = lshr <8 x i32> %47, %broadcast.splat36
  %51 = select i1 %23, <8 x i32> %50, <8 x i32> zeroinitializer
  %52 = or <8 x i32> %49, %51
  %53 = add <8 x i32> %47, %46
  %54 = xor <8 x i32> %52, %53
  %55 = shl <8 x i32> %54, %broadcast.splat38
  %56 = select i1 %27, <8 x i32> %55, <8 x i32> zeroinitializer
  %57 = lshr <8 x i32> %54, %broadcast.splat40
  %58 = select i1 %28, <8 x i32> %57, <8 x i32> zeroinitializer
  %59 = or <8 x i32> %56, %58
  %60 = add <8 x i32> %54, %53
  %61 = xor <8 x i32> %59, %60
  %62 = shl <8 x i32> %61, %broadcast.splat42
  %63 = select i1 %32, <8 x i32> %62, <8 x i32> zeroinitializer
  %64 = lshr <8 x i32> %61, %broadcast.splat44
  %65 = select i1 %33, <8 x i32> %64, <8 x i32> zeroinitializer
  %66 = add <8 x i32> %61, %60
  %67 = or <8 x i32> %63, %65
  %68 = xor <8 x i32> %67, %66
  %69 = add <8 x i32> %broadcast.splat46, %68
  %70 = getelementptr inbounds nuw [512 x i32], ptr %14, i64 0, i64 %39
  store <8 x i32> %69, ptr %70, align 4, !alias.scope !27, !noalias !28
  %71 = or disjoint i64 %39, 8
  %72 = getelementptr inbounds nuw [512 x i32], ptr %8, i64 0, i64 %71
  %wide.load49.1 = load <8 x i32>, ptr %72, align 4, !invariant.load !3, !alias.scope !12, !noalias !25
  %73 = shl <8 x i32> %wide.load49.1, %broadcast.splat30
  %74 = lshr <8 x i32> %wide.load49.1, %broadcast.splat32
  %75 = select i1 %18, <8 x i32> %74, <8 x i32> zeroinitializer
  %76 = getelementptr inbounds nuw [512 x i32], ptr %10, i64 0, i64 %71
  %wide.load50.1 = load <8 x i32>, ptr %76, align 4, !invariant.load !3, !alias.scope !14, !noalias !26
  %77 = or <8 x i32> %73, %75
  %78 = add <8 x i32> %wide.load50.1, %wide.load49.1
  %79 = xor <8 x i32> %77, %78
  %80 = shl <8 x i32> %79, %broadcast.splat34
  %81 = select i1 %22, <8 x i32> %80, <8 x i32> zeroinitializer
  %82 = lshr <8 x i32> %79, %broadcast.splat36
  %83 = select i1 %23, <8 x i32> %82, <8 x i32> zeroinitializer
  %84 = or <8 x i32> %81, %83
  %85 = add <8 x i32> %79, %78
  %86 = xor <8 x i32> %84, %85
  %87 = shl <8 x i32> %86, %broadcast.splat38
  %88 = select i1 %27, <8 x i32> %87, <8 x i32> zeroinitializer
  %89 = lshr <8 x i32> %86, %broadcast.splat40
  %90 = select i1 %28, <8 x i32> %89, <8 x i32> zeroinitializer
  %91 = or <8 x i32> %88, %90
  %92 = add <8 x i32> %86, %85
  %93 = xor <8 x i32> %91, %92
  %94 = shl <8 x i32> %93, %broadcast.splat42
  %95 = select i1 %32, <8 x i32> %94, <8 x i32> zeroinitializer
  %96 = lshr <8 x i32> %93, %broadcast.splat44
  %97 = select i1 %33, <8 x i32> %96, <8 x i32> zeroinitializer
  %98 = add <8 x i32> %93, %92
  %99 = or <8 x i32> %95, %97
  %100 = xor <8 x i32> %99, %98
  %101 = add <8 x i32> %broadcast.splat46, %100
  %102 = getelementptr inbounds nuw [512 x i32], ptr %14, i64 0, i64 %71
  store <8 x i32> %101, ptr %102, align 4, !alias.scope !27, !noalias !28
  %103 = add nuw nsw i64 %38, 1
  %exitcond9.not = icmp eq i64 %103, 32
  br i1 %exitcond9.not, label %broadcast_add_fusion_wrapped.exit, label %.split.us.us, !llvm.loop !29

.split:                                           ; preds = %.split.preheader, %.split
  %104 = phi i64 [ %165, %.split ], [ 0, %.split.preheader ]
  %105 = shl nuw nsw i64 %104, 4
  %106 = getelementptr inbounds nuw [512 x i32], ptr %8, i64 0, i64 %105
  %wide.load = load <8 x i32>, ptr %106, align 4, !invariant.load !3, !alias.scope !12, !noalias !25
  %107 = lshr <8 x i32> %wide.load, %broadcast.splat
  %108 = select i1 %18, <8 x i32> %107, <8 x i32> zeroinitializer
  %109 = getelementptr inbounds nuw [512 x i32], ptr %10, i64 0, i64 %105
  %wide.load26 = load <8 x i32>, ptr %109, align 4, !invariant.load !3, !alias.scope !14, !noalias !26
  %110 = add <8 x i32> %wide.load26, %wide.load
  %111 = xor <8 x i32> %108, %110
  %112 = shl <8 x i32> %111, %broadcast.splat13
  %113 = select i1 %22, <8 x i32> %112, <8 x i32> zeroinitializer
  %114 = lshr <8 x i32> %111, %broadcast.splat15
  %115 = select i1 %23, <8 x i32> %114, <8 x i32> zeroinitializer
  %116 = or <8 x i32> %113, %115
  %117 = add <8 x i32> %111, %110
  %118 = xor <8 x i32> %116, %117
  %119 = shl <8 x i32> %118, %broadcast.splat17
  %120 = select i1 %27, <8 x i32> %119, <8 x i32> zeroinitializer
  %121 = lshr <8 x i32> %118, %broadcast.splat19
  %122 = select i1 %28, <8 x i32> %121, <8 x i32> zeroinitializer
  %123 = or <8 x i32> %120, %122
  %124 = add <8 x i32> %118, %117
  %125 = xor <8 x i32> %123, %124
  %126 = shl <8 x i32> %125, %broadcast.splat21
  %127 = select i1 %32, <8 x i32> %126, <8 x i32> zeroinitializer
  %128 = lshr <8 x i32> %125, %broadcast.splat23
  %129 = select i1 %33, <8 x i32> %128, <8 x i32> zeroinitializer
  %130 = add <8 x i32> %125, %124
  %131 = or <8 x i32> %127, %129
  %132 = xor <8 x i32> %131, %130
  %133 = add <8 x i32> %broadcast.splat25, %132
  %134 = getelementptr inbounds nuw [512 x i32], ptr %14, i64 0, i64 %105
  store <8 x i32> %133, ptr %134, align 4, !alias.scope !27, !noalias !28
  %135 = or disjoint i64 %105, 8
  %136 = getelementptr inbounds nuw [512 x i32], ptr %8, i64 0, i64 %135
  %wide.load.1 = load <8 x i32>, ptr %136, align 4, !invariant.load !3, !alias.scope !12, !noalias !25
  %137 = lshr <8 x i32> %wide.load.1, %broadcast.splat
  %138 = select i1 %18, <8 x i32> %137, <8 x i32> zeroinitializer
  %139 = getelementptr inbounds nuw [512 x i32], ptr %10, i64 0, i64 %135
  %wide.load26.1 = load <8 x i32>, ptr %139, align 4, !invariant.load !3, !alias.scope !14, !noalias !26
  %140 = add <8 x i32> %wide.load26.1, %wide.load.1
  %141 = xor <8 x i32> %138, %140
  %142 = shl <8 x i32> %141, %broadcast.splat13
  %143 = select i1 %22, <8 x i32> %142, <8 x i32> zeroinitializer
  %144 = lshr <8 x i32> %141, %broadcast.splat15
  %145 = select i1 %23, <8 x i32> %144, <8 x i32> zeroinitializer
  %146 = or <8 x i32> %143, %145
  %147 = add <8 x i32> %141, %140
  %148 = xor <8 x i32> %146, %147
  %149 = shl <8 x i32> %148, %broadcast.splat17
  %150 = select i1 %27, <8 x i32> %149, <8 x i32> zeroinitializer
  %151 = lshr <8 x i32> %148, %broadcast.splat19
  %152 = select i1 %28, <8 x i32> %151, <8 x i32> zeroinitializer
  %153 = or <8 x i32> %150, %152
  %154 = add <8 x i32> %148, %147
  %155 = xor <8 x i32> %153, %154
  %156 = shl <8 x i32> %155, %broadcast.splat21
  %157 = select i1 %32, <8 x i32> %156, <8 x i32> zeroinitializer
  %158 = lshr <8 x i32> %155, %broadcast.splat23
  %159 = select i1 %33, <8 x i32> %158, <8 x i32> zeroinitializer
  %160 = add <8 x i32> %155, %154
  %161 = or <8 x i32> %157, %159
  %162 = xor <8 x i32> %161, %160
  %163 = add <8 x i32> %broadcast.splat25, %162
  %164 = getelementptr inbounds nuw [512 x i32], ptr %14, i64 0, i64 %135
  store <8 x i32> %163, ptr %164, align 4, !alias.scope !27, !noalias !28
  %165 = add nuw nsw i64 %104, 1
  %exitcond7.not = icmp eq i64 %165, 32
  br i1 %exitcond7.not, label %broadcast_add_fusion_wrapped.exit, label %.split

broadcast_add_fusion_wrapped.exit:                ; preds = %.split, %.split.us.us
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { nofree norecurse nosync nounwind uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 9}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 4}
!5 = !{i64 16}
!6 = !{i64 2048}
!7 = !{!8}
!8 = distinct !{!8, !9, !"broadcast_add_fusion_wrapped: argument 0"}
!9 = distinct !{!9, !"broadcast_add_fusion_wrapped"}
!10 = !{!11}
!11 = distinct !{!11, !9, !"broadcast_add_fusion_wrapped: argument 1"}
!12 = !{!13}
!13 = distinct !{!13, !9, !"broadcast_add_fusion_wrapped: argument 2"}
!14 = !{!15}
!15 = distinct !{!15, !9, !"broadcast_add_fusion_wrapped: argument 3"}
!16 = !{!17}
!17 = distinct !{!17, !9, !"broadcast_add_fusion_wrapped: argument 4"}
!18 = !{!19}
!19 = distinct !{!19, !9, !"broadcast_add_fusion_wrapped: argument 5"}
!20 = !{!21, !8, !13, !15, !17, !19}
!21 = distinct !{!21, !22, !"xla.slice_index=5"}
!22 = distinct !{!22}
!23 = !{!21, !11, !13, !15, !17, !19}
!24 = !{!21, !8, !11, !13, !15, !19}
!25 = !{!21, !8, !11, !15, !17, !19}
!26 = !{!21, !8, !11, !13, !17, !19}
!27 = !{!21, !19}
!28 = !{!8, !11, !13, !15, !17}
!29 = distinct !{!29, !30}
!30 = !{!"llvm.loop.unswitch.nontrivial.disable"}
