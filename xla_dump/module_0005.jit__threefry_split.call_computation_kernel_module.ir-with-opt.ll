; ModuleID = '__compute_module_call_computation_kernel_module'
source_filename = "__compute_module_call_computation_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @call_kernel(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg19_gep = getelementptr i8, ptr %args, i64 304
  %arg19 = load ptr, ptr %arg19_gep, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %arg20_gep = getelementptr i8, ptr %args, i64 320
  %arg20 = load ptr, ptr %arg20_gep, align 8, !invariant.load !3, !dereferenceable !5, !align !5
  %arg21_gep = getelementptr i8, ptr %args, i64 336
  %arg21 = load ptr, ptr %arg21_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %arg22_gep = getelementptr i8, ptr %args, i64 352
  %arg22 = load ptr, ptr %arg22_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %arg23_gep = getelementptr i8, ptr %args, i64 368
  %arg23 = load ptr, ptr %arg23_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %arg25_gep = getelementptr i8, ptr %args, i64 400
  %arg25 = load ptr, ptr %arg25_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %arg26_gep = getelementptr i8, ptr %args, i64 416
  %arg26 = load ptr, ptr %arg26_gep, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %arg27_gep = getelementptr i8, ptr %args, i64 432
  %arg27 = load ptr, ptr %arg27_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %arg28_gep = getelementptr i8, ptr %args, i64 448
  %arg28 = load ptr, ptr %arg28_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %arg29_gep = getelementptr i8, ptr %args, i64 464
  %arg29 = load ptr, ptr %arg29_gep, align 8, !invariant.load !3, !dereferenceable !8, !align !5
  %arg30_gep = getelementptr i8, ptr %args, i64 480
  %arg30 = load ptr, ptr %arg30_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %arg31_gep = getelementptr i8, ptr %args, i64 496
  %arg31 = load ptr, ptr %arg31_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %arg32_gep = getelementptr i8, ptr %args, i64 512
  %arg32 = load ptr, ptr %arg32_gep, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %arg33_gep = getelementptr i8, ptr %args, i64 528
  %arg33 = load ptr, ptr %arg33_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %arg34_gep = getelementptr i8, ptr %args, i64 544
  %arg34 = load ptr, ptr %arg34_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %arg36_gep = getelementptr i8, ptr %args, i64 576
  %arg36 = load ptr, ptr %arg36_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %arg37_gep = getelementptr i8, ptr %args, i64 592
  %arg37 = load ptr, ptr %arg37_gep, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %arg38_gep = getelementptr i8, ptr %args, i64 608
  %arg38 = load ptr, ptr %arg38_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %2 = load i32, ptr %arg31, align 64, !alias.scope !9, !noalias !12
  %3 = icmp slt i32 %2, 5
  %4 = zext i1 %3 to i8
  store i8 %4, ptr %arg29, align 64, !alias.scope !19, !noalias !20
  br i1 %3, label %while.6.body.i.lr.ph, label %return

while.6.body.i.lr.ph:                             ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %arg21, i64 4
  %6 = getelementptr inbounds nuw i8, ptr %arg21, i64 8
  %7 = getelementptr inbounds nuw i8, ptr %arg21, i64 12
  %8 = getelementptr inbounds nuw i8, ptr %arg20, i64 8
  %9 = getelementptr inbounds nuw i8, ptr %arg20, i64 16
  %10 = getelementptr inbounds nuw i8, ptr %arg20, i64 24
  %11 = getelementptr inbounds nuw i8, ptr %arg20, i64 32
  %12 = getelementptr inbounds nuw i8, ptr %arg20, i64 40
  %13 = getelementptr inbounds nuw i8, ptr %arg20, i64 48
  %14 = getelementptr inbounds nuw i8, ptr %arg20, i64 56
  %15 = getelementptr inbounds nuw i8, ptr %arg37, i64 4
  %16 = getelementptr inbounds nuw i8, ptr %arg26, i64 4
  %17 = getelementptr inbounds nuw i8, ptr %arg32, i64 4
  %18 = getelementptr inbounds nuw i8, ptr %arg19, i64 4
  br label %while.6.body.i

while.6.body.i:                                   ; preds = %while.6.body.i.lr.ph, %while.6.body.i
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %arg27, ptr noundef nonnull align 64 dereferenceable(16) %arg30, i64 16, i1 false), !noalias !21
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %arg21, ptr noundef nonnull align 64 dereferenceable(16) %arg28, i64 16, i1 false), !noalias !21
  %19 = load i32, ptr %arg33, align 64, !noalias !21
  store i32 %19, ptr %arg22, align 64, !noalias !21
  %20 = load i32, ptr %arg23, align 64, !noalias !21
  store i32 %20, ptr %arg25, align 64, !noalias !21
  %21 = load i32, ptr %arg34, align 64, !noalias !21
  store i32 %21, ptr %arg38, align 64, !noalias !21
  %22 = load i64, ptr %arg19, align 64, !noalias !21
  store i64 %22, ptr %arg26, align 64, !noalias !21
  %23 = load i64, ptr %arg32, align 64, !noalias !21
  store i64 %23, ptr %arg37, align 64, !noalias !21
  %24 = load i32, ptr %arg31, align 64, !noalias !21
  store i32 %24, ptr %arg36, align 64, !noalias !21
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %arg30, ptr noundef nonnull align 64 dereferenceable(16) %arg21, i64 16, i1 false), !noalias !21
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %arg28, ptr noundef nonnull align 64 dereferenceable(16) %arg27, i64 16, i1 false), !noalias !21
  %25 = load i32, ptr %arg22, align 64, !noalias !21
  store i32 %25, ptr %arg23, align 64, !noalias !21
  %26 = load i32, ptr %arg38, align 64, !noalias !21
  store i32 %26, ptr %arg33, align 64, !noalias !21
  %27 = load i32, ptr %arg25, align 64, !noalias !21
  store i32 %27, ptr %arg34, align 64, !noalias !21
  %28 = load i32, ptr %arg21, align 64, !alias.scope !24, !noalias !26
  %shft.chk.i.i = icmp ult i32 %28, 32
  %29 = sub i32 32, %28
  %shft.chk1.i.i = icmp ult i32 %29, 32
  %30 = load i32, ptr %5, align 4, !alias.scope !24, !noalias !26
  %shft.chk2.i.i = icmp ult i32 %30, 32
  %31 = sub i32 32, %30
  %shft.chk4.i.i = icmp ult i32 %31, 32
  %32 = load i32, ptr %6, align 8, !alias.scope !24, !noalias !26
  %shft.chk5.i.i = icmp ult i32 %32, 32
  %33 = sub i32 32, %32
  %shft.chk7.i.i = icmp ult i32 %33, 32
  %34 = load i32, ptr %arg38, align 64, !alias.scope !36, !noalias !37
  %35 = load i32, ptr %arg37, align 64, !alias.scope !40, !noalias !41
  %36 = load i32, ptr %arg26, align 64, !alias.scope !42, !noalias !43
  %37 = add i32 %36, %35
  %38 = shl i32 %36, %28
  %39 = select i1 %shft.chk.i.i, i32 %38, i32 0
  %40 = lshr i32 %36, %29
  %41 = select i1 %shft.chk1.i.i, i32 %40, i32 0
  %42 = or i32 %41, %39
  %43 = xor i32 %42, %37
  %44 = add i32 %43, %37
  %45 = shl i32 %43, %30
  %46 = select i1 %shft.chk2.i.i, i32 %45, i32 0
  %47 = lshr i32 %43, %31
  %48 = select i1 %shft.chk4.i.i, i32 %47, i32 0
  %49 = or i32 %46, %48
  %50 = xor i32 %49, %44
  %51 = add i32 %50, %44
  %52 = shl i32 %50, %32
  %53 = select i1 %shft.chk5.i.i, i32 %52, i32 0
  %54 = lshr i32 %50, %33
  %55 = select i1 %shft.chk7.i.i, i32 %54, i32 0
  %56 = or i32 %53, %55
  %57 = xor i32 %56, %51
  %58 = add i32 %51, %34
  %59 = add i32 %58, %57
  store i32 %59, ptr %arg32, align 64, !alias.scope !44, !noalias !45
  %60 = load i32, ptr %15, align 4, !alias.scope !40, !noalias !41
  %61 = load i32, ptr %16, align 4, !alias.scope !42, !noalias !43
  %62 = add i32 %61, %60
  %63 = shl i32 %61, %28
  %64 = select i1 %shft.chk.i.i, i32 %63, i32 0
  %65 = lshr i32 %61, %29
  %66 = select i1 %shft.chk1.i.i, i32 %65, i32 0
  %67 = or i32 %66, %64
  %68 = xor i32 %67, %62
  %69 = add i32 %68, %62
  %70 = shl i32 %68, %30
  %71 = select i1 %shft.chk2.i.i, i32 %70, i32 0
  %72 = lshr i32 %68, %31
  %73 = select i1 %shft.chk4.i.i, i32 %72, i32 0
  %74 = or i32 %71, %73
  %75 = xor i32 %74, %69
  %76 = add i32 %75, %69
  %77 = shl i32 %75, %32
  %78 = select i1 %shft.chk5.i.i, i32 %77, i32 0
  %79 = lshr i32 %75, %33
  %80 = select i1 %shft.chk7.i.i, i32 %79, i32 0
  %81 = or i32 %78, %80
  %82 = xor i32 %81, %76
  %83 = add i32 %76, %34
  %84 = add i32 %83, %82
  store i32 %84, ptr %17, align 4, !alias.scope !44, !noalias !45
  %85 = load i32, ptr %7, align 4, !alias.scope !24, !noalias !26
  %shft.chk17.i.i = icmp ult i32 %85, 32
  %86 = sub i32 32, %85
  %shft.chk19.i.i = icmp ult i32 %86, 32
  %87 = load i32, ptr %arg25, align 64, !alias.scope !48, !noalias !49
  %88 = load i32, ptr %arg36, align 64, !alias.scope !50, !noalias !51
  %89 = add i32 %87, 1
  %90 = add i32 %89, %88
  %91 = add i32 %57, %51
  %92 = shl i32 %57, %85
  %93 = select i1 %shft.chk17.i.i, i32 %92, i32 0
  %94 = lshr i32 %57, %86
  %95 = select i1 %shft.chk19.i.i, i32 %94, i32 0
  %96 = or i32 %93, %95
  %97 = xor i32 %96, %91
  %98 = add i32 %90, %97
  store i32 %98, ptr %arg19, align 64, !alias.scope !53, !noalias !54
  %99 = add i32 %82, %76
  %100 = shl i32 %82, %85
  %101 = select i1 %shft.chk17.i.i, i32 %100, i32 0
  %102 = lshr i32 %82, %86
  %103 = select i1 %shft.chk19.i.i, i32 %102, i32 0
  %104 = or i32 %101, %103
  %105 = xor i32 %104, %99
  %106 = add i32 %90, %105
  store i32 %106, ptr %18, align 4, !alias.scope !53, !noalias !54
  %107 = add i32 %88, 1
  store i32 %107, ptr %arg31, align 64, !alias.scope !9, !noalias !55
  store ptr %arg31, ptr %arg20, align 64, !alias.scope !56, !noalias !57
  store ptr %arg32, ptr %8, align 8, !alias.scope !56, !noalias !57
  store ptr %arg19, ptr %9, align 16, !alias.scope !56, !noalias !57
  store ptr %arg34, ptr %10, align 8, !alias.scope !56, !noalias !57
  store ptr %arg23, ptr %11, align 32, !alias.scope !56, !noalias !57
  store ptr %arg33, ptr %12, align 8, !alias.scope !56, !noalias !57
  store ptr %arg28, ptr %13, align 16, !alias.scope !56, !noalias !57
  store ptr %arg30, ptr %14, align 8, !alias.scope !56, !noalias !57
  %108 = icmp slt i32 %107, 5
  %109 = zext i1 %108 to i8
  store i8 %109, ptr %arg29, align 64, !alias.scope !19, !noalias !20
  br i1 %108, label %while.6.body.i, label %return

return:                                           ; preds = %while.6.body.i, %1
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!xla_cpu_memory_region_name = !{!0, !1}
!llvm.module.flags = !{!2}

!0 = !{!"xla_cpu_emitter__computation_kernel_emitter__hlo_opcode__call"}
!1 = !{!"ir_emitter"}
!2 = !{i32 1, !"xla_dylib_index", i64 0}
!3 = !{}
!4 = !{i64 8}
!5 = !{i64 64}
!6 = !{i64 16}
!7 = !{i64 4}
!8 = !{i64 1}
!9 = !{!10}
!10 = !{!"buffer: {index:7, offset:768, size:4}", !11}
!11 = !{!"XLA global AA domain"}
!12 = !{!13, !14, !15, !17}
!13 = !{!"buffer: {index:6, offset:0, size:4}", !11}
!14 = !{!"buffer: {index:7, offset:64, size:1}", !11}
!15 = distinct !{!15, !16, !"while.6__1: %buffer_table"}
!16 = distinct !{!16, !"while.6__1"}
!17 = distinct !{!17, !18, !"while.5_computation: %buffer_table"}
!18 = distinct !{!18, !"while.5_computation"}
!19 = !{!14}
!20 = !{!13, !10, !15, !17}
!21 = !{!22, !17}
!22 = distinct !{!22, !23, !"while.6: %buffer_table"}
!23 = distinct !{!23, !"while.6"}
!24 = !{!25}
!25 = !{!"buffer: {index:7, offset:64, size:16}", !11}
!26 = !{!27, !28, !29, !30, !31, !32, !33, !34, !35, !22, !17}
!27 = !{!"buffer: {index:1, offset:0, size:16}", !11}
!28 = !{!"buffer: {index:7, offset:192, size:16}", !11}
!29 = !{!"buffer: {index:7, offset:256, size:8}", !11}
!30 = !{!"buffer: {index:7, offset:320, size:8}", !11}
!31 = !{!"buffer: {index:7, offset:384, size:8}", !11}
!32 = !{!"buffer: {index:7, offset:448, size:8}", !11}
!33 = !{!"buffer: {index:7, offset:512, size:4}", !11}
!34 = !{!"buffer: {index:7, offset:576, size:4}", !11}
!35 = !{!"buffer: {index:7, offset:640, size:4}", !11}
!36 = !{!35}
!37 = !{!25, !29, !30, !31, !38, !39, !22, !17}
!38 = !{!"buffer: {index:7, offset:832, size:4}", !11}
!39 = !{!"buffer: {index:7, offset:960, size:4}", !11}
!40 = !{!30}
!41 = !{!25, !29, !31, !32, !33, !34, !35, !22, !17}
!42 = !{!29}
!43 = !{!25, !30, !31, !32, !33, !34, !35, !22, !17}
!44 = !{!31}
!45 = !{!27, !46, !25, !28, !29, !30, !32, !35, !10, !38, !47, !39, !22, !17}
!46 = !{!"buffer: {index:7, offset:0, size:64}", !11}
!47 = !{!"buffer: {index:7, offset:896, size:4}", !11}
!48 = !{!33}
!49 = !{!25, !29, !30, !32, !34, !38, !47, !22, !17}
!50 = !{!34}
!51 = !{!52, !25, !29, !30, !32, !33, !10, !22, !17}
!52 = !{!"buffer: {index:0, offset:0, size:4}", !11}
!53 = !{!32}
!54 = !{!27, !46, !25, !28, !29, !30, !31, !33, !34, !10, !38, !47, !39, !22, !17}
!55 = !{!52, !27, !46, !28, !31, !32, !34, !38, !47, !39, !22, !17}
!56 = !{!46}
!57 = !{!27, !28, !31, !32, !10, !38, !47, !39, !22, !17}
