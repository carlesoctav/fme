; ModuleID = '__compute_module_broadcast.1_elemental_kernel_module'
source_filename = "__compute_module_broadcast.1_elemental_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @broadcast.1_kernel(ptr readonly captures(none) %0) local_unnamed_addr #0 {
broadcast.1.loop_body.dim.0:
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %1 = load float, ptr %arg0, align 64, !invariant.load !3, !noalias !7
  store float %1, ptr %arg1, align 64, !alias.scope !7
  %2 = getelementptr inbounds nuw i8, ptr %arg1, i64 4
  store float %1, ptr %2, align 4, !alias.scope !7
  %3 = getelementptr inbounds nuw i8, ptr %arg1, i64 8
  store float %1, ptr %3, align 8, !alias.scope !7
  %4 = getelementptr inbounds nuw i8, ptr %arg1, i64 12
  store float %1, ptr %4, align 4, !alias.scope !7
  %5 = getelementptr inbounds nuw i8, ptr %arg1, i64 16
  store float %1, ptr %5, align 16, !alias.scope !7
  %6 = getelementptr inbounds nuw i8, ptr %arg1, i64 20
  store float %1, ptr %6, align 4, !alias.scope !7
  %7 = getelementptr inbounds nuw i8, ptr %arg1, i64 24
  store float %1, ptr %7, align 8, !alias.scope !7
  %8 = getelementptr inbounds nuw i8, ptr %arg1, i64 28
  store float %1, ptr %8, align 4, !alias.scope !7
  %9 = getelementptr inbounds nuw i8, ptr %arg1, i64 32
  store float %1, ptr %9, align 32, !alias.scope !7
  %10 = getelementptr inbounds nuw i8, ptr %arg1, i64 36
  store float %1, ptr %10, align 4, !alias.scope !7
  %11 = getelementptr inbounds nuw i8, ptr %arg1, i64 40
  store float %1, ptr %11, align 8, !alias.scope !7
  %12 = getelementptr inbounds nuw i8, ptr %arg1, i64 44
  store float %1, ptr %12, align 4, !alias.scope !7
  %13 = getelementptr inbounds nuw i8, ptr %arg1, i64 48
  store float %1, ptr %13, align 16, !alias.scope !7
  %14 = getelementptr inbounds nuw i8, ptr %arg1, i64 52
  store float %1, ptr %14, align 4, !alias.scope !7
  %15 = getelementptr inbounds nuw i8, ptr %arg1, i64 56
  store float %1, ptr %15, align 8, !alias.scope !7
  %16 = getelementptr inbounds nuw i8, ptr %arg1, i64 60
  store float %1, ptr %16, align 4, !alias.scope !7
  %17 = getelementptr inbounds nuw i8, ptr %arg1, i64 64
  store float %1, ptr %17, align 64, !alias.scope !7
  %18 = getelementptr inbounds nuw i8, ptr %arg1, i64 68
  store float %1, ptr %18, align 4, !alias.scope !7
  %19 = getelementptr inbounds nuw i8, ptr %arg1, i64 72
  store float %1, ptr %19, align 8, !alias.scope !7
  %20 = getelementptr inbounds nuw i8, ptr %arg1, i64 76
  store float %1, ptr %20, align 4, !alias.scope !7
  %21 = getelementptr inbounds nuw i8, ptr %arg1, i64 80
  store float %1, ptr %21, align 16, !alias.scope !7
  %22 = getelementptr inbounds nuw i8, ptr %arg1, i64 84
  store float %1, ptr %22, align 4, !alias.scope !7
  %23 = getelementptr inbounds nuw i8, ptr %arg1, i64 88
  store float %1, ptr %23, align 8, !alias.scope !7
  %24 = getelementptr inbounds nuw i8, ptr %arg1, i64 92
  store float %1, ptr %24, align 4, !alias.scope !7
  %25 = getelementptr inbounds nuw i8, ptr %arg1, i64 96
  store float %1, ptr %25, align 32, !alias.scope !7
  %26 = getelementptr inbounds nuw i8, ptr %arg1, i64 100
  store float %1, ptr %26, align 4, !alias.scope !7
  %27 = getelementptr inbounds nuw i8, ptr %arg1, i64 104
  store float %1, ptr %27, align 8, !alias.scope !7
  %28 = getelementptr inbounds nuw i8, ptr %arg1, i64 108
  store float %1, ptr %28, align 4, !alias.scope !7
  %29 = getelementptr inbounds nuw i8, ptr %arg1, i64 112
  store float %1, ptr %29, align 16, !alias.scope !7
  %30 = getelementptr inbounds nuw i8, ptr %arg1, i64 116
  store float %1, ptr %30, align 4, !alias.scope !7
  %31 = getelementptr inbounds nuw i8, ptr %arg1, i64 120
  store float %1, ptr %31, align 8, !alias.scope !7
  %32 = getelementptr inbounds nuw i8, ptr %arg1, i64 124
  store float %1, ptr %32, align 4, !alias.scope !7
  ret ptr null
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!xla_cpu_memory_region_name = !{!0, !1}
!llvm.module.flags = !{!2}

!0 = !{!"xla_cpu_emitter__elemental_kernel_emitter__hlo_opcode__broadcast"}
!1 = !{!"ir_emitter"}
!2 = !{i32 1, !"xla_dylib_index", i64 1}
!3 = !{}
!4 = !{i64 4}
!5 = !{i64 64}
!6 = !{i64 128}
!7 = !{!8}
!8 = !{!"result slice: {index:4, offset:0, size:128}", !9}
!9 = !{!"XLA host kernel broadcast.1_kernel AA domain"}
