; ModuleID = '__compute_module_broadcast.0_elemental_kernel_module'
source_filename = "__compute_module_broadcast.0_elemental_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @broadcast.0_kernel(ptr readonly captures(none) %0) local_unnamed_addr #0 {
broadcast.0.loop_body.dim.0:
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !3, !dereferenceable !5, !align !5
  %1 = load float, ptr %arg0, align 64, !invariant.load !3, !noalias !6
  store float %1, ptr %arg1, align 64, !alias.scope !6
  %2 = getelementptr inbounds nuw i8, ptr %arg1, i64 4
  store float %1, ptr %2, align 4, !alias.scope !6
  %3 = getelementptr inbounds nuw i8, ptr %arg1, i64 8
  store float %1, ptr %3, align 8, !alias.scope !6
  %4 = getelementptr inbounds nuw i8, ptr %arg1, i64 12
  store float %1, ptr %4, align 4, !alias.scope !6
  %5 = getelementptr inbounds nuw i8, ptr %arg1, i64 16
  store float %1, ptr %5, align 16, !alias.scope !6
  %6 = getelementptr inbounds nuw i8, ptr %arg1, i64 20
  store float %1, ptr %6, align 4, !alias.scope !6
  %7 = getelementptr inbounds nuw i8, ptr %arg1, i64 24
  store float %1, ptr %7, align 8, !alias.scope !6
  %8 = getelementptr inbounds nuw i8, ptr %arg1, i64 28
  store float %1, ptr %8, align 4, !alias.scope !6
  %9 = getelementptr inbounds nuw i8, ptr %arg1, i64 32
  store float %1, ptr %9, align 32, !alias.scope !6
  %10 = getelementptr inbounds nuw i8, ptr %arg1, i64 36
  store float %1, ptr %10, align 4, !alias.scope !6
  %11 = getelementptr inbounds nuw i8, ptr %arg1, i64 40
  store float %1, ptr %11, align 8, !alias.scope !6
  %12 = getelementptr inbounds nuw i8, ptr %arg1, i64 44
  store float %1, ptr %12, align 4, !alias.scope !6
  %13 = getelementptr inbounds nuw i8, ptr %arg1, i64 48
  store float %1, ptr %13, align 16, !alias.scope !6
  %14 = getelementptr inbounds nuw i8, ptr %arg1, i64 52
  store float %1, ptr %14, align 4, !alias.scope !6
  %15 = getelementptr inbounds nuw i8, ptr %arg1, i64 56
  store float %1, ptr %15, align 8, !alias.scope !6
  %16 = getelementptr inbounds nuw i8, ptr %arg1, i64 60
  store float %1, ptr %16, align 4, !alias.scope !6
  ret ptr null
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!xla_cpu_memory_region_name = !{!0, !1}
!llvm.module.flags = !{!2}

!0 = !{!"xla_cpu_emitter__elemental_kernel_emitter__hlo_opcode__broadcast"}
!1 = !{!"ir_emitter"}
!2 = !{i32 1, !"xla_dylib_index", i64 0}
!3 = !{}
!4 = !{i64 4}
!5 = !{i64 64}
!6 = !{!7}
!7 = !{!"result slice: {index:5, offset:0, size:64}", !8}
!8 = !{!"XLA host kernel broadcast.0_kernel AA domain"}
