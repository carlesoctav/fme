; ModuleID = '__compute_module_add.65_elemental_kernel_module'
source_filename = "__compute_module_add.65_elemental_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @add.65_kernel(ptr readonly captures(none) %0) local_unnamed_addr #0 {
return:
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %arg2_gep = getelementptr i8, ptr %args, i64 32
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %1 = load i32, ptr %arg0, align 64, !invariant.load !3, !noalias !6
  %2 = load i32, ptr %arg1, align 64, !invariant.load !3, !noalias !6
  %3 = add i32 %2, %1
  store i32 %3, ptr %arg2, align 64, !alias.scope !6
  ret ptr null
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!xla_cpu_memory_region_name = !{!0, !1}
!llvm.module.flags = !{!2}

!0 = !{!"xla_cpu_emitter__elemental_kernel_emitter__hlo_opcode__add"}
!1 = !{!"ir_emitter"}
!2 = !{i32 1, !"xla_dylib_index", i64 1}
!3 = !{}
!4 = !{i64 4}
!5 = !{i64 64}
!6 = !{!7}
!7 = !{!"result slice: {index:7, offset:1600, size:4}", !8}
!8 = !{!"XLA host kernel add.65_kernel AA domain"}
