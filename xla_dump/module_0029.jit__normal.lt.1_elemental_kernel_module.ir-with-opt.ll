; ModuleID = '__compute_module_lt.1_elemental_kernel_module'
source_filename = "__compute_module_lt.1_elemental_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @lt.1_kernel(ptr readonly captures(none) %0) local_unnamed_addr #0 {
return:
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %arg2_gep = getelementptr i8, ptr %args, i64 32
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %1 = load i32, ptr %arg0, align 64, !invariant.load !3, !noalias !7
  %2 = load i32, ptr %arg1, align 64, !invariant.load !3, !noalias !7
  %3 = icmp slt i32 %1, %2
  %4 = zext i1 %3 to i8
  store i8 %4, ptr %arg2, align 64, !alias.scope !7
  ret ptr null
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!xla_cpu_memory_region_name = !{!0, !1}
!llvm.module.flags = !{!2}

!0 = !{!"xla_cpu_emitter__elemental_kernel_emitter__hlo_opcode__compare"}
!1 = !{!"ir_emitter"}
!2 = !{i32 1, !"xla_dylib_index", i64 0}
!3 = !{}
!4 = !{i64 4}
!5 = !{i64 64}
!6 = !{i64 1}
!7 = !{!8}
!8 = !{!"result slice: {index:0, offset:0, size:1}", !9}
!9 = !{!"XLA host kernel lt.1_kernel AA domain"}
