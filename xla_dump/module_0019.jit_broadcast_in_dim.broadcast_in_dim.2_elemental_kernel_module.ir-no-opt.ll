; ModuleID = '__compute_module_broadcast_in_dim.2_elemental_kernel_module'
source_filename = "__compute_module_broadcast_in_dim.2_elemental_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_NumWorkGroups = type { i64, i64, i64 }
%XLA_CPU_WorkGroupId = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

; Function Attrs: uwtable
define ptr @broadcast_in_dim.2_kernel(ptr %0) #0 {
  %broadcast_in_dim.2.invar_address.dim.0 = alloca i64, align 8
  %num_workgroups_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 0
  %num_workgroups = load ptr, ptr %num_workgroups_gep, align 8
  %num_workgroups_x_gep = getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, ptr %num_workgroups, i32 0, i32 0
  %num_workgroups_y_gep = getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, ptr %num_workgroups, i32 0, i32 1
  %num_workgroups_z_gep = getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, ptr %num_workgroups, i32 0, i32 2
  %num_workgroups_x = load i64, ptr %num_workgroups_x_gep, align 4
  %num_workgroups_y = load i64, ptr %num_workgroups_y_gep, align 4
  %num_workgroups_z = load i64, ptr %num_workgroups_z_gep, align 4
  %workgroup_id_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %workgroup_id = load ptr, ptr %workgroup_id_gep, align 8
  %workgroup_id_x_gep = getelementptr inbounds nuw %XLA_CPU_WorkGroupId, ptr %workgroup_id, i32 0, i32 0
  %workgroup_id_y_gep = getelementptr inbounds nuw %XLA_CPU_WorkGroupId, ptr %workgroup_id, i32 0, i32 1
  %workgroup_id_z_gep = getelementptr inbounds nuw %XLA_CPU_WorkGroupId, ptr %workgroup_id, i32 0, i32 2
  %workgroup_id_x = load i64, ptr %workgroup_id_x_gep, align 4
  %workgroup_id_y = load i64, ptr %workgroup_id_y_gep, align 4
  %workgroup_id_z = load i64, ptr %workgroup_id_z_gep, align 4
  %args_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args = load ptr, ptr %args_gep, align 8
  %arg0_gep = getelementptr %XLA_CPU_KernelArg, ptr %args, i32 0, i32 0
  %arg0 = load ptr, ptr %arg0_gep, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %args_gep1 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args2 = load ptr, ptr %args_gep1, align 8
  %arg1_gep = getelementptr %XLA_CPU_KernelArg, ptr %args2, i32 1, i32 0
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !3, !dereferenceable !5, !align !5
  store i64 0, ptr %broadcast_in_dim.2.invar_address.dim.0, align 4
  br label %broadcast_in_dim.2.loop_header.dim.0

broadcast_in_dim.2.loop_header.dim.0:             ; preds = %broadcast_in_dim.2.loop_body.dim.0, %1
  %broadcast_in_dim.2.indvar.dim.0 = load i64, ptr %broadcast_in_dim.2.invar_address.dim.0, align 4
  %2 = icmp uge i64 %broadcast_in_dim.2.indvar.dim.0, 16
  br i1 %2, label %broadcast_in_dim.2.loop_exit.dim.0, label %broadcast_in_dim.2.loop_body.dim.0

broadcast_in_dim.2.loop_body.dim.0:               ; preds = %broadcast_in_dim.2.loop_header.dim.0
  %3 = load float, ptr %arg0, align 4, !invariant.load !3, !noalias !6
  %4 = getelementptr inbounds [16 x float], ptr %arg1, i64 0, i64 %broadcast_in_dim.2.indvar.dim.0
  store float %3, ptr %4, align 4, !alias.scope !6
  %invar.inc = add nuw nsw i64 %broadcast_in_dim.2.indvar.dim.0, 1
  store i64 %invar.inc, ptr %broadcast_in_dim.2.invar_address.dim.0, align 4
  br label %broadcast_in_dim.2.loop_header.dim.0

broadcast_in_dim.2.loop_exit.dim.0:               ; preds = %broadcast_in_dim.2.loop_header.dim.0
  br label %return

return:                                           ; preds = %broadcast_in_dim.2.loop_exit.dim.0
  ret ptr null
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!xla_cpu_memory_region_name = !{!0, !1}
!llvm.module.flags = !{!2}

!0 = !{!"xla_cpu_emitter__elemental_kernel_emitter__hlo_opcode__broadcast"}
!1 = !{!"ir_emitter"}
!2 = !{i32 1, !"xla_dylib_index", i64 0}
!3 = !{}
!4 = !{i64 4}
!5 = !{i64 64}
!6 = !{!7}
!7 = !{!"result slice: {index:0, offset:0, size:64}", !8}
!8 = !{!"XLA host kernel broadcast_in_dim.2_kernel AA domain"}
