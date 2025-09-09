module @broadcast_multiply_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @broadcast_multiply_fusion(%arg0: tensor<16x8xf32> {llvm.align = 64 : index, llvm.dereferenceable = 512 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<f32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<16x8xf32> {llvm.align = 64 : index, llvm.dereferenceable = 512 : index, xla.slice_index = 2 : index}) -> tensor<16x8xf32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %0 = xla.workgroup_id  x {xla.range = [0 : index, 0 : index]}
    %1 = xla.workgroup_id  y {xla.range = [0 : index, 0 : index]}
    %2 = xla.workgroup_id  z {xla.range = [0 : index, 0 : index]}
    %3 = scf.forall (%arg3, %arg4, %arg5) in (1, 1, 1) shared_outs(%arg6 = %arg2) -> (tensor<16x8xf32>) {
      %xla_loop = xla.loop (%arg3, %arg4, %arg5, %0, %1, %2)[%i, %j] -> (%ra, %rb) in #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z)[s0, s1] -> (s0, s1), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0], s0 in [0, 15], s1 in [0, 7]"> iter_args(%iter = %arg6) -> (tensor<16x8xf32>) {
        %pure_call = xla.pure_call @fused_computation_mul_0(%arg0, %arg1, %ra, %rb) : (tensor<16x8xf32>, tensor<f32>, index, index) -> f32
        %inserted = tensor.insert %pure_call into %iter[%ra, %rb] : tensor<16x8xf32>
        xla.yield %inserted : tensor<16x8xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %xla_loop into %arg6[0, 0] [16, 8] [1, 1] : tensor<16x8xf32> into tensor<16x8xf32>
      }
    }
    return %3 : tensor<16x8xf32>
  }
  func.func private @fused_computation_mul_0(%arg0: tensor<16x8xf32>, %arg1: tensor<f32>, %arg2: index {xla.range = [0 : index, 15 : index]}, %arg3: index {xla.range = [0 : index, 7 : index]}) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %extracted = tensor.extract %arg0[%arg2, %arg3] : tensor<16x8xf32>
    %extracted_0 = tensor.extract %arg1[] : tensor<f32>
    %0 = arith.mulf %extracted, %extracted_0 : f32
    return %0 : f32
  }
}