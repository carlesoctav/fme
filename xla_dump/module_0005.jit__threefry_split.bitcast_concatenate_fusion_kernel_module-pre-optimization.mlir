module @bitcast_concatenate_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__concatenate_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @bitcast_concatenate_fusion(%arg0: tensor<2xi32> {llvm.align = 64 : index, llvm.dereferenceable = 8 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<2xi32> {llvm.align = 64 : index, llvm.dereferenceable = 8 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<2x2xi32> {llvm.align = 64 : index, llvm.dereferenceable = 16 : index, xla.slice_index = 2 : index}) -> tensor<2x2xi32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %0 = xla.workgroup_id  x {xla.range = [0 : index, 0 : index]}
    %1 = xla.workgroup_id  y {xla.range = [0 : index, 0 : index]}
    %2 = xla.workgroup_id  z {xla.range = [0 : index, 0 : index]}
    %3 = scf.forall (%arg3, %arg4, %arg5) in (1, 1, 1) shared_outs(%arg6 = %arg2) -> (tensor<2x2xi32>) {
      %xla_loop = xla.loop (%arg3, %arg4, %arg5, %0, %1, %2)[%i] -> (%ra, %rb) in #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z)[s0] -> (s0, 0), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0], s0 in [0, 1]"> iter_args(%iter = %arg2) -> (tensor<2x2xi32>) {
        %4 = xla.apply_indexing #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z)[s0] -> (0), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0], s0 in [0, 1]">(%arg3, %arg4, %arg5, %0, %1, %2)[%i]
        %pure_call = xla.pure_call @fused_computation_2_bitcast_16(%arg0, %arg1, %i, %4) : (tensor<2xi32>, tensor<2xi32>, index, index) -> i32
        %pure_call_1 = xla.pure_call @fused_computation_2__epilogue__concatenate_0(%arg0, %arg1, %ra, %rb, %pure_call) : (tensor<2xi32>, tensor<2xi32>, index, index, i32) -> i32
        %inserted = tensor.insert %pure_call_1 into %iter[%ra, %rb] : tensor<2x2xi32>
        xla.yield %inserted : tensor<2x2xi32>
      }
      %xla_loop_0 = xla.loop (%arg3, %arg4, %arg5, %0, %1, %2)[%i] -> (%ra, %rb) in #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z)[s0] -> (s0, 1), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0], s0 in [0, 1]"> iter_args(%iter = %xla_loop) -> (tensor<2x2xi32>) {
        %4 = xla.apply_indexing #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z)[s0] -> (0), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0], s0 in [0, 1]">(%arg3, %arg4, %arg5, %0, %1, %2)[%i]
        %pure_call = xla.pure_call @fused_computation_2_bitcast_15(%arg0, %arg1, %i, %4) : (tensor<2xi32>, tensor<2xi32>, index, index) -> i32
        %pure_call_1 = xla.pure_call @fused_computation_2__epilogue__concatenate_0(%arg0, %arg1, %ra, %rb, %pure_call) : (tensor<2xi32>, tensor<2xi32>, index, index, i32) -> i32
        %inserted = tensor.insert %pure_call_1 into %iter[%ra, %rb] : tensor<2x2xi32>
        xla.yield %inserted : tensor<2x2xi32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %xla_loop_0 into %arg6[0, 0] [2, 2] [1, 1] : tensor<2x2xi32> into tensor<2x2xi32>
      }
    }
    return %3 : tensor<2x2xi32>
  }
  func.func private @fused_computation_2_bitcast_15(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>, %arg2: index {xla.range = [0 : index, 1 : index]}, %arg3: index {xla.range = [0 : index, 0 : index]}) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, no_compute = true} {
    %extracted = tensor.extract %arg0[%arg2] : tensor<2xi32>
    return %extracted : i32
  }
  func.func private @fused_computation_2_bitcast_16(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>, %arg2: index {xla.range = [0 : index, 1 : index]}, %arg3: index {xla.range = [0 : index, 0 : index]}) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, no_compute = true} {
    %extracted = tensor.extract %arg1[%arg2] : tensor<2xi32>
    return %extracted : i32
  }
  func.func private @fused_computation_2__epilogue__concatenate_0(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>, %arg2: index {xla.range = [0 : index, 1 : index]}, %arg3: index {xla.range = [0 : index, 1 : index]}, %arg4: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, no_compute = true} {
    return %arg4 : i32
  }
}