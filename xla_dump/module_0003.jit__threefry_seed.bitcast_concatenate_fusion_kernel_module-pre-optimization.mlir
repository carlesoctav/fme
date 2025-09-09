module @bitcast_concatenate_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__concatenate_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @bitcast_concatenate_fusion(%arg0: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<2xi32> {llvm.align = 64 : index, llvm.dereferenceable = 8 : index, xla.slice_index = 1 : index}) -> tensor<2xi32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %0 = xla.workgroup_id  x {xla.range = [0 : index, 0 : index]}
    %1 = xla.workgroup_id  y {xla.range = [0 : index, 0 : index]}
    %2 = xla.workgroup_id  z {xla.range = [0 : index, 0 : index]}
    %3 = scf.forall (%arg2, %arg3, %arg4) in (1, 1, 1) shared_outs(%arg5 = %arg1) -> (tensor<2xi32>) {
      %xla_loop = xla.loop (%arg2, %arg3, %arg4, %0, %1, %2)[] -> (%ra) in #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z) -> (0), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0]"> iter_args(%iter = %arg1) -> (tensor<2xi32>) {
        %4 = xla.apply_indexing #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z) -> (0), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0]">(%arg2, %arg3, %arg4, %0, %1, %2)
        %pure_call = xla.pure_call @fused_computation_bitcast_3(%arg0, %4) : (tensor<i32>, index) -> i32
        %pure_call_1 = xla.pure_call @fused_computation__epilogue__concatenate_0(%arg0, %ra, %pure_call) : (tensor<i32>, index, i32) -> i32
        %inserted = tensor.insert %pure_call_1 into %iter[%ra] : tensor<2xi32>
        xla.yield %inserted : tensor<2xi32>
      }
      %xla_loop_0 = xla.loop (%arg2, %arg3, %arg4, %0, %1, %2)[] -> (%ra) in #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z) -> (1), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0]"> iter_args(%iter = %xla_loop) -> (tensor<2xi32>) {
        %4 = xla.apply_indexing #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z) -> (0), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0]">(%arg2, %arg3, %arg4, %0, %1, %2)
        %pure_call = xla.pure_call @fused_computation_bitcast_2(%arg0, %4) : (tensor<i32>, index) -> i32
        %pure_call_1 = xla.pure_call @fused_computation__epilogue__concatenate_0(%arg0, %ra, %pure_call) : (tensor<i32>, index, i32) -> i32
        %inserted = tensor.insert %pure_call_1 into %iter[%ra] : tensor<2xi32>
        xla.yield %inserted : tensor<2xi32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %xla_loop_0 into %arg5[0] [2] [1] : tensor<2xi32> into tensor<2xi32>
      }
    }
    return %3 : tensor<2xi32>
  }
  func.func private @fused_computation_bitcast_2(%arg0: tensor<i32>, %arg1: index {xla.range = [0 : index, 0 : index]}) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %pure_call = xla.pure_call @fused_computation_param_0_1(%arg0) : (tensor<i32>) -> i32
    return %pure_call : i32
  }
  func.func private @fused_computation_bitcast_3(%arg0: tensor<i32>, %arg1: index {xla.range = [0 : index, 0 : index]}) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c32_i32 = arith.constant 32 : i32
    %pure_call = xla.pure_call @fused_computation_param_0_1(%arg0) : (tensor<i32>) -> i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.shrui %pure_call, %c32_i32 : i32
    %c32_i32_0 = arith.constant 32 : i32
    %1 = arith.cmpi ugt, %c32_i32_0, %c32_i32 : i32
    %2 = arith.select %1, %0, %c0_i32 : i32
    return %2 : i32
  }
  func.func private @fused_computation_param_0_1(%arg0: tensor<i32>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, no_compute = true} {
    %extracted = tensor.extract %arg0[] : tensor<i32>
    return %extracted : i32
  }
  func.func private @fused_computation__epilogue__concatenate_0(%arg0: tensor<i32>, %arg1: index {xla.range = [0 : index, 1 : index]}, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, no_compute = true} {
    return %arg2 : i32
  }
}