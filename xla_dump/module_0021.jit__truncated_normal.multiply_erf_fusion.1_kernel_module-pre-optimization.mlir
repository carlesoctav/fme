module @multiply_erf_fusion.1_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @multiply_erf_fusion.1(%arg0: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<f32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.slice_index = 1 : index}) -> tensor<f32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %0 = xla.workgroup_id  x {xla.range = [0 : index, 0 : index]}
    %1 = xla.workgroup_id  y {xla.range = [0 : index, 0 : index]}
    %2 = xla.workgroup_id  z {xla.range = [0 : index, 0 : index]}
    %3 = scf.forall (%arg2, %arg3, %arg4) in (1, 1, 1) shared_outs(%arg5 = %arg1) -> (tensor<f32>) {
      %xla_loop = xla.loop (%arg2, %arg3, %arg4, %0, %1, %2)[] -> () in #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z) -> (), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0]"> iter_args(%iter = %arg5) -> (tensor<f32>) {
        %pure_call = xla.pure_call @fused_computation_9_erf_1(%arg0) : (tensor<i32>) -> f32
        %inserted = tensor.insert %pure_call into %iter[] : tensor<f32>
        xla.yield %inserted : tensor<f32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %xla_loop into %arg5[] [] [] : tensor<f32> into tensor<f32>
      }
    }
    return %3 : tensor<f32>
  }
  func.func private @fused_computation_9_erf_1(%arg0: tensor<i32>) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %extracted = tensor.extract %arg0[] : tensor<i32>
    %0 = arith.sitofp %extracted : i32 to f32
    %cst = arith.constant 0.707106769 : f32
    %1 = arith.mulf %0, %cst : f32
    %2 = math.erf %1 : f32
    return %2 : f32
  }
}