module @slice_bitcast_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @slice_bitcast_fusion(%arg0: tensor<2xi32> {llvm.align = 64 : index, llvm.dereferenceable = 8 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.slice_index = 1 : index}) -> tensor<i32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %0 = xla.workgroup_id  x {xla.range = [0 : index, 0 : index]}
    %1 = xla.workgroup_id  y {xla.range = [0 : index, 0 : index]}
    %2 = xla.workgroup_id  z {xla.range = [0 : index, 0 : index]}
    %3 = scf.forall (%arg2, %arg3, %arg4) in (1, 1, 1) shared_outs(%arg5 = %arg1) -> (tensor<i32>) {
      %xla_loop = xla.loop (%arg2, %arg3, %arg4, %0, %1, %2)[] -> () in #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z) -> (), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0]"> iter_args(%iter = %arg5) -> (tensor<i32>) {
        %pure_call = xla.pure_call @fused_computation_5_bitcast_20(%arg0) : (tensor<2xi32>) -> i32
        %inserted = tensor.insert %pure_call into %iter[] : tensor<i32>
        xla.yield %inserted : tensor<i32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %xla_loop into %arg5[] [] [] : tensor<i32> into tensor<i32>
      }
    }
    return %3 : tensor<i32>
  }
  func.func private @fused_computation_5_bitcast_20(%arg0: tensor<2xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, no_compute = true} {
    %0 = xla.apply_indexing #xla.indexing_map<"() -> (0)">
    %1 = xla.apply_indexing #xla.indexing_map<"(d0) -> (d0 + 1), domain: d0 in [0, 0]">(%0)
    %extracted = tensor.extract %arg0[%1] : tensor<2xi32>
    return %extracted : i32
  }
}