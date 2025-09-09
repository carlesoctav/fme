module @slice_bitcast_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @slice_bitcast_fusion(%arg0: tensor<4xi32> {llvm.align = 64 : index, llvm.dereferenceable = 16 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<2xi32> {llvm.align = 64 : index, llvm.dereferenceable = 8 : index, xla.slice_index = 1 : index}) -> tensor<2xi32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %arg1) -> (tensor<2xi32>) {
      %1 = xla.apply_indexing #xla.indexing_map<"(d0) -> (d0 + 2), domain: d0 in [0, 1]">(%arg2)
      %extracted = tensor.extract %arg0[%1] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=1">]} : tensor<4xi32>
      %inserted = tensor.insert %extracted into %arg3[%arg2] {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=1">]} : tensor<2xi32>
      scf.yield %inserted : tensor<2xi32>
    }
    return %0 : tensor<2xi32>
  }
}