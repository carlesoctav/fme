module @slice_bitcast_fusion.1_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @slice_bitcast_fusion.1(%arg0: tensor<2xi32> {llvm.align = 64 : index, llvm.dereferenceable = 8 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.slice_index = 1 : index}) -> tensor<i32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %arg0[%c0] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=1">]} : tensor<2xi32>
    %inserted = tensor.insert %extracted into %arg1[] {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=1">]} : tensor<i32>
    return %inserted : tensor<i32>
  }
}