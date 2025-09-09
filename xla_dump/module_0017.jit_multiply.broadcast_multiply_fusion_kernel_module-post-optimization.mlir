module @broadcast_multiply_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @broadcast_multiply_fusion(%arg0: tensor<128xf32> {llvm.align = 64 : index, llvm.dereferenceable = 512 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<f32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<128xf32> {llvm.align = 64 : index, llvm.dereferenceable = 512 : index, xla.slice_index = 2 : index}) -> tensor<128xf32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %extracted = tensor.extract %arg1[] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=2">]} : tensor<f32>
    %0 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %arg2) -> (tensor<128xf32>) {
      %1 = scf.for %arg5 = %c0 to %c8 step %c1 iter_args(%arg6 = %arg4) -> (tensor<128xf32>) {
        %2 = xla.apply_indexing #xla.indexing_map<"(d0, d1) -> (d0 * 8 + d1), domain: d0 in [0, 15], d1 in [0, 7]">(%arg3, %arg5)
        %extracted_0 = tensor.extract %arg0[%2] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=2">]} : tensor<128xf32>
        %3 = arith.mulf %extracted_0, %extracted : f32
        %inserted = tensor.insert %3 into %arg6[%2] {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=2">]} : tensor<128xf32>
        scf.yield %inserted : tensor<128xf32>
      }
      scf.yield %1 : tensor<128xf32>
    }
    return %0 : tensor<128xf32>
  }
}