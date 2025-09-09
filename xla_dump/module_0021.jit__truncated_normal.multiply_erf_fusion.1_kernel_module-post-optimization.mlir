module @multiply_erf_fusion.1_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @multiply_erf_fusion.1(%arg0: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<f32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.slice_index = 1 : index}) -> tensor<f32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %cst = arith.constant 0.707106769 : f32
    %extracted = tensor.extract %arg0[] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=1">]} : tensor<i32>
    %0 = arith.sitofp %extracted : i32 to f32
    %1 = arith.mulf %0, %cst : f32
    %2 = math.erf %1 : f32
    %inserted = tensor.insert %2 into %arg1[] {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=1">]} : tensor<f32>
    return %inserted : tensor<f32>
  }
}