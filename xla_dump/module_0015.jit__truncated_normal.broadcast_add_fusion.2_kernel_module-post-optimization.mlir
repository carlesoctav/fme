module @broadcast_add_fusion.2_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @broadcast_add_fusion.2(%arg0: tensor<2xi32> {llvm.align = 64 : index, llvm.dereferenceable = 8 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<128xi32> {llvm.align = 64 : index, llvm.dereferenceable = 512 : index, xla.slice_index = 1 : index}) -> tensor<128xi32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c8_i64 = arith.constant 8 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %extracted = tensor.extract %arg0[%c1] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=1">]} : tensor<2xi32>
    %0 = scf.for %arg2 = %c0 to %c16 step %c1 iter_args(%arg3 = %arg1) -> (tensor<128xi32>) {
      %1 = arith.index_castui %arg2 : index to i64
      %2 = arith.muli %1, %c8_i64 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i64
      %3 = scf.for %arg4 = %c0 to %c8 step %c1 iter_args(%arg5 = %arg3) -> (tensor<128xi32>) {
        %4 = arith.index_castui %arg4 : index to i64
        %5 = arith.addi %2, %4 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i64
        %6 = arith.trunci %5 : i64 to i32
        %7 = arith.addi %6, %extracted {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
        %8 = xla.apply_indexing #xla.indexing_map<"(d0, d1) -> (d0 * 8 + d1), domain: d0 in [0, 15], d1 in [0, 7]">(%arg2, %arg4)
        %inserted = tensor.insert %7 into %arg5[%8] {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=1">]} : tensor<128xi32>
        scf.yield %inserted : tensor<128xi32>
      }
      scf.yield %3 : tensor<128xi32>
    }
    return %0 : tensor<128xi32>
  }
}