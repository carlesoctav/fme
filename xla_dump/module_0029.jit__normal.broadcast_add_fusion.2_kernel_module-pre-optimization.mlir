module @broadcast_add_fusion.2_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @broadcast_add_fusion.2(%arg0: tensor<2xi32> {llvm.align = 64 : index, llvm.dereferenceable = 8 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<12x8xi32> {llvm.align = 64 : index, llvm.dereferenceable = 384 : index, xla.slice_index = 1 : index}) -> tensor<12x8xi32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %0 = xla.workgroup_id  x {xla.range = [0 : index, 0 : index]}
    %1 = xla.workgroup_id  y {xla.range = [0 : index, 0 : index]}
    %2 = xla.workgroup_id  z {xla.range = [0 : index, 0 : index]}
    %3 = scf.forall (%arg2, %arg3, %arg4) in (1, 1, 1) shared_outs(%arg5 = %arg1) -> (tensor<12x8xi32>) {
      %xla_loop = xla.loop (%arg2, %arg3, %arg4, %0, %1, %2)[%i, %j] -> (%ra, %rb) in #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z)[s0, s1] -> (s0, s1), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0], s0 in [0, 11], s1 in [0, 7]"> iter_args(%iter = %arg5) -> (tensor<12x8xi32>) {
        %pure_call = xla.pure_call @fused_computation_4_add_100(%arg0, %ra, %rb) : (tensor<2xi32>, index, index) -> i32
        %inserted = tensor.insert %pure_call into %iter[%ra, %rb] : tensor<12x8xi32>
        xla.yield %inserted : tensor<12x8xi32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %xla_loop into %arg5[0, 0] [12, 8] [1, 1] : tensor<12x8xi32> into tensor<12x8xi32>
      }
    }
    return %3 : tensor<12x8xi32>
  }
  func.func private @fused_computation_4_add_100(%arg0: tensor<2xi32>, %arg1: index {xla.range = [0 : index, 11 : index]}, %arg2: index {xla.range = [0 : index, 7 : index]}) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = arith.index_castui %arg1 : index to i64
    %c8_i64 = arith.constant 8 : i64
    %1 = arith.muli %0, %c8_i64 : i64
    %2 = arith.index_castui %arg2 : index to i64
    %3 = arith.addi %1, %2 : i64
    %4 = arith.trunci %3 : i64 to i32
    %5 = xla.apply_indexing #xla.indexing_map<"() -> (0)">
    %6 = xla.apply_indexing #xla.indexing_map<"(d0) -> (d0 + 1), domain: d0 in [0, 0]">(%5)
    %extracted = tensor.extract %arg0[%6] : tensor<2xi32>
    %7 = arith.addi %4, %extracted : i32
    return %7 : i32
  }
}