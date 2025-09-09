module @broadcast_add_fusion.1_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @broadcast_add_fusion.1(%arg0: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<4xi32> {llvm.align = 64 : index, llvm.dereferenceable = 16 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<16x8xi32> {llvm.align = 64 : index, llvm.dereferenceable = 512 : index, xla.invariant, xla.slice_index = 2 : index}, %arg3: tensor<16x8xi32> {llvm.align = 64 : index, llvm.dereferenceable = 512 : index, xla.invariant, xla.slice_index = 3 : index}, %arg4: tensor<16x8xi32> {llvm.align = 64 : index, llvm.dereferenceable = 512 : index, xla.slice_index = 4 : index}) -> tensor<16x8xi32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %0 = xla.workgroup_id  x {xla.range = [0 : index, 0 : index]}
    %1 = xla.workgroup_id  y {xla.range = [0 : index, 0 : index]}
    %2 = xla.workgroup_id  z {xla.range = [0 : index, 0 : index]}
    %3 = scf.forall (%arg5, %arg6, %arg7) in (1, 1, 1) shared_outs(%arg8 = %arg4) -> (tensor<16x8xi32>) {
      %xla_loop = xla.loop (%arg5, %arg6, %arg7, %0, %1, %2)[%i, %j] -> (%ra, %rb) in #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z)[s0, s1] -> (s0, s1), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0], s0 in [0, 15], s1 in [0, 7]"> iter_args(%iter = %arg8) -> (tensor<16x8xi32>) {
        %pure_call = xla.pure_call @fused_computation_1_add_78(%arg0, %arg1, %arg2, %arg3, %ra, %rb) : (tensor<i32>, tensor<4xi32>, tensor<16x8xi32>, tensor<16x8xi32>, index, index) -> i32
        %inserted = tensor.insert %pure_call into %iter[%ra, %rb] : tensor<16x8xi32>
        xla.yield %inserted : tensor<16x8xi32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %xla_loop into %arg8[0, 0] [16, 8] [1, 1] : tensor<16x8xi32> into tensor<16x8xi32>
      }
    }
    return %3 : tensor<16x8xi32>
  }
  func.func private @fused_computation_1_add_78(%arg0: tensor<i32>, %arg1: tensor<4xi32>, %arg2: tensor<16x8xi32>, %arg3: tensor<16x8xi32>, %arg4: index {xla.range = [0 : index, 15 : index]}, %arg5: index {xla.range = [0 : index, 7 : index]}) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = xla.apply_indexing #xla.indexing_map<"() -> (0)">
    %pure_call = xla.pure_call @fused_computation_1_param_1_11(%arg0, %arg1, %arg2, %arg3, %0) : (tensor<i32>, tensor<4xi32>, tensor<16x8xi32>, tensor<16x8xi32>, index) -> i32
    %c32_i32 = arith.constant 32 : i32
    %1 = arith.subi %c32_i32, %pure_call : i32
    %extracted = tensor.extract %arg2[%arg4, %arg5] : tensor<16x8xi32>
    %c0_i32 = arith.constant 0 : i32
    %2 = arith.shli %extracted, %pure_call : i32
    %c32_i32_0 = arith.constant 32 : i32
    %3 = arith.cmpi ugt, %c32_i32_0, %pure_call : i32
    %4 = arith.select %3, %2, %c0_i32 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %5 = arith.shrui %extracted, %1 : i32
    %c32_i32_2 = arith.constant 32 : i32
    %6 = arith.cmpi ugt, %c32_i32_2, %1 : i32
    %7 = arith.select %6, %5, %c0_i32_1 : i32
    %8 = xla.apply_indexing #xla.indexing_map<"() -> (0)">
    %9 = xla.apply_indexing #xla.indexing_map<"(d0) -> (d0 + 1), domain: d0 in [0, 0]">(%8)
    %pure_call_3 = xla.pure_call @fused_computation_1_param_1_11(%arg0, %arg1, %arg2, %arg3, %9) : (tensor<i32>, tensor<4xi32>, tensor<16x8xi32>, tensor<16x8xi32>, index) -> i32
    %10 = arith.subi %c32_i32, %pure_call_3 : i32
    %extracted_4 = tensor.extract %arg3[%arg4, %arg5] : tensor<16x8xi32>
    %extracted_5 = tensor.extract %arg2[%arg4, %arg5] : tensor<16x8xi32>
    %11 = arith.ori %4, %7 : i32
    %12 = arith.addi %extracted_4, %extracted_5 : i32
    %13 = arith.xori %12, %11 : i32
    %c0_i32_6 = arith.constant 0 : i32
    %14 = arith.shli %13, %pure_call_3 : i32
    %c32_i32_7 = arith.constant 32 : i32
    %15 = arith.cmpi ugt, %c32_i32_7, %pure_call_3 : i32
    %16 = arith.select %15, %14, %c0_i32_6 : i32
    %c0_i32_8 = arith.constant 0 : i32
    %17 = arith.shrui %13, %10 : i32
    %c32_i32_9 = arith.constant 32 : i32
    %18 = arith.cmpi ugt, %c32_i32_9, %10 : i32
    %19 = arith.select %18, %17, %c0_i32_8 : i32
    %20 = xla.apply_indexing #xla.indexing_map<"() -> (0)">
    %21 = xla.apply_indexing #xla.indexing_map<"(d0) -> (d0 + 2), domain: d0 in [0, 0]">(%20)
    %pure_call_10 = xla.pure_call @fused_computation_1_param_1_11(%arg0, %arg1, %arg2, %arg3, %21) : (tensor<i32>, tensor<4xi32>, tensor<16x8xi32>, tensor<16x8xi32>, index) -> i32
    %22 = arith.subi %c32_i32, %pure_call_10 : i32
    %23 = arith.addi %extracted_4, %extracted_5 : i32
    %24 = arith.xori %23, %11 : i32
    %25 = arith.ori %16, %19 : i32
    %26 = arith.addi %23, %24 : i32
    %27 = arith.xori %26, %25 : i32
    %c0_i32_11 = arith.constant 0 : i32
    %28 = arith.shli %27, %pure_call_10 : i32
    %c32_i32_12 = arith.constant 32 : i32
    %29 = arith.cmpi ugt, %c32_i32_12, %pure_call_10 : i32
    %30 = arith.select %29, %28, %c0_i32_11 : i32
    %c0_i32_13 = arith.constant 0 : i32
    %31 = arith.shrui %27, %22 : i32
    %c32_i32_14 = arith.constant 32 : i32
    %32 = arith.cmpi ugt, %c32_i32_14, %22 : i32
    %33 = arith.select %32, %31, %c0_i32_13 : i32
    %34 = arith.addi %23, %24 : i32
    %35 = arith.xori %34, %25 : i32
    %36 = arith.ori %30, %33 : i32
    %37 = arith.addi %34, %35 : i32
    %38 = arith.xori %37, %36 : i32
    %39 = arith.addi %37, %38 : i32
    %extracted_15 = tensor.extract %arg0[] : tensor<i32>
    %40 = arith.addi %39, %extracted_15 : i32
    return %40 : i32
  }
  func.func private @fused_computation_1_param_1_11(%arg0: tensor<i32>, %arg1: tensor<4xi32>, %arg2: tensor<16x8xi32>, %arg3: tensor<16x8xi32>, %arg4: index {xla.range = [0 : index, 3 : index]}) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>, no_compute = true} {
    %extracted = tensor.extract %arg1[%arg4] : tensor<4xi32>
    return %extracted : i32
  }
}