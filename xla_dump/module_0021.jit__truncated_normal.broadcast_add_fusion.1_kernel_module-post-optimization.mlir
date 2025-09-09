module @broadcast_add_fusion.1_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @broadcast_add_fusion.1(%arg0: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<4xi32> {llvm.align = 64 : index, llvm.dereferenceable = 16 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<512xi32> {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, xla.invariant, xla.slice_index = 2 : index}, %arg3: tensor<512xi32> {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, xla.invariant, xla.slice_index = 3 : index}, %arg4: tensor<512xi32> {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, xla.slice_index = 4 : index}) -> tensor<512xi32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %arg1[%c0] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=4">]} : tensor<4xi32>
    %0 = arith.subi %c32_i32, %extracted : i32
    %1 = arith.cmpi ult, %extracted, %c32_i32 : i32
    %2 = arith.cmpi ult, %0, %c32_i32 : i32
    %extracted_0 = tensor.extract %arg1[%c1] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=4">]} : tensor<4xi32>
    %3 = arith.subi %c32_i32, %extracted_0 : i32
    %4 = arith.cmpi ult, %extracted_0, %c32_i32 : i32
    %5 = arith.cmpi ult, %3, %c32_i32 : i32
    %extracted_1 = tensor.extract %arg1[%c2] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=4">]} : tensor<4xi32>
    %6 = arith.subi %c32_i32, %extracted_1 : i32
    %7 = arith.cmpi ult, %extracted_1, %c32_i32 : i32
    %8 = arith.cmpi ult, %6, %c32_i32 : i32
    %extracted_2 = tensor.extract %arg0[] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=4">]} : tensor<i32>
    %9 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %arg4) -> (tensor<512xi32>) {
      %10 = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%arg8 = %arg6) -> (tensor<512xi32>) {
        %11 = xla.apply_indexing #xla.indexing_map<"(d0, d1) -> (d0 * 16 + d1), domain: d0 in [0, 31], d1 in [0, 15]">(%arg5, %arg7)
        %extracted_3 = tensor.extract %arg2[%11] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=4">]} : tensor<512xi32>
        %12 = arith.shli %extracted_3, %extracted : i32
        %13 = arith.select %1, %12, %c0_i32 : i32
        %14 = arith.shrui %extracted_3, %0 : i32
        %15 = arith.select %2, %14, %c0_i32 : i32
        %extracted_4 = tensor.extract %arg3[%11] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=4">]} : tensor<512xi32>
        %16 = arith.ori %13, %15 : i32
        %17 = arith.addi %extracted_4, %extracted_3 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
        %18 = arith.xori %17, %16 : i32
        %19 = arith.shli %18, %extracted_0 : i32
        %20 = arith.select %4, %19, %c0_i32 : i32
        %21 = arith.shrui %18, %3 : i32
        %22 = arith.select %5, %21, %c0_i32 : i32
        %23 = arith.ori %20, %22 : i32
        %24 = arith.addi %17, %18 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
        %25 = arith.xori %24, %23 : i32
        %26 = arith.shli %25, %extracted_1 : i32
        %27 = arith.select %7, %26, %c0_i32 : i32
        %28 = arith.shrui %25, %6 : i32
        %29 = arith.select %8, %28, %c0_i32 : i32
        %30 = arith.ori %27, %29 : i32
        %31 = arith.addi %24, %25 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
        %32 = arith.xori %31, %30 : i32
        %33 = arith.addi %31, %32 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
        %34 = arith.addi %33, %extracted_2 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
        %inserted = tensor.insert %34 into %arg8[%11] {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=4">]} : tensor<512xi32>
        scf.yield %inserted : tensor<512xi32>
      }
      scf.yield %10 : tensor<512xi32>
    }
    return %9 : tensor<512xi32>
  }
}