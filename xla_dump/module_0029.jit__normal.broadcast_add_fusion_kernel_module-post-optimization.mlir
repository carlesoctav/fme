module @broadcast_add_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @broadcast_add_fusion(%arg0: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<4xi32> {llvm.align = 64 : index, llvm.dereferenceable = 16 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<96xi32> {llvm.align = 64 : index, llvm.dereferenceable = 384 : index, xla.invariant, xla.slice_index = 2 : index}, %arg3: tensor<96xi32> {llvm.align = 64 : index, llvm.dereferenceable = 384 : index, xla.invariant, xla.slice_index = 3 : index}, %arg4: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 4 : index}, %arg5: tensor<96xi32> {llvm.align = 64 : index, llvm.dereferenceable = 384 : index, xla.slice_index = 5 : index}) -> tensor<96xi32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c8 = arith.constant 8 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %arg1[%c0] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=5">]} : tensor<4xi32>
    %0 = arith.subi %c32_i32, %extracted : i32
    %1 = arith.cmpi ult, %extracted, %c32_i32 : i32
    %2 = arith.cmpi ult, %0, %c32_i32 : i32
    %extracted_0 = tensor.extract %arg1[%c1] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=5">]} : tensor<4xi32>
    %3 = arith.subi %c32_i32, %extracted_0 : i32
    %4 = arith.cmpi ult, %extracted_0, %c32_i32 : i32
    %5 = arith.cmpi ult, %3, %c32_i32 : i32
    %extracted_1 = tensor.extract %arg1[%c2] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=5">]} : tensor<4xi32>
    %6 = arith.subi %c32_i32, %extracted_1 : i32
    %7 = arith.cmpi ult, %extracted_1, %c32_i32 : i32
    %8 = arith.cmpi ult, %6, %c32_i32 : i32
    %extracted_2 = tensor.extract %arg1[%c3] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=5">]} : tensor<4xi32>
    %9 = arith.subi %c32_i32, %extracted_2 : i32
    %10 = arith.cmpi ult, %extracted_2, %c32_i32 : i32
    %11 = arith.cmpi ult, %9, %c32_i32 : i32
    %extracted_3 = tensor.extract %arg0[] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=5">]} : tensor<i32>
    %extracted_4 = tensor.extract %arg4[] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=5">]} : tensor<i32>
    %12 = arith.addi %extracted_4, %c1_i32 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %13 = scf.for %arg6 = %c0 to %c12 step %c1 iter_args(%arg7 = %arg5) -> (tensor<96xi32>) {
      %14 = scf.for %arg8 = %c0 to %c8 step %c1 iter_args(%arg9 = %arg7) -> (tensor<96xi32>) {
        %15 = xla.apply_indexing #xla.indexing_map<"(d0, d1) -> (d0 * 8 + d1), domain: d0 in [0, 11], d1 in [0, 7]">(%arg6, %arg8)
        %extracted_5 = tensor.extract %arg2[%15] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=5">]} : tensor<96xi32>
        %16 = arith.shli %extracted_5, %extracted : i32
        %17 = arith.select %1, %16, %c0_i32 : i32
        %18 = arith.shrui %extracted_5, %0 : i32
        %19 = arith.select %2, %18, %c0_i32 : i32
        %extracted_6 = tensor.extract %arg3[%15] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=5">]} : tensor<96xi32>
        %20 = arith.ori %17, %19 : i32
        %21 = arith.addi %extracted_6, %extracted_5 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
        %22 = arith.xori %21, %20 : i32
        %23 = arith.shli %22, %extracted_0 : i32
        %24 = arith.select %4, %23, %c0_i32 : i32
        %25 = arith.shrui %22, %3 : i32
        %26 = arith.select %5, %25, %c0_i32 : i32
        %27 = arith.ori %24, %26 : i32
        %28 = arith.addi %21, %22 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
        %29 = arith.xori %28, %27 : i32
        %30 = arith.shli %29, %extracted_1 : i32
        %31 = arith.select %7, %30, %c0_i32 : i32
        %32 = arith.shrui %29, %6 : i32
        %33 = arith.select %8, %32, %c0_i32 : i32
        %34 = arith.ori %31, %33 : i32
        %35 = arith.addi %28, %29 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
        %36 = arith.xori %35, %34 : i32
        %37 = arith.shli %36, %extracted_2 : i32
        %38 = arith.select %10, %37, %c0_i32 : i32
        %39 = arith.shrui %36, %9 : i32
        %40 = arith.select %11, %39, %c0_i32 : i32
        %41 = arith.addi %35, %36 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
        %42 = arith.ori %38, %40 : i32
        %43 = arith.xori %41, %42 : i32
        %44 = arith.addi %43, %extracted_3 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
        %45 = arith.addi %44, %12 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
        %inserted = tensor.insert %45 into %arg9[%15] {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=5">]} : tensor<96xi32>
        scf.yield %inserted : tensor<96xi32>
      }
      scf.yield %14 : tensor<96xi32>
    }
    return %13 : tensor<96xi32>
  }
}