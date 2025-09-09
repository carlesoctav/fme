module @maximum_minimum_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @maximum_minimum_fusion(%arg0: tensor<f32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<f32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<512xi32> {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, xla.invariant, xla.slice_index = 2 : index}, %arg3: tensor<512xi32> {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, xla.invariant, xla.slice_index = 3 : index}, %arg4: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 4 : index}, %arg5: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 5 : index}, %arg6: tensor<512xf32> {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, xla.slice_index = 6 : index}) -> tensor<512xf32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c-2147483647_i32 = arith.constant -2147483647 : i32
    %c2143289344_i32 = arith.constant 2143289344 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c2139095040_i32 = arith.constant 2139095040 : i32
    %c2147483647_i32 = arith.constant 2147483647 : i32
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %cst = arith.constant 1.41421354 : f32
    %cst_0 = arith.constant 0x7F800000 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 2.83297682 : f32
    %cst_3 = arith.constant 1.50140941 : f32
    %cst_4 = arith.constant 1.00167406 : f32
    %cst_5 = arith.constant 0.246640727 : f32
    %cst_6 = arith.constant 0.00943887047 : f32
    %cst_7 = arith.constant -0.00417768164 : f32
    %cst_8 = arith.constant -0.0076224613 : f32
    %cst_9 = arith.constant -0.00125372503 : f32
    %cst_10 = arith.constant 0.00573950773 : f32
    %cst_11 = arith.constant 2.1858087E-4 : f32
    %cst_12 = arith.constant -0.00367342844 : f32
    %cst_13 = arith.constant -4.39150654E-6 : f32
    %cst_14 = arith.constant 0.00134934322 : f32
    %cst_15 = arith.constant -3.5233877E-6 : f32
    %cst_16 = arith.constant -3.000000e+00 : f32
    %cst_17 = arith.constant -2.500000e+00 : f32
    %cst_18 = arith.constant 5.000000e+00 : f32
    %cst_19 = arith.constant -1.000000e+00 : f32
    %c1065353216_i32 = arith.constant 1065353216 : i32
    %c0_i32 = arith.constant 0 : i32
    %c9_i32 = arith.constant 9 : i32
    %cst_20 = arith.constant 2.81022636E-8 : f32
    %cst_21 = arith.constant -2.00214257E-4 : f32
    %cst_22 = arith.constant 3.43273939E-7 : f32
    %cst_23 = arith.constant 1.00950558E-4 : f32
    %extracted = tensor.extract %arg0[] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=6">]} : tensor<f32>
    %extracted_24 = tensor.extract %arg1[] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=6">]} : tensor<f32>
    %0 = arith.subf %extracted, %extracted_24 : f32
    %extracted_25 = tensor.extract %arg4[] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=6">]} : tensor<i32>
    %1 = arith.sitofp %extracted_25 : i32 to f32
    %2 = arith.bitcast %1 : f32 to i32
    %3 = arith.andi %2, %c-2147483648_i32 : i32
    %4 = arith.andi %2, %c2147483647_i32 : i32
    %5 = arith.cmpi sgt, %4, %c2139095040_i32 : i32
    %6 = arith.cmpi ne, %3, %c0_i32 : i32
    %7 = arith.ori %5, %6 : i1
    %8 = arith.select %7, %c-1_i32, %c1_i32 : i32
    %9 = arith.cmpi eq, %4, %c0_i32 : i32
    %10 = arith.addi %2, %8 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %11 = arith.select %9, %c1_i32, %10 : i32
    %12 = arith.cmpf une, %1, %1 : f32
    %13 = arith.select %12, %c2143289344_i32, %11 : i32
    %14 = arith.bitcast %13 : i32 to f32
    %extracted_26 = tensor.extract %arg5[] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=6">]} : tensor<i32>
    %15 = arith.sitofp %extracted_26 : i32 to f32
    %16 = arith.bitcast %15 : f32 to i32
    %17 = arith.andi %16, %c-2147483648_i32 : i32
    %18 = arith.andi %16, %c2147483647_i32 : i32
    %19 = arith.cmpi sgt, %18, %c2139095040_i32 : i32
    %20 = arith.cmpi ne, %17, %c-2147483648_i32 : i32
    %21 = arith.ori %19, %20 : i1
    %22 = arith.select %21, %c-1_i32, %c1_i32 : i32
    %23 = arith.cmpi eq, %18, %c0_i32 : i32
    %24 = arith.addi %16, %22 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %25 = arith.select %23, %c-2147483647_i32, %24 : i32
    %26 = arith.cmpf une, %15, %15 : f32
    %27 = arith.select %26, %c2143289344_i32, %25 : i32
    %28 = arith.bitcast %27 : i32 to f32
    %29 = scf.for %arg7 = %c0 to %c32 step %c1 iter_args(%arg8 = %arg6) -> (tensor<512xf32>) {
      %30 = scf.for %arg9 = %c0 to %c16 step %c1 iter_args(%arg10 = %arg8) -> (tensor<512xf32>) {
        %31 = xla.apply_indexing #xla.indexing_map<"(d0, d1) -> (d0 * 16 + d1), domain: d0 in [0, 31], d1 in [0, 15]">(%arg7, %arg9)
        %extracted_27 = tensor.extract %arg2[%31] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=6">]} : tensor<512xi32>
        %extracted_28 = tensor.extract %arg3[%31] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=6">]} : tensor<512xi32>
        %32 = arith.xori %extracted_27, %extracted_28 : i32
        %33 = arith.shrui %32, %c9_i32 : i32
        %34 = arith.ori %33, %c1065353216_i32 : i32
        %35 = arith.bitcast %34 : i32 to f32
        %36 = arith.addf %35, %cst_19 : f32
        %37 = arith.mulf %36, %0 : f32
        %38 = arith.addf %37, %extracted_24 : f32
        %39 = arith.maximumf %extracted_24, %38 : f32
        %40 = arith.negf %39 : f32
        %41 = arith.mulf %39, %40 : f32
        %42 = math.log1p %41 : f32
        %43 = arith.negf %42 : f32
        %44 = arith.cmpf olt, %43, %cst_18 : f32
        %45 = arith.select %44, %cst_20, %cst_21 : f32
        %46 = arith.select %44, %cst_22, %cst_23 : f32
        %47 = math.sqrt %43 : f32
        %48 = arith.addf %43, %cst_17 : f32
        %49 = arith.addf %47, %cst_16 : f32
        %50 = arith.select %44, %48, %49 : f32
        %51 = arith.mulf %45, %50 : f32
        %52 = arith.addf %46, %51 : f32
        %53 = arith.select %44, %cst_15, %cst_14 : f32
        %54 = arith.mulf %52, %50 : f32
        %55 = arith.addf %53, %54 : f32
        %56 = arith.select %44, %cst_13, %cst_12 : f32
        %57 = arith.mulf %55, %50 : f32
        %58 = arith.addf %56, %57 : f32
        %59 = arith.select %44, %cst_11, %cst_10 : f32
        %60 = arith.mulf %58, %50 : f32
        %61 = arith.addf %59, %60 : f32
        %62 = arith.select %44, %cst_9, %cst_8 : f32
        %63 = arith.mulf %61, %50 : f32
        %64 = arith.addf %62, %63 : f32
        %65 = arith.select %44, %cst_7, %cst_6 : f32
        %66 = arith.mulf %64, %50 : f32
        %67 = arith.addf %65, %66 : f32
        %68 = arith.select %44, %cst_5, %cst_4 : f32
        %69 = arith.mulf %67, %50 : f32
        %70 = arith.addf %68, %69 : f32
        %71 = arith.select %44, %cst_3, %cst_2 : f32
        %72 = arith.mulf %70, %50 : f32
        %73 = math.absf %39 : f32
        %74 = arith.addf %71, %72 : f32
        %75 = arith.cmpf oeq, %73, %cst_1 : f32
        %76 = arith.mulf %39, %cst_0 : f32
        %77 = arith.mulf %74, %39 : f32
        %78 = arith.select %75, %76, %77 : f32
        %79 = arith.mulf %78, %cst : f32
        %80 = arith.maximumf %14, %79 : f32
        %81 = arith.minimumf %28, %80 : f32
        %inserted = tensor.insert %81 into %arg10[%31] {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=6">]} : tensor<512xf32>
        scf.yield %inserted : tensor<512xf32>
      }
      scf.yield %30 : tensor<512xf32>
    }
    return %29 : tensor<512xf32>
  }
}