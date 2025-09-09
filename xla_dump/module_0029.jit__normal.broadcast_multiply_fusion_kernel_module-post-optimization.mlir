module @broadcast_multiply_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @broadcast_multiply_fusion(%arg0: tensor<96xi32> {llvm.align = 64 : index, llvm.dereferenceable = 384 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<96xi32> {llvm.align = 64 : index, llvm.dereferenceable = 384 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<96xf32> {llvm.align = 64 : index, llvm.dereferenceable = 384 : index, xla.slice_index = 2 : index}) -> tensor<96xf32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c8 = arith.constant 8 : index
    %c12 = arith.constant 12 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
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
    %cst_19 = arith.constant -0.99999994 : f32
    %cst_20 = arith.constant 2.000000e+00 : f32
    %cst_21 = arith.constant -1.000000e+00 : f32
    %c1065353216_i32 = arith.constant 1065353216 : i32
    %c9_i32 = arith.constant 9 : i32
    %cst_22 = arith.constant 2.81022636E-8 : f32
    %cst_23 = arith.constant -2.00214257E-4 : f32
    %cst_24 = arith.constant 3.43273939E-7 : f32
    %cst_25 = arith.constant 1.00950558E-4 : f32
    %0 = scf.for %arg3 = %c0 to %c12 step %c1 iter_args(%arg4 = %arg2) -> (tensor<96xf32>) {
      %1 = scf.for %arg5 = %c0 to %c8 step %c1 iter_args(%arg6 = %arg4) -> (tensor<96xf32>) {
        %2 = xla.apply_indexing #xla.indexing_map<"(d0, d1) -> (d0 * 8 + d1), domain: d0 in [0, 11], d1 in [0, 7]">(%arg3, %arg5)
        %extracted = tensor.extract %arg0[%2] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=2">]} : tensor<96xi32>
        %extracted_26 = tensor.extract %arg1[%2] {llvm.noalias = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=2">]} : tensor<96xi32>
        %3 = arith.xori %extracted, %extracted_26 : i32
        %4 = arith.shrui %3, %c9_i32 : i32
        %5 = arith.ori %4, %c1065353216_i32 : i32
        %6 = arith.bitcast %5 : i32 to f32
        %7 = arith.addf %6, %cst_21 : f32
        %8 = arith.mulf %7, %cst_20 : f32
        %9 = arith.addf %8, %cst_19 : f32
        %10 = arith.maximumf %9, %cst_19 : f32
        %11 = arith.negf %10 : f32
        %12 = arith.mulf %10, %11 : f32
        %13 = math.log1p %12 : f32
        %14 = arith.negf %13 : f32
        %15 = arith.cmpf olt, %14, %cst_18 : f32
        %16 = arith.select %15, %cst_22, %cst_23 : f32
        %17 = arith.select %15, %cst_24, %cst_25 : f32
        %18 = math.sqrt %14 : f32
        %19 = arith.addf %14, %cst_17 : f32
        %20 = arith.addf %18, %cst_16 : f32
        %21 = arith.select %15, %19, %20 : f32
        %22 = arith.mulf %16, %21 : f32
        %23 = arith.addf %17, %22 : f32
        %24 = arith.select %15, %cst_15, %cst_14 : f32
        %25 = arith.mulf %23, %21 : f32
        %26 = arith.addf %24, %25 : f32
        %27 = arith.select %15, %cst_13, %cst_12 : f32
        %28 = arith.mulf %26, %21 : f32
        %29 = arith.addf %27, %28 : f32
        %30 = arith.select %15, %cst_11, %cst_10 : f32
        %31 = arith.mulf %29, %21 : f32
        %32 = arith.addf %30, %31 : f32
        %33 = arith.select %15, %cst_9, %cst_8 : f32
        %34 = arith.mulf %32, %21 : f32
        %35 = arith.addf %33, %34 : f32
        %36 = arith.select %15, %cst_7, %cst_6 : f32
        %37 = arith.mulf %35, %21 : f32
        %38 = arith.addf %36, %37 : f32
        %39 = arith.select %15, %cst_5, %cst_4 : f32
        %40 = arith.mulf %38, %21 : f32
        %41 = arith.addf %39, %40 : f32
        %42 = arith.select %15, %cst_3, %cst_2 : f32
        %43 = arith.mulf %41, %21 : f32
        %44 = math.absf %10 : f32
        %45 = arith.addf %42, %43 : f32
        %46 = arith.cmpf oeq, %44, %cst_1 : f32
        %47 = arith.mulf %10, %cst_0 : f32
        %48 = arith.mulf %45, %10 : f32
        %49 = arith.select %46, %47, %48 : f32
        %50 = arith.mulf %49, %cst : f32
        %inserted = tensor.insert %50 into %arg6[%2] {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=2">]} : tensor<96xf32>
        scf.yield %inserted : tensor<96xf32>
      }
      scf.yield %1 : tensor<96xf32>
    }
    return %0 : tensor<96xf32>
  }
}