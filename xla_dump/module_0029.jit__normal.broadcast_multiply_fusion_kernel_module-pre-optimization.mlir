module @broadcast_multiply_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @broadcast_multiply_fusion(%arg0: tensor<12x8xi32> {llvm.align = 64 : index, llvm.dereferenceable = 384 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<12x8xi32> {llvm.align = 64 : index, llvm.dereferenceable = 384 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<12x8xf32> {llvm.align = 64 : index, llvm.dereferenceable = 384 : index, xla.slice_index = 2 : index}) -> tensor<12x8xf32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %0 = xla.workgroup_id  x {xla.range = [0 : index, 0 : index]}
    %1 = xla.workgroup_id  y {xla.range = [0 : index, 0 : index]}
    %2 = xla.workgroup_id  z {xla.range = [0 : index, 0 : index]}
    %3 = scf.forall (%arg3, %arg4, %arg5) in (1, 1, 1) shared_outs(%arg6 = %arg2) -> (tensor<12x8xf32>) {
      %xla_loop = xla.loop (%arg3, %arg4, %arg5, %0, %1, %2)[%i, %j] -> (%ra, %rb) in #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z)[s0, s1] -> (s0, s1), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0], s0 in [0, 11], s1 in [0, 7]"> iter_args(%iter = %arg6) -> (tensor<12x8xf32>) {
        %pure_call = xla.pure_call @fused_computation_2_mul_11(%arg0, %arg1, %ra, %rb) : (tensor<12x8xi32>, tensor<12x8xi32>, index, index) -> f32
        %inserted = tensor.insert %pure_call into %iter[%ra, %rb] : tensor<12x8xf32>
        xla.yield %inserted : tensor<12x8xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %xla_loop into %arg6[0, 0] [12, 8] [1, 1] : tensor<12x8xf32> into tensor<12x8xf32>
      }
    }
    return %3 : tensor<12x8xf32>
  }
  func.func private @fused_computation_2_mul_11(%arg0: tensor<12x8xi32>, %arg1: tensor<12x8xi32>, %arg2: index {xla.range = [0 : index, 11 : index]}, %arg3: index {xla.range = [0 : index, 7 : index]}) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 2.81022636E-8 : f32
    %cst_0 = arith.constant -2.00214257E-4 : f32
    %cst_1 = arith.constant 3.43273939E-7 : f32
    %cst_2 = arith.constant 1.00950558E-4 : f32
    %extracted = tensor.extract %arg0[%arg2, %arg3] : tensor<12x8xi32>
    %extracted_3 = tensor.extract %arg1[%arg2, %arg3] : tensor<12x8xi32>
    %0 = arith.xori %extracted, %extracted_3 : i32
    %c9_i32 = arith.constant 9 : i32
    %c0_i32 = arith.constant 0 : i32
    %1 = arith.shrui %0, %c9_i32 : i32
    %c32_i32 = arith.constant 32 : i32
    %2 = arith.cmpi ugt, %c32_i32, %c9_i32 : i32
    %3 = arith.select %2, %1, %c0_i32 : i32
    %c1065353216_i32 = arith.constant 1065353216 : i32
    %4 = arith.ori %3, %c1065353216_i32 : i32
    %5 = arith.bitcast %4 : i32 to f32
    %cst_4 = arith.constant -1.000000e+00 : f32
    %6 = arith.addf %5, %cst_4 : f32
    %cst_5 = arith.constant 2.000000e+00 : f32
    %7 = arith.mulf %6, %cst_5 : f32
    %cst_6 = arith.constant -0.99999994 : f32
    %8 = arith.addf %7, %cst_6 : f32
    %9 = arith.maximumf %cst_6, %8 : f32
    %10 = arith.negf %9 : f32
    %11 = arith.mulf %9, %10 : f32
    %12 = math.log1p %11 : f32
    %13 = arith.negf %12 : f32
    %cst_7 = arith.constant 5.000000e+00 : f32
    %14 = arith.cmpf olt, %13, %cst_7 : f32
    %15 = arith.extui %14 : i1 to i8
    %16 = arith.select %14, %cst, %cst_0 : f32
    %17 = arith.select %14, %cst_1, %cst_2 : f32
    %cst_8 = arith.constant -2.500000e+00 : f32
    %18 = math.sqrt %13 : f32
    %cst_9 = arith.constant -3.000000e+00 : f32
    %19 = arith.addf %13, %cst_8 : f32
    %20 = arith.addf %18, %cst_9 : f32
    %21 = arith.select %14, %19, %20 : f32
    %22 = arith.mulf %16, %21 : f32
    %cst_10 = arith.constant -3.5233877E-6 : f32
    %cst_11 = arith.constant 0.00134934322 : f32
    %23 = arith.addf %17, %22 : f32
    %24 = arith.select %14, %cst_10, %cst_11 : f32
    %25 = arith.mulf %23, %21 : f32
    %cst_12 = arith.constant -4.39150654E-6 : f32
    %cst_13 = arith.constant -0.00367342844 : f32
    %26 = arith.addf %24, %25 : f32
    %27 = arith.select %14, %cst_12, %cst_13 : f32
    %28 = arith.mulf %26, %21 : f32
    %cst_14 = arith.constant 2.1858087E-4 : f32
    %cst_15 = arith.constant 0.00573950773 : f32
    %29 = arith.addf %27, %28 : f32
    %30 = arith.select %14, %cst_14, %cst_15 : f32
    %31 = arith.mulf %29, %21 : f32
    %cst_16 = arith.constant -0.00125372503 : f32
    %cst_17 = arith.constant -0.0076224613 : f32
    %32 = arith.addf %30, %31 : f32
    %extracted_18 = tensor.extract %arg0[%arg2, %arg3] : tensor<12x8xi32>
    %extracted_19 = tensor.extract %arg1[%arg2, %arg3] : tensor<12x8xi32>
    %33 = arith.select %14, %cst_16, %cst_17 : f32
    %34 = arith.mulf %32, %21 : f32
    %35 = arith.xori %extracted_18, %extracted_19 : i32
    %36 = arith.negf %9 : f32
    %cst_20 = arith.constant -0.00417768164 : f32
    %cst_21 = arith.constant 0.00943887047 : f32
    %37 = arith.addf %33, %34 : f32
    %c0_i32_22 = arith.constant 0 : i32
    %38 = arith.shrui %35, %c9_i32 : i32
    %c32_i32_23 = arith.constant 32 : i32
    %39 = arith.cmpi ugt, %c32_i32_23, %c9_i32 : i32
    %40 = arith.select %39, %38, %c0_i32_22 : i32
    %41 = arith.mulf %9, %36 : f32
    %42 = arith.select %14, %cst_20, %cst_21 : f32
    %43 = arith.mulf %37, %21 : f32
    %44 = arith.ori %40, %c1065353216_i32 : i32
    %45 = math.log1p %41 : f32
    %cst_24 = arith.constant 0.246640727 : f32
    %cst_25 = arith.constant 1.00167406 : f32
    %46 = arith.addf %42, %43 : f32
    %47 = math.sqrt %13 : f32
    %48 = arith.bitcast %44 : i32 to f32
    %49 = arith.negf %45 : f32
    %50 = arith.select %14, %cst_24, %cst_25 : f32
    %51 = arith.mulf %46, %21 : f32
    %52 = arith.addf %49, %cst_8 : f32
    %53 = arith.addf %47, %cst_9 : f32
    %54 = arith.addf %48, %cst_4 : f32
    %55 = arith.cmpf olt, %49, %cst_7 : f32
    %56 = arith.extui %55 : i1 to i8
    %cst_26 = arith.constant 1.50140941 : f32
    %cst_27 = arith.constant 2.83297682 : f32
    %57 = arith.addf %50, %51 : f32
    %58 = arith.select %55, %52, %53 : f32
    %59 = arith.mulf %54, %cst_5 : f32
    %60 = arith.select %55, %cst_26, %cst_27 : f32
    %61 = arith.mulf %57, %58 : f32
    %62 = arith.addf %59, %cst_6 : f32
    %63 = math.absf %9 : f32
    %cst_28 = arith.constant 1.000000e+00 : f32
    %cst_29 = arith.constant 0x7F800000 : f32
    %64 = arith.addf %60, %61 : f32
    %65 = arith.maximumf %cst_6, %62 : f32
    %66 = arith.cmpf oeq, %63, %cst_28 : f32
    %67 = arith.extui %66 : i1 to i8
    %68 = arith.mulf %65, %cst_29 : f32
    %69 = arith.mulf %64, %65 : f32
    %70 = arith.select %66, %68, %69 : f32
    %cst_30 = arith.constant 1.41421354 : f32
    %71 = arith.mulf %70, %cst_30 : f32
    return %71 : f32
  }
}