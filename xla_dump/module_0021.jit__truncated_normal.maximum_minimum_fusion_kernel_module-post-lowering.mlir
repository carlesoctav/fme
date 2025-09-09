module @maximum_minimum_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @xla.log1p.f32(f32) -> f32 attributes {sym_visibility = "private"}
  llvm.func @maximum_minimum_fusion(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr inbounds %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %2 = llvm.load %1 invariant : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %4 = llvm.load %3 invariant dereferenceable<bytes = 4> : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %2[1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %6 = llvm.load %5 invariant dereferenceable<bytes = 4> : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %2[2, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %8 = llvm.load %7 invariant dereferenceable<bytes = 2048> : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %2[3, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %10 = llvm.load %9 invariant dereferenceable<bytes = 2048> : !llvm.ptr -> !llvm.ptr
    %11 = llvm.getelementptr inbounds %2[4, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %12 = llvm.load %11 invariant dereferenceable<bytes = 4> : !llvm.ptr -> !llvm.ptr
    %13 = llvm.getelementptr inbounds %2[5, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %14 = llvm.load %13 invariant dereferenceable<bytes = 4> : !llvm.ptr -> !llvm.ptr
    %15 = llvm.getelementptr inbounds %2[6, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %16 = llvm.load %15 invariant dereferenceable<bytes = 2048> : !llvm.ptr -> !llvm.ptr
    %17 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %18 = llvm.load %17 : !llvm.ptr -> !llvm.ptr
    %19 = llvm.getelementptr inbounds %18[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %20 = llvm.load %19 invariant : !llvm.ptr -> i64
    %21 = llvm.getelementptr inbounds %18[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %22 = llvm.load %21 invariant : !llvm.ptr -> i64
    %23 = llvm.getelementptr inbounds %18[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %24 = llvm.load %23 invariant : !llvm.ptr -> i64
    llvm.call @maximum_minimum_fusion_wrapped(%4, %6, %8, %10, %12, %14, %16, %20, %22, %24) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return %0 : !llvm.ptr
  }
  llvm.func internal @maximum_minimum_fusion_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias, xla.invariant}, %arg2: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, llvm.noalias, xla.invariant}, %arg3: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, llvm.noalias, xla.invariant}, %arg4: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias, xla.invariant}, %arg5: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias, xla.invariant}, %arg6: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, llvm.noalias}, %arg7: i64, %arg8: i64, %arg9: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(1.00950558E-4 : f32) : f32
    %1 = llvm.mlir.constant(3.43273939E-7 : f32) : f32
    %2 = llvm.mlir.constant(-2.00214257E-4 : f32) : f32
    %3 = llvm.mlir.constant(2.81022636E-8 : f32) : f32
    %4 = llvm.mlir.constant(9 : i32) : i32
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(1065353216 : i32) : i32
    %7 = llvm.mlir.constant(-1.000000e+00 : f32) : f32
    %8 = llvm.mlir.constant(5.000000e+00 : f32) : f32
    %9 = llvm.mlir.constant(-2.500000e+00 : f32) : f32
    %10 = llvm.mlir.constant(-3.000000e+00 : f32) : f32
    %11 = llvm.mlir.constant(-3.5233877E-6 : f32) : f32
    %12 = llvm.mlir.constant(0.00134934322 : f32) : f32
    %13 = llvm.mlir.constant(-4.39150654E-6 : f32) : f32
    %14 = llvm.mlir.constant(-0.00367342844 : f32) : f32
    %15 = llvm.mlir.constant(2.1858087E-4 : f32) : f32
    %16 = llvm.mlir.constant(0.00573950773 : f32) : f32
    %17 = llvm.mlir.constant(-0.00125372503 : f32) : f32
    %18 = llvm.mlir.constant(-0.0076224613 : f32) : f32
    %19 = llvm.mlir.constant(-0.00417768164 : f32) : f32
    %20 = llvm.mlir.constant(0.00943887047 : f32) : f32
    %21 = llvm.mlir.constant(0.246640727 : f32) : f32
    %22 = llvm.mlir.constant(1.00167406 : f32) : f32
    %23 = llvm.mlir.constant(1.50140941 : f32) : f32
    %24 = llvm.mlir.constant(2.83297682 : f32) : f32
    %25 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %26 = llvm.mlir.constant(0x7F800000 : f32) : f32
    %27 = llvm.mlir.constant(1.41421354 : f32) : f32
    %28 = llvm.mlir.constant(-2147483648 : i32) : i32
    %29 = llvm.mlir.constant(2147483647 : i32) : i32
    %30 = llvm.mlir.constant(2139095040 : i32) : i32
    %31 = llvm.mlir.constant(-1 : i32) : i32
    %32 = llvm.mlir.constant(1 : i32) : i32
    %33 = llvm.mlir.constant(2143289344 : i32) : i32
    %34 = llvm.mlir.constant(-2147483647 : i32) : i32
    %35 = llvm.mlir.constant(1 : index) : i64
    %36 = llvm.mlir.constant(0 : index) : i64
    %37 = llvm.mlir.constant(32 : index) : i64
    %38 = llvm.mlir.constant(16 : index) : i64
    %39 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x f32>
    %40 = llvm.load %39 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=6">]} : !llvm.ptr -> f32
    %41 = llvm.getelementptr inbounds %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x f32>
    %42 = llvm.load %41 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=6">]} : !llvm.ptr -> f32
    %43 = llvm.fsub %40, %42 : f32
    %44 = llvm.getelementptr inbounds %arg4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %45 = llvm.load %44 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=6">]} : !llvm.ptr -> i32
    %46 = llvm.sitofp %45 : i32 to f32
    %47 = llvm.bitcast %46 : f32 to i32
    %48 = llvm.and %47, %28 : i32
    %49 = llvm.and %47, %29 : i32
    %50 = llvm.icmp "sgt" %49, %30 : i32
    %51 = llvm.icmp "ne" %48, %5 : i32
    %52 = llvm.or %50, %51 : i1
    %53 = llvm.select %52, %31, %32 : i1, i32
    %54 = llvm.icmp "eq" %49, %5 : i32
    %55 = llvm.add %47, %53 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %56 = llvm.select %54, %32, %55 : i1, i32
    %57 = llvm.fcmp "une" %46, %46 : f32
    %58 = llvm.select %57, %33, %56 : i1, i32
    %59 = llvm.bitcast %58 : i32 to f32
    %60 = llvm.getelementptr inbounds %arg5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %61 = llvm.load %60 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=6">]} : !llvm.ptr -> i32
    %62 = llvm.sitofp %61 : i32 to f32
    %63 = llvm.bitcast %62 : f32 to i32
    %64 = llvm.and %63, %28 : i32
    %65 = llvm.and %63, %29 : i32
    %66 = llvm.icmp "sgt" %65, %30 : i32
    %67 = llvm.icmp "ne" %64, %28 : i32
    %68 = llvm.or %66, %67 : i1
    %69 = llvm.select %68, %31, %32 : i1, i32
    %70 = llvm.icmp "eq" %65, %5 : i32
    %71 = llvm.add %63, %69 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %72 = llvm.select %70, %34, %71 : i1, i32
    %73 = llvm.fcmp "une" %62, %62 : f32
    %74 = llvm.select %73, %33, %72 : i1, i32
    %75 = llvm.bitcast %74 : i32 to f32
    llvm.br ^bb1(%36 : i64)
  ^bb1(%76: i64):  // 2 preds: ^bb0, ^bb5
    %77 = llvm.icmp "slt" %76, %37 : i64
    llvm.cond_br %77, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %78 = llvm.mul %76, %38 overflow<nsw> : i64
    llvm.br ^bb3(%36 : i64)
  ^bb3(%79: i64):  // 2 preds: ^bb2, ^bb4
    %80 = llvm.icmp "slt" %79, %38 : i64
    llvm.cond_br %80, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %81 = llvm.add %78, %79 overflow<nsw> : i64
    %82 = llvm.getelementptr inbounds %arg2[0, %81] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x i32>
    %83 = llvm.load %82 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=6">]} : !llvm.ptr -> i32
    %84 = llvm.getelementptr inbounds %arg3[0, %81] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x i32>
    %85 = llvm.load %84 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=6">]} : !llvm.ptr -> i32
    %86 = llvm.xor %83, %85 : i32
    %87 = llvm.lshr %86, %4 : i32
    %88 = llvm.or %87, %6 : i32
    %89 = llvm.bitcast %88 : i32 to f32
    %90 = llvm.fadd %89, %7 : f32
    %91 = llvm.fmul %90, %43 : f32
    %92 = llvm.fadd %91, %42 : f32
    %93 = llvm.intr.maximum(%42, %92) : (f32, f32) -> f32
    %94 = llvm.fneg %93 : f32
    %95 = llvm.fmul %93, %94 : f32
    %96 = llvm.call @xla.log1p.f32(%95) : (f32) -> f32
    %97 = llvm.fneg %96 : f32
    %98 = llvm.fcmp "olt" %97, %8 : f32
    %99 = llvm.select %98, %3, %2 : i1, f32
    %100 = llvm.select %98, %1, %0 : i1, f32
    %101 = llvm.intr.sqrt(%97) : (f32) -> f32
    %102 = llvm.fadd %97, %9 : f32
    %103 = llvm.fadd %101, %10 : f32
    %104 = llvm.select %98, %102, %103 : i1, f32
    %105 = llvm.fmul %99, %104 : f32
    %106 = llvm.fadd %100, %105 : f32
    %107 = llvm.select %98, %11, %12 : i1, f32
    %108 = llvm.fmul %106, %104 : f32
    %109 = llvm.fadd %107, %108 : f32
    %110 = llvm.select %98, %13, %14 : i1, f32
    %111 = llvm.fmul %109, %104 : f32
    %112 = llvm.fadd %110, %111 : f32
    %113 = llvm.select %98, %15, %16 : i1, f32
    %114 = llvm.fmul %112, %104 : f32
    %115 = llvm.fadd %113, %114 : f32
    %116 = llvm.select %98, %17, %18 : i1, f32
    %117 = llvm.fmul %115, %104 : f32
    %118 = llvm.fadd %116, %117 : f32
    %119 = llvm.select %98, %19, %20 : i1, f32
    %120 = llvm.fmul %118, %104 : f32
    %121 = llvm.fadd %119, %120 : f32
    %122 = llvm.select %98, %21, %22 : i1, f32
    %123 = llvm.fmul %121, %104 : f32
    %124 = llvm.fadd %122, %123 : f32
    %125 = llvm.select %98, %23, %24 : i1, f32
    %126 = llvm.fmul %124, %104 : f32
    %127 = llvm.intr.fabs(%93) : (f32) -> f32
    %128 = llvm.fadd %125, %126 : f32
    %129 = llvm.fcmp "oeq" %127, %25 : f32
    %130 = llvm.fmul %93, %26 : f32
    %131 = llvm.fmul %128, %93 : f32
    %132 = llvm.select %129, %130, %131 : i1, f32
    %133 = llvm.fmul %132, %27 : f32
    %134 = llvm.intr.maximum(%59, %133) : (f32, f32) -> f32
    %135 = llvm.intr.minimum(%75, %134) : (f32, f32) -> f32
    %136 = llvm.getelementptr inbounds %arg6[0, %81] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x f32>
    llvm.store %135, %136 {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=6">]} : f32, !llvm.ptr
    %137 = llvm.add %79, %35 : i64
    llvm.br ^bb3(%137 : i64)
  ^bb5:  // pred: ^bb3
    %138 = llvm.add %76, %35 : i64
    llvm.br ^bb1(%138 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}