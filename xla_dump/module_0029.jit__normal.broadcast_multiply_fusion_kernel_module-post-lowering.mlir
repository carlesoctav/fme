module @broadcast_multiply_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @xla.log1p.f32(f32) -> f32 attributes {sym_visibility = "private"}
  llvm.func @broadcast_multiply_fusion(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr inbounds %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %2 = llvm.load %1 invariant : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %4 = llvm.load %3 invariant dereferenceable<bytes = 384> : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %2[1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %6 = llvm.load %5 invariant dereferenceable<bytes = 384> : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %2[2, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %8 = llvm.load %7 invariant dereferenceable<bytes = 384> : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %10 = llvm.load %9 : !llvm.ptr -> !llvm.ptr
    %11 = llvm.getelementptr inbounds %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %12 = llvm.load %11 invariant : !llvm.ptr -> i64
    %13 = llvm.getelementptr inbounds %10[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %14 = llvm.load %13 invariant : !llvm.ptr -> i64
    %15 = llvm.getelementptr inbounds %10[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %16 = llvm.load %15 invariant : !llvm.ptr -> i64
    llvm.call @broadcast_multiply_fusion_wrapped(%4, %6, %8, %12, %14, %16) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return %0 : !llvm.ptr
  }
  llvm.func internal @broadcast_multiply_fusion_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 384 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 384 : index, llvm.noalias, xla.invariant}, %arg2: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 384 : index, llvm.noalias}, %arg3: i64, %arg4: i64, %arg5: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(1.00950558E-4 : f32) : f32
    %1 = llvm.mlir.constant(3.43273939E-7 : f32) : f32
    %2 = llvm.mlir.constant(-2.00214257E-4 : f32) : f32
    %3 = llvm.mlir.constant(2.81022636E-8 : f32) : f32
    %4 = llvm.mlir.constant(9 : i32) : i32
    %5 = llvm.mlir.constant(1065353216 : i32) : i32
    %6 = llvm.mlir.constant(-1.000000e+00 : f32) : f32
    %7 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %8 = llvm.mlir.constant(-0.99999994 : f32) : f32
    %9 = llvm.mlir.constant(5.000000e+00 : f32) : f32
    %10 = llvm.mlir.constant(-2.500000e+00 : f32) : f32
    %11 = llvm.mlir.constant(-3.000000e+00 : f32) : f32
    %12 = llvm.mlir.constant(-3.5233877E-6 : f32) : f32
    %13 = llvm.mlir.constant(0.00134934322 : f32) : f32
    %14 = llvm.mlir.constant(-4.39150654E-6 : f32) : f32
    %15 = llvm.mlir.constant(-0.00367342844 : f32) : f32
    %16 = llvm.mlir.constant(2.1858087E-4 : f32) : f32
    %17 = llvm.mlir.constant(0.00573950773 : f32) : f32
    %18 = llvm.mlir.constant(-0.00125372503 : f32) : f32
    %19 = llvm.mlir.constant(-0.0076224613 : f32) : f32
    %20 = llvm.mlir.constant(-0.00417768164 : f32) : f32
    %21 = llvm.mlir.constant(0.00943887047 : f32) : f32
    %22 = llvm.mlir.constant(0.246640727 : f32) : f32
    %23 = llvm.mlir.constant(1.00167406 : f32) : f32
    %24 = llvm.mlir.constant(1.50140941 : f32) : f32
    %25 = llvm.mlir.constant(2.83297682 : f32) : f32
    %26 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %27 = llvm.mlir.constant(0x7F800000 : f32) : f32
    %28 = llvm.mlir.constant(1.41421354 : f32) : f32
    %29 = llvm.mlir.constant(1 : index) : i64
    %30 = llvm.mlir.constant(0 : index) : i64
    %31 = llvm.mlir.constant(12 : index) : i64
    %32 = llvm.mlir.constant(8 : index) : i64
    llvm.br ^bb1(%30 : i64)
  ^bb1(%33: i64):  // 2 preds: ^bb0, ^bb5
    %34 = llvm.icmp "slt" %33, %31 : i64
    llvm.cond_br %34, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %35 = llvm.mul %33, %32 overflow<nsw> : i64
    llvm.br ^bb3(%30 : i64)
  ^bb3(%36: i64):  // 2 preds: ^bb2, ^bb4
    %37 = llvm.icmp "slt" %36, %32 : i64
    llvm.cond_br %37, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %38 = llvm.add %35, %36 overflow<nsw> : i64
    %39 = llvm.getelementptr inbounds %arg0[0, %38] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<96 x i32>
    %40 = llvm.load %39 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=2">]} : !llvm.ptr -> i32
    %41 = llvm.getelementptr inbounds %arg1[0, %38] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<96 x i32>
    %42 = llvm.load %41 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=2">]} : !llvm.ptr -> i32
    %43 = llvm.xor %40, %42 : i32
    %44 = llvm.lshr %43, %4 : i32
    %45 = llvm.or %44, %5 : i32
    %46 = llvm.bitcast %45 : i32 to f32
    %47 = llvm.fadd %46, %6 : f32
    %48 = llvm.fmul %47, %7 : f32
    %49 = llvm.fadd %48, %8 : f32
    %50 = llvm.intr.maximum(%49, %8) : (f32, f32) -> f32
    %51 = llvm.fneg %50 : f32
    %52 = llvm.fmul %50, %51 : f32
    %53 = llvm.call @xla.log1p.f32(%52) : (f32) -> f32
    %54 = llvm.fneg %53 : f32
    %55 = llvm.fcmp "olt" %54, %9 : f32
    %56 = llvm.select %55, %3, %2 : i1, f32
    %57 = llvm.select %55, %1, %0 : i1, f32
    %58 = llvm.intr.sqrt(%54) : (f32) -> f32
    %59 = llvm.fadd %54, %10 : f32
    %60 = llvm.fadd %58, %11 : f32
    %61 = llvm.select %55, %59, %60 : i1, f32
    %62 = llvm.fmul %56, %61 : f32
    %63 = llvm.fadd %57, %62 : f32
    %64 = llvm.select %55, %12, %13 : i1, f32
    %65 = llvm.fmul %63, %61 : f32
    %66 = llvm.fadd %64, %65 : f32
    %67 = llvm.select %55, %14, %15 : i1, f32
    %68 = llvm.fmul %66, %61 : f32
    %69 = llvm.fadd %67, %68 : f32
    %70 = llvm.select %55, %16, %17 : i1, f32
    %71 = llvm.fmul %69, %61 : f32
    %72 = llvm.fadd %70, %71 : f32
    %73 = llvm.select %55, %18, %19 : i1, f32
    %74 = llvm.fmul %72, %61 : f32
    %75 = llvm.fadd %73, %74 : f32
    %76 = llvm.select %55, %20, %21 : i1, f32
    %77 = llvm.fmul %75, %61 : f32
    %78 = llvm.fadd %76, %77 : f32
    %79 = llvm.select %55, %22, %23 : i1, f32
    %80 = llvm.fmul %78, %61 : f32
    %81 = llvm.fadd %79, %80 : f32
    %82 = llvm.select %55, %24, %25 : i1, f32
    %83 = llvm.fmul %81, %61 : f32
    %84 = llvm.intr.fabs(%50) : (f32) -> f32
    %85 = llvm.fadd %82, %83 : f32
    %86 = llvm.fcmp "oeq" %84, %26 : f32
    %87 = llvm.fmul %50, %27 : f32
    %88 = llvm.fmul %85, %50 : f32
    %89 = llvm.select %86, %87, %88 : i1, f32
    %90 = llvm.fmul %89, %28 : f32
    %91 = llvm.getelementptr inbounds %arg2[0, %38] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<96 x f32>
    llvm.store %90, %91 {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=2">]} : f32, !llvm.ptr
    %92 = llvm.add %36, %29 : i64
    llvm.br ^bb3(%92 : i64)
  ^bb5:  // pred: ^bb3
    %93 = llvm.add %33, %29 : i64
    llvm.br ^bb1(%93 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}