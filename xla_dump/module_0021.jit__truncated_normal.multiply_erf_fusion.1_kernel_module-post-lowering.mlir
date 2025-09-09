module @multiply_erf_fusion.1_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @multiply_erf_fusion.1(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr inbounds %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %2 = llvm.load %1 invariant : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %4 = llvm.load %3 invariant dereferenceable<bytes = 4> : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %2[1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %6 = llvm.load %5 invariant dereferenceable<bytes = 4> : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %8 = llvm.load %7 : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %10 = llvm.load %9 invariant : !llvm.ptr -> i64
    %11 = llvm.getelementptr inbounds %8[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %12 = llvm.load %11 invariant : !llvm.ptr -> i64
    %13 = llvm.getelementptr inbounds %8[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %14 = llvm.load %13 invariant : !llvm.ptr -> i64
    llvm.call @multiply_erf_fusion.1_wrapped(%4, %6, %10, %12, %14) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return %0 : !llvm.ptr
  }
  llvm.func internal @multiply_erf_fusion.1_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias}, %arg2: i64, %arg3: i64, %arg4: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %1 = llvm.mlir.constant(0.497469246 : f32) : f32
    %2 = llvm.mlir.constant(0.110985048 : f32) : f32
    %3 = llvm.mlir.constant(0.0140704699 : f32) : f32
    %4 = llvm.mlir.constant(0.00101796258 : f32) : f32
    %5 = llvm.mlir.constant(2.35479656E-5 : f32) : f32
    %6 = llvm.mlir.constant(-1.17916031E-7 : f32) : f32
    %7 = llvm.mlir.constant(1.12837911 : f32) : f32
    %8 = llvm.mlir.constant(0.185208321 : f32) : f32
    %9 = llvm.mlir.constant(0.0509556942 : f32) : f32
    %10 = llvm.mlir.constant(0.00340829091 : f32) : f32
    %11 = llvm.mlir.constant(2.29050653E-4 : f32) : f32
    %12 = llvm.mlir.constant(3.74392128 : f32) : f32
    %13 = llvm.mlir.constant(-3.74392128 : f32) : f32
    %14 = llvm.mlir.constant(0.707106769 : f32) : f32
    %15 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %16 = llvm.load %15 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=1">]} : !llvm.ptr -> i32
    %17 = llvm.sitofp %16 : i32 to f32
    %18 = llvm.fmul %17, %14 : f32
    %19 = llvm.intr.maximum(%18, %13) : (f32, f32) -> f32
    %20 = llvm.intr.minimum(%19, %12) : (f32, f32) -> f32
    %21 = llvm.fmul %20, %20 : f32
    %22 = llvm.intr.fma(%11, %21, %10) : (f32, f32, f32) -> f32
    %23 = llvm.intr.fma(%22, %21, %9) : (f32, f32, f32) -> f32
    %24 = llvm.intr.fma(%23, %21, %8) : (f32, f32, f32) -> f32
    %25 = llvm.intr.fma(%24, %21, %7) : (f32, f32, f32) -> f32
    %26 = llvm.fmul %20, %25 : f32
    %27 = llvm.intr.fma(%6, %21, %5) : (f32, f32, f32) -> f32
    %28 = llvm.intr.fma(%27, %21, %4) : (f32, f32, f32) -> f32
    %29 = llvm.intr.fma(%28, %21, %3) : (f32, f32, f32) -> f32
    %30 = llvm.intr.fma(%29, %21, %2) : (f32, f32, f32) -> f32
    %31 = llvm.intr.fma(%30, %21, %1) : (f32, f32, f32) -> f32
    %32 = llvm.intr.fma(%31, %21, %0) : (f32, f32, f32) -> f32
    %33 = llvm.fdiv %26, %32 : f32
    %34 = llvm.getelementptr inbounds %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x f32>
    llvm.store %33, %34 {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=1">]} : f32, !llvm.ptr
    llvm.return
  }
}