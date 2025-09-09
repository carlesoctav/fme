module @broadcast_add_fusion.1_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @broadcast_add_fusion.1(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr inbounds %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %2 = llvm.load %1 invariant : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %4 = llvm.load %3 invariant dereferenceable<bytes = 4> : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %2[1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %6 = llvm.load %5 invariant dereferenceable<bytes = 16> : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %2[2, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %8 = llvm.load %7 invariant dereferenceable<bytes = 2048> : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %2[3, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %10 = llvm.load %9 invariant dereferenceable<bytes = 2048> : !llvm.ptr -> !llvm.ptr
    %11 = llvm.getelementptr inbounds %2[4, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %12 = llvm.load %11 invariant dereferenceable<bytes = 2048> : !llvm.ptr -> !llvm.ptr
    %13 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %14 = llvm.load %13 : !llvm.ptr -> !llvm.ptr
    %15 = llvm.getelementptr inbounds %14[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %16 = llvm.load %15 invariant : !llvm.ptr -> i64
    %17 = llvm.getelementptr inbounds %14[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %18 = llvm.load %17 invariant : !llvm.ptr -> i64
    %19 = llvm.getelementptr inbounds %14[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %20 = llvm.load %19 invariant : !llvm.ptr -> i64
    llvm.call @broadcast_add_fusion.1_wrapped(%4, %6, %8, %10, %12, %16, %18, %20) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return %0 : !llvm.ptr
  }
  llvm.func internal @broadcast_add_fusion.1_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 16 : index, llvm.noalias, xla.invariant}, %arg2: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, llvm.noalias, xla.invariant}, %arg3: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, llvm.noalias, xla.invariant}, %arg4: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, llvm.noalias}, %arg5: i64, %arg6: i64, %arg7: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(32 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(32 : index) : i64
    %5 = llvm.mlir.constant(16 : index) : i64
    %6 = llvm.getelementptr inbounds %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %7 = llvm.load %6 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=4">]} : !llvm.ptr -> i32
    %8 = llvm.sub %1, %7 : i32
    %9 = llvm.icmp "ult" %7, %1 : i32
    %10 = llvm.icmp "ult" %8, %1 : i32
    %11 = llvm.getelementptr inbounds %arg1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %12 = llvm.load %11 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=4">]} : !llvm.ptr -> i32
    %13 = llvm.sub %1, %12 : i32
    %14 = llvm.icmp "ult" %12, %1 : i32
    %15 = llvm.icmp "ult" %13, %1 : i32
    %16 = llvm.getelementptr inbounds %arg1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %17 = llvm.load %16 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=4">]} : !llvm.ptr -> i32
    %18 = llvm.sub %1, %17 : i32
    %19 = llvm.icmp "ult" %17, %1 : i32
    %20 = llvm.icmp "ult" %18, %1 : i32
    %21 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %22 = llvm.load %21 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=4">]} : !llvm.ptr -> i32
    llvm.br ^bb1(%0 : i64)
  ^bb1(%23: i64):  // 2 preds: ^bb0, ^bb5
    %24 = llvm.icmp "slt" %23, %4 : i64
    llvm.cond_br %24, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %25 = llvm.mul %23, %5 overflow<nsw> : i64
    llvm.br ^bb3(%0 : i64)
  ^bb3(%26: i64):  // 2 preds: ^bb2, ^bb4
    %27 = llvm.icmp "slt" %26, %5 : i64
    llvm.cond_br %27, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %28 = llvm.add %25, %26 overflow<nsw> : i64
    %29 = llvm.getelementptr inbounds %arg2[0, %28] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x i32>
    %30 = llvm.load %29 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=4">]} : !llvm.ptr -> i32
    %31 = llvm.shl %30, %7 : i32
    %32 = llvm.select %9, %31, %2 : i1, i32
    %33 = llvm.lshr %30, %8 : i32
    %34 = llvm.select %10, %33, %2 : i1, i32
    %35 = llvm.getelementptr inbounds %arg3[0, %28] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x i32>
    %36 = llvm.load %35 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=4">]} : !llvm.ptr -> i32
    %37 = llvm.or %32, %34 : i32
    %38 = llvm.add %36, %30 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %39 = llvm.xor %38, %37 : i32
    %40 = llvm.shl %39, %12 : i32
    %41 = llvm.select %14, %40, %2 : i1, i32
    %42 = llvm.lshr %39, %13 : i32
    %43 = llvm.select %15, %42, %2 : i1, i32
    %44 = llvm.or %41, %43 : i32
    %45 = llvm.add %38, %39 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %46 = llvm.xor %45, %44 : i32
    %47 = llvm.shl %46, %17 : i32
    %48 = llvm.select %19, %47, %2 : i1, i32
    %49 = llvm.lshr %46, %18 : i32
    %50 = llvm.select %20, %49, %2 : i1, i32
    %51 = llvm.or %48, %50 : i32
    %52 = llvm.add %45, %46 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %53 = llvm.xor %52, %51 : i32
    %54 = llvm.add %52, %53 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %55 = llvm.add %54, %22 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %56 = llvm.getelementptr inbounds %arg4[0, %28] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x i32>
    llvm.store %55, %56 {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=4">]} : i32, !llvm.ptr
    %57 = llvm.add %26, %3 : i64
    llvm.br ^bb3(%57 : i64)
  ^bb5:  // pred: ^bb3
    %58 = llvm.add %23, %3 : i64
    llvm.br ^bb1(%58 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}