module @broadcast_add_fusion.2_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @broadcast_add_fusion.2(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr inbounds %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %2 = llvm.load %1 invariant : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %4 = llvm.load %3 invariant dereferenceable<bytes = 8> : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %2[1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %6 = llvm.load %5 invariant dereferenceable<bytes = 512> : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %8 = llvm.load %7 : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %10 = llvm.load %9 invariant : !llvm.ptr -> i64
    %11 = llvm.getelementptr inbounds %8[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %12 = llvm.load %11 invariant : !llvm.ptr -> i64
    %13 = llvm.getelementptr inbounds %8[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %14 = llvm.load %13 invariant : !llvm.ptr -> i64
    llvm.call @broadcast_add_fusion.2_wrapped(%4, %6, %10, %12, %14) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return %0 : !llvm.ptr
  }
  llvm.func internal @broadcast_add_fusion.2_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 8 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 512 : index, llvm.noalias}, %arg2: i64, %arg3: i64, %arg4: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(8 : index) : i64
    %1 = llvm.mlir.constant(16 : index) : i64
    %2 = llvm.mlir.constant(0 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(8 : i64) : i64
    %5 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i32>
    %6 = llvm.load %5 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=1">]} : !llvm.ptr -> i32
    llvm.br ^bb1(%2 : i64)
  ^bb1(%7: i64):  // 2 preds: ^bb0, ^bb5
    %8 = llvm.icmp "slt" %7, %1 : i64
    llvm.cond_br %8, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %9 = llvm.mul %7, %4 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i64
    %10 = llvm.mul %7, %0 overflow<nsw> : i64
    llvm.br ^bb3(%2 : i64)
  ^bb3(%11: i64):  // 2 preds: ^bb2, ^bb4
    %12 = llvm.icmp "slt" %11, %0 : i64
    llvm.cond_br %12, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %13 = llvm.add %9, %11 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i64
    %14 = llvm.trunc %13 : i64 to i32
    %15 = llvm.add %14, %6 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %16 = llvm.add %10, %11 overflow<nsw> : i64
    %17 = llvm.getelementptr inbounds %arg1[0, %16] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<128 x i32>
    llvm.store %15, %17 {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=1">]} : i32, !llvm.ptr
    %18 = llvm.add %11, %3 : i64
    llvm.br ^bb3(%18 : i64)
  ^bb5:  // pred: ^bb3
    %19 = llvm.add %7, %3 : i64
    llvm.br ^bb1(%19 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}