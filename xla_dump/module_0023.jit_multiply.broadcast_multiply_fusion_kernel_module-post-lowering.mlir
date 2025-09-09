module @broadcast_multiply_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @broadcast_multiply_fusion(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr inbounds %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %2 = llvm.load %1 invariant : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %4 = llvm.load %3 invariant dereferenceable<bytes = 2048> : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %2[1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %6 = llvm.load %5 invariant dereferenceable<bytes = 4> : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %2[2, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %8 = llvm.load %7 invariant dereferenceable<bytes = 2048> : !llvm.ptr -> !llvm.ptr
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
  llvm.func internal @broadcast_multiply_fusion_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias, xla.invariant}, %arg2: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, llvm.noalias}, %arg3: i64, %arg4: i64, %arg5: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(32 : index) : i64
    %3 = llvm.mlir.constant(16 : index) : i64
    %4 = llvm.getelementptr inbounds %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x f32>
    %5 = llvm.load %4 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=2">]} : !llvm.ptr -> f32
    llvm.br ^bb1(%1 : i64)
  ^bb1(%6: i64):  // 2 preds: ^bb0, ^bb5
    %7 = llvm.icmp "slt" %6, %2 : i64
    llvm.cond_br %7, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %8 = llvm.mul %6, %3 overflow<nsw> : i64
    llvm.br ^bb3(%1 : i64)
  ^bb3(%9: i64):  // 2 preds: ^bb2, ^bb4
    %10 = llvm.icmp "slt" %9, %3 : i64
    llvm.cond_br %10, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %11 = llvm.add %8, %9 overflow<nsw> : i64
    %12 = llvm.getelementptr inbounds %arg0[0, %11] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x f32>
    %13 = llvm.load %12 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=2">]} : !llvm.ptr -> f32
    %14 = llvm.fmul %13, %5 : f32
    %15 = llvm.getelementptr inbounds %arg2[0, %11] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x f32>
    llvm.store %14, %15 {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=2">]} : f32, !llvm.ptr
    %16 = llvm.add %9, %0 : i64
    llvm.br ^bb3(%16 : i64)
  ^bb5:  // pred: ^bb3
    %17 = llvm.add %6, %0 : i64
    llvm.br ^bb1(%17 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}