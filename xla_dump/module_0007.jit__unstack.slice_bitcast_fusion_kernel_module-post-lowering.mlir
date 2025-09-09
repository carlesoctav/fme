module @slice_bitcast_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @slice_bitcast_fusion(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr inbounds %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %2 = llvm.load %1 invariant : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %4 = llvm.load %3 invariant dereferenceable<bytes = 16> : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %2[1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %6 = llvm.load %5 invariant dereferenceable<bytes = 8> : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %8 = llvm.load %7 : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %10 = llvm.load %9 invariant : !llvm.ptr -> i64
    %11 = llvm.getelementptr inbounds %8[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %12 = llvm.load %11 invariant : !llvm.ptr -> i64
    %13 = llvm.getelementptr inbounds %8[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %14 = llvm.load %13 invariant : !llvm.ptr -> i64
    llvm.call @slice_bitcast_fusion_wrapped(%4, %6, %10, %12, %14) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return %0 : !llvm.ptr
  }
  llvm.func internal @slice_bitcast_fusion_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 16 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 8 : index, llvm.noalias}, %arg2: i64, %arg3: i64, %arg4: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(2 : index) : i64
    llvm.br ^bb1(%1 : i64)
  ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb2
    %4 = llvm.icmp "slt" %3, %2 : i64
    llvm.cond_br %4, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %5 = llvm.add %3, %2 overflow<nsw> : i64
    %6 = llvm.getelementptr inbounds %arg0[0, %5] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i32>
    %7 = llvm.load %6 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=1">]} : !llvm.ptr -> i32
    %8 = llvm.getelementptr inbounds %arg1[0, %3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2 x i32>
    llvm.store %7, %8 {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=1">]} : i32, !llvm.ptr
    %9 = llvm.add %3, %0 : i64
    llvm.br ^bb1(%9 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
}