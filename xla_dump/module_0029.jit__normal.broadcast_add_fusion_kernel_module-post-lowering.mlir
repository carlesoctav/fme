module @broadcast_add_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @broadcast_add_fusion(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr inbounds %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %2 = llvm.load %1 invariant : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %4 = llvm.load %3 invariant dereferenceable<bytes = 4> : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %2[1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %6 = llvm.load %5 invariant dereferenceable<bytes = 16> : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %2[2, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %8 = llvm.load %7 invariant dereferenceable<bytes = 384> : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %2[3, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %10 = llvm.load %9 invariant dereferenceable<bytes = 384> : !llvm.ptr -> !llvm.ptr
    %11 = llvm.getelementptr inbounds %2[4, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %12 = llvm.load %11 invariant dereferenceable<bytes = 4> : !llvm.ptr -> !llvm.ptr
    %13 = llvm.getelementptr inbounds %2[5, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %14 = llvm.load %13 invariant dereferenceable<bytes = 384> : !llvm.ptr -> !llvm.ptr
    %15 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %16 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
    %17 = llvm.getelementptr inbounds %16[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %18 = llvm.load %17 invariant : !llvm.ptr -> i64
    %19 = llvm.getelementptr inbounds %16[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %20 = llvm.load %19 invariant : !llvm.ptr -> i64
    %21 = llvm.getelementptr inbounds %16[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %22 = llvm.load %21 invariant : !llvm.ptr -> i64
    llvm.call @broadcast_add_fusion_wrapped(%4, %6, %8, %10, %12, %14, %18, %20, %22) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return %0 : !llvm.ptr
  }
  llvm.func internal @broadcast_add_fusion_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 16 : index, llvm.noalias, xla.invariant}, %arg2: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 384 : index, llvm.noalias, xla.invariant}, %arg3: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 384 : index, llvm.noalias, xla.invariant}, %arg4: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias, xla.invariant}, %arg5: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 384 : index, llvm.noalias}, %arg6: i64, %arg7: i64, %arg8: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(32 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(12 : index) : i64
    %6 = llvm.mlir.constant(8 : index) : i64
    %7 = llvm.getelementptr inbounds %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %8 = llvm.load %7 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=5">]} : !llvm.ptr -> i32
    %9 = llvm.sub %1, %8 : i32
    %10 = llvm.icmp "ult" %8, %1 : i32
    %11 = llvm.icmp "ult" %9, %1 : i32
    %12 = llvm.getelementptr inbounds %arg1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %13 = llvm.load %12 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=5">]} : !llvm.ptr -> i32
    %14 = llvm.sub %1, %13 : i32
    %15 = llvm.icmp "ult" %13, %1 : i32
    %16 = llvm.icmp "ult" %14, %1 : i32
    %17 = llvm.getelementptr inbounds %arg1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %18 = llvm.load %17 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=5">]} : !llvm.ptr -> i32
    %19 = llvm.sub %1, %18 : i32
    %20 = llvm.icmp "ult" %18, %1 : i32
    %21 = llvm.icmp "ult" %19, %1 : i32
    %22 = llvm.getelementptr inbounds %arg1[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %23 = llvm.load %22 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=5">]} : !llvm.ptr -> i32
    %24 = llvm.sub %1, %23 : i32
    %25 = llvm.icmp "ult" %23, %1 : i32
    %26 = llvm.icmp "ult" %24, %1 : i32
    %27 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %28 = llvm.load %27 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=5">]} : !llvm.ptr -> i32
    %29 = llvm.getelementptr inbounds %arg4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %30 = llvm.load %29 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=5">]} : !llvm.ptr -> i32
    %31 = llvm.add %30, %3 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    llvm.br ^bb1(%0 : i64)
  ^bb1(%32: i64):  // 2 preds: ^bb0, ^bb5
    %33 = llvm.icmp "slt" %32, %5 : i64
    llvm.cond_br %33, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %34 = llvm.mul %32, %6 overflow<nsw> : i64
    llvm.br ^bb3(%0 : i64)
  ^bb3(%35: i64):  // 2 preds: ^bb2, ^bb4
    %36 = llvm.icmp "slt" %35, %6 : i64
    llvm.cond_br %36, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %37 = llvm.add %34, %35 overflow<nsw> : i64
    %38 = llvm.getelementptr inbounds %arg2[0, %37] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<96 x i32>
    %39 = llvm.load %38 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=5">]} : !llvm.ptr -> i32
    %40 = llvm.shl %39, %8 : i32
    %41 = llvm.select %10, %40, %2 : i1, i32
    %42 = llvm.lshr %39, %9 : i32
    %43 = llvm.select %11, %42, %2 : i1, i32
    %44 = llvm.getelementptr inbounds %arg3[0, %37] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<96 x i32>
    %45 = llvm.load %44 invariant {noalias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=5">]} : !llvm.ptr -> i32
    %46 = llvm.or %41, %43 : i32
    %47 = llvm.add %45, %39 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %48 = llvm.xor %47, %46 : i32
    %49 = llvm.shl %48, %13 : i32
    %50 = llvm.select %15, %49, %2 : i1, i32
    %51 = llvm.lshr %48, %14 : i32
    %52 = llvm.select %16, %51, %2 : i1, i32
    %53 = llvm.or %50, %52 : i32
    %54 = llvm.add %47, %48 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %55 = llvm.xor %54, %53 : i32
    %56 = llvm.shl %55, %18 : i32
    %57 = llvm.select %20, %56, %2 : i1, i32
    %58 = llvm.lshr %55, %19 : i32
    %59 = llvm.select %21, %58, %2 : i1, i32
    %60 = llvm.or %57, %59 : i32
    %61 = llvm.add %54, %55 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %62 = llvm.xor %61, %60 : i32
    %63 = llvm.shl %62, %23 : i32
    %64 = llvm.select %25, %63, %2 : i1, i32
    %65 = llvm.lshr %62, %24 : i32
    %66 = llvm.select %26, %65, %2 : i1, i32
    %67 = llvm.add %61, %62 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %68 = llvm.or %64, %66 : i32
    %69 = llvm.xor %67, %68 : i32
    %70 = llvm.add %69, %28 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %71 = llvm.add %70, %31 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %72 = llvm.getelementptr inbounds %arg5[0, %37] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<96 x i32>
    llvm.store %71, %72 {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>>, description = "xla.slice_index=5">]} : i32, !llvm.ptr
    %73 = llvm.add %35, %4 : i64
    llvm.br ^bb3(%73 : i64)
  ^bb5:  // pred: ^bb3
    %74 = llvm.add %32, %4 : i64
    llvm.br ^bb1(%74 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}