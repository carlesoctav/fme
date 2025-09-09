; ModuleID = '__compute_module_call_computation_kernel_module'
source_filename = "__compute_module_call_computation_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_NumWorkGroups = type { i64, i64, i64 }
%XLA_CPU_WorkGroupId = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@constant.15 = private unnamed_addr constant [4 x i8] c"\01\00\00\00", align 4
@0 = private unnamed_addr constant [16 x i8] c"\0D\00\00\00\0F\00\00\00\1A\00\00\00\06\00\00\00", align 16
@1 = private unnamed_addr constant [16 x i8] c"\11\00\00\00\1D\00\00\00\10\00\00\00\18\00\00\00", align 16
@2 = private unnamed_addr constant [4 x i8] zeroinitializer, align 4
@constant.14 = private unnamed_addr constant [4 x i8] c"\05\00\00\00", align 4
@3 = private unnamed_addr constant [4 x i8] c" \00\00\00"
@4 = private unnamed_addr constant [4 x i8] c"\01\00\00\00"
@5 = private unnamed_addr constant [4 x i8] c" \00\00\00"

; Function Attrs: uwtable
define ptr @call_kernel(ptr %0) #0 {
  %num_workgroups_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 0
  %num_workgroups = load ptr, ptr %num_workgroups_gep, align 8
  %num_workgroups_x_gep = getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, ptr %num_workgroups, i32 0, i32 0
  %num_workgroups_y_gep = getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, ptr %num_workgroups, i32 0, i32 1
  %num_workgroups_z_gep = getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, ptr %num_workgroups, i32 0, i32 2
  %num_workgroups_x = load i64, ptr %num_workgroups_x_gep, align 4
  %num_workgroups_y = load i64, ptr %num_workgroups_y_gep, align 4
  %num_workgroups_z = load i64, ptr %num_workgroups_z_gep, align 4
  %workgroup_id_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %workgroup_id = load ptr, ptr %workgroup_id_gep, align 8
  %workgroup_id_x_gep = getelementptr inbounds nuw %XLA_CPU_WorkGroupId, ptr %workgroup_id, i32 0, i32 0
  %workgroup_id_y_gep = getelementptr inbounds nuw %XLA_CPU_WorkGroupId, ptr %workgroup_id, i32 0, i32 1
  %workgroup_id_z_gep = getelementptr inbounds nuw %XLA_CPU_WorkGroupId, ptr %workgroup_id, i32 0, i32 2
  %workgroup_id_x = load i64, ptr %workgroup_id_x_gep, align 4
  %workgroup_id_y = load i64, ptr %workgroup_id_y_gep, align 4
  %workgroup_id_z = load i64, ptr %workgroup_id_z_gep, align 4
  %args_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args = load ptr, ptr %args_gep, align 8
  %arg0_gep = getelementptr %XLA_CPU_KernelArg, ptr %args, i32 0, i32 0
  %arg0 = load ptr, ptr %arg0_gep, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %args_gep1 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args2 = load ptr, ptr %args_gep1, align 8
  %arg1_gep = getelementptr %XLA_CPU_KernelArg, ptr %args2, i32 1, i32 0
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep3 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args4 = load ptr, ptr %args_gep3, align 8
  %arg2_gep = getelementptr %XLA_CPU_KernelArg, ptr %args4, i32 2, i32 0
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %args_gep5 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args6 = load ptr, ptr %args_gep5, align 8
  %arg3_gep = getelementptr %XLA_CPU_KernelArg, ptr %args6, i32 3, i32 0
  %arg3 = load ptr, ptr %arg3_gep, align 8, !invariant.load !3, !dereferenceable !5, !align !5
  %args_gep7 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args8 = load ptr, ptr %args_gep7, align 8
  %arg4_gep = getelementptr %XLA_CPU_KernelArg, ptr %args8, i32 4, i32 0
  %arg4 = load ptr, ptr %arg4_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep9 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args10 = load ptr, ptr %args_gep9, align 8
  %arg5_gep = getelementptr %XLA_CPU_KernelArg, ptr %args10, i32 5, i32 0
  %arg5 = load ptr, ptr %arg5_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep11 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args12 = load ptr, ptr %args_gep11, align 8
  %arg6_gep = getelementptr %XLA_CPU_KernelArg, ptr %args12, i32 6, i32 0
  %arg6 = load ptr, ptr %arg6_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %args_gep13 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args14 = load ptr, ptr %args_gep13, align 8
  %arg7_gep = getelementptr %XLA_CPU_KernelArg, ptr %args14, i32 7, i32 0
  %arg7 = load ptr, ptr %arg7_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep15 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args16 = load ptr, ptr %args_gep15, align 8
  %arg8_gep = getelementptr %XLA_CPU_KernelArg, ptr %args16, i32 8, i32 0
  %arg8 = load ptr, ptr %arg8_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep17 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args18 = load ptr, ptr %args_gep17, align 8
  %arg9_gep = getelementptr %XLA_CPU_KernelArg, ptr %args18, i32 9, i32 0
  %arg9 = load ptr, ptr %arg9_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep19 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args20 = load ptr, ptr %args_gep19, align 8
  %arg10_gep = getelementptr %XLA_CPU_KernelArg, ptr %args20, i32 10, i32 0
  %arg10 = load ptr, ptr %arg10_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep21 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args22 = load ptr, ptr %args_gep21, align 8
  %arg11_gep = getelementptr %XLA_CPU_KernelArg, ptr %args22, i32 11, i32 0
  %arg11 = load ptr, ptr %arg11_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %args_gep23 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args24 = load ptr, ptr %args_gep23, align 8
  %arg12_gep = getelementptr %XLA_CPU_KernelArg, ptr %args24, i32 12, i32 0
  %arg12 = load ptr, ptr %arg12_gep, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %args_gep25 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args26 = load ptr, ptr %args_gep25, align 8
  %arg13_gep = getelementptr %XLA_CPU_KernelArg, ptr %args26, i32 13, i32 0
  %arg13 = load ptr, ptr %arg13_gep, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %args_gep27 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args28 = load ptr, ptr %args_gep27, align 8
  %arg14_gep = getelementptr %XLA_CPU_KernelArg, ptr %args28, i32 14, i32 0
  %arg14 = load ptr, ptr %arg14_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep29 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args30 = load ptr, ptr %args_gep29, align 8
  %arg15_gep = getelementptr %XLA_CPU_KernelArg, ptr %args30, i32 15, i32 0
  %arg15 = load ptr, ptr %arg15_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep31 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args32 = load ptr, ptr %args_gep31, align 8
  %arg16_gep = getelementptr %XLA_CPU_KernelArg, ptr %args32, i32 16, i32 0
  %arg16 = load ptr, ptr %arg16_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %args_gep33 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args34 = load ptr, ptr %args_gep33, align 8
  %arg17_gep = getelementptr %XLA_CPU_KernelArg, ptr %args34, i32 17, i32 0
  %arg17 = load ptr, ptr %arg17_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep35 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args36 = load ptr, ptr %args_gep35, align 8
  %arg18_gep = getelementptr %XLA_CPU_KernelArg, ptr %args36, i32 18, i32 0
  %arg18 = load ptr, ptr %arg18_gep, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %args_gep37 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args38 = load ptr, ptr %args_gep37, align 8
  %arg19_gep = getelementptr %XLA_CPU_KernelArg, ptr %args38, i32 19, i32 0
  %arg19 = load ptr, ptr %arg19_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %args_gep39 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args40 = load ptr, ptr %args_gep39, align 8
  %arg20_gep = getelementptr %XLA_CPU_KernelArg, ptr %args40, i32 20, i32 0
  %arg20 = load ptr, ptr %arg20_gep, align 8, !invariant.load !3, !dereferenceable !5, !align !5
  %args_gep41 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args42 = load ptr, ptr %args_gep41, align 8
  %arg21_gep = getelementptr %XLA_CPU_KernelArg, ptr %args42, i32 21, i32 0
  %arg21 = load ptr, ptr %arg21_gep, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %args_gep43 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args44 = load ptr, ptr %args_gep43, align 8
  %arg22_gep = getelementptr %XLA_CPU_KernelArg, ptr %args44, i32 22, i32 0
  %arg22 = load ptr, ptr %arg22_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep45 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args46 = load ptr, ptr %args_gep45, align 8
  %arg23_gep = getelementptr %XLA_CPU_KernelArg, ptr %args46, i32 23, i32 0
  %arg23 = load ptr, ptr %arg23_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep47 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args48 = load ptr, ptr %args_gep47, align 8
  %arg24_gep = getelementptr %XLA_CPU_KernelArg, ptr %args48, i32 24, i32 0
  %arg24 = load ptr, ptr %arg24_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep49 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args50 = load ptr, ptr %args_gep49, align 8
  %arg25_gep = getelementptr %XLA_CPU_KernelArg, ptr %args50, i32 25, i32 0
  %arg25 = load ptr, ptr %arg25_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep51 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args52 = load ptr, ptr %args_gep51, align 8
  %arg26_gep = getelementptr %XLA_CPU_KernelArg, ptr %args52, i32 26, i32 0
  %arg26 = load ptr, ptr %arg26_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %args_gep53 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args54 = load ptr, ptr %args_gep53, align 8
  %arg27_gep = getelementptr %XLA_CPU_KernelArg, ptr %args54, i32 27, i32 0
  %arg27 = load ptr, ptr %arg27_gep, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %args_gep55 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args56 = load ptr, ptr %args_gep55, align 8
  %arg28_gep = getelementptr %XLA_CPU_KernelArg, ptr %args56, i32 28, i32 0
  %arg28 = load ptr, ptr %arg28_gep, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %args_gep57 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args58 = load ptr, ptr %args_gep57, align 8
  %arg29_gep = getelementptr %XLA_CPU_KernelArg, ptr %args58, i32 29, i32 0
  %arg29 = load ptr, ptr %arg29_gep, align 8, !invariant.load !3, !dereferenceable !8, !align !5
  %args_gep59 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args60 = load ptr, ptr %args_gep59, align 8
  %arg30_gep = getelementptr %XLA_CPU_KernelArg, ptr %args60, i32 30, i32 0
  %arg30 = load ptr, ptr %arg30_gep, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %args_gep61 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args62 = load ptr, ptr %args_gep61, align 8
  %arg31_gep = getelementptr %XLA_CPU_KernelArg, ptr %args62, i32 31, i32 0
  %arg31 = load ptr, ptr %arg31_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep63 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args64 = load ptr, ptr %args_gep63, align 8
  %arg32_gep = getelementptr %XLA_CPU_KernelArg, ptr %args64, i32 32, i32 0
  %arg32 = load ptr, ptr %arg32_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %args_gep65 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args66 = load ptr, ptr %args_gep65, align 8
  %arg33_gep = getelementptr %XLA_CPU_KernelArg, ptr %args66, i32 33, i32 0
  %arg33 = load ptr, ptr %arg33_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep67 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args68 = load ptr, ptr %args_gep67, align 8
  %arg34_gep = getelementptr %XLA_CPU_KernelArg, ptr %args68, i32 34, i32 0
  %arg34 = load ptr, ptr %arg34_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep69 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args70 = load ptr, ptr %args_gep69, align 8
  %arg35_gep = getelementptr %XLA_CPU_KernelArg, ptr %args70, i32 35, i32 0
  %arg35 = load ptr, ptr %arg35_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep71 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args72 = load ptr, ptr %args_gep71, align 8
  %arg36_gep = getelementptr %XLA_CPU_KernelArg, ptr %args72, i32 36, i32 0
  %arg36 = load ptr, ptr %arg36_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %args_gep73 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args74 = load ptr, ptr %args_gep73, align 8
  %arg37_gep = getelementptr %XLA_CPU_KernelArg, ptr %args74, i32 37, i32 0
  %arg37 = load ptr, ptr %arg37_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %args_gep75 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args76 = load ptr, ptr %args_gep75, align 8
  %arg38_gep = getelementptr %XLA_CPU_KernelArg, ptr %args76, i32 38, i32 0
  %arg38 = load ptr, ptr %arg38_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %buffer_table = alloca ptr, i64 39, align 8
  %2 = getelementptr inbounds ptr, ptr %buffer_table, i64 0
  store ptr %arg0, ptr %2, align 8
  %3 = getelementptr inbounds ptr, ptr %buffer_table, i64 1
  store ptr %arg1, ptr %3, align 8
  %4 = getelementptr inbounds ptr, ptr %buffer_table, i64 2
  store ptr %arg2, ptr %4, align 8
  %5 = getelementptr inbounds ptr, ptr %buffer_table, i64 3
  store ptr %arg3, ptr %5, align 8
  %6 = getelementptr inbounds ptr, ptr %buffer_table, i64 4
  store ptr %arg4, ptr %6, align 8
  %7 = getelementptr inbounds ptr, ptr %buffer_table, i64 5
  store ptr %arg5, ptr %7, align 8
  %8 = getelementptr inbounds ptr, ptr %buffer_table, i64 6
  store ptr %arg6, ptr %8, align 8
  %9 = getelementptr inbounds ptr, ptr %buffer_table, i64 7
  store ptr %arg7, ptr %9, align 8
  %10 = getelementptr inbounds ptr, ptr %buffer_table, i64 8
  store ptr %arg8, ptr %10, align 8
  %11 = getelementptr inbounds ptr, ptr %buffer_table, i64 9
  store ptr %arg9, ptr %11, align 8
  %12 = getelementptr inbounds ptr, ptr %buffer_table, i64 10
  store ptr %arg10, ptr %12, align 8
  %13 = getelementptr inbounds ptr, ptr %buffer_table, i64 11
  store ptr %arg11, ptr %13, align 8
  %14 = getelementptr inbounds ptr, ptr %buffer_table, i64 12
  store ptr %arg12, ptr %14, align 8
  %15 = getelementptr inbounds ptr, ptr %buffer_table, i64 13
  store ptr %arg13, ptr %15, align 8
  %16 = getelementptr inbounds ptr, ptr %buffer_table, i64 14
  store ptr %arg14, ptr %16, align 8
  %17 = getelementptr inbounds ptr, ptr %buffer_table, i64 15
  store ptr %arg15, ptr %17, align 8
  %18 = getelementptr inbounds ptr, ptr %buffer_table, i64 16
  store ptr %arg16, ptr %18, align 8
  %19 = getelementptr inbounds ptr, ptr %buffer_table, i64 17
  store ptr %arg17, ptr %19, align 8
  %20 = getelementptr inbounds ptr, ptr %buffer_table, i64 18
  store ptr %arg18, ptr %20, align 8
  %21 = getelementptr inbounds ptr, ptr %buffer_table, i64 19
  store ptr %arg19, ptr %21, align 8
  %22 = getelementptr inbounds ptr, ptr %buffer_table, i64 20
  store ptr %arg20, ptr %22, align 8
  %23 = getelementptr inbounds ptr, ptr %buffer_table, i64 21
  store ptr %arg21, ptr %23, align 8
  %24 = getelementptr inbounds ptr, ptr %buffer_table, i64 22
  store ptr %arg22, ptr %24, align 8
  %25 = getelementptr inbounds ptr, ptr %buffer_table, i64 23
  store ptr %arg23, ptr %25, align 8
  %26 = getelementptr inbounds ptr, ptr %buffer_table, i64 24
  store ptr %arg24, ptr %26, align 8
  %27 = getelementptr inbounds ptr, ptr %buffer_table, i64 25
  store ptr %arg25, ptr %27, align 8
  %28 = getelementptr inbounds ptr, ptr %buffer_table, i64 26
  store ptr %arg26, ptr %28, align 8
  %29 = getelementptr inbounds ptr, ptr %buffer_table, i64 27
  store ptr %arg27, ptr %29, align 8
  %30 = getelementptr inbounds ptr, ptr %buffer_table, i64 28
  store ptr %arg28, ptr %30, align 8
  %31 = getelementptr inbounds ptr, ptr %buffer_table, i64 29
  store ptr %arg29, ptr %31, align 8
  %32 = getelementptr inbounds ptr, ptr %buffer_table, i64 30
  store ptr %arg30, ptr %32, align 8
  %33 = getelementptr inbounds ptr, ptr %buffer_table, i64 31
  store ptr %arg31, ptr %33, align 8
  %34 = getelementptr inbounds ptr, ptr %buffer_table, i64 32
  store ptr %arg32, ptr %34, align 8
  %35 = getelementptr inbounds ptr, ptr %buffer_table, i64 33
  store ptr %arg33, ptr %35, align 8
  %36 = getelementptr inbounds ptr, ptr %buffer_table, i64 34
  store ptr %arg34, ptr %36, align 8
  %37 = getelementptr inbounds ptr, ptr %buffer_table, i64 35
  store ptr %arg35, ptr %37, align 8
  %38 = getelementptr inbounds ptr, ptr %buffer_table, i64 36
  store ptr %arg36, ptr %38, align 8
  %39 = getelementptr inbounds ptr, ptr %buffer_table, i64 37
  store ptr %arg37, ptr %39, align 8
  %40 = getelementptr inbounds ptr, ptr %buffer_table, i64 38
  store ptr %arg38, ptr %40, align 8
  call void @while.5_computation(ptr null, ptr null, ptr null, ptr %buffer_table, ptr null, ptr null)
  br label %return

return:                                           ; preds = %1
  ret ptr null
}

; Function Attrs: alwaysinline uwtable
define internal void @while.6(ptr %retval, ptr noalias %run_options, ptr noalias %params, ptr noalias %buffer_table, ptr noalias %status, ptr noalias %prof_counters) #1 {
entry:
  %broadcast_add_fusion.kLoop_fusion.invar_address.dim.0 = alloca i64, align 8
  %broadcast_add_fusion.1.kLoop_fusion.invar_address.dim.0 = alloca i64, align 8
  %0 = getelementptr inbounds ptr, ptr %buffer_table, i64 20
  %arg_tuple.4 = load ptr, ptr %0, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  %1 = getelementptr inbounds ptr, ptr %buffer_table, i64 30
  %2 = load ptr, ptr %1, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  %3 = getelementptr inbounds ptr, ptr %buffer_table, i64 28
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4, !align !4
  %5 = getelementptr inbounds ptr, ptr %buffer_table, i64 33
  %6 = load ptr, ptr %5, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  %7 = getelementptr inbounds ptr, ptr %buffer_table, i64 23
  %8 = load ptr, ptr %7, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  %9 = getelementptr inbounds ptr, ptr %buffer_table, i64 34
  %10 = load ptr, ptr %9, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  %11 = getelementptr inbounds ptr, ptr %buffer_table, i64 19
  %12 = load ptr, ptr %11, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  %13 = getelementptr inbounds ptr, ptr %buffer_table, i64 32
  %14 = load ptr, ptr %13, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  %15 = getelementptr inbounds ptr, ptr %buffer_table, i64 31
  %16 = load ptr, ptr %15, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  %17 = getelementptr inbounds ptr, ptr %buffer_table, i64 27
  %copy.15 = load ptr, ptr %17, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %copy.15, ptr align 1 %2, i64 16, i1 false)
  %18 = getelementptr inbounds ptr, ptr %buffer_table, i64 21
  %copy.14 = load ptr, ptr %18, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %copy.14, ptr align 1 %4, i64 16, i1 false)
  %19 = getelementptr inbounds ptr, ptr %buffer_table, i64 22
  %copy.13 = load ptr, ptr %19, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %copy.13, ptr align 1 %6, i64 4, i1 false)
  %20 = getelementptr inbounds ptr, ptr %buffer_table, i64 25
  %copy.12 = load ptr, ptr %20, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %copy.12, ptr align 1 %8, i64 4, i1 false)
  %21 = getelementptr inbounds ptr, ptr %buffer_table, i64 38
  %copy.11 = load ptr, ptr %21, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %copy.11, ptr align 1 %10, i64 4, i1 false)
  %22 = getelementptr inbounds ptr, ptr %buffer_table, i64 26
  %copy.10 = load ptr, ptr %22, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %copy.10, ptr align 1 %12, i64 8, i1 false)
  %23 = getelementptr inbounds ptr, ptr %buffer_table, i64 37
  %copy.9 = load ptr, ptr %23, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %copy.9, ptr align 1 %14, i64 8, i1 false)
  %24 = getelementptr inbounds ptr, ptr %buffer_table, i64 36
  %copy.8 = load ptr, ptr %24, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %copy.8, ptr align 1 %16, i64 4, i1 false)
  %25 = getelementptr inbounds ptr, ptr %buffer_table, i64 30
  %copy.23 = load ptr, ptr %25, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %copy.23, ptr align 1 %copy.14, i64 16, i1 false)
  %26 = getelementptr inbounds ptr, ptr %buffer_table, i64 28
  %copy.22 = load ptr, ptr %26, align 8, !invariant.load !3, !dereferenceable !4, !align !4
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %copy.22, ptr align 1 %copy.15, i64 16, i1 false)
  %27 = getelementptr inbounds ptr, ptr %buffer_table, i64 23
  %copy.20 = load ptr, ptr %27, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %copy.20, ptr align 1 %copy.13, i64 4, i1 false)
  %28 = getelementptr inbounds ptr, ptr %buffer_table, i64 33
  %copy.21 = load ptr, ptr %28, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %copy.21, ptr align 1 %copy.11, i64 4, i1 false)
  %29 = getelementptr inbounds ptr, ptr %buffer_table, i64 34
  %copy.19 = load ptr, ptr %29, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %copy.19, ptr align 1 %copy.12, i64 4, i1 false)
  %30 = getelementptr inbounds ptr, ptr %buffer_table, i64 32
  %broadcast_add_fusion.1 = load ptr, ptr %30, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  store i64 0, ptr %broadcast_add_fusion.1.kLoop_fusion.invar_address.dim.0, align 4
  br label %broadcast_add_fusion.1.kLoop_fusion.loop_header.dim.0

return:                                           ; preds = %broadcast_add_fusion.kLoop_fusion.loop_exit.dim.0
  ret void

broadcast_add_fusion.1.kLoop_fusion.loop_header.dim.0: ; preds = %broadcast_add_fusion.1.kLoop_fusion.loop_body.dim.0, %entry
  %broadcast_add_fusion.1.kLoop_fusion.indvar.dim.0 = load i64, ptr %broadcast_add_fusion.1.kLoop_fusion.invar_address.dim.0, align 4
  %31 = icmp uge i64 %broadcast_add_fusion.1.kLoop_fusion.indvar.dim.0, 2
  br i1 %31, label %broadcast_add_fusion.1.kLoop_fusion.loop_exit.dim.0, label %broadcast_add_fusion.1.kLoop_fusion.loop_body.dim.0

broadcast_add_fusion.1.kLoop_fusion.loop_body.dim.0: ; preds = %broadcast_add_fusion.1.kLoop_fusion.loop_header.dim.0
  %32 = getelementptr inbounds [2 x i32], ptr %copy.9, i64 0, i64 %broadcast_add_fusion.1.kLoop_fusion.indvar.dim.0
  %33 = load i32, ptr %32, align 4, !alias.scope !10, !noalias !13
  %34 = getelementptr inbounds [2 x i32], ptr %copy.10, i64 0, i64 %broadcast_add_fusion.1.kLoop_fusion.indvar.dim.0
  %35 = load i32, ptr %34, align 4, !alias.scope !21, !noalias !22
  %36 = add i32 %33, %35
  %37 = getelementptr inbounds [2 x i32], ptr %copy.10, i64 0, i64 %broadcast_add_fusion.1.kLoop_fusion.indvar.dim.0
  %38 = load i32, ptr %37, align 4, !alias.scope !21, !noalias !22
  %39 = getelementptr inbounds [4 x i32], ptr %copy.14, i64 0, i64 0
  %40 = load i32, ptr %39, align 4, !alias.scope !23, !noalias !24
  %41 = shl i32 %38, %40
  %shft.chk = icmp ult i32 %40, 32
  %42 = select i1 %shft.chk, i32 %41, i32 0
  %43 = getelementptr inbounds [2 x i32], ptr %copy.10, i64 0, i64 %broadcast_add_fusion.1.kLoop_fusion.indvar.dim.0
  %44 = load i32, ptr %43, align 4, !alias.scope !21, !noalias !22
  %constant.24 = load i32, ptr @3, align 4
  %45 = sub i32 %constant.24, %40
  %46 = lshr i32 %44, %45
  %shft.chk1 = icmp ult i32 %45, 32
  %47 = select i1 %shft.chk1, i32 %46, i32 0
  %48 = or i32 %42, %47
  %49 = xor i32 %36, %48
  %50 = add i32 %36, %49
  %51 = getelementptr inbounds [4 x i32], ptr %copy.14, i64 0, i64 1
  %52 = load i32, ptr %51, align 4, !alias.scope !23, !noalias !24
  %53 = shl i32 %49, %52
  %shft.chk2 = icmp ult i32 %52, 32
  %54 = select i1 %shft.chk2, i32 %53, i32 0
  %constant.243 = load i32, ptr @3, align 4
  %55 = sub i32 %constant.243, %52
  %56 = lshr i32 %49, %55
  %shft.chk4 = icmp ult i32 %55, 32
  %57 = select i1 %shft.chk4, i32 %56, i32 0
  %58 = or i32 %54, %57
  %59 = xor i32 %50, %58
  %60 = add i32 %50, %59
  %61 = getelementptr inbounds [4 x i32], ptr %copy.14, i64 0, i64 2
  %62 = load i32, ptr %61, align 4, !alias.scope !23, !noalias !24
  %63 = shl i32 %59, %62
  %shft.chk5 = icmp ult i32 %62, 32
  %64 = select i1 %shft.chk5, i32 %63, i32 0
  %constant.246 = load i32, ptr @3, align 4
  %65 = sub i32 %constant.246, %62
  %66 = lshr i32 %59, %65
  %shft.chk7 = icmp ult i32 %65, 32
  %67 = select i1 %shft.chk7, i32 %66, i32 0
  %68 = or i32 %64, %67
  %69 = xor i32 %60, %68
  %70 = add i32 %60, %69
  %71 = load i32, ptr %copy.11, align 4, !alias.scope !27, !noalias !28
  %72 = add i32 %70, %71
  %73 = getelementptr inbounds [2 x i32], ptr %broadcast_add_fusion.1, i64 0, i64 %broadcast_add_fusion.1.kLoop_fusion.indvar.dim.0
  store i32 %72, ptr %73, align 4, !alias.scope !31, !noalias !32
  %invar.inc = add nuw nsw i64 %broadcast_add_fusion.1.kLoop_fusion.indvar.dim.0, 1
  store i64 %invar.inc, ptr %broadcast_add_fusion.1.kLoop_fusion.invar_address.dim.0, align 4
  br label %broadcast_add_fusion.1.kLoop_fusion.loop_header.dim.0

broadcast_add_fusion.1.kLoop_fusion.loop_exit.dim.0: ; preds = %broadcast_add_fusion.1.kLoop_fusion.loop_header.dim.0
  %74 = getelementptr inbounds ptr, ptr %buffer_table, i64 19
  %broadcast_add_fusion = load ptr, ptr %74, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  store i64 0, ptr %broadcast_add_fusion.kLoop_fusion.invar_address.dim.0, align 4
  br label %broadcast_add_fusion.kLoop_fusion.loop_header.dim.0

broadcast_add_fusion.kLoop_fusion.loop_header.dim.0: ; preds = %broadcast_add_fusion.kLoop_fusion.loop_body.dim.0, %broadcast_add_fusion.1.kLoop_fusion.loop_exit.dim.0
  %broadcast_add_fusion.kLoop_fusion.indvar.dim.0 = load i64, ptr %broadcast_add_fusion.kLoop_fusion.invar_address.dim.0, align 4
  %75 = icmp uge i64 %broadcast_add_fusion.kLoop_fusion.indvar.dim.0, 2
  br i1 %75, label %broadcast_add_fusion.kLoop_fusion.loop_exit.dim.0, label %broadcast_add_fusion.kLoop_fusion.loop_body.dim.0

broadcast_add_fusion.kLoop_fusion.loop_body.dim.0: ; preds = %broadcast_add_fusion.kLoop_fusion.loop_header.dim.0
  %76 = getelementptr inbounds [2 x i32], ptr %copy.9, i64 0, i64 %broadcast_add_fusion.kLoop_fusion.indvar.dim.0
  %77 = load i32, ptr %76, align 4, !alias.scope !10, !noalias !13
  %78 = getelementptr inbounds [2 x i32], ptr %copy.10, i64 0, i64 %broadcast_add_fusion.kLoop_fusion.indvar.dim.0
  %79 = load i32, ptr %78, align 4, !alias.scope !21, !noalias !22
  %80 = add i32 %77, %79
  %81 = getelementptr inbounds [2 x i32], ptr %copy.10, i64 0, i64 %broadcast_add_fusion.kLoop_fusion.indvar.dim.0
  %82 = load i32, ptr %81, align 4, !alias.scope !21, !noalias !22
  %83 = getelementptr inbounds [4 x i32], ptr %copy.14, i64 0, i64 0
  %84 = load i32, ptr %83, align 4, !alias.scope !23, !noalias !24
  %85 = shl i32 %82, %84
  %shft.chk9 = icmp ult i32 %84, 32
  %86 = select i1 %shft.chk9, i32 %85, i32 0
  %87 = getelementptr inbounds [2 x i32], ptr %copy.10, i64 0, i64 %broadcast_add_fusion.kLoop_fusion.indvar.dim.0
  %88 = load i32, ptr %87, align 4, !alias.scope !21, !noalias !22
  %constant.22 = load i32, ptr @5, align 4
  %89 = sub i32 %constant.22, %84
  %90 = lshr i32 %88, %89
  %shft.chk10 = icmp ult i32 %89, 32
  %91 = select i1 %shft.chk10, i32 %90, i32 0
  %92 = or i32 %86, %91
  %93 = xor i32 %80, %92
  %94 = add i32 %80, %93
  %95 = getelementptr inbounds [4 x i32], ptr %copy.14, i64 0, i64 1
  %96 = load i32, ptr %95, align 4, !alias.scope !23, !noalias !24
  %97 = shl i32 %93, %96
  %shft.chk11 = icmp ult i32 %96, 32
  %98 = select i1 %shft.chk11, i32 %97, i32 0
  %constant.2212 = load i32, ptr @5, align 4
  %99 = sub i32 %constant.2212, %96
  %100 = lshr i32 %93, %99
  %shft.chk13 = icmp ult i32 %99, 32
  %101 = select i1 %shft.chk13, i32 %100, i32 0
  %102 = or i32 %98, %101
  %103 = xor i32 %94, %102
  %104 = add i32 %94, %103
  %105 = getelementptr inbounds [4 x i32], ptr %copy.14, i64 0, i64 2
  %106 = load i32, ptr %105, align 4, !alias.scope !23, !noalias !24
  %107 = shl i32 %103, %106
  %shft.chk14 = icmp ult i32 %106, 32
  %108 = select i1 %shft.chk14, i32 %107, i32 0
  %constant.2215 = load i32, ptr @5, align 4
  %109 = sub i32 %constant.2215, %106
  %110 = lshr i32 %103, %109
  %shft.chk16 = icmp ult i32 %109, 32
  %111 = select i1 %shft.chk16, i32 %110, i32 0
  %112 = or i32 %108, %111
  %113 = xor i32 %104, %112
  %114 = add i32 %104, %113
  %115 = getelementptr inbounds [4 x i32], ptr %copy.14, i64 0, i64 3
  %116 = load i32, ptr %115, align 4, !alias.scope !23, !noalias !24
  %117 = shl i32 %113, %116
  %shft.chk17 = icmp ult i32 %116, 32
  %118 = select i1 %shft.chk17, i32 %117, i32 0
  %constant.2218 = load i32, ptr @5, align 4
  %119 = sub i32 %constant.2218, %116
  %120 = lshr i32 %113, %119
  %shft.chk19 = icmp ult i32 %119, 32
  %121 = select i1 %shft.chk19, i32 %120, i32 0
  %122 = or i32 %118, %121
  %123 = xor i32 %114, %122
  %124 = load i32, ptr %copy.12, align 4, !alias.scope !36, !noalias !37
  %125 = add i32 %123, %124
  %126 = load i32, ptr %copy.8, align 4, !alias.scope !38, !noalias !39
  %constant.23 = load i32, ptr @4, align 4
  %127 = add i32 %126, %constant.23
  %128 = add i32 %125, %127
  %129 = getelementptr inbounds [2 x i32], ptr %broadcast_add_fusion, i64 0, i64 %broadcast_add_fusion.kLoop_fusion.indvar.dim.0
  store i32 %128, ptr %129, align 4, !alias.scope !41, !noalias !42
  %invar.inc8 = add nuw nsw i64 %broadcast_add_fusion.kLoop_fusion.indvar.dim.0, 1
  store i64 %invar.inc8, ptr %broadcast_add_fusion.kLoop_fusion.invar_address.dim.0, align 4
  br label %broadcast_add_fusion.kLoop_fusion.loop_header.dim.0

broadcast_add_fusion.kLoop_fusion.loop_exit.dim.0: ; preds = %broadcast_add_fusion.kLoop_fusion.loop_header.dim.0
  %130 = getelementptr inbounds ptr, ptr %buffer_table, i64 31
  %add.52 = load ptr, ptr %130, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  %131 = load i32, ptr %copy.8, align 4, !alias.scope !38, !noalias !39
  %132 = load i32, ptr @constant.15, align 4, !alias.scope !43, !noalias !44
  %133 = add i32 %131, %132
  store i32 %133, ptr %add.52, align 4, !alias.scope !45, !noalias !46
  %134 = getelementptr inbounds ptr, ptr %buffer_table, i64 20
  %tuple.13 = load ptr, ptr %134, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  %135 = getelementptr inbounds [8 x ptr], ptr %tuple.13, i64 0, i64 0
  store ptr %add.52, ptr %135, align 8, !alias.scope !47, !noalias !48
  %136 = getelementptr inbounds [8 x ptr], ptr %tuple.13, i64 0, i64 1
  store ptr %broadcast_add_fusion.1, ptr %136, align 8, !alias.scope !47, !noalias !48
  %137 = getelementptr inbounds [8 x ptr], ptr %tuple.13, i64 0, i64 2
  store ptr %broadcast_add_fusion, ptr %137, align 8, !alias.scope !47, !noalias !48
  %138 = getelementptr inbounds [8 x ptr], ptr %tuple.13, i64 0, i64 3
  store ptr %copy.19, ptr %138, align 8, !alias.scope !47, !noalias !48
  %139 = getelementptr inbounds [8 x ptr], ptr %tuple.13, i64 0, i64 4
  store ptr %copy.20, ptr %139, align 8, !alias.scope !47, !noalias !48
  %140 = getelementptr inbounds [8 x ptr], ptr %tuple.13, i64 0, i64 5
  store ptr %copy.21, ptr %140, align 8, !alias.scope !47, !noalias !48
  %141 = getelementptr inbounds [8 x ptr], ptr %tuple.13, i64 0, i64 6
  store ptr %copy.22, ptr %141, align 8, !alias.scope !47, !noalias !48
  %142 = getelementptr inbounds [8 x ptr], ptr %tuple.13, i64 0, i64 7
  store ptr %copy.23, ptr %142, align 8, !alias.scope !47, !noalias !48
  br label %return
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: alwaysinline uwtable
define internal void @while.6__1(ptr %retval, ptr noalias %run_options, ptr noalias %params, ptr noalias %buffer_table, ptr noalias %status, ptr noalias %prof_counters) #1 {
entry:
  %0 = getelementptr inbounds ptr, ptr %buffer_table, i64 20
  %arg_tuple.3 = load ptr, ptr %0, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  %1 = getelementptr inbounds ptr, ptr %buffer_table, i64 31
  %2 = load ptr, ptr %1, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  %3 = getelementptr inbounds ptr, ptr %buffer_table, i64 29
  %lt.1 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  %4 = load i32, ptr %2, align 4, !alias.scope !45, !noalias !49
  %5 = load i32, ptr @constant.14, align 4, !alias.scope !52, !noalias !53
  %6 = icmp slt i32 %4, %5
  %7 = zext i1 %6 to i8
  store i8 %7, ptr %lt.1, align 1, !alias.scope !54, !noalias !55
  br label %return

return:                                           ; preds = %entry
  ret void
}

; Function Attrs: alwaysinline uwtable
define internal void @while.5_computation(ptr %retval, ptr noalias %run_options, ptr noalias %params, ptr noalias %buffer_table, ptr noalias %status, ptr noalias %prof_counters) #1 {
entry:
  %0 = getelementptr inbounds ptr, ptr %buffer_table, i64 20
  %tuple.14 = load ptr, ptr %0, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  %1 = getelementptr inbounds ptr, ptr %buffer_table, i64 20
  %while.6 = load ptr, ptr %1, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  br label %while.6.header

return:                                           ; preds = %while.6.exit
  ret void

while.6.header:                                   ; preds = %while.6.body, %entry
  call void @while.6__1(ptr null, ptr %run_options, ptr null, ptr %buffer_table, ptr %status, ptr %prof_counters)
  %2 = getelementptr inbounds ptr, ptr %buffer_table, i64 29
  %3 = load ptr, ptr %2, align 8, !invariant.load !3, !dereferenceable !9, !align !5
  %4 = load i8, ptr %3, align 1
  %5 = icmp ne i8 %4, 0
  br i1 %5, label %while.6.body, label %while.6.exit

while.6.body:                                     ; preds = %while.6.header
  call void @while.6(ptr null, ptr %run_options, ptr null, ptr %buffer_table, ptr %status, ptr %prof_counters)
  br label %while.6.header

while.6.exit:                                     ; preds = %while.6.header
  br label %return
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline uwtable "denormal-fp-math"="preserve-sign" "no-frame-pointer-elim"="false" }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!xla_cpu_memory_region_name = !{!0, !1}
!llvm.module.flags = !{!2}

!0 = !{!"xla_cpu_emitter__computation_kernel_emitter__hlo_opcode__call"}
!1 = !{!"ir_emitter"}
!2 = !{i32 1, !"xla_dylib_index", i64 0}
!3 = !{}
!4 = !{i64 16}
!5 = !{i64 64}
!6 = !{i64 4}
!7 = !{i64 8}
!8 = !{i64 1}
!9 = !{i64 964}
!10 = !{!11}
!11 = !{!"buffer: {index:7, offset:320, size:8}", !12}
!12 = !{!"XLA global AA domain"}
!13 = !{!14, !15, !16, !17, !18, !19, !20}
!14 = !{!"buffer: {index:7, offset:64, size:16}", !12}
!15 = !{!"buffer: {index:7, offset:256, size:8}", !12}
!16 = !{!"buffer: {index:7, offset:384, size:8}", !12}
!17 = !{!"buffer: {index:7, offset:448, size:8}", !12}
!18 = !{!"buffer: {index:7, offset:512, size:4}", !12}
!19 = !{!"buffer: {index:7, offset:576, size:4}", !12}
!20 = !{!"buffer: {index:7, offset:640, size:4}", !12}
!21 = !{!15}
!22 = !{!14, !11, !16, !17, !18, !19, !20}
!23 = !{!14}
!24 = !{!25, !26, !15, !11, !16, !17, !18, !19, !20}
!25 = !{!"buffer: {index:1, offset:0, size:16}", !12}
!26 = !{!"buffer: {index:7, offset:192, size:16}", !12}
!27 = !{!20}
!28 = !{!14, !15, !11, !16, !29, !30}
!29 = !{!"buffer: {index:7, offset:832, size:4}", !12}
!30 = !{!"buffer: {index:7, offset:960, size:4}", !12}
!31 = !{!16}
!32 = !{!25, !33, !14, !26, !15, !11, !17, !20, !34, !29, !35, !30}
!33 = !{!"buffer: {index:7, offset:0, size:64}", !12}
!34 = !{!"buffer: {index:7, offset:768, size:4}", !12}
!35 = !{!"buffer: {index:7, offset:896, size:4}", !12}
!36 = !{!18}
!37 = !{!14, !15, !11, !17, !19, !29, !35}
!38 = !{!19}
!39 = !{!40, !14, !15, !11, !17, !18, !34}
!40 = !{!"buffer: {index:0, offset:0, size:4}", !12}
!41 = !{!17}
!42 = !{!25, !33, !14, !26, !15, !11, !16, !18, !19, !34, !29, !35, !30}
!43 = !{!40}
!44 = !{!19, !34}
!45 = !{!34}
!46 = !{!40, !25, !33, !26, !16, !17, !19, !29, !35, !30}
!47 = !{!33}
!48 = !{!25, !26, !16, !17, !34, !29, !35, !30}
!49 = !{!50, !51}
!50 = !{!"buffer: {index:6, offset:0, size:4}", !12}
!51 = !{!"buffer: {index:7, offset:64, size:1}", !12}
!52 = !{!50}
!53 = !{!51, !34}
!54 = !{!51}
!55 = !{!50, !34}
