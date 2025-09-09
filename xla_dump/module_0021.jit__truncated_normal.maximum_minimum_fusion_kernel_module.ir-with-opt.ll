; ModuleID = '__compute_module_maximum_minimum_fusion_kernel_module'
source_filename = "__compute_module_maximum_minimum_fusion_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@llvm.compiler.used = appending global [2 x ptr] [ptr @xla.log1p.v2f32, ptr @xla.log1p.v4f32], section "llvm.metadata"

; Function Attrs: uwtable
define noalias noundef ptr @maximum_minimum_fusion(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = load ptr, ptr %5, align 8, !invariant.load !3, !dereferenceable !4
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %8 = load ptr, ptr %7, align 8, !invariant.load !3, !dereferenceable !5
  %9 = getelementptr inbounds nuw i8, ptr %3, i64 48
  %10 = load ptr, ptr %9, align 8, !invariant.load !3, !dereferenceable !5
  %11 = getelementptr inbounds nuw i8, ptr %3, i64 64
  %12 = load ptr, ptr %11, align 8, !invariant.load !3, !dereferenceable !4
  %13 = getelementptr inbounds nuw i8, ptr %3, i64 80
  %14 = load ptr, ptr %13, align 8, !invariant.load !3, !dereferenceable !4
  %15 = getelementptr inbounds nuw i8, ptr %3, i64 96
  %16 = load ptr, ptr %15, align 8, !invariant.load !3, !dereferenceable !5
  tail call void @llvm.experimental.noalias.scope.decl(metadata !6)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !9)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !11)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !13)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !15)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !17)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !19)
  %17 = load float, ptr %4, align 4, !invariant.load !3, !alias.scope !6, !noalias !21
  %18 = load float, ptr %6, align 4, !invariant.load !3, !alias.scope !9, !noalias !24
  %19 = fsub float %17, %18
  %20 = load i32, ptr %12, align 4, !invariant.load !3, !alias.scope !15, !noalias !25
  %21 = sitofp i32 %20 to float
  %22 = bitcast float %21 to i32
  %23 = tail call float @llvm.fabs.f32(float %21)
  %24 = bitcast float %23 to i32
  %25 = icmp samesign ugt i32 %24, 2139095040
  %26 = icmp slt i32 %20, 0
  %27 = or i1 %26, %25
  %28 = select i1 %27, i32 -1, i32 1
  %29 = icmp eq i32 %24, 0
  %30 = add i32 %28, %22
  %31 = select i1 %29, i32 1, i32 %30
  %32 = load i32, ptr %14, align 4, !invariant.load !3, !alias.scope !17, !noalias !26
  %33 = sitofp i32 %32 to float
  %34 = bitcast float %33 to i32
  %35 = tail call float @llvm.fabs.f32(float %33)
  %36 = bitcast float %35 to i32
  %37 = icmp samesign ugt i32 %36, 2139095040
  %38 = icmp sgt i32 %32, -1
  %39 = or i1 %38, %37
  %40 = select i1 %39, i32 -1, i32 1
  %41 = icmp eq i32 %36, 0
  %42 = add i32 %40, %34
  %43 = select i1 %41, i32 -2147483647, i32 %42
  %44 = insertelement <8 x i32> poison, i32 %43, i64 0
  %broadcast.splatinsert7 = bitcast <8 x i32> %44 to <8 x float>
  %broadcast.splat8 = shufflevector <8 x float> %broadcast.splatinsert7, <8 x float> poison, <8 x i32> zeroinitializer
  %45 = insertelement <8 x i32> poison, i32 %31, i64 0
  %broadcast.splatinsert5 = bitcast <8 x i32> %45 to <8 x float>
  %broadcast.splat6 = shufflevector <8 x float> %broadcast.splatinsert5, <8 x float> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert3 = insertelement <8 x float> poison, float %18, i64 0
  %broadcast.splat4 = shufflevector <8 x float> %broadcast.splatinsert3, <8 x float> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert = insertelement <8 x float> poison, float %19, i64 0
  %broadcast.splat = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <8 x i32> zeroinitializer
  br label %vector.ph

vector.ph:                                        ; preds = %1, %vector.ph
  %46 = phi i64 [ 0, %1 ], [ %229, %vector.ph ]
  %47 = shl nuw nsw i64 %46, 4
  %48 = getelementptr inbounds nuw [512 x i32], ptr %8, i64 0, i64 %47
  %wide.load = load <8 x i32>, ptr %48, align 4, !invariant.load !3, !alias.scope !11, !noalias !27
  %49 = getelementptr inbounds nuw [512 x i32], ptr %10, i64 0, i64 %47
  %wide.load9 = load <8 x i32>, ptr %49, align 4, !invariant.load !3, !alias.scope !13, !noalias !28
  %50 = xor <8 x i32> %wide.load9, %wide.load
  %51 = lshr <8 x i32> %50, splat (i32 9)
  %52 = or disjoint <8 x i32> %51, splat (i32 1065353216)
  %53 = bitcast <8 x i32> %52 to <8 x float>
  %54 = fadd <8 x float> %53, splat (float -1.000000e+00)
  %55 = fmul <8 x float> %broadcast.splat, %54
  %56 = fadd <8 x float> %broadcast.splat4, %55
  %57 = tail call <8 x float> @llvm.maximum.v8f32(<8 x float> %broadcast.splat4, <8 x float> %56)
  %58 = fneg <8 x float> %57
  %59 = fmul <8 x float> %57, %58
  %60 = fadd <8 x float> %59, splat (float 1.000000e+00)
  %61 = call <8 x float> @llvm.log.v8f32(<8 x float> %60)
  %62 = fmul <8 x float> %59, %59
  %63 = fmul <8 x float> %59, zeroinitializer
  %64 = fadd <8 x float> %63, splat (float 1.000000e+00)
  %65 = fmul <8 x float> %64, %59
  %66 = fadd <8 x float> %65, splat (float 0x402E2035A0000000)
  %67 = fmul <8 x float> %66, %59
  %68 = fadd <8 x float> %67, splat (float 0x4054C30B60000000)
  %69 = fmul <8 x float> %68, %59
  %70 = fadd <8 x float> %69, splat (float 0x406BB865A0000000)
  %71 = fmul <8 x float> %70, %59
  %72 = fadd <8 x float> %71, splat (float 0x4073519460000000)
  %73 = fmul <8 x float> %72, %59
  %74 = fadd <8 x float> %73, splat (float 0x406B0DB140000000)
  %75 = fmul <8 x float> %74, %59
  %76 = fadd <8 x float> %75, splat (float 0x404E0F3040000000)
  %77 = fadd <8 x float> %63, splat (float 0x3F07BC0960000000)
  %78 = fmul <8 x float> %77, %59
  %79 = fadd <8 x float> %78, splat (float 0x3FDFE818A0000000)
  %80 = fmul <8 x float> %79, %59
  %81 = fadd <8 x float> %80, splat (float 0x401A509F40000000)
  %82 = fmul <8 x float> %81, %59
  %83 = fadd <8 x float> %82, splat (float 0x403DE97380000000)
  %84 = fmul <8 x float> %83, %59
  %85 = fadd <8 x float> %84, splat (float 0x404E798EC0000000)
  %86 = fmul <8 x float> %85, %59
  %87 = fadd <8 x float> %86, splat (float 0x404C8E75A0000000)
  %88 = fmul <8 x float> %87, %59
  %89 = fadd <8 x float> %88, splat (float 0x40340A2020000000)
  %90 = fdiv <8 x float> %89, %76
  %91 = fmul <8 x float> %59, %62
  %92 = fmul <8 x float> %91, %90
  %93 = fmul <8 x float> %62, splat (float -5.000000e-01)
  %94 = fadd <8 x float> %93, %92
  %95 = fadd <8 x float> %59, %94
  %96 = call <8 x float> @llvm.fabs.v8f32(<8 x float> %59)
  %97 = fcmp olt <8 x float> %96, splat (float 0x3FDA8279A0000000)
  %98 = select <8 x i1> %97, <8 x float> %95, <8 x float> %61
  %99 = fneg <8 x float> %98
  %100 = fcmp ogt <8 x float> %98, splat (float -5.000000e+00)
  %101 = select <8 x i1> %100, <8 x float> splat (float 0x3E5E2CB100000000), <8 x float> splat (float 0xBF2A3E1360000000)
  %102 = select <8 x i1> %100, <8 x float> splat (float 0x3E970966C0000000), <8 x float> splat (float 0x3F1A76AD60000000)
  %103 = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %99)
  %104 = fsub <8 x float> splat (float -2.500000e+00), %98
  %105 = fadd <8 x float> %103, splat (float -3.000000e+00)
  %106 = select <8 x i1> %100, <8 x float> %104, <8 x float> %105
  %107 = fmul <8 x float> %101, %106
  %108 = fadd <8 x float> %102, %107
  %109 = select <8 x i1> %100, <8 x float> splat (float 0xBECD8E6AE0000000), <8 x float> splat (float 0x3F561B8E40000000)
  %110 = fmul <8 x float> %106, %108
  %111 = fadd <8 x float> %109, %110
  %112 = select <8 x i1> %100, <8 x float> splat (float 0xBED26B5820000000), <8 x float> splat (float 0xBF6E17BCE0000000)
  %113 = fmul <8 x float> %106, %111
  %114 = fadd <8 x float> %112, %113
  %115 = select <8 x i1> %100, <8 x float> splat (float 0x3F2CA65B60000000), <8 x float> splat (float 0x3F77824F60000000)
  %116 = fmul <8 x float> %106, %114
  %117 = fadd <8 x float> %115, %116
  %118 = select <8 x i1> %100, <8 x float> splat (float 0xBF548A8100000000), <8 x float> splat (float 0xBF7F38BAE0000000)
  %119 = fmul <8 x float> %106, %117
  %120 = fadd <8 x float> %118, %119
  %121 = select <8 x i1> %100, <8 x float> splat (float 0xBF711C9DE0000000), <8 x float> splat (float 0x3F8354AFC0000000)
  %122 = fmul <8 x float> %106, %120
  %123 = fadd <8 x float> %121, %122
  %124 = select <8 x i1> %100, <8 x float> splat (float 0x3FCF91EC60000000), <8 x float> splat (float 0x3FF006DB60000000)
  %125 = fmul <8 x float> %106, %123
  %126 = fadd <8 x float> %124, %125
  %127 = select <8 x i1> %100, <8 x float> splat (float 0x3FF805C5E0000000), <8 x float> splat (float 0x4006A9EFC0000000)
  %128 = fmul <8 x float> %106, %126
  %129 = tail call <8 x float> @llvm.fabs.v8f32(<8 x float> %57)
  %130 = fadd <8 x float> %127, %128
  %131 = fcmp oeq <8 x float> %129, splat (float 1.000000e+00)
  %132 = select <8 x i1> %131, <8 x float> splat (float 0x7FF0000000000000), <8 x float> %130
  %133 = fmul <8 x float> %57, %132
  %134 = fmul <8 x float> %133, splat (float 0x3FF6A09E60000000)
  %135 = tail call <8 x float> @llvm.maximum.v8f32(<8 x float> %broadcast.splat6, <8 x float> %134)
  %136 = tail call <8 x float> @llvm.minimum.v8f32(<8 x float> %broadcast.splat8, <8 x float> %135)
  %137 = getelementptr inbounds nuw [512 x float], ptr %16, i64 0, i64 %47
  store <8 x float> %136, ptr %137, align 4, !alias.scope !29, !noalias !30
  %138 = or disjoint i64 %47, 8
  %139 = getelementptr inbounds nuw [512 x i32], ptr %8, i64 0, i64 %138
  %wide.load.1 = load <8 x i32>, ptr %139, align 4, !invariant.load !3, !alias.scope !11, !noalias !27
  %140 = getelementptr inbounds nuw [512 x i32], ptr %10, i64 0, i64 %138
  %wide.load9.1 = load <8 x i32>, ptr %140, align 4, !invariant.load !3, !alias.scope !13, !noalias !28
  %141 = xor <8 x i32> %wide.load9.1, %wide.load.1
  %142 = lshr <8 x i32> %141, splat (i32 9)
  %143 = or disjoint <8 x i32> %142, splat (i32 1065353216)
  %144 = bitcast <8 x i32> %143 to <8 x float>
  %145 = fadd <8 x float> %144, splat (float -1.000000e+00)
  %146 = fmul <8 x float> %broadcast.splat, %145
  %147 = fadd <8 x float> %broadcast.splat4, %146
  %148 = tail call <8 x float> @llvm.maximum.v8f32(<8 x float> %broadcast.splat4, <8 x float> %147)
  %149 = fneg <8 x float> %148
  %150 = fmul <8 x float> %148, %149
  %151 = fadd <8 x float> %150, splat (float 1.000000e+00)
  %152 = call <8 x float> @llvm.log.v8f32(<8 x float> %151)
  %153 = fmul <8 x float> %150, %150
  %154 = fmul <8 x float> %150, zeroinitializer
  %155 = fadd <8 x float> %154, splat (float 1.000000e+00)
  %156 = fmul <8 x float> %155, %150
  %157 = fadd <8 x float> %156, splat (float 0x402E2035A0000000)
  %158 = fmul <8 x float> %157, %150
  %159 = fadd <8 x float> %158, splat (float 0x4054C30B60000000)
  %160 = fmul <8 x float> %159, %150
  %161 = fadd <8 x float> %160, splat (float 0x406BB865A0000000)
  %162 = fmul <8 x float> %161, %150
  %163 = fadd <8 x float> %162, splat (float 0x4073519460000000)
  %164 = fmul <8 x float> %163, %150
  %165 = fadd <8 x float> %164, splat (float 0x406B0DB140000000)
  %166 = fmul <8 x float> %165, %150
  %167 = fadd <8 x float> %166, splat (float 0x404E0F3040000000)
  %168 = fadd <8 x float> %154, splat (float 0x3F07BC0960000000)
  %169 = fmul <8 x float> %168, %150
  %170 = fadd <8 x float> %169, splat (float 0x3FDFE818A0000000)
  %171 = fmul <8 x float> %170, %150
  %172 = fadd <8 x float> %171, splat (float 0x401A509F40000000)
  %173 = fmul <8 x float> %172, %150
  %174 = fadd <8 x float> %173, splat (float 0x403DE97380000000)
  %175 = fmul <8 x float> %174, %150
  %176 = fadd <8 x float> %175, splat (float 0x404E798EC0000000)
  %177 = fmul <8 x float> %176, %150
  %178 = fadd <8 x float> %177, splat (float 0x404C8E75A0000000)
  %179 = fmul <8 x float> %178, %150
  %180 = fadd <8 x float> %179, splat (float 0x40340A2020000000)
  %181 = fdiv <8 x float> %180, %167
  %182 = fmul <8 x float> %150, %153
  %183 = fmul <8 x float> %182, %181
  %184 = fmul <8 x float> %153, splat (float -5.000000e-01)
  %185 = fadd <8 x float> %184, %183
  %186 = fadd <8 x float> %150, %185
  %187 = call <8 x float> @llvm.fabs.v8f32(<8 x float> %150)
  %188 = fcmp olt <8 x float> %187, splat (float 0x3FDA8279A0000000)
  %189 = select <8 x i1> %188, <8 x float> %186, <8 x float> %152
  %190 = fneg <8 x float> %189
  %191 = fcmp ogt <8 x float> %189, splat (float -5.000000e+00)
  %192 = select <8 x i1> %191, <8 x float> splat (float 0x3E5E2CB100000000), <8 x float> splat (float 0xBF2A3E1360000000)
  %193 = select <8 x i1> %191, <8 x float> splat (float 0x3E970966C0000000), <8 x float> splat (float 0x3F1A76AD60000000)
  %194 = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %190)
  %195 = fsub <8 x float> splat (float -2.500000e+00), %189
  %196 = fadd <8 x float> %194, splat (float -3.000000e+00)
  %197 = select <8 x i1> %191, <8 x float> %195, <8 x float> %196
  %198 = fmul <8 x float> %192, %197
  %199 = fadd <8 x float> %193, %198
  %200 = select <8 x i1> %191, <8 x float> splat (float 0xBECD8E6AE0000000), <8 x float> splat (float 0x3F561B8E40000000)
  %201 = fmul <8 x float> %197, %199
  %202 = fadd <8 x float> %200, %201
  %203 = select <8 x i1> %191, <8 x float> splat (float 0xBED26B5820000000), <8 x float> splat (float 0xBF6E17BCE0000000)
  %204 = fmul <8 x float> %197, %202
  %205 = fadd <8 x float> %203, %204
  %206 = select <8 x i1> %191, <8 x float> splat (float 0x3F2CA65B60000000), <8 x float> splat (float 0x3F77824F60000000)
  %207 = fmul <8 x float> %197, %205
  %208 = fadd <8 x float> %206, %207
  %209 = select <8 x i1> %191, <8 x float> splat (float 0xBF548A8100000000), <8 x float> splat (float 0xBF7F38BAE0000000)
  %210 = fmul <8 x float> %197, %208
  %211 = fadd <8 x float> %209, %210
  %212 = select <8 x i1> %191, <8 x float> splat (float 0xBF711C9DE0000000), <8 x float> splat (float 0x3F8354AFC0000000)
  %213 = fmul <8 x float> %197, %211
  %214 = fadd <8 x float> %212, %213
  %215 = select <8 x i1> %191, <8 x float> splat (float 0x3FCF91EC60000000), <8 x float> splat (float 0x3FF006DB60000000)
  %216 = fmul <8 x float> %197, %214
  %217 = fadd <8 x float> %215, %216
  %218 = select <8 x i1> %191, <8 x float> splat (float 0x3FF805C5E0000000), <8 x float> splat (float 0x4006A9EFC0000000)
  %219 = fmul <8 x float> %197, %217
  %220 = tail call <8 x float> @llvm.fabs.v8f32(<8 x float> %148)
  %221 = fadd <8 x float> %218, %219
  %222 = fcmp oeq <8 x float> %220, splat (float 1.000000e+00)
  %223 = select <8 x i1> %222, <8 x float> splat (float 0x7FF0000000000000), <8 x float> %221
  %224 = fmul <8 x float> %148, %223
  %225 = fmul <8 x float> %224, splat (float 0x3FF6A09E60000000)
  %226 = tail call <8 x float> @llvm.maximum.v8f32(<8 x float> %broadcast.splat6, <8 x float> %225)
  %227 = tail call <8 x float> @llvm.minimum.v8f32(<8 x float> %broadcast.splat8, <8 x float> %226)
  %228 = getelementptr inbounds nuw [512 x float], ptr %16, i64 0, i64 %138
  store <8 x float> %227, ptr %228, align 4, !alias.scope !29, !noalias !30
  %229 = add nuw nsw i64 %46, 1
  %exitcond2.not = icmp eq i64 %229, 32
  br i1 %exitcond2.not, label %maximum_minimum_fusion_wrapped.exit, label %vector.ph

maximum_minimum_fusion_wrapped.exit:              ; preds = %vector.ph
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #2

declare <2 x float> @xla.log1p.v2f32(<2 x float>) local_unnamed_addr

declare <4 x float> @xla.log1p.v4f32(<4 x float>) local_unnamed_addr

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.maximum.v8f32(<8 x float>, <8 x float>) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.fabs.v8f32(<8 x float>) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.minimum.v8f32(<8 x float>, <8 x float>) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.log.v8f32(<8 x float>) #3

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 10}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 4}
!5 = !{i64 2048}
!6 = !{!7}
!7 = distinct !{!7, !8, !"maximum_minimum_fusion_wrapped: argument 0"}
!8 = distinct !{!8, !"maximum_minimum_fusion_wrapped"}
!9 = !{!10}
!10 = distinct !{!10, !8, !"maximum_minimum_fusion_wrapped: argument 1"}
!11 = !{!12}
!12 = distinct !{!12, !8, !"maximum_minimum_fusion_wrapped: argument 2"}
!13 = !{!14}
!14 = distinct !{!14, !8, !"maximum_minimum_fusion_wrapped: argument 3"}
!15 = !{!16}
!16 = distinct !{!16, !8, !"maximum_minimum_fusion_wrapped: argument 4"}
!17 = !{!18}
!18 = distinct !{!18, !8, !"maximum_minimum_fusion_wrapped: argument 5"}
!19 = !{!20}
!20 = distinct !{!20, !8, !"maximum_minimum_fusion_wrapped: argument 6"}
!21 = !{!22, !10, !12, !14, !16, !18, !20}
!22 = distinct !{!22, !23, !"xla.slice_index=6"}
!23 = distinct !{!23}
!24 = !{!22, !7, !12, !14, !16, !18, !20}
!25 = !{!22, !7, !10, !12, !14, !18, !20}
!26 = !{!22, !7, !10, !12, !14, !16, !20}
!27 = !{!22, !7, !10, !14, !16, !18, !20}
!28 = !{!22, !7, !10, !12, !16, !18, !20}
!29 = !{!22, !20}
!30 = !{!7, !10, !12, !14, !16, !18}
