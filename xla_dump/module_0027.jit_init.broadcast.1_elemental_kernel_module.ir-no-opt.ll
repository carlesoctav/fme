; ModuleID = '__compute_module_broadcast.1_elemental_kernel_module'
source_filename = "__compute_module_broadcast.1_elemental_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_NumWorkGroups = type { i64, i64, i64 }
%XLA_CPU_WorkGroupId = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@0 = private unnamed_addr constant [2048 x i8] c"E\8E\FC\BEA\EA;>\AC\AE\22=j(B\BE\87(L>p\9A0>\D9\8Em>\88t\9D\BD7\A8m=\06g\8C\BD\96e\A0\BE\CD\04\F5\BEO\9B\AA\BDk\09\81=\F1\AD/=\06\D2\18\BE\9A\99\DE\BE\F9Z\83=\9D\D7\80>C\E6\A5>\18td=bF!>p2F=\9F6\96=\D6\C9\85>)\FC\0C\BDl\14\AC>Y\EC\C6>\CF\06\8C>t\CB\F3\BDO]\B4>\F3\CD\C2;\9C\05\02\BFP\B7i>\C6\0B\D5=C\9F\AE>\B2fx=\BB\88\D9\BE\9A\F5\E4>\DE\E0\D1\BE)\A1c\BE\A5<\96=\90\E6\E6>\C0K\DD\BEY$\F6>0\F4\B2>\A2\FF\89\BE\E0)\00>\C1\B8>\BE\94\E1\AD\BD\DB\CC\C1;KJ\F9\BD\05\C1~>\139\06\BD\C9:r=#(\A8\BC\B7\EF\C8\BEkH\B4=~Y#\BD\ACXg\BC\FC*s\BE\AFG\C9\BE\A2\9EB>\FA4\C7\BD\83Y\119\9Cl\D3\BE\0B\B8*\BE7\9D\8B\BD{B\B0\BEh\C9;\BE\FF\C6\84\BE)\86U>nKc\BC\01]\0C>$\1E\93>\A4h\89>ijP>\C8T^\BEQ\18,>\AEBM\BE\DF\A3\C1>I\F2@=oCD>\C38s\BEec\B4=)X\ED=`]\AB\BC\D2\8D_>\0B\1A\F6\BE\FE\A6q\BE\EDVh>'#6\BEEh,\BE\9Ff\F1=7p\91\BE.\AD\B8>\B0\A6\9B>\0ET\E6>\02\C0\B1\BE:\93\FD\BD=\F2\DB>J\BE\00>\90\9F\91\BDN\BDI\BE\82\F6\01?\9C\1D\0B\BE\15\04r\BEZ<6\BE0\9B\80\BD\14\B6a>\B7\B1\D1>\F4m\D4>?\1D\AE>?\99}=\94\80\9A=r\88\B8<f\B0\06>\9E\82\C5>\BD\8F\EC\BE\FD\EC\1E\BE\EF\E8=\BERc\8F\BD\BCf\F3\BD\C0\E0t\BET#\B8\BDP\CA\9C=|\D6\B0\BE\BCMw>6\8C\FC>\C2\D2\B5>\A7\06[=\BD\CF\08?\B7\AFS\BE\13`4\BE\DCpF>Rf\04>_\B2]\BC&\18$==\D4\9F>3M\AF\BE\EA\DD\E7=\1Da\B4>\FE;[\BC\0D\97\9C\BE\18\A4\96>:\05:\BE\1D\B9\A5\BD0\BD\0C\BEx\DC\8C\BE\85\92\0B?\FF\A0K>\E9\FE\0F\BD\92\E63>X?\98\BC8A\8F>\BA8\C0>\1Bc\1F<[\D1\FF\BE\C8\17\E2\BE=\ED\22>'3w<[\FA\86\BD\8A\CE(\BEx\EDm< \FF\E6\BE\D9\AF#\BD\C0\22o=\D1\AA\A9=l\BF7\BE\DEFJ=\15L\B5>+\0F\8C>\DF\B38>b}\B9\BEE\00Y\BE\B4!\EF\BE\8C\97%>\84\AA\04>\B7\BA\EC>-Y\01?\EANU\BE\DD\A7\89\BE\16]\FF\BE\80\B8\EF\BD#]4\BE=a\B4\BC\09[\B2>\CE\F8\81\BED\12\1B>.\EC\EE\BDA\F54\BEgaU>1\BA\84\BC\B0\E1F\BD\EF\E2O\BE\88\9A\EE\BE\9A\E4\0A<\C9e\95>?\EA\91\BD~\82\80\BD\D3t\03<\C5\E5d\BE\B5O\82>n\92\EA\BE0n\CE\BD\EF\C2\EB;\BB\01\D2=\AB\9C\8C\BE?\E5\C2\BC\16\E7\FB\BER\0A\87\BEO\80\E9>\91W\0F>\F8J\8D>\05\DA\03>[\A3\AB\BE@\CA\C2>\82\BC\C8\BC\BF\C3\C4\BE\BA\D4o=\1D\E8\F7<\05\EB\97>R\DA\A2\BE?\06\8C=-.\AF>Po\AC\BCe\\Z\BEkk\EB\BE:\A9T\BE\00\80\C4\BE\16\FE\9C>Pzc>`>\AB\BE\E4Y4;qa4>\F6-\B8\BD\8EM\8D>\80\19\DF\BE:k\85\BE\FA\A4D>\1E\B5\E4=R \AC\BD\94o\DC>\1C\FC\8A\BE\C3\19\09>M\D8\DD\BEiw\03>\DE\D4.\BD\9F\1C;=B\DC\07\BF\02+\AC\BE\CA\08\C3\BDUAI=\BB\BA\C8\BE\92p\AB\BE\D8\C8w>\9A\F8\C1>0\7F\A2\BE\B6\A4\86\BE\E0\89\DD=\0B\A6\0C>\AEB\9A=\B5\FE\17\BEzA\17\BE\97v\C7=P\0C%\BE(n\8C>\0E;\9D\BD(\05\0F\BE\C2B\BE=\F6=\E0\BD4I,>\D8Ka\BDE\A2\97\BC\C06P>Ft\98\BD\FF[Y=\0E\18F\BE~\F0`\BEVu\BE=\D8\05\04\BF\0C\DD\87\BE9\12J\BE\C94\9F\BE\97\EC\16\BE\10\89\BD=C(|>|\F9\1C>$\C0\85>5\E7\EC=\B4\0C\83\BE\E7s\AB\BD\E1\D1\94>\1Ac\8C>9J@\BE\E1{\0E\BE\7F[|>\A5\B1\93\BE\19\E1I\BC1G\E1>d!B=\92~/=\F8\DF\D9\BD\DB\87q>_F\BF=&Fd\BE\BB\17\83>\02\B16>2u\04>\C9\BC>>\D5\8B\F6\BE%I\80>Xc{\BEH\B7\04=\22\EC\14\BD\DA\EC^=%\FE\D9=\8E\13_=\0E\B9\96=\F4\E5L>\F5\07\AF\BEC\FFW\BE*\C6\FB=\99\9A7\BE\F2fF>\\\FC\92>\E9\90\E5<q\BE\E2\BE\F2\0D\02\BE\0F\9F\8C<H\0E\B4>\12\A9\E5>\7F-v\BD\87\19\07?\D9\B0\D6\BD\EBHV=\A5;k>\1A\07\F7\BD)\81\D7=\EAS\C4=\B0\9D\EA=\D0\7Fz>y\ED\AA\BE*:%>\A1\D9\0B\BER@\0A>\FE!\0E\BF\EE\D8#\BE\E3\02\A5>\CC\\Y>Kx3\BE\96\BB\DE\BD#\87\BE:3X\A2>5.\15>\F0\F3\98>\F5\E3\DB>\EB\1F\A2\BE\9Fkw>P\80\07>\D9z\8B\BE-\85,>\EB_z=,\DE\E3>\9A\C1\17=P[\85>\1B\F5}>\02\C1==#\F1T>T\16\CE>\00\E1L\BE[\D8\E6\BDE\1Cp\BE\FFW\A0\BE\\\E5\BF>U\C1\15\BEL3p\BE\A4\1B\D2>\EEG\85>I+\D6\BE\81n\B2<@\D2\B9=\B9\CC\B4\BE\17\EC\AE>wft>\AAr\86>,\F2\8C>\09m`\BE\94\16\84>S.U=F\E3\CC\BDK9\EC\BBw\EC\91\BE\B1\09\C8\BE\BF;\08?\06\81\05\BDi\FF\85=\1F\AF\0D=~ZA:\B8\94\F7>\E5\9C\81\BE!H\F4\BE\86@]\BE\C0\D6\17\BE\8D7\B7><\AE\9A\BE\E1\B1\C1=\8FR\AE\BD\D7[}>\17\BD\AA\BE\A4-\81>\E4P\AB=\CE\E1T>a\E6\D9>\D8\C7\17\BE\F3Ra\BE\9A[v;\83Q\BE\BC\\N\CD=\18\18\A6\BE\E7\11N>\B5B\19=I#\84\BE\AD\ED\BC\BE\98o\BD>\A2\BD\AB=\F8\1B\FC\BE\EAGF>\A2Pp\BDY@w\BE\A9%\84>\DD\D2\94\BE\04\E0\8C\BDa\0C(\BDx\CC\A4\BC\B4K\F1>!4'\BE\F2/\\\BDjK\D6=*\0E\98\BER\8B\BF\BD\8E\1Az\BE\0C1V\BE\06\E2c\BE\A2\D8\1C\BE?\BE\0F?Vu\A9>\9Er_\BD\BB\9ED>.,\FE\BA\F6_\BF>\BB\FF\06?\D4\D8o\BEm\1C3\BE\15yJ>\C2@V\BEGH\89\BEE\B9\D5=\EBQ0\BE?\93\B3=_\F1\9E\BE\9C>\86=Q\F0\D1=-\BAj>(y\AD\BD\D9\E4\F5<\B5\A7\B1>\8F\1E\82<f\E90\BD$b\0C\BE\1F\99\AD<\F7\C3\94\BE\05zu\BE\B5\1F\BE\BC\F9\E77>\94\AA->\FC\F66=\EE2P\BD'\A8U\BC15\C0>\98\\\F3\BE\1Fr|\BD\CC\A8\B5\BDj\F7\F3>[\A4\DE\BEbC\87\BE\A2i\CF=\07\F2\1B\BE@\CD=\BE\E9\0D6>\15\BC\B6>\8D\22f<2\EC\BE\BE\0D\CCr>R\0A\B1=\1B\88\94\BE\83\E9\9B\BE\0A\1D\C6=\FEP\1F=/R\03>=mW\BD\09\97 \BE\C78\16>\82\BE\0B\BE\DFj\FF\BD\C5\CF\A2\BERt\89\BE\CA\84\97=<\19\18>\15\80o>\F2b\F7\BE(\9B\0F\BF", align 64
@1 = private unnamed_addr constant [512 x i8] c"p\93\E4\BA\AC\0Bm\BE\E1\1F\13?C\D8\9D=\10SD?\D0\B5\7F>\FD&\C0\BE]'F>\86\DB;?\9A\92%>\10\B6\1E?42\D9\BD\BEE}\BE\C9R/\BE\9Cm\87>\80n\8D\BE\0E\0D\B9=o\E0\DE\BC\89\9E\C3>\F8o\B0>\FF\FF\BB>\AA4\E4>J\10\88>O\AE<\BF\81C\8D>\E6\F0\85\BDK\AB\B1\BE\A9w=\BE\A5\93\0F>\11\EAc>\99\16I\BE\06\DB\EC\BEq\F9\12\BE\98\1B\0D>+J\BA\BE/m\FA\BE\7F\AE\D8\BC\FA\B1_>\D3CC\BC\C4?\C1\BE\9E\7F\A4\BE)\ED\05?*\C6\C9=XB\B6\BD\1F\C6\FF\BD\197\F0>\D0\BB\8A\BC)\BBT\BE\C4\D7\AF=\D7\\>\BE\15\FF\01>\A4\17\15\BE_\F5\1E\BE9Ww\BC\E5*\9D<\AA\B2w>Yj0\BF\A9~\D9>\C0\87\0D\BE\1B\1B>?Xw\A2=\\U0\BF\91\B9&\BFS\B6\F4;}\E4&\BD\8F'*\BE\DB\BC2\BF\B1&\95>\A7l{\BEo\16\C2>B\15\C7\BB}\06%\BE@\AE*\BF\0F\C1'?7\88\06> \D1\D2=\C4\90\8B=\B6\B1\F5>2\B0\0A\BE\ED\85\0A\BF\D8\F4'\BD\A1\16\87\BE:\06\14=o\A8\11\BFl=:>\046\BE\BE\87j?\BE\1Fgk\BE\10+n\BDY\F8\1E?!\F0\B3\BE\B8\91\8F>\B5\99\13?\DC\8E\E7>\D2a\BA\BE\8A\A0\AD\BE\BD`\82=\82q\D2\BE\D5\D5\BD=\9A\9A\E0=6\B2\9A>)\02\9B\BE\93\22\89\BEd\C8\B2=\8E-\83>R\94x\BC5:\C5\BEM\93\BA>\1E\B2\0B?\C5o#=\A6g\EE\BE;\AFF>A\9D\E3\BA*Q\D4\BD\EC\AC=>;\EB\97\BE\1A\05\A5=u\06\CA=\97\A8'?\1E\CC%>\9B\DET>\D6]\F8\BE\E5#\18\BE!\9A\C3<\BA\C5\9A>N\AB\04?A@\B8>\99;%\BF", align 64
@2 = private unnamed_addr constant [4 x i8] zeroinitializer, align 4

; Function Attrs: uwtable
define ptr @broadcast.1_kernel(ptr %0) #0 {
  %broadcast.1.invar_address.dim.0 = alloca i64, align 8
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
  store i64 0, ptr %broadcast.1.invar_address.dim.0, align 4
  br label %broadcast.1.loop_header.dim.0

broadcast.1.loop_header.dim.0:                    ; preds = %broadcast.1.loop_body.dim.0, %1
  %broadcast.1.indvar.dim.0 = load i64, ptr %broadcast.1.invar_address.dim.0, align 4
  %2 = icmp uge i64 %broadcast.1.indvar.dim.0, 32
  br i1 %2, label %broadcast.1.loop_exit.dim.0, label %broadcast.1.loop_body.dim.0

broadcast.1.loop_body.dim.0:                      ; preds = %broadcast.1.loop_header.dim.0
  %3 = load float, ptr %arg0, align 4, !invariant.load !3, !noalias !7
  %4 = getelementptr inbounds [32 x float], ptr %arg1, i64 0, i64 %broadcast.1.indvar.dim.0
  store float %3, ptr %4, align 4, !alias.scope !7
  %invar.inc = add nuw nsw i64 %broadcast.1.indvar.dim.0, 1
  store i64 %invar.inc, ptr %broadcast.1.invar_address.dim.0, align 4
  br label %broadcast.1.loop_header.dim.0

broadcast.1.loop_exit.dim.0:                      ; preds = %broadcast.1.loop_header.dim.0
  br label %return

return:                                           ; preds = %broadcast.1.loop_exit.dim.0
  ret ptr null
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!xla_cpu_memory_region_name = !{!0, !1}
!llvm.module.flags = !{!2}

!0 = !{!"xla_cpu_emitter__elemental_kernel_emitter__hlo_opcode__broadcast"}
!1 = !{!"ir_emitter"}
!2 = !{i32 1, !"xla_dylib_index", i64 1}
!3 = !{}
!4 = !{i64 4}
!5 = !{i64 64}
!6 = !{i64 128}
!7 = !{!8}
!8 = !{!"result slice: {index:4, offset:0, size:128}", !9}
!9 = !{!"XLA host kernel broadcast.1_kernel AA domain"}
