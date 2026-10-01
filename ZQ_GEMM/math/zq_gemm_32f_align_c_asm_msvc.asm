;-----------------------------------------------------------------------------
; zq_gemm_32f_align_c_asm_msvc.asm
;
; ZQ_GEMM 手写汇编微内核的 MSVC 版本 (MASM / ml64, Intel 语法)。
;
; 为什么需要这个文件: MSVC **x64 目标根本不支持函数体内联汇编**, 对 __asm {}
; 会直接报 C4235 (non-standard extension: '__asm' keyword is not supported in
; this context)。x86 32 位才支持 __asm, x64 只能用独立的 .asm 文件走 ml64。
; 所以 Windows 下这三个微内核用 MASM 写在��里, 由 zq_gemm_32f_align_c_asm.c
; 声明并调用; GCC/Clang 那边仍然用函数体内 __asm__ volatile (AT&T)。
; 指令序列两边完全一致。
;
; 三个微内核 (与 C 侧的静态内联汇编版一一对应):
;   zq_gemm_32f_asm_core_m2n4 : 2 行 x 4 列, 8 个 ymm 累加器
;   zq_gemm_32f_asm_core_m1n8 : 1 行 x 8 列, B 分两批加载
;   zq_gemm_32f_asm_core_m1n4 : 1 行 x 4 列
; 每个微内核只处理 K 维的前 k8*8 个元素 (k8 = K/8), 结果**覆盖写入** C 的 4
; (或 8) 个 float, K 尾部由 C 侧用标量累加补上; k8 == 0 时写出全 0。
;
; 调用约定 (x64 Windows): 前 4 个整型/指针参数在 rcx/rdx/r8/r9, 其余在栈上
; (第 5 个参数在 [rsp+28h], 调用者已预留 32 字节 shadow space)。
;
; !! callee-saved 浮点寄存器 !!
; Windows x64 ABI 与 System V AMD64 **不一样**:
;   - System V AMD64 (Linux/macOS): x87 与全部 xmm0-xmm31 都是 caller-saved,
;     随便用。
;   - Windows x64: 只有 xmm0-xmm5 是 volatile, **xmm6-xmm15 是非易失的
;     (callee-saved)**, 调用方会把值留在里面跨越调用。
; 本文件的微内核要用到 ymm0-ymm14 (8 个累加器 + 6 个操作数 + 1 个归约临时),
; 必然踩到 xmm6-xmm14。如果不保存/恢复, 返回后调用方的 xmm6-xmm15 就成了
; 垃圾 —— 表现为"调用者的 double 局部变量变成 -1.7e30 之类的乱码、耗时算出来是
; 1e30 / inf", 而且随 MSVC 怎么分配寄存器而变, 很难复现。
; (这正是 2026-10-01 Windows 端内存被破坏的根因; Linux 端因为 XMM 全是
;  caller-saved 所以一直没暴露。)
; 因此三个微内核统一用 ZQA_PROLOGUE / ZQA_EPILOGUE 保存/恢复 xmm6-xmm15。
;
; 通用寄存器只用 caller-saved 的 rax rcx rdx r8 r9 r10 r11, 不碰 callee-saved
; 的 rbx rbp rsi rdi r12-r15, 所以 GPR 侧不需要额外保存。
;
; 栈: 入口 rsp ≡ 8 (mod 16) (调用者的 call 压了 8 字节返回地址)。
; ZQA_FRAME = 0A8h = 168 = 10*16 (xmm6-xmm15) + 8 字节补齐, 且 168 ≡ 8 (mod 16),
; 所以 sub rsp, ZQA_FRAME 之后 rsp ≡ 0 (mod 16), 保存区天然 16 字节对齐;
; add rsp, ZQA_FRAME 之后 rsp 回到入口值, 出口 16 字节对齐不变。
; 减 rsp 之后, 栈上传参的偏移整体要加 ZQA_FRAME, 见 ZQA_ARG5..ZQA_ARG8。
;
; FMA: Windows 侧 ZQ_CNN_USE_SSETYPE 固定为 AVX2 (含 FMA3), 且 /arch:AVX2,
; 这里直接用 vfmadd231ps。
;-----------------------------------------------------------------------------

_TEXT SEGMENT

; xmm6-xmm15 保存区大小 (见上面的对齐推导)
ZQA_FRAME      EQU     0A8h
; 减 rsp, ZQA_FRAME 之后的栈上传参偏移
ZQA_ARG5       EQU     0D0h        ; = 28h + ZQA_FRAME
ZQA_ARG6       EQU     0D8h        ; = 30h + ZQA_FRAME
ZQA_ARG7       EQU     0E0h        ; = 38h + ZQA_FRAME
ZQA_ARG8       EQU     0E8h        ; = 40h + ZQA_FRAME

; 保存 / 恢复 xmm6-xmm15 (Windows x64 callee-saved)
ZQA_PROLOGUE MACRO
        sub     rsp, ZQA_FRAME
        vmovups xmmword ptr [rsp+00h], xmm6
        vmovups xmmword ptr [rsp+10h], xmm7
        vmovups xmmword ptr [rsp+20h], xmm8
        vmovups xmmword ptr [rsp+30h], xmm9
        vmovups xmmword ptr [rsp+40h], xmm10
        vmovups xmmword ptr [rsp+50h], xmm11
        vmovups xmmword ptr [rsp+60h], xmm12
        vmovups xmmword ptr [rsp+70h], xmm13
        vmovups xmmword ptr [rsp+80h], xmm14
        vmovups xmmword ptr [rsp+90h], xmm15
        ENDM

ZQA_EPILOGUE MACRO
        vmovups xmm6,  xmmword ptr [rsp+00h]
        vmovups xmm7,  xmmword ptr [rsp+10h]
        vmovups xmm8,  xmmword ptr [rsp+20h]
        vmovups xmm9,  xmmword ptr [rsp+30h]
        vmovups xmm10, xmmword ptr [rsp+40h]
        vmovups xmm11, xmmword ptr [rsp+50h]
        vmovups xmm12, xmmword ptr [rsp+60h]
        vmovups xmm13, xmmword ptr [rsp+70h]
        vmovups xmm14, xmmword ptr [rsp+80h]
        vmovups xmm15, xmmword ptr [rsp+90h]
        add     rsp, ZQA_FRAME
        ENDM





;-----------------------------------------------------------------------------
; 2 行 x 4 列
;   rcx = a0 (第 0 行), rdx = b0, r8d = k8, r9d = s0 = lda*4  (第 1 行 = [r10+rcx])
;   [rsp+ZQA_ARG5] = s1 (ldb*4 字节), [rsp+ZQA_ARG6] = s3 (3*ldb*4 字节)
;   [rsp+ZQA_ARG7] = c0, [rsp+ZQA_ARG8] = c1
;   ymm0-ymm3 : C 第 0 行的 4 列   ymm4-ymm7 : C 第 1 行的 4 列
;   ymm8/ymm9 : A 两行             ymm10-13  : Bt 的 4 行
;   K 循环**没有**做 2 路展开: 每个块 6 次 load + 8 条 FMA, 在 Zen 3 上
;   8 条 FMA 4 周期、6 次 load 3 周期、19 条 uop 发射 3.2 周期 —— 本来就是
;   FMA 吞吐瓶颈, 展开只能省下 dec/jnz 那一条, 实测无收益。
;-----------------------------------------------------------------------------
zq_gemm_32f_asm_core_m2n4 PROC
        ZQA_PROLOGUE
        mov     eax, r8d
        mov     r10, rcx
        mov     ecx, r9d                      ; s0 = lda*4, 顺带清掉 rcx 高 32 位
        mov     r9, rdx                       ; b0 (必须在取走 r9d 之后再覆盖)
        mov     r8d, DWORD PTR [rsp+ZQA_ARG5]
        mov     edx, DWORD PTR [rsp+ZQA_ARG6]
        vxorps  ymm0, ymm0, ymm0
        vxorps  ymm1, ymm1, ymm1
        vxorps  ymm2, ymm2, ymm2
        vxorps  ymm3, ymm3, ymm3
        vxorps  ymm4, ymm4, ymm4
        vxorps  ymm5, ymm5, ymm5
        vxorps  ymm6, ymm6, ymm6
        vxorps  ymm7, ymm7, ymm7
        test    eax, eax
        jz      L_m2n4_done
L_m2n4_loop:
        vmovups ymm8,  [r10]
        vmovups ymm9,  [r10+rcx]
        vmovups ymm10, [r9]
        vmovups ymm11, [r9+r8]
        vmovups ymm12, [r9+r8*2]
        vmovups ymm13, [r9+rdx]
        vfmadd231ps ymm0, ymm8, ymm10
        vfmadd231ps ymm1, ymm8, ymm11
        vfmadd231ps ymm2, ymm8, ymm12
        vfmadd231ps ymm3, ymm8, ymm13
        vfmadd231ps ymm4, ymm9, ymm10
        vfmadd231ps ymm5, ymm9, ymm11
        vfmadd231ps ymm6, ymm9, ymm12
        vfmadd231ps ymm7, ymm9, ymm13
        add     r10, 32
        add     r9, 32
        dec     eax
        jnz     L_m2n4_loop
L_m2n4_done:
        mov     r9, QWORD PTR [rsp+ZQA_ARG7]
        mov     r10, QWORD PTR [rsp+ZQA_ARG8]
        ; 4 个累加器 -> 4 个连续 float: vhaddps 两级树, 比逐累加器
        ; vextract+vaddps+vhaddps x2 少一半指令, 依赖链也短一半。
        ; vhaddps 只能配对处理, 所以 (ymm0,ymm1) 一组、(ymm2,ymm3) 一组。
        vhaddps ymm0, ymm0, ymm1
        vhaddps ymm2, ymm2, ymm3
        vextractf128 xmm8, ymm0, 1
        vaddps  xmm0, xmm0, xmm8
        vhaddps xmm0, xmm0, xmm0
        vextractf128 xmm8, ymm2, 1
        vaddps  xmm2, xmm2, xmm8
        vhaddps xmm2, xmm2, xmm2
        vshufps xmm0, xmm0, xmm2, 44h
        vmovups [r9], xmm0
        vhaddps ymm4, ymm4, ymm5
        vhaddps ymm6, ymm6, ymm7
        vextractf128 xmm8, ymm4, 1
        vaddps  xmm4, xmm4, xmm8
        vhaddps xmm4, xmm4, xmm4
        vextractf128 xmm8, ymm6, 1
        vaddps  xmm6, xmm6, xmm8
        vhaddps xmm6, xmm6, xmm6
        vshufps xmm4, xmm4, xmm6, 44h
        vmovups [r10], xmm4
        vzeroupper
        ZQA_EPILOGUE
        ret
zq_gemm_32f_asm_core_m2n4 ENDP


;-----------------------------------------------------------------------------
; 1 行 x 8 列 (B 分两批加载)
;   rcx = a0, rdx = b0 (第 0 列行), r8 = b4 (第 4 列行), r9d = k8
;   [rsp+ZQA_ARG5] = s1, [rsp+ZQA_ARG6] = s3, [rsp+ZQA_ARG7] = c0
;   ymm0-ymm3 : 列 0..3 的累加器   ymm4-ymm7 : 列 4..7 的累加器
;-----------------------------------------------------------------------------
zq_gemm_32f_asm_core_m1n8 PROC
        ZQA_PROLOGUE
        mov     eax, r9d
        mov     r10, rcx
        mov     r9, rdx
        mov     r11, r8
        mov     r8d, DWORD PTR [rsp+ZQA_ARG5]
        mov     edx, DWORD PTR [rsp+ZQA_ARG6]
        mov     rcx, QWORD PTR [rsp+ZQA_ARG7]
        vxorps  ymm0, ymm0, ymm0
        vxorps  ymm1, ymm1, ymm1
        vxorps  ymm2, ymm2, ymm2
        vxorps  ymm3, ymm3, ymm3
        vxorps  ymm4, ymm4, ymm4
        vxorps  ymm5, ymm5, ymm5
        vxorps  ymm6, ymm6, ymm6
        vxorps  ymm7, ymm7, ymm7
        test    eax, eax
        jz      L_m1n8_done
L_m1n8_loop:
        vmovups ymm8,  [r10]
        vmovups ymm9,  [r9]
        vmovups ymm10, [r9+r8]
        vmovups ymm11, [r9+r8*2]
        vmovups ymm12, [r9+rdx]
        vfmadd231ps ymm0, ymm8, ymm9
        vfmadd231ps ymm1, ymm8, ymm10
        vfmadd231ps ymm2, ymm8, ymm11
        vfmadd231ps ymm3, ymm8, ymm12
        vmovups ymm9,  [r11]
        vmovups ymm10, [r11+r8]
        vmovups ymm11, [r11+r8*2]
        vmovups ymm12, [r11+rdx]
        vfmadd231ps ymm4, ymm8, ymm9
        vfmadd231ps ymm5, ymm8, ymm10
        vfmadd231ps ymm6, ymm8, ymm11
        vfmadd231ps ymm7, ymm8, ymm12
        add     r10, 32
        add     r9, 32
        add     r11, 32
        dec     eax
        jnz     L_m1n8_loop
L_m1n8_done:
        lea     r10, [rcx+16]
        vhaddps ymm0, ymm0, ymm1
        vhaddps ymm2, ymm2, ymm3
        vextractf128 xmm8, ymm0, 1
        vaddps  xmm0, xmm0, xmm8
        vhaddps xmm0, xmm0, xmm0
        vextractf128 xmm8, ymm2, 1
        vaddps  xmm2, xmm2, xmm8
        vhaddps xmm2, xmm2, xmm2
        vshufps xmm0, xmm0, xmm2, 44h
        vmovups [rcx], xmm0
        vhaddps ymm4, ymm4, ymm5
        vhaddps ymm6, ymm6, ymm7
        vextractf128 xmm8, ymm4, 1
        vaddps  xmm4, xmm4, xmm8
        vhaddps xmm4, xmm4, xmm4
        vextractf128 xmm8, ymm6, 1
        vaddps  xmm6, xmm6, xmm8
        vhaddps xmm6, xmm6, xmm6
        vshufps xmm4, xmm4, xmm6, 44h
        vmovups [rcx+16], xmm4
        vzeroupper
        ZQA_EPILOGUE
        ret
zq_gemm_32f_asm_core_m1n8 ENDP

;-----------------------------------------------------------------------------
; 1 行 x 4 列
;   rcx = a0, rdx = b0, r8d = k8, r9d = s1
;   [rsp+ZQA_ARG5] = s3, [rsp+ZQA_ARG6] = c0
;   (x64 Windows ABI: 前 4 个整型/指针参数走 rcx/rdx/r8/r9, 第 5 个起才上栈;
;    偏移已含 ZQA_FRAME, 见文件头的说明)
;-----------------------------------------------------------------------------
zq_gemm_32f_asm_core_m1n4 PROC
        ZQA_PROLOGUE
        mov     eax, r8d
        mov     r8d, r9d                      ; s1 必须在 r9 被 b0 覆盖之前取走
        mov     r10, rcx
        mov     r9, rdx
        mov     edx, DWORD PTR [rsp+ZQA_ARG5]
        mov     rcx, QWORD PTR [rsp+ZQA_ARG6]
        vxorps  ymm0, ymm0, ymm0
        vxorps  ymm1, ymm1, ymm1
        vxorps  ymm2, ymm2, ymm2
        vxorps  ymm3, ymm3, ymm3
        test    eax, eax
        jz      L_m1n4_done
L_m1n4_loop:
        vmovups ymm8,  [r10]
        vmovups ymm9,  [r9]
        vmovups ymm10, [r9+r8]
        vmovups ymm11, [r9+r8*2]
        vmovups ymm12, [r9+rdx]
        vfmadd231ps ymm0, ymm8, ymm9
        vfmadd231ps ymm1, ymm8, ymm10
        vfmadd231ps ymm2, ymm8, ymm11
        vfmadd231ps ymm3, ymm8, ymm12
        add     r10, 32
        add     r9, 32
        dec     eax
        jnz     L_m1n4_loop
L_m1n4_done:
        vhaddps ymm0, ymm0, ymm1
        vhaddps ymm2, ymm2, ymm3
        vextractf128 xmm8, ymm0, 1
        vaddps  xmm0, xmm0, xmm8
        vhaddps xmm0, xmm0, xmm0
        vextractf128 xmm8, ymm2, 1
        vaddps  xmm2, xmm2, xmm8
        vhaddps xmm2, xmm2, xmm2
        vshufps xmm0, xmm0, xmm2, 44h
        vmovups [rcx], xmm0
        vzeroupper
        ZQA_EPILOGUE
        ret
zq_gemm_32f_asm_core_m1n4 ENDP

;-----------------------------------------------------------------------------
; 4 行 x 1 列 (N=1 专用) —— 沿 K 方向 ymm 累加, b 向量只加载一次、复用给 4 行 A
;   rcx = a0 (指向块内第 0 行), rdx = b0, r8d = k8, r9d = s1 = lda*4
;   [rsp+ZQA_ARG5] = s3 = 3*lda*4
;   [rsp+ZQA_ARG6] = c0 (指向块内第 0 行)
;   [rsp+ZQA_ARG7] = ldc4 = ldc*4, [rsp+ZQA_ARG8] = ldc12 = 3*ldc*4
;   ymm0-ymm3 : 4 行的累加器, ymm8 = b 当前 K 块, ymm9 = A 当前 K 块
;-----------------------------------------------------------------------------
zq_gemm_32f_asm_core_m4n1 PROC
        ZQA_PROLOGUE
        mov     eax, r8d                      ; k8 (ml64 不允许 64 位目的 <- 32 位源)
        mov     r11d, r9d                     ; s1 = lda*4, 顺带清掉 r11 高 32 位
        mov     r9, rdx                       ; b0
        mov     r10, rcx                      ; a0 = 块内第 0 行
        mov     edx, DWORD PTR [rsp+ZQA_ARG5] ; s3 = 3*lda*4
        vxorps  ymm0, ymm0, ymm0
        vxorps  ymm1, ymm1, ymm1
        vxorps  ymm2, ymm2, ymm2
        vxorps  ymm3, ymm3, ymm3
        test    rax, rax
        jz      L_m4n1_done
L_m4n1_loop:
        vmovups ymm8, [r9]
        vmovups ymm9, [r10]
        vfmadd231ps ymm0, ymm9, ymm8
        vmovups ymm9, [r10+r11]
        vfmadd231ps ymm1, ymm9, ymm8
        vmovups ymm9, [r10+r11*2]
        vfmadd231ps ymm2, ymm9, ymm8
        vmovups ymm9, [r10+rdx]
        vfmadd231ps ymm3, ymm9, ymm8
        add     r10, 32
        add     r9, 32
        dec     rax
        jnz     L_m4n1_loop
L_m4n1_done:
        mov     rcx, QWORD PTR [rsp+ZQA_ARG6]
        mov     r8d, DWORD PTR [rsp+ZQA_ARG7]
        mov     edx, DWORD PTR [rsp+ZQA_ARG8]
        ; 4 行各自是独立标量, 归约结果留在 lane 0, 4 条 vmovss 散写
        vhaddps ymm0, ymm0, ymm1
        vhaddps ymm2, ymm2, ymm3
        vextractf128 xmm8, ymm0, 1
        vaddps  xmm0, xmm0, xmm8
        vhaddps xmm0, xmm0, xmm0
        vextractf128 xmm8, ymm2, 1
        vaddps  xmm2, xmm2, xmm8
        vhaddps xmm2, xmm2, xmm2
        ; 4 个结果散写到 4 行, lane 1 用 vshufps 0x55 复制出来
        vshufps xmm1, xmm0, xmm0, 55h
        vshufps xmm3, xmm2, xmm2, 55h
        vmovss  DWORD PTR [rcx],      xmm0
        vmovss  DWORD PTR [rcx+r8],   xmm1
        vmovss  DWORD PTR [rcx+r8*2], xmm2
        vmovss  DWORD PTR [rcx+rdx],  xmm3
        vzeroupper
        ZQA_EPILOGUE
        ret
zq_gemm_32f_asm_core_m4n1 ENDP

;-----------------------------------------------------------------------------
; 1 行 x 1 列 (M 尾部 1~3 行 / M<4)
;   rcx = a0, rdx = b0, r8d = k8, r9 = c0   (4 个参数全走寄存器, 没有栈上传参)
;-----------------------------------------------------------------------------
zq_gemm_32f_asm_core_m1n1 PROC
        ZQA_PROLOGUE
        mov     eax, r8d                      ; k8
        mov     r10, rcx                      ; a0
        mov     rcx, r9                       ; c0
        mov     r9, rdx                       ; b0
        vxorps  ymm0, ymm0, ymm0
        test    rax, rax
        jz      L_m1n1_done
L_m1n1_loop:
        vmovups ymm8, [r9]
        vmovups ymm9, [r10]
        vfmadd231ps ymm0, ymm9, ymm8
        add     r10, 32
        add     r9, 32
        dec     rax
        jnz     L_m1n1_loop
L_m1n1_done:
        vextractf128 xmm8, ymm0, 1
        vaddps  xmm0, xmm0, xmm8
        vhaddps xmm0, xmm0, xmm0
        vhaddps xmm0, xmm0, xmm0
        vmovss  DWORD PTR [rcx], xmm0
        vzeroupper
        ZQA_EPILOGUE
        ret
zq_gemm_32f_asm_core_m1n1 ENDP

_TEXT ENDS
END
