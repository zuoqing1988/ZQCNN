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
; 本文件只用 caller-saved 寄存器 (rax rcx rdx r8 r9 r10 r11) 与 caller-saved 的
; ymm0-ymm15, 不碰 callee-saved 寄存器, 也不调整 rsp, 所以不需要保存/恢复任何
; 寄存器, 栈在函数入口/出口都保持 16 字节对齐。
;
; FMA: Windows 侧 ZQ_CNN_USE_SSETYPE 固定为 AVX2 (含 FMA3), 且 /arch:AVX2,
; 这里直接用 vfmadd231ps。
;-----------------------------------------------------------------------------

_TEXT SEGMENT





;-----------------------------------------------------------------------------
; 2 行 x 4 列
;   rcx = a0, rdx = a1, r8 = b0, r9d = k8
;   [rsp+28h] = s1 (ldb*4 字节), [rsp+30h] = s3 (3*ldb*4 字节)
;   [rsp+38h] = c0, [rsp+40h] = c1
;   ymm0-ymm3 : C 第 0 行的 4 列   ymm4-ymm7 : C 第 1 行的 4 列
;   ymm8/ymm9 : A 两行             ymm10-13  : Bt 的 4 行
;-----------------------------------------------------------------------------
zq_gemm_32f_asm_core_m2n4 PROC
        ret
        mov     eax, r9d
        mov     r10, rcx
        mov     r11, rdx
        mov     r9, r8
        mov     r8d, DWORD PTR [rsp+28h]
        mov     edx, DWORD PTR [rsp+30h]
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
        vmovups ymm9,  [r11]
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
        add     r11, 32
        add     r9, 32
        dec     eax
        jnz     L_m2n4_loop
L_m2n4_done:
        mov     r9, QWORD PTR [rsp+38h]
        mov     r10, QWORD PTR [rsp+40h]
        vextractf128 xmm14, ymm0, 1
        vaddps  xmm0, xmm0, xmm14
        vhaddps xmm0, xmm0, xmm0
        vhaddps xmm0, xmm0, xmm0
        vextractf128 xmm14, ymm1, 1
        vaddps  xmm1, xmm1, xmm14
        vhaddps xmm1, xmm1, xmm1
        vhaddps xmm1, xmm1, xmm1
        vextractf128 xmm14, ymm2, 1
        vaddps  xmm2, xmm2, xmm14
        vhaddps xmm2, xmm2, xmm2
        vhaddps xmm2, xmm2, xmm2
        vextractf128 xmm14, ymm3, 1
        vaddps  xmm3, xmm3, xmm14
        vhaddps xmm3, xmm3, xmm3
        vhaddps xmm3, xmm3, xmm3
        vinsertps xmm0, xmm0, xmm1, 10h
        vinsertps xmm0, xmm0, xmm2, 20h
        vinsertps xmm0, xmm0, xmm3, 30h
        vmovups [r9], xmm0
        vextractf128 xmm14, ymm4, 1
        vaddps  xmm4, xmm4, xmm14
        vhaddps xmm4, xmm4, xmm4
        vhaddps xmm4, xmm4, xmm4
        vextractf128 xmm14, ymm5, 1
        vaddps  xmm5, xmm5, xmm14
        vhaddps xmm5, xmm5, xmm5
        vhaddps xmm5, xmm5, xmm5
        vextractf128 xmm14, ymm6, 1
        vaddps  xmm6, xmm6, xmm14
        vhaddps xmm6, xmm6, xmm6
        vhaddps xmm6, xmm6, xmm6
        vextractf128 xmm14, ymm7, 1
        vaddps  xmm7, xmm7, xmm14
        vhaddps xmm7, xmm7, xmm7
        vhaddps xmm7, xmm7, xmm7
        vinsertps xmm4, xmm4, xmm5, 10h
        vinsertps xmm4, xmm4, xmm6, 20h
        vinsertps xmm4, xmm4, xmm7, 30h
        vmovups [r10], xmm4
        vzeroupper
        ret
zq_gemm_32f_asm_core_m2n4 ENDP

;-----------------------------------------------------------------------------
; 1 行 x 8 列 (B 分两批加载)
;   rcx = a0, rdx = b0 (第 0 列行), r8 = b4 (第 4 列行), r9d = k8
;   [rsp+28h] = s1, [rsp+30h] = s3, [rsp+38h] = c0
;   ymm0-ymm3 : 列 0..3 的累加器   ymm4-ymm7 : 列 4..7 的累加器
;-----------------------------------------------------------------------------
zq_gemm_32f_asm_core_m1n8 PROC
        ret
        mov     eax, r9d
        mov     r10, rcx
        mov     r9, rdx
        mov     r11, r8
        mov     r8d, DWORD PTR [rsp+28h]
        mov     edx, DWORD PTR [rsp+30h]
        mov     rcx, QWORD PTR [rsp+38h]
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
        vextractf128 xmm14, ymm0, 1
        vaddps  xmm0, xmm0, xmm14
        vhaddps xmm0, xmm0, xmm0
        vhaddps xmm0, xmm0, xmm0
        vextractf128 xmm14, ymm1, 1
        vaddps  xmm1, xmm1, xmm14
        vhaddps xmm1, xmm1, xmm1
        vhaddps xmm1, xmm1, xmm1
        vextractf128 xmm14, ymm2, 1
        vaddps  xmm2, xmm2, xmm14
        vhaddps xmm2, xmm2, xmm2
        vhaddps xmm2, xmm2, xmm2
        vextractf128 xmm14, ymm3, 1
        vaddps  xmm3, xmm3, xmm14
        vhaddps xmm3, xmm3, xmm3
        vhaddps xmm3, xmm3, xmm3
        vinsertps xmm0, xmm0, xmm1, 10h
        vinsertps xmm0, xmm0, xmm2, 20h
        vinsertps xmm0, xmm0, xmm3, 30h
        vmovups [rcx], xmm0
        vextractf128 xmm14, ymm4, 1
        vaddps  xmm4, xmm4, xmm14
        vhaddps xmm4, xmm4, xmm4
        vhaddps xmm4, xmm4, xmm4
        vextractf128 xmm14, ymm5, 1
        vaddps  xmm5, xmm5, xmm14
        vhaddps xmm5, xmm5, xmm5
        vhaddps xmm5, xmm5, xmm5
        vextractf128 xmm14, ymm6, 1
        vaddps  xmm6, xmm6, xmm14
        vhaddps xmm6, xmm6, xmm6
        vhaddps xmm6, xmm6, xmm6
        vextractf128 xmm14, ymm7, 1
        vaddps  xmm7, xmm7, xmm14
        vhaddps xmm7, xmm7, xmm7
        vhaddps xmm7, xmm7, xmm7
        vinsertps xmm4, xmm4, xmm5, 10h
        vinsertps xmm4, xmm4, xmm6, 20h
        vinsertps xmm4, xmm4, xmm7, 30h
        vmovups [rcx+16], xmm4        ; c0 + 列 4..7（原来错写成 [r10]，r10 是 A 行指针，
                                     ; 会把结果写进调用方的 A 缓冲区造成堆破坏）
        vzeroupper
        ret
zq_gemm_32f_asm_core_m1n8 ENDP

;-----------------------------------------------------------------------------
; 1 行 x 4 列
;   rcx = a0, rdx = b0, r8d = k8, r9d = s1
;   [rsp+28h] = s3, [rsp+30h] = c0
;   (x64 Windows ABI: 前 4 个整型/指针参数走 rcx/rdx/r8/r9, 第 5 个起才上栈)
;-----------------------------------------------------------------------------
zq_gemm_32f_asm_core_m1n4 PROC
        ret
        mov     eax, r8d
        mov     r8d, r9d                      ; s1 必须在 r9 被 b0 覆盖之前取走
        mov     r10, rcx
        mov     r9, rdx
        mov     edx, DWORD PTR [rsp+28h]
        mov     rcx, QWORD PTR [rsp+30h]
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
        vextractf128 xmm14, ymm0, 1
        vaddps  xmm0, xmm0, xmm14
        vhaddps xmm0, xmm0, xmm0
        vhaddps xmm0, xmm0, xmm0
        vextractf128 xmm14, ymm1, 1
        vaddps  xmm1, xmm1, xmm14
        vhaddps xmm1, xmm1, xmm1
        vhaddps xmm1, xmm1, xmm1
        vextractf128 xmm14, ymm2, 1
        vaddps  xmm2, xmm2, xmm14
        vhaddps xmm2, xmm2, xmm2
        vhaddps xmm2, xmm2, xmm2
        vextractf128 xmm14, ymm3, 1
        vaddps  xmm3, xmm3, xmm14
        vhaddps xmm3, xmm3, xmm3
        vhaddps xmm3, xmm3, xmm3
        vinsertps xmm0, xmm0, xmm1, 10h
        vinsertps xmm0, xmm0, xmm2, 20h
        vinsertps xmm0, xmm0, xmm3, 30h
        vmovups [rcx], xmm0
        vzeroupper
        ret
zq_gemm_32f_asm_core_m1n4 ENDP

_TEXT ENDS
END
