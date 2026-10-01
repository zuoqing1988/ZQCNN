/*
 * zq_gemm_32f_align_c_asm.c
 *
 * ZQ_GEMM 手写内联汇编 (inline assembly) 内核。
 *
 * 本文件与 intrinsic 版 (zq_gemm_32f_align_c.c / zq_gemm_32f_align_c_raw.h /
 * zq_gemm_32f_auto.c) 完全并列存在, 不修改、不依赖它们的任何内部实现,
 * 只在没有汇编内核的平台/编译器上调用 intrinsic 入口作为回落。
 *
 * 数据布局与语义 (与 intrinsic 版完全一致):
 *   C[i][j] = sum_k A[i*lda + k] * Bt[j*ldb + k]
 *   A : M x K 行主序   Bt: N x K 行主序 (注意是 N x K)   C : M x N 行主序
 *   C 是被覆盖写入 (*C = sum), 不是累加 (+=)
 *
 * ============================ 微内核设计 ============================
 *
 * 1) M2 x N4  (2 行 x 4 列) —— 主力微内核, 8 个 ymm 累加器
 *
 *   ymm0  ymm1  ymm2  ymm3   : C 第 0 行的 4 列 (列 0..3)
 *   ymm4  ymm5  ymm6  ymm7   : C 第 1 行的 4 列 (列 0..3)
 *   ymm8                        : A 第 0 行当前 K 块 (8 float)
 *   ymm9                        : A 第 1 行当前 K 块
 *   ymm10 ymm11 ymm12 ymm13    : Bt 的 4 行当前 K 块
 *   ymm14                       : 水平归约临时
 *   ymm15                       : 无 FMA 时 vmulps 的临时
 *
 *   通用寄存器: 全部只用 caller-saved (rax rcx rdx r8 r9 r10 r11),
 *   所以整个 asm 块里不需要 push/pop, 也不碰 callee-saved 寄存器。
 *     eax  = K 迭代次数 (K/8)
 *     r10  = A 第 0 行指针  (每次 +32)      循环结束后作废, 复用为写回指针
 *     r11  = A 第 1 行指针  (每次 +32)
 *     r9   = Bt 第 0 行指针 (每次 +32)      循环结束后作废, 复用为写回指针
 *     r8d  = ldb * 4 (字节)                 Bt 相对行偏移
 *     edx  = ldb * 12 (字节) = 3 * ldb * 4
 *   Bt 的 4 个行地址: [r9] / [r9+r8] / [r9+r8*2] / [r9+rdx]
 *   (第 3 个用预乘的 3*ldb, 因为 x86 寻址只支持 base+index*scale, 表达不了 r9+r8+rdx)
 *
 * 2) M1 x N8  (1 行 x 8 列) —— B 分两批加载, 复用于两批
 *
 *   ymm0..ymm3 : 列 0..3 的累加器,  ymm4..ymm7 : 列 4..7 的累加器
 *   ymm8  : A 当前 K 块
 *   ymm9..ymm12 : B 的 4 个向量 (先列 0..3, 再列 4..7)
 *   eax = 迭代次数, r10 = A, r9 = Bt+0, r11 = Bt+4*ldb,
 *   r8d = ldb*4, edx = ldb*12, rcx = C 写回指针
 *   列 0..3: [r9]/[r9+r8]/[r9+r8*2]/[r9+rdx]
 *   列 4..7: [r11]/[r11+r8]/[r11+r8*2]/[r11+rdx]
 *
 * 3) M1 x N4  (1 行 x 4 列) —— M 尾部单行, 以及 N 尾部剩下的 4 列
 *
 *   ymm0..ymm3 : 4 个累加器, ymm8 = A, ymm9..ymm12 = B
 *   eax = 迭代次数, r10 = A, r9 = Bt, r8d/edx = ldb 偏移, rcx = C
 *
 * 水平归约 (每个累加器):
 *   vextractf128 $1 -> vaddps -> vhaddps x2, 得到 4 个 lane 相同的 xmm,
 *   一次 vmovups 写 4 个 float。写入范围严格落在 n..n+3 <= N-1 内。
 *
 * K 维处理: SIMD 只做 [0, K & ~7), 剩下 1~7 个元素由 C 标量循环补加。
 * 因此不依赖 A/Bt 行尾补零 (intrinsic 版依赖, 两边都能用)。
 *
 * FMA: 由 ZQ_CNN_USE_FMADD256 决定 —— 与 intrinsic 版用同一个宏,
 * 保证 A/B 对比时两边走同一条指令路径:
 *   1 -> vfmadd231ps        0 -> vmulps + vaddps
 * 宏在 C 层分派, 不在 __asm {} 块里写 #if。
 *
 * GCC 侧输入传递: 全部输入用 "m" 约束。
 *   原因: 固定寄存器的 asm 里, 如果输入用 "r" 约束, 编译器可能把后读的
 *   输入分配到先写的寄存器上, 读到的就是被破坏的值; 用 "m" 约束后每个
 *   输入都在栈上有一份独立的拷贝, 什么时候读都安全, 因此可以把 C 写回
 *   指针留到循环之后再读 (对应 MSVC 的 __asm{} 直接读 C 变量的行为)。
 *
 * ============================ 平台 ============================
 *   MSVC x64        : 函数体内 __asm { } (Intel 语法)
 *   GCC/Clang x64   : 函数体内 __asm__ volatile (AT&T 语法)
 *   其他 / 无 AVX   : 整段用 #if 编译掉, 所有 _asm 符号回落调用
 *                      zq_gemm_32f_AnoTrans_Btrans_auto (intrinsic 版)
 */

#include "zq_gemm_32f_align_c_asm.h"
#include "zq_gemm_32f_align_c.h"

/* ====================================================================== *
 * 平台与指令集判定
 * ====================================================================== */

#if defined(__ARM_NEON) && __ARM_NEON
#define ZQA_NEON 1
#else
#define ZQA_NEON 0
#endif

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
#define ZQA_MSVC_X64 1
#else
#define ZQA_MSVC_X64 0
#endif

#if !ZQA_MSVC_X64 && (defined(__x86_64__) || defined(__amd64__)) && (defined(__GNUC__) || defined(__clang__))
#define ZQA_GNU_X64 1
#else
#define ZQA_GNU_X64 0
#endif

/* 只有 x86-64 + 支持内联汇编的编译器 + 配置里打开了 AVX 才给汇编实现 */
#if !ZQA_NEON && (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX) && (ZQA_MSVC_X64 || ZQA_GNU_X64)
#define ZQA_IMPL 1
#else
#define ZQA_IMPL 0
#endif

/* 与 intrinsic 版共用同一个 FMA 开关, 保证 A/B 对比公平 */
#if defined(ZQ_CNN_USE_FMADD256) && ZQ_CNN_USE_FMADD256
#define ZQA_HAVE_FMA 1
#else
#define ZQA_HAVE_FMA 0
#endif

/* ====================================================================== *
 * 汇编片段原语 (两套语法, 同名宏)
 * ====================================================================== */

#if ZQA_IMPL

#if ZQA_MSVC_X64

/* ---------------------------------------------------------------------- *
 * MSVC 分支: 三个微内核由 zq_gemm_32f_align_c_asm_msvc.asm (MASM / ml64) 提供。
 *
 * MSVC 的 x64 目标**不支持函数体内联汇编** —— 对 __asm { } 直接报
 *   error C4235: non-standard extension: '__asm' keyword is not supported in
 *   this context
 * (只有 x86 32 位目标支持 __asm; x64 只能用独立的 .asm 文件)。
 * 所以 Windows 下内核是独立汇编文件, 这里只做声明; 指令序列与下面
 * GCC/Clang 的内联汇编版逐条对应。
 * ---------------------------------------------------------------------- */
#define ZQA_NOINLINE __declspec(noinline)

void zq_gemm_32f_asm_core_m2n4(const float* a0, const float* a1, const float* b0,
	int k8, int s1, int s3, float* c0, float* c1);
void zq_gemm_32f_asm_core_m1n8(const float* a0, const float* b0, const float* b4,
	int k8, int s1, int s3, float* c0);
void zq_gemm_32f_asm_core_m1n4(const float* a0, const float* b0,
	int k8, int s1, int s3, float* c0);


#else /* ZQA_GNU_X64, AT&T 语法: 函数体内 __asm__ volatile */

/* 微内核必须独立成函数: 一个 C 函数里只能有一处 __asm__ 块带同名的
   数字局部标号, 内联展开会出现标号重复。 */
#define ZQA_NOINLINE __attribute__((noinline))

/* 注意 1: 下面的宏参数一律用 # 字符串化 —— AT&T 汇编模板是字符串,
   形如 "vmovups " m ", %%" #r 的写法如果不用 # , 展开后会变成
   "..." 裸标识符 "...", C 层面就不是合法的字符串拼接了。
   而 m / a 之类的参数传进来的已经是一个宏展开后的字符串字面量,
   所以它们用普通替换即可。
   注意 2: 模板里每一个寄存器名都必须写成 %%ymm0 / %%r10 这种形式。
   单个 % 会被 GCC 当成操作数引用 (%ymm0 里的 "y" 被当成修饰字母),
   报 "invalid 'asm': operand number missing after %-letter"。 */
#define ZQA_MOV32(d, s)   "movl %[" #s "], %%" #d "\n\t"
#define ZQA_MOV64(d, s)   "movq %[" #s "], %%" #d "\n\t"
#define ZQA_XOR(r)        "vxorps %%" #r ", %%" #r ", %%" #r "\n\t"
#define ZQA_LD(r, m)      "vmovups " m ", %%" #r "\n\t"
#define ZQA_ST(m, r)      "vmovups %%" #r ", " m "\n\t"   /* AT&T 存储: 寄存器在前 */
#define ZQA_LEA(d, m)     "leaq " m ", %%" #d "\n\t"
#define ZQA_ADD32(d)      "addq $32, %%" #d "\n\t"
#define ZQA_DECEAX        "decl %%eax\n\t"
#define ZQA_TSTEAX        "testl %%eax, %%eax\n\t"
#define ZQA_M0(b)         "(%%" #b ")"
#define ZQA_M1(b, i)      "(%%" #b ",%%" #i ",1)"
#define ZQA_M2(b, i, s)   "(%%" #b ",%%" #i "," #s ")"
#define ZQA_MI(b, o)      #o "(%%" #b ")"
#define ZQA_VZ            "vzeroupper\n\t"
#define ZQA_HSUM(r, rl) \
	"vextractf128 $1, %%" #r ", %%xmm14\n\t" \
	"vaddps %%xmm14, %%" #rl ", %%" #rl "\n\t" \
	"vhaddps %%" #rl ", %%" #rl ", %%" #rl "\n\t" \
	"vhaddps %%" #rl ", %%" #rl ", %%" #rl "\n\t"
#define ZQA_INS(d, s, i)  "vinsertps $" #i ", %%" #s ", %%" #d ", %%" #d "\n\t"
#define ZQA_PACK4(a, b, c, d) ZQA_INS(a, b, 0x10) ZQA_INS(a, c, 0x20) ZQA_INS(a, d, 0x30)

#if ZQA_HAVE_FMA
#define ZQA_FMA(d, a, b) "vfmadd231ps %%" #b ", %%" #a ", %%" #d "\n\t"
#else
#define ZQA_FMA(d, a, b) "vmulps %%" #b ", %%" #a ", %%ymm15\n\tvaddps %%ymm15, %%" #d ", %%" #d "\n\t"
#endif

/* ====================================================================== *
 * 三个微内核的循环体 (MASM 版与内联汇编版共用同一套结构)
 * ====================================================================== */

/* 2 行 x 4 列, 8 个累加器 */
#define ZQA_BODY_M2N4 \
	ZQA_LD(ymm8,  ZQA_M0(r10)) \
	ZQA_LD(ymm9,  ZQA_M0(r11)) \
	ZQA_LD(ymm10, ZQA_M0(r9)) \
	ZQA_LD(ymm11, ZQA_M1(r9, r8)) \
	ZQA_LD(ymm12, ZQA_M2(r9, r8, 2)) \
	ZQA_LD(ymm13, ZQA_M1(r9, rdx)) \
	ZQA_FMA(ymm0, ymm8, ymm10) \
	ZQA_FMA(ymm1, ymm8, ymm11) \
	ZQA_FMA(ymm2, ymm8, ymm12) \
	ZQA_FMA(ymm3, ymm8, ymm13) \
	ZQA_FMA(ymm4, ymm9, ymm10) \
	ZQA_FMA(ymm5, ymm9, ymm11) \
	ZQA_FMA(ymm6, ymm9, ymm12) \
	ZQA_FMA(ymm7, ymm9, ymm13)

/* 1 行 x 8 列, B 分两批加载 */
#define ZQA_BODY_M1N8 \
	ZQA_LD(ymm8,  ZQA_M0(r10)) \
	ZQA_LD(ymm9,  ZQA_M0(r9)) \
	ZQA_LD(ymm10, ZQA_M1(r9, r8)) \
	ZQA_LD(ymm11, ZQA_M2(r9, r8, 2)) \
	ZQA_LD(ymm12, ZQA_M1(r9, rdx)) \
	ZQA_FMA(ymm0, ymm8, ymm9) \
	ZQA_FMA(ymm1, ymm8, ymm10) \
	ZQA_FMA(ymm2, ymm8, ymm11) \
	ZQA_FMA(ymm3, ymm8, ymm12) \
	ZQA_LD(ymm9,  ZQA_M0(r11)) \
	ZQA_LD(ymm10, ZQA_M1(r11, r8)) \
	ZQA_LD(ymm11, ZQA_M2(r11, r8, 2)) \
	ZQA_LD(ymm12, ZQA_M1(r11, rdx)) \
	ZQA_FMA(ymm4, ymm8, ymm9) \
	ZQA_FMA(ymm5, ymm8, ymm10) \
	ZQA_FMA(ymm6, ymm8, ymm11) \
	ZQA_FMA(ymm7, ymm8, ymm12)

/* 1 行 x 4 列 */
#define ZQA_BODY_M1N4 \
	ZQA_LD(ymm8,  ZQA_M0(r10)) \
	ZQA_LD(ymm9,  ZQA_M0(r9)) \
	ZQA_LD(ymm10, ZQA_M1(r9, r8)) \
	ZQA_LD(ymm11, ZQA_M2(r9, r8, 2)) \
	ZQA_LD(ymm12, ZQA_M1(r9, rdx)) \
	ZQA_FMA(ymm0, ymm8, ymm9) \
	ZQA_FMA(ymm1, ymm8, ymm10) \
	ZQA_FMA(ymm2, ymm8, ymm11) \
	ZQA_FMA(ymm3, ymm8, ymm12)

/* ====================================================================== *
 * 微内核 1: 2 行 x 4 列
 *
 * 只处理 SIMD 部分 (K 维前 k8*8 个元素), 结果覆盖写入 c0[0..3] / c1[0..3];
 * K 尾部由调用方用标量累加补上。k8 == 0 时写出全 0, 与 intrinsic 版
 * "先覆盖写入再补加 K 尾部" 的行为一致。
 * ====================================================================== */

static ZQA_NOINLINE void zq_gemm_32f_asm_core_m2n4(const float* a0, const float* a1, const float* b0,
	int k8, int s1, int s3, float* c0, float* c1)
{
	__asm__ volatile (
		ZQA_MOV32(eax, k8)
		ZQA_MOV64(r10, a0)
		ZQA_MOV64(r11, a1)
		ZQA_MOV64(r9, b0)
		ZQA_MOV32(r8d, s1)
		ZQA_MOV32(edx, s3)
		ZQA_XOR(ymm0) ZQA_XOR(ymm1) ZQA_XOR(ymm2) ZQA_XOR(ymm3)
		ZQA_XOR(ymm4) ZQA_XOR(ymm5) ZQA_XOR(ymm6) ZQA_XOR(ymm7)
		ZQA_TSTEAX
		"jz 1f\n\t"
		"0:\n\t"
		ZQA_BODY_M2N4
		ZQA_ADD32(r10)
		ZQA_ADD32(r11)
		ZQA_ADD32(r9)
		ZQA_DECEAX
		"jnz 0b\n\t"
		"1:\n\t"
		ZQA_MOV64(r9, c0)
		ZQA_MOV64(r10, c1)
		ZQA_HSUM(ymm0, xmm0) ZQA_HSUM(ymm1, xmm1) ZQA_HSUM(ymm2, xmm2) ZQA_HSUM(ymm3, xmm3)
		ZQA_PACK4(xmm0, xmm1, xmm2, xmm3)
		ZQA_ST(ZQA_M0(r9), xmm0)
		ZQA_HSUM(ymm4, xmm4) ZQA_HSUM(ymm5, xmm5) ZQA_HSUM(ymm6, xmm6) ZQA_HSUM(ymm7, xmm7)
		ZQA_PACK4(xmm4, xmm5, xmm6, xmm7)
		ZQA_ST(ZQA_M0(r10), xmm4)
		ZQA_VZ
		:
		: [k8]"m"(k8), [a0]"m"(a0), [a1]"m"(a1), [b0]"m"(b0),
		  [s1]"m"(s1), [s3]"m"(s3), [c0]"m"(c0), [c1]"m"(c1)
		: "cc", "memory", "rax", "rcx", "rdx", "r8", "r9", "r10", "r11",
		  "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"
	);
}

/* ====================================================================== *
 * 微内核 2: 1 行 x 8 列
 * b0 指向 8 列块的第 0 行, b4 指向第 4 行 (两者都随 K 维 +32 推进)
 * ====================================================================== */

static ZQA_NOINLINE void zq_gemm_32f_asm_core_m1n8(const float* a0, const float* b0, const float* b4,
	int k8, int s1, int s3, float* c0)
{
	__asm__ volatile (
		ZQA_MOV32(eax, k8)
		ZQA_MOV64(r10, a0)
		ZQA_MOV64(r9, b0)
		ZQA_MOV64(r11, b4)
		ZQA_MOV32(r8d, s1)
		ZQA_MOV32(edx, s3)
		ZQA_MOV64(rcx, c0)
		ZQA_XOR(ymm0) ZQA_XOR(ymm1) ZQA_XOR(ymm2) ZQA_XOR(ymm3)
		ZQA_XOR(ymm4) ZQA_XOR(ymm5) ZQA_XOR(ymm6) ZQA_XOR(ymm7)
		ZQA_TSTEAX
		"jz 1f\n\t"
		"0:\n\t"
		ZQA_BODY_M1N8
		ZQA_ADD32(r10)
		ZQA_ADD32(r9)
		ZQA_ADD32(r11)
		ZQA_DECEAX
		"jnz 0b\n\t"
		"1:\n\t"
		ZQA_LEA(r10, ZQA_MI(rcx, 16))
		ZQA_HSUM(ymm0, xmm0) ZQA_HSUM(ymm1, xmm1) ZQA_HSUM(ymm2, xmm2) ZQA_HSUM(ymm3, xmm3)
		ZQA_PACK4(xmm0, xmm1, xmm2, xmm3)
		ZQA_ST(ZQA_M0(rcx), xmm0)
		ZQA_HSUM(ymm4, xmm4) ZQA_HSUM(ymm5, xmm5) ZQA_HSUM(ymm6, xmm6) ZQA_HSUM(ymm7, xmm7)
		ZQA_PACK4(xmm4, xmm5, xmm6, xmm7)
		ZQA_ST(ZQA_M0(r10), xmm4)
		ZQA_VZ
		:
		: [k8]"m"(k8), [a0]"m"(a0), [b0]"m"(b0), [b4]"m"(b4),
		  [s1]"m"(s1), [s3]"m"(s3), [c0]"m"(c0)
		: "cc", "memory", "rax", "rcx", "rdx", "r8", "r9", "r10", "r11",
		  "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"
	);
}

/* ====================================================================== *
 * 微内核 3: 1 行 x 4 列
 * ====================================================================== */

static ZQA_NOINLINE void zq_gemm_32f_asm_core_m1n4(const float* a0, const float* b0,
	int k8, int s1, int s3, float* c0)
{
	__asm__ volatile (
		ZQA_MOV32(eax, k8)
		ZQA_MOV64(r10, a0)
		ZQA_MOV64(r9, b0)
		ZQA_MOV32(r8d, s1)
		ZQA_MOV32(edx, s3)
		ZQA_MOV64(rcx, c0)
		ZQA_XOR(ymm0) ZQA_XOR(ymm1) ZQA_XOR(ymm2) ZQA_XOR(ymm3)
		ZQA_TSTEAX
		"jz 1f\n\t"
		"0:\n\t"
		ZQA_BODY_M1N4
		ZQA_ADD32(r10)
		ZQA_ADD32(r9)
		ZQA_DECEAX
		"jnz 0b\n\t"
		"1:\n\t"
		ZQA_HSUM(ymm0, xmm0) ZQA_HSUM(ymm1, xmm1) ZQA_HSUM(ymm2, xmm2) ZQA_HSUM(ymm3, xmm3)
		ZQA_PACK4(xmm0, xmm1, xmm2, xmm3)
		ZQA_ST(ZQA_M0(rcx), xmm0)
		ZQA_VZ
		:
		: [k8]"m"(k8), [a0]"m"(a0), [b0]"m"(b0),
		  [s1]"m"(s1), [s3]"m"(s3), [c0]"m"(c0)
		: "cc", "memory", "rax", "rcx", "rdx", "r8", "r9", "r10", "r11",
		  "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"
	);
}

#endif /* ZQA_GNU_X64 : GCC/Clang 内联汇编实现结束 */

/* ====================================================================== *
 * 公共驱动: 任意 M / N / K
 *
 * 主分块 mb x nb (mb = 1/2/4/8, nb = 4/8) 走汇编微内核,
 * M / N / K 的尾部全部用 C 标量循环兜底, 结果与 intrinsic 版一致。
 * ====================================================================== */

/* 公共驱动体, MB / NB 是宏参数 (编译期常量)。
   每个分块形状生成一份独立函数, 编译器可以把 m / n / i / j 循环完全展开并
   常量化, 避免运行期分支和循环开销 —— 微内核每调用一次只有几十条指令,
   这部分开销在 K 较小时会占很大比例。 */
#define ZQA_DEFINE_KERNEL(FNAME, MB, NB) \
static void FNAME(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc) \
{ \
	int m, n, k, i, j; \
	int k8 = K >> 3; \
	int kstart = k8 << 3; \
	int s1 = ldb << 2; \
	int s3 = s1 * 3; \
	const int mb = MB, nb = NB; \
	const int npair = NB >> 2; \
 \
	{ extern int printf(const char*, ...); static int zqa_dbg = 0; if (zqa_dbg < 25) { zqa_dbg++; printf("[K %s M=%d N=%d K=%d k8=%d s1=%d s3=%d A=%p Bt=%p C=%p lda=%d ldb=%d ldc=%d]\n", #FNAME, M, N, K, k8, s1, s3, (void*)A, (void*)Bt, (void*)C, lda, ldb, ldc); } } \
 \
	if (M <= 0 || N <= 0) \
		return; \
 \
	for (m = 0; m + mb <= M; m += mb) \
	{ \
		const float* Ab = A + (size_t)m * lda; \
		float* Cb = C + (size_t)m * ldc; \
 \
		for (n = 0; n + nb <= N; n += nb) \
		{ \
			const float* Bb = Bt + (size_t)n * ldb; \
			float* Cc = Cb + n; \
 \
			if (MB == 1) \
			{ \
				if (NB == 8) \
					zq_gemm_32f_asm_core_m1n8(Ab, Bb, Bb + 4 * ldb, k8, s1, s3, Cc); \
				else \
					zq_gemm_32f_asm_core_m1n4(Ab, Bb, k8, s1, s3, Cc); \
			} \
			else \
			{ \
				for (i = 0; i < mb; i += 2) \
				{ \
					for (j = 0; j < npair; j++) \
					{ \
						zq_gemm_32f_asm_core_m2n4(Ab + i * lda, Ab + (i + 1) * lda, \
							Bb + j * 4 * ldb, k8, s1, s3, \
							Cc + i * ldc + j * 4, Cc + (i + 1) * ldc + j * 4); \
					} \
				} \
			} \
 \
			/* K 尾部: 微内核已覆盖写入, 这里补加剩余的 K 个元素 */ \
			for (k = kstart; k < K; k++) \
			{ \
				for (i = 0; i < mb; i++) \
				{ \
					float a = Ab[i * lda + k]; \
					for (j = 0; j < nb; j++) \
						Cc[i * ldc + j] += a * Bb[j * ldb + k]; \
				} \
			} \
		} \
 \
		/* N 尾部: 整列标量 */ \
		for (; n < N; n++) \
		{ \
			const float* b1 = Bt + (size_t)n * ldb; \
			for (i = 0; i < mb; i++) \
			{ \
				const float* a1 = Ab + i * lda; \
				float* c1 = Cb + i * ldc + n; \
				float sum = 0; \
				for (k = 0; k < K; k++) \
					sum += a1[k] * b1[k]; \
				*c1 = sum; \
			} \
		} \
	} \
 \
	/* M 尾部: 剩下的 1 .. mb-1 行, 行内仍优先走 1x8 / 1x4 微内核 */ \
	for (; m < M; m++) \
	{ \
		const float* a1 = A + (size_t)m * lda; \
		float* c1 = C + (size_t)m * ldc; \
		int n2 = 0; \
 \
		for (; n2 + 8 <= N; n2 += 8) \
			zq_gemm_32f_asm_core_m1n8(a1, Bt + (size_t)n2 * ldb, Bt + (size_t)(n2 + 4) * ldb, k8, s1, s3, c1 + n2); \
		for (; n2 + 4 <= N; n2 += 4) \
			zq_gemm_32f_asm_core_m1n4(a1, Bt + (size_t)n2 * ldb, k8, s1, s3, c1 + n2); \
		for (k = kstart; k < K; k++) \
		{ \
			float a = a1[k]; \
			for (j = 0; j < n2; j++) \
				c1[j] += a * Bt[(size_t)j * ldb + k]; \
		} \
		for (; n2 < N; n2++) \
		{ \
			const float* b1 = Bt + (size_t)n2 * ldb; \
			float sum = 0; \
			for (k = 0; k < K; k++) \
				sum += a1[k] * b1[k]; \
			c1[n2] = sum; \
		} \
	} \
}

ZQA_DEFINE_KERNEL(zq_gemm_32f_asm_k1n4, 1, 4)
ZQA_DEFINE_KERNEL(zq_gemm_32f_asm_k1n8, 1, 8)
ZQA_DEFINE_KERNEL(zq_gemm_32f_asm_k2n4, 2, 4)
ZQA_DEFINE_KERNEL(zq_gemm_32f_asm_k2n8, 2, 8)
ZQA_DEFINE_KERNEL(zq_gemm_32f_asm_k4n4, 4, 4)
ZQA_DEFINE_KERNEL(zq_gemm_32f_asm_k4n8, 4, 8)
ZQA_DEFINE_KERNEL(zq_gemm_32f_asm_k8n4, 8, 4)

#else /* ZQA_IMPL == 0 : 没有汇编内核时的占位, 保证符号存在 */

#define ZQA_FALLBACK \
	zq_gemm_32f_AnoTrans_Btrans_auto(M, N, K, A, lda, Bt, ldb, C, ldc)

#endif /* ZQA_IMPL */

/* ====================================================================== *
 * 对外接口 (签名与语义与 intrinsic 版同名函数完全一致)
 * ====================================================================== */

void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N4_asm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
#if ZQA_IMPL
	zq_gemm_32f_asm_k1n4(M, N, K, A, lda, Bt, ldb, C, ldc);
#else
	ZQA_FALLBACK;
#endif
}

void zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N8_asm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
#if ZQA_IMPL
	zq_gemm_32f_asm_k1n8(M, N, K, A, lda, Bt, ldb, C, ldc);
#else
	ZQA_FALLBACK;
#endif
}

void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N4_asm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
#if ZQA_IMPL
	zq_gemm_32f_asm_k2n4(M, N, K, A, lda, Bt, ldb, C, ldc);
#else
	ZQA_FALLBACK;
#endif
}

void zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N8_asm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
#if ZQA_IMPL
	zq_gemm_32f_asm_k2n8(M, N, K, A, lda, Bt, ldb, C, ldc);
#else
	ZQA_FALLBACK;
#endif
}

void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N4_asm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
#if ZQA_IMPL
	zq_gemm_32f_asm_k4n4(M, N, K, A, lda, Bt, ldb, C, ldc);
#else
	ZQA_FALLBACK;
#endif
}

void zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N8_asm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
#if ZQA_IMPL
	zq_gemm_32f_asm_k4n8(M, N, K, A, lda, Bt, ldb, C, ldc);
#else
	ZQA_FALLBACK;
#endif
}

void zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_N4_asm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
#if ZQA_IMPL
	zq_gemm_32f_asm_k8n4(M, N, K, A, lda, Bt, ldb, C, ldc);
#else
	ZQA_FALLBACK;
#endif
}

void zq_gemm_32f_AnoTrans_Btrans_auto_asm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
#if ZQA_IMPL
	int m = 0;

	if (M <= 0 || N <= 0)
		return;

	/* 每次只让子内核处理它自己那一块 (MB 行), 指针相应下移;
	   不能传 M - m, 否则第 m 块之后的行会被反复重算。 */
	while (m + 8 <= M)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_N4_asm(8, N, K, A + (size_t)m * lda, lda, Bt, ldb, C + (size_t)m * ldc, ldc);
		m += 8;
	}
	if (m + 4 <= M)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N8_asm(4, N, K, A + (size_t)m * lda, lda, Bt, ldb, C + (size_t)m * ldc, ldc);
		m += 4;
	}
	while (m + 2 <= M)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N4_asm(2, N, K, A + (size_t)m * lda, lda, Bt, ldb, C + (size_t)m * ldc, ldc);
		m += 2;
	}
	if (m < M)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N8_asm(1, N, K, A + (size_t)m * lda, lda, Bt, ldb, C + (size_t)m * ldc, ldc);
	}
#else
	zq_gemm_32f_AnoTrans_Btrans_auto(M, N, K, A, lda, Bt, ldb, C, ldc);
#endif
}
