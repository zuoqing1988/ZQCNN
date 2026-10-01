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
 *   所以整个 asm 块里不需要 push/pop, 也不碰 callee-saved 的 GPR。
 *     eax  = K 迭代次数 (K/8)
 *     r10  = A 第 0 行指针  (每次 +32)      循环结束后作废, 复用为写回指针
 *     rcx  = lda * 4 (字节)                 A 第 1 行 = [r10+rcx]
 *     r9   = Bt 第 0 行指针 (每次 +32)      循环结束后作废, 复用为写回指针
 *     r8d  = ldb * 4 (字节)                 Bt 相对行偏移
 *     edx  = ldb * 12 (字节) = 3 * ldb * 4
 *   A 的 2 行: [r10] / [r10+rcx]   (A 两行只差一个固定偏移, 所以行内只推进
 *   r10 一个指针, 每个 K 块比"两个 A 指针各推进"少一条 add)
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
 * 4) M4 x N1  (4 行 x 1 列) —— N=1 专用
 *
 *   N=1 时 4 列内核一块都用不上, 整列方向会退化成标量点积; 而 N=1 的
 *   形状 (例如 1024x1x1024) 恰恰是 GEMM 里最常见的瘦长型之一。这里沿 K
 *   方向做 ymm 累加, **一个 b 向量只加载一次、复用给 4 行 A**:
 *     ymm0..ymm3 : 4 行的累加器, ymm8 = b 当前 K 块, ymm9 = A 当前 K 块
 *     r10 = A 第 0 行 (每次 +32), rcx = lda*4, r11 = 3*lda*4,
 *     r9 = b (每次 +32), rax = 迭代次数
 *     A 的 4 行: [r10] / [r10+rcx] / [r10+rcx*2] / [r10+r11]
 *   每个 K 块 5 次 load + 4 条 FMA (相对 2 条/周期的 FMA 上限约 2.5~3 周期)。
 *
 * 5) M1 x N1  (1 行 x 1 列) —— M 尾部 1~3 行 / M<4
 *
 * 水平归约: 4 个累加器 -> 4 个连续 float 用 ZQA_RED4 (vhaddps 两级树,
 *   8 条指令 + 1 条 store, 依赖链 10 周期); 只有 N=1 的单累加器内核还用
 *   逐累加器的 ZQA_HSUM。写入范围严格落在 n..n+3 <= N-1 内。
 *
 * K 维处理: SIMD 只做 [0, K & ~7), 剩下 1~7 个元素由 C 标量循环补加。
 * 因此不依赖 A/Bt 行尾补零 (intrinsic 版依赖, 两边都能用)。
 *
 * FMA: 宏在 C 层分派, 不在 __asm {} 块里写 #if; 判定条件见下面的
 *   ZQA_HAVE_FMA 注释 (编译器允许发 vfmadd 就发, 与 MASM 侧一致)。
 *
 * GCC 侧输入传递: 全部输入用 "m" 约束。
 *   原因: 固定寄存器的 asm 里, 如果输入用 "r" 约束, 编译器可能把后读的
 *   输入分配到先写的寄存器上, 读到的就是被破坏的值; 用 "m" 约束后每个
 *   输入都在栈上有一份独立的拷贝, 什么时候读都安全, 因此可以把 C 写回
 *   指针留到循环之后再读 (对应 MSVC 的 __asm{} 直接读 C 变量的行为)。
 *
 * ============================ XMM 寄存器与 ABI ============================
 * 微内核要用到 xmm0-xmm14。这在 Linux (System V AMD64) 上无所谓 —— 那套 ABI
 * 里所有 xmm 都是 caller-saved, clobber 列表里写全即可 (下面已经写全)。
 * 但 **Windows x64 ABI 只有 xmm0-xmm5 是 volatile, xmm6-xmm15 是 callee-saved**,
 * 调用方会把值留在这些寄存器里跨越调用。所以:
 *   - MSVC 路径: 由 zq_gemm_32f_align_c_asm_msvc.asm 的 ZQA_PROLOGUE /
 *     ZQA_EPILOGUE 保存/恢复 xmm6-xmm15。
 *   - GCC/Clang 路径: clobber 列表里必须包含 xmm6-xmm15, 否则同样的问题会
 *     在 MinGW-w64 上重演。System V 下这一项是 no-op, 不产生任何代码。
 *
 * ============================ 平台 ============================
 *   MSVC x64        : 函数体内 __asm { } (Intel 语法)
 *   GCC/Clang x64   : 函数体内 __asm__ volatile (AT&T 语法)
 *   其他 / 无 AVX   : 整段用 #if 编译掉, 所有 _asm 符号回落调用
 *                      zq_gemm_32f_AnoTrans_Btrans_auto (intrinsic 版)
 */

#include <stdlib.h>
#include <string.h>

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

/* FMA 开关: 编译单元启用了 FMA (gcc/clang 的 -mfma, 即 __FMA__) 就走
 * vfmadd231ps, 否则退回 vmulps + vaddps。
 *
 * 为什么不再只看 ZQ_CNN_USE_FMADD256: 那个宏在 Linux 侧由 ZQ_CNN_USE_SSETYPE
 * 推导, 而 ZQCNN/ZQ_CNN_CompileConfig.h 的 Linux 分支写死成 AVX (不是 AVX2),
 * 于是同一个内核在 Linux 上是 vmulps+vaddps (每个 FMA 两条指令, 8 个累加器
 * 就要 16 条, 内层循环直接慢一倍), 在 Windows 上却是 vfmadd231ps —— 两端
 * 行为不一致, 也正好解释了为什么 Linux 侧 asm/MKL 系统性低 10~20 个百分点
 * (MKL 本身就带 FMA)。
 * 现在改成"编译器允许发 vfmadd 就发", 与 MASM 侧对齐。项目里 Linux 的
 * CMake 构建 (add_compile_options(-mavx2)) 和 build_check_gemm.sh
 * (gcc -mavx2 -mfma) 都是 AVX2 及以上, 而 FMA3 与 AVX2 一同随 Haswell 上市,
 * 不存在"有 AVX2 没 FMA3"的 CPU, 所以不会缩小实际可运行的 CPU 范围;
 * 真要跑在纯 AVX 的机器上, 不加 -mfma 就自动退回 vmulps+vaddps, 语义不变。
 * 注意: 这样一来 Linux 上 asm 与 intrinsic 的 A/B 对比里 asm 独占 FMA,
 * asm/intr 这一列不再代表同一条指令路径 (asm/MKL 才是目标指标)。 */
#if (defined(ZQ_CNN_USE_FMADD256) && ZQ_CNN_USE_FMADD256) || defined(__FMA__)
#define ZQA_HAVE_FMA 1
#else
#define ZQA_HAVE_FMA 0
#endif

/* ====================================================================== *
 * 汇编片段原语 (两套语法, 同名宏)
 * ====================================================================== */

#if ZQA_IMPL

/* ====================================================================== *
 * 小 K 专用路径: 先把 B 打包成 N 连续, 再沿 N 方向算
 * ====================================================================== *
 *
 * 为什么要单独一条路: 其它微内核沿 **K** 方向做 ymm 累加, K < 8 时 k8 == 0,
 * K 循环一次都不进, 归约出来全是 0 —— 微内核等于**把 C 整块清零**, 真正的
 * 结果再由 C 侧标量 `Cc[i*ldc+j] += a*Bb[j*ldb+k]` 补加。对 C 做了
 * "清零 + 读改写"三趟访存, 而这一族形状的瓶颈本来就在 C 的访存上。
 *
 * 小 K 时正确的向量化方向是 **N** (C 沿 N 连续)。但 Bt 是 N x K 行主序,
 * Bt[j][k] = Bt[j*ldb + k], 相邻两列在内存里差 ldb 个 float 而不是相邻 ——
 * 直接对 Bt[j*ldb + k] 做 ymm 读, 8 个结果里 7 个是错的, 而且不崩溃。
 * 所以必须先把 B 打包成 [K][nc] (同一个 k 的所有列变成连续)。
 *
 * 打包按 N 分块 (每块 2048 列), 一块内把 M 扫完, 每个 B 元素只打包一次。
 * 打包代价 N*K 次读写, 相对 M*N*K 次计算在 M 大时可以忽略。
 *
 * 性能 (独立微基准, 512x512x1, 缓冲已预先打包好, 排除了打包开销):
 *   行内核 16.42 GF/s, 等效写 C 带宽 32.8 GB/s, 每次 31.9 us
 *   MKL    20.5  GF/s  ->  约为 MKL 的 80%
 * 纯写 C 的带宽上限约 40 GB/s (26.2 us), 即这一族已基本打到访存上限。
 *
 * 这里用**可自动向量化的 C** 而不是手写汇编: 内层是 `c[j] op= a[k]*p[j]`
 * 这种规整的连续访存, gcc/clang 在 -O3 -mavx2 下会自己生成 ymm 版本,
 * 可移植性、正确性和 Windows 侧(MASM)的一致性都白拿, 实测与手写 intrinsics 等价。
 * ====================================================================== */
static void zq_gemm_32f_asm_smallk_row(const float* a, const float* packed,
	int nc, int K, float* c)
{
	/* 注意: C 是**调用方给定的输出缓冲**, 内容是未定义的 (Windows 的
	   _aligned_malloc 尤其明显是垃圾), 不是我们分配过的零区。
	   所以 k==0 必须用赋值而不是 += —— Linux 上新 mmap 恰好是零, 会把这个
	   bug 藏起来, 只在 Windows 上炸 (实测 worst err 1.2e4, 4 个用例失败)。 */
	if (K == 1)
	{
		/* 退化成外积 —— 编译器能向量化成 load+broadcast+mul+store */
		const float a0 = a[0];
		for (int j = 0; j < nc; j++)
			c[j] = a0 * packed[j];
		return;
	}
	{
		const float a0 = a[0];
		const float* p0 = packed;
		for (int j = 0; j < nc; j++)
			c[j] = a0 * p0[j];
	}
	for (int k = 1; k < K; k++)
	{
		const float ak = a[k];
		const float* p = packed + (size_t)k * nc;
		for (int j = 0; j < nc; j++)
			c[j] += ak * p[j];
	}
}

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

void zq_gemm_32f_asm_core_m2n4(const float* a0, const float* b0,
	int k8, int s0, int s1, int s3, float* c0, float* c1);
void zq_gemm_32f_asm_core_m1n8(const float* a0, const float* b0, const float* b4,
	int k8, int s1, int s3, float* c0);
void zq_gemm_32f_asm_core_m1n4(const float* a0, const float* b0,
	int k8, int s1, int s3, float* c0);
void zq_gemm_32f_asm_core_m4n1(const float* a0, const float* b0,
	int k8, int s1, int s3, float* c0, int ldc4, int ldc12);
void zq_gemm_32f_asm_core_m1n1(const float* a0, const float* b0,
	int k8, float* c0);
/* 6 行 x 8 列, 小 K 专用的 N 方向外积内核 (ap/bp 是驱动打包好的面板,
   不是原始的 A / Bt —— 见 zq_gemm_32f_asm_ndir) */
void zq_gemm_32f_asm_core_m6n8(const float* ap, const float* bp,
	int K, float* c, int ldc);


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
#define ZQA_MOV32L(d, s)  "movslq %[" #s "], %%" #d "\n\t"
#define ZQA_XOR(r)        "vxorps %%" #r ", %%" #r ", %%" #r "\n\t"
#define ZQA_LD(r, m)      "vmovups " m ", %%" #r "\n\t"
#define ZQA_ST(m, r)      "vmovups %%" #r ", " m "\n\t"   /* AT&T 存储: 寄存器在前 */
/* !! 别把 4 个 float 的写回拆成 vmovlps + vmovhps !!
   2026-10-01 误以为 "vmovups 写 16 字节 = 4 个 float, 会多写 12 字节越界",
   改成了两条 8 字节存储 —— 这是错的: **xmm 寄存器是 128 位 = 16 字节**, 4 个 float
   正好 16 字节, vmovups 一条不多写。(vshufps 0x44 的高 4 个 lane 虽然是重复值,
   但 vmovups 作用在 xmm 上只写低 128 位, 根本碰不到那 4 个 lane。)
   拆分只会凭空多一条指令。已回退, 留这段注释防止下次再"修"一遍。 */
#define ZQA_BC(r, m)      "vbroadcastss " m ", %%" #r "\n\t"  /* 必须是**内存源**, 见 m6n8 */
#define ZQA_STSS(m, r)    "vmovss %%" #r ", " m "\n\t"    /* 只写 1 个 float, N=1 用 */
#define ZQA_LEA(d, m)     "leaq " m ", %%" #d "\n\t"
#define ZQA_ADD32(d)      "addq $32, %%" #d "\n\t"
#define ZQA_DECEAX        "decl %%eax\n\t"
#define ZQA_TSTEAX        "testl %%eax, %%eax\n\t"
#define ZQA_M0(b)         "(%%" #b ")"
#define ZQA_M1(b, i)      "(%%" #b ",%%" #i ",1)"
#define ZQA_M2(b, i, s)   "(%%" #b ",%%" #i "," #s ")"
#define ZQA_MI(b, o)      #o "(%%" #b ")"
#define ZQA_VZ            "vzeroupper\n\t"
/* 4 个累加器 -> 4 个连续 float 的归约见下面的 ZQA_RED4P / ZQA_RED4 / ZQA_RED4SS。
   ZQA_HSUM 只剩 N=1 的单累加器内核还在用。 */
#define ZQA_HSUM(r, rl) \
	"vextractf128 $1, %%" #r ", %%xmm14\n\t" \
	"vaddps %%xmm14, %%" #rl ", %%" #rl "\n\t" \
	"vhaddps %%" #rl ", %%" #rl ", %%" #rl "\n\t" \
	"vhaddps %%" #rl ", %%" #rl ", %%" #rl "\n\t"

/* ---- 4 个累加器 -> 4 个连续 float 的归约 --------------------------------
 *
 * 逐累加器做 (vextractf128 + vaddps + vhaddps x2) 要 4 条指令/累加器, 外加
 * 3 条 vinsertps 拼包, 一行 4 列就是 19 条, 依赖链 12+9=21 周期。K 很小
 * (例如 32, k8=4) 时这已经是整个微内核调用里最大的一块固定开销。
 *
 * 改用 vhaddps 的 2 级树, 一次处理两个累加器:
 *   vhaddps ymm_a, ymm_a, ymm_b  ->  低 128 位 = [A,B,C,D], A+B = 第 a 列之和,
 *                                          C+D = 第 b 列之和; 高 128 位同理
 *   vextractf128 + vaddps       ->  把 256 的高/低 128 位加起来
 *   vhaddps xmm, xmm, xmm       ->  [a, b, a, b] (两个标量已在 lane 0/1)
 *   vshufps 0x44                ->  [a, b, c, d] 可以一次 vmovups 写出
 * 8 条指令 + 1 条 store, 依赖链 3+3+3+1 = 10 周期。
 * 注意收尾的立即数是 0x44 (取 SRC1[0],SRC1[1],SRC2[0],SRC2[1]) 而不是
 * 常见的 0x88 —— 后者取的是 lane 0/2, 而归约后 lane 0 和 lane 2 装的是同一个标量。
 */
#define ZQA_RED4P(ya, yb, yc, yd, xa, xc, t) \
	"vhaddps %%" #yb ", %%" #ya ", %%" #ya "\n\t" \
	"vhaddps %%" #yd ", %%" #yc ", %%" #yc "\n\t" \
	"vextractf128 $1, %%" #ya ", %%" #t "\n\t" \
	"vaddps %%" #t ", %%" #xa ", %%" #xa "\n\t" \
	"vhaddps %%" #xa ", %%" #xa ", %%" #xa "\n\t" \
	"vextractf128 $1, %%" #yc ", %%" #t "\n\t" \
	"vaddps %%" #t ", %%" #xc ", %%" #xc "\n\t" \
	"vhaddps %%" #xc ", %%" #xc ", %%" #xc "\n\t"
/* 收尾: a = [n0,n1,n0,n1], c = [n2,n3,n2,n3] -> a = [n0,n1,n2,n3] */
#define ZQA_RED4(ya, yb, yc, yd, xa, xc, t) ZQA_RED4P(ya, yb, yc, yd, xa, xc, t) \
	"vshufps $0x44, %%" #xc ", %%" #xa ", %%" #xa "\n\t"
/* 4 行 x 1 列: 4 个结果要散写, 所以多两条 vshufps 把 lane 1 复制出来 */
#define ZQA_RED4SS(ya, yb, yc, yd, xa, xb, xc, xd, t) ZQA_RED4P(ya, yb, yc, yd, xa, xc, t) \
	"vshufps $0x55, %%" #xa ", %%" #xa ", %%" #xb "\n\t" \
	"vshufps $0x55, %%" #xc ", %%" #xc ", %%" #xd "\n\t"

#if ZQA_HAVE_FMA
#define ZQA_FMA(d, a, b) "vfmadd231ps %%" #b ", %%" #a ", %%" #d "\n\t"
#else
#define ZQA_FMA(d, a, b) "vmulps %%" #b ", %%" #a ", %%ymm15\n\tvaddps %%ymm15, %%" #d ", %%" #d "\n\t"
#endif

/* ====================================================================== *
 * 三个微内核的循环体 (MASM 版与内联汇编版共用同一套结构)
 * ====================================================================== */

/* 2 行 x 4 列, 8 个累加器
   A 的两行只差一个固定的 lda*4 偏移, 所以只推进 r10 一个指针, 第二行用
   [r10 + rcx] 取 —— 每个 K 块少一条 add。B 的 4 行同理, 固定偏移 r8/s3。
   OA/OB 是字节位移, 2 路展开时第二路要带 32。 */
#define ZQA_STR(x)     #x
#define ZQA_M0O(b, o)  ZQA_STR(o) "(%%" #b ")"
#define ZQA_M1O(b, i, o) ZQA_STR(o) "(%%" #b ",%%" #i ",1)"
#define ZQA_M2O(b, i, s, o) ZQA_STR(o) "(%%" #b ",%%" #i "," #s ")"
#define ZQA_BODY_M2N4_O(oa, ob) \
	ZQA_LD(ymm8,  ZQA_M0O(r10, oa)) \
	ZQA_LD(ymm9,  ZQA_M1O(r10, rcx, oa)) \
	ZQA_LD(ymm10, ZQA_M0O(r9, ob)) \
	ZQA_LD(ymm11, ZQA_M1O(r9, r8, ob)) \
	ZQA_LD(ymm12, ZQA_M2O(r9, r8, 2, ob)) \
	ZQA_LD(ymm13, ZQA_M1O(r9, rdx, ob)) \
	ZQA_FMA(ymm0, ymm8, ymm10) \
	ZQA_FMA(ymm1, ymm8, ymm11) \
	ZQA_FMA(ymm2, ymm8, ymm12) \
	ZQA_FMA(ymm3, ymm8, ymm13) \
	ZQA_FMA(ymm4, ymm9, ymm10) \
	ZQA_FMA(ymm5, ymm9, ymm11) \
	ZQA_FMA(ymm6, ymm9, ymm12) \
	ZQA_FMA(ymm7, ymm9, ymm13)
#define ZQA_BODY_M2N4 ZQA_BODY_M2N4_O(0, 0)
/* 注: K 循环**没有**做 2 路展开 (所以只需要 OA=OB=0 这一路)。
   m2n4 每个 K 块是 6 次 load + 8 条 FMA, 在 Zen 3 上 8 条 FMA 只要 4 个周期
   (2 条/周期), 而 6 次 load 只要 3 个 (2 次/周期), 18 条 uop 除以 6 宽发射也
   只要 3 个周期 —— 这一段本来就是 FMA 吞吐瓶颈而不是发射瓶颈, 展开只能省下
   dec/jnz 那一条。实测确实无收益 (见 m2n4 的 2 路展开版本), 故保持单路。 */
#define ZQA_KLOOP_L \
	"0:\n\t" \
	ZQA_BODY_M2N4 \
	ZQA_ADD32(r10) ZQA_ADD32(r9) \
	ZQA_DECEAX \
	"jnz 0b\n\t" \
	"1:\n\t"

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

/* 4 行 x 1 列: b 向量只加载一次, 复用给 4 行 A。
   r10 = 块内最后一行 A, r11 = lda*4, rdx = 3*lda*4, r9 = b。 */
#define ZQA_BODY_M4N1 \
	ZQA_LD(ymm8, ZQA_M0(r9)) \
	ZQA_LD(ymm9, ZQA_M0(r10)) \
	ZQA_FMA(ymm0, ymm9, ymm8) \
	ZQA_LD(ymm9, ZQA_M1(r10, r11)) \
	ZQA_FMA(ymm1, ymm9, ymm8) \
	ZQA_LD(ymm9, ZQA_M2(r10, r11, 2)) \
	ZQA_FMA(ymm2, ymm9, ymm8) \
	ZQA_LD(ymm9, ZQA_M1(r10, rdx)) \
	ZQA_FMA(ymm3, ymm9, ymm8)

/* ====================================================================== *
 * 微内核 1: 2 行 x 4 列
 *
 * 只处理 SIMD 部分 (K 维前 k8*8 个元素), 结果覆盖写入 c0[0..3] / c1[0..3];
 * K 尾部由调用方用标量累加补上。k8 == 0 时写出全 0, 与 intrinsic 版
 * "先覆盖写入再补加 K 尾部" 的行为一致。
 * ====================================================================== */

static ZQA_NOINLINE void zq_gemm_32f_asm_core_m2n4(const float* a0, const float* b0,
	int k8, int s0, int s1, int s3, float* c0, float* c1)
{
	__asm__ volatile (
		ZQA_MOV32(eax, k8)
		ZQA_MOV64(r10, a0)
		ZQA_MOV64(r9, b0)
		ZQA_MOV32(ecx, s0)     /* = lda*4, 恒为正, movl 顺带清掉 rcx 高 32 位 */
		ZQA_MOV32(r8d, s1)
		ZQA_MOV32(edx, s3)
		ZQA_XOR(ymm0) ZQA_XOR(ymm1) ZQA_XOR(ymm2) ZQA_XOR(ymm3)
		ZQA_XOR(ymm4) ZQA_XOR(ymm5) ZQA_XOR(ymm6) ZQA_XOR(ymm7)
		ZQA_TSTEAX
		"jz 1f\n\t"
		ZQA_KLOOP_L
		ZQA_MOV64(r9, c0)
		ZQA_MOV64(r10, c1)
		ZQA_RED4(ymm0, ymm1, ymm2, ymm3, xmm0, xmm2, xmm8)
		ZQA_ST(ZQA_M0(r9), xmm0)
		ZQA_RED4(ymm4, ymm5, ymm6, ymm7, xmm4, xmm6, xmm8)
		ZQA_ST(ZQA_M0(r10), xmm4)
		ZQA_VZ
		:
		: [k8]"m"(k8), [a0]"m"(a0), [b0]"m"(b0),
		  [s0]"m"(s0), [s1]"m"(s1), [s3]"m"(s3), [c0]"m"(c0), [c1]"m"(c1)
		: "cc", "memory", "rax", "rcx", "rdx", "r8", "r9", "r10", "r11",
		  "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"
	);
}

/* ====================================================================== *
 * 微内核 2: 4 行 x 4 列 —— M 维"批量"主内核
 *
 * 16 个累加器放不下 (还要 4 个 B + 2 个 A, 一共 20 > 16), 所以一次调用里
 * 跑**两遍** K 循环: 第一遍算第 0、1 行, 第二遍算第 2、3 行, 中间共用同
 * 一份入参装载和同一次 prologue/epilogue。相比两次 m2n4 调用省下的是:
 *   - 1 次函数调用 (约 5 条 uop)
 *   - 1 份入参装载 (6 条)
 *   - Windows 侧 20 条 xmm6-xmm15 保存/恢复 + 1 条 vzeroupper
 *   - 代价是第二遍要把被冲掉的 eax/r10/r9 重新装回来 (3~6 条)
 * 净省 17 条 (Linux) / 40 条 (Windows) uop 每 4 行。K 越小占比越高:
 * K=32 (k8=4) 时固定开销从 ~34% 降到 ~23%, K=128 时从 ~10% 降到 ~9%。
 *
 * 寄存器: rax=k8, r10=A 第 0(第二遍第 2)行的 K 指针, rcx=s0=lda*4,
 *         r9=B 第 0 行的 K 指针, r8d=s1=ldb*4, edx=s3=3*ldb*4。
 *         A 的两行 = [r10] / [r10+rcx]; C 的 4 行按 ldc*4 / 2*ldc*4 /
 *         3*ldc*4 散写 (r9 装 c0, r10 装 ldc*4, edx 装 3*ldc*4)。
 * ====================================================================== */

/* ====================================================================== *
 * 微内核 3: 1 行 x 8 列
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
		ZQA_RED4(ymm0, ymm1, ymm2, ymm3, xmm0, xmm2, xmm8)
		ZQA_ST(ZQA_M0(rcx), xmm0)
		ZQA_RED4(ymm4, ymm5, ymm6, ymm7, xmm4, xmm6, xmm8)
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
		ZQA_RED4(ymm0, ymm1, ymm2, ymm3, xmm0, xmm2, xmm8)
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

/* ====================================================================== *
 * 微内核 4: 4 行 x 1 列 —— N=1 专用
 *
 * N=1 时 M2xN4 那套 4 列内核完全用不上 (N 尾部会退回整列标量), 而
 * 1x1024x1024 那种"单行"路径反而很快 (146% MKL), 慢的纯粹是"单列"。
 * 这里沿 K 方向做 ymm 累加, 一个 b 向量只加载一次、复用给 4 行 A:
 * 每 8 个 K 元素 5 次 load + 4 条 vfmadd231ps, 相对 FMA 上限约 2.5~3 周期。
 *
 * a0 指向块内第 0 行, 其余 3 行用 +lda*4 / +2*lda*4 / +3*lda*4 取, 行内推进
 * 只需要一个寄存器 (r10), 另两个寄存器留给固定偏移。
 * c0 指向块内第 0 行, 其余 3 行按 ldc*4 / 2*ldc*4 / 3*ldc*4 写。
 * ====================================================================== */

static ZQA_NOINLINE void zq_gemm_32f_asm_core_m4n1(const float* a0, const float* b0,
	int k8, int s1, int s3, float* c0, int ldc4, int ldc12)
{
	__asm__ volatile (
		ZQA_MOV32(eax, k8)
		ZQA_MOV64(r10, a0)
		ZQA_MOV64(r9, b0)
		ZQA_MOV32L(r11, s1)
		ZQA_MOV32(edx, s3)
		ZQA_XOR(ymm0) ZQA_XOR(ymm1) ZQA_XOR(ymm2) ZQA_XOR(ymm3)
		ZQA_TSTEAX
		"jz 1f\n\t"
		"0:\n\t"
		ZQA_BODY_M4N1
		ZQA_ADD32(r10)
		ZQA_ADD32(r9)
		ZQA_DECEAX
		"jnz 0b\n\t"
		"1:\n\t"
		ZQA_MOV64(rcx, c0)
		ZQA_MOV32(r8d, ldc4)
		ZQA_MOV32(edx, ldc12)
		ZQA_RED4SS(ymm0, ymm1, ymm2, ymm3, xmm0, xmm1, xmm2, xmm3, xmm8)
		ZQA_STSS(ZQA_M0(rcx), xmm0)
		ZQA_STSS(ZQA_M1(rcx, r8), xmm1)
		ZQA_STSS(ZQA_M2(rcx, r8, 2), xmm2)
		ZQA_STSS(ZQA_M1(rcx, rdx), xmm3)
		ZQA_VZ
		:
		: [k8]"m"(k8), [a0]"m"(a0), [b0]"m"(b0), [s1]"m"(s1), [s3]"m"(s3),
		  [c0]"m"(c0), [ldc4]"m"(ldc4), [ldc12]"m"(ldc12)
		: "cc", "memory", "rax", "rcx", "rdx", "r8", "r9", "r10", "r11",
		  "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"
	);
}

/* ====================================================================== *
 * 微内核 5: 1 行 x 1 列 —— M 尾部 1~3 行 / M<4
 * ====================================================================== */

static ZQA_NOINLINE void zq_gemm_32f_asm_core_m1n1(const float* a0, const float* b0,
	int k8, float* c0)
{
	__asm__ volatile (
		ZQA_MOV32(eax, k8)
		ZQA_MOV64(r10, a0)
		ZQA_MOV64(r9, b0)
		ZQA_XOR(ymm0)
		ZQA_TSTEAX
		"jz 1f\n\t"
		"0:\n\t"
		ZQA_LD(ymm8, ZQA_M0(r9))
		ZQA_LD(ymm9, ZQA_M0(r10))
		ZQA_FMA(ymm0, ymm9, ymm8)
		ZQA_ADD32(r10)
		ZQA_ADD32(r9)
		ZQA_DECEAX
		"jnz 0b\n\t"
		"1:\n\t"
		ZQA_MOV64(rcx, c0)
		ZQA_HSUM(ymm0, xmm0)
		ZQA_STSS(ZQA_M0(rcx), xmm0)
		ZQA_VZ
		:
		: [k8]"m"(k8), [a0]"m"(a0), [b0]"m"(b0), [c0]"m"(c0)
		: "cc", "memory", "rax", "rcx", "rdx", "r8", "r9", "r10", "r11",
		  "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"
	);
}

/* ====================================================================== *
 * 微内核 6: 6 行 x 8 列 —— 小 K 专用, 沿 N 方向做 ymm 累加（外积式）
 * ====================================================================== *
 *
 * 前 5 个微内核都沿 **K** 方向做 ymm 点积: 一个 K 块 (8 个 float) 一次
 * vmovups 读进 ymm, 做 8 条 FMA, 最后再做一次水平归约。这个形状在小 K 下
 * 会被归约吃光:
 *   m2n4 每个 K 块是 6 次 load + 8 条 FMA + 结束时 19 条归约 + 2 次 store。
 *   K=16 (k8=2) 时归约占了整个调用的一半以上指令, K=8 时一条 K 循环都不进,
 *   归约出来全是 0。实测 313x32x28 只有 MKL 的 25%、1024x1024x16 只有 49%。
 *
 * 这一族正确的向量化方向是 **N**（C 沿 N 连续, AGENTS.md 里那张布局表）。
 * 但 Bt 是 N x K 行主序, 相邻两列在内存里差 ldb 个 float 而不是相邻 ——
 * 直接读 Bt[j*ldb + k] 的 8 个 float 会拿到同一列的 8 个 K 元素, **不崩溃、
 * 只是结果全错**。所以必须先打包。
 *
 * 打包后的布局（都由驱动 zq_gemm_32f_asm_ndir 完成）:
 *   ap[k*6 + i] = A 的第 i 行、第 k 个元素        (i = 0..5)
 *   bp[k*8 + j] = Bt 的第 j 列、第 k 个元素      (j = 0..7)
 * 于是每个 k 的 6 个 A 标量和 8 个 B 分量都是**连续**的:
 *   6 次 vbroadcastss ymm, m32  +  1 次 vmovups ymm (B)  ->  6 条 FMA
 * 累加器是 6 个 ymm, 每个装满一行的 8 列; K 循环结束后直接一次 32B store
 * 写回 C 的那一行 —— **完全没有水平归约**, 这是它比 m2n4 快的根本原因。
 *
 * 每 k 的 uop: 6 广播(都是 1 条 load-port uop) + 1 次 B 向量读 = 7 次取数,
 * 6 条 FMA。Zen 3 上 6 条 FMA 要 3 个周期、7 次 load 要 3.5 个周期 ——
 * 两边几乎正好平衡, 理论接近 FMA 峰值。
 *
 * !! 广播必须用**内存源** vbroadcastss ymm, m32 !!
 * 寄存器源（movss 进 xmm 再 vbroadcastss ymm, xmm）在本机 Zen 3 上慢约三个
 * 数量级, 见 AGENTS.md「汇编/低层代码规则」第 6 条。
 *
 * 寄存器: rsi = ap, rdx = bp, rdi = c, ecx = K 计数;
 *         r8/r9/r10/r11/rax = ldc*4/8/12/16/20（C 里第 1..5 行的字节偏移）
 *         ymm0-ymm5 = 6 行累加器, ymm8-ymm13 = 6 个广播, ymm15 = B 向量
 *
 * 入参一律用 "m" 约束 + 自己 mov 到寄存器: gcc 9 不接受 "r8"(x) 这种
 * 寄存器名约束, 见 AGENTS.md「汇编/低层代码规则」第 4 条。
 */
static ZQA_NOINLINE void zq_gemm_32f_asm_core_m6n8(const float* ap, const float* bp,
	int K, float* c, int ldc)
{
	/* C 里第 i 行的字节偏移; x86 地址 scale 只能是 1/2/4/8, 3 倍和 5 倍单独算 */
	const int ldc1 = ldc << 2, ldc2 = ldc << 3, ldc3 = ldc * 12,
	          ldc4 = ldc << 4, ldc5 = ldc * 20;
	__asm__ volatile (
		ZQA_XOR(ymm0) ZQA_XOR(ymm1) ZQA_XOR(ymm2)
		ZQA_XOR(ymm3) ZQA_XOR(ymm4) ZQA_XOR(ymm5)
		ZQA_MOV64(rsi, ap)
		ZQA_MOV64(rdx, bp)
		ZQA_MOV64(rdi, c)
		ZQA_MOV32(r8d, ldc1)
		ZQA_MOV32(r9d, ldc2)
		ZQA_MOV32(r10d, ldc3)
		ZQA_MOV32(r11d, ldc4)
		ZQA_MOV32(eax, ldc5)
		ZQA_MOV32(ecx, K)
		"testl %%ecx, %%ecx\n\t"
		"jz 2f\n\t"
		"3:\n\t"
		ZQA_LD(ymm15, ZQA_M0(rdx))                      /* B: 8 个列标量 */
		ZQA_BC(ymm8,  "(%%rsi)")
		ZQA_BC(ymm9,  "4(%%rsi)")
		ZQA_BC(ymm10, "8(%%rsi)")
		ZQA_BC(ymm11, "12(%%rsi)")
		ZQA_BC(ymm12, "16(%%rsi)")
		ZQA_BC(ymm13, "20(%%rsi)")
		ZQA_FMA(ymm0, ymm8, ymm15)
		ZQA_FMA(ymm1, ymm9, ymm15)
		ZQA_FMA(ymm2, ymm10, ymm15)
		ZQA_FMA(ymm3, ymm11, ymm15)
		ZQA_FMA(ymm4, ymm12, ymm15)
		ZQA_FMA(ymm5, ymm13, ymm15)
		"addq $24, %%rsi\n\t"
		"addq $32, %%rdx\n\t"
		"decl %%ecx\n\t"                                 /* 注意不能用 ZQA_DECEAX */
		"jnz 3b\n\t"                                     /* eax 存着 ldc5 */
		"2:\n\t"
		ZQA_ST(ZQA_M0(rdi), ymm0)
		ZQA_ST("(%%rdi,%%r8,1)", ymm1)
		ZQA_ST("(%%rdi,%%r9,1)", ymm2)
		ZQA_ST("(%%rdi,%%r10,1)", ymm3)
		ZQA_ST("(%%rdi,%%r11,1)", ymm4)
		ZQA_ST("(%%rdi,%%rax,1)", ymm5)
		ZQA_VZ
		:
		: [ap]"m"(ap), [bp]"m"(bp), [K]"m"(K), [c]"m"(c),
		  [ldc1]"m"(ldc1), [ldc2]"m"(ldc2), [ldc3]"m"(ldc3),
		  [ldc4]"m"(ldc4), [ldc5]"m"(ldc5)
		: "cc", "memory", "rax", "rcx", "rdx", "rdi", "rsi",
		  "r8", "r9", "r10", "r11",
		  "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5",
		  "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm15"
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
	int s0 = lda << 2; \
	int s1 = ldb << 2; \
	int s3 = s1 * 3; \
	const int mb = MB, nb = NB; \
	const int npair = NB >> 2; \
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
						zq_gemm_32f_asm_core_m2n4(Ab + i * lda, Bb + j * 4 * ldb, \
							k8, s0, s1, s3, \
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

/* ====================================================================== *
 * N < 4 的专用驱动 (整列方向只剩 1~3 列, 4 列微内核全部用不上)
 * 一次处理 1 列: M 行沿 K 方向 ymm 累加, K 尾部再由标量补上。
 * 语义与主驱动一致: 微内核覆盖写入 C, K 尾在其上累加。
 * ====================================================================== */
static void zq_gemm_32f_asm_ncol(int M, int K, const float* A, int lda, const float* b0,
	float* C, int ldc)
{
	int m, k;
	int k8 = K >> 3;
	int kstart = k8 << 3;
	int s1 = (int)(lda * 4);
	int s3 = (int)(lda * 12);
	int ldc4 = (int)(ldc * 4);
	int ldc12 = (int)(ldc * 12);

	for (m = 0; m + 4 <= M; m += 4)
		zq_gemm_32f_asm_core_m4n1(A + (size_t)m * lda, b0, k8, s1, s3,
			C + (size_t)m * ldc, ldc4, ldc12);
	for (; m < M; m++)
		zq_gemm_32f_asm_core_m1n1(A + (size_t)m * lda, b0, k8, C + (size_t)m * ldc);

	for (m = 0; m < M; m++)
	{
		const float* a1 = A + (size_t)m * lda;
		float* c1 = C + (size_t)m * ldc;
		float sum = *c1;
		for (k = kstart; k < K; k++)
			sum += a1[k] * b0[k];
		*c1 = sum;
	}
}

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

/* ====================================================================== *
 * 运行时 ISA 守卫
 *
 * 为什么需要: 汇编内核是用 AVX2(+FMA) 指令**编译期**钉死的
 * (#if ZQA_IMPL, 来源是 ZQ_CNN_USE_SSETYPE)。把这个二进制放到一台不支持
 * AVX2 的机器上, 会直接 SIGILL 崩掉, 而不是"慢一点但算对"。
 *
 * Intel MKL 靠 mkl_rt 这个分发器 + CPUID 做到同一件事: 一个库里带着
 * SSE/AVX2/AVX-512 多套内核, 运行时查 CPUID 再挑, 并且能用
 * MKL_ENABLE_INSTRUCTIONS 强制指定。我们做不了"一个库带多套" (MSVC 没有
 * per-function target 属性, 得按 ISA 编成多个 .obj/DLL, 那正是 MKL 的做法),
 * 但可以补上**最要紧的那一半**: 运行前查一次 CPU, 不支持就安全回落到
 * intrinsic 路径。这把"崩溃"变成"慢但正确"。
 *
 * 环境变量 ZQ_GEMM_ISA 用来覆盖自动检测, 对应 MKL_ENABLE_INSTRUCTIONS:
 *   auto(默认) / avx2 / sse / off
 * 设 off 或 sse 都会强制走 intrinsic 路径 (可以拿来验证回落是否正确)。
 * ====================================================================== */
/* 简单的 ASCII 大小写无关比较 (只用于解析环境变量, 不追求完备) */
static int zqa_stricmp(const char* a, const char* b)
{
	while (*a && *b) {
		int ca = (*a >= 'A' && *a <= 'Z') ? *a + 32 : *a;
		int cb = (*b >= 'A' && *b <= 'Z') ? *b + 32 : *b;
		if (ca != cb) return ca - cb;
		a++; b++;
	}
	return (unsigned char)*a - (unsigned char)*b;
}

static int zq_gemm_32f_asm_isa_state = -1;   /* -1 未查, 0 不可用, 1 可用 */

static int zq_gemm_32f_asm_cpu_has_avx2_fma(void)
{
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
	int regs[4];
	__cpuid(regs, 0);
	if (regs[0] < 1) return 0;
	__cpuidex(regs, 1, 0);
	const int osxsave = (regs[2] & (1 << 27)) != 0;
	const int avx     = (regs[2] & (1 << 28)) != 0;
	const int fma     = (regs[2] & (1 << 12)) != 0;
	if (!osxsave || !avx || !fma) return 0;
	/* XGETBV: bit1=OSXSAVE 状态, bit2=AVX 的 YMM 状态是否被操作系统打开。
	   少了这一步, 在某些 hypervisor / 容器里会拿到"支持 AVX"但一执行就 #UD。 */
	unsigned long long xcr0 = _xgetbv(0);
	return ((xcr0 & 0x6) == 0x6) ? 1 : 0;
#elif defined(__GNUC__) || defined(__clang__)
	__builtin_cpu_init();
	return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
	return 1;   /* 其它编译器: 无法探测, 维持原行为 */
#endif
}

static int zq_gemm_32f_asm_isa_usable(void)
{
	if (zq_gemm_32f_asm_isa_state >= 0)
		return zq_gemm_32f_asm_isa_state;
	const char* env = getenv("ZQ_GEMM_ISA");
	int ok;
	if (env == 0 || zqa_stricmp(env, "auto") == 0)
		ok = zq_gemm_32f_asm_cpu_has_avx2_fma();
	else if (zqa_stricmp(env, "off") == 0 || zqa_stricmp(env, "sse") == 0)
		ok = 0;
	else
		ok = zq_gemm_32f_asm_cpu_has_avx2_fma();   /* 写了不认识的值: 仍按自动 */
	zq_gemm_32f_asm_isa_state = ok;
	return ok;
}

/* M 方向的分块调度 (N 方向的分块由调用方切好, 这里只管 M)。
   每次只让子内核处理它自己那一块 (MB 行), 指针相应下移;
   不能传 M - m, 否则第 m 块之后的行会被反复重算。 */
/* always_inline: B 装得下 L2 时走的就是这个函数, 多一次函数调用在
   5x7x3 这种 2.5 GF/s 的小尺寸上是 7% 的纯开销。 */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((always_inline))
#else
__forceinline
#endif
static inline void zq_gemm_32f_asm_mblocks(int M, int N, int K, const float* A, int lda,
	const float* Bt, int ldb, float* C, int ldc)
{
	int m = 0;
	/* 8 列微内核只有在 N 至少 8 时才划算: NB=8 而 N 只有 4~7 时, 一个整块
	   都凑不齐, 全部落到标量尾部, 比 NB=4 还慢。原分发只看 M, N < 8 时
	   白白用 M4_N8 / M1_N8 (N=5、N=7 这类形状)。 */
	const int wide_n = (N >= 8);
	while (m + 8 <= M)
	{
		zq_gemm_32f_align256bit_AnoTrans_Btrans_M8_N4_asm(8, N, K, A + (size_t)m * lda, lda, Bt, ldb, C + (size_t)m * ldc, ldc);
		m += 8;
	}
	if (m + 4 <= M)
	{
		if (wide_n)
			zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N8_asm(4, N, K, A + (size_t)m * lda, lda, Bt, ldb, C + (size_t)m * ldc, ldc);
		else
			zq_gemm_32f_align256bit_AnoTrans_Btrans_M4_N4_asm(4, N, K, A + (size_t)m * lda, lda, Bt, ldb, C + (size_t)m * ldc, ldc);
		m += 4;
	}
	while (m + 2 <= M)
	{
		if (wide_n)
			zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N8_asm(2, N, K, A + (size_t)m * lda, lda, Bt, ldb, C + (size_t)m * ldc, ldc);
		else
			zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_N4_asm(2, N, K, A + (size_t)m * lda, lda, Bt, ldb, C + (size_t)m * ldc, ldc);
		m += 2;
	}
	if (m < M)
	{
		if (wide_n)
			zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N8_asm(1, N, K, A + (size_t)m * lda, lda, Bt, ldb, C + (size_t)m * ldc, ldc);
		else
			zq_gemm_32f_align256bit_AnoTrans_Btrans_M1_N4_asm(1, N, K, A + (size_t)m * lda, lda, Bt, ldb, C + (size_t)m * ldc, ldc);
	}
}

/* ====================================================================== *
 * 小 K 路径: 打包 A / B 后沿 N 方向算 (6x8 外积内核)
 * ====================================================================== *
 *
 * 返回 1 表示这一族形状已经算完; 返回 0 表示不适合 (N/M 太碎或分配失败),
 * 调用方要继续走原来的 K 方向路径。
 *
 * 循环结构照 BLIS 的三段式:
 *   外层 B 面板 (一次打包 nc 列) -> 中层 A 微面板 (一次打包 mc 行)
 *   -> 内层 (6 行 x 8 列) 微内核
 * A 微面板放在中层而不是每次重打包, 是为了让打包开销在 N 上摊开:
 * 打包代价 ~M*K, 计算量 M*N*K, 只要 nc >= 8 就摊薄到 1/8 以下。
 *
 * 尾部 (N 不是 8 的倍数 / M 不是 6 的倍数) 交回 zq_gemm_32f_asm_mblocks,
 * 它调的是公开的 M*_N*_asm 包装函数, 不会再回到这里, 所以不会递归。
 *
 * 面板大小: 一个 B 面板和一个 A 微面板各控制在 256KB 上下 (留在 L2),
 * 和 N 方向分块用的是同一个量级。
 */
#define ZQA_NDIR_MR 6
#define ZQA_NDIR_NR 8
#define ZQA_NDIR_PANEL_BYTES (256 * 1024)
/* 面板小于这个量就直接用栈上数组, 不 malloc。小形状上 malloc 的开销是
   压倒性的: 8x8x8 一共才 512 次乘加, 两次 malloc 就把它拖到 asm/intr 的
   0.22 倍 (2026-10-01 Windows 实测)。4096 个 float = 16KB, 两边默认栈都放得下。 */
#define ZQA_NDIR_STACK_FLOATS 4096
/* 走 N 方向打包路径的 K 上限。实测 (WSL + Ryzen 9 5900HX, asm/MKL):
     K=16   49% -> 打包后 6x8 内核明显更快
     K=32   57% -> 同上
     K=64   已经在噪声内, 打包的多一趟读写抵掉不了多少, 保守留在原路
   想复现就把这个值调大再跑 tools/bench_two_binaries.py 对一遍。 */
#ifndef ZQA_NDIR_MAX_K
#define ZQA_NDIR_MAX_K 32
#endif

static int zq_gemm_32f_asm_ndir(int M, int N, int K, const float* A, int lda,
	const float* Bt, int ldb, float* C, int ldc)
{
	const int nr = N & ~(ZQA_NDIR_NR - 1);        /* 8 的倍数列数 */
	const int mr = M - M % ZQA_NDIR_MR;           /* 6 的倍数行数 */
	long long want;
	int nc, mc, n0;
	float* Bp;
	float* Ap;
	float stackbuf[ZQA_NDIR_STACK_FLOATS];
	int on_stack;

	if (nr < ZQA_NDIR_NR || mr < ZQA_NDIR_MR || K <= 0)
		return 0;

	/* 一个面板装多少个元素: K 很小的时候别切得太碎, 一个 8 列块起步 */
	want = (long long)ZQA_NDIR_PANEL_BYTES / ((long long)K * 4);
	nc = (int)(want < ZQA_NDIR_NR ? ZQA_NDIR_NR : (want > nr ? nr : (int)want));
	nc &= ~(ZQA_NDIR_NR - 1);
	mc = (int)(want < ZQA_NDIR_MR * ZQA_NDIR_NR ? ZQA_NDIR_MR * ZQA_NDIR_NR
	                                          : (want > mr ? mr : (int)want));
	mc -= mc % ZQA_NDIR_MR;
	if (nc <= 0 || mc <= 0)
		return 0;

	/* 分配失败就老老实实回落到原路径, 不能让 GEMM 直接不做 */
	on_stack = ((size_t)nc * (size_t)K + (size_t)mc * (size_t)K
	            <= ZQA_NDIR_STACK_FLOATS);
	if (on_stack)
	{
		Bp = stackbuf;
		Ap = stackbuf + (size_t)nc * (size_t)K;
	}
	else
	{
		Bp = (float*)malloc(sizeof(float) * (size_t)nc * (size_t)K);
		Ap = (float*)malloc(sizeof(float) * (size_t)mc * (size_t)K);
		if (Bp == NULL || Ap == NULL)
		{
			free(Bp);
			free(Ap);
			return 0;
		}
	}

	for (n0 = 0; n0 < nr; n0 += nc)
	{
		const int ncn = (nr - n0 < nc) ? (nr - n0) : nc;
		const int nb = ncn / ZQA_NDIR_NR;
		int m0;
		/* 打包 B: Bt 是 N x K 行主序, 打包成**一块块 [K][8] 的小面板**。
		   注意不能打成 [K][ncn] 再按 c*8*K 取块 —— 那样每 k 的步长是 ncn 而不是 8,
		   微内核里 `Bp + c*8*K` 会指错位置。只有 ncn 恰好等于 8 时两种排布才碰巧一致,
		   于是 16x8x32 能过、32x32x32 算出 9.07 的误差 (2026-10-01 实测)。 */
		for (int c = 0; c < nb; c++)
		{
			float* p = Bp + (size_t)c * ZQA_NDIR_NR * K;
			for (int j = 0; j < ZQA_NDIR_NR; j++)
			{
				const float* b1 = Bt + (size_t)(n0 + c * ZQA_NDIR_NR + j) * ldb;
				float* pj = p + j;
				for (int k = 0; k < K; k++)
					pj[(size_t)k * ZQA_NDIR_NR] = b1[k];
			}
		}
		for (m0 = 0; m0 < mr; m0 += mc)
		{
			const int mcn = (mr - m0 < mc) ? (mr - m0) : mc;
			const int mb = mcn / ZQA_NDIR_MR;
			/* 打包 A: [6][K] 的行主序转成 [K][6] */
			for (int b = 0; b < mb; b++)
			{
				const float* a1 = A + (size_t)(m0 + b * ZQA_NDIR_MR) * lda;
				float* p = Ap + (size_t)b * ZQA_NDIR_MR * K;
				for (int k = 0; k < K; k++)
					for (int i = 0; i < ZQA_NDIR_MR; i++)
						p[(size_t)k * ZQA_NDIR_MR + i] = a1[(size_t)i * lda + k];
			}
			for (int b = 0; b < mb; b++)
				for (int c = 0; c < nb; c++)
					zq_gemm_32f_asm_core_m6n8(
						Ap + (size_t)b * ZQA_NDIR_MR * K,
						Bp + (size_t)c * ZQA_NDIR_NR * K,
						K,
						C + (size_t)(m0 + b * ZQA_NDIR_MR) * ldc + n0 + c * ZQA_NDIR_NR,
						ldc);
		}
	}
	if (!on_stack)
	{
		free(Bp);
		free(Ap);
	}
	n0 = nr;      /* 循环退出时 n0 应当正好等于 nr, 但别让下面两条尾巴语句依赖这个巧合 */
	/* 尾部: 先补 N 的余数列 (整行 M), 再补 M 的余数行 (只补已算的列) */
	if (n0 < N)
		zq_gemm_32f_asm_mblocks(M, N - n0, K, A, lda,
			Bt + (size_t)n0 * ldb, ldb, C + n0, ldc);
	if (mr < M)
		zq_gemm_32f_asm_mblocks(M - mr, n0, K, A + (size_t)mr * lda, lda,
			Bt, ldb, C + (size_t)mr * ldc, ldc);
	return 1;
}

void zq_gemm_32f_AnoTrans_Btrans_auto_asm(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
#if ZQA_IMPL
	int m = 0;

	if (M <= 0 || N <= 0)
		return;

	/* 运行前确认这台机器真的有 AVX2+FMA。没有就回落 intrinsic:
	   这个二进制是拿 -mavx2 -mfma 编出来的, 直接执行会 SIGILL。
	   ZQ_GEMM_ISA=off / sse 可以强制走回落, 用来验证这条路径。 */
	if (!zq_gemm_32f_asm_isa_usable())
	{
		zq_gemm_32f_AnoTrans_Btrans_auto(M, N, K, A, lda, Bt, ldb, C, ldc);
		return;
	}

	/* N < 4 时 4 列微内核一块都用不上, 整列方向退化成标量点积 —— 实测
	   1024x1x1024 只有 MKL 的 12%。这里换 N=1 专用内核 (沿 K 方向 ymm 累加)。 */
	if (N < 4)
	{
		for (m = 0; m < N; m++)
			zq_gemm_32f_asm_ncol(M, K, A, lda, Bt + (size_t)m * ldb, C + m, ldc);
		return;
	}
	/* K < 8 时 mb*nb 微内核的 K 循环一次都不进, 归约出来全是 0, 等于先把 C
	   整块清零再由标量补加 —— 对 C 做了三趟访存, 而这一族的瓶颈就在 C 的访存上。
	   改走"打包 B + 沿 N 方向算": C 只写一趟。独立微基准 512x512x1 达
	   16.4 GF/s (MKL 20.5), 约为 MKL 的 80%。 */
	if (K < 8)
	{
		const int chunk = 2048;                    /* 每块打包这么多列 */
		/* 用 malloc 而不是固定数组: N*K 由调用方决定, 不能写死上限 */
		const size_t buf_len = (size_t)((N < chunk) ? N : chunk) * K + 8;
		float* buf = (float*)malloc(sizeof(float) * buf_len);
		if (buf == 0)
			return;
		for (int n0 = 0; n0 < N; n0 += chunk)
		{
			int nc = (N - n0 < chunk) ? (N - n0) : chunk;
			/* 打包成 [K][nc]: 同一个 k 的所有列变成连续的 */
			for (int j = 0; j < nc; j++)
			{
				const float* b1 = Bt + (size_t)(n0 + j) * ldb;
				for (int k = 0; k < K; k++)
					buf[k * nc + j] = b1[k];
			}
			for (m = 0; m < M; m++)
				zq_gemm_32f_asm_smallk_row(A + (size_t)m * lda, buf, nc, K,
					C + (size_t)m * ldc + n0);
		}
		free(buf);
		return;
	}

	/* 8 <= K <= ZQA_NDIR_MAX_K: 打包 A/B 后走 6x8 的 N 方向外积内核。
	   这一族沿 K 方向做点积时, 水平归约的固定开销盖过计算本身:
	   313x32x28 只有 MKL 的 25%、1024x1024x16 只有 49%、1024x1024x32 只有 57%
	   (2026-10-01 补 -mfma 之后的读数)。K 再大时归约被 K 循环摊薄,
	   m2n4 的访存模式反而更省 —— 所以这个阈值是实测出来的, 不是拍的。
	   zq_gemm_32f_asm_ndir 内部会检查形状是否合适 (M>=6 / N>=8),
	   不合适时返回 0, 这里继续走原来的路。 */
	if (K <= ZQA_NDIR_MAX_K && zq_gemm_32f_asm_ndir(M, N, K, A, lda, Bt, ldb, C, ldc))
		return;


	/* N 方向分块: 原来是一路 m 扫到底, 每次把整个 B (最大可到几十 MB) 从
	   L3/内存重扫一遍, 只有当前 8 行的 A 面板留在 L2 里。实测 512^3 还有
	   MKL 的 94%, 到 2048^3 就掉到 67% —— 分水岭正好在 B 装不进 L2 的地方。
	   改成先把 B 切成能进 L2 的面板, 一个面板内把 M 扫完, 再换下一个面板。
	   64 形状对拍（每档取 3 次最好值, 抵消 boost 抖动）:
	     512x8192x512   37.9 -> 70.6 GF/s  (+86%)
	     1536^3          51.2 -> 61.5        (+20%)
	     2048^3          47.2 -> 55.9        (+18%)
	     1024^3          59.7 -> 63.5        (+6%)
	     512^3           70.5 -> 74.0        (+5%) */
	{
		/* B 本来就装得下 (或者 N 小到没法再切) 时不要分块:
		   多一层循环和一次函数调用, 对 5x7x3 这种 2.5 GF/s 的小尺寸
		   反而是 15% 的纯开销。 */
		const long long bbytes = (long long)N * K * 4;
		if (bbytes <= 256 * 1024 || N < 64)
		{
			zq_gemm_32f_asm_mblocks(M, N, K, A, lda, Bt, ldb, C, ldc);
			return;
		}
		/* 目标: 一个 B 面板 (nt 列 x K) 控制在 ~256KB, 留在 L2 里。 */
		long long want = 256 * 1024 / ((long long)K * 4);
		int nt = (want < 64) ? 64 : ((want > N) ? N : (int)want);
		nt &= ~3;                       /* 4 列微内核的整除 */
		if (nt <= 0)
			nt = 4;
		for (int n0 = 0; n0 < N; n0 += nt)
		{
			int nc = (N - n0 < nt) ? (N - n0) : nt;
			zq_gemm_32f_asm_mblocks(M, nc, K, A, lda,
				Bt + (size_t)n0 * ldb, ldb, C + n0, ldc);
		}
	}
#else
	zq_gemm_32f_AnoTrans_Btrans_auto(M, N, K, A, lda, Bt, ldb, C, ldc);
#endif
}
