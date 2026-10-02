# CHANGELOG 2026-10-02

## 新增/变更：第十一轮审计 —— 修掉自己刚加的 6x8 内核里一个静默算错的 bug

### 变更文件
- `ZQ_GEMM/math/zq_gemm_32f_align_c_asm.c`
  - 新增 `ZQA_FMA7` 宏：6x8 内核专用的 FMA 宏，无 FMA 时用 `ymm7` 而不是 `ymm15`
  - 6x8 内核的 6 条 FMA 全部改用 `ZQA_FMA7`，clobber 列表补 `xmm7`
- `audit_k3_20261001.md`：新增**附录 V**（第十一轮）
- `AGENTS.md`：行尾一节补第 5 条（只看退出码不够 + sample 要在产物目录跑）、
  第 6 条补 heredoc 变体；汇编一节补第 9 条（借寄存器当暂存是跨内核耦合）

### 问题

`ZQA_FMA` 在没有 FMA 时展开成两条指令，借 **`ymm15`** 当 `vmulps` 的暂存：

```
vmulps %ymm15, %ymm8, %ymm15     <-- 借 ymm15
vaddps %ymm15, %ymm0, %ymm0
```

而 2026-10-01 新加的 6x8 外积内核把 **B 向量放在 `ymm15`**，每个 k 循环要用它做
6 次 FMA。于是 `-mfma` 缺席时，第 1 条 FMA 的 `vmulps` 把 B 向量冲掉，后 5 条全错 ——
**不崩溃、不越界，只是结果错**。

### 实测结果

Linux 去掉 `-mfma` 编译，`SampleGEMMAsmCompare`：

```
   17    19    27 |    6.136e+00 ...   <== FAIL
   16     8    32 |    5.180e+00 ...   <== FAIL
   32    32    32 |    5.709e+00 ...   <== FAIL
  313    32    28 |    7.365e+00 ...   <== FAIL
result: FAIL (4 case(s) failed)
```

四个失败用例的 K 是 27/32/32/28，**全在 `ZQA_NDIR_MAX_K = 32` 里** ——
只有走新路径的形状会错，其余 14 个用例照常通过。这种「只错一部分」的形态比全错
更难发现。

修复：6x8 内核改用 `ZQA_FMA7`（无 FMA 时暂存用该内核空闲的 `ymm7`）。
同一套「不带 -mfma」的构建修复后 `PASS (0 case(s) failed)`，worst 3.8e-06。

### 是怎么发现的

不是靠读代码，是靠**编两次、把 sample 输出逐字节 diff**：
`tools/run_sample_regression.sh` 只看退出码，两次跑都是全 rc=0，但输出内容不一样。
顺带确认了 `-mfma` **不改变任何推理结果**：MTCNN / MTCNN_NCHWC4 / SSD /
CascadeOnet / CascadeOnet_Interface / FaceDetectorMTCNN / MTCNNLoadFromCode
七份输出在两个构建之间逐字节相同（滤掉耗时行后）。

另外发现一个一次性脚本的坑：**sample 必须在产物目录
`cmake-out-unix-x64/Release` 里跑**（CMake 把 `model/` 和 `data/` 联接到了那里），
从仓库根跑只会打一行 `empty image` —— 看着像跑过了，其实什么都没验。
`run_sample_regression.sh` 一直是对的。

### 验证
- Windows Release 全量构建 0 error，`SampleGEMMAsmCompare` PASS（worst 1.9e-06）
- Linux `C_FLAGS = -O3 -DNDEBUG -fPIC -Ofast -ffast-math -mavx2 -mfma`，
  `SampleGEMMAsmCompare` PASS（worst 3.8e-06）
- Linux 去掉 `-mfma` 的构建也 PASS（worst 3.8e-06）—— **这条分支现在有人走过了**
- 两平台 sample 回归全 rc=0
- `tools/check_line_endings.py` / `tools/check_text_encoding.py` 均 OK

### 注意事项
1. 其余 5 个微内核（m2n4 / m1n8 / m1n4 / m4n1 / m1n1）只用 ymm0-ymm13，
   `ymm15` 对它们空闲，所以**只有 6x8 撞上了这个坑**；MASM 侧写死
   `vfmadd231ps`，不受影响。
2. 已写进 `AGENTS.md` 汇编一节第 9 条：**编译期分支都要有人走过**。默认构建永远
   只走「有 FMA」那条，去掉 `-mfma` 是唯一能触发另一条的办法。

## 新增/变更：给 ZQ_MergeSort 补一个独立测试（推翻「无法验证」这个前提）

### 变更文件
- `tools/zq_mergesort_check.cpp`（新增）：`ZQ_MergeSort` 的脱离 ZQlibFaceID 的
  独立回归测试（正确性 + ASan/LSan + OOM 探针）
- `audit_k3_20261001.md`：附录 O 的「不修」结论按新证据修订

### 为什么做这件事

附录 O 把 `ZQ_MergeSort::_mergeSort_OOC` 的若干问题记为**不修**，理由是
「这是 3rdparty 第三方头，改动面大且**无法在无数据库的机器上端到端验证**」。

但 `ZQ_MergeSort.h` 只 include 了 4 个标准头（`stdlib.h` / `string.h` /
`stdio.h` / `iostream`），模板成员全是 static，**完全能脱离 ZQlibFaceID 单独
编译**。补上 MSVC 的 `__int64` / `__max` / `__min` 之后 gcc 9.4 直接编过。
**「无法验证」这个前提本身就不成立。**

### 实测结果

- 正确性：`MergeSort_OOC<float>` 14 种规模（0/1/2/3/7/8/9/15/16/17/100/1000/
  4096/5000）× 4 种 `max_mem_size_in_KB`（1/4/16/1024）× 升/降序，
  逐个与 `std::sort` 的结果**逐元素相等**；
  `MergeSortWithData_OOC<float>` 14 种规模 × 2 个方向，**载荷与键保持对应**
  （用每个 id 的原始分数反查）。
- **ASan + LeakSanitizer 全程无报告**：无泄漏、无越界、无 use-after-free。
  这同时推翻了附录 O 里「~20 条错误返回路径全泄漏 4 块缓冲」的说法 ——
  那些路径在本测试覆盖到的成功路径上没有出现泄漏。
- 空输入（n=0）两个入口都返回 false，是合理的契约，测试已按此断言。
- OOM 探针：用 `ulimit -v` 把地址空间卡到 256MB，让 `val_size*block_size*2`
  （256MB）分配失败，**没有崩溃，返回 false**。

### 仍然遗留（**未取得端到端证据**，不声称已修）

`ZQ_MergeSort.h` 里一共 8 处 `malloc` **全部不判空**：
`val_block_buffer` / `data_block_buffer` / `out_val_buffer` / `out_data_buffer`
（都在 `_mergeSort_OOC` 里）以及 `_mergeSortWithData` 的
`tmp_data_left` / `tmp_data_right`。分配失败时返回值直接拿去 `memcpy` /
`fread`。

我的 OOM 探针没有复现出崩溃，所以**按「按构造正确 + 现有调用点（`sort_score_file`
的 8 处）在内存充足时不会失败」记录**，不改代码。要复现崩溃需要更精确地卡住
某一次分配，而这一步没做成。

### 注意事项
1. `ZQ_MergeSort.h` 里还留着调试输出：每次归并轮次都 `printf` 形如
   `1/3 1024` 的进度行（跑本测试时刷了 90 多行）。这是上游代码的行为，
   本轮不改，但它会污染任何调用它的程序的 stdout。
2. 头文件在 gcc 下需要 `__int64` / `__min` / `__max`；测试文件里在
   `#include` **之前**补了 typedef/macro，顺序反了就编不过。

## 新增/变更：把「第三方头无法验证」变成一张表 —— 143 个 ZQlib 头里 81 个能独立编译

### 变更文件
- `tools/probe_zqlib_headers.py`（新增）：逐个生成最小翻译单元探测 ZQlib 头能否独立编译
- `3rdparty/include/ZQlib/ZQ_MergeSort.h`：补 `#include <vector>`
- `3rdparty/include/ZQlib/ZQ_Kmeans.h`：补 `#include <math.h>`
- `tools/bench_gemm_ab.py`：修 WSL 喂脚本时的 CRLF 问题（见下）
- `audit_k3_20261001.md`：新增**附录 X**

### 起因

ZQ_MergeSort 那条「不修」被推翻之后（附录 W），剩下的疑问是：
**还有多少「这是第三方头、无法验证所以不修」其实同样站不住？**
不逐个量一下就永远不知道。143 个头，人工翻不动 —— 所以写了个探测器。

### 工具做法

给每个头生成一个只 `#include` 它的最小翻译单元，前面垫一层
MSVC 兼容 shim（`__int64` / `__min` / `__max` / `_fseeki64` / `_ftelli64` /
`fopen_s` / `strcpy_s` / `sprintf_s`），用 gcc `-fsyntax-only -std=c++11` 编，
按第一条 error 分类成 OK / NEEDS_LIB / MSVC_ONLY / BROKEN。

### 实测结果（gcc, Linux, 143 个头）

| 分类 | 修复前 | 修复后 |
|---|---|---|
| **OK（可以单独验证）** | **81** | **83** |
| NEEDS_LIB（确实要 Windows/MFC/OpenCV） | 6 | 6 |
| MSVC_ONLY（补个 shim 就能救） | 1 | 1 |
| BROKEN（缺兄弟头或真有问题） | 55 | 53 |

真正无法验证的只有 6 个：`ZQ_Logger` / `ZQ_ProtectedData` / `ZQ_PutTextCN` /
`ZQ_SemaphoreEx`（要 `windows.h`）、`ZQ_MFC_Utils`（要 MFC `afx`）、
`ZQ_StereoDisparity_CV2`（要 `opencv2/`）。其余 137 个都可以端到端测。

### 顺带修掉两个「头不自足」的真缺陷

两个头都**只靠 MSVC 的传递 include 才能编过**，libstdc++ 下直接失败：

- `ZQ_MergeSort.h`：`_mergeSort_OOC` 里用了 `std::vector<char>`，但只 include 了
  `<iostream>`。MSVC 的 `<iostream>` 会顺带带进 `<vector>`，libstdc++ 不会。
  实测报错：`ZQ_MergeSort.h:928:9: error: 'vector' is not a member of 'std'`。
  **任何恰好先 include 了 `<vector>` 的调用方都会把这个坑掩盖掉** ——
  我自己手写的第一版测试就是这样蒙混过去的。
- `ZQ_Kmeans.h`：用了 `fabs()` 却没 include `<math.h>`。

补上之后两个头都进 OK 列表。补完重新跑 `tools/zq_mergesort_check.cpp`：仍 PASS。

### 顺带修掉 tools/bench_gemm_ab.py 的一个真 bug

它的 `wsl(script)` 用 `subprocess.run(text=True, input=script)` 把脚本喂给
`bash -s`。**Windows 上文本模式会把 `
` 翻译成 `
`**，WSL 里的 bash 于是看到：

```
bash: line 1: set: +: invalid option
bash: line 2: cd: $'zqprobe2
': No such file or directory
bash: line 56: syntax error: unexpected end of file
```

一行都没跑。改成 `input=script.encode('utf-8')` 后正常。修完用**空对照**
（同一个文件跟自己比）验证工具本身可用：64 个形状里 61 个在 8% 噪声内、
1 个 A 快、2 个 B 快。

**无法断言此前用这个工具报出的数字是否受影响**（它当时也可能是在别的调用方式下
正常跑的）。本轮所有新数字都来自 `tools/bench_two_binaries.py` 与
`tools/gemm_mkl_ratio.py`，两者都走 `wsl bash -c`，不走这条 stdin 路径。

### 注意事项
1. 剩下 53 个 BROKEN 里绝大多数不是「头有问题」，而是**缺兄弟头 / 缺 OpenCV**
   （例如 `ZQ_TaucsBase.h` 缺 taucs、`ZQ_ImageIO.h` 缺 `opencv2/`、
   `ZQ_WinSockBase.h` 缺 winsock）。要真审这批得先把依赖装齐，不在本轮范围。
2. 附录 O 里还有几条以「第三方头 / 死代码」为由的「不修」，现在有工具可以逐条
   复核 —— 下一轮按 `probe_zqlib_headers.py` 的 OK 列表逐个重判。

## 新增/变更：按附录 X 的探测结果重判「不修」项 —— ZQ_Kmeans 的 k<=0 已修

### 变更文件
- `3rdparty/include/ZQlib/ZQ_Kmeans.h`：4 个入口补 `k <= 0` 守卫；
  `_select_init_center` 补完整入参守卫
- `tools/zq_kmeans_check.cpp`（新增）：`ZQ_Kmeans` 的独立回归测试
- `audit_k3_20261001.md`：附录 O 那条「不修」按新证据修订

### 起因

附录 X 证明 143 个 ZQlib 头里 83 个能独立编译，于是附录 O 里
「`ZQ_Kmeans.h` 的 `k == 0` 往零长数组写 — 不修：仅经死代码链引用」
这条的**第二个理由**（没法验证）也不成立了，可以直接测。

### 实测：bug 确实存在，ASan 精确定位

`tools/zq_kmeans_check.cpp`（ASan + LeakSanitizer）跑 `k = 0`：

```
ERROR: AddressSanitizer: heap-buffer-overflow ... READ of size 4
    #0 ZQ::ZQ_Kmeans<float>::Kmeans_with_init  ZQ_Kmeans.h:54
0x... is located 0 bytes to the right of 1-byte region
allocated by thread T0 here:
    #1 operator new[](unsigned long)
    #2 ZQ::ZQ_Kmeans<float>::Kmeans_with_init  ZQ_Kmeans.h:28
```

机理：守卫只判了 `k > nPts`、**没判 `k <= 0`**。`k == 0` 时 `new int[0]`
返回的是一个 1 字节的合法指针，而 `min_kid` 初始化为 0、`j` 循环不执行，
于是每个点的 `idx[i]` 都是 0，紧接着 `sum_kid[kid]++` 就是**零长数组越界写**。
`k < 0` 更糟：`new int[-1]` 抛 `std::bad_array_new_length`，没人接 → 进程终止。

### 修法

4 个公开入口（`Kmeans_with_init` / `KmeansNormVec_with_init` / `Kmeans` /
`KmeansNormVec`）的守卫统一加上 `k <= 0`。

顺带修 `_select_init_center`：它是 **public**（头里 `//private:` 被注释掉了），
里面 `rand() % (nPts - i)` 在 `k > nPts` 时 **i 取到 nPts 就除以 0**。
内部两个入口调它之前已经判过，但外部可以直接调，所以自己也要判。

### 实测结果

| | 修复前 | 修复后 |
|---|---|---|
| `k=0`（四个入口） | **ASan heap-buffer-overflow** | 返回 false，无报告 |
| `k=-1/-2`（四个入口） | 未捕获异常 → terminate | 返回 false，无报告 |
| `_select_init_center(k=nPts+1)` | 除以 0 | 返回 false |
| `_select_init_center(k=0)` | — | 返回 false |
| 正常路径 k=1/2/3 | — | PASS（每点都被分到最近中心） |
| 其它入参守卫（nPts/dim/k>nPts/3 个 NULL） | — | PASS |
| LeakSanitizer | — | 无报告 |

### 可达性（说清楚，不夸大）

本仓库里 `ZQ_Kmeans` 的**唯一**使用者是 `ZQ_LazySnapping.h` 的 4 处调用，
而它自己用 `if (ifore_k > 0)` / `if (iback_k > 0)` 包着，**不会传 k==0**；
`ZQ_LazySnapping.h` 又全仓零 includers。所以这条 bug 在**本仓库内不可达**，
属于「库头里的潜伏缺陷」。

修它的理由不是「本仓库有洞」，而是：现在它**可验证**了，改一行守卫就能让
一个潜在的堆越界写变成干净的 `return false`，成本几乎为零。

### 验证
- Windows Release 全量构建 0 error，`SampleGEMMAsmCompare` PASS（1.9e-06）
- Linux 全量构建 0 error，sample 回归 8 个全 rc=0
- `tools/zq_kmeans_check.cpp`：ASan + LSan 全程无报告，RESULT: PASS
- `tools/check_line_endings.py` / `tools/check_text_encoding.py` 均 OK

## 新增/变更：ZQ_QuickSort 的 FindKthMax(vals, idx, ...) 静默破坏调用方数组

### 变更文件
- `3rdparty/include/ZQlib/ZQ_QuickSort.h`：`_findKthMax(T*, int*, ...)` 出口一行
- `tools/zq_quicksort_check.cpp`（新增）：`ZQ_QuickSort` 的独立回归测试
- `audit_k3_20261001.md`：新增**附录 Z**

### 问题

`ZQ_QuickSort.h:247`（`_findKthMax` 带 idx 的那个重载）：

```cpp
vals[i] = tmp_val;
vals[i] = tmp_idx;      // <-- 应该是 idx[i] = tmp_idx;
```

复制粘贴时漏改了左值。同文件里的 `_quickSort(vals, idx, ...)` 写法是对的
（`vals[i] = tmp_val; idx[i] = tmp_idx;`），所以这是一处孤立的错。

后果两条：
1. `vals[i]` 刚写进去的枢轴值被「下标」覆盖 —— 调用方拿到的数组**不再是原数组
   的一个排列**，元素被凭空替换掉了；
2. `idx[i]` 从头到尾没被写过 —— 下标数组是错的。

**为什么返回值一直是对的**：`output`/`out_idx` 取自局部变量 `tmp_val`/`tmp_idx`，
而唯一被写坏的槽位 `i` 恰好**不在**两边递归区间里（`[start, i-1]` 与
`[i+1, end]`）。所以只测返回值的测试永远发现不了 —— 必须测**数组状态**。

### 实测（tools/zq_quicksort_check.cpp，ASan + LSan）

新增的断言「就地重排后 vals 仍是原数组的一个排列」在 n=1,2,3,5,8,17,64,1000
**全部失败**；单看返回值（output 是第 k 大、out_idx 指向对应元素）则全部通过 ——
正是上面说的「返回值对、数组坏」。

修完：

| | 修复前 | 修复后 |
|---|---|---|
| 排列性（n=1..1000） | 全部 FAIL | PASS |
| `QuickSort` 两个重载 vs `std::sort` | — | PASS |
| `FindKthMax` 单数组版 output = 第 k 大 | — | PASS |
| `FindKthMax` 带 idx 版 output / out_idx | — | PASS |
| ASan / LeakSanitizer | — | 无报告 |

**测试有牙齿**：用 `git show HEAD:...ZQ_QuickSort.h` 取修复前的头重编同一个测试，
n=1 起就 FAIL；换回修复后的头立刻 PASS。

### 可达性（如实记录）

仓库里 `FindKthMax` 的**全部**调用点（`ZQ_ImageProcessing.h` 6 处、
`ZQ_CameraCalibrationBino.h` 2 处）都走**单数组**重载，那条路走的是
`_findKthMax(vals, start, end, k, output)`，出口只有 `vals[i] = tmp_val;`，
**没有这个 bug**。带 idx 的重载全仓零调用点。

也就是说：仓库内不可达，但它是一个 **public API**（头里没有 `//private:`），
任何用这个库的外部代码调 6 参数版本就会静默拿到一个被改坏的数组。现在它可验证了，
改一个字符就能修好。

### 顺带记一笔测试本身踩的两个坑

1. 第一版测试数据用 `(rand>>8) % 100000 / 100.0f`，值必然重复，于是
   「排完之后下标该怎么排」本来就不唯一（`std::sort` 不稳定、快排也不稳定）——
   拿它当期望值是在测一个没有定义的东西，一上来报了假的 FAIL。
   改成先造互异值再打乱。
2. 排列性断言第一版写成 `sort(after) == v`，拿**排序后的结果**去比**没排序的
   原数组**，于是修完代码仍然 FAIL。两次都是测试自己的错，不是被测代码的错 ——
   写完先问一句「这个断言在正确实现下会不会通过」。

## 新增/变更：重判「不修」项第 2 批（BitonicSort / BitStream / Huffman / LSQRSolver）

### 变更文件
- `tools/zq_bitonicsort_check.cpp`（新增）：`ZQ_BitonicSort` 的独立回归测试
- `audit_k3_20261001.md`：新增**附录 AA**

### 结论：四个候选里三个干净，一个本机确实测不了

| 头 | 结论 |
|---|---|
| `ZQ_BitonicSort` | **干净**，新增测试全部 PASS |
| `ZQ_BitStream` | 无可达越界；`malloc` 不判空（潜伏，未复现） |
| `ZQ_Huffman` | 看着像洞的那处其实是**写对了的防御代码**；两条遗留未复现 |
| `ZQ_LSQRSolver` | 依赖 taucs，本机无法编译验证 |

### ZQ_BitonicSort 的测试（新增）

`Sort` 两个重载 len = 1,2,4,…,4096 × 升降序与 `std::sort` 逐元素相等；
带 idx 重载的值序列 + 下标序列；非 2 的幂（3,5,6,7,9,10,12,100,1000）必须返回
false；len = -2/-1/0/1 的边界；`Sort_Recursive(n, start_idx)` 只把
`[start, start+n)` 排好。**一条 FAIL 都没有**，ASan + LSan 无报告。

另外手工复核了两处：
- `len` 不是 2 的幂时 `cur_len` 一路折半、每步判 `%2 != 0` 就返回 false，
  所以不会硬跑；实测确认。
- `_merge` 里 `sort_block_size/2`、`merge_block_size/2` 不会除以 0，
  因为外层 `merge_lvl` 从 `sort_lvl` 递减到 1，恒 `>= 1`。

### ZQ_Huffman：我一开始怀疑的那处其实是对的

`ImportFromBitStream` 从**输入位流**读 `index` 直接当 `code[]` 的下标，
而 `code` 是固定 256 个指针 —— 看着像「不可信输入驱动下标」的经典洞。

复核下来不是：`index` 声明成 `unsigned char`，必然 0..255，正好落在 `code[256]` 内。
条目数那儿的 `if (N > 256) return false;` 看起来多余（n 是 unsigned char），
但配合 `N = bv ? n+256 : n` 把整个区间卡死了。**这是写得对的防御代码。**

遗留两条，都需要构造输入/卡内存才能复现，本轮**未取得端到端证据，不改**：
1. `malloc` 不判空 → OOM 时 `memset(NULL, ...)`；
2. 同一个 `index` 在一条流里出现两次时，第一次的 `malloc` 被覆盖而泄漏。

### ZQ_LSQRSolver
include 了 `ZQ_TaucsBase.h`，而它在附录 X 的探测里属于 BROKEN（缺 taucs 本体）。
**要装齐 taucs 才能测**，本轮不做。

### 注意事项
负结果同样写进报告：不然下一轮会把 `ZQ_BitonicSort` 再翻一遍，
或者更糟 —— 默认「没人报过 = 没人看过」。

验证：Windows Release 0 error；Linux 0 error；
quicksort / bitonicsort / kmeans / mergesort 四个独立测试均 PASS；
两套检查工具均 OK。

## 新增/变更：第 3 批 —— 3x3 中值滤波两条真缺陷（活的调用方）+ 2 个公开 API 缺陷 + 1 处编译错误

### 变更文件
- `3rdparty/include/ZQlib/ZQ_ImageProcessing.h`
  - `Sort_decend_3elements`：中间那步改成换 1/2（原来照抄第一步换 0/1）
  - `MedianFilter33_1channel`：第二列写 `col[1]`（原来写成 `col[0]`），**两份拷贝都改**
    （`ZQLIB_USE_OPENMP` 那条虽然是死代码，但它同样坏，保持一致）
- `3rdparty/include/ZQlib/ZQ_Quaternion.h`：`operator+=` 里 `w += w.z;` → `w += v.w;`
- `3rdparty/include/ZQlib/ZQ_MinIndependentSets.h`：`if (ou == 0)` → `if (out == 0)`
- `tools/zqlib_msvc_shim.h`（新增）：MSVC→gcc 兼容垫片，多个测试共用
- `tools/zq_imageprocessing_check.cpp`（新增）：中值滤波的独立回归测试
- `audit_k3_20261001.md`：新增**附录 AB**；头部统计换成「截至第十三轮」口径

### 1. `Sort_decend_3elements` 从来没排过第 3 个元素

```cpp
if (values[0] < values[1]) { swap(0,1) }          // 第一步
if (values[1] < values[2]) { swap(0,1) }          // <-- 照抄了第一步！应该是 swap(1,2)
if (values[0] < values[1]) { swap(0,1) }
```

三步都在换 0/1，`values[2]` 从头到尾没参与过任何交换。
实测 `{0,1,2}` 的结果是 `{1,0,2}` —— 既不是降序，`values[0]` 也不是最大值。

### 2. `MedianFilter33_1channel` 的第二列写错了槽位

```cpp
col[0][0] = tmpImg[h*padding_width + 1];    // <-- 应该是 col[1][0]
col[0][1] = ...;
col[0][2] = ...;
Sort_decend_3elements(col[1]);              // 排的是从来没被赋值的 col[1]
```

于是 `col[1]` 在第一次迭代读的是**未初始化的栈内存**，之后是上一轮的残值，
再被 `__min(col[0][0], __min(col[1][0], col[2][0]))` 读走。

**两条叠在一起 = 3×3 中值滤波的结果完全不对**，而它有活的调用方：
`ZQ_FindCorners.h:1562-1563` 各调一次。

### 实测（tools/zq_imageprocessing_check.cpp，ASan + LSan）

修复前：

```
  FAIL: Sort_decend_3elements({0,1,2}) 应得降序, 实得 {1,0,2}
  FAIL: Sort_decend_3elements({0,2,1}) 应得降序, 实得 {2,0,1}
  ... 共 6/9 组失败
  FAIL: MedianFilter33_1channel(3x3)   有 1/9   个像素与暴力中值不符 (首个 (0,1): 期望 100 实得 10)
  FAIL: MedianFilter33_1channel(5x4)   有 1/20  个像素与暴力中值不符 (首个 (0,1): 期望 100 实得 10)
  FAIL: MedianFilter33_1channel(7x6)   有 1/42  个像素与暴力中值不符
  FAIL: MedianFilter33_1channel(16x16) 有 1/256 个像素与暴力中值不符
```

参考实现是「把 9 个像素（含 padding 的边缘复制）排序取中间那个」，不依赖被测代码。

修复后：全部 PASS，ASan + LSan 无报告。值得注意的是修复前**每种尺寸都恰好错 1 个
像素**（左边缘那一列），且错的值正是上一次循环的残值 —— 这正是「col[1] 未赋值」
的特征，也说明只有读多个尺寸才能把它和「算法整体写错」区分开。

### 3. `ZQ_Quaternion::operator+=` 的 `w.z`

```cpp
w += w.z;      // <-- 应该是 w += v.w;
```

`w` 是 `double`，`w.z` 是在 double 上取成员：

```
error: request for member 'z' in ... which is of non-class type 'double'
```

**任何编译器都过不去** —— 与第 1、2 条（`ou`、`radius_distortion`）属同一族
「自己源码就编不过」。

> **更正**：本条最初被描述成「自引用，只是算错」，那是错的 —— 它不是静默算错，
> 是根本编不过。拿修复前的头重编 `tools/zq_quaternion_check.cpp` 会在编译期直接失败，
> 这个测试有牙齿。已修。

### 4. `ZQ_MinIndependentSets.h` 在非 Windows 下编不过

```cpp
out = fopen(name, "w");
if (ou == 0)        // <-- ou 未声明
```

一个字母的笔误，`ZQ_RawSets::Print` 在 gcc 下是硬编译错误。
这正是它落在探测器的 BROKEN 桶里、拿不到独立测试的原因。
修掉之后它进入 OK 列表 —— **可独立编译的头从 83 涨到 84**。

### 验证
- Windows Release 全量构建 0 error；Linux 全量构建 0 error
- Linux sample 回归 8 个全 rc=0
- `tools/run_zqlib_checks.py`：5/5 PASS（新增 zq_imageprocessing）
- `tools/probe_zqlib_headers.py`：OK 列表 83 → **84**
- `check_line_endings.py` / `check_text_encoding.py` 均 OK

### 注意事项
1. `ZQ_ImageProcessing.h` 也需要 MSVC 垫片（用了 `__min`/`__max`），
   垫片抽成了 `tools/zqlib_msvc_shim.h`，各测试在 include 目标头**之前**包含它。
2. `ZQLIB_USE_OPENMP` 全仓从未定义（只有 `#ifdef`，没有 `#define` 也没有 `-D`），
   所以 `MedianFilter33_1channel` 里那份拷贝是死代码 —— 但它同样坏，本次一并修，
   免得将来有人打开 OpenMP 时又踩一遍。
3. 本批的缺陷来自一次并行扫描代理，**每一条都由我逐条回读源码确认过**才动手；
   代理报告里另有 5 条「中等置信度」项（`ZQ_KDTree.h` 的 `k<=0` / `npts==0`、
   `ZQ_WeightedMedian.h` 的 `num<=0`、`ZQ_CubicInterpolation.h` 的无界递归、
   `ZQ_FindLargestSubMatrix.h` 的无符号乘法溢出、`ZQ_Matrix.h` 的自赋值），
   本轮未处理，已记入附录 AB 待下一批。

## 新增/变更：第 4 批 —— KDTree / WeightedMedian / CubicInterpolation / FindLargestSubMatrix 六条

### 变更文件
- `3rdparty/include/ZQlib/ZQ_KDTree.h`
  - 去掉嵌套类上重复的 `template<class T>`（gcc 报 "shadows template parameter"，MSVC 放行）
  - 34 处 `ZQ_KDTree_Node<T>` 改成 `ZQ_KDTree_Node`（嵌套类不再是模板）
  - `BuildKDTree` 守卫 `npts < 0` -> `npts <= 0`
  - 四个搜索入口（BruteForceSearch / AnnSearch / AnnSearchWithInitalRadius /
    AnnFixRadiusSearch）补 `k <= 0`
  - `_recursive_ann_fix_radius_search` 的叶节点循环补 `cur_k >= k` 上限
- `3rdparty/include/ZQlib/ZQ_WeightedMedian.h`：`FindMedian` 补 `num <= 0`
- `3rdparty/include/ZQlib/ZQ_CubicInterpolation.h`：`ZQ_nCubicInterpolate` 补
  `n <= 0` 与 `n > 8` 的上界（原来只有 `n == 1` 终止，且 `1 << ((n-1)*2)` 会溢出）
- `3rdparty/include/ZQlib/ZQ_FindLargestSubMatrix.h`：补 `in_width/in_height == 0`，
  乘积改成 `(size_t)` 相乘
- `tools/zq_batch4_check.cpp`（新增）：六条的独立回归测试
- `audit_k3_20261001.md`：新增**附录 AC**

### 这批是怎么来的

上一批（附录 AB）用一次并行扫描代理扫出了 10 条候选，我只处理了「已确认」的 4 条，
把 5 条「中等置信度」留到了这一批。**这批的每一条我都先回读源码确认，再动手**，
其中两条的实际情况比代理描述的还严重一点。

### 1. `ZQ_KDTree.h` 在 gcc 下根本编不过（先修这个才谈得上测）

```cpp
template<class T>
class ZQ_KDTree {
    template<class T>          // <-- 嵌套类里重复声明与外层同名的模板参数
    class ZQ_KDTree_Node { ... };
```
gcc 报 `declaration of template parameter 'T' shadows template parameter`，
MSVC 放行。所以它在探测器的 BROKEN 桶里 —— **这正是它一直拿不到独立测试的原因**。
去掉重复声明后，34 处 `ZQ_KDTree_Node<T>` 要同步改成 `ZQ_KDTree_Node`。
修完它才进 OK 列表，测试才编得过。

### 2. `BuildKDTree` 接受 `npts == 0`

守卫只有 `npts < 0`。`npts == 0` 通过后 `pts_idx = new int[0]`，
而 `_find_min_max`（:114）无条件读 `pts[pts_idx[0]][d]` —— **零长数组越界读**。

### 3. 四个搜索入口只判 `tree->npts < k`，没判 `k <= 0`

与 `ZQ_Kmeans` 完全同一形状。`k == 0` 时 `_update_search_result` 走 `cur_k == k`
分支读 `out_dis2[k-1]` = `out_dis2[-1]`。
**ASan 实测**（修复前）：`stack-buffer-overflow READ`，`ZQ_KDTree.h:383`。

### 4. `_recursive_ann_fix_radius_search` 写穿调用方缓冲（这条最严重）

叶节点循环里

```cpp
out_idx[cur_k] = cur_idx;
out_dis2[cur_k] = cur_dis2;
cur_k++;
```

**没有任何 `cur_k < k` 检查** —— 半径内的点数超过 `k` 时直接写穿。
这条**与 `k` 的取值无关**：用 12 个点、`k = 3`、半径覆盖全部 12 个点就能触发，
不需要任何非法入参。兄弟函数 `_recursive_ann_search` /
`_recursive_ann_search_with_initial_radius` 都以 `k` 为容量上限，这里漏了。

### 5. `ZQ_WeightedMedian::FindMedian` 的 `num` 从不判

`num == 0` 时两个循环都不进、`inf_num` 保持 0，
`output = sort_vals[num - inf_num]` = `sort_vals[0]` **读零长数组**；
`num < 0` 时 `new T[-1]` 抛未捕获异常。
**ASan 实测**（修复前）：`heap-buffer-overflow READ`，`ZQ_WeightedMedian.h:46`，
分配点 `:22`。

### 6. `ZQ_nCubicInterpolate` 无界递归

只有 `n == 1` 会终止。`n == 0` 先做一次 `1 << -2`（UB）再递归进 `n = -1, -2, ...`
直到爆栈；`n >= 16` 时 `(n-1)*2 >= 30`，`1 << 30` 起也是 UB。已补 `n <= 0` 与 `n > 8`。

### 7. `FindLargestSubMatrix` 的整数溢出

`new unsigned int[in_height*in_width]` 是 unsigned × unsigned，乘积在 2^32 处回绕
**之后**才拓宽到 size_t —— 65536×65536 会 new 出长度 0 的数组再写 2^32 个元素；
另外 `in_height == 0` 时 `(in_height - 1)*in_width + w` 也回绕成巨大下标。
两处都判掉，乘积改成 `(size_t)` 相乘。

### 实测（tools/zq_batch4_check.cpp，ASan + LSan）

修复前逐条被 ASan 抓到（KDTree 的 `out_dis2[-1]`、WeightedMedian 的
`sort_vals[0]` 两处都是精确到行号的报告）。修复后 **6/6 全 PASS**，无报告。

`tools/run_zqlib_checks.py` 现有 6 个测试全部通过。

### 验证
- Windows Release 全量构建 0 error；Linux 全量构建 0 error
- Linux sample 回归 8 个全 rc=0
- `tools/probe_zqlib_headers.py`：OK 列表 84 -> **85**（KDTree 进 OK）
- `check_line_endings.py` / `check_text_encoding.py` 均 OK

### 一条方法论记录
代理报告把 `ZQ_KDTree.h` 归在「可独立编译」一类（它自己也标注了这个推断是
按 `#include` 图猜的、没能跑探测器）。**实际上它连编译都过不去** —— 又一次
「先自己回读、再动手」拦下了一个会写进报告的错误结论。

## 新增/变更：第 5 批 —— Laplacian 边界越界（已修）+ 一个据实记录的未决疑点

### 变更文件
- `3rdparty/include/ZQlib/ZQ_ImageProcessing.h`：`Laplacian` 两趟的边界特判
  各自套上 `if (width >= 2)` / `if (height >= 2)`（**只改活的两处**）
- `tools/zq_imageprocessing_check.cpp`：加 Laplacian 的越界回归用例
- `audit_k3_20261001.md`：新增**附录 AD**

### 已修：四条边界特判没有维度守卫

左边界读 `src[i*width+1]`、右边界读 `src[i*width+width-2]`、上边界读 `src[1*width+j]`、
下边界读 `src[(height-2)*width+j]` —— 通用路径 `ImageFilter2D` 每个 tap 都过
`EnforceRange`，这四条没有。`width==1` 时右边界变成 `i-1`，`i==0` 就是 **-1**。

**ASan 实测**（修复前）：`heap-buffer-overflow READ`，`ZQ_ImageProcessing.h:1023`，
"0 bytes to the left of" 缓冲区。

修复后 ASan 不再报，且输出缓冲里没有未被写过的元素（哨兵值验证）。

### 据实记录：输出语义本轮没查清

写参考实现时发现 8x6 的 `dst[0]` 实得 2，而按「可分离两趟、两趟都读 pSrcImage」
推出来应该是 402。已排除越界残留、部分元素未写、`#ifdef` 漏编译三种可能。
**因此本轮刻意没有写像素值断言**，只断言「不崩」+「每个元素都被写过」——
把一个对不上的参考值写成断言只会得到永远失败的测试。

**`Laplacian` 的输出语义目前是未验证状态**；全仓零调用点，本仓库没有 ground truth
可对照。该函数本来就是死代码性质，但没有证据就不下结论。

### 只改活的两处

`ZQLIB_USE_OPENMP` 全仓从未定义，Laplacian 里有 4 份几乎一样的代码，
本轮**只改了非 OpenMP 的两处**，死代码留在原样并在注释里说明。
理由见附录 AD.3：跨行改写脚本在这种「多份拷贝、只有一份是活的」代码上
连续两次改错位置（一次只包住两条边界特判里的第一条，一次因两份的 `for (int c...)`
写法不同只匹配到一处），最后逐个 Edit 手改活代码才收敛。

### 验证
- Windows Release 全量构建 0 error；Linux 全量构建 0 error
- Linux sample 回归 8 个全 rc=0
- `tools/run_zqlib_checks.py`：6/6 PASS
- `check_line_endings.py` / `check_text_encoding.py` 均 OK

## 新增/变更：第 6 批 —— ZQ_Matrix 自赋值、ScanLinePolygonFill「裁了但不用」

### 变更文件
- `3rdparty/include/ZQlib/ZQ_Matrix.h`：`operator=` 加自赋值保护
- `3rdparty/include/ZQlib/ZQ_ScanLinePolygonFill.h`：`ScanLinePolygonFillWithClip`
  里三处 `polygon_pts` 改成 `out_poly`
- `tools/zq_batch6_check.cpp`（新增）：两条的独立回归测试
- `audit_k3_20261001.md`：新增**附录 AE**

### 1. `ZQ_Matrix::operator=` 缺自赋值保护

```cpp
if (data) free(data);              // 先释放自己的
data = (T*)malloc(...);
memcpy(data, other.data, ...);     // a = a 时 other.data 就是刚 free 掉的那块
```

实测：自赋值之后 3×2 的 6 个元素**全部**变成垃圾值。已修
（开头加 `if (this == &other) return;`）。拷贝构造没有这个问题 ——
它读 other 时还不拥有 data，不需要额外处理。

### 2. `ScanLinePolygonFillWithClip` 把裁剪结果丢掉了

只有「out_poly.size() < 3 就返回」那一步用了裁剪结果，**后面三处
（取 minmax / 建边表 / 扫描填充）全用原始的 polygon_pts** ——
这个「WithClip」根本没有做任何裁剪。

实测：顶点 `(-5,4) / (12,4) / (4,-5)` 的三角形在 8×8 的图上

| | 返回像素数 | 落在 8×8 之外的 |
|---|---|---|
| 修复前 | 72 | **40** |
| 修复后 | 28 | **0** |

唯一调用方 `FillOneStrokeWithClip` 传 width/height 进来本意就是想限制笔画范围，
修复前这个约定不成立。已修。

### 可达性

两条的头全仓零 includers，属「外部使用者会踩到」的形态，不是本仓库运行时的洞。

这两条补完了并行扫描代理 15 条候选里最后的两条**已确认**项。剩下那条
`ZQ_MinIndependentSets.h` 析构不判空属调用方契约（同文件里的兄弟析构反而判了，
是不一致而非漏洞），不改。

至此，探测器 OK 列表里那 85 个能独立编译的 ZQlib 头已被系统性过了一遍。

### 验证
- Windows Release 全量构建 0 error；Linux 全量构建 0 error
- Linux sample 回归 8 个全 rc=0
- `tools/run_zqlib_checks.py`：7/7 PASS（新增 zq_batch6）
- `check_line_endings.py` / `check_text_encoding.py` 均 OK

## 新增/变更：把 BROKEN 桶分类，顺手修掉两类「自己就编不过」的 ZQlib 头

### 变更文件
- `3rdparty/include/ZQlib/ZQ_CameraProjection.h`：`undistort_points` 里
  `radius_distortion` → `radial_distortion`（少了一个 `ad`）
- `3rdparty/include/ZQlib/ZQ_SparseMatrix.h`：3 处依赖类型补 `typename`
- `audit_k3_20261001.md`：新增**附录 AF**

### 先分类：50 个 BROKEN 不是同一种东西

| 缺什么 | 头数 | 能不能救 |
|---|---|---|
| `ZQ_TaucsBase.h` 需要 taucs 本体 | 20 | 要先装 taucs，本轮不做 |
| `ZQ_WinSockBase.h` 需要 `winsock2.h` | 7 | Windows-only，本机测不了 |
| `ZQ_ImageIO.h` 需要 `opencv2/` | 4 | 本机有 OpenCV，但要加 include 路径与链接 |
| **自己源码里就有错** | ~15 | 能救，而且就是真缺陷 |
| 其余（依赖链更深） | ~4 | 先不动 |

### 1. ZQ_CameraProjection::undistort_points 的 radius_distortion

    double radial_distortion = 1.0 + k*radius_2;              // 声明的是 radial_
    x_out[i * 2 + 0] = x_in[i * 2 + 0] / radius_distortion;   // 用的是 radius_

`error: 'radius_distortion' was not declared in this scope; did you mean
'radial_distortion'?` —— **任何编译器都过不去**，MSVC 也一样。说明这个函数从来没
被编译过（整条路径是死的，因为它零 includers）。已修。

连带效果：5 个头依赖它，OK 列表 85 -> 87。

### 2. ZQ_SparseMatrix.h 的依赖类型缺 typename

    std::vector<SparseMatrixElement>::const_iterator rit;   // 3 处都缺

`SparseMatrixElement` 是当前模板的嵌套类型，所以这是依赖类型，模板两阶段名字查找
要求 `typename`。MSVC 放行、gcc 报 `need 'typename' ... dependent scope`。已修。

连带效果：OK 列表 87 -> 88。

### 这一族的四个实例

| 位置 | 笔误 | 表现 |
|---|---|---|
| ZQ_MinIndependentSets.h:207 | `ou` → `out` | 未声明标识符 |
| ZQ_KDTree.h:15 | 嵌套类重复 `template<class T>` | shadows template parameter |
| ZQ_CameraProjection.h:635 | `radius_distortion` → `radial_distortion` | 未声明标识符 |
| ZQ_SparseMatrix.h:277/314/337 | 缺 `typename` | dependent scope |

共同点：**都是编译期就能发现的死代码** —— 编不过所以永远没人调，于是也永远没人
发现。MSVC 放行了其中两个，说明上游多半在 MSVC 上写的，而这几个头从没在 MSVC 上
被编译过（零 includers，MSVC 也编不到）。

检测成本几乎为零：probe_zqlib_headers.py 跑一遍就全出来了。

### 验证
- Windows Release 全量构建 0 error；Linux 全量构建 0 error
- `tools/probe_zqlib_headers.py`：OK 列表 85 -> **88**
- `tools/run_zqlib_checks.py`：7/7 PASS
- `check_line_endings.py` / `check_text_encoding.py` 均 OK

## 新增/变更：修 5 处「自己源码就编不过」的 ZQlib 头 —— 可验证的头 88 -> 103

### 变更文件
- `3rdparty/include/ZQlib/ZQ_TaucsBase.h`：第 47 行 `std::map<int,T>::const_iterator`
  补 `typename`
- `3rdparty/include/ZQlib/ZQ_LazySnapping.h`：补 `#include <ctime>`
- `3rdparty/include/ZQlib/ZQ_CompressedImageRaw.h`：4 处 `ZQ_Wavelet<T>::PaddingMode`
  补 `typename`
- `3rdparty/include/ZQlib/ZQ_MGMRESSolver.h`：补 `#include <iostream>`
- `3rdparty/include/ZQlib/ZQ_StereoMatching.h`：补 `#include <climits>`
- `audit_k3_20261001.md`：新增**附录 AG**

### 五处

1. `ZQ_TaucsBase.h:47` `std::map<int,T>::const_iterator rit;` 缺 `typename` ——
   T 是模板参数，`std::map<int,T>` 是依赖类型，两阶段查找要求写 typename。
   MSVC 放行、gcc 报 `need 'typename' ... dependent scope`。**这一个修好解锁 13 个头**。
2. `ZQ_LazySnapping.h` 用了 clock()/clock_t 却没 include <ctime>
3. `ZQ_CompressedImageRaw.h` 的 `ZQ_Wavelet<T>::PaddingMode` 是依赖类型，
   4 处（:76/:169/:312/:427）都缺 typename
4. `ZQ_MGMRESSolver.h` 用了 cerr 却没 include <iostream>
5. `ZQ_StereoMatching.h` 用了 INT_MAX 却没 include <climits>

全部是「编译期就能发现」的问题：头编不过 → 永远没人 include → 永远没人发现。
MSVC 对其中三类（缺 typename、模板参数遮蔽）都放行，说明上游多半在 MSVC 上写的，
而这些头在 MSVC 上也从没被编译过（零 includers）。

### 效果

| | 附录 AF 前 | 现在 |
|---|---|---|
| **OK（能独立编译 ⇒ 能验证）** | 81 | **103** |
| NEEDS_LIB | 6 | 6 |
| MSVC_ONLY | 1 | 1 |
| BROKEN | 55 | **33** |

可验证覆盖率从 57% 提到 72%。

### 剩下 33 个 BROKEN 缺什么

- `GL/glew.h`（GLSLShader）
- `ZQ_ImageIO.h:8` 的 `opencv2\opencv.hpp` —— 注意是**反斜杠**路径（Windows 风格），
  Linux 下 `\o` 会被当转义序列。本机有 OpenCV，给足 include 路径也许能过
- `winsock2.h`（7 个 WinSock 系，Windows-only）
- `ZQ_Calibration.h:1553` 调 `ZQ_Rodrigues_R2r_fun`，实际成员叫 `ZQ_Rodrigues_R2r`
  —— 本轮没改（要确认调用点的 T 怎么传），记为待办

### 元观察

前十八轮审计主要花在读业务逻辑上，而这一族（编译期就能发现的死代码）验证成本
几乎为零：一条 `python tools/probe_zqlib_headers.py` 就把 143 个头的可编译性摆成
一张表，然后「自己源码有错」的那十几个一个一个冒出来，产出密度高得多。

教训可推广：**先花五分钟确认「这个东西到底能不能被自动检查」，再决定要不要投入
人工精读。**

### 验证

Windows Release 0 error；Linux 0 error；sample 回归 8 个全 rc=0；
probe OK 列表 88 -> 103；run_zqlib_checks.py 7/7 PASS；两套检查工具均 OK。

## 新增/变更：打开 OpenCV 那条线之后又冒出 5 处真缺陷（可验证头 103 -> 106）

### 变更文件
- `3rdparty/include/ZQlib/ZQ_ImageIO.h`
  - `#include "opencv2\opencv.hpp"` 的反斜杠改正斜杠
  - `cv::Mat` 返回值函数里的 `return 0;` -> `return cv::Mat();`
- `3rdparty/include/ZQlib/ZQ_MGMRESSolver.h`
  - `cerr` / `cout` 8 处补 `std::` 限定
  - 补 `#include <iostream>`
  - `r8vec_uniform_01` 里补回丢失的局部声明 `long k;`
  - `r[i] = ...` 改成 `out[i] = ...`
- `3rdparty/include/ZQlib/ZQ_LazySnappingGUI.h`：`opencv\cv.h` -> `opencv2/opencv.h`
- `audit_k3_20261001.md`：新增**附录 AH**

### 为什么单独跑一遍「带 OpenCV 头」的对照

附录 AG 之后还剩 33 个 BROKEN，其中 4 个报的是
`fatal error: opencv2\opencv.hpp: No such file or directory` —— 分类器把它们归到
BROKEN，但其实只是缺 OpenCV，而本机明明有 `/usr/local/include/opencv2/opencv.hpp`。

单独跑 `-I/usr/local/include` 的对照后发现 **5 处真缺陷**：

1. `ZQ_ImageIO.h:8` 反斜杠 —— gcc 在 `\o` 处把它当转义序列起始，报「No such file」
   看起来像缺文件，其实是路径分隔符问题。正斜杠在两边都能用。
2. `ZQ_ImageIO.h:339` `cv::Mat` 返回值函数里写 `return 0;` ——
   `could not convert '0' from 'int' to 'cv::Mat'`。改成 `return cv::Mat();`。
3. `ZQ_MGMRESSolver.h` 的 `cerr` / `cout` 在全局命名空间裸用，8 处要补 `std::`。
4. `ZQ_MGMRESSolver.h:1485` `k = seed / 127773;` 的 `k` **没有声明** ——
   从 Burkardt 的 `r8vec_uniform_01` 移植成 C++ 模板时把 `long int k;` 丢了。
5. `ZQ_MGMRESSolver.h:1498` `r[i] = ...` 的 `r` **没有声明** —— 同一处移植把输出
   参数从 `r` 改名成 `out`，这一行漏改。

4 和 5 是同一个函数里连着的两处移植遗漏，说明这段从 C 翻成 C++ 时没有逐行对过；
而正因为编不过，**它从来没被编译过**。

### 顺带：ZQ_LazySnappingGUI.h 引用的是 OpenCV 1.x

`<opencv\cv.h>` 是 OpenCV 1.x 的 C API 头，2/3/4 都没有（对应 `opencv2/opencv.h`），
文件里还有 5 处 `CvMemStorage` / `CvMat` / `cvLoadImage` 在新版里不存在。
**这个头不是路径问题，是整个基于一个已消失的 API。** 把路径改成
`opencv2/opencv.h` 只是让错误信息指向真正的原因（符号不存在）而不是误导人的
「No such file or directory」。本轮不修（要修等于重写整个 GUI 层），记为待办。

### 效果

| | 附录 AG 后 | 现在 |
|---|---|---|
| **OK（能独立编译 => 能验证）** | 103 | **106** |
| NEEDS_LIB | 6 | 8 |
| MSVC_ONLY | 1 | 1 |
| BROKEN | 33 | **28** |

3 个头从 BROKEN 挪到 NEEDS_LIB（现在能正确报告「需要 OpenCV」而不是含混的
「文件不存在」），另外 3 个从 BROKEN 变成 OK。

### 方法论

**探测器的分类结论本身也要复核。** 它把 4 个头报成 BROKEN，理由是
「找不到 opencv2\opencv.hpp」；但那条错误信息里有个**反斜杠**，说明它根本不是
「缺文件」。如果直接采信，这 4 个头会一直留在 BROKEN 桶里，而它们后面还藏着
第 2、3、4、5 条真缺陷。

**自动分类给出的错误信息要读一遍再采信** —— 尤其当错误信息本身就长得可疑。

### 验证

Windows Release 0 error；Linux 0 error；sample 回归 8 个全 rc=0；
probe OK 列表 103 -> 106；run_zqlib_checks.py 7/7 PASS；两套检查工具均 OK。

## 新增/变更：清掉三簇「参数签名改过、调用点没跟着改」（可验证头 106 -> 114）

### 变更文件
- `ZQ_GEMM/../3rdparty/include/ZQlib/ZQ_CameraCalibration.h`：`radius_distortion`
  的第二个副本改正（连带 5 个头）
- `3rdparty/include/ZQlib/ZQ_CameraPoseEstimation.h`：补 `ZQ_DoubleImage.h` 与
  `<vector>`（连带 4 个头）
- `3rdparty/include/ZQlib/ZQ_PoissonSolver.h`
  - `void SolveOpenPoisson` 里的 `return 0;` -> `return;`
  - 6 处多余的 `datatype` 实参去掉（连带 3 个头）
- `audit_k3_20261001.md`：新增**附录 AI**

### 三簇

1. **`ZQ_CameraCalibration.h:785-786`** 是 `radius_distortion` 笔误的**第二份副本**
   （附录 AF 修的是 `ZQ_CameraProjection.h:635`）。两个头各自维护了一份几乎相同的
   `undistort_points` —— 这份代码是复制粘贴出来的。

   这一条本身值得记：我第一次只按文件名找到 CameraProjection 修掉就以为完事了，
   是后来跑簇扫描才发现 CameraCalibration 里还有一份。

2. **`ZQ_CameraPoseEstimation.h`** 用了 `ZQ_DImage<T>` 和 `std::vector`，
   而这个文件**一个都没 include**。

3. **`ZQ_PoissonSolver.h` 的 `datatype` 是一个已不存在的变量** ——
   `RegularGridtoMAC` 等函数的签名后来多了一个 `bool use_period_coord`，
   调用点还在传 `datatype`，于是每个调用点同时报「未声明」和「实参个数不对」。
   顺带修同文件 `:451` 的 `return 0;`（那个函数返回 void）。

### 效果

| | 附录 AH 后 | 现在 |
|---|---|---|
| **OK（能独立编译 => 能验证）** | 106 | **114** |
| NEEDS_LIB | 8 | 8 |
| MSVC_ONLY | 1 | 1 |
| BROKEN | 28 | **20** |

从附录 AF 开始算：**81 -> 114**，可验证覆盖率从 **57% 提到 80%**。

### 怎么搜的教训

`radius_distortion` 在两个文件里各有一份，只按「我知道的文件」逐个看就会漏。

**审计同类缺陷必须按「错误特征」全局搜，不能按「文件」逐个看** —— 复制粘贴出来的
代码会带着同一个错误散落在多个文件里，而每个文件内部看都是自洽的。
这与 AGENTS.md 行尾一节第 6 条（不要凭上一轮清单判断覆盖面，要全仓枚举同类站点）
是同一条纪律的两个面。

### 剩下 20 个 BROKEN

- 7 个 ZQ_WinSock* 要 winsock2.h（Windows-only）
- 1 个 GLSLShader 要 GL/glew.h
- `ZQ_Calibration.h` 的 `ZQ_Rodrigues_R2r_fun`（实际成员是 `ZQ_Rodrigues_R2r`，
  且是模板静态成员）—— 本轮仍未改，记为待办
- 单头无连带的一批：BlendTwoImages3D 缺 <ctime>、GridDeformation3D 的 ZQ_DImage3D
  未声明、ShapeDeformation 缺 typename、StructureFromTexture 缺模板实参、
  RBFKernel/SplinePCHIP 缺 <cmath> —— 下一轮批量清

### 验证

Windows Release 0 error；Linux 0 error；sample 回归全 rc=0；
probe OK 列表 106 -> 114；run_zqlib_checks.py 7/7 PASS；两套检查工具均 OK。

## 新增/变更：清掉最后一批单头缺陷（可验证头 114 -> 118）+ 一条决定不修的 API 漂移

### 变更文件
- `3rdparty/include/ZQlib/ZQ_BlendTwoImages3D.h`：补 `#include <ctime>`（用了 clock()）
- `3rdparty/include/ZQlib/ZQ_GridDeformation3D.h`：补 `ZQ_DoubleImage3D.h`
- `3rdparty/include/ZQlib/ZQ_ShapeDeformation.h`：5 处 `std::map<int,T>::iterator`
  补 `typename`
- `3rdparty/include/ZQlib/ZQ_StructureFromTexture.h`
  - 3 处 `ZQ_SparseMatrix` 补模板实参 `<float>`
  - 补 `#include "ZQ_TaucsBase.h"`
  - 21 处 `TaucsBase::` 补命名空间限定 -> `ZQ_TaucsBase::`
- `3rdparty/include/ZQlib/ZQ_Calibration.h`：一处 `ZQ_Rodrigues_R2r_fun` ->
  `ZQ_Rodrigues_R2r<T>`（**只改这一处**，理由见下）
- `audit_k3_20261001.md`：新增**附录 AJ**

### 决定不修：ZQ_Calibration.h 引用了四个不存在的 ZQ_Rodrigues 成员

ZQ_Rodrigues.h 里实际只有：

    static void ZQ_Rodrigues_r2R(const T* r, T* R, T* dRdr = 0);   // 返回 void
    static bool ZQ_Rodrigues_R2r(const T* R, T* r, T* drdR = 0);   // 返回 bool

而 ZQ_Calibration.h 调的是：

    ZQ_Rodrigues_r2R_fun(...)     不存在（8 处）
    ZQ_Rodrigues_R2r_fun(...)     不存在（9 处）
    ZQ_Rodrigues_r2R_jac(...)     不存在（7 处）—— 要的是**雅可比**，完全没有对应物
    ZQ_Rodrigues_autoscale(...)   不存在（4 处）

只把最机械的一处 R2r_fun 改名改了。其余刻意不改：
1. `r2R` 返回 void，而调用点写的是 `if(!..._r2R_fun(...))` —— 返回类型就不对，
   改名仍然编不过；
2. `r2R_jac`（雅可比）和 `autoscale` 根本没有对应实现，补它们等于**重写
   Rodrigues 参数化的数值微分**，是新功能不是修 bug；
3. 没有可对照的 ground truth（零调用点，也没有原始实现可查）。

**所以 ZQ_Calibration.h 保持 BROKEN，记为已知待办：它需要重写一批数值函数，
不是三行改动。不要把它当成「一个拼写错误」来处理。**

### 效果

| | 附录 AI 后 | 现在 |
|---|---|---|
| **OK（能独立编译 => 可验证）** | 114 | **118** |
| NEEDS_LIB | 8 | 8 |
| MSVC_ONLY | 1 | 1 |
| BROKEN | 20 | **16** |

从附录 AF 开始算：**81 -> 118**，可验证覆盖率 **57% -> 82%**。

### 剩下 16 个 BROKEN 已经全是「真的要外部依赖或要重写」

- 7 个 ZQ_WinSock* 要 winsock2.h（Windows-only）
- 1 个 GLSLShader 要 GL/glew.h
- 1 个 ZQ_Calibration.h（AJ.2 的 Rodrigues API 漂移）
- 1 个 ZQ_LazySnappingGUI.h：`opencv\cv.h` 是 OpenCV 1.x 的 C API，文件里还有
  CvMemStorage / CvMat / cvLoadImage 在新版里不存在；路径已改成 opencv2/opencv.h
  让报错指向真正原因，但要修等于重写整个 GUI 层
- 其余是 taucs / 更深的依赖链

**能靠「补一个 include / 加一个 typename / 改一个名字」救回来的已经救完了。**
剩下的每一处都需要装依赖或重写代码。

### 验证

Windows Release 0 error；Linux 0 error；sample 回归全 rc=0；
probe OK 列表 114 -> 118；run_zqlib_checks.py 7/7 PASS；两套检查工具均 OK。

## 新增/变更：给 ZQ_Quaternion / ZQ_RBFKernel 补测试，并更正一处描述

### 变更文件
- `tools/zq_quaternion_check.cpp`（新增）：ZQ_Quaternion + ZQ_RBFKernel 的独立测试
- `audit_k3_20261001.md`：新增**附录 AK**；更正附录 AB.3 对 `w += w.z` 的描述
- `docs-changelogs/CHANGELOG_2026-10-02.md`：同处更正

### 新增测试的覆盖点

`tools/run_zqlib_checks.py` 现在有 **8 个**测试。

ZQ_Quaternion：
- `operator+` / `operator-` / `operator*` / `Dot` / `Length` 逐个对照手算值
- **`operator+=`**（附录 AB 修的那处）
- `Quat2Rot` -> `Rot2Quat` 往返：6 个用例（4 个绕单轴 + 2 个任意四元数，
  **先归一化**）；断言 `R` 正交（`R*R^T == I`，最大偏差 < 1e-12）与往返点积
  绝对值 == 1

ZQ_RBFKernel：紧支撑核与全局核各几个分支，重点是支撑半径边界与 `flag` 的真实语义。

### 更正：`w += w.z` 不是「算错」，是**编不过**

附录 AB 把它描述成「自引用，只是把 z 加错了地方」。错的：`w` 是 `double`，
`w.z` 是在 double 上取成员，gcc 直接报

    error: request for member 'z' in ... which is of non-class type 'double'

所以它和 `ou`、`radius_distortion` 属同一族「自己源码就编不过」，**不是**会静默
算错的 bug。已在附录 AB 与本文件里更正。

怎么发现的：拿修复前的头（`git show 013ef94^:...ZQ_Quaternion.h`）重编同一个测试，
**在编译期就失败了** —— 这正好也证明了测试有牙齿。

### 写测试时自己踩的三个坑

1. **喂了非单位四元数**给 `Quat2Rot` -> 算出不正交的 R，两条假 FAIL。
   该函数假定输入已归一化，测试要先归一化。
2. **误解了 `flag` 的语义**：`_compact_kernel` / `_global_kernel` 的 `flag` 表示
   「RBF_TYPE 这个分支被识别了」，**不是**「参数在有效范围内」——每个 case 里
   都是无条件 `flag = true`。按后者断言又测出两条假 FAIL。
3. 与附录 Z.5 完全一样：**写完一条断言先问「它在正确实现下会不会通过」**。
   这一次三条里错了两条。

### 顺带记下一个不改的观察

`_global_kernel` / `_compact_kernel` 都没对 `sigma` / `radius` 做下界检查：
`x = fabs(distance/sigma)` 在 sigma==0 时得到 inf 或 NaN，GLOBAL_TPS 于是返回 inf。
测试里把这个行为打印出来但不作为断言。

不崩、不越界，属于「契约型风险」：调用方传了非法的 sigma，得到 inf 会污染下游
计算但不会立即暴露。本轮不改 —— 要改得先确定这个库对非法参数的约定是
「返回 NaN/inf」还是「返回 false」，而没有可对照的 ground truth。

### 验证

`tools/run_zqlib_checks.py` 8/8 PASS。

## 新增/变更：把 ZQlib 可编译性探测变成回归门禁（--check-baseline / --save-baseline）

### 变更文件
- `tools/probe_zqlib_headers.py`：新增 `--save-baseline` / `--check-baseline`
- `tools/zqlib_probe_baseline.txt`（新增）：逐头分类基线（143 行）
- `AGENTS.md` 构建一节新增第 7 条
- `reports/README.md`：复现命令一节补上门禁用法
- `audit_k3_20261001.md`：新增**附录 AL**

### 门禁做什么

    # 任何一个头从 OK 变成非 OK 就退出 1
    python tools/probe_zqlib_headers.py --check-baseline tools/zqlib_probe_baseline.txt
    # 修好或新增头之后更新基线
    python tools/probe_zqlib_headers.py --save-baseline tools/zqlib_probe_baseline.txt

基线当前：OK 118 / NEEDS_LIB 8 / MSVC_ONLY 1 / BROKEN 16。
比对分别报 REGRESSION / IMPROVED / NEW / REMOVED。

往 ZQlib 里加新头、或改现有头改到编不过，这一步会立刻抓到 ——
而附录 AG~AJ 里那二十来条缺陷**全都是这一类**。

### 验证门禁时抓出了探测工具自己的 bug

做法是故意把一个 OK 的头（ZQ_Ray2D.h）改坏再跑门禁。第一次跑出来是：

    NEW — 基线里没有的新头:
       ZQ_Ray2D                                 BROKEN
    REMOVED — 基线里有、现在目录里没有了:
       ZQ_Advection.h
       ...（其余 140 个）

**报的是「新增」而不是「回退」** —— 门禁形同虚设。原因是脚本自己一处参数错位：

    "echo 'R|%s|OK|'; else echo \"R|%s|ERR|...\"; fi" % (stem, stem, h, stem, stem)
                                                          ^^^ ERR 分支打的是 stem（不带 .h）

OK 分支打 `h`（带 .h）、ERR 分支打 `stem`（不带），于是**任何编不过的头在比对时
都对不上基线里的键**，永远只会被当成「新增」。已修（两个分支都打 h）。
修完再跑同一条命令得到 `ZQ_Ray2D.h OK -> BROKEN` + 退出码 1；还原后退出码 0。

> 这一条值得记：`--check-baseline` 是刚写的，按惯例做「故意弄坏再跑一遍」的验证，
> 结果第一个抓到的 bug 在**工具自己**身上而不在被测代码上。
> **新写的检查工具必须先证明自己会失败**，否则只是加了一个永远返回「没问题」的脚本。

## 新增/变更：再补两个 ZQlib 测试（ZQ_Matrix / ZQ_Kahansum），9 个测试全绿

### 变更文件
- `tools/zq_matrix_check.cpp`（新增）
- `audit_k3_20261001.md`：新增**附录 AM**

### ZQ_Matrix：最难被自动工具抓到的一条

附录 AE.1 修的 `operator=` 自赋值 use-after-free：

- **ASan 抓不到**。glibc 的 free 只是把块放回空闲链表，紧接着的 malloc 很可能
  拿回**同一个地址**，于是 memcpy 变成自我拷贝，结果完全正确。实测没有触发任何
  use-after-free 报告。
- 能抓到它的只有一条断言：**「自赋值之后内容必须一字不差」**。

验证有牙齿：用 `git show 35854d8^:...ZQ_Matrix.h`（修复前的头）重编同一个测试 ->

    FAIL: Matrix: a = a 之后内容应与自赋值前完全相同（曾因先 free 后 memcpy 而变垃圾）
    RESULT: FAIL

还原后 PASS。

（第一次挑错了参照提交：取的是 `5deb40e^`，那已经在修复之后，所以「修复前的
代码」也 PASS。用 `git log -- <file>` 确认到底哪一笔改了它。）

其余覆盖：GetData/SetData 的越界标志（-1 / nRow / nCol），越界写之后原有内容不能被
破坏；Transpose 的元素对应关系；MatrixMul 与手算逐元素比对 + 维度不匹配时返回 false。

### ZQ_Kahansum：求 1e6 个 0.1

    naive=100000.0000013329   |naive-exact| = 1.333e-06
    kahan=100000.0000000000   |kahan-exact| = 0.000e+00

补偿求和在这里误差恰好为 0。另外覆盖 n==0 / n==1 / n<0（循环不进入，不越界读）。

### 写测试时自己踩的坑（同族第三次）

第一版加了一条「对照组有效性」断言：「朴素累加 1e6 次 0.1 的相对误差应 > 1e-6，
否则这个用例测不出东西」。实测只有 **1.3e-11** —— 我对 0.1 这个特定值的误差估计
过于悲观，断言本身是错的，测出假 FAIL。

这是附录 Z.5 之后同一族的第三次（前两次分别错在「重复值的下标顺序本就不唯一」
和「flag 的语义被误解」）。三条合起来是同一句：**断言的依据必须来自实测或阅读，
不能来自直觉。**

### 验证

tools/run_zqlib_checks.py 9/9 PASS。

## 新增/变更：把审计这一轮加的检查收成一个入口 tools/run_audit_checks.py

### 变更文件
- `tools/run_audit_checks.py`（新增）
- `AGENTS.md` 构建一节新增第 6 条（后面两条顺延）
- `reports/README.md` 复现命令一节补上统一入口
- `audit_k3_20261001.md`：新增**附录 AN**

### 为什么需要

前面二十五轮陆续加了五样东西：两个文本卫生检查、9 组第三方头库的 ASan 测试、
一个 ZQlib 可编译性门禁。**五个命令要人记住，迟早会有人只跑其中三个** ——
而且不会有人发现漏了哪个。

    python tools/run_audit_checks.py            # 全跑（含门禁，约 2.5 分钟）
    python tools/run_audit_checks.py --quick    # 跳过门禁，约 20 秒

| 组 | 内容 | 耗时 |
|---|---|---|
| A1 | check_line_endings.py —— multi-CR / lone-CR / CRLF+LF 混用 | 秒级 |
| A2 | check_text_encoding.py —— UTF-8 有损解码残留（U+FFFD） | 秒级 |
| B | run_zqlib_checks.py —— 9 组 ZQlib 独立测试（ASan + LSan） | ~20 秒 |
| C | probe_zqlib_headers.py --check-baseline —— 可编译性门禁 | ~2 分钟 |

任何一组失败就整体退出 1。改完东西先跑它，比逐个记命令可靠。

### 为什么是 Python 而不是 .sh

第一版写的是 tools/run_audit_checks.sh，想放进 WSL 里跑。跑不通：

    AttributeError: 'module' object has no attribute 'run'

因为 B、C 两组里的工具本身是「Windows 侧 Python -> wsl ... bash -s 喂脚本 ->
在 WSL 里编译」。放进一个 WSL 里的 shell 脚本去调用，会在 WSL 里再起一个 Python，
然后那个 Python 想调 wsl —— 而 subprocess 是 **Windows 的**标准库，WSL 里没有。

所以编排脚本必须待在 Windows 侧，由它去调各个工具（那些工具再自己去叫 WSL）。
第一版 shell 脚本已删除。

### 踩的两个坑

1. **基线路径解析错**：子进程以仓库根为 cwd，我传相对路径 zqlib_probe_baseline.txt，
   解析成 <仓库根>/zqlib_probe_baseline.txt，而文件在 tools/ 下 -> 报「读不到基线」。
   改成传绝对路径。
2. **输出顺序是乱的**：父进程的 print 走 Python 缓冲，子进程直接写同一个 fd，
   于是所有子进程输出排在父进程所有 print 之前，读日志像是门禁先跑了再去跑 A1。
   加 sys.stdout.flush() 才对。

第二个不是功能问题，但会让日志读起来是错的（看起来像执行顺序不对），排查时非常误导。

### 验证

全跑：4 组全 OK，退出码 0。--quick：3 组 OK + 门禁跳过，退出码 0。

## 新增/变更：把「双平台跑通」也接进统一入口（--with-build）

### 变更文件
- `tools/run_audit_checks.py`：新增 `--with-build`（D 组）
- `AGENTS.md` 构建一节第 6 条补上 --with-build 的说明
- `reports/README.md` 复现命令一节补上
- `audit_k3_20261001.md`：新增**附录 AO**

### D 组内容

| 组 | 内容 |
|---|---|
| D1 | Windows 全量构建（cmake --build build_x64 --config Release） |
| D2 | Linux 全量构建（WSL 里 cd /tmp/zqb2 && make -j8） |
| D3 | Linux sample 回归（tools/run_sample_regression.sh，8 个） |
| D4 | Windows 关键 sample 6 个（SampleGEMMAsmCompare / MTCNN / NCHWC4 / SSD / CascadeOnet / FaceDetectorMTCNN） |

### 修掉一处「看着跑过了其实没跑」

D4 里 Windows 的 sample 必须把 cwd 设成**产物目录**
cmake-out-win32-x64/release/Release，因为 CMake 把 model/ 和 data/ 联接到了那里。
从仓库根跑，sample 只会打一行 `empty image` 然后返回 0。

**看着像跑过了，其实什么都没验。** 2026-10-02 写 capture 脚本时踩过一次，
这次写 D4 又踩一次，所以在代码里留了注释。

### 「Windows 和 Linux 都能完全跑通」现在有了可重复执行的形式

    python tools/run_audit_checks.py --with-build

一次跑完：双平台构建 + 双平台 sample + 文本卫生 + 9 组 ZQlib 测试 + 可编译性门禁。
任何一组失败就退出 1。

### 验证

--with-build --quick：13 个子组全 OK，退出码 0。

## 新增/变更：改 ZQlib 之前本该先算的一张表（可达性）

### 变更文件
- `tools/zqlib_reachability.py`（新增）
- `audit_k3_20261001.md`：新增**附录 AP**

### 问题

这一轮改了 `3rdparty/include/ZQlib/` 下 **26 个头**。改之前有个必须先回答的问题：
**它们真的进过产物吗？**

直觉会说「第三方库、主工程用不到」—— 但**直觉在这里是部分错的**，而且错得最有
欺骗性：直接 grep「谁 include 了它」很容易得出「没人用」（因为 include 链是传递的），
而实际上主工程里有一批 sample 确实用到了我改过的 5 个头。

### 工具

从主工程所有 .h/.cpp/.c（ZQCNN / ZQlibFaceID / SamplesZQ* / model，262 个）出发，
按 C 的 #include "..." 语义走**传递闭包**：

    主工程源码文件: 262 个
    ZQlib 头总数  : 143
    可达          : 16
    不可达        : 127   <- 改了不会进任何产物

### 本轮改过的 26 个头里，可达 5 个

| 头 | 改动 | 影响哪些 sample |
|---|---|---|
| ZQ_MergeSort.h | 补 #include <vector> | **15 个**（人脸库 / LFW 评估那一大片，经 ZQ_FaceDatabaseMaker.h 的 sort_score_file） |
| ZQ_Matrix.h | operator= 自赋值保护 | 4 个 |
| ZQ_QuickSort.h | idx[i] 笔误 | SampleLnet106 |
| ZQ_ImageProcessing.h | 中值滤波两条 + Laplacian 边界 | SampleLnet106 |
| ZQ_Kmeans.h | 补 <math.h> + k<=0 守卫 | 传递可达 |

其余 21 个头完全不可达 —— 改错了 sample 也发现不了，**只能靠 ASan 测试验**
（这就是 tools/run_zqlib_checks.py 存在的原因）。

### 为什么这条有操作意义

1. **风险分级有了依据**：可达的 5 个必须跑双平台 sample 回归；不可达的 21 个只能靠
   独立测试。把两者混为一谈，要么白花时间跑回归，要么漏掉真正会崩的路径。
2. **ZQ_MergeSort.h 那处改动其实是行为中性的**：它只是补 #include <vector>，
   而这些 sample 之前在 Linux 上能编过，说明它们的 includer 恰好先带进了 <vector>
   （正是 AGENTS.md 行尾一节第 6 条那一类「靠运气」）。补上之后就不再依赖那个运气。
3. **中值滤波那条只影响 ZQ_FindCorners.h 的调用点**，而 ZQ_FindCorners.h 本身不在
   可达集里 —— 所以那个「活的调用方」在**产物层面**其实也是死的。附录 AB 的可达性
   说明仍然成立（它是那个头的调用方），但影响面要按可达性表来划，不能只看调用图。

### 结论

改 ZQlib 头之前先跑 `python tools/zqlib_reachability.py`，它会直接告诉你
「你改的东西会不会进产物」：不可达的头 = 只能靠独立测试验；可达的头 = 双平台
sample 回归兜底。

## 新增/变更：可达 ≠ 可跑 —— 附录 AP 的下半集

### 变更文件
- `audit_k3_20261001.md`：新增**附录 AQ**

### 现象

按附录 AP 的可达性表，把那批「链接了我改过的头」的 sample 在两个平台各跑一遍：

    Windows: SampleSwapFace rc=1、SampleCropImagesForArcFace rc=1、
             SampleFaceDatabaseNCNN rc=1、SampleEvaluationOnLFWArcFaceMiniCaffe rc=53 ...
    Linux  : 同样一批 rc=1；另有 8 个 MISSING

第一眼像「我的改动弄坏了 19 个 sample」。

### 查下来全是「缺命令行参数」

    ===== SampleSwapFace =====
    SampleSwapFace.exe img1 img2 out
    ===== SampleCropImagesForArcFace =====
    Use: SampleCropImagesForArcFace.exe src_root dst_root [max_thread_num] ...

这些是 **usage 消息**，不是崩溃。它们要人脸库目录、模型文件、输出路径，仓库里没有。

而且这正是 tools/sweep_samples_win.py 早就知道的：

    if rc not in ('0', '1', 'TIMEOUT'):
        ... '<<< 需要看'

`rc == 1`（usage）与 TIMEOUT 一直被当作可接受，附录 T 的口径就是这条。
所以附录 T 没说错，是我这次的临时脚本用「rc 必须为 0」当判据才误报。

### 结论

附录 AP 说「可达的 5 个头要靠双平台 sample 回归兜底」—— 这句话在本机站不住：

| | 链接了那 5 个头 | 本机能否真正跑起来验 |
|---|---|---|
| sample 数 | ~19 | 基本不能（缺人脸库/模型/输出目录） |
| 可用信号 | 只有 usage 与超时 | 没有 |

于是这 5 个头的验证**只能靠 run_zqlib_checks.py 里那 9 组 ASan 测试**；
也正因为如此「测试必须有牙齿」在这里格外重要 —— 我为 ZQ_Matrix 的自赋值、
ZQ_QuickSort 的 idx[i]、ZQ_Quaternion 的 w.z **逐个用「修复前的头重编同一个测试」
验过**（附录 AM / AK 记了过程）。

**如果当时只是「改了 + 跑一遍 sample 看没崩」，这三条改动会全部静默通过** ——
因为能跑的那些 sample 根本不经过这些代码路径。

## 新增/变更：Windows 侧从来没被验证过的那一半 —— 新增 MSVC 侧探测，查出 3 件事

### 变更文件
- `3rdparty/include/ZQlib/ZQ_Logger.h`：`wcslen(msg)` -> `_tcslen(msg)`（两处）
- `3rdparty/include/ZQlib/ZQ_LazySnapping.h`：修掉我自己的一个乱码字（`眉` -> `不`）
- `3rdparty/include/ZQlib/ZQ_Quaternion.h`：注释与已更正的结论对齐
- `tools/probe_zqlib_headers_msvc.py`（新增）：用 `cl /Zs` 在 Windows 本地逐头语法检查
- `tools/check_text_encoding.py`：新增**第三类检查**（罕见汉字清单，供人工核对）
- `tools/run_audit_checks.py`：新增 `--msvc-probe`
- `audit_k3_20261001.md`：新增**附录 AR**

### 缺口

前面二十几轮，ZQlib 的改动**只在 gcc 下验证过** —— probe_zqlib_headers.py 是
Windows 侧脚本但把编译外包给了 WSL。而这一轮改的 26 个头里有 5 个**真的链接进了
Windows 侧 sample**（附录 AP）。也就是说那些改动在 Windows 上一次都没被编译过。

### 查出来的三件事

**① 一条只在 MSVC 才看得见的真 bug（连带 8 个头）**

    WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), msg, wcslen(msg), NULL, NULL);

C 的 wcslen 是 `wcslen(const wchar_t*, size_t)`，**要两个参数**；MSVC 这里只能
找到 C++ 的 `std::wcslen(const wchar_t*)`，签名对不上：

    error C2664: 'size_t wcslen(const wchar_t *)': no matching overloaded function found

连带 ZQ_Logger.h 自己和 7 个 WinSock 系的头在 MSVC 下都编不过（ZQ_SemaphoreEx.h
也经由 ZQ_Logger.h 中招）。该文件本来就 include 了 `<tchar.h>`，改成 `_tcslen(msg)`
才对（ANSI 构建下 TCHAR=char，_tcslen 自动走 strlen）。两处都改了。

**这条是纯 MSVC 侧问题** —— gcc 那边压根没有 wcslen/windows.h，gcc 探测永远看不到它。
这就是「只在一侧验证」的具体代价。

**② 一个我自己的编码错误，check_text_encoding.py 查不出来**

    // clock()/clock_t 本来眉不能编过          <-- 「眉」应为「不」

**这是合法的 UTF-8，只是有一个字错了**。来源是在 python 里手写 `hex 转义` 的 UTF-8
字节时写错了一位。已修，并给该工具加了第三类检查：把全仓汉字按出现次数排序、
列出罕见字供人眼扫一眼（1130 个不同汉字 -> 221 个低频的）。它是**筛子不是判官** ——
用得对的生僻字也会在里面，作用只是把「不可能人工核对」变成「几秒」。
默认 `--rarity 3`，`--no-rare` 可关掉。

顺带：工具输出中文时必须把 stdout 改成 utf-8（本机控制台 GBK），否则罕见字清单
自己就是乱码 —— 恰恰在最需要人眼核对的时候看不清。

**③ 一条与已更正结论矛盾的注释**

ZQ_Quaternion.h 的注释还写着「自引用，只是把 z 加错了地方」，但附录 AK.2 已更正：
`w.z` 是在 double 上取成员，**根本编不过**。注释已改成与结论一致。

### MSVC 全量结果

    143 个头：OK 129，自身编译错 0
    剩下 14 个全是 fatal error C1083（找不到 opencv2/ jpeglib.h GL/glew.h libav* stdafx.h）

那 14 个是**探测器的局限**：本机只装了 OpenCV 的源码树，没有预编译的 include
目录；主工程里那些路径是有的（SampleLnet106 就直接 include 了 opencv2/opencv.hpp）。
判断「自身编译错」时要把 C1083 排除掉再看。

    gcc  : 118 OK / 25 BROKEN（9 外部依赖 + 16 需重写）
    MSVC : 129 OK / 0 自身编译错 / 14 外部依赖

MSVC 通过的更多，是因为它对「缺 typename」「嵌套模板参数遮蔽」这类错误是放行的 ——
而那正是前面几轮修掉的一批。

### 两条教训

1. **只在一侧验证等于没验证**。wcslen 那条只在 MSVC 出现；反过来，缺 typename
   那一批只在 gcc 出现。两个方向都要走。
2. **「自动检查查不出」要主动想一遍**。U+FFFD / 非法 UTF-8 / 罕见字是三类不同的
   失败模式，工具只能覆盖前两类 —— 第三类必须靠人眼，所以工具的职责是
   **把要人眼看的量收敛到能看完**。

### 验证

`python tools/run_audit_checks.py --quick --msvc-probe`：4 组全 OK，退出码 0。

## 新增/变更：把 9 个 ASan 测试搬到 Windows 上真跑一遍（附录 AS）

### 变更文件
- `tools/run_zqlib_checks_msvc.bat`（新增，**必须纯 ASCII**）
- `tools/run_audit_checks.py`：新增 `--msvc-asan`

### 缺口

附录 AR 补的是**编译**侧（MSVC `cl /Zs`），运行侧仍然是单边的：
`run_zqlib_checks.py` 把编译外包给 WSL，所以那 9 个回归测试本质是 gcc/Linux 的结果。
而附录 AP 算过：本轮改过的 26 个头里有 5 个**真的链接进了 Windows 侧 sample**，
也就是说那些改动在 Windows 上**一次都没被运行过**。sample 又因为缺人脸库跑不起来。

### 实测结果

    cl /nologo /EHsc /std:c++14 /O1 /Zi /utf-8 /fsanitize=address ^
       /I3rdparty\include\ZQlib /Itools tools\%%T_check.cpp

    zq_batch4 PASS / zq_batch6 PASS / zq_bitonicsort PASS / zq_imageprocessing PASS
    zq_kmeans PASS / zq_matrix PASS / zq_mergesort PASS / zq_quaternion PASS
    zq_quicksort PASS
    9 tests, 0 failed

### 三个坑

1. `/utf-8` 必加 —— 不加时 MSVC 按 GBK 读源码，中文注释直接 C2001/C2143
2. `.bat` 本身必须纯 ASCII —— cmd 用 OEM 代码页读批处理，注释里的中文会把解析器搞坏
3. `cd /d "%~dp0.."` 而不是写死 `D:\ZQCNN`；`/Fo` `/Fd` 指到 `%TEMP%`，
   否则 cl 在仓库根留下 `vc140.pdb`（这次清掉了）

### 顺带：换一个 sanitizer

给 `run_zqlib_checks.py` 加 `--ubsan`，同一批 9 个测试用
`-fsanitize=undefined` 再跑一遍（抓 ASan 看不见的有符号溢出/移位越界等）。
结果 **9/9 全干净**。

> **一个必须记的坑**：UBSan 默认只打一行 `runtime error:` 然后**继续跑**，
> `rc` 恒为 0。我最初就是按 `rc == 0` 判通过的 —— 那一栏永远是 0，等于什么都没查。
> 正确做法是数 `runtime error:` 的行数。

---

## 新增/变更：开一条新审计轴 —— gcc `-Wall -Wextra`（附录 AT）

### 问题

前面三十几轮找缺陷靠的是「**编不过**」这一根轴。118 个头都能编过 ——
也就是说，**能编过的那些头里编译器早就看见了问题，只是默认一声不吭**。

### 变更文件

源码：

- `3rdparty/include/ZQlib/ZQ_ConstrainedDelaunayTriangulation.h`（AT.4，死掉的空指针守卫 + 构造后丢掉的异常）
- `3rdparty/include/ZQlib/ZQ_MathBase.h`（AT.5 条件数读错行距 + AT.8 malloc/delete[] 错配）
- `3rdparty/include/ZQlib/ZQ_ObjLoader.h`（AT.8 三处 + AT.9 四处 ignored-qualifiers）
- `3rdparty/include/ZQlib/ZQ_TaucsBase.h`（AT.8 四处）
- `3rdparty/include/ZQlib/ZQ_MinIndependentSets.h`（AT.8 一处）
- `3rdparty/include/ZQlib/ZQ_Huffman.h`（AT.10 `%d` 配 `unsigned long`）
- `3rdparty/include/ZQlib/ZQ_StereoMatching.h`（AT.10 `%I64d` 是 MSVC 专有 + `(int64_t)` 被 sizeof 吃掉；外加一个罕见字 `眉`）
- `3rdparty/include/ZQlib/ZQ_MarchingCube.h`（AT.11 初始化列表顺序）
- `3rdparty/include/ZQlib/ZQ_CPURayCasting.h`（AT.11 braced-init 收窄）
- `3rdparty/include/ZQlib/ZQ_BinaryImageProcessing.h`（AT.11 补括号，语义未变）

工具：

- `tools/warn_sweep_zqlib.py`（新增）：`-Wall -Wextra` 扫 143 个头，HIGH/MED/LOW 分桶
- `tools/check_alloc_delete.py`（新增）：malloc/new 与 delete[]/free 错配扫描，**带内建自测**
- `tools/zq_mathbase_check.cpp`（新增）：第 10 个回归测试
- `tools/zqlib_probe_shim.h`（新增）：转发到 `zqlib_msvc_shim.h`
- `tools/probe_zqlib_headers.py`：改为共用 `zqlib_msvc_shim.h`，不再内联一份
- `tools/zqlib_warn_baseline.txt`（新增）：HIGH 桶基线（**当前为空**）
- `tools/run_audit_checks.py`：新增 A3/A4 组、`--warn-sweep`、`--ubsan`
- `audit_k3_20261001.md`：新增**附录 AS / AT**

### 实测结果

    143 个头 / 4892 行警告：HIGH 43, MED 108, LOW 699
    修完后 HIGH = 0，基线为空

HIGH 桶逐条判定：

| # | 位置 | 判定 |
|---|---|---|
| 1-2 | `ZQ_ConstrainedDelaunayTriangulation.h:1985,2101` `-Waddress` | **真缺陷** |
| 3 | `ZQ_Huffman.h:426` `-Wformat=` | **真缺陷** |
| 4 | `ZQ_StereoMatching.h` 6 处 `-Wformat=` | **真缺陷** |
| 5 | `ZQ_MarchingCube.h:105` `-Wreorder` | 隐患，当前无数值影响 |
| 6 | `ZQ_CPURayCasting.h:282-284` `-Wnarrowing` | 隐患（MSVC /W4 报 C4248） |
| 7 | `ZQ_BinaryImageProcessing.h` 3 处 `-Wparentheses` | **不是缺陷**（优先级本来就对） |

### 最有价值的两条

**① `&ot == NULL` 是一个永远不会触发的守卫**

    Triangle& ot = t->NeighborAcross(p);   // 内部已经 *neighbors_[k] 解引用
    Point&  op = *ot.OppositePoint(*t, p); // 而且这里已经用上了
    if (&ot == NULL) { assert(0); }        // 守卫写在用完之后

`neighbors_` 初始化为 NULL。真为 NULL 的话 `*neighbors_[k]` 早就是 UB、
程序在 `NeighborAcross` 里就崩了，走不到这个 `if`。
修法：拆出不解引用的 `Triangle* NeighborAcrossPtr(Point&)`，两个调用点先判空再解引用。
另外 `EdgeEvent` 里两处 `std::runtime_error("...")` 是**构造一个临时异常然后丢掉**，
加上 `assert(0)` 在 Release 下被编掉 —— 那个「不可能发生」的分支在正式构建里
静悄悄 return 出去，返回一个没被旋转过的 triangle。已改成真 `throw`。

**② `Cond_by_double_svd` 的条件数在「高矩阵」上是垃圾**

`Smat` 是 `sdim x sdim`、行距 `sdim`，但代码写的是 `S[(N-1)*col + (N-1)]`。
`col == N` 时碰巧相等（所以一直没人发现）；`row < col` 时读到了
`memset(sdim*sdim)` 区域之外的**未初始化堆内存**：

    高矩阵 3x4   cond = -1.74492e+07   <- 修前（负数，纯垃圾）
    高矩阵 3x4   cond =  70.2337      <- 修后
    宽矩阵 4x3   cond =  82.257       <- 修前修后一致（本来就对）

in-tree 影响面：`ZQ_CameraCalibration.h:889/1036/1809`、
`ZQ_CameraCalibrationMono.h:125/272/1046` 调 `Cond_by_double_svd(JJ, 2*nPts, 6, ...)`，
`nPts <= 2` 时正好落进高矩阵分支。

### 关于 `malloc` 配 `delete[]`

`Cond_by_double_svd` 的测试在 ASan 下报出

    ERROR: AddressSanitizer: alloc-dealloc-mismatch (malloc vs operator delete [])

于是做了 `tools/check_alloc_delete.py`。**全仓 706 个源文件，命中 5 处**（全在 ZQlib），
已全部修掉。工具带**内建自测**并作为 A3 组常驻 —— 一个「什么都查不出来」的检查工具
比没有这个工具更危险，它会让人以为这块已经审过了。

写自测时抓到工具自己两个 bug（同一类：正则用 `search` 只看一行的第一个匹配，
于是「一行里两次 malloc」时第二个变量不登记）。已改成 `finditer`。

### 注意事项

1. **HIGH 桶也不是判官。** 43 条 HIGH 里 `ZQ_BinaryImageProcessing.h` 那 3 处
   `-Wparentheses` 就是误报：`&&` 优先级本来就高于 `||`，代码是对的。
2. **假绿要专门防。** 一个编不过的头，它的告警文件里全是 `error:` 没有 `warning:`，
   于是它在 HIGH 桶里显示「0 条」—— 看着最干净其实根本没被扫。
   工具因此单独报告「几个头编不过」（本机 9 个）。
3. **基线只记 HIGH。** 一个天天报 3000 条的门禁等于没有门禁。
4. **垫片只有一份。** MSVC→gcc 兼容垫片历史上散落过三份，会悄悄漂移；
   现在 `zqlib_msvc_shim.h` 是唯一真实定义，`zqlib_probe_shim.h` 是转发头。

### 验证

    python tools/run_audit_checks.py --quick --msvc-asan   # 6 组全 OK
    python tools/run_zqlib_checks.py --ubsan               # 9/9 全干净
    python tools/check_alloc_delete.py --selftest          # 3 命中 0 误报
    python tools/check_alloc_delete.py                     # 全仓无命中
    python tools/warn_sweep_zqlib.py --check-baseline tools/zqlib_warn_baseline.txt
    python tools/check_text_encoding.py                    # 623 文件 OK

## 新增/变更：把 `-Wall -Wextra` 这根轴搬到主工程（附录 AU）

### 问题

附录 AT 在 143 个**第三方**头上挖出 5 条真缺陷。主工程 `ZQCNN/` 已经被人工精读
十三轮，边际收益理应更低 —— **但没人算过**，而「没人算过」正是报告开头那条
元发现（「无法验证」是会自我实现的结论）的另一种写法。

### 变更文件

生产代码（**全部零行为变化**）：

- `ZQCNN/ZQ_CNN_Layer.h`：8 个构造函数的初始化列表按**声明顺序**重排；
  `buffer`/`buffer_len` 补 `= 0`
- `ZQCNN/ZQ_CNN_Layer_NCHWC.h`：同上
- `ZQCNN/ZQ_CNN_Net.h`：构造函数重排 + `input_C/H/W` 补 `= 0`
- `ZQCNN/ZQ_CNN_Net_NCHWC.h`：同上
- `ZQCNN/ZQ_CNN_Tensor4D.h` / `ZQ_CNN_Tensor4D_NCHWC.h`：
  19 处无意义的 `const int GetX() const` 去掉返回类型上的 `const`；
  `float* const GetFirstPixelPtr()` 同理
- `ZQCNN/ZQ_CNN_Tensor4D.cpp` / `ZQ_CNN_Tensor4D_NCHWC.cpp`：
  14 处 `&&` 混在 `||` 里补括号（**语义本来就对**，只是消歧义）

工具：

- `tools/warn_sweep_src.py`（新增）：同一套 HIGH/MED/LOW 分桶扫主工程 43 个 TU
- `tools/zqcnn_warn_baseline.txt`（新增）：主工程 HIGH 桶基线（**当前为空**）
- `tools/check_uninit_members.py`（新增）：扫「类成员没在构造函数初始化列表里」，
  **带内建自测**
- `tools/run_audit_checks.py`：新增 A5/A6 组、`--src-sweep`
- `audit_k3_20261001.md`：新增**附录 AU / AV**

### 实测结果

```
43 个 TU / 311 行警告：HIGH 42  MED 72  LOW 197
修完后：              HIGH  0  MED 12  LOW 197  （12 是跨头重复计数，唯一 5 处）
```

| 类别 | 条数 | 判定 |
|---|---|---|
| `-Wreorder` | 26（8 个构造函数） | 隐患，当前无数值影响，已按声明顺序重排 |
| `-Wparentheses` | 14 | **不是缺陷**（`&&` 优先级本就高于 `||`），补括号 |
| `-Wignored-qualifiers` | 19 | 返回类型上的 `const` 对内建类型无意义，已去 |
| `-Wclass-memaccess` | 1 | **不是缺陷**，但 `memset(this,...)`/`fread(this,...)` 是隐患，记录不改 |
| `-Wdiscarded-qualifiers` | 4 | `free()` 收到 const 指针，形式 UB 实践无害，在手写内核头里，记录不改 |

### gcc 抓不到的那一类：未初始化类成员

`-Wmissing-field-initializers` 只管聚合初始化，不管构造函数；clang 有
`-Weffc++`，gcc 没有对应物。所以另写了 `tools/check_uninit_members.py`。

判定为**真隐患**并已修的 5 处：

| 位置 | 说明 |
|---|---|
| `ZQ_CNN_Layer::buffer` / `buffer_len` | 构造到 `ZQ_CNN_Net` 赋 `layers[i]->buffer` 之间是未初始化窗口。实测那段窗口无人读它（`->buffer` 全仓只有 4 处赋值、都在 `Forward` 之前；`use_buffer ? buffer : 0` 全在 `Forward` 内），**今天不是活 bug**，形状与附录 AT.4 的 `&ot == NULL` 一致 |
| `ZQ_CNN_Layer_NCHWC::buffer` / `buffer_len` | 同上 |
| `ZQ_CNN_Layer_UpSampling::sample_type` | 构造函数**完全不碰**它，唯一赋值点在 `ReadParam` |
| `ZQ_CNN_Net::input_C/input_H/input_W` | 只在 `LoadModel` 的 `GetTopDim` 那一处被赋值；`GetInputDim()` 在那之前调用会返回栈垃圾 |
| `ZQ_CNN_Net_NCHWC` 同上 | 同上 |

判定为**不是缺陷**的：`ZQ_CNN_Tensor4D::firstPixelData/rawData` 等 4 个 ——
6 个派生类的构造函数（Align0/128bit/256bit、NCHWC1/4/8）**全都**赋了值。
这正是本工具的已知局限（只看单文件、不看继承链），但换成"整个类体里有没有
被赋过值"这个判据之后，这类假阳性自动消失了。

### 工具自己踩的坑

第一版 `warn_sweep_src.py` 让 **6 个 TU 报 error**：

1. `.c` 用了 `g++` → `zq_avx_mathfun.c` 报 `narrowing conversion of '2147483648'`
   （C++11 braced-init 的检查，**在 C 里完全合法**）。按这条 error 去"修"生产代码
   就是为了迎合一个错误的编译器模式去改没问题的文件。
2. 少了 `-DZQ_CNN_USE_ZQ_GEMM=1`（`CMakeLists.txt:15` 默认值）→
   `zq_lstm_32f_align_c` `invalid conversion`
3. 少了 `-mavx2 -mfma`（`CMakeLists.txt:113`）

而「编不过」在按 `-W` 分桶的视角下看起来是「这个文件 0 条高信号警告」，
也就是**最干净的那一类**。这就是两个 sweep 工具都单独报告
「N 个目标编不过、告警没被扫到」的原因。

### 注意事项

- `ZQ_CNN_BBox240` 的 `memset(this, 0, sizeof(...))` + `fread(this, ...)`
  **不调用**子对象 `ZQ_CNN_BBox106` 的构造函数（标准上 UB）。今天所有成员都是
  平凡标量，行为无差异；但这套写法隐含「这个类永远只有平凡标量成员」的前提，
  加一个指针成员就会出事。要改得连 `fread` 一起改成逐字段反序列化 ——
  格式层面的重构，**本轮明确不改**。
- `zq_cnn_convolution_gemm_32f_align_c_raw.h` 里 4 处 `free(const*)` 同理：
  零行为变化，但要动的是**手写内核头**（AGENTS.md 对它有一整节规矩），记录不改。

### 验证

```
python tools/run_audit_checks.py --with-build                  D1~D4 全 OK
python tools/run_audit_checks.py --warn-sweep --src-sweep --msvc-asan
                                                              ALL CHECKS PASSED, exit 0
```

Windows 全量构建 0 error；Linux 全量构建 rc=0；Linux sample 回归 8 个全 rc=0；
Windows 侧 6 个关键 sample（含 `SampleGEMMAsmCompare` 对拍）全 rc=0。

## 新增/变更：MSVC `/analyze` 查出两个六年前的 `if (1 || ...)`（附录 AW）

### 变更文件

- `ZQCNN/ZQ_CNN_Forward_SSEUtils.cpp`：两处恒真条件各加一段注释（**零行为变化**）
- `tools/msvc_analyze.bat`（新增）：`/analyze` 扫主工程 7 个 TU
- `tools/capture_sample_outputs.sh`（新增）：跑 sample 并把**计时噪声归一化**后存
- `tools/ab_diff_sample_outputs.sh`（新增）
- `tools/ab_time_two_binaries.sh`（新增）：两个二进制**交替**跑比中位耗时
- `audit_k3_20261001.md`：新增**附录 AW**

### 问题

附录 AT/AU 用的是 gcc `-Wall -Wextra`。MSVC 侧到这一轮为止只做过
"能不能编过"的检查（附录 AR 的 `cl /Zs`）。`/analyze` 覆盖的是
**gcc 根本没有对应警告**的一类：C6001（解引用 NULL）、C6385/C6386（缓冲区溢出）、
C4701（可能未初始化的局部变量）、**C6235（恒真/恒假条件）**、C6246（变量遮蔽），
而且它在单个 TU 内是过程间分析。

### 实测结果

7 个 TU / 约 46 秒：

| TU | /analyze |
|---|---|
| `ZQ_CNN_Forward_SSEUtils.cpp` | **6 条**（2×C6235 + 4×C6246） |
| `ZQ_CNN_SSDDetectorPytorch.cpp` | 20 条（全 C6246，来自 ZQ_CNN_Layer.h） |
| 其余 5 个 TU | 0 |

**没有 C6001 / C6385 / C6386 / C4701** —— 前三轮高危缺陷里那些
"解析层不做范围校验"的路径，Code Analysis 一个都没额外捞出来。

### 缺陷本体

```cpp
if (1 || (out_HW >= 16 && filter_HWC >= 32 && filter_N >= 4)
    || ((out_HW >= 16 && filter_H == 1 && filter_W == 1 && filter_C >= 8 && filter_N >= 4)))
```

`1 ||` 让整个条件恒为真。`git log -L` 查出来是 2019-02-25 的 `75d4af2`
（"尝试支持arm_neon"）引入的，**六年了**。后果：

1. 那个形状守卫是**死代码**；
2. 它下面约 **470 行**手写标量卷积核（`kernel1x1_C4` / `kernel3x3` / `kernel5x5` /
   `general` 以及 Align0 / Align256bit 的对应版本）在默认构建
   （`ZQ_CNN_USE_ZQ_GEMM=1`）下**完全不可达**。

### 三个实验证明它今天不做任何事

**① 带探针跑 sample**，看守卫会挡掉哪些形状：

| sample | 会被挡掉的卷积次数 | 典型形状 |
|---|---|---|
| `SampleMTCNN` | 900 | `out 3x5 C=24 N=1 / filt 1x1x16 N=24`（out_HW=15, filter_HWC=16） |
| `SampleMTCNNLoadFromCode` | **10600** | `out 60x60 C=2 N=1 / filt 1x1x24 N=2`（filter_HWC=24 < 32） |
| `SampleCascadeOnet` | 18 | `out 1x1 C=128 / filt 1x1x128`（out_HW=1） |
| `SampleSSD` | **0** | — |
| `SampleFaceDetectorMTCNN` | **0** | — |

**② 去掉 `1 ||`，比输出**（编两个只差这两行的二进制）：
7 个 sample 去掉计时噪声后**逐字节相同**。两条路径对这些形状结果完全一致。

**③ 交替跑比性能**（`tools/ab_time_two_binaries.sh`，ABABAB）：

```
SampleMTCNN             n=15  A 8.674 ms  B 8.774 ms  B/A = 1.012
SampleMTCNNLoadFromCode  n=7  A 40.484 ms B 41.367 ms B/A = 1.022
```

**都在本机 7% 的噪声下限之内。**

### 处置：保留 `1 ||`，加注释，**不擅自恢复守卫**

`1 ||` 不是正确性缺陷，但它是"一个六年没人碰过的、让 470 行代码变成死代码的开关"。

**保留** `1 ||` 并在两处加注释写清来龙去脉与实测数据。
**不恢复守卫** —— 恢复它等于让 470 行**从未在当前配置下跑过**的标量核重新上线，
那是**增加**风险而不是减少风险；尤其 `SampleSSD` 与 `SampleFaceDetectorMTCNN`
对那条路径的覆盖是 **0 次**。

后续由所有者二选一：① 删掉不可达的 backup method；② 逐形状验证那些标量核之后
再恢复守卫。**本轮不替这个决定背书。**

### 注意事项

1. **这类恒真条件 gcc 完全看不见**（`-Wall -Wextra` 没有对应项；Clang 有
   `-Wconstant-logical-operand` 但只在 `-Weverything` 里）。这是
   "双平台各走各的检查"又一个具体收益。
2. **C6246 的 24 条不修**。`ZQ_CNN_Layer.h` 里 `dst_len` 被 20 个 `LoadParam`
   重载各自遮蔽，读的**就是**被遮蔽的那个参数，行为完全正确；参数名与成员名
   同名在这里是有意的写法。改名会牵动 20 个函数，收益为负。
3. **做输出对比前必须先归一化计时**。第一版直接 diff sample stdout，
   得到 33 行差异而**全是噪声**（同一份二进制连跑两次，`stage 1: cost`
   在 1.481~1.510 ms 之间跳、GF/s 差 20% 以上）。
   `tools/capture_sample_outputs.sh` 在存之前把时间/吞吐数字替换成占位符，
   保留 `nms cost: <T>ms, (159-->24)` 里的**检测框数量** —— 那才是要比的。

### 顺带查出一条已存在的非确定性

归一化之后仍有一处两次运行不同：

```
SampleMTCNNLoadFromCode.txt
< nms cost: <T>ms, (6067-->649)
> nms cost: <T>ms, (6098-->649)
```

**同一二进制、同一输入，NMS 之前的候选框数量每次都不同**（实测 6007/6068/6098），
**NMS 之后恒为 649**。

不是本轮引入的（两个变体都出现）。CNN 前向在单线程、确定性输入下给出不同的
原始候选数，通常意味着并行归约的求和顺序不固定（`ZQ_CNN_Net` 用 OpenMP）或
某处读了未初始化内存。**本轮没有定位到根因**，如实记录。实际影响被 NMS 吸收掉了，
但"前向结果随运行变化"在需要严格可复现的场合是不能接受的。

### 验证

    python tools/check_line_endings.py                line endings OK
    python tools/check_text_encoding.py               629 files, no U+FFFD
    wsl make -j8                                      rc=0
    A/B 输出对比（注释版 vs 原版）                     7 个 sample 逐字节相同

## 新增/变更：附录 AX —— `zq_cnn_lrn` 的堆越界写（ASan 坐实）+ 附录 AW.6 的数据竞争

### 变更文件

**内核（真缺陷）**

- `ZQCNN/layers_c/zq_cnn_lrn_32f_align_c_raw.h`：
  ① `pad_size` 向下取整导致 `local_size==1 && C%align!=0` 时**堆写越界**（最多 7 个 float）
  ② `local_sum_buf` 只给 `C` 个 float，而 `zq_mm_load_ps` 以 `align` 为步长读
  → **堆读越界**

**并发（真缺陷）**

- `ZQCNN/ZQ_CNN_MTCNN.h`：P-net 的 `#pragma omp parallel for` 缺
  `reduction(+:before_count, after_count)`，两个**诊断计数器**在多线程里无锁 `+=`
  → 数据竞争（UB），表现为 printf 出来的 pre-NMS 候选框数每次运行都不同

**测试 / 工具**

- `tools/zq_lrn_check.cpp`（新增）：第 11 个回归测试，直接调 LRN 内核，
  C 取 1..17 全部余数类 × local_size 取 1/3/5/7/9，与标量参考对拍
- `tools/run_zqlib_checks.py`：新增 `EXTRA_SOURCES`/`EXTRA_LINK`/`EXTRA_INC`/
  `EXTRA_CXXFLAGS` 四张按测试名的表（测内核的测试要额外编两个 math `.c`，
  且主 TU 也必须带 `-mavx2 -mfma`）
- `tools/msvc_analyze.bat`：扩到 14 个 TU（加了 `math/` 与几个 `layers_c/` 内核）
- `tools/probe_nondeterminism.sh`（新增）：把全部 nms 计数抓出来跨多次运行比对
- `audit_k3_20261001.md`：新增**附录 AX**

### 缺陷本体

```c
pad_size = local_size / 2 + zq_mm_align_size - 1;
pad_size = pad_size - pad_size%zq_mm_align_size;     // 向下取整
len = C + (pad_size << 1);
square_buf = _aligned_malloc(sizeof(float)*len, ...);
for (c = 0, square_ptr = square_buf + pad_size; c < C; c += zq_mm_align_size, ...)
    zq_mm_store_ps(square_ptr, ...);                  // 一次写 align 个 float
```

最后一下写到 `pad_size + ceil(C/align)*align - 1`，而缓冲区只有 `len` 个。
`local_size == 1` 时 `pad_size` 被向下取整成 **0**，于是 `C % align != 0` 就必然越界。

**可达性**：`LRN_across_channels` 只校验 `local_size % 2 != 1`，
`local_size == 1` 通过；`local_size` 与 `C` 都来自模型文件（`.zqparams`），
按本报告的威胁模型是**不可信输入**。

### ASan 实测（默认构建是 AVX2，align=8）

    ==114945==ERROR: AddressSanitizer: heap-buffer-overflow
    WRITE of size 32 at 0x605000000020
        #1 zq_cnn_lrn_across_channels_32f_align256bit  zq_cnn_lrn_32f_align_c_raw.h:64
    0x605000000024 is located 0 bytes to the right of 4-byte region

修完第一处再跑，ASan 立刻指出第二处（`:102` 读越界）。
两处都修完：**69 个用例全过，无越界，数值与标量参考一致（相对误差 ≤ 4.1e-6）**。

### 修法（两处都是零数值变化）

```c
if (pad_size < zq_mm_align_size) pad_size = zq_mm_align_size;   // 新增
local_sum_buf = _aligned_malloc(sizeof(float)*(C + zq_mm_align_size), ...);
```

多出来的 pad 元素被初始化循环填 0，累加窗口随 `pad_size` 整体平移，
覆盖的**相对区间**没变 —— 对拍验证了这一点。

### 顺带：AW.6 那条非确定性，根因找到了

同一二进制、同一输入，pre-NMS 候选框数每次不同（6007/6067/6068/6098 四种），
而 post-NMS 那个数恒定。原因是：

```cpp
int before_count = 0, after_count = 0;                 // 声明在 parallel 区域**外面**
#pragma omp parallel for schedule(dynamic, chunk_size) num_threads(thread_num)
    for (int bb = 0; bb < block_num; bb++) {
        ...
        before_count += tmp_before_count;               // 多线程无锁 +=，没有 reduction
        after_count  += tmp_after_count;
    }
...
after_count = bounding_boxes[i].size();                 // 被整个覆盖，所以第二个数恒定
printf("nms cost: %.3f ms, (%d-->%d)\n", ..., before_count, after_count);
```

**货真价实的数据竞争（UB）**。但**检测结果不受影响**：
`bounding_boxes[i]` / `bounding_scores[i]` 按 `bb` 下标分，每个线程只碰自己那几个；
这两个计数器除了 printf 没有任何消费者。属于"诊断数字不准"，不是"算法结果错"。

已加 `reduction(+:before_count, after_count)`。**计算结果一字不变**，
数字变稳定且正确。

> 诚实说明：竞争窗口很窄 —— 修之前连跑 8 次也没复现、修之后连跑 10 次全恒定，
> 所以**拿不出一个确定性的 A/B 证据**。判定依据是三条独立事实：
> ① OpenMP 语义上「共享变量在并行区里无 reduction 无 atomic 地 `+=`」本身就是 UB；
> ② 观测到的现象（第一个数变、第二个不变）与"第二个在 915 行被覆盖"的代码结构
>    **精确吻合**；③ 加上 reduction 之后竞争从定义上消失。

### 注意事项

1. **MSVC `/analyze` 指错了行，但没白跑。** 它报的是 `:68`（安全，只是零余量）
   和 `:73`（安全，卡在边界），真正越界的 `:64` 它**没报**。是它让我去读了那个文件，
   而那个文件里确实有 bug。**静态分析器是筛子，动态实测才是判据。**
2. **这一类不是通用模式。** `pad_size - pad_size%align` 这种向下取整在整个
   `layers_c` / `layers_nchwc` 的 `*_raw.h` 里**只有 LRN 这一处**；
   其它内核按 `C % align32 == 0 / % align16 == 0 / ...` 分派，不存在同类问题。
3. **`.c` 必须用 gcc 编**（g++ 会把 `zq_avx_mathfun.c` 的
   `_PS256_CONST_TYPE(sign_mask, int, 0x80000000)` 判成 narrowing 直接失败）；
   测内核的测试主 TU 也要带 `-mavx2 -mfma`，否则 `_mm256_set1_ps` 报
   `target specific option mismatch`。
4. **我的第一版 LRN 测试把对齐宽度写成 4（实际是 8）**，
   于是像素地址 16 字节步进而 `_mm256_load_ps` 要求 32 字节对齐 → SIGSEGV，
   **而且 ASan 报出来的故障地址是 0x000000000000**，第一反应会误判成空指针。

## 新增/变更：附录 AY —— `/analyze` 铺到全部 43 个 TU，查出 2 处真缺陷 + 1 段死代码

### 变更文件

- `ZQCNN/layers_c/zq_cnn_batchnormscale_32f_align_c_raw.h`：
  2 对 `_aligned_malloc` 补 NULL 守卫
- `ZQCNN/layers_nchwc/zq_cnn_batchnormscale_nchwc_raw.h`：
  同上 2 对；另把读模型参数的循环上界从 `ceil_C` 改成 `in_C`，
  `[in_C, ceil_C)` 改为清零
- `tools/run_msvc_analyze.py`（新增）：枚举 43 个 TU 并驱动 `/analyze`，
  按 **gbk** 解析 cl 的输出
- `tools/msvc_analyze.bat`：删掉坏掉的 `goto`/label 扫描逻辑，改成"收一串文件参数"
- `tools/zq_bns_check.cpp`（新增）：钉住上面两处修复
- `audit_k3_20261001.md`：新增**附录 AY**

### 全量总账（43 个 TU）

    C6386  58  缓冲区溢出（写）        C6011  24  解引用 NULL 指针   <-- 真缺陷
    C6385  43  缓冲区溢出（读）        C6326  18  可能的算术溢出
    C6246  33  变量遮蔽               C4090   8  switch 漏枚举
                                      C6387   4  指针可能为 0
                                      C6235   2  恒真条件（AW 的两个 1 ||）

8 个 TU 有发现，其余 35 个干净。

### 真缺陷 ①：`_aligned_malloc` 没判 NULL（4 对 8 个分配点）

`in_C` / `ceil_C` 来自模型文件（不可信输入），一个巨大的通道数就能让分配失败，
而代码紧接着就解引用。函数返回 `void`，失败时释放兄弟再返回；
**正常路径行为一字不变**。

### 真缺陷 ②：读模型参数时用了 `ceil_C` 上界

`a`/`b` 这两个补零向量要按 `ceil_C` 填满（主内核整宽读），
但 `slope_data` / `var_data` / `mean_data` / `bias_data` 是**每通道一个 float**
的模型参数、长度只有 `in_C`。`ceil_C > in_C` 时四个数组各被多读 `align-1` 个。

ASan 实测（`tools/zq_bns_check.cpp`）：

    ERROR: AddressSanitizer: heap-buffer-overflow READ of size 4
        #1 zq_cnn_batchnormscale_mean_var_scale_bias_nchwc4  ..._raw.h:40
    0x... is located 0 bytes to the right of 20-byte region   <- 5 个 float 的模型数组

### 死代码：那个文件根本没有调用方

查第三件事（主内核索引约定不对）时顺藤摸瓜发现：

    $ grep -rn "zq_cnn_batchnormscale_mean_var_scale_bias_nchwc" --include=*.h --include=*.c --include=*.cpp .
    （除本测试外，零引用）

真正在跑的 NCHWC 路径是 `ZQ_CNN_Forward_SSEUtils_NCHWC.h:23` 的
`BatchNormScaleBias_Compute_b_a`，它**自己有一份标量 C++ 实现**，
循环上界就是 `C`、**没有越界**。

文件本身仍被 CMake 的 `file(GLOB .../layers_nchwc/*.c)` 编进库 ——
这份死代码每次构建都会编一遍，只是没人调。而它那个"不对"的索引约定
（`c` 循环用**整张图**的步长跨过去，是 **NCHW** 布局的约定）
也就说得通了：从 NCHW 版复制过来**没改完**。

**处置**：①②已修；**索引约定不修** —— 修它等于按 NCHWC 布局把这个文件整个重写，
而它没有调用方。正确做法是删掉它，但删一个被 CMake glob 进来的文件属于结构性改动，
本轮明确不做，记在附录里由所有者决定。

`tools/zq_bns_check.cpp` 保留，用来把已修的 ①②钉住；文件头写明
「③ 仍会触发并 abort，这是**已知未修项，不是回归**」。

### 工具：批处理的"扫描全部"逻辑坏掉了

`msvc_analyze.bat` 原来用 `goto` + label + `EnableDelayedExpansion` 枚举文件，
结果 **cmd 开始把 `rem` 注释的片段当命令执行**（`'ses' 不是内部或外部命令`、
`'1.md' 不是…`）。枚举挪到 Python 侧，bat 只负责"收一串文件参数，逐个编"。

顺带修掉一个**假绿**：`cl` 的输出是**本地代码页**（本机 GBK），
按 utf-8 读会得到一堆 U+FFFD，而要匹配的 `warning C6386` 恰好是 ASCII ——
统计会显示"0 条"，看起来像"全部干净"。现在固定按 gbk 读。

### 注意事项

- **C6xxx 不是一律不可信。** 附录 AX 里 C6386 的行号不准（它没报真正越界那行），
  而这里 C6011 的行号是准的。更准确的表述：**缓冲区/NULL 类（C6011/C6385/C6386/C6387）
  值得逐条查；C6386 里"路径推断型"的那部分要靠动态实测确认。**
- **LRN 修完之后，那 7 条 C6386 + 6 条 C6385 依然在报**，而 ASan 证明那里干净 ——
  因为缓冲区大小依赖一个 MSVC 证不出来的不变式。这条留作"静态分析器是筛子"的又一例。
- C6246（33 处）、C4090（8 处）、C6387（4 处）本轮判定为不改，理由见附录 AY.6。
- **C6326 算术溢出 18 处已核对：不是缺陷。** 全部是
  `if (zq_mm_align_size >= 4)`，而 `zq_mm_align_size` 是每个变体的**编译期常量**
  （nchwc1=1 / nchwc4=4 / nchwc8=8），两处（1287 / 1407）**都有 else 分支**
  （1352 / 1491），nchwc1 走标量路径。属刻意的 SIMD 宽度分派。


## 新增/变更：附录 BD —— pooling 的 stride=0 是**模型可控的除零**（已修）

### 变更文件
- `ZQCNN/ZQ_CNN_Layer.h`：`ZQ_CNN_Layer_Pooling::ReadParam` 增加
  `kernel_H/kernel_W/stride_H/stride_W <= 0` 的值域校验
- `audit_k3_20261001.md`：新增**附录 BD**

### 问题

`ReadParam` 只校验参数**在不在**（`has_kernelH` 之类），**不校验值**：

    else if (_my_strcmpi("stride_H", paras[n][0].c_str()) == 0) {
        if (paras[n].size() >= 2) { has_strideH = true; stride_H = atoi(paras[n][1].c_str()); }
    }
    ...
    if (!global_pool)
        return has_kernelH && has_kernelW && has_strideH && has_strideW && has_bottom && has_top && has_name;

于是模型文件里写 `stride: 0` 会一路走到

    need_H = (int)ceil((float)(in_H + pad_H_top + pad_H_bottom - kernel_H) / stride_H + 1);

**浮点除以 0** -> ±inf，而 **`(int)ceil(inf)` 是未定义行为**（x86 上
cvttss2si 给 INT_MIN，恰好被后面的 `need_H <= 0` 挡掉 —— **那是巧合不是保证**；
主工程还开着 `-Ofast -ffast-math`，编译器有理由假设这种转换不会发生）。

`kernel_size: 0` 不崩，但池化循环一次都不执行，**输出整片变成 -FLT_MAX（max）/ 0（avg）**。

`stride_H < 0` 会让 need_H 变成很大的正数，那条路已被附录 H3/H4 的
`0x7FFFFFFF` 上界校验挡住。

这正是本报告开头那条首要漏洞画像 ——「模型/配置文件的解析层几乎不做
范围与一致性校验」—— 只是这一次落在**除零**上而不是缓冲区上。

### 修法

`ReadParam` 返回 `bool`、调用方会中止模型加载，所以在它返回前加：

    if (!global_pool
        && (kernel_H <= 0 || kernel_W <= 0 || stride_H <= 0 || stride_W <= 0))
    {
        std::cout << "Layer " << name << " invalid pooling params: kernel "
                  << kernel_H << "x" << kernel_W
                  << " stride " << stride_H << "x" << stride_W
                  << " (must all be > 0)
";
        return false;
    }

**零行为变化**：所有合法模型的 kernel/stride 本来就都 > 0。

### 顺带答了附录 BB.5 留下的待办

BB 说「内核要求 `out_H = ceil((in_H - kernel_H)/stride_H) + 1`，
契约被破坏时的行为不在覆盖范围内，那属于调用方的校验责任」。
查了 `ZQ_CNN_Forward_SSEUtils::MaxPooling`（1530 行）之后确认：
**没有任何外部调用方能传错** —— `need_H/need_W` 是 wrapper 自己
从 `in_H/in_W/kernel/stride/pad` **现算**的，一路到内核中间没有别的层。
`nopadding_*` 那族也只在 `pad_*` 全 0 时被调用，与语义一致。
**这条契约由 `MaxPooling` 自己保证，BB 的测试按公式传参是对的。**

### 验证

    wsl make -j8                     100% Built，无 error
    run_sample_regression.sh          8 个 sample 全 rc=0


## 新增/变更：附录 BE —— 卷积的 `stride=0` 是 **SIGFPE**（比 BD 更硬）

### 怎么找到的

BD 在 pooling 上查到"模型可控的除零"之后做了一次全量普查：
`ZQ_CNN_Layer.h` 里有 **25 个** `ReadParam` 会用 `atoi` 取参数，
其中**只有 2 个**校验了值（Pooling —— 刚修的那个 —— 和 Softmax/Reduction 的 axis）。

先逐个确认了"没校验"的那批是不是真没兜住：Reshape / Permute / Flatten /
Concat / Reduction 的 `axis` 都有显式校验（Concat 在
`_concat_NCHW_get_size` 第一行 `if (axis < 0 || axis >= 4) return false;`），
LSTM 的 `hidden_dim` 靠每次使用都过 `ChangeSize`、而 `ChangeSize` 有附录 H3/H4
加的 `0x7FFFFFFF` 上界校验兜住。

**只有 Convolution / DepthwiseConvolution / DeConvolution 两层都没有。**

### 缺陷本体：7 处整数除以 stride

`ZQ_CNN_Forward_SSEUtils.h` 里 Convolution / DepthwiseConvolution 的 7 个 wrapper：

    int need_H = (in_H - (filter_H-1)*dilation_H - 1 + (padH_top+padH_bottom)) / strideH + 1;
    int need_W = (in_W - (filter_W-1)*dilation_W - 1 + (padW_left+padW_right)) / strideW + 1;

`strideH` 来自**模型文件**。**这是整数除法** —— `INT_MIN / 0` 在 x86 上是 `idiv`，
直接 **SIGFPE**，进程当场死。实测最小复现确认（`caught signal 8`）。

**比 BD 硬得多**：

| | pooling（BD） | 卷积（BE） |
|---|---|---|
| 除法类型 | 浮点 | **整数** |
| 除以 0 的结果 | ±inf | **trap** |
| 后果 | `(int)ceil(inf)` 是 UB；x86 给 INT_MIN，**恰好**被后面的 `need_H <= 0` 挡掉 | **SIGFPE，确定性崩** |
| 有没有兜底 | 有（但是巧合） | **没有** |

`DeConvolution` 不中招：它那四处是 `((in_H-1)*strideH + 1 - ...)`，**乘法**不是除法。

### 修法（纵深防御，两层）

**第一层 —— wrapper 里加守卫**（7 处，`ZQ_CNN_Forward_SSEUtils.h`）：

    // 审计修复 2026-10-02（附录 BE）：strideH/strideW 来自**模型文件**（不可信输入），
    // 下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，直接 SIGFPE、进程死。
    if (strideH <= 0 || strideW <= 0)
        return false;

**第二层 —— `ReadParam` 里拒绝**（3 处，`ZQ_CNN_Layer_Convolution` /
`_DepthwiseConvolution` / `_DeConvolution`）：kernel/stride/dilate 任一 <= 0 即
`return false` 并打一行原因。

第一层是关键（wrapper 是 public API，层只是它的一个调用方）；
第二层让错误在**加载模型时**就以一条可读的消息暴露出来。

**零行为变化**：合法模型的 kernel/stride/dilate 本来就都 > 0。

### 这条普查本身的价值

BD 是一次"碰巧看见了"，BE 证明了它是**一类**。25 个会 `atoi` 参数的 ReadParam
里只有 3 个现在自己校验值域；其余的要么在 Forward 里有守卫，要么靠 `ChangeSize`
的 `0x7FFFFFFF` 上界兜住。**这个"分层兜底"的结构本身是健康的** ——
关键是每一层要么自己查、要么下游一定查。BD/BE 暴露的是**两层都没有**的那一处，
所以修的时候两层都补上。

### 验证

    wsl make -j8                      100% Built，无 error
    run_sample_regression.sh           8 个 sample 全 rc=0


## 新增/变更：附录 BF —— `Tile` 的整数回绕导致堆溢出写（已修）

### 变更文件
- `ZQCNN/ZQ_CNN_Tensor4D.h`：`Tile` 的四个乘积改用 `__int64` 并做上界校验
- `audit_k3_20261001.md`：新增**附录 BF**

### 接着 BE 的普查往下走

25 个会 `atoi` 参数的 `ReadParam` 里，剩下的逐个走完：

| 层 | 参数 | 结论 |
|---|---|---|
| PriorBox | `step_h/step_w` | OK `ZQ_CNN_Forward_SSEUtils.cpp` 的 4365/4556/4675 三处都有 `if (step_w == 0 || step_h == 0)` |
| InnerProduct | `kernel_*` / `num_output` | OK 输出尺寸从 filter 张量的实际形状推导，并查形状一致性；无除法 |
| UpSampling | `align_type` | OK 只被当作 `sample_align_type == 1` 做**比较**，全仓没有 `[align_type]` 下标用法 |
| UpSampling | `dst_h/dst_w` | OK 过 `ChangeSize` 的 `0x7FFFFFFF` 上界 |
| DetectionOutput | `keep_top_k` / `nms_top_k` | OK `GetMaxScoreIndex` 是 `if (top_k > -1 && top_k < size()) resize(top_k)`，只截断不增长 |
| LRN | `local_size` | OK `local_size % 2 != 1` 拒掉 0 与负数；`==1` 的越界已在 AX 修掉 |
| **Tile** | `tile_n/h/w/c` | **两处都没有兜住** |

### 缺陷本体

`ZQ_CNN_Tensor4D::Tile`（`ZQ_CNN_Tensor4D.h:240`）：

    int out_C = C*tile_c;                       // <-- 未检查的整数乘法
    ...
    for (int tc = 0; tc < tile_c; tc++) {        // <-- 按 tile_c 的**原始值**循环
        memcpy(out_c_ptr, in_c_ptr, sizeof(float)*C);
        out_c_ptr += C;
    }

`tile_*` 来自模型文件（`ZQ_CNN_Layer_Tile::ReadParam`，也是 `atoi`、无值域校验）。

**关键在"分配按回绕后的值、写入按原始值"**：

    N=H=W=1, C=3, tile_c=0x55555556
      3 * 0x55555556 = 0x100000002，截成 int 是 **2**
      -> out_C = 2，ChangeSize(1,1,1,2) 成功
      -> 循环却按 0x55555556 次 memcpy 12 字节并前进 12 字节
      -> **堆缓冲区溢出写**（越界约 10 GB）

`out_N/out_H/out_W` 同理。

**这与附录 H3/H4 修的那类不是同一处**：那次是 `ChangeSize` **内部**
`dst_sliceStep*dst_N*sizeof(float)` 的溢出；这次是**调用方**传给 `ChangeSize`
的尺寸本身就已经是回绕过的错误值，`ChangeSize` 看不到问题。

### 修法

四个乘积改用 `__int64`，要求落在 `[1, 0x7FFFFFFF]`，并拒掉 `tile_* <= 0`。
**零行为变化**：合法模型的 `tile_*` 是 1 或 2，乘积远小于上限。
NCHWC 那条线**没有 Tile**（`ZQ_CNN_Tensor4D_NCHWC.h` 里搜不到），只需改这一处。

### 教训：溢出防护要作用在**乘法发生的那一行**

BE 的结论是"每一层要么自己查、要么下游一定查"，BF 是个反例：
`ChangeSize` 的上界校验**只检查传进来的值**，
它无法知道**这个值本身是不是某个乘积回绕的结果**。
`Tile` 原本把"检查"和"使用"放在同一函数里，中间隔着一次"回绕后的错误尺寸"，
于是检查全过、使用越界。

附录 H3/H4 加在 `ChangeSize` 里的 `__int64` 中间量之所以有效，
是因为那次溢出**发生在 `ChangeSize` 内部**；同样的手法搬到调用方就不管用了。

### 验证

    wsl make -j8                      100% Built，无 error
    run_sample_regression.sh           8 个 sample 全 rc=0


## 新增/变更：附录 BG —— 值域普查做成门禁，当场抓出我自己的漏网

### 变更文件
- `tools/check_param_domain.py`（新增）：扫「ReadParam 只校验参数在不在、不校验值」
- `tools/param_domain_baseline.txt`（新增）：61 条"已校验"的基线
- `tools/run_audit_checks.py`：新增 A7/A8 组
- `ZQCNN/ZQ_CNN_Forward_SSEUtils_NCHWC.cpp`：**25 处**整数除 stride 守卫
- `ZQCNN/ZQ_CNN_Layer_NCHWC.h`：3 处 ReadParam 值域校验
- `audit_k3_20261001.md`：新增**附录 BG**

### 为什么要做这个工具

BD / BE / BF 三轮都从同一件事长出来：`kernel_*` / `stride_*` / `tile_*` 全部来自
**不可信的模型文件**，而 `ReadParam` 只检查"这一行在不在"（`has_strideH` 之类）。

本轮把 25 个会 `atoi` 参数的 `ReadParam` 全过了一遍、逐个确认了每个参数最终
有没有被兜住（结论记在附录 BF.1），但那份结论**写在报告里** —— 下次有人删掉某个
守卫、或者新增一个层类，没有任何机制会提醒。

`check_param_domain.py` 把「哪些 (层, 参数) 在 `ReadParam` 里被校验过」变成
**可回归的基线**。它只做这一件事：对每个 `ReadParam` 抽出 `atoi` 变量，
在**同一函数体**里找值域校验（`x <= 0` / `x < 1` / `x != 0` / `invalid x`）。
它**不做**「下游有没有兜住」—— 那是语义判断，正则做不了，结论仍在 BF.1。

### 工具自己踩的三个坑（都由 --selfcheck 逮到）

1. **`find_check` 写成了"正则字符串非空吗"而不是"匹不匹配"**
   （`if r.pattern % re.escape(var):` 恒为真）。于是工具报告
   **"全部 211 个 (层, 参数) 都已校验"**，而实际上 BD/BE/BF 三条缺陷就在这一族里。
   自测里那个**故意不校验**的 `tag` 立刻被抓出来。
   > 兑现 AGENTS.md 那条「一个"什么都查不出来"的检查工具必须自带自测」。
   > 不写自测的话，这个工具会**绿着**挡住后续所有同类缺陷。
2. **匹配模式太松**：原本还有「出现过这个变量、同一句里又有 return false」
   这种写法，会把 `return has_a && has_b && has_name` 误判成值域校验。已删，
   并明确不认枚举/集合限制（`x == 0 || x == 1`）与 `x &&` 这类布尔用法。
3. **`--save-baseline` 不带路径会 IndexError**（手写 argparse 的通病）。已修。

### 工具当场抓出：**BE 的修复只做了一半**

    ZQ_CNN_Layer_NCHWC.h   ZQ_CNN_Layer_NCHWC_Convolution          已校验 0 / 未校验 11
    ZQ_CNN_Layer_NCHWC.h   ZQ_CNN_Layer_NCHWC_DepthwiseConvolution  已校验 0 / 未校验 11
    ZQ_CNN_Layer_NCHWC.h   ZQ_CNN_Layer_NCHWC_Pooling               已校验 0 / 未校验 7

**NCHWC 那三个层类一个都没修**。查下去发现不止 `ReadParam` ——
`ZQ_CNN_Forward_SSEUtils_NCHWC.cpp` 里有 **25 处**和 BE 那 7 处**一字不差**的
整数除法：

    int need_H = (in_H - (filter_H-1)*dilation_H - 1 + (padH << 1)) / strideH + 1;

NCHWC 的卷积代码是 NCHW 那边**整份复制**过去的，我上一轮只改了原版。
`stride: 0` 写进一个 NCHWC 模型照样 SIGFPE。

> 这是「收口一类缺陷必须全仓枚举、不能只信上一轮的清单」的重演，
> 只不过这次清单是我自己刚写的。

**已补**：25 处 wrapper 守卫 + 3 处 ReadParam 值域校验
（`ZQ_CNN_Layer_NCHWC.h` 里 `name` 要写全 `ZQ_CNN_Layer_NCHWC<Tensor4D>::name`）。

### 现在的状态

基线 61 条（只记已校验的那些）。未校验的 `num_output` / `pad_*` / `with_bias` /
`type` / `operation` 在 BF.1 里已逐个确认过是被兜住的
（`ChangeSize` 的上界、`== 0` 布尔、集合限制），**不需要在 ReadParam 里再查一遍**。

### 接入统一入口

    A7  ReadParam 值域校验基线自测 (check_param_domain --selfcheck)
    A8  ReadParam 值域校验基线比对 (check_param_domain --check-baseline)

A8 在**已校验的参数少了一条**时退出 1。


## 新增/变更：附录 BH —— 把"同一类收口"也做成门禁（check_div_guard）

### 变更文件
- `tools/check_div_guard.py`（新增）：扫整棵树里"除以模型参数"的语句
- `tools/div_guard_allowlist.txt`（新增）：人工白名单（守卫在别处），强制写理由
- `tools/run_audit_checks.py`：新增 A9/A10 组
- `audit_k3_20261001.md`：新增**附录 BH**

### 为什么还要再做一次

BG 的工具抓出了我自己的漏网：BE 修了 `ZQ_CNN_Forward_SSEUtils.h` 里 7 处
`/ strideH`，**忘了 NCHWC 那份复制品里的 25 处**。BG 把"层类 ReadParam 有没有
值域校验"变成了门禁，但**转发层里那 25 处除法**不归它管 —— 只堵住了一半的路径。

`check_div_guard.py` 补的就是这一半：扫 `ZQCNN/` + `ZQlibFaceID/` +
`SamplesZQCNN/` + `SamplesZQlibFaceID/` 里每一处"除以一个模型参数"的语句，
检查它上方有没有 `<= 0` 的守卫。

### 现在的数字

    "除以模型参数" 守卫普查：95 处 / 已守卫 64 / 白名单 31 / **待查 0**

- 64 处已守卫：BE 的 7 处 + BG 补的 NCHWC 25 处 + 其它
- 31 处白名单：全在 `ZQ_CNN_MTCNN*.h`（6 个文件），守卫在**另一个方法**里 ——
  `InitFromBuffer` 的 `this->pnet_stride = __max(1, pnet_stride);`
  （`ZQ_CNN_MTCNN.h:227`，223 行本来就有注释说明）。
  本工具只看得到除法**所在函数体内上方 8 行**，看不到那里，所以走白名单，
  并且**白名单强制要求写理由**。
- 0 处待查

### 工具自己踩的四个坑

1. **上下文越过函数边界**：直接取"上方 8 行"，于是函数 A 的守卫会"罩住"
   紧随其后的函数 B —— 自测里那个**故意没守卫**的 `f2` 被误判成有守卫。
   修法：往上遇到第一个「行首是 } 的行」就截断。
2. **没处理块注释**：`/*if (...) std::cout << "... step_h/step_w ...";*/`
   里的字符串被当成真除法。
3. **手写块注释状态机写错了**：补第 2 条时用了 `while ... break` 的写法，
   结果把**真代码**也抹掉了 —— 命中数从 96 变成 131，一眼就看得出不对。
   > 兑现 AGENTS.md 那条「别自己发明半吊子的解析器」：需要解析 C++ 语法时，
   > 正确答案是**用编译器**（gcc -E / MSVC /Zs），不是自己写正则状态机。
4. **白名单静默匹配不上**：扫描结果走 `os.path.relpath`（Windows 上是**反斜杠**），
   而白名单是人手写的（习惯写**正斜杠**）。症状是"白名单 0 条、待查 31 条" ——
   看起来像根本没有白名单机制。**白名单匹配不上是最危险的一类工具 bug**：
   它不会报错，只会让工具退化成"什么都不拦"。修法：读入时归一化。

### 接入统一入口

    A9   "除以模型参数" 守卫普查自测 (check_div_guard --selfcheck)
    A10  "除以模型参数" 守卫普查 (check_div_guard)

A10 在**出现一条既没有同函数守卫、也不在白名单里的除法**时退出 1。

### 至此"模型可控的值"这一族一共堵了四处

| 附录 | 层 | 后果 | 状态 |
|---|---|---|---|
| BD | Pooling 的 stride=0 | 浮点除零 -> (int)ceil(inf) 是 UB | 已修（NCHW + NCHWC） |
| BE | 卷积的 stride=0 | **整数除零 -> SIGFPE** | 已修（NCHW + NCHWC 共 32 处） |
| BF | Tile 的 tile_* | 整数回绕 -> **堆溢出写** | 已修 |
| BG/BH | 门禁 | 防止再次"只修一半" | 已上线 |

### 验证

    run_audit_checks.py --quick     A1~A10 + B 全 OK


## 新增/变更：附录 BI —— 把 BC 那条未决项做完（内核没 bug，是我的测试用错形状）

### 结论先说

`zq_cnn_innerproduct_gemm_32f_align{128,256}bit_same_pixstep_batch`
**是正确的**。附录 BC 记的"第一个用例就崩"从头到尾是测试的问题。
144 个用例（生产形状下）全部通过，相对误差 <= 2.6e-08，无 ASan 报错、无泄漏。

### 生产调用点的三个约定（BC 里我全猜错了）

唯一调用点 `ZQ_CNN_Forward_SSEUtils.cpp:2417`：

| 约定 | 内容 | BC 里我用的 |
|---|---|---|
| 形状门槛 | `out_N >= 16 && filter_N >= 16` 才走 GEMM，其余走 `..._noborder` | **N=1、filter_N=1..17**（生产根本不这么调） |
| `filter_sliceStep` | **K = H*W*C**（逻辑长度），同时当 sgemm 的 `ldb` | **K*Fpad**（大了 8 倍，im2col 只填 1/8 的列） |
| 三个 out 步长 | 传**同一个** `out_sliceStep`（out 是 [N,1,1,filter_N]） | 各传各的 |

### filter 布局是 **f 优先**（`[filter_N][K]`）

对拍时**两种布局都算**，结果一目了然：

    a128 malloc align=4 N=16 2x3x8 F=16  A(k优先)=3.43e-01 B(f优先)=2.17e-08  ok f优先[F][K]
    a256 malloc align=8 N=20 1x1x16 F=33  A(k优先)=2.35e-01 B(f优先)=2.39e-08  ok f优先[F][K]

内核把 `filters_data` 当 `Bt`（`ldb = K`）传给 sgemm，而实际张量是 NCHW 的
`[filter_N][H][W][C]`，即 `filters[f*K + k]`。两个数差 4 个数量级。
**同时算两种布局**才没把它误报成"内核错"。

> 对拍失败时先怀疑自己的参考实现：一个**恒定**的相对误差（这里恒为 ~0.3，
> 换形状/对齐/分配方式都不变）说明差异是**结构性的**，不是"某处算错一点点"。

### buffer 路径的所有权

`buffer != 0` 时内核把 `*buffer` 存进调用方的槽位就不再管了。第一版测试里
`buf` 是局部变量、不 free，LeakSanitizer 报 **84 KB x 36** —— 那是测试的假泄漏。
**这条约定内核里没有任何注释**，将来有人改成自己 free 就是 double free，
值得补一行（本轮未改 `layers_c/` 的注释风格，等下次动那个文件时一起）。

### 接入方式：默认不跑

要链 `ZQ_GEMM` 三个 TU，其中 `zq_gemm_32f_align_c.c` **单个编一次 >5 分钟**
（98 MB 的 .o）。放进去会让日常回归从 30 秒变成 6 分钟以上，所以给了 `--with-slow`：

    python tools/run_zqlib_checks.py --with-slow zq_innerproduct    # 约 7 分钟

不加时跳过，并**把理由整段打出来**（与 SKIP 表同一个规矩）。
日常回归仍是 14 组 ASan 测试（~30 秒）。

### 顺带修的链接问题

`EXTRA_SOURCES` 一开始只编了 ZQ_GEMM 那三个 TU、**没编内核自己**，
于是 undefined reference。补上
`ZQCNN/layers_c/zq_cnn_innerproduct_gemm_32f_align_c.c` 之后通过。

### 我在这一个测试上错了四次

1. 声明的参数个数写错（BC.4）
2. 形状用错（约定 ①）
3. `filter_sliceStep` 约定搞错（约定 ②）
4. filter 布局搞反（BI.3）

四次**每一次都有一个很自信的"解释"**。**一个观察对不上时，先怀疑观察工具本身** ——
这条已经写进 AGENTS.md，但执行上我显然没做到。


## 新增/变更：附录 BJ —— 16 个文件共享一条**没有任何文档**的所有权约定

### 起因

附录 BI 里那个 LeakSanitizer 假泄漏（84 KB x 36）其实是个信号：
测试作者（含我）**不知道 `*buffer` 到底归谁**，只能靠 ASan 报出来的现象反推。

反过来说：这条约定一旦被误解成"内核会替你释放"，改代码的人就会在
内核里加一句 `_aligned_free(*buffer)` —— **直接变成 double free**。

### 现状：全 0

    $ grep -rl "void\*\* *buffer" --include=*.h ZQCNN/layers_c ZQCNN/layers_nchwc | wc -l
    16
    $ 对每个文件 grep "所有权|调用方负责|归调用方"
    （16 个文件全部是 0）

带这对参数的公开头/源文件共 16 个（convolution / deconvolution / innerproduct /
lstm 的 NCHW 版 + NCHWC 版 + packed4/prepack4 变体），**没有一处**说明
`*buffer` 归谁。

### 补的契约说明

在 6 个**公开头**的文件开头各加一段（`layers_c/` 的 convolution / deconvolution /
innerproduct / lstm，`layers_nchwc/` 的 convolution / innerproduct）：

    buffer == NULL  —— 内核自己 _aligned_malloc / _aligned_free，用完即走。
    buffer != NULL  —— 读写的是**调用方**持有的两块内存：
                        *buffer      指向一块 _aligned_malloc 出来的内存
                                      （或者 NULL，表示"还没分配过"）；
                        *buffer_len  是它的字节数。
                      容量不够时内核会 _aligned_free(*buffer) 再重新分配，
                      并把新指针/新长度写回去。
                      **返回之后这块内存归调用方，内核不再持有、也不再释放它。**

同时写清了**为什么**要写这一段（BI 里那个假泄漏的例子），
免得后来人觉得"这是废话注释"给删了。

**零行为变化**（纯注释）。验证：Linux 全量构建无 error，8 个 sample 全 rc=0。

### 顺带一个观察

这 16 个文件是"同一份契约的 16 个副本"，而它们之间的差异（是否 packed、
是否 prepack、是否 nchwc）**从来不体现在参数上** —— 调用方无法从参数签名
看出"这个变体会不会把 `*buffer` 换掉"（实际上它们**都会**）。
这一族将来要重构，第一件事就是把上面那段契约变成**一个**地方，而不是 16 个。


## 新增/变更：附录 BK —— 人脸库解析路径的回归测试（写测试的过程查出 3 条真缺陷）

### 为什么挑它

本报告的威胁模型把**人脸库文件**（.feat / .names）明确列为**不可信输入**，
而 `ZQ_FaceDatabaseCompact::LoadFromFile` 就是它的解析入口 —— **此前零测试覆盖**。
整个 `ZQlibFaceID/` 的 29 个头都在这个状态。

写测试的第一步是"让这个头能独立编译"，而这一撞就撞出了三条。

### 缺陷 ①：ZQ_FaceRecognizerUtils.h 用 std::cout 却没 include <iostream>

    ZQlibFaceID/ZQ_FaceRecognizerUtils.h:219:10: error: 'cout' is not a member of 'std'
      219 |     std::cout << "failed to solve
";

它只 include 了 opencv2 的三个头 + time.h + omp.h，`std::cout` 完全是
**靠 OpenCV 的头传递带进来**的。换个 OpenCV 版本或 include 顺序就直接编不过。
与附录 AG 修的那批 ZQlib「头不自足」同一类。已补。

### 缺陷 ②③：GenerateRandomDatabase 里的两个问题

    int* tmp_person_face_num = (int*)malloc(sizeof(int)*num_person);   // 不判 NULL
    for (int i = 0; i < num_person; i++) tmp_person_face_num[i] = ...;  // 直接写
    __int64* tmp_person_face_offset = (__int64*)malloc(...);            // 同上
    ...
    __int64 num_all_feats = num_person * num_feat_per_person;            // 两个 int 相乘！

**②** 两个 malloc 都不判 NULL，紧接着就写。旁边的 `tmp_all_feats` 反而判了 ——
同一段代码里两种写法。

**③** 那个 `__int64` 是**装饰性的**：乘积在两个 int 相乘时就已经回绕，事后加宽没用。
同一文件的 `_load_feats` 用的 `total_face_num` 才是真的 `__int64` 累加 ——
**同一个类里的两处，一处对一处错**。

后果不是越界（`needed_bytes` 和后面两个写入循环按同一个回绕值走，自洽），
而是**静默申请到错误大小的库**：请求 1e10 个特征会拿到 1.4e9 个。

### 顺带：三处 printf/sprintf 的格式符与实参宽度不匹配

| 行 | 原写法 | 实参类型 |
|---|---|---|
| 79 | `sprintf(buf, "%d", i)` | `__int64` |
| 123 | `printf("need %d MB 
", needed_bytes/1024/1024)` | `__int64` |
| 126 | `printf("...need %ld bytes
", needed_bytes)` | `__int64` |

x86-64 上 varargs 传 64 位、`%d` 只读低 32 位 —— 数值"恰好对"，但属于未定义行为。
与附录 AT.10 的 ZQ_Huffman.h 同一类。已改成 `%lld` + 显式强转。

### 结论：**加载路径本身是干净的**

13 个用例（2 正常 + 11 畸形）全过，无 ASan 报错、无泄漏：
空文件 / 文件不存在 / dim=0 / dim=-5 / person_num=0 / person_num=-1 /
某人脸数为 0 / 为负 / 特征区被截断 / names 人数不一致 / 人脸数累加 int 回绕，
全部**干净返回 false**。

也就是说 `_load_feats` 里的 `__int64` 用法是**对的**（对比 BK.3③ 的
`GenerateRandomDatabase` 是错的）。**要修的是生成库的那条路，不是解析库的那条路** ——
这个区别只看代码看不出来，得两个都测。

### 我自己的测试也挂过一次

最后一个用例（回绕）第一版直接复用了 `make_feats`，于是它自己要去
`std::vector` 里塞 `8 维 x 0x7FFFFFFE` 个 float（约 17 GB）—— **挂的是测试**。
改成只写头部 + 16 个 float 之后正常。

以及：`dim/person_num/total_face_num` 都是 private、没有公开取值接口，
只能断言 `LoadFromFile` 的**返回值**。想更严就得加 getter，**本轮不加**
（那是 API 变更，不是审计）。

### 接入

归到 `--with-slow`（要 OpenCV 头，ASan 下编一次约 3 分钟），
OpenCV 路径从 `build_x64/CMakeCache.txt` 取，取不到就整体跳过并说明原因
（与 probe_zqlib_headers_msvc.py 对那 14 个 C1083 的处理同一思路）。
日常回归仍是 14 组 ASan 测试（~90 秒）。

### 顺带记一个编译器提醒

`ZQ_FaceDatabaseCompact.h:287` 的 `fgets(line, 199, in)` 忽略返回值
（`-Wunused-result`）。判断用的是下一行的 `line[0] == '\0'`，**功能上是对的**，
但 ferror 时会误判成 EOF。本轮未改（改动会引入新分支），记在这里。

## 新增/变更：附录 BL —— 人脸库**分析路径**查出 4 条真缺陷（含 1 条堆越界写）

### 变更文件
- `ZQlibFaceID/ZQ_FaceDatabase.h`
  - 新增 `_check_analyzable()`：四个分析入口统一走它（BL.1）
  - 新增 `_check_pivot_square_size()`：算 `cur_num*cur_num` **之前**查（BL.2）
  - 新增 `_fseek64()`：offset 是 `__int64`，Windows 的 `fseek` 只收 32 位（BL.3）
  - `_export_similarity_for_all_pairs`：校验提到两个 `fopen` 之前；单线程分支
    补上 `same_pair_num` / `notsame_pair_num`；并行分支改成"先算每人各占多少字节，
    各自 fseek 到自己那一段写"（BL.1 / BL.3 / BL.5）
  - `_select_subset` / `_detect_lowest_pair` / `_detect_repeat_person`：
    并行分支从 `omp critical` + `push_back` 改成"按 p/i 分槽位"（BL.2 / BL.3）
  - `_find_the_best_matches`：四路输出要么一起推要么一起不推（BL.4）
- `ZQlibFaceID/ZQ_FaceDatabaseCompact.h`
  - 同样的 `_fseek64()` / `_check_pivot_square_size()`，两处 `cur_num^2` 矩阵
  - `_export_similarity_for_all_pairs` / `_detect_repeat_person`：
    同样的确定性写出；补 `person_num<=0 || dim<=0` 检查（BL.2 / BL.3 / BL.7）
  - `GenerateRandomDatabase`：`_aligned_malloc` 失败那条 `return` 补上两个 `free`（BL.7）
- `tools/zq_facedb2_check.cpp`：新测试，6 个用例（ASan + LeakSanitizer）
- `tools/run_zqlib_checks.py`：登记 `zq_facedb2`（`--with-slow`）
- `audit_k3_20261001.md`：新增**附录 BL**

### 4 条真缺陷

| 编号 | 缺陷 | 严重度 | 证据 |
|---|---|---|---|
| BL.1 | 空库上直接 `persons[0].features[0].length` | SEGV / heap-use-after-free | ASan |
| BL.2 | `cur_num*cur_num` int 回绕，65536 时回绕成 0 | **堆越界写** | ASan |
| BL.3 | 并行分支用 `critical` 收集，产物逐次不同 | 不可复现 | 8 线程 x 8 次逐字节比对 |
| BL.4 | `Search` 四路输出长度可以不等 | 调用方越界 | 长度断言 |
| BL.5 | 单线程分支不写 `same/notsame_pair_num` | 调用方拿到初值 | 计数断言 |

BL.2 的实测（修之前）：

```
==153549==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000000
WRITE memory access
    #0 ZQ::ZQ_FaceDatabase::_select_subset(...) ZQ_FaceDatabase.h:815
    #1 ZQ::ZQ_FaceDatabase::SelectSubset(...) ZQ_FaceDatabase.h:45
```

815 行就是 `scores[i*cur_num + i] = 1;` —— 加载器允许单人最多 1e7 个特征，
`65536*65536` 在 int 里正好回绕成 0，`vector<float>(0)` 分配"成功"，紧接着这一行
就是 4 字节越界写。

### BL.5 是测试算出来的，不是读代码看出来的

我第一版把同人对写成 `4*6+3*3+5*10+2*1+6*15 = 175`（真值 `C(4,2)+C(3,2)+C(5,2)+C(2,2)+C(6,2) = 35`），
断言失败后差点去改库。**是同一条测试里"4 线程与单线程一致"那一行过了**才让我
判清楚：库给的是 35，两边一致，所以是**我算错了**。
（`all_pair_num` 我也写错过一次：20 张脸是 C(20,2)=190，我写成了 78。）

### 我的测试自己挂过两次，都记下来

1. **生成器写错了布局**。`make_feats_plain` 第一版把所有人的 `feat_num` 全写在
   前面再写所有特征 —— 那是 **compact** 的布局；非 compact 是"每个人的 `feat_num`
   紧跟在他自己的特征前面"。结果用例 1/5 全挂，而**用例 2（畸形输入）全绿** ——
   看着全绿的测试比没有测试更危险，它验的是"随便什么文件都被拒绝"。
   修法：用例 2 开头加一条**基线断言**（同一生成器的干净文件必须能加载）。
2. **人数太少导致假绿**。第一版确定性用例只有 40 个人，而
   `DetectLowestPair` / `ExportSimilarityForAllPairs` 的 `chunk_size` 是 100
   —— 40 个人只有一个 chunk，`parallel for` 把整个 chunk 交给同一个线程，
   顺序当然是确定的。**"人数少时看着正常，人数一多就不可复现"**，
   这比"一直不确定"更难发现。改成 300 个人后两个入口才真的红。

### 工具教训

* **ASan abort 时 stdout 是块缓冲的，崩溃前的输出全丢**。第一版探针崩溃后
  只看到一段 ASan 报告，前面所有 `ok` 行都不见了 —— 而那些行正是要看的证据。
  现在 `main` 开头加 `setvbuf(stdout, NULL, _IONBF, 0)`。
* **同名模式的两份拷贝要一起测**。`ZQ_FaceDatabaseCompact` 里那个
  `cur_num*cur_num` 我第一版只改了 `only_pivot=false` 那条分支，
  是新加的用例 6 抓到 `only_pivot=true`（**默认值**）那条还没改。
  与附录 BG 栽的跟头同型（NCHW 修了 NCHWC 忘了）。

### 接入

`zq_facedb2` 归到 `--with-slow`（与 `zq_facedb` 同一套 OpenCV 头探测，
ASan 下编一次约 3 分钟）。日常回归仍是 14 组 ASan 测试（~90 秒）。

注意用例 4 里**故意不验** `ExportSimilarityForAllPairs`：它不建 `cur_num^2` 矩阵
所以确实不受 BL.2 影响，但它是 O(N^2) —— 65536 张脸就是 2.1e9 次点积，
ASan 下要跑好几分钟，不适合进门禁；那条路在用例 1 里已经验过了。

### 实测结果

```
[case 1] 正常库 15 项（含 4 线程 flag 文件与单线程逐字节相同、save/load 往返）  全 ok
[case 2] 畸形文件 9 条 + 基线断言 1 条                                          全 ok
[case 3] 空库 4 入口 x 2（默认构造 / Clear 之后）                              全 ok
[case 4] 65536 x 1 维：非 compact 2 入口 + compact 1 入口                      全 ok
[case 5] 300 人：8 线程 x 8 次 x 5 份产物 + 8 线程 vs 单线程                    全 ok
        all=404550 same=900 notsame=403650
[case 6] compact 库：BL.2 + BL.3 + BL.7                                         全 ok
PASSED
```

编译告警只有一条**既有**的：`ZQ_FaceDatabase.h:447` / `ZQ_FaceDatabaseCompact.h:293`
的 `fgets` 忽略返回值（`-Wunused-result`，附录 BK.8 已记，本轮未改）。

## 新增/变更：附录 BM —— 两处丢弃返回值的 `cv::invert`（退化输入下空指针解引用）

### 变更文件
- `ZQlibFaceID/ZQ_FaceRecognizerUtils.h`
  - `_findNonreflectiveSimilarity`：`void` -> `bool`（原来 `cv::solve` 失败后
    只 printf 一句就 return，transform 留在**空 cv::Mat**，调用方无从检查）
  - `_findSimilarity`：`void` -> `bool`，检查两次 `_findNonreflectiveSimilarity`
    和 `cv::invert` 的返回值
  - `CropImage_112x96 / _112x112 / _160x160 / _256x256_dot85`：
    失败往上抛，而不是无条件 `return true`
- `tools/check_uncked_cv_return.py`：新门禁
- `tools/run_audit_checks.py`：接进 A11 / A12 两组
- `audit_k3_20261001.md`：新增**附录 BM**

### 缺陷

```cpp
// 修之前 ZQlibFaceID/ZQ_FaceRecognizerUtils.h:423-426
cv::Mat tmp;
if (norm1 < norm2)
    cv::invert(transform1, tmp, cv::DECOMP_SVD);   // 返回值丢掉
else
    cv::invert(transform2, tmp, cv::DECOMP_SVD);
...
trans.ptr<TmpType>(i)[j] = tmp.ptr<TmpType>(j)[i];  // 空 Mat -> 空指针
```

`cv::invert` 对**奇异**矩阵返回 `false` 并把 dst 留成**空 `cv::Mat`**。
`Tinv = [sc -ss 0; ss sc 0; tx ty 1]` 恰好奇异的条件是 `sc*sc+ss*ss == 0`,
也就是相似变换退化成零尺度 —— 5 个输入点退化（全 0 / 共线 / 有重复）时正是这样。

第二层：四个 `CropImage_*` 原来**无条件** `return true`，所以调用方
`ZQ_FaceRecognizerSphereFace::AlignAndCropFeature` 里那个
`if (!CropImage_xxx(...)) return false;` **永远不成立** ——
"点退化"这件事在整个调用链上没有任何一层能知道。

### 可达性（如实说，没那么严重）

两个自带检测器都填 `ppoint`（`ZQ_CNN_MTCNN.h:1302`、
`ZQ_FaceDetectorLibFaceDetect.h:196`），所以正常图像上不会退化。
`ZQ_CNN_BBox` 构造函数是 `memset(this, 0, ...)`，**没填就是全 0**，
而 `ZQ_FaceDatabaseMaker.h:557` 直接从 BBox 取 5 点
—— 接一个不填 `ppoint` 的第三方检测器就退化。
另外 `AlignAndCropFeature` 的 `face5point_x/y` 是**调用方给的裸指针**，
API 上没有任何前置条件。所以定 **MED**（API 契约缺陷），不是 HIGH。

### 没有 ASan 运行时证据 —— 如实记下来

本机 `3rdparty/opencv` 只有 Windows 的 `opencv_world342.lib`，**没有 Linux 的 `.so`**，
所以这一条做不了运行时验证。它是**读代码 + 与同仓库那份已经加固的拷贝对照**定位的：
`ZQCNN/ZQ_CNN_FaceCropUtils.h:88` 是同一段算法的另一份拷贝，
**本来就返回 bool 且调用方检查了** —— 同一段算法两份拷贝，OpenCV 这份把加固丢了。
不假装有运行时证据。

### 门禁 `check_uncked_cv_return.py`

只收"返回值是 bool 且失败时会把 OutputArray 留成空"的
`cv::solve` / `cv::invert` / `cv::gemm`，
判定"这一行以 `cv::xxx(` 开头"（把返回值整个丢掉当独立语句用）。

**这个门禁自己怎么被验证的**（一个从不匹配的正则 + 零命中 = 永远绿的门禁，
比没有门禁更坏）：

* 正则层 11 条构造样例
* **遍历层**：真的往被扫的目录写一个临时 `.cpp`，跑完整 `find_hits`，
  确认 2 处 -> 2 处，且临时文件已清掉
* 再拿修前状态（`git stash`）验过：报 2 条、exit=1；`git stash pop` 后
  报 0 条、exit=0

KNOWN_GAPS 写进了工具 docstring：只认"本行以 cv::xxx( 开头"这一种形状；
不判断检查得对不对；只覆盖 `cv::` 限定写法；没有运行时证据。

### 实测结果

```
A11 check_uncked_cv_return --selfcheck: OK
    selfcheck: 11 条判定全部符合预期
    selfcheck: 遍历层也通过（临时文件 2 处 -> find_hits 2 处，跑完已清理）
A12 check_uncked_cv_return: OK
    OK: 全部 cv::solve/invert/gemm 调用的返回值都被用上了（扫了 0 处）
--quick 全量: ALL CHECKS PASSED
```

### 全树只有这 2 处

一句 grep 就能确认：
`^\s*cv::(invert|solve|gemm)\s*\(` 整个仓库只命中 `ZQ_FaceRecognizerUtils.h`
的 407 / 409 两行。所以这一条是"形状很典型、量很少"，修完顺手做成门禁就够。

## 新增/变更：附录 BN —— NCHWC innerproduct 整族（21 个变体）首次覆盖：1 条静默错算 + 1 条 dispatcher 崩溃

### 变更文件
- `ZQCNN/layers_nchwc/zq_cnn_innerproduct_gemm_nchwc_raw.h`
  - noborders 的形参 `out_sliceStep` -> `out_imStep`（**BN.2**）
- `ZQCNN/layers_nchwc/zq_cnn_innerproduct_gemm_nchwc.h`
  - 9 个 noborders 声明的最后一个参数改名
- `ZQCNN/ZQ_CNN_Forward_SSEUtils_NCHWC.cpp`
  - 12 个 noborders 调用点改传 `out_imStep`（**BN.2**）
- `tools/zq_nchwc_ip_check.cpp`：新测试，21 个内核变体 x 14 组形状 x 2 种 buffer 模式
- `tools/zq_gemm_shape_check.cpp`：新测试，dispatcher 的 (M,N,K) 形状表
- `tools/run_zqlib_checks.py`：两个测试登记在 SKIP 里，理由逐条写明
- `audit_k3_20261001.md`：新增**附录 BN**

### BN.2【已修】noborders 用 slice 步长当 image 步长 -> N>1 时结果互相覆盖

`zq_cnn_innerproduct_nchwc{1,4,8}_noborder*` 用 `out_sliceStep` 跳下一张图，
但 out 是 `[N,1,1,K]` 的 NCHWC 张量：

```
widthStep = 1*align,  sliceStep = widthStep*realH = align,  imStep = ceil(K/align)*sliceStep
```

N>1 时相邻两张图的 K 个结果被写到相隔 `align` 个 float 的地方，**互相覆盖，
且缓冲区尾部根本没被写**。同文件 `general`（im2col+GEMM）那条路用的是
`imStep`（ldc=filter_N），是对的。12 个调用点**全部**传错。

实测（N=2, C=8, K=4, align=1）：

```
参考:      0.455   0.201  -0.512   0.352  -0.930  -0.446   0.474  -0.094
general:   完全正确
noborders: 0.455  -0.930  -0.446   0.474  -0.930  -0.446   0.474  -0.094
```

`[0]` 段的 k=1..3 被 `[1]` 段的 k=0..2 覆盖；`[1]` 的 k=1..3 保持初值。

**为什么 sample 回归一直绿**：所有 sample 都一次一张图（`out_N == 1`），
此时 `n*1 + k` 恰好就是唯一那张图。**batch > 1 就会踩到。**

修法：12 个调用点改传 `out_imStep`；同时把 9 个声明 + raw 头里的形参**改名**
（`general` 的同名参数不能改，它确实是 slice 步长）。改名不是为了好看 ——
参数名写着 sliceStep 而实际要 imStep，就是这次缺陷的成因。

### BN.3【未修】dispatcher 在 K=27 / K=108 且 M>=2 时崩溃

新测试跑到第 61 个用例（`align=1 general, N=2 H=3 W=3 C=12 K=7`
-> sgemm 的 M=2, N=7, K=108）就 SEGV。把参数直接喂给
`zq_gemm_32f_AnoTrans_Btrans_auto` 整张表：

```
K=27  : M=1 全过；M=2..8 基本全崩
K=108 : M=1 全过；M=2..8 基本全崩
K=512 : M=1..8 x N=1..9 全过
K=3136: 全过
M,N >= 16（生产条件那一侧）: 0/16 通过
共 304 个用例; 崩溃 134, 结果错 0
```

**既有测试为什么没抓到**：`zq_innerproduct_check.cpp` 直接调
`zq_cnn_innerproduct_gemm_32f_align128bit_same_pixstep_batch`，
**根本不经过 auto dispatcher**；而 NCHW 生产路径有 `out_N >= 16 && filter_N >= 16`
守卫（附录 BC 查出来的）。两者叠加 = dispatcher 在 M<16 / N<16 一次没测过，
连 M,N>=16 那区也只在直接调具体内核时测过。

**波及面**：dispatcher 是这 16 个调用点共用的入口 ——
NCHW conv(4) / deconv(4) / innerproduct(4) + NCHWC conv(4) / innerproduct(4)。
**K=27 就是 3x3x3，即每个 CNN 的 RGB 首层** => **batch>1 的 3x3x3 卷积就崩**。
sample 全绿是因为它们都 out_N == 1。

**这一轮不修**：要动 `zq_gemm_32f_align_c_raw.h`（1.6 万行）里 **~20 个内核族**
的 M 尾处理（`M2_N4_Kgeneral` 的主循环是 `for (m = 0; m < M - 1; m += 2)`
**根本没有 M 尾循环**）。这是本仓库最吃性能的一段代码，而用户明确要求 GEMM
性能对标 MKL —— 没有配套性能回归就改它，风险远大于收益。
下一片单独做：先加 dispatcher 入口的**保守**守卫（不支持的形状挡在选内核之前，
走正确的标量兜底），再谈逐族补尾处理。

**不假装已修**：两个测试都进 `run_zqlib_checks.py` 的 SKIP，理由写明
「已定位未修 / 部分已修」，并写"修好之后把这一条删掉即可"。

### 我的测试错了两次（都记下来）

1. **`uses_bias` 表写错**。三个 prelu 变体真名是 `**_with_bias_prelu`**，
   它们**要** bias。我把 prelu 单列成一个变体漏了 bias，于是
   `general+prelu N=1 C=8 K=1` 一上来就红。判清楚靠一个 **N=1 最小探针**：
   `general`（im2col+GEMM）与 `noborder`（逐元素 FMA）**两个完全不同的实现
   同时算对**，才说明是我这边错 —— 附录 BI 的教训用上了。
2. **noborders 生产条件抄错**。align=1/4/8 三处条件**不一样**，
   filter 那一项是 `align*in_W == filter_widthStep`，我第一版漏了 `align*`。
   **照抄的时候要连前面的系数一起抄。**

### 两条工具教训

* **崩溃会吃掉整张表**：ASan 碰到 SEGV 直接 abort，304 个用例只跑到第 12 个。
  改成**每个用例 fork 一个子进程** + 父进程 `waitpid` 判信号，
  才拿到完整边界图。一崩就停的测试只能告诉你有一个坏了。
* **ASan 默认是 `exit(1)` 不是 abort** —— "崩溃"与"算错"在 waitpid 看来都是
  exit code 1，第一版混在一起报了"134 个 FAIL"。要靠
  `__asan_default_options(){ return "abort_on_error=1"; }`（在 ASan 初始化
  **之前**被调用）才能分开。另外子进程 stderr 要接 `/dev/null`，否则 ASan
  的报告会把父进程 stdout 上的网格拦腰截断。

## 新增/变更：附录 BO —— dispatcher 的形状安全图（3240 个用例）+ 修正判据

### 变更文件
- `tools/zq_gemm_shape_check.cpp`：从 304 个用例扩到 **3240 个**（M 12 档 x N 15 档 x K 18 档），
  **判据从相对误差改成后向误差**，每 K 一张 M x N 网格
- `tools/run_zqlib_checks.py`：`SKIP['zq_gemm_shape']` 的理由按实测结果重写
- `audit_k3_20261001.md`：新增**附录 BO**

### 为什么要画这张图

BN.3 定位到 dispatcher 在 K=27/K=108 且 M>=2 时崩溃，但只测了 M<=8/N<=9/4 个 K。
**那点数据不足以写出一道保守的守卫** —— 守卫只能拦"现在就是坏的"，
而"哪些形状是好的"必须测出来，不能靠读 1.6 万行内核去推。

### 根因（实测坐实）

精确大小的缓冲区 + `M=2 N=4 K=25`：

```
#0 _mm_load_ps                                            xmmintrin.h:927
#1 zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N4_Kgeneral  zq_gemm_32f_align_c_raw.h:3161
#2 zq_gemm_32f_align128bit_AnoTrans_Btrans_M2_N4           zq_gemm_32f_align_c_raw.h:16376
#3 zq_gemm_32f_AnoTrans_Btrans_auto                       zq_gemm_32f_auto.c:574
```

三件事凑在一起：
1. `lda = ldb = K`，而 K 不必是向量宽度的整数倍 —— 第 m 行基址 `A + m*lda` 不对齐
2. k 方向循环全程用 `zq_mm_load_ps`（= 对齐的 `movaps`）——
   raw 头里 **250 次** `zq_mm_load_ps`、**0 次** `zq_mm_loadu_ps`
   （那个非对齐宏就在旁边定义着，一个都没用）
3. 循环上界是 `padK` 而不是 `K`；紧跟的标量收尾 `for (; k < K; k++)`
   在 `K < padK` 时一次都不执行

**同仓库的另一份实现就是反证**：汇编版 `vmovups` 9 次、`vmovaps` **0 次**，
全程非对齐载入，所以对任意 lda/ldb 都安全。同一份数学，一份假设了对齐、一份没假设。

### 3240 个格子的结论

```
K      K%8   ok    崩溃   结果错   special-case?
16     0     180   0      0        是
17     1     24    156    0        否
24     0     180   0      0        是
27     3     24    156    0        **是**（照样崩）
28     4     180   0      0        是
32/64/72/128/144/256/512/1024  0  180  0  是
33     1     24    156    0        否
108    4     22    158    0        否
112    0     180   0      0        否
150    6     22    158    0        否
3136   0     180   0      0        否
合计: ok 2456, 崩溃 784, 结果错 0
```

**两个反例把规则钉死**：
* **K=28 安全，尽管 28%8=4** —— 28 == `zq_mm_align_size7`，
  `M2_N4` 有一支 `KeqAlign7` 专用内核，尾巴按 7 个元素处理**恰好 K 个**，不依赖对齐。
* **K=27 会崩，尽管 dispatcher 里有 `K == 27` 的分支** —— 那个分支把 K=27
  交给 `M2_N4`，而 `M2_N4` 自己的 K 阶梯里没有 27 这一档，落到 `*_Kgeneral`。
  **dispatcher 有 K=27 的分支 != 选中的内核族能处理 K=27。**

判别式：**K % 8 == 0 一定安全**（10 个 K、1800 个格子、零崩溃，其中 112/3136
连 special-case 都没有）；**K % 8 != 0 就崩**，24 与 28 靠专用内核幸免。
**全程只有崩溃、零个"结果错"** —— 这是个响的失败，不是静默的正确性危害。

### 顺带查出：既有门禁的形状表整个落在安全区

`tools/zq_gemm_oob_check.c` 测的 10 个形状（{2,4,64}…{7,4,40}），
**K 全部是 8 的倍数、N 全部是 4 的倍数** —— 结构上不可能发现这条缺陷。
这就是"测试跑了很多次全绿"的真实含义。

### 判据：我错了第三次（这一条最值钱）

第一版用 `|got-exp| / max(|exp|,1)`、阈值 2e-4，报了两处"结果错"：

```
M=1024 N=64 K=1024 max_rel=5.5e-02 @ (m=441,n=11) got=0.000046 exp=0.000043
```

**是我的判据错了。** 1024 个量级 ~0.29 的乘积求和，中间值在 ±9；
某个 (m,n) 上抵消到 4.6e-5，float32 只有 7.2 位有效数字，抵消 5 个数量级之后
**绝对**误差自然还有 1e-6 量级 —— 除以 4.6e-5 就是 12%。**这是 float32 的固有性质。**

换成后向误差 `|got-exp| / (||A_m||_2 * ||B_n||_2)` 之后：
K=16 那一整张 180 个格子**全过**（第一版报的 M>=20/N>=24 一片 `x` 全是假的），
K=1024 全过，而 K%8!=0 那一片**仍然是崩溃**（崩溃与判据无关，BN.3 不受影响）。

**又一次差点把假缺陷写进报告**（与 BC/BI/BN.4 一脉相承，四次了）。规矩：

> **先确认判据本身是对的，再去解释结果。**
> GEMM 的判据用后向误差，不要用相对误差 —— 后者对抵消敏感，而点积天生就有抵消。

### 修法已经定死（留给下一片）

在 dispatcher 入口加守卫，把 `K % align != 0` 挡到一条正确的兜底路径：

* 上表显示 **`K % 8 == 0` 那一片 1800 个格子现在全是好的**，
  守卫一个都碰不到 —— **性能对标 MKL 的那一片形状一个都不动**
* 被改道的那一片现在**全是崩溃**，不存在"原来更快"
* 这是一个**可证的无性能回退**的修改。BN.3 当时不敢动，就是因为不知道哪些形状是好的

注意守卫要写成用**各编译块自己的 align 常量**（AVX 块 8 / SSE 块 4），
不能写死 8，否则只编到 SSE 的配置下会拦掉一批本来安全的形状。

顺带：汇编 dispatcher `zq_gemm_32f_AnoTrans_Btrans_auto_asm`
**已经有完整的入口守卫**（`M<=0||N<=0` 直接返回、N<4 走专用核、K<8 走打包路径、
ndir 内部还会查形状后返回 0 让上层换路）。也就是**修法的样板就在同一个仓库里**。

## 新增/变更：附录 BP —— BN.3 / BN.4 修掉：3240 个格子全绿 + 新查到一条 bias 被数了 N 次

### 变更文件
- `ZQ_GEMM/math/zq_gemm_32f_auto.c`
  - 新增 `zq_gemm_32f_AnoTrans_Btrans_fallback()`（朴素三重循环，double 累加，覆盖写 C）
  - x86 的 `zq_gemm_32f_AnoTrans_Btrans_auto` 入口加守卫：`K % ZQ_GEMM_K_ALIGN != 0`
    -> 走 fallback。`ZQ_GEMM_K_ALIGN` 按档取（AVX 8 / SSE 4），**不写死 8**
- `ZQCNN/layers_nchwc/zq_cnn_innerproduct_gemm_nchwc_raw.h`
  - `noborders` 的 bias 从"预置进 SIMD 累加器的每个 lane"改成"归约后加一次"（BN.4）
- `tools/zq_gemm_shape_check.cpp`：加 14 个 production 形状（M 到 3136 / N 到 512）
- `tools/zq_nchwc_ip_check.cpp`：判据换后向误差；收窄到**生产真的会走的配置**；
  修 buffer 所有权（附录 BJ 那条约定）
- `tools/run_zqlib_checks.py`：`SKIP` 清空（两个测试都真的绿了）
- `audit_k3_20261001.md`：新增**附录 BP**

### BN.3【已修】dispatcher 入口守卫

守卫放在 `SWAP_A_Bt` **之前**（换了 A/Bt 之后 lda/ldb 互换了），
判的是 K 本身（swap 不改变 K）。兜底路径累加用 double：这条路现在只能被
"原本会崩"的形状走到，正确性优先；真要优化它应该上分块/向量化并配 MKL 对标，
不是先猜着优化。

### 为什么这是可证的无性能回退（实测兑现）

* `K%8==0` 那一片（1800 个格子）现在全是好的，守卫一条都碰不到
* 被改道的那一片现在全是崩溃，不存在"原来更快"
* 补测了 production 尺度的 14 个形状（M 到 3136 / N 到 512）

```
zq_gemm_shape_check:  ok 3254, 崩溃 0, 结果错 0     <-- 修之前 2456 / 784 / 0
zq_nchwc_ip_check:    588/588 PASSED, LSan 干净
   NCHWC1 align=1: 168/168   NCHWC4 align=4: 252/252   NCHWC8 align=8: 168/168
```

### BN.4【已修】noborders 把 bias 数了 align 次

修好守卫之后 `zq_nchwc_ip_check` 终于能跑过第 61 个用例，**当场又查出一条新的**
（"修好一个缺陷解锁了后面的覆盖"）：

```c
sum_vec = zq_mm_set1_ps(bias[out_c]);   // 预置进 SIMD 累加器的每个 lane
...
*out_c_ptr = zq_final_sum_q;            // 把 align 个 lane 全加起来 -> bias 被数 align 次
```

align=1 看不出来，**align=4/8 就是 4 倍 / 8 倍 bias** —— 而 noborders 恰好是
align=4/8 的默认路径。实测（NCHWC4, N=1, C=8, K=1）：
`exp=dot+bias=0.507058`，`got=0.664558=0.454558+4*0.0525`，一分不差。
改成"先归约出点积、再加一次 bias"，align=1 结果完全不变。

### 对齐假设这一族：还有 3 处，已定位、当前模型到不了、未修

| 位置 | 触发条件 | 实测 |
|---|---|---|
| `..._col2im.h:205` `for (kc=0; kc<out_C; kc+=align)` | filter_N 不是 align 倍数 | align=4, K=17 -> SEGV |
| `noborders` 的 `in_hwc += align` + 对齐载入 | H*W*C 不是 align 倍数 | align=4 -> SEGV |
| `packed4` 的 4 路打包 | C 不是 4 的倍数 | C=5 -> 错, max_rel=9.8e-2 |

三处都要两个条件同时成立才到得了。核过 shipped 模型：SphereFace/ArcFace 的
innerproduct 输入是 7*7*512 / 1*1*256 之类、filter_N 是 512/128/10，全是 4/8 倍数；
MTCNN 没有 innerproduct；`ZQ_CNN_Layer_NCHWC_InnerProduct` 把 bottoms 原样透传而
shipped net 的 innerproduct 输入都是 border=0。**既不修、也不当已修**，
测试里对这三种配置直接跳过并把理由写在代码注释里。

### 顺带查到：调用方契约是三条，不是两条

写计时 harness 时用 `std::vector<float>` 当 A/Bt/C，**before/after 都第一次调用就崩**。
256bit 那一族用的是**对齐**载入（`_mm256_load_ps`），glibc malloc 只给 16 字节，
而 **ASan 的分配器给 32** —— 所以 BO 那 3240 个用例在 ASan 下从没暴露过这条。
所以 ZQ_GEMM 的调用方契约是：① lda(=K) 是向量宽度倍数（BP.1 补上守卫）
② 缓冲区 32 字节对齐 ③ C 是覆盖写（beta=0）不是累加。
第 2 条任何地方都没写，生产已经满足，**不需要改代码，但要记下来**。

### 我自己把控制流改坏了一次（第五次"假绿"）

用正则批量给 5 个提前 `return true` 补 `_aligned_free(buffer)` 时，
它把语句插到了 `if` 和它的 `return` **之间**：

```c
if (bw != 0 || bh != 0)
    if (buffer) _aligned_free(buffer);
    return true;      // <-- 现在是无条件的
```

于是 noborders 分支每个用例都直接返回 —— 全部"跳过"全部"通过"，
`NCHWC1 168/168` 看着完美，其实一条 noborders 都没跑。是 LSan 的残留泄漏
把我引到那里的。规矩已进 AGENTS.md：**批量改代码后必须重新看一遍被改那几行的
控制流**；`grep -c` 数出来是 6 看着对，但计数不是证据。

### 没有做的：MKL 性能对标

BP.7 如实记着没量。结构性论证（对 K%align==0 逐字节不变）+ 功能性证据
（3254 格子全绿）都在，但"GEMM 性能相对 MKL 无变化"这句话仍需在
`SampleGEMMAsmCompare` 那条链上跑一轮 A/B 才有数字。
**不把"应该没影响"说成"量过没影响"。**

## 新增/变更：附录 BQ —— 补上 BP.7 欠的那一轮计时（守卫无可测量回退）

### 变更文件
- `tools/zq_gemm_ab_perf.cpp`：新加，A/B 计时驱动
- `tools/ab_time_two_binaries.sh`：输出里**写死**的 A/B 标签改成打印真实路径
- `audit_k3_20261001.md`：新增**附录 BQ**

### 欠的是什么

BP.1 给 dispatcher 入口加了守卫 `K % ZQ_GEMM_K_ALIGN != 0 -> fallback`，
我给的论证是**结构性**的（对 K%align==0 的形状，守卫之后的函数体逐字节不变）
+ 3254 个格子全绿的**功能性**证据。BP.7 如实写着"没量"。

### 怎么量的

* before = `acc6653` 的父提交那份 `zq_gemm_32f_auto.c`；脚本里**断言过**
  里面没有 `ZQ_GEMM_K_ALIGN`，否则直接退出（不然 before 就不是 before 了）
* 两个版本链**同一份** `zq_gemm_32f_align_c.o` / `_asm.o`（那个要编 5 分钟以上，
  只编一次），所以差里只有守卫 + fallback
* 计时用 `tools/ab_time_two_binaries.sh`：ABABAB 交替、9 轮、取中位数；
  该脚本自己记着"本机噪声下限约 7%"
* 形状全部取 `K % 8 == 0` —— 那一片才是"性能对标 MKL 的形状"，守卫碰不到
* 驱动每个形状算 best-of-3，缓冲区 32 字节对齐（BP.8 那条契约），
  并打一个 `chk=` 结果和 —— 全 0 的话"很快"就没有意义

### 结果（before/after，全部落在噪声里）

| 形状 (MxNxK) | after 中位 (ms) | before 中位 (ms) | before/after |
|---|---|---|---|
| 512 x 512 x 512 | 10.188 | 10.129 | 0.994 |
| 1024 x 1024 x 512 | 41.012 | 40.877 | 0.997 |
| 1024 x 1024 x 1024 | 87.629 | 89.678 | 1.023 |
| 2048 x 2048 x 512 | 177.514 | 174.158 | 0.981 |
| 784 x 256 x 288 | 2.327 | 2.402 | 1.032 |
| 3136 x 64 x 144 | 2.277 | 2.269 | 0.997 |
| 512 x 10 x 32 | 0.021 | 0.021 | 1.008 |

全部在 ±3.2% 以内（噪声下限约 7%），而且方向是**混的**（三个 <1、四个 >1）
—— 真有回退会**系统性地** >1。
**守卫对 K%align==0 的形状没有可测量的性能影响。**

### 顺带修了一个会让人把 A/B 读反的工具缺陷

`ab_time_two_binaries.sh` 输出里的标签是**写死**的：
`A (1 || 全走 GEMM)` / `B (守卫恢复)` —— 那是某一轮 A/B 的遗留描述。
脚本是通用工具、被复用过多轮，而标签没跟着改。我这次跑的时候
A 其实是 after、B 是 before，照着标签读会把方向整个读反。
改成直接打印两个二进制的路径。

**通用工具的输出里不要写死某一次的具体描述** —— 复用的人不会去看脚本，
只会看输出。同一条教训的另一个版本是 BP.5（正则改坏控制流）。

### 驱动踩到的两个坑

1. **无参数时 `argc == 1`**。我第一版写了 `if (argc < 2) { usage; return 2; }`，
   而那个脚本是**不带参数**调用二进制的 —— 于是每次只打一行 usage，
   grep 取不到样本，脚本报"没取到样本 A=0 B=0"。看不出来是"没采到"。
2. **gcc 的 `<malloc.h>` 在这台机器上不声明 `_aligned_malloc`**。
   早先几个测试能用是因为它们 include 了 ZQCNN 的头被顺带带进来；
   这个驱动只 include 标准头。改成按平台分流的 `zq_ab_aligned`
   （Windows 用 `_aligned_malloc`，Linux 用 `posix_memalign`）。

### 范围

BP.7 那句"没量"现在可以划掉，但**只限于"守卫有没有引入回退"这一件事**。
"GEMM 相对 MKL 的整体性能"是另一件事，仍以
`reports/ZQ_GEMM_汇编内核性能对比.md` 里那份测量为准，本轮没有重测它。

## 新增/变更：附录 BR —— 「8 个 sample 全绿」其实是 6 个真跑了 + 2 个平台桩

### 变更文件
- `tools/run_sample_regression.sh`：输出不再丢进 /dev/null，改为报告**状态**
- `audit_k3_20261001.md`：新增**附录 BR**

### 怎么发现的

本来是去看 NCHWC 卷积那一族（BO 之后剩下的最大一片未覆盖面），
路上读到 `ZQ_CNN_Forward_SSEUtils_NCHWC.cpp` 的 3x3 分派在 x86 上是
`return false`，于是想验证「SampleMTCNN_NCHWC4 在 Linux 上到底跑不跑得起来」。
去看回归脚本，发现它把输出全丢、**只看退出码**。

### 缺陷：Linux 回归 8 个里有 2 个是「平台桩」

```
=== SampleFaceDetectorMTCNN     rc=0   ./SampleFaceDetectorMTCNN only support windows
=== SampleCascadeOnet_Interface rc=0   not support in linux
```

两个都 `rc=0`。所以历次「Linux sample 8/8 全绿」里，有 2 个在 Linux 上
只是打印了一句话就退出了。**桩本身不是缺陷**（Windows-only 的 sample 在
Linux 上说一声"不支持"是正常行为），缺陷在**回归把桩也记成了通过** ——
它让「双平台都跑通了」这个结论比证据支持的更强。

不是个别现象：全仓扫「平台桩」措辞命中一串
（SampleMatMulNEON / SampleCropImagesForArcFace / SampleEvaluationOnLFW* 等）。

### 修法：报告状态，不只报退出码

```
OK    跑完了、有输出、rc=0          不判失败
STUB  自己说了"这个平台不支持"        不判失败，但**报出来**（并打上那半句）
NOUT  一行输出都没有                判失败
FAIL  rc != 0                      判失败
```

有 NOUT / FAIL 就退出非 0，所以 run_audit_checks.py 的 D3 组会真的红。

修之后：

```
SampleMTCNN              OK    21 行输出
SampleMTCNN_NCHWC4       OK    29 行输出
SampleSSD                OK    10 行输出
SampleFaceDetectorMTCNN  STUB  [./SampleFaceDetectorMTCNN only support windows]
SampleCascadeOnet        OK     2 行输出
SampleCascadeOnet_Interface STUB [not support in linux]
SampleMTCNNLoadFromCode  OK    23 行输出
SampleGEMMAsmCompare     OK    39 行输出
---- 真跑了的 6 个；本平台不支持的桩 2 个；问题 0 个 ----
```

### Windows 侧顺手也查了：6 个都是真跑的

D4 组同样只看退出码。实测 6 个 exe 全是真跑：

| sample | rc | 输出 |
|---|---|---|
| SampleGEMMAsmCompare | 0 | 39 行 |
| SampleMTCNN | 0 | 21 行 |
| SampleMTCNN_NCHWC4 | 0 | 29 行 |
| SampleSSD | 0 | 10 行 |
| SampleCascadeOnet | 0 | 2 行（只打三个模型大小） |
| SampleFaceDetectorMTCNN | 0 | `1000 iters cost 10.385 secs` —— 真跑了一个测速循环 |

所以两个平台的样本清单本来就不一样，现在知道为什么：Linux 那 8 个里混了
两个 Windows-only 的桩。**两边都没有真缺陷** —— 这一条是关于**证据强度**的：
「双平台都跑通了」此前只在 6 个 sample 上有证据，却按 8 个报。

### 顺带记一条更弱的观察（不作为缺陷）

`SampleCascadeOnet` 在两个平台上都只打印三行模型大小，**没有任何检测结果输出**，
所以它的 rc=0 只能证明"模型加载了"。本轮没查它是真跑了检测还是提前返回 ——
**没有证据，不记成缺陷**。

### 状态

`run_sample_regression.sh` 已改。Windows 侧（D4）**仍然只看退出码** ——
本轮查过 6 个都是真跑的所以没改；要改是同一套做法。

## 新增/变更：附录 BS —— NCHWC「no_padding」卷积族（MTCNN 真正走的那一支）

### 变更文件
- `tools/zq_nchwc_conv_check.cpp`：新加，NCHWC 3x3 的 6 个内核 x 9 组形状 x 2 种 buffer
- `tools/run_zqlib_checks.py`：登记 `zq_nchwc_conv`（`--with-slow`）
- `audit_k3_20261001.md`：新增**附录 BS**

### 先纠正我自己的一个判断

上一轮（BR）我说「NCHWC 3x3 卷积在 x86 上是 `return false`，所以 packed 那一族
在 x86 是死代码」。**这句话只对了一半** —— 那个 `return false` 属于 **packed 族**。
真正在 x86 上跑 MTCNN 3x3 卷积的是**另一个族**：

    zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3{,_with_bias,_with_bias_prelu}
    zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_C3{,_with_bias,_with_bias_prelu}

`SampleMTCNN_NCHWC4` 在 Linux 上确实检出 92/88/45/29 张脸（BR.2 记过），
所以走的一定是这一族。**这一族此前零测试覆盖。**

### 覆盖范围与取舍

先只覆盖 NCHWC4 的 3x3 两支 x 3 个激活动作 = 6 个内核。
**不全铺开**是因为这一族每种对齐约 19 个入口 x 3 种对齐，参数契约
（stride/dilation/padding 各自怎么处理）每支都不一样，一次全铺很容易变成
「照着调用点抄参数、但不知道该填几」—— 那是附录 BI 栽过的坑。宁可少而对。
1x1 / 2x2 / general 留给下一片。

布局仍然不用自己推：用真实的 ZQ_CNN_Tensor4D_NCHWC4 类算 stride、
用 ConvertFromCompactNCHW 填普通 [N][C][H][W]。判据用**后向误差**（BO.3 的坑）。

### 我的测试自己写错了一处（ASan 当场抓到）

给输出填哨兵时用了 `n*oIS + oh*oSS + ow*oWS + k` —— 那是 **NCHW** 的算法。
NCHWC4 是 `[n][c/4][h][w][4]`：

    #define OUT_IDX(nn, ohh, oww, kk) \
        ((nn) * oIS + ((kk) / 4) * oSS + (ohh) * oWS + (oww) * 4 + ((kk) % 4))

ASan 报 heap-buffer-overflow（写到缓冲区右边 96 字节）。

值得记的是**为什么同一个简化式在 innerproduct 那个测试里是对的**：
innerproduct 的输出是 [N,1,1,K]，那里 sliceStep 恰好等于 align，
于是 `(k/4)*4 + k%4 == k`。**同一个错法在两个测试里表现完全不同** ——
所以不能靠"上次这么写是对的"来判断。

### 【已定位，未修】filter_N % 4 != 0 会在 col2im 里越界

隔离实验（一次只动一个变量）：

| 组 | C | H=W | stride | K | oW | oW%4 | 结果 |
|---|---|---|---|---|---|---|---|
| A | 3 | 28 | 1 | 12 | 26 | 2 | 通过 |
| B | 3 | 28 | 1 | 10 | 26 | 2 | SEGV (col2im.h:124) |
| C | 3 | 27 | 1 | 10 | 25 | 1 | SEGV (col2im.h:60) |

A 与 B 只差 K；B 与 C 的 K 相同、只把 oW 的余数从 2 换成 1。所以：
* 与 stride / H / W **无关**
* 与 out_W%4 是哪一支 **无关**（B 崩在 ==2 那一支，C 崩在**另一支**）
* 唯一自变量是 **filter_N % 4**

`zq_cnn_convolution_gemm_nchwc_col2im.h` 里每一支都是
`for (kc = 0; kc < out_C; kc += zq_mm_align_size, ...)`（一次处理 4 个 filter），
`out_C` 就是 `filter_N`，不是 4 的倍数时最后一组会多处理 2~3 个 —— 越界读 matrix_C、
越界写输出。与附录 BP.4 第 1 条同族。

**wrapper 从不校验它**：`ZQ_CNN_Forward_SSEUtils_NCHWC::Convolution*` 只查了
`filter_C == in_C` 和 `filter_N == bias_C`（ZQ_CNN_Forward_SSEUtils_NCHWC.cpp:2777），
**没有查 filter_N % 4**。所以输出通道数不是 4 的倍数的模型，在 NCHWC 3x3 卷积上
会**直接段错误**，而不是干净地 return false。

**为什么 MTCNN 现在没事**（实测不是推测）：P-net 的 3x3 卷积输出通道数 10/16/32，
其中 16、32 是 4 的倍数；conv1（10 通道）的输出宽度落在 out_W%4 != 2 的那一支上；
唯一 filter_N=2 的那一层（conv4-1）**是 1x1**，走另一个内核。**三件事凑巧都躲开了。**

**本轮不修**：要动的是 packed 微内核的 col2im 收尾（每支都要补尾巴），
而守卫该加在哪一层、1x1/2x2/general 那几支是不是同样要求 4 的倍数，本轮都**没测**。
加一个只覆盖 3x3 的半截守卫比不加以更坏（让人以为契约已被守住）。留给下一片。

### 实测结果

    共 54 个用例（跳过 0 个形状不合法），PASSED (g_fail = 0)

**内核本身是对的** —— 6 个内核、9 组形状、两种 buffer 模式，逐位落在后向误差
1e-5 以内。找到的是**契约**问题（filter_N % 4），不是算错。

本轮只加了测试与文档，没有动库代码。
  python tools/run_audit_checks.py --quick   ->  ALL CHECKS PASSED
  python tools/run_zqlib_checks.py --with-slow zq_nchwc_conv

## 新增/变更：附录 BT —— NCHWC no_padding 卷积的 filter_N%4 契约图（并更正 BS 的一条结论）

### 变更文件
- `tools/zq_nchwc_conv_check.cpp`：从「只测 3x3 两支」扩成**六支 × 3 个激活动作**
  的契约图；每个用例 fork 一个子进程，崩溃只算该用例失败
- `audit_k3_20261001.md`：新增**附录 BT**

### 为什么要画这张表

BS 只测了 3x3。**守卫该加在哪一层，取决于 1x1 / 2x2 / general 那三支是不是
同样要求 `filter_N` 是 4 的倍数** —— 没测就加守卫，等于拿"看起来修好了"
换"其实只堵了一个门"。四个分支的**参数列表完全相同**，所以一张表铺得开。

### 表（C=8；C3 两支用 C=3，H=W=20，stride=1）

```
kernel       K%4    plain(无bias)     with_bias        with_bias_prelu
general      0      ok    ok           ok    ok        ok    ok
general      2      WRONG WRONG        ok    ok        ok    ok
kernel1x1    0      ok    ok           ok    ok        ok    ok
kernel1x1    2      WRONG WRONG        ok    ok        ok    ok
kernel2x2    0      ok    ok           ok    ok        ok    ok
kernel2x2    2      WRONG WRONG        ok    ok        ok    ok
kernel2x2_C3 0      WRONG WRONG        WRONG WRONG     WRONG WRONG
kernel2x2_C3 2      WRONG WRONG        WRONG WRONG     WRONG WRONG
kernel3x3    0      ok    ok           ok    ok        ok    ok
kernel3x3    2      WRONG WRONG        ok    ok        ok    ok
kernel3x3_C3 0      ok    ok           ok    ok        ok    ok
kernel3x3_C3 2      WRONG WRONG        ok    ok        ok    ok

共 72 个用例：崩溃 0，结果错 22
```

### 结论一：六支契约**统一** —— 都要 filter_N % 4 == 0（并更正 BS 一条结论）

BS 说 `filter_N%4 != 0` 会**越界**（ASan 报 SEGV）。
这张表里同样是 `K%4 != 0`，六支有五支是**结果错、不是崩**。
崩的那次（BS 的 B/C 两组）用的是 **C=3**，而这张表里 C3 两支用 C=3 + **K%4==0**，
反而不崩。

也就是说：**"越界(崩)"与"结果错"是两个不同的触发条件，BS 把它们混成了一条。**
K%4 != 0 -> 越界多处理 2~3 个 -> 多出来的写到 out_C 之外，于是真正该被写的
那几个 channel 可能根本没被写（还留着哨兵 -12345），表现为"结果错"；
只有当越界写同时跨到**未映射的页**时才会 SEGV。BS 看到 SEGV 只是因为那次的
缓冲区布局让越界恰好跨页。

**不变的核心结论仍然成立**：filter_N % 4 != 0 就不对，而 wrapper 从不校验。

### 结论二：kernel2x2_C3 在**所有**配置下都错 —— 本轮不下结论

三种可能分不清：(1) 它真有 bug；(2) 它还有一条我没满足的前置条件（dilation？
某种 filter 打包？）也就是**我违约了**；(3) 它在生产里根本不可达。
按 BC/BI 的规矩，**分不清就不下结论**。这是下一片的第一件事。
（`kernel3x3_C3` 在 K%4==0 下是 ok 的，所以"C3 后缀"本身不等于坏。）

### 守卫该加在哪一层 —— 形状清楚了，但修法待定

K%4!=0 的要求六支统一，所以守卫应该加在
`ZQ_CNN_Forward_SSEUtils_NCHWC::Convolution*`（它现在只查
`filter_C == in_C` 与 `filter_N == bias_C`），而不是逐个内核去补 col2im 的尾巴。

**但本轮仍不落地**，两个理由：
1. kernel2x2_C3 那条还没定性；如果它是"生产不可达"，守卫该不该覆盖 2x2 另说。
2. 补 col2im 的尾巴是更彻底的修法（让内核对任意 out_C 都对），守卫只是
   "把不合法挡在门外"。两者取舍需要先知道"到底有多少模型会用非 4 倍数的
   filter_N" —— 而 shipped 的 SphereFace / ArcFace / MTCNN **全都是 4 的倍数**，
   所以**改 col2im 的收益目前为零、风险不为零**。

**定位完成、形状清楚、修法待定 —— 这比加一个半截守卫诚实。**

### 测试本身又踩了两个已知的坑（都当场抓到）

1. **子进程里把 run_case 跑了两遍**：
   `_exit(run_case(...) == 2 ? 3 : run_case(...))` 两个分支各调一次，
   等于内核跑两遍、只看第二遍的结果。已改成先取返回值。
2. **子进程 stderr 没接 /dev/null**，ASan 报告会把父进程 stdout 上那一格
   拦腰截断（BN.5 第一次踩的就是这个）。

这次**没有**再犯 BS.3 那个错：输出索引用 OUT_IDX
（n*imStep + (k/4)*sliceStep + oh*widthStep + ow*4 + k%4），不是 NCHW 的简化式。

### 补充：门禁里的分类与 SKIP 登记

K%4 != 0 那一档在测试里被标成「**只报告**、不判失败」：门禁要 pin 住的是
**「合法形状必须算对」**，不是「非法形状必须算错」—— 后者会在有人把 col2im 的
尾巴补好之后把门禁变红。唯一让本测试为红的是 kernel2x2_C3 在 K%4==0 那一档，
所以 `zq_nchwc_conv` 登记在 `run_zqlib_checks.py` 的 SKIP 里，理由写明
「已知未修 / 未定性」，定性之后才谈得上移出。

### 补充：kernel2x2_C3 已定性（附录 BU）

BT.4 留的三个可能收敛成两个：它的 `matrix_B_rows` 里硬编码了
`filter_H*filter_W*3`（`zq_cnn_convolution_gemm_nchwc_raw.h:1015`），
即**要求 filter 张量按 3x3 形状存放**，而 wrapper 是按模型文件里的 kernel_size
建张量的（2x2 就是 2x2），于是 matrix_B_rows 与实际布局对不上 -> 结果错。

**生产不可达，已证实**：扫全仓 model/*.zqparams，kernel_size 分布
{1:68, 3:69, 2:9, 7:1}，其中真正的 **2x2 Convolution**（排除 Pooling）只有两处：
  det2:conv3  bottom=pool2  上一层 conv2 -> in_C=16   -> 走 kernel2x2（非 C3）
  det3:conv4  bottom=pool3  上一层 conv3 -> in_C=64   -> 走 kernel2x2（非 C3）
两个的 in_C 都不是 3，所以 shipped 模型里没有任何一层走 `_C3`。

**本轮不改行为**：修法两条都不纯赚 —— 在 wrapper 里对
`filter_C==3 && filter_H==2` 直接 return false，会关掉一个"按 3x3 存放 filter"
的用户可能正在用的功能；去补 matrix_B_rows 支持真正的 2x2 存放，是给一条
没有模型在用的路径加功能，而改的是 im2col 行数计算这种核心算式。
该做的是把前置条件写在 raw 头 `filter_C` 注释旁（现在只有 `// must be in_C`），
但那要先确认"3x3 存放"确实是当初的意图而不是笔误 —— 要读改动历史，
超出本轮范围。

SKIP 理由相应改写：不是"未定性"，而是"生产不可达 / 前置条件未文档化"，
并且**保持红** —— 哪天有人按那个前提去修了它，这个测试会提醒重新评估。

### 补充：kernel2x2_C3 的两个假设都被推翻（附录 BV）

1. **"matrix_B_rows 是复制粘贴写错"** —— 推翻。同一文件里六支有五支写的是
   `filter_H*filter_W*align_C`，只有它不同，看着像漏改；我把它改成一致之后
   **测试结果一点没变**（6 个用例照样全错），改动已回退。
2. **"那个 *3 是要求 filter 按 3x3 存放"（BU.5 的定性）** —— **作废**。
   读它的 im2col（`raw.h:1072-1096`）就明白：每个 filter 写的是
   **2 行 × 3 通道 = 12 个值**，`*3` 是把 C==3 硬编码进去的（函数名就叫 `_C3`），
   `matrix_B_rows = 2*2*3 = 12` 正好，**是对的**。真要 3x3 存放应该是
   `3*3*align_C` 而不是 `2*2*3`。

**新线索（未证实是根因）**：那段 im2col 把 3 个通道**复制**进第 4 个槽位
（`cp_dst_ptr[3] = filter_pix_ptr[0]`），所以只有**输入侧第 4 通道恰好为 0**
时才无害 —— 而生产里这一层的输入是上一层卷积的输出，上一层的 col2im
**不一定**把补齐通道清零。这条解释不了我的测试为什么仍错（我的测试用
`ConvertFromCompactNCHW` 填的，补齐通道确实是 0），所以它是线索不是根因。

**改动：零。** 净产出是两条被推翻的假设 + 一条新线索 + 作废 BU.5 的错误定性。
`zq_nchwc_conv` 保持红、保持在 SKIP，理由改成"根因未定 + 生产不可达"。

### 补充：kernel2x2_C3 又推翻三条假设（附录 BW），改动仍为零

1. **"输入张量的补齐通道参与计算"（BV.4 的线索）** —— 排除。给输入的第 4 个
   （补齐）通道填 0 还是 7.0，**输出签名逐位相同**（相对差 0.000e+00）。
   内核自己从通道 0 构造补齐槽位，压根没读张量的那部分。
2. **"问题出在「2x2 + C=3」本身"** —— 排除。对照支 `kernel2x2`（C=8 时是**对的**）
   强行喂 C=3，后向误差 3.238e-07，**一样是对的**。所以故障特指
   `kernel2x2_C3` 这个函数体。
3. **"补齐槽位应该填 0 而不是通道 0 的副本"** —— 排除**且这个补丁有害**。
   按此把全仓 10 处（含 `kernel3x3_C3` 的 3 处）改成 0 之后，
   `kernel2x2_C3` 一点没变，而 `kernel3x3_C3`（本来是对的）会被改坏。
   **已回退**并确认 `kernel3x3_C3` 回到 ok。
   这一条说明"读代码推出一个不一致"不等于"那就是要改的地方"：
   `kernel3x3_C3` 用同一套展开却是对的，说明"补齐槽位放通道 0"在别的上下文里
   本身成立（很可能是为了让某条 SIMD 路径不读未初始化内存）。

现在已排除：输入侧、filter_N 对齐、通用 vs 特化、补齐槽位取值。
根因**仍未定**，但收窄到 `kernel2x2_C3` 的函数体内、且是它那套手写 im2col
**之后**的 gemm / col2im 部分 —— 因为 `kernel3x3_C3` 的 im2col 写法几乎一样
却是对的，差异一定在后面。

### 补充：kernel2x2_C3 的根因找到了（附录 BX），改动仍为零

第五条假设也推翻了：gemm 的 K（matrix_A_cols=16）与 B 的行距（matrix_B_rows=12）
在 `kernel2x2_C3` 里不相等，而其余五支都相等。改 `matrix_A_cols` 之后
结果一点没变（BV.2 那次改的是 `matrix_B_rows` —— **两个方向都试过，都没用**）。

**根因**：它的 filter im2col **按 3 行 filter 的形状在走**。每个 filter 读
四组 x 三通道 = 12 个槽位（= 3 行 x align 4），但 2x2 的 filter 在 NCHWC4 里
只有 2 行 = 8 个 float：

  cp[0..2]  = 行0 通道0-2            偏移  0
  cp[3..5]  = filter_pix_ptr += align_size -> 偏移 4 = **行1**
  cp[6..8]  = filter_row_ptr += filter_widthStep -> 偏移 8 = **下一个 filter**
  cp[9..11] = 再 +4 -> 下一个 filter 里 +4 的地方

所以第 3、4 组读到的**根本不是本 filter 的数据**。这也解释了 BW.1 那个观测
（输出与输入张量的补齐通道逐位无关）—— 多读的那几组压根不是去读补齐通道，
而是去读了**别的 filter**。

**为什么之前四条假设全落空**：错的是**读哪几个位置**，不是读到什么值；
两侧（输入与 filter）成对地错，所以任何"对称的改动"（两边都改 0、
两边都改成一致）都不解决问题。

修法要按 2 行重写展开并**重定该支的 K 维定义**（matrix_A_cols /
matrix_B_rows / B 缓冲区大小 / col2im 的 K 步进要一起改），而这一支
**没有第二个实现可对照**。加上生产不可达（附录 BU.4 实测），
本轮不改 —— 收益为零、风险不为零。

本轮改动：零（又一次修改尝试被回退）。只更新文档与 SKIP 理由。

### 补充：附录 BY —— x86 上每个 NCHWC 卷积层都在做一次永远不会被读的权重打包

**事实**：`ZQ_CNN_Layer_NCHWC::ConvolutionWithBias/PReLU/...` 里
`packedfilters` 一共只被 4 个地方用（301/336/373/406 行），**四处全在
`#if __ARM_NEON` 里**，每处的 `#else` 都是把 `*filters` 直接传下去。
也就是说 **x86 上 Forward 一次都不读 packedfilters**。

但 `ZQ_CNN_Layer_NCHWC_Convolution::Prepack()`（789 行）原来是无条件执行的，
而 `ZQ_CNN_Net_NCHWC::_prepack()` 对每一层都调一次 Prepack()
（ZQ_CNN_Net_NCHWC.h:1144）。合起来：**x86 上每个卷积层都白白分配并填了一份
完整的 filter 副本，然后永远不读。**

**修法**：给 `Prepack()` 加 `#if __ARM_NEON` 守卫。
**只挡卷积这一类** —— `InnerProduct::Prepack`（2535 行）那一处**不能**挡，
内积的 packed 路径（`packedM4N4_kernel1x1`）在 x86 上是真被用到的
（没有 ARM 守卫），一起挡掉会让 x86 内积失效。

**实测**（SampleMTCNN_NCHWC4 峰值 RSS，3 遍取中位数，A/B 是去守卫/加守卫各重编一次）：

    after （带守卫）    29000 KB
    before（无条件）   29724 KB
    差                  724 KB

**我第一版估的是"≈10 MB × 线程数"，估错了。** 原因在
`ZQ_CNN_Forward_SSEUtils_NCHWC.cpp:1677-1718`：`ConvolutionPrePack`
**只打包两种形状** —— 1x1，以及 3x3 且 C<=4；其它形状直接落空返回。
MTCNN 的 3x3 卷积绝大多数 C>4（conv2 的 C=10、conv3 的 C=16），
**根本没进这个打包**，所以被浪费的只有 1x1 那一层。
规模与"被打包的卷积层数 × 线程数"成正比；1x1 卷积占大头的网络会省得多。

> 方法论：**先估再量，量完要按量的写。** 报告里凡是数字都应该是测出来的。

**附带更正附录 BR 的一句判断**：BR 里写"packed 族在 x86 上只有 1x1 的
packedM4N4 三变体可达"—— **错**。卷积的 packed 族整族是 ARM-only，
x86 可达的只有**内积**的 packed，不是卷积的。

回归：双平台全量构建 + sample 回归 + 告警扫描 + MSVC ASan -> ALL CHECKS PASSED

### 补充：附录 BZ —— NCHWC8（align=8）的带 bias 卷积把 bias 整条丢了

把 zq_nchwc_conv_check 从只测 align=4 扩到 align=4 + align=8
（`SampleLnet106` / `SampleSphereFaceNet` 就是走 NCHWC8 的，所以是生产在跑、
零测试覆盖）。对齐宽度 4->8 意味着布局公式、补齐槽位数、内核名全变，
align=8 那一整族是独立的。

结果：

  NCHWC8  align=8，K%8==0（门禁那一档）
    general / kernel1x1 / kernel2x2 / kernel3x3 / kernel3x3_C3
      plain              全对
      with_bias          全错
      with_bias_prelu    全错

错法由实测钉死（测试在每个失败格多打 got / exp / 差 / (got-exp)/bias）：

    [详细] align=8 general    k=4: got=1.31665814 exp=1.80765820 差=-0.49100007
    [详细] 该 filter 的 bias = +0.49100003   (got-exp)/bias = -1.0000
    [详细] align=8 kernel1x1  k=4: (got-exp)/bias = -1.0000
    [详细] align=8 kernel3x3_C3: (got-exp)/bias = -1.0000

**`(got-exp)/bias = -1.0000` 精确成立**，而参考里 exp = dot + bias，
所以 **got = dot —— bias 根本没被加上**。

**生产影响**：`SampleLnet106` / `SampleSphereFaceNet` 的卷积基本都带 bias，
也就是说 x86 上这两个 sample 算出来的特征**少了 bias 项**。
而**它们不在 `run_sample_regression.sh` 的清单里**，所以回归看不见 ——
与附录 BR 同一个盲区（只跑 rc=0、不比对结果）。

**根因未定位**（附录 BZ.5 给了线索：col2im 的 `out_W%4==0` 那一支每趟只处理
4 个通道 a0..a3，而 `kc` 的步长是 align=8；但实测是"一个都没加上"，
所以要么走的不是这一支、要么 bias 向量载入取错了位置，两种都还没排除）。

**本轮不改**：align=4 那一支现在是**对的**，而 align=8 与它共用同一段 col2im；
根因没定就动，六支一起坏的风险太大。测试保持红、保持 SKIP。

### 撤回：附录 BZ 的两条结论（附录 CA）—— 两个独立复现把它自己推翻了

BZ.2 说「NCHWC8 的 `with_bias`/`with_bias_prelu` 六支全错、bias 根本没被加上」，
并据此推出「`SampleLnet106` / `SampleSphereFaceNet` 的特征在 x86 上少 bias」。
**这三条全部撤回。**

两个**与 `tools/zq_nchwc_conv_check.cpp` 没有一行共用代码**的小程序
（align=8 一个、align=4 一个）跑同一个调用，逐格统计：

    align=8 kernel1x1_with_bias  N=1 H=20 W=20 C=8 K=8  out=20x20
    k     正确   无bias    其它错  未写过
    k=0   400      0          0          0
    ... （k=1..7 同样 400/0/0/0）
    k=0(0,0)    got=0.930794   exp=0.930794
    k=1(7,5)    got=-0.431416   exp=-0.431416

**8 个输出通道 x 400 个格子全部与参考值一致**（align=4 同样 400/400）。
所以错的是**我的门禁**，不是库。

**顺带查出来：改测试本身改坏了 align=4。** 把 align=8 加进来的那次重写
（提交 52c9e9e）同时让 align=4 的带 bias 路径从 6 个失败变成 26 个
（五支的 with_bias/with_bias_prelu 全红）。可疑点是那次把逐支调用改成了
`##ALIGN##` 宏拼接，以及把「最差格」明细挪到**子进程**里打印
（子进程写 stdout 会截断父进程的网格行）。**门禁已回退**到 52c9e9e^，
即 BS/BT/BW 验证过的那一版，顺手修掉一个 printf 多余实参的告警。

**为什么 BZ 会错**（与 BO.3 同类但更值得记）：两处都是**从一大堆数字里挑了
一个**（"最差格"、"最大相对误差"）当结论的证据 —— 一个被挑出来的极值格
不具代表性。而这次**判据是对的**（两个程序用同一个后向误差、同一阈值），
假结论却还在，说明**判据对了不等于测量对了**。

补进 AGENTS.md 的两条：
* 一个"最差格"不能用来概括整体；要下"某条路径整体如何"的结论必须**逐格统计**。
* **改测试之后要拿"独立复现"对一遍**，尤其当改动引入了新的宏/模板间接层 ——
  CA 里 align=4 的 6 -> 26 就是这么发现的；只看"新增的 align=8 报了新东西"
  不会想到它同时把老的那一段也弄坏了。

保留的部分：align=4 六支的契约图（`filter_N % 4 == 0`）、`kernel2x2_C3` 的
根因（BX.2）、以及"这两个 sample 不在 `run_sample_regression.sh` 清单里"
（那是独立观察到的事实，只是不再附带"它们算错了"这个断言）。
**align=8 那一族至今没测过。**
## 新增/变更：附录 CB —— `kernel2x2_C3` 的三处独立缺陷全部修掉，align=1/4/8 逐格全对

### 变更文件

* `ZQCNN/layers_nchwc/zq_cnn_convolution_gemm_nchwc_raw.h`
  —— 只改 `zq_cnn_conv_no_padding_gemm_nchwc_kernel2x2_C3` 这一个函数，三处：
  1. `matrix_A_cols` 由 `filter_H*filter_W*align_C` 改成 C3 的补齐式
     `(filter_H*filter_W*3 + align - 1)/align*align`，并令 `matrix_B_rows = matrix_A_cols`
     （改之前两者不等：align=1 时 4 vs 12、align=4 时 16 vs 12、align=8 时 32 vs 16）
  2. filter 的 im2col 循环头补上漏掉的 `cp_dst_ptr += matrix_B_rows`
     （同文件另外五个函数都有，只有这一支没有）
  3. B 侧与 A 侧各套一层 `if (zq_mm_align_size >= 4) {交错} else {平面}`，
     与 `kernel3x3_C3` 逐行对齐；顺带删掉因此不再被引用的 `align_C`
* `tools/zq_nchwc_conv8_check.cpp`（新增）
  —— align=8 那一族的独立门禁，与 `zq_nchwc_conv_check.cpp` **没有一行共用代码**：
  内核名在调用点写全、走函数指针表（签名写错会编译报错，不做宏拼接）；
  判据是后向误差 + **逐格统计**；每用例 fork 一个子进程。
  另修一处门禁自身的分类 bug：「只报告」档被照常计成失败（见"注意事项"）
* `tools/zq_nchwc_conv_check.cpp`
  —— 修 `ncrash` 重复计数（旧代码先 `ncrash++` 再改成 `info:WRONG`，
  同一个用例进了两个计数器，汇总行凭空多出 12 次"崩溃"）；更新已过时的尾注
* `tools/run_zqlib_checks.py`
  —— `zq_nchwc_conv` **移出 SKIP**；新登记 `zq_nchwc_conv8`
  （`EXTRA_SOURCES` / `EXTRA_LINK` / `EXTRA_INC` / `EXTRA_CXXFLAGS` / `SLOW` 五处）
* `audit_k3_20261001.md` —— 新增附录 CB
* `AGENTS.md` —— 新增三节（见下）
* `docs-changelogs/CHANGELOG_2026-10-02.md`
  —— 顺手清掉一个**裸 NUL 字节**（第 2564 行，正文引 C 代码 `line[0] == '\0'`
  时写成了裸 NUL 而不是反斜杠+0）。它让 `grep` 把整个 changelog 当**二进制**文件，
  `grep -c "^## "` 直接输出 `Binary file ... matches`。清掉之后 46 个章节可正常检索

### 实测结果

判据：后向误差 `|got-exp| / sqrt(sum(a²f²))`，阈值 1e-5，**逐格统计**（不用最差格）。
独立复现程序与两道门禁都**没有共用代码**。

| 对齐 | 分支 | 修之前 | 修之后 |
|---|---|---|---|
| 1 | `kernel2x2_C3` | 对 0 / 错 **2888**（全错），最差 3.295e+00 | 对 **2888** / 错 0，最差 1.266e-07 |
| 4 | `kernel2x2_C3` | 对 0 / 错 **2888**（全错），最差 5.539e+00 | 对 **2888** / 错 0，最差 1.266e-07 |
| 8 | `kernel2x2_C3` | 对 0 / 错 **2888**（全错），最差 4.461e+00 | 对 **2888** / 错 0，最差 2.985e-07 |
| 1/4/8 | `kernel3x3_C3`（对照组，未改动） | 全对 | 全对 |

2888 = 19×19×8，是 `H=W=20`、2×2、stride 1、`C=3`、`K=8` 的**每一个输出元素**，不是抽样。

两道门禁（各自单独编 `conv.o`）：

* `zq_nchwc_conv`（align=4）：`崩溃 0，应对但仍错 0，只报告 12`，退出码 0 —— **移出 SKIP**
* `zq_nchwc_conv8`（align=8）：`全对 60，有错 0，崩溃/搭建失败 0`，退出码 0

A/B（同一套编译参数，只换 `conv.o`）确认无回归：
旧 `全对 50 / 有错 10 / 崩溃 12` → 新 `全对 60 / 有错 0 / 崩溃 12`。

`python tools/run_audit_checks.py --quick` → `ALL CHECKS PASSED`。
`python tools/check_text_encoding.py` → `OK: 649 text files, all strict UTF-8, no U+FFFD`。

### 注意事项

1. **附录 BX 有三条结论要更正**（详见 CB.6）：
   * BX.2「第 3、4 组读到的是**下一个 filter** 的数据」**不对** —— 一个 2×2、C=3 的
     filter 在 NCHWC 里占 2 行，四组读的 `行0像素0 / 行0像素1 / 行1像素0 / 行1像素1`
     **全都在本 filter 内**，12 个槽位对 2×2 恰好完整。真正读串的是缺陷 2（写指针不步进）
   * BX.3 第 5 条「gemm 的 K 与 B 的行距不一致 —— 排除」**不是排除，是被挡住**。
     它确实是缺陷之一，只是当时另外两处独立地让结果全错，改它看不出差别
   * BX.4「要重定该支的 K 维定义 …… col2im 的 K 步进要一起改」**高估了** ——
     查 `zq_cnn_convolution_gemm_nchwc_col2im.h`，它只用 `matrix_B_cols` 和
     `matrix_B_rows % align` 这个判据，**根本不按 K 步进**，一个字都不用改
2. **「改了没变化」不等于假设被推翻**（已进 AGENTS.md）。BX 连着五次"改了没用"，
   本该在那时就优先假设"不止一处缺陷"
3. **「生产不可达所以不改」这条判据打了补丁**（已进 AGENTS.md）。
   新判据是「**附近有没有可逐行对照的正确实现**」+「**是不是内存安全问题**」。
   这次两条都满足（`kernel3x3_C3` 就在同一个文件里；缺陷 1 是越界读），
   所以 BX.4 那个"不修"的理由不成立
4. **契约之外的调用是段错误，不是垃圾值**（已进 AGENTS.md）。
   `filter_N % align == 0` 被违反时，`plain` 变体直接 SIGSEGV（实测 exit 139），
   `with_bias` / `with_bias_prelu` 只是算错。库**不做任何参数校验**。
   所以门禁里"故意违约"那一档必须标成「只报告、不判失败」，**而且判定代码要真的读那个标记**
   —— `zq_nchwc_conv8_check` 自己标了"只报告"却在判定处照常 `g_crash++`，
   凭空多出 12 个"崩溃"；`zq_nchwc_conv_check` 则是先 `ncrash++` 再改成 `info:WRONG`。
   **两处都是门禁自己的 bug，不是被测代码的**（已用改动前的 `conv.o` 做 A/B 确认）
5. **NCHWC1 与 NCHWC4/8 的布局不同**（本轮新查清，可能是以后还会踩的坑）：
   C=3 时 **NCHWC1 是平面布局**（通道相隔 `H*W`，`widthStep=W`、`sliceStep=H*W`），
   NCHWC4/8 是**交错布局**（通道相邻，`widthStep=align*W*C_pad`）。
   凡是"读连续三个 float 当作一个像素的 3 个通道"的手写展开，
   **只对 NCHWC4/8 成立**，NCHWC1 必须读 `[in_sliceStep]` / `[in_sliceStep2]`。
   证据：把 `ConvertFromCompactNCHW` 之后的内存落点打出来（`in[i]=i`），
   NCHWC1 的 `p[0],p[1],p[2]` 是 0,1,2（同一通道的三个相邻像素），
   NCHWC4/8 的 `p[0],p[1],p[2]` 是 0,16,32（同一像素的三个通道）
实测补记：ASan 实证了缺陷 1 的越界读
--------------------------------------
缺陷 2（`matrix_A_cols` 用错公式）不是"算错"，是**越界读**。用全部 TU 都带
`-fsanitize=address` 重编之后（`buffer=0` 走三次独立 `memalign`，越界才落红区）：

```
==186679==ERROR: AddressSanitizer: heap-buffer-overflow
READ of size 32 at 0x616000000280 thread T0
    #1 zq_gemm_32f_align256bit_AnoTrans_Btrans_M2_caseNdiv4_Keq32  zq_gemm_32f_align_c_raw.h:8617
    #3 zq_gemm_32f_AnoTrans_Btrans_auto                             zq_gemm_32f_auto.c:565
    #4 zq_cnn_conv_no_padding_gemm_nchwc8_kernel2x2_C3
                                    zq_cnn_convolution_gemm_nchwc_raw.h:1147
0x616000000280 is located 0 bytes to the right of 512-byte region
allocated by ... zq_cnn_convolution_gemm_nchwc_raw.h:1056
```

那个 512 字节的块正是 `matrix_Bt`（`matrix_B_rows=16` x `filter_N=8` x 4 字节），
分配点在 1056 行、越界读在 1147 行的 gemm 调用里，与"align=8 时 gemm 按 ldb=32
读一块只有 16·K 的缓冲区"的推算完全一致。修掉之后同一个探测程序不再被拦下，
且 ASan 下复跑 align=1/4/8 逐格全对（最差后向误差 1.266e-07 / 1.266e-07 / 2.985e-07）。

**顺带一条方法论坑**：第一次探测用的是从别处复用来的、**未插桩**的 `.o`，
ASan 什么都没报，看着像"没有越界"。ASan 是**逐翻译单元编译期生效**的，
链一个没带 `-fsanitize=address` 编出来的 `.o`，那些访存就完全不被检查。
**ASan 报不出来的时候，先确认被测的那个 TU 本身插过桩。**

## 新增/变更：附录 CC —— 把 CB 的三条契约推广成一次全仓扫描，净结论是「没有新缺陷」

### 变更文件

* `audit_k3_20261001.md` —— 新增附录 CC
* `docs-changelogs/CHANGELOG_2026-10-02.md` —— 本节

**本次没有改动任何生产代码。** 这一轮是纯审计：把附录 CB 在一支内核里
找到的三条契约拿到全仓逐条扫，扫完的结论是「三处都只有 CB 修的那一个实例」。

### 扫描的三条契约与结果

| 契约 | 全仓实例数 | 有问题的实例 |
|---|---|---|
| `matrix_A_cols == matrix_B_rows`（CB 缺陷 2 的形态） | 13 处函数定义 | 0 |
| 循环写缓冲必须步进写指针（CB 缺陷 1 的形态） | 7 + 4 处循环 | 0 |
| `_C3` 调用点必须有 `C == 3` 守卫 | 29 处调用 | 0 |

补充说明：

* **K 维定义**：NCHWC 卷积六支、NCHWC innerproduct 一支、NCHW 卷积八支逐个列出
  两个定义比对。NCHWC 六支现在全部相等 —— 这是 CB 修完之后才成立的
* **写指针步进**：NCHWC 族 7 处 filter im2col 循环里，`general` / `kernel2x2` /
  `kernel3x3` 三支是**按像素步进**（每像素 `+= zq_mm_align_size`），累加起来正好
  `filter_H*filter_W*align_C`，**是另一种正确写法、不是缺陷**（与这三支在
  align=4/align=8 两道门禁全绿一致）。NCHW 族是嵌套循环结构，也没有这个问题
* **`_C3` 守卫**：`_C3` 把通道数 3 硬编码进 im2col 展开。实测传 C=4 或 C=6 时，
  输出**与 C=3 逐位相同** —— 不报错、不崩溃，只是安静地只算前 3 个通道，
  **比崩溃更危险**。`ZQ_CNN_Forward_SSEUtils_NCHWC.cpp` 里 29 个调用点
  全部有 `filter_C == 3` 或 `C == 3` 守卫

### 被排除的三个候选

1. **`same_pixstep_kernel1x1`**：`matrix_A_cols = in_pixelStep` 而
   `matrix_B_rows = filter_pixelStep`，**确实不相等**，gemm 也确实以
   `ldb = in_pixelStep` 读 `filters_data`。但唯一生产调用点被
   `ZQ_CNN_Forward_SSEUtils.cpp:216` 的 `if (in_pixStep == filter_pixStep)` 整个包住。
   **教训：`matrix_A_cols != matrix_B_rows` 这个形态不等于有缺陷** ——
   还要看"相等"这个前提是谁在保证、在哪一层保证，只看内核内部会误报
2. **自造检查器的假阴性**：`_C3` 守卫扫描器第一版只认 `filter_C` / `in_C` / `need_C`，
   而 `prepack8_other_kernel3x3_C3` 那个站点所在函数的局部变量就叫 `C`
   （`ZQ_CNN_Forward_SSEUtils_NCHWC.cpp:1705`），于是误报成"没有守卫"。
   那个站点本来就是我**手工读过、确认有守卫**的，正好当自测样本 ——
   AGENTS.md「写检查类工具时先跑一个已知答案的小用例」又一次生效
3. **一处被硬编码短路的启发式**：`ZQ_CNN_Forward_SSEUtils.cpp:218` 的
   `if (1||(out_HW >= 16 && ...))`，让"小形状不走快路径"这条判断**当前是关着的**。
   不是缺陷（快路径内部还有第二道守卫会退回通用路径），
   但它是留在生产代码里的调试开关 —— 谁删掉那个 `1` 就会静默改变分派行为

### 注意事项

**这一轮扫描的价值不在于找到东西，而在于把"这一族只有这一处"变成可核对的结论。**
附录 CB 那种"连续五次假设全落空"的局面（连着五次"改了没变化"却没人想过
"是不是不止一处缺陷"），根源就是没人做过这次全仓枚举。
把"已排除"逐条写下来，下一轮才不用从零起步 —— 这与 AGENTS.md
「推不动的时候就把排除了什么记下来」是同一条。
## 新增/变更：把 CC 的 `_C3` 守卫普查固化成常驻门禁（check_c3_guards，A13/A14）

### 变更文件

* `tools/check_c3_guards.py`（新增）—— 普查 `_C3` 内核的每个调用点上方有没有 `C == 3` 守卫，
  带 `--selfcheck`
* `tools/run_audit_checks.py` —— 登记为 A13（自测）与 A14（普查）
* `audit_k3_20261001.md` —— 附录 CC.6 补记这轮的两处订正
* `AGENTS.md` —— 「写检查类工具」那节补一条
* `docs-changelogs/CHANGELOG_2026-10-02.md` —— 本节

**没有改动任何生产代码。**

### 为什么要这个门禁

NCHWC 那一族的 `_C3` 内核把通道数 **3 硬编码**进 im2col 展开
（`matrix_B_rows` 里的 `* 3`）。实测：传 C=4 或 C=6 进去，
输出**与 C=3 逐位相同** —— 不报错、不崩溃，只是安静地只算前 3 个通道。
**这比崩溃更危险**，所以每个调用点都必须有守卫。
现状 29 个 NCHWC 调用点全部有 `filter_C == 3` 或 `C == 3`；
这个门禁保证以后新增调用点时不会漏。

### 两处订正（都是**我自己的检查器**错了，不是库错了）

1. **第一版正则只认 `filter_C` / `in_C` / `need_C`**，而
   `prepack8_other_kernel3x3_C3` 那个站点所在函数的局部变量就叫 `C`
   （`ZQ_CNN_Forward_SSEUtils_NCHWC.cpp:1705`），于是误报成"没有守卫"。
   那个站点我本来**手工读过、确认有守卫**，正好拿来当自测样本 ——
   AGENTS.md「写检查类工具时先跑一个已知答案的小用例」又一次生效
2. **第二版把 `NCHWC_ONLY` 写成 `r'\bnchwc'`，一条都匹配不上** ——
   内核名是 `..._gemm_nchwc4_kernel3x3_C3`，下划线是单词字符，
   `_` 与 `n` 之间**没有词边界**。扫出 0 个调用点，自测当场全 BAD 报出来。
   这个"扫到 0 个就报坏"的兜底（AGENTS.md 坑 #2）救了它

### 一个必须记下来的**同名不同义**

**两族都叫 `_C3`，含义却不同：**

| 族 | im2col 做法 | 调用侧守卫 | 硬编码 3？ |
|---|---|---|---|
| **NCHWC**（`layers_nchwc/`） | 手写展开，读 3 个 float 当一个像素的 3 个通道 | `filter_C == 3` / `C == 3` | **是** |
| **NCHW**（`layers_c/`） | `memcpy(dst, src, sizeof(float)*filter_C)`、`cp_dst_ptr += filter_C`、`padded_len` 按实际 C 推导 | `in_C <= 4` / `in_C <= 8` | **否**（通用实现） |

NCHW 的 `_C3` 指的是**小 C 变体**，**不要求 C 恰好等于 3**。
第一版检查器没做这个限定，把 `ZQ_CNN_Forward_SSEUtils.cpp` 里 3 处
`in_C <= 4` / `in_C <= 8` 的守卫报成了"缺守卫"。
**如果没去手工核实那 3 处，就会把一条"缺陷"写进审计报告。**

> 教训：**按名字 grep 出来的候选，必须去看它那一族的实现**，
> 不能凭名字的含义套规则。同一个后缀在两个模块里可以完全是两件事。
> 门禁的自测样本里也因此加了 3 条"NCHW 的 `_C3` **必须不被计入**"的负样本。

### 自测样本（9 条，3 条"必须报" + 3 条 NCHW"必须不报"）

```
[OK ] NCHWC 标准写法 filter_C == 3                        期望 没守卫 0/调用点 1，实得 0/1
[OK ] NCHWC 局部变量就叫 C                                期望 没守卫 0/调用点 1，实得 0/1
[OK ] NCHWC in_C == 3 且带后缀 _with_bias_prelu           期望 没守卫 0/调用点 1，实得 0/1
[OK ] 故意不合格：完全没有守卫                             期望 没守卫 1/调用点 1，实得 1/1
[OK ] 故意不合格：守卫是 C == 4（等式方向对但值不对）       期望 没守卫 1/调用点 1，实得 1/1
[OK ] 故意不合格：out_C == 3 不算守卫                      期望 没守卫 1/调用点 1，实得 1/1
[OK ] NCHW 小 C 变体守的是 in_C <= 4 —— 不该被算成 NCHWC 调用点   期望 0/0，实得 0/0
[OK ] NCHW 256bit 变体 in_C <= 8 —— 同上                       期望 0/0，实得 0/0
[OK ] NCHW 连守卫都没有也不该被报 —— 它根本不是 NCHWC 内核        期望 0/0，实得 0/0
```

实测：

```
ZQCNN\ZQ_CNN_Forward_SSEUtils_NCHWC.cpp   29 个 _C3 调用点，0 个没守卫
ZQCNN\ZQ_CNN_Forward_SSEUtils.cpp           0 个 _C3 调用点，0 个没守卫  <- NCHW 那一族，正确排除
合计 29 个 **NCHWC** _C3 调用点，0 个缺 C==3 守卫
```

`python tools/run_audit_checks.py --quick` → A13/A14 均 OK。
## 新增/变更：附录 CD —— NCHWC 的 packed4 微内核族在 x86 上是整块死代码（21 个内核 / 6276 行 / 零执行）

### 变更文件

* `audit_k3_20261001.md` —— 新增附录 CD
* `docs-changelogs/CHANGELOG_2026-10-02.md` —— 本节

**没有改动任何生产代码。** 本轮是纯审计 + 一次环境能力探测。

### 结论

| | 数量 |
|---|---|
| `zq_cnn_convolution_gemm_nchwc_packed4.h` 的函数定义 | **7** |
| 每个 × plain / with_bias / with_bias_prelu | **21 个内核** |
| 该文件行数 | **6276** |
| x86 的 `conv.o` 里实际存在的 packed 符号 | **9**（3 个无 ARM 守卫的定义 × 3） |
| **x86 上会被调用的 packed 内核** | **0** |

即：**9 个编出来但永不执行，另外 12 个（4 个 ARM 守卫定义 × 3）连符号都没有。**
这 6276 行手写微内核在 Windows 与 Linux 上**一次都没跑过**。

### 三层独立核实（互相印证）

1. **调用点**：维护 `#if` 栈扫 `ZQ_CNN_Forward_SSEUtils_NCHWC.cpp`，
   12 个 packed4 调用点**全部**在 `#if __ARM_NEON && __ARM_NEON_ARMV8` 里，
   x86 会编进去的 0 个
2. **定义**：同样扫 `packed4.h` 自身的 `#if` 栈，7 个定义里 4 个带 ARM 守卫
   （M4N8 1x1 / M8N8 1x1 / M4N8 3x3_C3 / M8N8 3x3_C3），3 个不带
3. **符号**：`nm conv_fixed.o | grep -ci packed` = 9，
   与「3 个无守卫定义 × 3 个激活动作」分毫不差

> 顺带纠正一个容易想当然的地方：`packed4.h` 的 `#include`
> （`zq_cnn_convolution_gemm_nchwc.c` 的 124 / 162 / 200 / 313 / 339 行）
> **并不在** ARM 守卫里，所以这 6276 行在 x86 上**确实被编译了**
> （也因此能吃到 `-Wall` 告警），只是**永不执行**。**「编了」和「跑了」是两回事。**

### 顺带补全了附录 BY 的另一半

BY 当时报的是「x86 上每个 NCHWC 卷积层都在做一次永远不会被读的权重打包，
实测省掉 724 KB」—— 当时处理的是**分配**。用同一套 `#if` 栈扫 prepack 侧：

```
 1699  ..._prepack4_kernel1x1        #if __ARM_NEON || (SSE)     <- x86 会编、也会被调
 1714  ..._prepack4_kernel3x3_C3C4   #if __ARM_NEON || (SSE)     <- x86 会编、也会被调
 1693  ..._prepack8_other_kernel1x1  #if __ARM_NEON && ARMV8     <- x86 不编
 1708  ..._prepack8_other_kernel3x3_C3  #if __ARM_NEON && ARMV8  <- x86 不编
```

**生产侧（打包）x86 会跑，消费侧（packed 内核）x86 一概不调** ——
打包出来的 `packedfilters.data` 在 x86 上是纯粹的死数据，**调用本身也不该发生**。

### 对 CB / CC 覆盖结论的修正

* 我这两轮做的两道门禁（`zq_nchwc_conv_check` / `zq_nchwc_conv8_check`）测的全是 **raw** 族
* 当时以为「packed 族没测 = 覆盖缺口」。核实之后结论**反过来**：
  **在 x86 上 raw 族就是唯一的卷积路径**，门禁覆盖面是完整的
  （CB.3 那句「general / kernel2x2 / kernel3x3 三支全绿」是真的全绿）
* 但这**不等于 packed 族没问题**，而是**等于 packed 族在 x86 上无法被证伪**。
  CB 的经验是**同一族里相邻两个函数可以差出三处独立缺陷**
  （`kernel2x2_C3` vs `kernel3x3_C3`），所以「没跑过」和「是对的」之间没有任何逻辑关系

### 想验证它需要什么（本机做不到，如实记录）

最小条件是 ARM 交叉编译 + qemu-user：

```
$ which aarch64-linux-gnu-gcc qemu-aarch64     ->（空）
$ apt-cache policy qemu-user-static gcc-aarch64-linux-gnu
qemu-user-static:      Installed: (none)   Candidate: 1:4.2-3ubuntu6.30
gcc-aarch64-linux-gnu: Installed: (none)   Candidate: 4:9.3.0-1ubuntu2
$ sudo -n true
sudo: a password is required
```

两个包 apt 源里都有，但**本机 sudo 需要密码**，装不了。
**本轮不对 packed 族做任何运行时验证，也不做任何「它应该是对的」的断言。**
要做得装 `gcc-aarch64-linux-gnu` + `qemu-user-static`，用
`-D__ARM_NEON=1 -D__ARM_NEON_ARMV8` 交叉编译，并在 qemu 下跑与 CB 同一套逐格判据的门禁。

与本仓库已记的另外两条同类限制并列：
Linux 侧**没有 OpenCV 的 `.so`**（凡 OpenCV 路径只能读代码）、
Windows 侧 MSVC 与 gcc 的检查覆盖面不同。

### 一条不打算改的观察

`packedM8N8_other_kernel3x3_C3` 开头 `if (*buffer_len < need_buffer_size)`
**直接解引用、没有 `buffer != 0` 的分支**，而 raw 族把 `buffer == 0` 定义成
「内部分配、结束时释放」并显式支持。**这不是缺陷** —— 唯一调用方永远传真实 buffer，
与 CC.5 里 `same_pixstep_kernel1x1` 同一类。记下来是因为将来若要给 packed 族写门禁，
**别按 raw 族的用法传 `buffer = 0`**。
## 新增/变更：附录 CE —— NCHW 卷积（x86 主生产路径）第一次有数值门禁，当场查出一条生产可达的静默数据损坏

### 变更文件

* `ZQCNN/layers_c/zq_cnn_convolution_gemm_32f_align_c.c`
  —— `zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep_batch`
  的 `matrix_A_cols` 由 `in_C` 改成 `filter_H*filter_W*in_C`。
  **两处拷贝都改了**：531 行（x86 那份）与 896 行（`#if __ARM_NEON && __ARM_NEON_FP16` 那份）
* `tools/zq_nchw_conv_check.cpp`（新增）—— 16 个 NCHW 卷积入口的数值门禁
* `tools/run_zqlib_checks.py` —— 登记 `zq_nchw_conv`（`EXTRA_SOURCES` / `EXTRA_LINK` /
  `EXTRA_INC` / `EXTRA_CXXFLAGS` / `SLOW` 五处）
* `audit_k3_20261001.md` —— 新增附录 CE
* `AGENTS.md` —— 行尾/编码那节补一条「Edit 工具会抹掉 UTF-8 BOM」
* `docs-changelogs/CHANGELOG_2026-10-02.md` —— 本节

### 这条缺陷

`zq_cnn_conv_no_padding_gemm_32f_align0_same_or_notsame_pixstep_batch`
的 im2col 循环走满 `filter_H*filter_W` 个位置、每位置写 `in_C` 个 float
（3×3、C=8 时每输出像素 72 个），但 `matrix_A_row_ptr += matrix_A_cols`
用的 `matrix_A_cols = in_C`（8）—— **少了 `filter_H*filter_W*` 因子**，
`matrix_A` 的相邻行互相覆盖。1×1 时 `filter_H*filter_W*in_C == in_C` 恰好相等，
所以这个错误在 1×1 上完全看不出来。

**修法不是猜的**：同一文件同一段里紧挨着的**非 batch 兄弟**（350 行 / 717 行）
写的就是 `filter_H*filter_W*in_C`，两处只差这一个表达式。

### 实测

独立复现（与门禁**没有一行共用代码**），判据：后向误差，阈值 1e-5，逐格统计：

| 用例 | 修之前 | 修之后 |
|---|---|---|
| N=16, 3×3, C=8, K=8 | 正确 1 / 错 **12799**，最差 3.783e+00 | 正确 **12800** / 错 0，最差 1.231e-06 |
| N=16, 1×1, C=8, K=8 | 正确 18432 / 错 0 | 正确 18432 / 错 0（未受影响） |
| N=2, 3×3, C=8, K=8 | 正确 0 / 错 **5184**，最差 3.811e+00 | 正确 **5184** / 错 0，最差 1.188e-06 |
| N=2, 1×1, C=8, K=8 | 正确 6400 / 错 0 | 正确 6400 / 错 0（未受影响） |

门禁 `zq_nchw_conv`：**修之前 56 个用例里 4 个全错（EXIT=1）→ 修之后 56/56 全对（EXIT=0）**。
16 个入口覆盖 `align0` / `align128bit` / `align256bit` × `same_pixstep` /
`same_pixstep_kernel1x1` / `same_pixstep_C4` / `same_pixstep_batch` /
`same_or_notsame_pixstep` / `same_or_notsame_pixstep_C3` / `same_or_notsame_pixstep_batch`。

### 生产可达性

`ZQ_CNN_Forward_SSEUtils.cpp` 的 727 与 926 两处调它，触发条件：

* `align_mode` **既不是** ALIGN_128bit **也不是** ALIGN_256bit
  —— 即张量用**不做通道对齐**的 `Align0` 变体分配
* 且 `out_N >= 16`（大 batch）、`out_NHW >= 8`
* 且卷积核**不是 1×1**

**shipped 的 sample 全部走 ALIGN_128bit / ALIGN_256bit，所以 sample 回归抓不到它** ——
这正是"没有数值门禁"的代价。

### 注意事项

1. **这是静默数据损坏，不是内存安全问题。** 我第一反应是"缓冲区少分配 9 倍 → 堆溢出"，
   用 ASan 去验，**ASan 什么都没报**。手算给出了原因：
   ```
   need_A = 648*8*4 = 20,736 字节（按错的 matrix_A_cols 算）
   need_B = 72 *8*4 =  2,304 字节（matrix_Bt 紧跟在 A 后面）
   total  = 23,040 字节
   实际最大写入偏移 = 647*8*4 + 72*4 = 20,992 < 23,040
   ```
   **溢出部分全部落在 B 缓冲区里，没有跑出整块分配。**
   教训：看到"少分配"先算一遍**最大写入偏移 vs 分配总长**，再决定要不要报成内存安全问题。
   **"少分配了 9 倍"和"越界"是两件事。**
2. **ARM FP16 那份拷贝（896 行）本机无法运行**（没有 ARM 工具链，见附录 CD.6）。
   它与同段非 batch 兄弟（717 行）**只差这一个表达式**，修法逐字相同 ——
   这一点如实标注，**不宣称它被验证过**
3. **门禁自己也有一个缺陷（已修）**：第一版"让 `in_pixStep != filter_pixStep`"
   写的是"把 f_pixStep 也补到 4 的倍数"，对 C=8 算出来还是 8，等于**根本没测到**。
   现在改成 `f_pixStep = in_pixStep + 4`，并把两边的 pixelStep 写进结果文件、
   在用例行里直接打出来 —— 免得又出现"以为测了、其实没测"
4. **`align0` 在 NCHW 这一族里不等于"标量模板实例"** —— 它是**手写的通用实现**
   （`zq_gemm_32f_align0_AnoTrans_Btrans` + memcpy），与 `_raw.h` 那套
   `zq_mm_*` 模板完全是两套代码。所以 NCHW 与 NCHWC 的同名函数**不能互为参照**
5. **顺带记一个工具坑**：`Edit` 工具会抹掉文件头的 UTF-8 BOM。
   改 `zq_cnn_convolution_gemm_32f_align_c.c` 时它在 `git diff` 里表现为
   "第一行被改了一行内容"，混在两处真正的改动里。已还原并记进 AGENTS.md
## 新增/变更：附录 CF —— NCHWC depthwise 门禁，284 个生产层第一次有覆盖（首跑全绿，且验证过门禁有牙齿）

### 变更文件

* `tools/zq_nchwc_depthwise_check.cpp`（新增）—— 63 个入口的数值门禁
* `tools/run_zqlib_checks.py` —— 登记 `zq_nchwc_depthwise`
  （`EXTRA_SOURCES` / `EXTRA_LINK` / `EXTRA_INC` / `EXTRA_CXXFLAGS` 四处；**不进 SLOW**，
  编一次约 17 秒，所以默认回归里就会跑到）
* `audit_k3_20261001.md` —— 新增附录 CF
* `docs-changelogs/CHANGELOG_2026-10-02.md` —— 本节

**没有改动任何生产代码。**

### 背景

`model/` 下的 shipped 模型里有 **284 个 `DepthwiseConvolution` 层**
（MobileNetSSD / Pose / det1-dw\* / det2-dw\* / det3-dw\* …），
而 22 个 `zq_*_check.cpp` **一个都没覆盖它**。

这一族与普通卷积是**两套完全不同的代码**：不走 gemm，是手写的
「每个通道一个 filter」SIMD 展开（`filter_N == 1`、`filter_C == in_C`、`out_C == in_C`），
所以附录 CB / CE 修的那些问题对它一概无效。附录 CE 刚在 NCHW 卷积里查出一条
100% 算错的生产可达缺陷 —— 同一层里"另一个变体没测过"的教训在这里直接适用。

### 覆盖面与结果

7 个基础变体（`general` / `kernel3x3` / `kernel5x5_s1d1` / `kernel3x3_s1d1` /
`kernel3x3_s2d1` / `kernel2x2` / `kernel2x2_s1d1`）
× 3 种对齐（NCHWC1/4/8）× 3 个激活动作 = **63 个入口**，
每个入口 3 组形状（`C=align`、`C=2*align` 跨两个对齐组、`N=2`），合计 **189 个用例**。

```
共 189 个用例：全对 189，有错 0，崩溃/搭建失败 0
```

门禁做法沿用 CB / CE 已验证过的部分：内核名写全走函数指针表（不拼接）、
后向误差 + 逐格统计、每用例 fork 一个子进程、
**用真实的 `ZQ_CNN_Tensor4D_NCHWC1/4/8` 类**分配与填充张量、
输出缓冲区预填 `-12345.0f`（内核没写就会看到哨兵值而不是"恰好通过"）。

### 首跑就绿，怎么知道它不是"什么都没测"

做了两件事：

1. **确认 63 个入口真是 63 个不同符号**：
   `nm zq_dw.o | grep " T .*depthwise.*nchwc" | sort -u | wc -l` → **63**
   （其中 21 个是不带 `_with_bias` 的 plain 变体 = 7 基础 × 3 对齐，与预期一致）
2. **变异测试**：把门禁**自己**参考实现里的 filter 下标改错
   （`flt[ch*fH*fW + fh*fW + fw]` → `flt[fw*fH*fW + fh*fW + ch]`），
   **只改门禁、不碰被测代码**，重跑：
   ```
   共 189 个用例：全对 123，有错 66，崩溃/搭建失败 0
   ```
   **66 个用例立刻变红。** 绿不是因为判据松，而是因为库真的算对了。变异体已删除

> **一个从不报错的检查工具，和一个坏掉的检查工具，在输出上长得一模一样。**
> 唯一能区分它们的方法是**故意把它弄坏，看它会不会叫** ——
> 这是 CB 那几轮"假绿"教训的直接应用。

### 这条门禁**没有**覆盖到什么（如实记下来）

* **形状**：只有 `H=W=17`、`N∈{1,2}`、`C∈{align, 2*align}`。
  没测 `C` **不是** align 倍数的情形
* **dilation > 1 完全没测** —— 而 `kernel5x5_s1d1` / `kernel3x3_s1d1` /
  `kernel2x2_s1d1` 这些名字里的 `s1d1` 正是在强调"dilation 被硬编码成 1"，
  也就是说**这些变体不支持 dilation != 1**，但**没有任何地方拦住**调用方传一个进来
* **契约之外**：与 CB 的 `kernel2x2_C3` 一样，这些内核不对非法形状做校验
* **ARM**：三种对齐在 x86 上都编了、都跑了；NCHWC 在 ARM 上还有 NEON 专用路径，
  本机无法验证（见附录 CD.6）

### 下一个明确的空白

`addbias_prelu` / `addbias_prelu_sure_slope_lessthan1` 仍然**没有门禁**。
它被 NCHWC 那一族的 `with_bias_prelu` 三个变体直接调用
（`zq_cnn_conv_no_padding_gemm_nchwc*_*_with_bias_prelu` 里的
`zq_mm_fmadd_ps(slope_v, min(0,x), max(0,x))` 就是它），
所以它一旦错，CB 修好的那些路径会一起错 —— 只是目前没有独立的门禁盯它。
## 新增/变更：附录 CG —— NCHWC 激活层门禁（15 个入口，75 个用例全对）+ 变异测试抓出我自己门禁里的一个 bug

### 变更文件

* `tools/zq_nchwc_act_check.cpp`（新增）—— NCHWC 激活层门禁
* `tools/run_zqlib_checks.py` —— 登记 `zq_nchwc_act`（`EXTRA_SOURCES` / `EXTRA_LINK` /
  `EXTRA_INC` / `EXTRA_CXXFLAGS` 四处；**不进 SLOW**，编一次约 8 秒）
* `audit_k3_20261001.md` —— 新增附录 CG
* `docs-changelogs/CHANGELOG_2026-10-02.md` —— 本节

**没有改动任何生产代码。**

### 背景

CF 收尾时点名 `addbias_prelu` 是下一个空白：NCHWC 卷积的 `with_bias_prelu`
三个变体内部**直接调 prelu**（`zq_mm_fmadd_ps(slope_v, min(0,x), max(0,x))`），
它一旦错，CB 修好的那十几个卷积入口会一起错，而没有独立的门禁盯它。

### 覆盖面

头里那两个宏别名（`zq_cnn_prelu_nchwc` / `..._sure_slope_lessthan1`）只是
`zq_cnn_prelu_nchwc.c` 在 include 时的重命名，**真实定义**是 15 个符号（已用 `nm` 核实）：

  addbias              zq_cnn_addbias_nchwc{1,4,8}                                  bias
  prelu                zq_cnn_prelu_nchwc{1,4,8}                                    slope
  prelu_sure           zq_cnn_prelu_nchwc{1,4,8}_sure_slope_lessthan1               slope
  addbias_prelu        zq_cnn_addbias_prelu_nchwc{1,4,8}                            bias, slope
  addbias_prelu_sure   zq_cnn_addbias_prelu_nchwc{1,4,8}_sure_slope_lessthan1       bias, slope

每个 5 组形状：`W ∈ {12,13,14,15}`（覆盖内核自己的 `in_W%4==0/1/2/3` 四条分派）
+ `N=2, C=align`。`C = align+2` 的那四组**故意用不是 align 倍数的通道数**。
判据用**逐元素**后向误差（`max(|y|,1)` 归一），阈值 1e-6 ——
这里没有任何归约，所以可以比卷积那套严得多。

结果：**共 75 个用例：全对 75，有错 0，崩溃 0**。

### 重点：变异测试抓出**我自己门禁里的一个 bug**

首跑全绿后，把门禁**自己**参考实现里的 slope 偏 0.001（只改门禁、不碰被测代码）再跑：

  第一次：全对 65，有错 10     <- 只抓到 10/75，太少
  修门禁后：全对 17，有错 58    <- 剩下的 17 个是变异本就不影响的 addbias

根因在门禁里：

    std::vector<float> bv(A), sl(A);        // 按 align 开 —— 错
    for (int k = 0; k < A; k++) { ... }

而用例里 `C = align + 2 > A`，于是**通道 `align` 与 `align+1` 的 slope 是 0**。
内核那边 `slope_v = zq_mm_load_ps(slope + c)` 每次读 `align` 个 float，
`c` 走到最后一个不满的组时**读过界**；而我的参考值也用 `sl[align] = 0`，
两边"**恰好一致**"，所以门禁全绿 —— **那 4 组用例根本没在测 prelu 的斜率**。
改成按补齐后的通道数开（`paddedC = (C + A - 1) / A * A`）之后，
未变异版本仍是 75/75 全绿，变异版本抓到 58/75。

> 一个从不报错的检查工具，和一个坏掉的检查工具，在输出上长得一模一样。
> 唯一能区分它们的方法是**故意把它弄坏，看它会不会叫** ——
> 而且"叫了多少"本身就是信号：只抓到 10/75 的时候，正确的结论不是"库还行"，
> 而是"我的工具还有 20 组是空的"。
>
> 这是本会话第七次栽在"自己的检查工具给出可信的错误答案"上
> （B 最差格 / CA align=8 段 / CC 守卫正则 / CD 三层核实 / CE diff_pixstep /
>  CF 参考下标 / CG slope 数组长度）。

### 顺带记一条

`zq_cnn_addbias_nchwc.h` 里 `zq_cnn_addbias_nchwc1` 被**声明了两次**。
C++ 下重复声明合法、不会报错，但说明这份头是手工维护的 ——
**"声明存在"不等于"只有一处声明"**，写门禁时别用声明条数当符号个数，要用 `nm`。

### 剩下的空白

| 层 | 状态 |
|---|---|
| NCHWC 卷积（raw 族） | zq_nchwc_conv / zq_nchwc_conv8（CB） |
| NCHWC depthwise | zq_nchwc_depthwise（CF） |
| NCHWC 激活层 | 本附录 |
| NCHW 卷积 | zq_nchw_conv（CE） |
| NCHW pooling / eltwise / lrn | 已有门禁 |
| NCHWC pooling / relu / softmax / batchnormscale / resize | **仍无门禁** |
| NCHWC packing 那几支 | x86 上不可达（CD） |
## 新增/变更：附录 CH —— NCHWC relu + eltwise 门禁（15 个入口，57 个用例全对，变异测试符合预期）

### 变更文件

* `tools/zq_nchwc_elt_relu_check.cpp`（新增）—— NCHWC relu + eltwise 门禁
* `tools/run_zqlib_checks.py` —— 登记 `zq_nchwc_elt_relu`
  （`EXTRA_SOURCES` / `EXTRA_LINK` / `EXTRA_INC` / `EXTRA_CXXFLAGS` 四处；**不进 SLOW**）
* `audit_k3_20261001.md` —— 新增附录 CH
* `docs-changelogs/CHANGELOG_2026-10-02.md` —— 本节

**没有改动任何生产代码。**

### 覆盖面与语义

15 个入口（`nm` 核实）：`relu`(3) + `eltwise{sum, max, mul, sum_with_weight}`(12)，
各 × NCHWC1/4/8。语义**逐条从源码读出来**：

| 内核 | 语义 | 出处 |
|---|---|---|
| `relu_nchwc(data,…,slope)` | 就地；`slope == 0` 时 `out = max(0,x)`，否则 `out = slope*min(0,x) + max(0,x)` | `zq_cnn_relu_nchwc_raw.h:23` |
| `eltwise_sum` | `out = Σ in[i]` | 先写 `in[0]+in[1]`，再对 `tensor_id >= 2` 累加 |
| `eltwise_max` | `out = max_i in[i]` | 两种写法：`max(in_pix, in1_pix)` 或累加式 |
| `eltwise_mul` | `out = Π in[i]` | 同上 |
| `eltwise_sum_with_weight` | `out = Σ weight[i]·in[i]` | `weight` 是**每张输入一个标量**（`set1_ps(weight[i])` 广播），**不是逐通道** |

最后一条特别值得记：`weight` 的类型是 `const float*`，第一眼看着像逐通道权重，
实际是每张张量一个。写错会得到一个"看起来在跑、全错"的门禁。

### 结果

`relu` 特意跑了 `slope == 0` 与 `slope != 0` **两条分支**；
`eltwise` 特意跑了输入张量数 **2 / 3 / 4**（内核对第 3 个及以后另有一段循环）。

  共 57 个用例：全对 57，有错 0，崩溃/搭建失败 0

### 变异测试：抓到 18/57，与预期完全一致

按 CG 立下的规矩做：把门禁**自己**的参考实现改错两处，重跑。

* MUTANT1：`relu` 参考里 `slope * x` 改成 `slope * (x + 0.001f)` —— 只影响 `slope != 0` 分支
* MUTANT2：`eltwise_sum` 参考里每一项乘 `1.001f`

  共 57 个用例：全对 39，有错 18，崩溃/搭建失败 0

**18 = 6 + 12**：6 是三个 align 的 relu 中 `slope != 0` 的用例
（3 个入口 × 3 个形状里只有 2 个走那条分支 → 6），
12 是 `eltwise_sum` 的 3 个入口 × 4 个形状。
**MUTANT1 本来就不该影响 `slope == 0` 的 3 个用例，它们保持绿是对的。**

> 这是第三次做变异测试（CF / CG / CH），也是第一次**抓到数量与预期完全对上**。
> 前两次的价值都在"发现门禁自己有洞"（CF 的 filter 下标、CG 的 slope 数组长度）；
> 这一次说明那两次修完之后门禁的灵敏度是真的了。

### 剩下的空白

| 层 | 状态 |
|---|---|
| NCHWC 卷积（raw 族） | zq_nchwc_conv / zq_nchwc_conv8（CB） |
| NCHWC depthwise | zq_nchwc_depthwise（CF） |
| NCHWC 激活层（addbias / prelu） | zq_nchwc_act（CG） |
| NCHWC relu / eltwise | 本附录 |
| NCHW 卷积 | zq_nchw_conv（CE） |
| NCHW pooling / eltwise / lrn | 已有门禁 |
| **NCHWC pooling（24 个入口）** | **仍无门禁 —— 下一个** |
| NCHWC batchnormscale（12）/ softmax（5）/ resize（6） | 仍无门禁 |
| NCHWC packing 那几支 | x86 上不可达（CD） |

`pooling` 是剩下最大的一块（24 个入口），而且头里有**重复声明**
（`zq_cnn_avgpooling_nopadding_nodivided_nchwc4_general` 出现两次），
写门禁时要用 `nm` 取真实符号表，别用声明条数（CG.5 已经吃过一次亏）。
## 新增/变更：附录 CI —— NCHWC pooling 门禁（24 个入口 / 114 个用例全对，变异测试精准命中边界分支）

### 变更文件

* `tools/zq_nchwc_pool_check.cpp`（新增）—— NCHWC pooling 门禁
* `tools/run_zqlib_checks.py` —— 登记 `zq_nchwc_pool`
  （`EXTRA_SOURCES` / `EXTRA_LINK` / `EXTRA_INC` / `EXTRA_CXXFLAGS` 四处；**不进 SLOW**）
* `audit_k3_20261001.md` —— 新增附录 CI
* `docs-changelogs/CHANGELOG_2026-10-02.md` —— 本节

**没有改动任何生产代码。**

### 覆盖面与结果

24 个入口（`nm` 核实，**不是**按头里的声明数 —— 头里有重复声明，
`zq_cnn_avgpooling_nopadding_nodivided_nchwc4_general` 出现两次）：
avg/max × nodivided/suredivided × nchwc1/4/8。

  nodivided_general     每个入口 9 个用例   <- 6 个 (in-k)%s==0 + 3 个刻意不整除
  suredivided_general   每个入口 6 个用例
  suredivided_k2x2/k3x3 每个入口 2 个用例

  共 114 个用例：全对 114，有错 0，崩溃/搭建失败 0

### 这一族的名字有歧义，代码没有

```c
int final_kH = __min(kernel_H, in_H - (out_H - 1)*stride_H);
int final_kW = __min(kernel_W, in_W - (out_W - 1)*stride_W);
```

| | 边界分支 | 除数 |
|---|---|---|
| `nodivided_general` | **有**四条（末列 / 末行 / 角上各一条） | 内部 `kH*kW`；末列 `kH*final_kW`；末行 `final_kH*kW`；角上 `final_kH*final_kW` |
| `suredivided_*` | **没有**，只有一个循环 | 一律 `kH*kW` |

**`nodivided` 才是"按实际窗口收窄"的那一个**，`suredivided` 反而不管边界、一律除满。

`suredivided` 的**契约**：每个窗口都必须放得下，即
`(in_H - kernel_H) % stride_H == 0` 且 `out_H = (in_H-kernel_H)/stride_H + 1`。
分派器用 `suredivided = (in_H+pad-kernel_H)%stride_H==0 && (…)` 保证
（`ZQ_CNN_Forward_SSEUtils.h:1604`）。**不满足就会越界读** —— 库不做任何校验。

max pooling 不做除法，所以两者的区别**只在边界处理**。

### 另一个契约：kernel2x2 / kernel3x3 把窗口写死了

它们**完全忽略 `kernel_H`/`kernel_W`**（每行固定 2 次/3 次 load、行数也固定），
却仍用 `1/(kernel_H*kernel_W)` 做除数 —— 传 3×3 给 kernel2x2 会算出
"2×2 的和 ÷ 9"，**静默出错**。

**分派器有守卫**（`ZQ_CNN_Forward_SSEUtils_NCHWC.cpp:3638`）：
`if (kernel_H==2 && kernel_W==2) … else if (3&&3) … else general`。
与附录 CC.5 的 `same_pixstep_kernel1x1` 完全同一形态 ——
契约在调用侧强制，内核内部不重复校验。所以门禁也必须按契约喂。

顺带：头里的契约注释写的是 `out_H must be ceil((in_H - filter_H)/stride_H) + 1`，
**参数名写的是 `filter_H`，实际叫 `kernel_H`**，又是一条过时注释。

### 门禁自己踩的坑

第一版给所有入口都喂 `(in - k) % s == 0` 的形状（因为 `suredivided` 的契约要求这样），
但**对 `nodivided` 来说这恰好让 `final_kH` 恒等于 `kernel_H`，
边界分支一次都执行不到**。补了 3 个刻意让 `(in-k) % s != 0` 的用例，
`nodivided` 每入口从 6 组增到 9 组，总数 96 → 114。

### 变异测试：6/6 精准命中

注入一处专门针对边界分支的变异（门禁自己的参考不再用收窄后的 `fh*fw` 做除数）：

    - else y = sum / (double)(e.divided ? (kH * kW) : (fh * fw));
    + else y = sum / (double)(kH * kW);      /* MUTANT */

  共 114 个用例：全对 108，有错 6，崩溃/搭建失败 0

6 个红的是 3 个 `avg nodivided` 入口各 2 个**真正发生裁剪**的用例；
我加的第 3 组 `s == 1` 那一组 `(in-k) % 1` **恒为 0**、根本不会裁剪，
**本来就不该红** —— 事实如此，它确实保持绿了。

> 这是第四次做变异测试（CF / CG / CH / CI），也是第二次**抓到数量与预期完全对上**。
> 它同时证明了两件事：① 门禁**能**测出边界分支上的差别；
> ② 门禁**没有**在那些"其实没差别"的地方制造假信号。

### 剩下的空白

  NCHWC 卷积（raw 族）              zq_nchwc_conv / zq_nchwc_conv8（CB）
  NCHWC depthwise                   zq_nchwc_depthwise（CF）
  NCHWC 激活层                      zq_nchwc_act（CG）
  NCHWC relu / eltwise              zq_nchwc_elt_relu（CH）
  NCHWC pooling                     本附录
  NCHW 卷积                         zq_nchw_conv（CE）
  NCHW pooling / eltwise / lrn      已有门禁
  NCHWC batchnormscale(12) / softmax(5) / resize(6)   仍无门禁
  NCHWC packing 那几支              x86 上不可达（CD）

`batchnormscale` 是下一块，它的参考实现最容易写错 ——
`batchnorm_b_a_nchwc` 的参数名叫 `b_data, a_data`，而代码是
`fmadd(x, b_vec, a_vec)` = `x*b + a`，也就是 **b 乘、a 加**，与名字的直觉相反。
写它之前必须先把四段的实际算式逐条抄下来。
## 新增/变更：附录 CJ —— NCHWC batchnormscale 门禁（12 入口 / 30 用例）+ 顺带查出五道门禁共有的一个假绿来源

### 变更文件

* `tools/zq_nchwc_bn_check.cpp`（新增）—— NCHWC batchnormscale 门禁
* `tools/zq_nchwc_depthwise_check.cpp` / `zq_nchwc_act_check.cpp` / `zq_nchwc_elt_relu_check.cpp` / `zq_nchwc_pool_check.cpp`
  —— 全部补上「结果文件缺失 = 判失败」的守卫；depthwise 与 act 的逐通道数组改成 32 字节对齐
* `tools/run_zqlib_checks.py` —— 登记 `zq_nchwc_bn`（四处；**不进 SLOW**）
* `audit_k3_20261001.md` —— 新增附录 CJ；**并给附录 CF.4 加了"本节的'全对'是假的"的更正**
* `AGENTS.md` —— 「写检查类工具」那节补三条（没读到=失败 / 逐通道数组两个坑 / 参数名不是语义）
* `docs-changelogs/CHANGELOG_2026-10-02.md` —— 本节

`ZQCNN/` 下**没有改动任何生产代码**。

### 覆盖面与结果

12 个入口：`scale` / `batchnorm_b_a` / `batchnorm_mean_var` /
`batchnormscale_mean_var_scale_bias` 各 × NCHWC1/4/8。
`scale` 的 `bias == NULL` 与 `bias != NULL` 两个分支都测。

  共 30 个用例：全对 30，有错 0，崩溃/搭建失败 0

### 这一族的参数名与直觉相反

| 内核 | 算式 |
|---|---|
| `scale_nchwc(data,…,scale,bias)` | `bias != NULL` → `x*scale[c]+bias[c]`；`bias == NULL` → `x*scale[c]` |
| `batchnorm_b_a_nchwc(data,…,b_data,a_data)` | `x*b_data[c] + a_data[c]` |
| `batchnorm_mean_var_nchwc(data,…,mean,var,eps)` | `b=1/sqrt(max(var+eps,1e-32))`，`a=-mean*b`，`out=x*b+a` |
| `batchnormscale_mean_var_scale_bias_nchwc(…)` | `b=scale/sqrt(max(var+eps,1e-32))`，`a=bias-mean*b`，`out=x*b+a` |

**`batchnorm_b_a` 的第一个参数是乘数、第二个是加数**（代码是 `fmadd(x, b_vec, a_vec)`）。
按名字写成"x*a+b"会得到一个**看起来在跑、全错**的门禁。

另外 `batchnorm_mean_var` 算完 a/b 之后**直接调 `batchnorm_b_a_nchwc`**
（raw.h:123），所以 b_a 的问题会在两处一起出现。

### 重点：变异测试抓出**两个**问题，第二个是通用的

MUTANT1 把 b/a 顺序反了、MUTANT2 漏掉 scale 因子，
第一次跑只有 **11/30** 变红（预期 12）—— `nchwc8 batchnorm_b_a N=2 C=8` 没红。

**（一）我的门禁违反了 AGENTS.md 里已有的对齐契约。** 直连探针一跑就现形：

    AddressSanitizer: SEGV on unknown address 0x000000000000
        #0 _mm256_load_ps
        #1 zq_cnn_batchnorm_b_a_nchwc8  zq_cnn_batchnormscale_nchwc_raw.h:216

`zq_mm_load_ps` 在 align=8 下是 `_mm256_load_ps`，**要求 32 字节对齐**，
而 `std::vector<float>` 只给 16。AGENTS.md「ZQ_GEMM 的调用方契约」第 2 条
早就写着这条，是我的门禁没照做。

**（二）"结果文件缺失 = 通过"—— 五道门禁共有的洞。** 原来的判定：

    if (f) { if (fscanf(...) != N) ok = bad = 0; fclose(f); }
    if (WIFSIGNALED(st)) { g_crash++; …; return; }
    if (bad > 0) { g_bad++; …; } else { g_ok++; }        // <-- 洞

**ASan 撞上 SEGV 时默认走 `Die()` → `_exit(1)`，不发信号。**
于是 `WIFSIGNALED` 为假、退出码也不对、`bad` 保持 0
→ **一个段错误被记成了"通过"**。
补上 `if (!have) { g_crash++; …; return; }`。

### 补上之后，附录 CF 的"189 个全对"被推翻

守卫一加，`zq_nchwc_depthwise` 立刻变红：

    共 189 个用例：全对 119，有错 0，崩溃/搭建失败 70

**70 个用例从未运行过**（`with_bias`/`with_bias_prelu` 要传 per-channel 数组，
align=8 下一样撞 16 字节对齐），而 CF.4 把它们全记成了"全对"。
按同样手法修好 depthwise 与 act 的数组之后：

| 门禁 | 修正前 | 修正后 |
|---|---|---|
| zq_nchwc_depthwise | 189 个里 70 个从未运行 | 189/189 全对 |
| zq_nchwc_act | 同类问题 | 75/75 全对 |
| zq_nchwc_bn | 新写，已含对齐处理 | 30/30 全对 |
| zq_nchwc_elt_relu / zq_nchwc_pool | 不传 per-channel 数组，本来就不受影响 | 57 / 114 全对 |

**depthwise 这一族的结论没有变**（它确实是对的），
**但 CF 当时那份证据不成立** —— 已在 CF.4 就地标注更正。

### 修正之后三道门禁的变异测试

| 门禁 | 变异 | 检出 | 该保持绿的 |
|---|---|---|---|
| zq_nchwc_depthwise | 参考的 filter 下标写错 | 189/189（126 算错 + 63 变异自身越界被 ASan 拦下） | 0 |
| zq_nchwc_act | 参考的 slope 偏 0.001 | 60/75 | 15（全是 addbias 入口，变异不影响） |
| zq_nchwc_bn | b/a 顺序反 + 漏掉 scale | 12/30 | 18（scale 与 batchnorm_mean_var 入口） |

三道门禁现在都做到"**该红的全红、该绿的绿**"。

### 教训（已进 AGENTS.md）

* **「子进程写文件 + 父进程读文件」的门禁，必须显式判「没读到」= 失败。**
  通则：**判据里必须区分「读到了但内容不对」和「根本没读到」**，
  前者是失败，后者**更**是失败 —— 而默认写法会把后者算成通过
* **逐通道数组两个坑一起防**：长度按 `ceil(C/align)*align` 开（CG）、
  对齐按 32 字节开（CJ）。两个坑在**同一个数组**上
* **参数名不是语义**：已踩三次（b_a 的 b/a、`weight` 的粒度、`nodivided` 的含义）。
  照抄参数名之前先去看 `fmadd`/`store` 那两行
## 新增/变更：附录 CK —— NCHWC softmax 门禁（5 入口 / 15 用例全对）+ 一条「该红的没红」的通用结论

### 变更文件

* `tools/zq_nchwc_softmax_check.cpp`（新增）—— NCHWC softmax 门禁
* `tools/run_zqlib_checks.py` —— 登记 `zq_nchwc_softmax`
  （`EXTRA_SOURCES` / `EXTRA_LINK` / `EXTRA_INC` / `EXTRA_CXXFLAGS` 四处；**不进 SLOW**）
* `audit_k3_20261001.md` —— 新增附录 CK
* `AGENTS.md` —— 「写检查类工具」那节补两条
* `docs-changelogs/CHANGELOG_2026-10-02.md` —— 本节

`ZQCNN/` 下**没有改动任何生产代码**。

### 覆盖面与结果

5 个入口（`nm` 核实）：`zq_cnn_softmax_nchwc1_C` / `_H` / `_W` /
`nchwc4_C` / `nchwc8_C`。语义是**就地**沿指定轴做标准 softmax：
`max = max(该轴)` → `v = exp(v-max)` → `sum = Σv` → `v = v/sum`。
`_H` / `_W` 只有 NCHWC1 一份，在 `zq_cnn_softmax_nchwc.c:196/245` 手写，
不是 `_raw.h` 那段的宏重命名。

每个入口 3 组形状：`C = align*2`（整对齐）、`C = align+3`（不是 align 倍数，
内核的对齐尾循环必须被走到）、`C = 1`（全部落在尾循环里）。

  共 15 个用例：全对 15，有错 0，崩溃/搭建失败 0

### 一处"看着像 bug 其实不是"的写法

对齐尾循环 `for (; c < in_C; c++, slice_ptr++)` 里的 `slice_ptr++` 看着漏了
`in_sliceStep`，**但它是对的** —— 主循环退出时指针已停在正确位置，
尾循环**先读后加**、加完就结束，那个 `++` 会被丢弃。

> 推论进 AGENTS.md：**看着可疑的 `++`，先确认"加完还有没有被用到"，别靠"看着像"。**

### 重点：第一个变异是**无效的**，而它无效的原因本身就是一条结论

我把参考实现的 `exp(v - max)` 改成 `exp(v)`，**15 个用例一个都没变红**。
因为 softmax 是**平移不变**的：

    exp(v_i - m) / Σ_j exp(v_j - m)  =  exp(v_i)·exp(-m) / (exp(-m)·Σ_j exp(v_j))  =  exp(v_i) / Σ_j exp(v_j)

减 max **在数学上冗余**，存在只是为了数值稳定（`v` 很大时 `exp(v)` 会溢出）。
我的输入在 [-1,1]，两边都不溢出，结果逐位相同。

> **「该红的没红」的第一反应应该是「我的变异是等价的」，而不是「库没执行那段」。**
> 门禁"测不到某处代码"不等于"那处没被覆盖"，也可能是那处在当前输入下与其他写法
> **数学等价**。要真正测到它，得喂**大到会溢出**的输入。
> 这条也适用于别处：查一个恒等式、查一个 clamp、查一个 max ——
> 先用代数判断"这个变异在数学上会不会改变结果"，别靠"跑一遍没变"就下结论。

顺带：第二处变异（把 `_W` 轴当成 `_H`）让 3 个用例**崩溃**而不是变红 ——
因为它让参考自己越界读了 `in`（`ih` 最大到 W-1=6，而 H=5）。
**门禁正确判成"没跑完"**（CJ.4 那道守卫），没把它算成通过。

### 换成有效变异之后：12/15，与预期完全一致

    - double y = v[i] / sum;      /* 归一化 */
    + double y = v[i] / AL;       /* MUTANT: 改成平均 */

  共 15 个用例：全对 3，有错 12，崩溃/搭建失败 0

**保持绿的正好 3 个** —— 三个 `_C` 变体的 `C = 1` 用例。
`AL == 1` 时 `v[0]/sum` 与 `v[0]/AL` 都是 `exp(0)/1 = 1`，
**这个变异对它们本来就无效**；而 `_H`/`_W` 的 `C=1` 用例轴长是 5/7，变异有效。

"该红的全红、该绿的绿"—— 这是第四次做到（CI / CH / CJ / CK）。

### 剩下的空白

  NCHWC 卷积 / depthwise / 激活层 / relu+eltwise / pooling /
  batchnormscale / **softmax**      CB / CF / CG / CH / CI / CJ / **CK**
  NCHW 卷积                         CE
  NCHW pooling / eltwise / lrn      已有门禁
  **NCHWC resize_with/without_safeborder（6 个入口）**   仍无门禁
  NCHWC packing 那几支              x86 上不可达（CD）

`resize` 单独留给下一轮：双线性插值的**角点对齐规则**
（`w1 = (2*ow + 1)*realW/wrapW - 1`）以及 `with_safeborder` / `without_safeborder`
的区别，与 AGENTS.md「三个张量变体对越界 rect 的策略互相冲突」那条直接相关，
不能照着名字写参考。
## 新增/变更：附录 CL —— NCHWC resize 门禁，当场查出并修掉一处越界读（同仓另一族是正确写法）

### 变更文件

* `ZQCNN/layers_nchwc/zq_cnn_resize_nchwc_raw.h`
  —— **修掉一处越界读**（本轮唯一的生产代码改动）：
  `y0`/`y1` 的边界钳位由 `__min(in_H, …)` 改为 `__min(in_H - 1, …)`
* `tools/zq_nchwc_resize_check.cpp`（新增）—— NCHWC resize 门禁（6 个入口）
* `tools/run_zqlib_checks.py` —— 登记 `zq_nchwc_resize`（四处；**不进 SLOW**）
* `audit_k3_20261001.md` —— 新增附录 CL
* `AGENTS.md` —— 「写检查类工具」那节补两条
* `docs-changelogs/CHANGELOG_2026-10-02.md` —— 本节

### 缺陷

`ZQCNN/layers_nchwc/zq_cnn_resize_nchwc_raw.h:176-177`：

    x0[w] = __min(in_W - 1, __max(0, x0[w]));    // x：上界 in_W - 1  正确
    ...
    y0    = __min(in_H,     __max(0, y0));       // y：上界 in_H      少了 -1
    y1    = __min(in_H,     __max(0, y1));       //   同一个函数里，x 那边是对的

坐标超出图时 y 被钳到 `in_H`，而合法行号上界是 `in_H - 1`，
于是 `in_row0_ptr = in_slice_ptr + y0*in_widthStep` **指到图外面一行**。

**同仓 A/B（决定性）**：`ZQCNN/layers_c/zq_cnn_resize_32f_align_c_raw.h`
里 NCHW 那一族（NCHW 是 x86 上的**主生产路径**）用的是
`y_nn = __min(in_H - 1, __max(0, y_nn))`，**5 处全是 `in_H - 1`**。
同一个功能的两份实现，一份对一份错 —— 这不是设计取舍，是笔误。

### ASan 实证

独立复现程序（与门禁无共用代码），`N=1 16x16 C=8`、`off=(0,0)`、
`rect=20x20`、`out=16x16`：

    第 12 行：未钳 y0=15 y1=16 -> 钳到 y0=15 y1=16   *** 越界（合法上界 15）***
    第 13 行：未钳 y0=16 y1=17 -> 钳到 y0=16 y1=16   *** 越界 ***
    第 14 行：未钳 y0=17 y1=18 -> 钳到 y0=16 y1=16   *** 越界 ***
    第 15 行：未钳 y0=18 y1=19 -> 钳到 y0=16 y1=16   *** 越界 ***

    ==202917==ERROR: AddressSanitizer: heap-buffer-overflow
    READ of size 32 at 0x625000004940 thread T0
        #0 _mm256_load_ps
        #1 zq_cnn_resize_without_safeborder_nchwc8
             ZQCNN/layers_nchwc/zq_cnn_resize_nchwc_raw.h:191
    0x625000004940 is located 0 bytes to the right of 8256-byte region

修掉那两行的 `- 1` 之后，**同一个探针不再被 ASan 拦下，函数正常返回**。

### 生产可达性

从 `ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp` 的 **6 个调用点**进入
（line 252 / 339 / 711 / 798 / 1171 / 1258），
也就是 **NCHWC 张量上做 resize 的那条路**，检测器（SSD / MTCNN 的 NCHWC 变体）走的就是它。

触发条件：**纵向映射把最后几行的 `y0`/`y1` 顶到 `in_H` 或以上**，
即 `in_off_y + in_rect_height > in_H`，或下采样倍率让 `coord_y` 走到图外。
这不是"非法输入" —— 按 AGENTS.md「三个张量变体对越界 rect 的策略互相冲突」那条，
**MTCNN 家族是故意传入越界 rect 的**（检测框不做图像边界裁剪）。

> 与 `with_safeborder` 的对照：那一支**根本不钳**、按坐标直读，
> 契约是"调用方保证安全边界"。所以 `without_safeborder` 存在的意义**就是**
> 替调用方兜住越界 —— 它反而漏了一格，方向完全错了。

### 门禁

6 个入口，两种配置：

* **cfgA**：降采样、off 从 0 起、rect 严格落在图内
  —— `w_step >= 1` 时 `coord_x_ini >= 0`、末尾 `x1` 仍在图内，**两个变体都安全**、期望相同
* **cfgB**：`rect = 20x20` 比 `16x16` 的图还大、`out = 16x16`
  —— `w_step = 1.25`（**非整数**，否则 `sx` 恒为 0、钳位就测不出来），
  `coord_x(15) = 18.875` → `x0 = 18 > 15`，**钳位必然被走到**。
  `with_safeborder` 在这个配置下会越界读，所以只给 `without` 跑

  共 9 个用例：全对 9，有错 0，崩溃/搭建失败 0

变异测试（把参考的 y 钳位退回 `in_H`，即复现修复前的写法）：
**3/9 变红** —— 正好是三个 `without_safeborder` 的 cfgB 用例，
cfgA 与 `with_safeborder` 保持绿。

### 门禁自己也踩了两次"无效用例"

第一版 cfgB 是"上采样到图边"（`rect=16, out=12`），算下来 `x0=14, x1=15`，
**根本没越界**。
第二版换成 `out == in`（`w_step = 1`），`x1 = in_W` 确实越界，
但此时 **`sx` 恒为 0**（坐标全是整数），而被验证的 `x1` 正是**被 `sx` 加权**的
（`r0 = v00 + (v01-v00)*sx`）—— 钳不钳对结果毫无影响。
加上"把参考里的钳位去掉"这个变异之后仍然 **0 红**，才把这两层无效性暴露出来。

> **一个用例"跑通了"不等于"它测到了东西"。**
> 判据是"变异掉你想验证的那一处，看它会不会变红"。
> 推论：**设计用例时要问「如果被测的那一行被删掉，这个用例会红吗」**
> —— 让目标代码处在一个"它不影响结果"的位置上，用例就是废的。
> （与 CK.4「该红的没红 = 我的变异可能等价」同源，只是这次**用例本身**无效。）

### NCHWC 这一族的覆盖到此为止

  NCHWC 卷积（raw 族）      zq_nchwc_conv / zq_nchwc_conv8   CB
  NCHWC depthwise           zq_nchwc_depthwise                CF
  NCHWC 激活层              zq_nchwc_act                      CG
  NCHWC relu / eltwise      zq_nchwc_elt_relu                 CH
  NCHWC pooling             zq_nchwc_pool                     CI
  NCHWC batchnormscale      zq_nchwc_bn                       CJ
  NCHWC softmax             zq_nchwc_softmax                  CK
  **NCHWC resize**          **zq_nchwc_resize**                **CL**
  NCHWC packing / prepack   x86 上不可达（CD）

`ZQ_CNN_Forward_SSEUtils_NCHWC.cpp` 里被调用到的 NCHWC 层至此**全部有门禁**。
## 新增/变更：附录 CM —— 把 CL 的教训做成常驻门禁（同一函数内各轴的钳位上界必须一致）

### 变更文件

* `tools/check_clamp_asymmetry.py`（新增）—— 扫「同一函数内不同轴的边界钳位上界不一致」
* `tools/run_audit_checks.py` —— 登记 A15（自测）/ A16（普查）
* `audit_k3_20261001.md` —— 新增附录 CM
* `docs-changelogs/CHANGELOG_2026-10-02.md` —— 本节

**`ZQCNN/` 下没有改动任何生产代码。**

### 为什么立这条门禁

附录 CL 查出的那处**越界读**不是偶然的笔误，它是一个**可以被机械检测的形状**：

    x0[w] = __min(in_W - 1, __max(0, x0[w]));    // 上界 in_W - 1   对
    y0    = __min(in_H,     __max(0, y0));       // 上界 in_H       错，少减 1

同一个函数里 W / H / C 三条轴访问的合法下标上界**本该是同一个值**。
少减一个 `- 1` 就是**读到最后一行/列之后**。

它同时满足三个条件，正好是静态检查最擅长的：
1. **判据完全局部** —— 不必理解算法，只看同一个函数里几处 `__min/MAX` 的上界
2. **"对"的版本有明确参照** —— 同一条轴的另外两条，以及同仓另一族的对应实现
3. **后果严重** —— 而且是那种"平时不发作"的越界

第 3 条尤其值得记：CL 那个缺陷在 `rect` 不超出图像时完全不会发作 ——
`w_step = 1` 的整数倍坐标让 `x1 = in_W` 恰好与 `x0+1` 落进同一个像素、
而 `sx` 又是 0。**"平时好好的"正是它能活下来的原因**（与 CL.6 同一个坑）。

### 门禁做什么

* 扫 `ZQCNN/` 下所有 `*_raw.h` 与 `*_32f_align_c.c`
  （`layers_nchwc` / `layers_c` / `math` / 根目录），**共 54 个文件**
* 按 `void zq_cnn_…` 切函数，函数内按 `__min` / `__max` / `MIN` / `MAX` 分组
* 同一组里出现**两条以上不同的轴**、且**偏移量集合不相等** → 报出来
* `__min` 与 `__max` **分开比**（上界与下界不是一回事）

  扫了 54 个源文件，报出 0 处「同一函数内不同轴的钳位上界不一致」

### 自测样本（5 条，2 条"必须报" + 3 条"必须不报"）

**样本直接从真实缺陷反推** —— 前两条就是附录 CL 修好前后那两行：

  [OK ] 附录 CL 修复前：y0 漏了 -1（必须被抓出来）      期望 1 处，实得 1 处
  [OK ] 同一处修好之后：两轴都是 -1（必须不再报）      期望 0 处，实得 0 处
  [OK ] 故意不合格：y 写成 H - 2（必须被抓出来）        期望 1 处，实得 1 处
  [OK ] 只有一条轴 —— 没有可比对象（不该报）            期望 0 处，实得 0 处
  [OK ] __min 与 __max 是两回事，分别比（不该报）      期望 1 处，实得 1 处

> 关键在**前两条成对**：只有"修好之后不再报"这一条不够 ——
> 一个永远返回 0 的扫描器也能通过它。加上"修复前必须报"，
> 才同时约束了**灵敏度**与**特异度**。
> 这与 CL.6 是同一条（"该红的没红 = 变异/用例可能无效"），但落点在**工具**上。

### 顺带做的跨文件对照

| 家族 | NCHWC 侧钳位 | NCHW 侧 | 对照结果 |
|---|---|---|---|
| resize | 4 | 26 | **完全一致**（修好之后） |
| pooling | 12 | 12 | **完全一致** |
| eltwise / batchnormscale / addbias / convolution / depthwise / innerproduct | 0 | 0 | 两边都没有钳位 |

resize 那一行在**修之前**是不一致的（`in_H` vs `in_H - 1`），
修完变成一致 —— 这本身就是修法正确的一个独立佐证。
其余六族两边都没有钳位表达式，说明**这一类缺陷只可能出现在 resize 这一族**。

### 边界：这门禁抓不到什么

* 上界写对了但**漏了下界**（少写 `__max(0, ...)`）
* **完全没钳**（`with_safeborder` 那一支就是有意不钳的）
* 钳位用错了**变量**（拿 `in_W` 去钳 y）
* 与算法语义相关的错误（插值权重、步长公式等）

所以它是**补充**而不是替代：CL 那个越界读是靠 `zq_nchwc_resize` 数值门禁
发现的，这条扫描是在那之后把"同类形状"固化成常驻检查。

### 净结论

**本轮没有找到第二处同类缺陷**（54 个文件、0 处不对称）。

`python tools/run_audit_checks.py --quick` → A15/A16 均 OK，ALL CHECKS PASSED。
## 新增/变更：附录 CN —— NCHW 的 remap，x86 默认路径的插值权重用错了变量（首次跑就红）

### 变更文件

* `ZQCNN/layers_c/zq_cnn_resize_32f_align_c_raw.h`
  —— **修掉一处生产缺陷**：`remap_without_safeborder` 与
  `remap_without_safeborder_fillval` 的 **SIMD 版本**里，
  双线性的**横向**插值用了 `sy` 而不是 `sx`。共 **4 行**
* `tools/zq_nchw_resize_check.cpp`（新增）—— NCHW resize/remap 门禁（15 个真实符号）
* `tools/run_zqlib_checks.py` —— 登记 `zq_nchw_resize`（四处）
* `audit_k3_20261001.md` —— 新增附录 CN
* `docs-changelogs/CHANGELOG_2026-10-02.md` —— 本节

### 缺陷

    result0 = v00 + (in[y0][x1] - v00) * sy;       // 横向插值，这里应该是 sx
    result1 = v10 + (in[y1][x1] - v10) * sy;       // 同上
    sum     = result0 + (result1 - result0) * sy;  // 纵向用 sy，这个是对的

`sx` 在上面算出来了（`sx = coord_x - x0_f`），**一次都没被用到**。

**对照物（决定性）**：同一个 .c 里手写的
`zq_cnn_remap_without_safeborder_32f_align0`（line 484）写的是**正确**的：

    result0 = v00 + dx0 * sx;     // 对
    result1 = v10 + dx1 * sx;     // 对
    sum     = result0 + dy * sy;  // 对

即：**错的是 SIMD 那份，对的是标量那份** —— 而 x86 上 SSE/AVX 才是默认路径，
标量的 align0 在这台机器上根本不会被选中。

> 与附录 CL 正好相反：CL 是 NCHWC 错、NCHW 对；CN 是同一文件里
> **手写标量对、raw 头 SIMD 错**。两次都是"同一份数学被抄两遍、抄错的那份没人跑"。

### 门禁首跑结果

15 个真实符号（`nm` 核实；头里是 **34 个声明**，含重复 —— 又是"声明数 ≠ 符号数"）：

  align1/4/8 resize_nn                 各 2 个用例：对 2，错 0      全绿
  align1/4/8 resize_with_safeborder    各 2 个用例：对 2，错 0      全绿
  align1/4/8 resize_without_safeborder 各 2 个用例：对 2，错 0      全绿
  align1       remap / remap_fillval   各 2 个用例：对 2，错 0      标量，对
  align4       remap / remap_fillval   map 全在图内 FAIL 256/256
                                        map 一半出图 FAIL 187/256
  align8       remap / remap_fillval   map 全在图内 FAIL 512/512
                                        map 一半出图 FAIL 376/512

  共 30 个用例：全对 22，有错 8

**9 个 resize_* 入口全绿、只有 remap 的 SSE/AVX 两档全错** ——
这个形状本身就是最强的线索：问题不在参数、不在参考实现，而在"哪一份实现"。

修掉 4 行之后：**30/30 全对**。

### 门禁要点

* `sample_align_type`（resize 末尾那个参数）**两种都跑**：
  `== 1` 时坐标原点直接取 `in_off`（**不做**半像素平移），
  否则用 `0.5*step - 0.5 + in_off`。**名字里没有任何提示**，不看实现一定会漏
* `resize_nn` 用的是 `(int)(coord + 0.5f)`（**四舍五入**，不是截断）
* `remap` 两种配置：map 全在图内 / **一半出图**（验 without 的钳位与 fillval 的判据
  —— 后者用**未钳位**的坐标判 `0 <= coord <= n-1`，不满足就整像素写 fillval）
* 沿用 CB~CM：名字写全走函数指针表、逐格统计、fork 子进程并
  显式判"没读到结果文件" = 失败（CJ.4）

### 变异测试

把参考实现的 remap 横向权重退回 `sy`（复现修复前的写法）：
**12/30 变红** —— 正好是 6 个 `remap` + 6 个 `remap_fillval`（3 对齐 × 2 配置），
9 个 `resize_*` 入口的 18 个用例保持绿（变异只碰了 remap 的参考分支）。
**该红的全红、该绿的绿。**

### 生产可达性与危害

`zq_cnn_remap_*` 由 `ZQ_CNN_Tensor4D.cpp` 调用 —— 那是 **NCHW 张量类**，
x86 上**所有检测器走的都是它**。remap 用于把检测框/关键点映射回原图坐标。

`sy` 来自 `map_y`、`sx` 来自 `map_x`，**两者没有任何关系**，
所以横向缩放完全失控 —— 横向尺寸变化越大错得越离谱
（门禁里 align8 的 `map 全在图内` 是 512/512 全错，最差相对误差 1.32，
即数量级级别的偏差）。

### 这一族的覆盖

  NCHW resize / remap       zq_nchw_resize        CN（本轮）
  NCHW 卷积                 zq_nchw_conv          CE
  NCHW pooling / eltwise / lrn   已有门禁
  NCHW innerproduct         zq_innerproduct_check / zq_nchwc_ip   BN
  NCHWC 全族                CB / CF / CG / CH / CI / CJ / CK / CL

`deconvolution` / `deconvolution_gemm` / `dropout` 仍无门禁。
其中 deconvolution 只有分派器 `ZQ_CNN_Forward_SSEUtils.cpp` 引用，
且 **shipped 模型里一次都没用到**（`model/*.zqparams` 里搜不到 Deconv），优先级排在最后。
