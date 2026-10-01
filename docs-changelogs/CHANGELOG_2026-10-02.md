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
