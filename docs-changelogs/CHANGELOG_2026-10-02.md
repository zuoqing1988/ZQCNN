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
（`-Wunused-result`）。判断用的是下一行的 `line[0] == ' '`，**功能上是对的**，
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
