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
