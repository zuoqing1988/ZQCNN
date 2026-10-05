# CHANGELOG 2026-10-06

（本日的第一条改动见下面「继承自 10-05 的收尾」与「v62 回归记录」。）

---

## 继承自 2026-10-05 的收尾

10-05 最后一轮（IX（五）/（六）、IX.23/IX.24、统计区补写）在本日树上的验证结果：

    v62 全量回归：55 个检查组，ALL CHECKS PASSED，RC=0

其中新增的两道常驻门禁在 v63 里第一次被完整跑过（v62 启动时还没注册）：

* A19/A20 `check_conv_overflow_guard`（附录 IX.19）
* A18 `check_mm_safety` 覆盖面扩大到整个 `ZQCNN/`（附录 IX.23）

---

## 记录：v62 全量回归

    python tools/run_audit_checks.py --with-build --warn-sweep --src-sweep \
        --bounds-sweep --ubsan-sweep --reachability --msvc-asan

    ALL CHECKS PASSED
    RC=0

    B 组 53/53 通过
    C5b（MNN 分叉头）all 7 headers compile, all guards present
    C11 MSVC /analyze 基线条数不变（新增的 `if (... == 0) return;` 早退没被报成新告警）
    C7 可达性：基线 36 条 -> 现在 36 条，无状态变化

---

## 新增/变更：IY —— `ZQ_Rodrigues` 补齐缺失成员，顺带翻出一条真算错

### 变更文件

* `3rdparty/include/ZQlib/ZQ_Rodrigues.h`（补 4 个成员 + 修 `R2r` 的 pi 因子）
* `tools/zq_rodrigues_check.cpp`（**新增**门禁）
* `tools/zqlib_probe_baseline.txt`（`ZQ_Calibration.h`: BROKEN -> OK）

### IY.1 `ZQ_Calibration.h` 引用的四个静态成员从来没存在过

`tools/probe_zqlib_headers.py` 一直把它列在 BROKEN 的 9 个里，
而那 9 个此前都被归为「需要 `windows.h` / OpenCV / GL，本机测不了」。
**只有这一个不是**：

    ZQ_Calibration.h:1851:22: error: 'ZQ_Rodrigues_r2R_fun' is not a member of 'ZQ::ZQ_Rodrigues'
    ZQ_Calibration.h:1919:22: error: 'ZQ_Rodrigues_r2R_jac'  is not a member of 'ZQ::ZQ_Rodrigues'

整个头**编不过**（任何平台）。它引用了 `ZQ_Rodrigues_r2R_fun` /
`ZQ_Rodrigues_r2R_jac`（2 参与 3 参两种形态）/ `ZQ_Rodrigues_R2r_fun` /
`ZQ_Rodrigues_autoscale` 四个不存在的成员。

前三个的语义由调用点**唯一确定**，底层 `ZQ_Rodrigues_r2R(r, R, dRdr = 0)`
本来就把 Jacobian 一并算好了（第三个形参），所以只是**带判空的薄包装**。
第四个 `autoscale` 没有仓库内 ground truth，按名字 + 用法定为
「把旋向量折到主值区间 `(-pi, pi]`」；这个定义**可验证**：
`R` 对旋向量 2*pi 周期，折与不折的 `R` 必须逐位相同（门禁第 2 条，实测 3.89e-16）。

### IY.2 真缺陷：`R2r` 在 θ = π 处漏乘了角度

「角度 = pi」分支原来写 `r[0] = x_abs * signs_mat[...]`，而
`x_abs = sqrt((R00+1)/2)` 是**轴的分量**，旋向量要的是**角度乘轴**。
于是 `r = (pi, 0, 0)` 往返一圈拿到 `(1, 0, 0)` —— 少了整整一个因子 pi，
**回代得到的旋转矩阵完全不是 R**（实测偏差正好 `pi - 1 = 2.1416`）。
补上 `* theta` 后往返偏差 0。

### IY.3 新门禁 `tools/zq_rodrigues_check.cpp`（ASan+LSan 与 UBSan 两轴）

    ok   r2R_fun 与 r2R 逐位相同            最大偏差 0
    ok   autoscale 之后 r2R 结果不变         最大偏差 3.89e-16
    ok   r2R_jac 的 Jacobian vs 中心差分     最大偏差 2.89e-10
    ok   R2r_fun 往返（r -> R -> r' -> R'）
    ok   四个薄包装的空指针都返回 false

> **一条要写下来的教训**：这条门禁的 Jacobian 判据**先后错了两轮** ——
> 先写成 `J[k*9+j]`（把 3x3 当 9x3 读，**直接越界**），
> 再写成 `J[j*9+k]`（行距取 9，而库里是 **3**：`dRdr[行*3 + 导数下标]`）。
> 两次都报成「库里的 Jacobian 算错了」，而**库里的 Jacobian 是对的**。
> 判据：**先确认被测对象的内存布局，再写判据**。

判据 4 也从「比 r 与 r'」改成「比 `r -> R -> r' -> R'`」：
`θ = pi` 处 `R(pi*n) = R(-pi*n)`（180° 旋转是对合），**轴的正负本来就二义**。

### IY.4 附带效果：可验证的头从 118 变成 127（+9）

`tools/zqlib_probe_baseline.txt` 里 `ZQ_Calibration.h` 从 `BROKEN` 变成 `OK`，
BROKEN 从 9 降到 8，且**剩下的 8 个全部是 `windows.h` / GL 依赖**（真的测不了）。

### 实测

    python tools/run_zqlib_checks.py rodrigues          -> 1/1 通过（ASan+LSan）
    python tools/run_zqlib_checks.py --ubsan rodrigues  -> 1/1 通过（UBSan）
    python tools/probe_zqlib_headers.py --check-baseline tools/zqlib_probe_baseline.txt
        OK: 127 -> 127 / 无回退、无新增   (rc=0)
    text/行尾卫生：OK: 755 text files, all strict UTF-8, no U+FFFD / line endings OK

---

## 新增/变更：IB —— ZQ_Calibration 第一次能被验，抓到成功路径上的内存泄漏

### 变更文件

* `3rdparty/include/ZQlib/ZQ_Calibration.h`（成功路径补 `delete[]`）
* `tools/zq_calibration_check.cpp`（**新增**门禁）

### IB.1 为什么值得验

4500 多行的头因为引用了四个**从来没存在过**的 `ZQ_Rodrigues` 成员而**整个编不过**（IY.1）。
补齐后它第一次能编能跑，而它是仓库里唯一一份 **ground truth 可自己构造**的数值算法：
投影模型 `proj_no_distortion` 公开，标定入口的参数含义由
`_calib_estimate_no_distortion_func` 完全写死。于是造一套真值（内参 + 两相机外参 + 3D 点），
用同一个 `proj_no_distortion` 生成 2D 观测喂回去，看它能不能还原。

顺带读出一条约定：`X3` 是**所有相机共用**的那一份（对每个相机传的是同一个 `X3` 指针），
`X2` 才按相机分段（`X2[N*2*cc+i]`）。

### IB.2 缺陷：成功路径上不释放 `hx` / `p`

    if(!ZQ_LevMar::ZQ_LevMar_Der<T>(...)) { delete []hx; delete []p; return false; }  // 只在失败分支
    avg_err_square = ...; memcpy(...); return true;                                    // 成功路径直接走

同文件里同形状的另外两个入口（`stickCalib_estimate_no_distortion_init:1145`、
`calib_estimate_int_rT_fix_k_with_init:2499`）两条路径都释放，**只有这一处漏了**。
LSan 实测：4 次调用漏 8 个块、3584 字节。

### IB.3 新门禁结果（ASan+LSan 与 UBSan 两轴都过）

    [init=truth]       avg_err_square = 0
       内参 0 / 外参 0 / 残差 0
    [init=perturbed]   avg_err_square = 2.34e-27
       内参 3.41e-13 / 外参 3.83e-15 / 残差 2.34e-27
    [init=perturbed x2] 内参 7.96e-13 / 外参 7.77e-15
    [pose]             位姿相对真值 1.35e-09，残差 1.27e-16
    退化输入：3D 点全在相机后面 -> 不崩、解全为有限值；点数为 0 -> 返回 false 不崩

**「初始化被扰动」那组比「真值当初值」强得多** ——
它同时验了目标函数**和解析 Jacobian**：能从 2% 的内参偏差与几像素的主点偏差
自己走回真值到 1e-13。

### 结论

除 IB.2 那处泄漏外，`ZQ_Calibration` 的数值是对的。

---

## 记录：v63 全量回归

    python tools/run_audit_checks.py --with-build --warn-sweep --src-sweep \
        --bounds-sweep --ubsan-sweep --reachability --msvc-asan

    ALL CHECKS PASSED
    RC=0

    57 个检查组，52 个显式 OK，0 个 FAILED
    B 组 **54/54 通过**（比 v62 的 53 多了 `zq_rodrigues`）

> 注：v63 启动之后 IY / IZ / IA / IB 才落地，所以紧接着又跑了 v64 拿最终树的单点。

### 本轮（B 组门禁）累计

从 v59 到 v63，B 组从 **51 道 -> 54 道**，A 组多出 A17~A20，
C5b 多出 3 条函数级守卫：

| 门禁 | 守的是 | 抓到过 |
| --- | --- | --- |
| `zq_deconv` | 三个 general 内核的索引映射（33 组形状） | —— |
| `zq_lstm` | LSTM_TF 内核 vs 独立参考实现（11 组） | 未对齐 SIMD 写（IX.3） |
| `zq_rodrigues` | Rodrigues 换算 + Jacobian vs 差分 | `R2r` 在 θ=π 漏乘角度（IY.2） |
| `zq_calibration` | 标定入口 vs 构造出来的 ground truth | 成功路径泄漏（IB.2） |
| A17/A18 `check_mm_safety` | 归约项数 / 栈数组对齐 / 分配判空 | ARM FP16 越界读 + 未对齐写 + 42 处不判空 |
| A19/A20 `check_conv_overflow_guard` | 每个「读 dilate」的卷积类都要溢出守卫 | NCHWC 那一族一处都没有（IX.19） |
| C5b +3 | MNN 分叉头的 BN 融合通道数守卫 | 那一族三处全漏（IX.14） |

---

## 记录：IC —— `ZQ_LSQRSolver` 的「不测」理由要更正

第十三轮的中危剩余项写着「依赖缺失、本机确实测不了的（`ZQ_LSQRSolver` 需要 taucs）」。
**这条已过时**：taucs 就 vendored 在同目录（`ZQ_taucs.h`），
`ZQ_LSQRSolver.h` 现在能独立编译，基线里已经是 `OK`。

真正的原因是**链接期**缺 BLAS：`ZQ_LSQRUtils::lsqr` 内部用 `cblas_dnrm2` / `cblas_dscal`，
门禁的 harness 里没有 CBLAS。而且它在仓库里只有 `ZQ_ClosedFormImageMatting.h`
一个 include 者，那个头又没有任何调用点 —— 整条链是死代码。

顺带记一条差点改错的地方：`_aprod` 的两个分支看着都像 bug
（`y += A*x` 没清零、`x += A'·y` 把 x/y 写反），但 `ZQ_LSQRUtils.h` 里附着的
契约注释写明 LSQR 就是**累加式**的（`y = y + A*x` / `x = x + A'·y`），
驱动侧在调用前已把 `u`/`v` 缩放过。**实现是对的。**

判据：**改之前先找到被调方的契约**，而不是按函数名 + 直觉推断。

---

## 新增/变更：ID —— H12「越界 rect 分歧」从「记为遗留」变成可判的两条断言

### 变更文件

* `tools/zq_resize_align_check.cpp`（**新增**门禁）
* `tools/run_zqlib_checks.py`（注册 `zq_resize_align` 的 EXTRA_SOURCES/LINK/INC/CXXFLAGS）

### 背景

H12 一直只有一句话、没有判据：「`Align0`/`Align256bit` 遇越界 rect 直接 `return false`，
`Align128bit` 却夹取后再算」。不修的理由是「MTCNN 全家依赖后者，改了会动检测结果」。

### 先把分歧的准确形状读出来

* `Align0::ResizeBilinearRect`（`ZQ_CNN_Tensor4D.cpp:253`）与 `Align256bit::` 同名函数：
  `if (src_off_x < 0 || ... || src_off_x + src_rect_w > W || ...) return false;`
* `Align128bit::ResizeBilinearRect`（`ZQ_CNN_Tensor4D.cpp:1108`）：
  把 rect 夹到 `[-border, 尺寸-1+border]` 再算 —— 它自己的注释写明为什么必须夹：
  MTCNN 的 NMS 检测框**不做边界裁剪**（16 处边界检查被注释掉），
  不夹就是**堆越界读**（实测纵向超出 34 像素）。
* 另外两个还硬拒，是因为 **MTCNN 的张量本来就是 Align128bit**
  （`ConvertFromBGR` 用 `ChangeSize(1,H,W,3,1,1)`）—— 那两条路**在生产里走不到**。
  所以分歧是**潜在的**，不是活跃的。

### 两条断言（都不是「保持现状」，是「把现状钉死」）

1. **rect 在界内时，三种对齐必须逐位相同** —— 同一个算法不能有三套分歧实现。
   4 组形状（含同尺寸走 `ROI`、缩小走 safeborder、放大贴边触发
   `can_call_safeborder = false`）**全部逐位相同**（最大偏差 0）。
2. **rect 越界时**返回值符合契约（0 / 1 / 0），且**夹取的结果与
   「显式传入那个被夹过的 rect」逐位相同**。

### 又一次「假阳性来自装置自己」

第一版报「夹取 != 显式传夹过的 rect，偏差 23.16」。原因：`run()` 无论成功与否
都会把 dst 拷进输出向量，而 Align0 / Align256bit 那两次**失败的调用复用了同一个
输出向量**，把前一次结果覆盖了。各自一个向量之后偏差 0。

> 与 IX.11 / IX.22 / IY.3 同源：**「实测不一致」要先排除「实测手段本身错了」。**

### 实测

    python tools/run_zqlib_checks.py resize_align          -> 1/1 通过（ASan+LSan）
    python tools/run_zqlib_checks.py --ubsan resize_align  -> 1/1 通过（UBSan）

---

## 记录：v64 全量回归（**最终树**的单点）

    python tools/run_audit_checks.py --with-build --warn-sweep --src-sweep \
        --bounds-sweep --ubsan-sweep --reachability --msvc-asan

    57 个检查组，52 个显式 OK，0 个 FAILED
    ALL CHECKS PASSED
    RC=0

    **B 组 56/56 通过**（起点是 51 道）：
        + zq_deconv        三个 general 内核的索引映射（33 组形状）
        + zq_lstm          LSTM_TF vs 独立参考实现（11 组）
        + zq_rodrigues     Rodrigues + Jacobian vs 中心差分
        + zq_calibration   标定 vs 自己构造的 ground truth
        + zq_resize_align  ResizeBilinearRect 三种对齐的一致性
    A 组：A17~A20 四道源码级门禁全过
    C 组：可编译性门禁 OK: 128 -> 128（比上轮 +1：ZQ_GLSLShader.h 变成可验证）
    C5b：all 7 headers compile, all guards present（9 条守卫）
    C11：MSVC /analyze 基线条数不变
    C16：BN/PReLU 接线门禁 OK
    D1/D2 双平台全量构建 0 error；D3/D4 双平台 sample 回归全过

这之后没有再改生产代码 —— 本轮（IV / IX / IW / IY / IZ / IA / IB / IC / ID）全部落在一个已验证的树上。

---

## 新增/变更：IE —— `typeid(T).name()` 判类型，21 个头在 GCC/Linux 上整族失效

### 变更文件

* `3rdparty/include/ZQlib/` 下 **21 个头、73 处**（逐处 `strcmp(typeid(T).name(), ...)`
  -> `std::is_same<T, float/double>::value`，并补 `#include <type_traits>`）
* `tools/check_typeid_name.py`（**新增**，门禁 A21/A22）
* `tools/zq_pcg_check.cpp`（**新增**，PCG 的最优性条件门禁）
* `tools/run_audit_checks.py`（注册 A21/A22）

### 缺陷

`typeid(T).name()` 返回的**字符串是实现定义的**：

| 编译器 | `typeid(float).name()` | `typeid(double).name()` |
| --- | --- | --- |
| MSVC | `"float"` | `"double"` |
| GCC / Clang（Itanium ABI） | `"f"` | `"d"` |

所以

```cpp
if      (strcmp(typeid(T).name(), "float")  == 0) { ... }
else if (strcmp(typeid(T).name(), "double") == 0) { ... }
else return false;                  // <-- GCC 上 T 是 double，恒走这里
```

在 **GCC 上每个函数都立刻 `return false`，一个数都算不出来，不报错也不崩**。

实测（gcc 9.4，`PCG` 求 SPD 系统最优解）：修前 `ret=false it=-1`、残差 0.9845（一步没走）；
修后 `ret=true it=n`、残差 **1.4e-16 ~ 3.6e-15**（迭代数恰好等于 n，
正是共轭梯度法在对称系统上的性质）。

### 影响面

`ZQ_PoissonSolver.h`(10) / `ZQ_PoissonSolver3D.h`(8) / `ZQ_PCGSolver.h`(8) /
`ZQ_ShapeDeformation.h`(6) / `ZQ_GridDeformation3D.h`(6) /
`ZQ_CameraCalibrationBino.h`(4) / `ZQ_StereoCalibration.h`(4) / `ZQ_GridDeformation.h`(4) /
`ZQ_DoubleImage.h`(4) / 其余 12 个 1~3 处。

分两类：
* **正确性受影响**：「`if / else if / else return false`」那一族 ——
  `PCG` / `PCG_sparse_unsquare` / `PCG_BQP` / `ZQ_taucs_ccs_matrix_time_vec` …
  **整个函数不工作**；
* **只影响性能**：「`if (double) {...} else { /* 同样上转 double 再算 */ }`
  那一族（`ZQ_SVD::Decompose`、`ZQ_MathBase::SVD_Decompose`）——
  GCC 上恒走 else，结果仍对，只是多一次上转。

> 这条正对上「windows 和 linux 都能完全跑通」这条硬要求：
> **同一份代码，Windows 上能算，Linux 上返回 false 且无任何提示。**

### 门禁

* **A21/A22 `check_typeid_name.py`** —— 报出所有
  `strcmp(typeid(T).name(), "float"/"double") ==/!= 0`，
  **特意放过只用来打印的那种**（否则会误报 3 处 `sprintf`）。
  变异测试：把 `ZQ_PCGSolver.h` 第一处改回 `strcmp` -> 门禁在第 94 行报出，rc=1。
* **`tools/zq_pcg_check.cpp`**（ASan+LSan 与 UBSan 两轴）——
  判据用**最优性条件**而不是"和另一个求解器比"：
  `PCG` 最小化 `0.5*x'Hx - f'x`，一阶条件就是 `H*x - f = 0`，
  所以**不需要任何 ground truth 文件**，而且**另一个实现也错的话不会一起错**。

```
=== ZQ_PCGSolver 最优性条件回归 ===
[dense n=4]        ret=true it=4   ||Hx-f||inf = 2.22e-16
[dense n=8]        ret=true it=8   ||Hx-f||inf = 1.388e-16
[dense n=12]       ret=true it=12  ||Hx-f||inf = 4.441e-16
[dense n=8 欠迭代] ret=true it=2   残差 0.9845 -> 0.04824（断言「下降」而不是「收敛」）
[laplacian m=4]    ret=true it=9   ||Hx-f||inf = 1.11e-15
[laplacian m=6]    ret=true it=16  ||Hx-f||inf = 3.553e-15
n=0 空矩阵不崩 / 全零矩阵（奇异）不崩且输出有限值
```

### 顺带验证：143 个头仍全部可编译

    python tools/probe_zqlib_headers.py --check-baseline tools/zqlib_probe_baseline.txt
        OK: 128 -> 128 / 无回退、无新增   (rc=0)
    python tools/run_zqlib_checks.py rodrigues / calibration  -> 各 1/1 通过

---

## 新增/变更：IF —— ZQ_FaceGroup 的 feat_dim==0 走成 fread(nullptr,...)，既有门禁从未执行到那条路径

### 变更文件

* `ZQlibFaceID/ZQ_FaceGroup.h`（读、写两侧各加一处 `if (feat_dim > 0)`）
* `tools/zq_facegroup_check.cpp`（新增 `OP_RT_DIMZERO` 用例，19 -> 20）

### 缺陷

守卫是 `feat_dim >= 0 && feat_dim < 65535`，**`feat_dim == 0` 被收下**；接着

```cpp
face_feats[i].ChangeSize(feat_dim);          // ChangeSize(0) 把 pData 置 0
flag = (feat_dim == fread(face_feats[i].pData, sizeof(float), feat_dim, in));
```

于是 `fread(nullptr, 4, 0, in)`。UBSan 实测：

    ZQlibFaceID/ZQ_FaceGroup.h: runtime error: null pointer passed as argument 1,
    which is declared to never be null

比 UB 本身更要紧的是**语义**：读回来的是一组**特征指针全为 0** 的记录，
后面任何 `feat.pData[k]` 都是空指针解引用。`WriteToFile` 那一侧一模一样。

### 为什么既有门禁没抓到

`tools/zq_facegroup_check.cpp` 的 `OP_RT_EMPTY` 取的是 `num = 0` 且 `dim = 0` ——
而内层 `for (int i = 0; i < num && flag; i++)` **一次都不跑**。
于是「`dim == 0` 但**真有特征记录**」这条路径**从来没被执行过**。

> 与 IW.2「判别形状落在退化形状上」同源：
> **一个退化形状会把另一个退化形状整个盖住**，两个都在，就都看不见。

### 修法（第一版改错了，记录一下）

先试的是把守卫改成 `feat_dim > 0`（与同族 `ZQ_FaceDatabaseCompact` 的
`dim <= 0 就拒` 对齐），**结果打破了既有的空组往返用例** ——
`OP_RT_EMPTY` 立刻变成「该收却拒了」。
说明 `feat_dim == 0` **本身合法**（空组），要修的不是守卫而是**传输那一行**。

最终：守卫保持 `>= 0`，读、写两侧各加 `if (feat_dim > 0)` 包住传输。
`ZQ_FaceSearchTarget::LoadFromFile` 走同一个 `ZQ_FaceGroup::LoadFromFile`，一处覆盖两边。

### 实测

    python tools/run_zqlib_checks.py facegroup          -> PASS
    python tools/run_zqlib_checks.py --ubsan facegroup  -> PASS（修前 runtime error）
    用例数 19 -> 20（新增 OP_RT_DIMZERO：num=3 且 feat_dim=0）


## 新增/变更：附录 IH —— ZQ_FaceIDPrecisionEvaluation 全套修复 + 新门禁 zq_lfw_eval

### 为什么是这一个头

附录 EG 把它从 `<opencv2\opencv.hpp>`（反斜杠）改成正斜杠，于是它在 Linux 上「能编过」了。
但附录 EH 之后的结论一直是：ZQlibFaceID 里**没有一个头 include 了 OpenCV**，
所以它们一道行为门禁都跑不了（本机 WSL 没装 OpenCV）。

这个头是那个结论的**唯一反例**，而且反例本身很说明问题：
它 include OpenCV 只为了 `cv::imread` / `cv::flip` 两个调用，
而 `EvaluationOnLFW` 的**全部逻辑**（解析 list 文件、抽特征、留一法定阈值、FAR/TAR 曲线）
就在这个头里 —— 是同目录里逻辑量最大的一个。
也就是说附录 EH 那份「零覆盖」名单里，**漏掉的恰恰是覆盖价值最高的那一个**。

做法：新增 `tools/opencv_stub/opencv2/opencv.hpp`（只给 cv::Mat / cv::imread / cv::flip 三样），
让头本身在 Linux 上编过并跑真行为。桩里的 imread 走**真 fopen** ——
「list 文件指向的图片全都不存在」正是触发 IH.1 那个空 vector 越界的最短路径，桩必须能造出这个场景。

### 查到的真缺陷（全部在 ZQlibFaceID/ZQ_FaceIDPrecisionEvaluation.h）

| 编号 | 位置 | 缺陷 |
| --- | --- | --- |
| IH.1 | `_compute_far_tar` | `singles` 为空时 `int dim = singles[0].feat.length;` **解引用空 vector 的第 0 号元素** -> SEGV |
| IH.2 | `_parse_lfw_list` | `part_num` / `half_pair_num` 只判 `> 0`，**无上界** |
| IH.3 | `_parse_lfw_list` | `2 * half_pair_num` 在 half_pair_num > 2^30 时 **int 回绕成负**；循环体一次不跑，连 fgets 的 NULL 检查也不执行，于是「一行数据都没有」被当成解析成功 |
| IH.4 | `EvaluationOnLFW` | `GetFeatDim()` 无校验，0/负值会让 `real_dim` 失去意义 |
| IH.5 | `EvaluationOnLFW` | 失败路径写 `return EXIT_FAILURE`，而函数返回 **bool**、EXIT_FAILURE==1 —— 「list 文件根本打不开」被报告成 **true**；七个 SampleEvaluationOnLFW* 直接把它当进程退出码 |
| IH.6 | `_parse_lfw_list` | 对不可信输入直接 `atoi`（越界是 UB） |
| IH.7 | `_parse_lfw_list` | 分隔段数既非 3 也非 4 的行被**静默丢掉**，头里说 10 折实际只解析出 3 对这种情况不可见 |
| IH.8 | `_compute_accuracy` | 留一法在 part_num==1 时把唯一那折也留掉，`mu.length==0` -> **所有 test score 恒为 0**，仍打印一行「0  0.00%」 |
| IH.9 | `_compute_far_tar` | `int all_num = (int)((long long)image_num*(image_num-1)/2);` —— 那个 (long long) 说明作者意识到会溢出，外面的 (int) 又掐了回去；image_num 到 65536 时 `vector<float>(负数)` 抛 length_error 且无人接 -> terminate |
| IH.10 | `_compute_far_tar` | `notsame_num == 0` 时打印 `cur_far_num / notsame_num` -> inf；`far_num[stage]` 全 0 让 cur_stage 每轮自增 |

### IH.1 的实测证据（修之前）

    ==1178329==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000028
      #0 _compute_far_tar ... ZQ_FaceIDPrecisionEvaluation.h:648
      #1 EvaluationOnLFW  ... ZQ_FaceIDPrecisionEvaluation.h:326

触发条件极其普通：**list 文件本身完全合法，只是 folder 写错 / 图片被挪走**，
每一对都被标成 invalid 并 erase 掉，`singles` 变空 —— 也就是最常见的用户错误直接崩进程。

### 门禁

新增 `tools/zq_lfw_eval_check.cpp`（tag `zq_lfw_eval`，B 组 57 -> 58），9 个用例：
图片全不存在 / header 与实际行数不符 / part_num=2000 万 / half_pair_num=2^30 /
正常 2x2 / 正常 2x2+use_flip / part_num=1 / 只有一对图片在 / list 文件不存在。
其中「list 文件不存在」是**对照项**（修之前就是 false），用来确认判据没写反。

### 踩到的坑（记录，避免重复）

1. **桩 imread 的造图路径必须和库自己拼的路径一致。** 第一版图省事写成 `img_%03d.jpg`
   放在 IMG_DIR 根下，结果**一张都读不到**，「正常 2x2」和「只有一对图片在」
   实际跑的是「全都不存在」，门禁照样报「没跑完」，看上去像库崩了。
   —— 附录 CA.3 的又一次：**观测手段没走到那条路径，结论就是假的**。
2. **RLIMIT_AS 与 ASan 不兼容。** 第一版想用「给子进程设 1GB 地址空间上限」
   来逼 `pairs.resize(2000万)` 分配失败，实测子进程只剩一行 `ERROR: Failed to mmap`。
   逐档试过 8/16/24/40/64/96/128/200 GB，**全部一样** —— 不是值不够大，是这条路本身不通。
   改用 **ru_maxrss 增量**：part_num=2000 万 -> resize 要 480MB，
   修好之后是在 resize **之前**就拒掉，涨 0。
3. **基类 `Init(const std::string model_name, ...)` 是按值传参上的顶层 const**，
   签名里被丢掉，桩里写 `const std::string&` 会得到「抽象类」编译错。
4. 桩 `cv::Mat` 的成员名就是 `data` / `step`（被测头按 `imgL.data`、`imgL.step[0]` 取），
   内部再叫 `_data` / `_step` 会直接编不过。

### 实测

    python tools/run_zqlib_checks.py zq_lfw_eval         -> 1/1 通过（9 个用例全对）
    修前同一道门禁：4 有错 + 1 崩溃（SEGV，栈已抓到）
    变异测试（把 IH.1 的空表守卫和 IH.5 的 return false 退回）-> 门禁重新变红
    cmake --build build_x64 --config Release --target 三个 SampleEvaluationOnLFW* -> RC=0，0 error
    python tools/check_text_encoding.py -> OK: 762 text files, all strict UTF-8, no U+FFFD
    python tools/check_line_endings.py  -> line endings OK

### 注意事项

- `EvaluationOnLFW` 的失败返回值语义变了（`true` -> `false`）。这是**修正**，不是回归：
  原先「解析失败」返回 `EXIT_FAILURE`==1 即 true，七个 sample 的失败分支永远进不去。
  但如果有外部代码依赖了「它总是返回 true」这个旧行为，需要一并检查。
- `_parse_lfw_list` 现在会在**一行都没解析出来**时返回 false（原先返回 true 并让下游崩）。
- `_compute_far_tar` 在 `image_num < 2` 或 `same_num/notsame_num` 为 0 时会**跳过并打印原因**，
  不再打一串 inf，也不再让 O(N^2) 分数表在超大 list 上把进程拖死。
- `tools/opencv_stub/` 只在 tools/ 的门禁里通过 -I 生效，主工程两个构建都看不到它。


## 新增/变更：附录 II —— MTCNN 五个变体的 SetPara / 多线程索引一致性（六个缺陷）

### 范围

`ZQCNN/ZQ_CNN_MTCNN*.h` 有**五份逐字拷贝**的 MTCNN 实现：
`ZQ_CNN_MTCNN.h`（主副本）、`_AspectRatio` / `_Interface` / `_NCHWC` / `ncnn`。
这一轮把五份一起过了一遍，挖出**六个**问题。

### 查到的真缺陷（全部逐条独立复核过，证据见下）

| 编号 | 位置 | 缺陷 | 修法 |
| --- | --- | --- | --- |
| II.1 | `ZQ_CNN_MTCNN_Interface.h` 多线程 lnet | `lnet[thread_id].Forward(...)` 之后读 blob 却用 `lnet[0].GetBlobByName(...)` —— 0 号线程此刻正在改写它的 blob，**数据竞争 + 读到别人 batch 的 landmark** | 改成 `lnet[thread_id]` |
| II.2 | 同上 | 下颌/眼周那 29 个点带一个**活的** `* 0.5`；单线程支路同一处是 `/**0.25*/`（注掉的），参考实现 `ZQ_CNN_MTCNN.h:1800/1802` 是 `/**0.5*/`（也是注掉的）⇒ **同一个 landmark 点在 thread_num=1 和 >1 下差一倍** | 去掉，两支都取 x1 |
| II.3 | 五个变体 | `mapH/mapW/maps` 按「通过 pnet_size 过滤的个数」建（**紧凑**下标），`task_scale_id.push_back(i)` 存 `scales` 的**全局**下标，消费端又用紧凑下标去取 `scales[i]` —— 三处混用，一个 scale 被过滤就同时错位，`maps[scale_id]` 那处是**越界写** | 在 `SetPara` 里**剔掉**会被过滤的 scale，让紧凑下标 == 全局下标 |
| II.4 | 五个变体 `SetPara` | 缓存失效条件只比 width/height/scale_factor，漏了 `pnet_size` / `min_size` / `special_handle_very_big_face` —— 而 `scales` 的生成恰恰依赖这三个。二次 SetPara 改 pnet_size 会用旧 scale，既几何错位又**打破 II.3 那个不变量** | 先存旧值再赋值，并把三个参数加进条件 |
| II.5 | `ZQ_CNN_MTCNN_Interface.h` 串行支路 | `thread_num <= 1` 分支（**不在任何 parallel 区内**）用 `omp_get_thread_num()` 去索引大小**恰好是 thread_num** 的 `pnet` / `task_pnet_images`。OpenMP 规定串行区嵌在调用方 parallel 区内时它返回**外层**线程号 => 越界写整对象 + 越界读 | 硬编码 `const int thread_id = 0;` |
| II.6 | `ZQ_CNN_MTCNN_ncnn.h` | **漏了**另外四份都有的 `pnet_size/pnet_stride = __max(1,...)` 夹取（上一轮修四份时漏的）。`pnet_size==0` 时整除 SIGFPE，`<0` 时 `while (minside > MIN_DET_SIZE)` 永不终止、scales 涨到 OOM | 补上 |

### 判定为笔误（而非取舍）的证据

- **II.1**：正确的写法就在**上一行注释里**（`//...lnet[thread_id].GetBlobByName("conv6-3")`），
  而且同仓 `ZQ_CNN_MTCNN.h:1782-1783` 用的正是 `lnet[thread_id].Forward` + `lnet[thread_id].GetBlobByName`。
- **II.2**：五个变体里**只有**本文件这两处有活的 `* 0.5`；
  紧邻的 else 分支（其余 77 个点）两支都是 x1，参考实现那一处是**注掉的**。三处指向 x1。
- **II.6**：另外四份在 2026-10-05 那轮就夹了（commit c1663d2），本文件没有 —— 
  又一次「一个副本有守卫、孪生副本没有」（IH.9 / BE.2 / IX.14 / conv_overflow 之后第 5 次）。

### 门禁

新增 `tools/check_mtcnn_setpara.py`，A23（自测）+ A24（普查）两条：
A23 五个变体的 pnet_size/pnet_stride 夹取、A24 `SetPara` 失效条件比较旧值、
A25 `pnet_images.resize(count)` 之后的不变量兜底、
A26 `*_Interface` 不得有活的 `* 0.5`、A27 `lnet[X].Forward` 之后最近的读 blob 必须也是 `lnet[X]`、
A28 `thread_num <= 1` 分支体内不得有 `omp_get_thread_num()`。

**为什么必须是源码级**：修复前后 4 个 MTCNN sample 的输出**逐字节相同**。
这既说明修得对，也说明 sample 对这六条是**瞎的** ——
仓内所有 sample 都 `thread_num=0`（被 `__max(1,...)` 夹成 1），
`lnet[0] == lnet[thread_id]`、`omp_get_thread_num()` 恒返回 0。

### 踩到的坑（都是门禁自己的，写下来免得重犯）

1. **第一版没剥注释**，A28 匹配到了**我自己写的修复说明**里那句
   「这里原来写的是 `omp_get_thread_num()`」，报出一个根本不存在的缺陷。
   修法：判定前统一 `strip_comments`（保留换行以免行号漂移）。
   附带好处是 A26 不再需要「数一下前面有没有 `/*`」这种脆办法 —— 
   `/**0.5*/` 是注释（对），活的 `* 0.5` 是代码（错），剥完就自然分开了。
2. **A27 第一版写成「Forward 之后 400 字符内」**。我加的那段解释注释约 500 字，
   把窗口撑破了，变异测试立刻发现 A27 抓不到。
   改成「**最近的下一个** `GetBlobByName` 必须同下标」，没有窗口，也就不受注释长度影响。
3. **A27 的下标字符类写成了 `[A-Za-z_][A-Za-z0-9_]*`**，于是 `lnet[0].GetBlobByName` 
   根本匹配不上 —— 变异测试把 A27 退回成 `lnet[0]` 之后门禁**照样全绿**。
   下标既可能是 `thread_id` 也可能是字面量 `0`，写死成标识符等于把最关键那个 case 排除在外。
4. **自测第一版只比「有没有报错」**。那样一个只缺 A23 的样本因为顺带也缺 A24/A25 
   会被判成「符合预期」—— 门禁自己糊弄自己。改成比对**触发的规则集合**。
5. 变异测试本身也踩了一次：我把 `/**0.25*0.5*/` 替换成 ` * 0.5;` 时只替换了
   `**0.25*0.5*/` 那一段，**留下了一个孤立的 `/`**，于是
   `keyPoint_ptr[...]/ * 0.5;;` 语法都不一样了，A26 当然不报。
   —— **变异测试写错 = 门禁看起来通过了**。改对之后 A26 立刻被抓到。

### 实测

    python tools/check_mtcnn_setpara.py --selfcheck   -> 9 cases, all as expected（RC=0）
    python tools/check_mtcnn_setpara.py               -> 5 个文件全 OK，合计 27 项（RC=0）
    变异测试：逐条退回 A23 / A26 / A27 / A28 -> 门禁**逐条变红**并点名对应规则
    Linux  -fsyntax-only（三个 MTCNN 头）              -> RC=0
    cmake --build build_x64 --config Release --target SampleMTCNN SampleMTCNN_Interface SampleMTCNN_AspectRatio -> RC=0，0 error
    A/B 对拍（归一化计时后）：SampleMTCNN / _NCHWC4 / _Interface / LoadFromCode **逐字节相同**
    python tools/check_text_encoding.py -> OK: 763 text files, all strict UTF-8, no U+FFFD
    python tools/check_line_endings.py  -> line endings OK

### 注意事项

- II.3 的修法（剔掉会被过滤的 scale）**依赖 scales 是升序**（原地压缩保序）。
  这一点写在了代码注释里；若将来改成降序，NMS 那边的假设也要一起改。
- II.5 只改了 `_Interface` 这一处。另外 5 处 `omp_get_thread_num()` 都在
  `#pragma omp parallel for num_threads(thread_num)` 里，`thread_id < thread_num` 有保证，**不用改**。
- `SampleMTCNN_AspectRatio` 在本机跑不通（`failed to open ../../model/handdet1-dw20-fast.zqparams`），
  是**既有**问题、与本轮改动无关（改前改后都一样）。
- II.2 的取值选择（x1 而不是 x0.5）依据是「三处一致」：else 分支、单线程支路、参考实现。
  如果后续拿到 LFW 级别的人工标注发现 x0.5 才对，那要改的是**三处一起**，不是一处。


## 追加：附录 II 第二批 —— II.7 ~ II.11（同一批 MTCNN 变体，五个新缺陷）

### 查到的真缺陷

| 编号 | 范围 | 缺陷 | 修法 |
| --- | --- | --- | --- |
| II.7 | **五个变体** | `SetPara` 里 `scale_factor = __max(0.5, __min(0.97, scale_factor));` 改的是**形参**，成员 `factor` 从构造函数 `factor = 0.709` 起**从没被赋过值**。后果两条，都是静默的：① 调用方传的 `scale_factor` **完全无效**（传 0.5 也照样按 0.709 建金字塔）；② `factor != scale_factor` 变成拿 0.709 跟**调用方传的原始值**比，只要没恰好传 0.709 条件**恒真**，于是每次 SetPara 都 `scales.clear()+pnet_images.clear()` 全量重建 | 先算 `new_factor`、留 `old_factor` 副本、在重建块之前 `this->factor = new_factor`，比较用 `old_factor != new_factor` |
| II.8 | `ZQ_CNN_MTCNN_AspectRatio.h` | Pnet **并行**分支用任务下标 `i` 判族（`if (i < ori_num)` / `else if (i < ori_num + xhalf_num)` / `int j = i - ori_num` / `int k = i - ori_num - xhalf_num`），而下面取张量用 `scale_id`。任务表按 scale 顺序追加，`scale_id <= i`，一个 scale 产出多个块时就分离 => `k` 可为负、`j`/`k` 可越界 => `std::vector::operator[]` 越界 -> 野张量上 `.ROI()` -> 野指针解引用。触发：1920x1080 + `thread_num>=2` + `min_face_size<=pnet_size`（`tasks(scale0)=40 > ori+xhalf=26`，`k=-26` 起） | 三处全改 `scale_id`（同函数单线程分支 `:905` 起就是正确样板） |
| II.9 | **四个变体** | 空任务守卫只判**外层** `task_src_off_x.size() == 0`，而外层大小是 `need_thread_num`、**恒 >= 1**，守卫永远不成立；真正会为空的是当前槽位 `task_src_off_x[pp]`。主副本的 Rnet 那一族早就是 `size()==0 \|\| task_src_off_x[pp].size()==0`，这几处是没同步的旧拷贝 | 21 处补上 `|| task_src_off_x[pp].size() == 0` |
| II.10 | **四个变体** | block 循环的 `#pragma omp parallel for schedule(dynamic, chunk_size) num_threads(thread_num)` **缺 `reduction(+:before_count, after_count)`** —— 两个共享 int 在 parallel for 里无锁 `+=` 是数据竞争，打印出来的数字每次都可能不同。主副本 `ZQ_CNN_MTCNN.h:932` 早就是带 reduction 的写法 | 四份补上 |
| II.11 | **五个变体** | `block_end_w[bb] = (bw == block_num - 1) ? scoreW : ...` 判的是**块总数**减一，应该是**每行的块数** `block_W_num - 1`（h 侧同理）。`bb == block_num-1` 只在「最后一行块的最后一列块」成立 => 只有那一个块延伸到 scoreW/scoreH，每行最后 `scoreW - block_W_num*width_per_block` 列、以及前 `block_H_num-1` 个行块的最后一整行**从来没被扫过**。不越界，是**召回率**缺陷：贴边的脸会漏 | 改用 `block_W_num - 1` / `block_H_num - 1` |

### 「孪生副本」计数

这一轮 II.6 / II.9 / II.10 三条都是「主副本修过、另外几份没同步」，
加上之前的 II.7（五份同款）—— 这是 IH.9 / BE.2 / IX.14 / conv_overflow 之后的**第 5、6、7 次**。
根源是这五份 MTCNN 实现是逐字拷贝，任何修改天然要改五遍。
本轮的应对不是「记得改五遍」，而是**门禁逐变体扫**（见下）。

### 门禁扩充：A29 ~ A33

`tools/check_mtcnn_setpara.py` 从 6 条规则扩到 **11 条**（A23~A33），自测从 8 例扩到 **17 例**：
A29 五个变体必须把夹取后的 `scale_factor` 落到成员 `factor`（且不得再有裸的 `scale_factor = __max(...)`）；
A30 AspectRatio 的**任务**循环里族判断/派生下标必须用 `scale_id`；
A31 有 `task_src_off_x` 的变体必须判当前槽位；
A32 `block_end_w/h` 必须用 `block_W_num/block_H_num`；
A33 block 循环 pragma 必须带 reduction 子句。

### 踩到的坑（门禁自己的，两条都是「过度敏感 / 过窄」）

1. **A30 第一版是全文扫 `if (i < ori_num)`，误报了 5 处。**
   那 5 处是 `for (int i = 0; i < total_scale_num; i++)` 的**尺度**循环 ——
   那里 `i` **就是**尺度下标，`if (i < ori_num)` / `maps[i]` / `mapH[i]` 全都正确。
   真正有问题的只有**任务**循环（`i < task_num` + `scale_id = task_scale_id[i]`）。
   改成**锚在 `scale_id = task_scale_id[i]` 上**往后看一个窗口，只在任务循环里判。
   —— 扫描器**过度敏感**和**过窄**一样有害：前者会让人养成「这条规则不准」的习惯，
   后者会让人以为覆盖了其实没有。
2. **A31 一开始对 `ZQ_CNN_MTCNN_ncnn.h` 报错**，说它一处 `task_src_off_x[pp].size()==0` 都没有。
   实际是**那个文件根本没有 `task_src_off_x` 这套变量**（换了自己的命名）——
   规则不适用。改成「文件里出现 `task_src_off_x` 才判」。
   这也是同一类错：把「不存在」当成「不满足」。
3. 自测骨架改成 `FULL` 公共前缀 + 每条只破坏一处，期望集合因此**可推导**，不再是拍脑袋写的。

### 实测

    python tools/check_mtcnn_setpara.py --selfcheck -> 17 cases, all as expected（RC=0）
    python tools/check_mtcnn_setpara.py             -> 5 文件全 OK，合计 47 项（RC=0）
    变异测试：逐条破坏 A29/A30/A31/A32/A33 -> 门禁**五条全部抓到**并点名
    Linux -fsyntax-only（MTCNN / AspectRatio / Interface / NCHWC 四个头）-> RC=0
    cmake --build build_x64 --config Release --target 四个 MTCNN sample -> RC=0，0 error
    A/B 对拍（归一化计时后）：
        SampleMTCNN          IDENTICAL
        SampleMTCNN_NCHWC4   IDENTICAL
        SampleMTCNN_Interface 候选 144->145 / Rnet 89->90，**最终检测数 15 不变**
        SampleMTCNNLoadFromCode 候选 1982->2002 / Rnet 977->984，**最终检测数 230 不变**
    python tools/check_text_encoding.py -> OK: 763 text files, all strict UTF-8, no U+FFFD
    python tools/check_line_endings.py  -> line endings OK

### 注意事项

- **II.11 会改变检测框数量，这是修复不是回归。** A/B 对拍里
  `first stage candidate count` 与 `run Rnet [...] times` 都**变大了**（144->145、1982->2002），
  因为原来根本没扫边缘那一条 cell；**最终 `candidate after nms` 一字未变**（15 / 230），
  说明多出来的都是边缘的重复候选、被 NMS 正常抑制掉了。
  `SampleMTCNN` / `SampleMTCNN_NCHWC4` 完全不变，是因为那两张图小到
`block_H_num == block_W_num == 1`，此时 `block_num-1` 与 `block_W_num-1` 恰好相等。
- **II.7 是行为变更**：以前 `scale_factor` 传什么都无效，现在会真的生效。
  仓内所有 sample 用的都是默认 0.709，所以实测输出不变；
  但如果有外部代码一直依赖「传了没用」这个行为来固定金字塔，行为会变。
- **II.8 的触发条件需要大图 + 多线程**，仓内 sample 全部 `thread_num=0`（夹成 1），
所以这条**没有任何运行时覆盖**，只能靠 A30 源码门禁钉住。
- `ZQ_CNN_MTCNN_old.h` 有同款的 II.9/II.10/II.11 形态，但它**不在任何构建里**
（全仓无 include），本轮**没有**改它 —— 改一个死文件只会增加 diff 噪音。
如果哪天它被重新启用，这三条要一起补。


## 追加：附录 II 第三批 —— II.12 ~ II.16（把子代理报告里剩下的 S6/S7/S9/S10/S11 收掉）

### 查到的真缺陷

| 编号 | 范围 | 缺陷 | 修法 |
| --- | --- | --- | --- |
| II.12 | **四个变体，30 处** | `GetBlobByName` 找不到就返回 **0**（`ZQ_CNN_Net.h:295-300`），而 `Init` 对 blob 名**零校验**、`SetPara`/`Find` 也不校验 —— 传一个概率层不叫 `prob1` 的模型进来，下一行 `score->GetH()` / `score->GetFirstPixelPtr()` 就是**空指针解引用**。同一个函数里 `keyPoint` **有**判空（`if (keyPoint != 0)`）、`score`/`location` 没有 —— 判据不一致本身就是信号 | 每处声明后加 `if (score == 0 || location == 0) { printf(...); continue; }` |
| II.13 | **五个变体** | `Init` 里 `thread_num` 只夹**下界**就 `pnet.resize(thread_num)` 并**逐份 LoadFrom** —— 每份都是一整套网络+权重。传 100000 就是 30 万份模型常驻内存，直接 OOM / 换页失败。`Init` 里除了 `ret` 之外没有任何资源预算 | 上界 128（超过 CPU 核数那么多份没有意义，每份独占一份 net 就是为了并行），超了打日志并夹到 128 |
| II.14 | **五个变体** | `SetPara` 公开、没有 `w/h` 的任何校验。`minside = min(w,h)` 为 0 时 `scales.push_back(pnet_size/minside)` 是 **+inf**，消费端 `(int)ceil(height*scales[i])` 是 float->int 的**未定义行为**，返回值再进 `if (changedH < pnet_size) continue;` —— 判据本身随之失效 | 入口加 `if (w <= 0 || h <= 0)` 夹到 1x1 |
| II.15 | **七个循环** | `special_handle_very_big_face` 的 `for (int tmp_size = last_size - 1; tmp_size >= pnet_size + 1; tmp_size -= 2)` 次数是 ~minside/2，**没有上界**。20000x20000 的图近 1 万个 scale -> `pnet_images.resize(1万)` -> 每个都分配 3x120x120x4 字节，**GB 级内存**。另外 `last_size > INT_MAX` 时 `int tmp_size = last_size - 1` 本身就是 UB | 循环条件加 `&& count < 2000` |
| II.16 | `ZQ_CNN_MTCNN_Interface.h` | bgr 版 `Find` 第一行就是 `if (width != _width || height != _height) return false;`，**张量版那个重载没有**。而 `scales`/`pnet_images`/`width`/`height` 全是按 SetPara 那对尺寸生成的，后面所有几何又都拿**成员** width/height 算 —— 换个尺寸的图进来不越界，但**所有几何都按过期尺寸算**，结果完全错乱且无任何提示。`ZQ_CNN_VideoFaceDetection_Interface.h` 走的就是这个重载 | 补 `if (input.GetW() != width || input.GetH() != height)` |

### 门禁扩充：A34 ~ A38（16 条规则，自测 22 例）

A34 `score`/`location` 的 GetBlobByName 之后必须判空；
A35 `Init` 的 `thread_num` 必须有上界；
A36 `SetPara` 入口必须有 `w <= 0 || h <= 0` 守卫；
A37 `special_handle_very_big_face` 的循环必须有 `count < N`；
A38 张量版 `Find` 必须有尺寸守卫。

**A35 一加上就抓到了真缺口**：它先报 `ZQ_CNN_MTCNN.h` / `_AspectRatio` / `_NCHWC` / `ncnn.h` 四份缺上界
（我第一版只改了 `ZQ_CNN_MTCNN_Interface.h`），补齐后才全绿。
这正是「门禁逐变体扫」比「记得改五遍」可靠的地方。

### 踩到的坑（这一轮三次，全是**同一个**根因：heredoc 吃转义）

1. `printf("...1x1\n")` 写进文件后 `
` 变成**真换行**，字符串字面量被劈成两行；
   修的时候又漏了 `w, h` 两个实参，gcc 报 `-Wformat=` 警告（不是错误，容易被忽略）。
2. 门禁里 `re.search(r'' + v + ...)` —— 写 `` 时少了一层，
   Python 把它解成**退格字符**，正则变成「退格 + score + == 0」，
   于是**所有** A34 点位都被误报成「没判空」，差点让我以为修复没生效。
3. 修 A38 时连续三轮：先把返回类型写死成 `bool`（换个返回类型就静默不执行），
   再是 heredoc 吃掉 `\s`，最后改成拼接还是不对。

**根因同一个**：在 `python - <<'PYEOF'` 里写正则和 C 字符串字面量，
转义要经过「shell 传递 -> Python 字符串字面量 -> 文件」三层。
AGENTS.md 已有的规矩（改 C/C++ 源码用 Edit/Write、Python 里要写真反斜杠用 `chr(92)` 拼接）
这次补上正则这一类：**写正则优先用不带转义的判据**。
A38 最后干脆不用正则了 —— `A38_FIND_MARK in text` 纯字符串判断，没有转义层。

### 实测

    python tools/check_mtcnn_setpara.py --selfcheck -> 22 cases, all as expected（RC=0）
    python tools/check_mtcnn_setpara.py             -> 5 文件全 OK，合计 72 项（RC=0）
    变异测试：逐条破坏 A34/A35/A36/A37/A38 -> 门禁**五条全部抓到**并点名
    Linux -fsyntax-only（四个 MTCNN 头）              -> RC=0
    cmake --build build_x64 --config Release --target 四个 MTCNN sample -> RC=0，0 error
    A/B 对拍（II.7~11 版 vs II.12~16 版）：四个 sample **全部 IDENTICAL**
    v65 全量回归（含 58 道 ZQlib 门禁）-> ALL CHECKS PASSED, RC=0
    python tools/check_text_encoding.py -> OK: 763 text files, all strict UTF-8, no U+FFFD
    python tools/check_line_endings.py  -> line endings OK

### 注意事项

- **II.12~II.16 全部是「只在非法输入 / 错误模型 / 错配尺寸时才触发」的守卫**，
  所以 A/B 对拍四个 sample 逐字节不变 —— 这是预期结果，不是「修复没生效」的证据。
  真要验它们得故意传错：给一个 blob 名不对的模型、传 `w=0`、给张量版 Find 换尺寸的图。
- **II.13 的上界 128 是个取舍**：真机器核数超过 128 的极少，
  而真有人要 256 份时会被夹到 128（会打日志）。
  如果将来支持超多线程，这个值要跟着 `omp_get_num_procs()` 走而不是写死。
- **II.15 的上界 2000 同样是取舍**：2000 个 scale 的分数表大约 2000*2000*4 = 16MB，
  可接受；真要处理超大图应该改用 `special_handle_very_big_face` 的**步进策略**
（比如按比例抽稀）而不是无脑加 scale。
- `ZQ_CNN_MTCNN_ncnn.h` **没有**张量版 Find（只有 bgr 那个，本来就有尺寸守卫），
  所以 A38 对它**不适用**、不报 —— 这是「不适用」不是「不满足」。
  ncnn.h 的 `Init` 形态也与另外四份不同（`pnet = std::vector<ncnn::Net>(thread_num)`，
  因为 `ncnn::Net` 的拷贝构造是 private，见附录 EW），所以补上界时是单独写的。


## 追加：附录 II 第四批 —— II.17 / II.18（S13 的像素缓冲契约 + #4 的框泄漏）

### 查到的真缺陷

| 编号 | 范围 | 缺陷 | 修法 |
| --- | --- | --- | --- |
| II.17 | **五个变体，全部 bgr 入口** | bgr 入口只校验宽高，**不校验像素缓冲本身**。`ConvertFromBGR` 里是 `bgr_row = BGR_img + h*_widthStep` 然后逐像素 `bgr_pix += 3`，于是 ① `bgr_img == nullptr` 立刻空指针解引用；② `_widthStep <= 0` 时 `bgr_row` 原地不动或**往回走**，越过缓冲区**前端**读；③ `_widthStep < _width*3` 时本行最后一个像素读到下一行，到了**最后一行**就越过整个缓冲末尾 —— 堆越界**读**。三种都是调用方一个笔误（传 nullptr、传 0、忘了算对齐填充），症状是随机崩溃或花屏，**没有任何提示** | 每个公开 bgr 入口（`Find` / `Find106` / `_Pnet_stage`）**各自**加 `if (bgr_img == 0 || _width <= 0 || _height <= 0 || _widthStep < _width*3) return false;` |
| II.18 | **四个变体，16 处** | Rnet/Onet 的 `ResizeBilinearRect` 失败时是**裸 `continue`** —— 于是**这一槽的框一个都没被评过**，却仍然 `exist=true`、`score` 还是**上一阶段的旧分数**（`task_secondBbox[pp]` 是从上一层整体拷来的），随后的汇总把它们全部并进下一阶段。这些框带着**偏高的旧分数**参加该阶段的 NMS，会把真正的框当 hero 抑制掉 | 失败分支先 `task_secondBbox[pp].clear()` / `task_thirdBbox[pp].clear()` 再 `continue` |

### 门禁扩充：A39 / A40（18 条规则）

A39 每个 bgr 入口都要有自己的像素缓冲守卫；A40 每个 Rnet/Onet resize 失败分支都要先清空该槽。

**A39 一加上就抓出三个我漏补的入口**：`AspectRatio` 的 `Find`、`NCHWC` 的 `Find` **和** `Find106`、
`ncnn` 的 `Find` —— 我第一版只在 `MTCNN.h` / `_Interface` 的 `Find`+`Find106` 和
另外三份的 `_Pnet_stage` 里加了，漏掉了「`Find` 委托给 `_Pnet_stage`」这条路上入口自身没有守卫。
补的时候顺带把守卫**也**放到公开入口（不只下游），理由写在代码注释里：
公开 API 的参数契约应该在入口就成立，不能依赖「我恰好调的那个函数会检查」。

### A39 的写法也踩了两次（都记在这里）

1. **第一版只判「文件里存在一处守卫」**，变异测试把 `Interface.h` 两个重载里的**一个**改坏，
   门禁照样全绿。同一函数的两个重载（`Find` / `Find106`）形状完全一样，最容易只改一处。
   改成**逐入口**判定。
2. **逐入口之后用固定 2000 字符窗口** —— 窗口跨进了下一个函数体，
   于是「这个入口没守卫、下一个有」又被判成全过。
   改成**窗口到下一个入口为止**。
   —— 这和 A27 那个 400 字符窗口是**完全同一个毛病**，一天之内犯了两次。
   规律：**任何"往后看 N 个字符"的规则都该改成"往后看到下一个同类为止"**。

### 另一个教训：源码门禁发现不了**编译不过**

给 MTCNN.h / _AspectRatio / _NCHWC 补 II.13 上界时，脚本用了 `re.subn(lambda m: NEW, ...)`，
`NEW` 里的 ``（缩进反向引用）**不会**在 lambda 形式下展开，
于是文件里留下了 13 行字面量 `thread_num = ...` 和 2 行 `	printf(...)`。
**门禁照样全绿** —— 因为 `thread_num = __max(1, thread_num)` 这个**模式**还在。
是随后的 `g++ -fsyntax-only` 报 `stray '' in program` 才抓到的。

也就是说：源码级门禁只能保证「模式在不在」，**保证不了「文件还能编」**。
后者只能靠每个平台各编一次。本轮每次改完都是「门禁 -> -fsyntax-only -> 两个平台各构建 -> A/B 对拍」
四步都过才提交，就是为了不让「门禁绿但编不过」溜过去。

### 实测

    python tools/check_mtcnn_setpara.py --selfcheck -> 22 cases, all as expected（RC=0）
    python tools/check_mtcnn_setpara.py             -> 5 文件全 OK，合计 82 项（RC=0）
    变异测试：A39（只改两个重载之一）/ A40（去掉 clear）-> 门禁逐条变红并点名
    Linux -fsyntax-only（四个 MTCNN 头）              -> RC=0
    cmake --build build_x64 --config Release --target 四个 MTCNN sample -> RC=0，0 error
    WSL make -j8 四个 sample                          -> 无 error
    A/B 对拍（v5 -> v6）：四个 sample **全部 IDENTICAL**
    python tools/check_text_encoding.py -> OK: 763 text files, all strict UTF-8, no U+FFFD
    python tools/check_line_endings.py  -> line endings OK

### 注意事项

- **II.17 的守卫放在每个公开 bgr 入口**，而不是只在 `ConvertFromBGR` 里加。
  `ZQ_CNN_Tensor4D.h` 的 `ConvertFromBGR` / `ConvertFromBGR2GRAY` **本身仍然不校验**参数，
  其他调用方（不经 MTCNN 的那些）仍然可能传 nullptr。
  这次没动它是因为 blast radius 太大（`ConvertFromBGR` 有 NCHWC 等多个 override 和大量调用方），
  单独开一轮处理更稳妥 —— 但这是个**已知的未修缺口**，不是「不存在」。
- **II.18 在当前仓库里不可达**：`Align128bit::ResizeBilinearRect` 只在
  `off > W-1+borderW` 或钳位后 `rect_w <= 0` 时返回 false，
  而 Pnet 侧产出的 `col1` 满足 `col1 >= -20 > -borderW`，所以这条路走不到。
  它是**防御性代码没写全**：一旦 rect 的来源变了（接 landmark/lnet、或者将来改了 Pnet 的裁剪），
  就是活 bug。修它不是为了现在，是为了那时候不用再查一遍。
- `ZQ_CNN_MTCNN_ncnn.h` 仍然**不在任何构建里**（全仓无 include，见附录 EW），
  本轮对它的改动只经过 `-fsyntax-only`（它需要 ncnn 头，本机 Linux 没有 Linux 版 ncnn 库，
  完整链接做不了）。这一点在 changelog 里重复记一次，免得以后误以为它被 CI 覆盖了。


## 追加：附录 II 第五批 —— II.19 / II.20（landmark 通道下标 + _select 选错框）

### 查到的真缺陷

| 编号 | 范围 | 缺陷 | 修法 |
| --- | --- | --- | --- |
| II.19 | **四个变体，14 处** | `int kp_num = __min(5, keyPoint->GetC() / 2);` 但下面**两半**都读：`keyPoint_ptr[i*sliceStep + num]` 与 `[... + num + 5]`。本行读到的最大下标是 `kp_num-1+5`，要不越出行必须 `kp_num <= C-5`：`C=4` -> kp_num=2、最大读 6 >= 4；`C=8` -> kp_num=4、最大读 8 >= 8；只有 `C=10` 才安全。越出行本身还在缓冲里（不算越界），但**最后一个样本**再读就越过缓冲末尾；而且即使不崩，取到的也是**下一个样本的坐标** —— 结果静默错乱 | 上限改成 `C - 5`，再 `__min` 到 5 并夹到 >= 0 |
| II.20 | **五个变体** | `_select` 是 `bbox.resize(limit_num)` —— 按**插入顺序**截断。`firstBbox`/`secondBbox` 的插入顺序是「先按 scale、再按 scale 内顺序」，而 scale 是从小到大走的，所以保留下来的恰好是**最小尺度**那批，而不是分数最高的。`SetLimit(r,o)` 的用途是给 Rnet/Onet 计算量封顶，封顶应该留最有希望的框 | 改成按 `score` 降序**稳定**取前 `limit_num` 个（`std::stable_sort` + 下标数组 + `swap`）。同分保持原插入序，否则同分框的相对次序取决于排序实现、输出不再可复现 |

### 门禁扩充：A41 / A42（20 条规则）

A41 有 `ppoint[num + 5]` 的地方，`kp_num` 必须是 `C - 5` 形式；
A42 有 `_select` 的地方必须按分数选、不得有 `bbox.resize(limit_num)`。

### 实测

    python tools/check_mtcnn_setpara.py --selfcheck -> 22 cases, all as expected（RC=0）
    python tools/check_mtcnn_setpara.py             -> 5 文件全 OK（RC=0）
    变异测试：A41（退回 C/2）/ A42（stable_sort -> sort）-> 门禁逐条变红并点名
    Linux -fsyntax-only（四个 MTCNN 头）              -> RC=0
    cmake --build build_x64 --config Release --target 四个 MTCNN sample -> RC=0，0 error
    WSL make -j8 四个 sample                          -> 无 error
    A/B 对拍（v6 -> v7）：四个 sample **全部 IDENTICAL**
    python tools/check_text_encoding.py -> OK: 763 text files, all strict UTF-8, no U+FFFD
    python tools/check_line_endings.py  -> line endings OK

### 注意事项

- **II.19 对现有模型是 no-op**：`conv6-3` 的 C 通常是 10 或 20，
  旧式 `min(5, C/2)` 与新式 `min(5, C-5)` 在这两种取值上结果**相同**（都是 5）。
  它只在 C < 10 的模型上才有差别 —— 而随仓模型都不是这种，
  所以「A/B 逐字节相同」是预期结果。
- **II.20 在当前仓库里也不生效**：`SetLimit` 的调用点在 sample 里是**注释掉的**
  （`SampleMTCNN.cpp:137`），所以 `_select` 根本没被调到。
  它是**未启用的公开 API**（`SetLimit` 本身是 public）—— 一旦有人打开这个开关，
  现在的行为就是「按插入序截断」。修它是为了那时不用再查一遍。
- `_select` 的 `width`/`height` 两个形参从头到尾没被用过（调用点传的是
  `input.GetW()`/`GetH()`），保留签名没动，免得影响外部调用。


## 追加：附录 II 第六批 —— II.21（调试开关只关不恢复）

### 缺陷

`ZQ_CNN_MTCNN*.h` 的 `_Pnet_stage` 单线程分支里：

    pnet[0].TurnOffShowDebugInfo();
    //pnet[0].TurnOnShowDebugInfo();      <- 恢复那一行**被注释掉了**，就挂在下一行
    _compute_Pnet_single_thread(input, maps, mapH, mapW);

两个问题叠在一起：

1. **只关不恢复**：用户 `TurnOnShowDebugInfo()` 之后，第一次 `Find` 就把 pnet 的调试
   **永久**关掉，之后再也开不回来。
2. **两条路径行为相反**：多线程分支（`_compute_Pnet_multi_thread`）**根本没有**
   `TurnOffShowDebugInfo()`。所以 `thread_num==1` 时 pnet 不打、`thread_num>1` 时照打 ——
   这不是取舍，是漏了。

根因是本类的 `show_debug_info` 和 `pnet[i].show_debug_info` 是**两份从不互相通知**的状态，
「该恢复成什么样」根本无从判断。

### 修法

* `TurnOnShowDebugInfo()` / `TurnOffShowDebugInfo()` 现在**同时传播**给 `pnet/rnet/onet/lnet`。
* `_Pnet_stage` 里按本类的 `show_debug_info` 记下原值，跑完再恢复。

### 门禁：A43（21 条规则）

`pnet[0].TurnOffShowDebugInfo()` 出现时，必须同时有 `pnet[0].TurnOnShowDebugInfo()` 恢复、
有 `const bool pnet_debug_was = ...` 存原值、且 `TurnOnShowDebugInfo` 里有 `pnet[i].` 的传播。
三条缺一即报错 —— 变异测试去掉恢复行，立刻被抓到。

### 实测：A/B 对拍**故意**有差异，而且这正是修复生效的证据

    SampleMTCNN          变化 97 行   SampleMTCNN_NCHWC4     变化 28 行
    SampleMTCNN_Interface 变化 265 行 SampleMTCNNLoadFromCode 变化 1128 行

**所有变化行都是 `<层级调试行>`**（`Conv layer:` / `DwConv layer:` / `BatchNorm layer:` /
`PReLU layer:` / `Innerproduct layer:` ...），**没有一行 `<` 是删减**，全是新增。
把层级调试行过滤掉之后，四个 sample 的检测结果与阶段计数**逐字节相同**：

    去掉所有层级调试行后：SampleMTCNN / _NCHWC4 / _Interface / LoadFromCode 全部 IDENTICAL

也就是说：`SampleMTCNNLoadFromCode.cpp:123` 那个 `mtcnn.TurnOnShowDebugInfo()` 
**以前是白调的** —— 它让 MTCNN 自己开始打阶段计时，但 pnet 那 800 多行层级调试
在第一次 `Find` 之后被永久吞掉了。现在恢复了。

### 注意事项

- **这是本轮唯一一个「A/B 对拍输出变多」的修复**。不是回归：
  加的是用户主动要求打开的调试输出，检测结果一个字节都没变（上面已验）。
- `tools/capture_sample_outputs.sh` 的 `NORM` 正则**不覆盖**层级调试行
  （它只归一化 `<T>ms` / `<GF>GF/s` 那几个数字），所以 `ab_diff_sample_outputs.sh` 
  对 `SampleMTCNNLoadFromCode` 的 A/B 会看到这上千行。
  归一化脚本已经会把耗时数字换成 `<T>ms`，但行数差异还在 —— 
  **这是预期的**，做 A/B 时要知道这一点，别当成数值回归。
- 传播是在 `Init` 之后才被调用的（sample 的顺序是 Init -> TurnOn -> 循环 Find），
  此时 net 全部建好，循环安全。如果有人在 `Init` **之前**调 `TurnOnShowDebugInfo()`，
  那时 vector 还是空的，循环不执行，`Init` 之后 pnet 的调试仍是关的 —— 
  这种调用顺序本身不合理，但不会被崩。
- `ZQ_CNN_MTCNN_AspectRatio.h` **没有 lnet 成员**（它走 xhalf/yhalf 三族分派），
  所以它的 `TurnOn/TurnOffShowDebugInfo()` 只传播 pnet/rnet/onet。
  这是 `-fsyntax-only` 抓到的（第一版照抄了四份里的写法，它多写了 lnet 那一行）。


## 追加：附录 IJ —— BBoxUtils NMS / SSD 解码契约（六个缺陷，含一个可达的堆越界读）

### 范围

`ZQCNN/ZQ_CNN_BBoxUtils.h`（759 行）是 **MTCNN / CascadeOnet / SSD / MXNET-SSD
四条检测线共用的**几何底座：`_nms`、`_refine_and_square_bbox`、`DecodeBBoxes*`、
`GetPriorBBoxes`、`JaccardOverlap` 都在这里。它此前**零行为门禁**，
而输入是「网络输出 + 模型文件」，两者都不可信。

### 查到的真缺陷

| 编号 | 位置 | 缺陷 | 修法 |
| --- | --- | --- | --- |
| IJ.1 | `ZQ_CNN_Forward_SSEUtils.cpp` `_detection_output`（SSD 主路径） | 只判 `len <= 0`，**从不把 `num_priors` 和三个 blob 的实际长度对账**。`num_priors` 来自**另一个张量**（Layer 里从 conf 的 H 推出来），而 Layer 只校验了 loc 的 C 和 conf 的 C，**没校验 prior 的 C**。`GetPriorBBoxes` 要读 `8*num_priors`，`prior_len` 只有 `4*num_priors` 时就是**堆越界读** | 照抄同文件 `_detection_output_MXNET` 里早就写好的守卫；prior 那项是 `num_priors*4*2`（bbox + variance 两半） |
| IJ.2 | `ZQ_CNN_BBoxUtils.h` `_nms`（单线程 + 并行两支） | IoU **混用两套面积约定**：交集用「含端点」（`+1`），而 `area` 的**所有**生产点都是「不含 +1」。同一个分母里两套口径 => IoU 本身不成立：12x12 算出 **1.42**（>1）、1x1 退化框算出 **-2**（`> threshold` 恒假 -> **永远不被抑制**）、3x2 分母 **0**（除零 -> +inf -> 误抑制一切）。「Min」模式更直接：`IOU / __min(area1, area2)`，零面积框除零得 +inf，而 **R-net / O-net 走的就是 Min** -> 一个零面积框抑制掉所有框 | 交集与面积统一到「不带 +1」，面积**就地重算**（不再直接用 `area` 字段做除数），分母再兜一次底 |
| IJ.3 | 同上 | `order`（来自**外部传入**的 `oriOrder`）只挡 `order < 0`，**不挡上界**。越界时 `boundingBox[order].exist = false` 越界**写**、`boundingBox[order].col1` 越界**读**。而 `ZQ_CNN_OrderScore` 的默认构造是 memset 到 0 —— **「漏填」会静默指向 0 号框**而不是报错 | 补 `order >= (int)boundingBox.size()` |
| IJ.4 | `BBoxUtils.h` 3 处 + MTCNN 25 处 | `it->area = (float)(row2 - row1) * (col2 - col1)` —— 减法在 **int** 里先算完再转 float，`|row2-row1| > 2^31` 就是 signed overflow UB（编译器可以假设永不溢出从而**删掉后面的检查**） | 先拓宽再相减，数值等价 |
| IJ.5 | `BBoxUtils.h` `DecodeBBoxesAll` | `if (find(label) == end()) { /*LOG(FATAL)*/ }` —— 判了**什么也不做**，下一行照样 `find(label)->second` 解引用 `end()`（UB）。这层保护**是假的**，唯一作用是让读代码的人以为这里被守住了 | 改成 `continue`（同仓 `Forward_SSEUtils.cpp:5111` 早就是这么写的） |
| IJ.6 | `Forward_SSEUtils.cpp` 调用点 | `GetLocPredictions` 的返回值被丢弃。它在 `share_location && num_loc_classes != 1` 时 return false 且**不 resize**，目前靠下游 `all_loc_preds.size() != num` 这个**二阶守卫**兜住 | 加返回值检查 |

### 判定为笔误（而非取舍）的证据

- **IJ.1**：同一份文件的两条 DetectionOutput 路径，`_detection_output_MXNET`
  早就有完整的 `num_anchors*4` 对账守卫（`Forward_SSEUtils.cpp:5270-5282`），
  主路径一条没有。
- **IJ.2**：同文件的 `JaccardOverlap`（`:710-736`）**内部是自洽的** ——
  它的交集和 `BBoxSize` 用同一个 `normalized` 开关。只有 `_nms` 跨了两套。
- **IJ.5**：`Forward_SSEUtils.cpp:5111` 的对应位置早就是 `continue`。

### A/B 对拍：**检测输出确实变了，这是预期的**

改之前先抓了基线（`/tmp/base_before/` + MTCNN 的 v8 输出），改之后：

    SampleSSD                  IDENTICAL
    SampleCascadeOnet          IDENTICAL
    SampleCascadeOnet_Interface IDENTICAL

    SampleMTCNN            first stage 45->51,  nms (159-->24) 变 (159-->27)
    SampleMTCNN_NCHWC4     nms (92-->11) 变 (92-->12),  (45-->5) 变 (45-->6)
    SampleMTCNN_Interface  final found num: 11 -> 12
    SampleMTCNNLoadFromCode  first stage 2002->1964,  after nms 230 -> 238

方向是**双向**的，不是单调变化 —— 这符合分析：旧公式在框够大时 IoU **偏大**（过抑制），
在退化框时 IoU **为负**（永不抑制），两个方向的错都存在。

**必须说清楚的事**：这里**没有 ground truth**，我不能声称「检测更准了」。
能说的是：① 旧公式可证明是错的（IoU>1、IoU<0、除零三种都能从公式直接推出）；
② 修完之后与同文件 `JaccardOverlap` 的约定一致；③ SSD / CascadeOnet 两条线**逐字节不变**，
说明这次改动只作用在 MTCNN 真正踩到退化框的那部分路径上。
**如果要用真值评估，得另接 LFW/WIDER 之类的标注集**，本仓没有。

### 口径选择是一个显式取舍

canonical MTCNN（`detect_face.py` / `nms.py`）用的是**含端点**口径：
`area = (x2-x1+1)*(y2-y1+1)`，交集也带 `+1` —— 两边都含端点，也是自洽的。
本轮选的是**不带端点**（连续坐标）这一套。
理由：① 改 `_nms` 一个函数的 blast radius 最小；
② `area` 字段有 28 个生产点、3 个读取点，动它牵连面大得多；
③ 两种口径**都自洽**，差别只在 1 像素的边界效应。
**这一点写下来是为了将来有人要切到 canonical 口径时知道该动哪里**：
要切就得把 28 个 `area` 生产点一起改成 `+1`，不能只改 `_nms` 一处。

### 门禁：新增 `tools/check_bbox_nms.py`（A25/A26，6 条规则，自测 12 例）

A1 `_nms` 的交集不得是 `+1` 口径、面积必须就地重算、分母必须兜底；
A2 `area` 不得出现 `(float)(row2 - row1)` 形态；
A3 `order` 必须有上界守卫；
A4 `find/end` 的分支体里必须有 `continue`；
A5 `_detection_output` 的三项长度对账（loc / conf / prior*2）必须齐全；
A6 `GetLocPredictions` 返回值必须检查。

### 踩到的坑（三次，同一根因：判据写出来但**扫不到**）

1. **A5 的 `num_priors` 守卫形态写错**：写成找 `num_priors > 0`，
   而源码里从来没有这种写法（守卫是 `num_priors <= 0` 就拒）——
   这条判据**恒假**、等于没判。
2. **A6 的正则忘了 `re.M`**：`^` 不加 MULTILINE 只匹配整个字符串开头，
   而调用点在文件中间，于是又一条**恒假**的判据。
   「扫不到」和「没问题」在报告里长得一模一样 —— 这是本会话第三次栽在这上面。
3. **A4 用正则匹配分支体失败**：`\{[^{}]*?\}` 分不出「判了 + continue」和
   「判了但什么都不做」—— **合格**样本里那个体里正好有 `continue;`，
   `}` 后面照样跟着 `find(label)->second`，于是合格样本被报成不合格。
   改成把分支体**取出来**（配对花括号）看里面有没有 `continue`。

另外这一轮又犯了**「不适用」当「不满足」**：A1~A4 最初是无条件判定，
于是「只含 `_nms` 的自测样本」被判成「也缺 find/end 守卫」。
四条规则都补上了前置条件（该构造在本文件里到底存不存在）。
这是本会话第二次犯（第一次是 A31）。

### 实测

    python tools/check_bbox_nms.py --selfcheck -> 12 cases, all as expected（RC=0）
    python tools/check_bbox_nms.py             -> 6 文件全 OK（RC=0）
    变异测试：逐条破坏 A1/A2/A3/A5/A6 -> 门禁**五条全部抓到**并点名
    python tools/check_mtcnn_setpara.py --selfcheck -> 22 cases, all as expected（RC=0）
    Linux -fsyntax-only（BBoxUtils / Forward_SSEUtils / 四个 MTCNN 头）-> RC=0
    cmake --build build_x64 --config Release -> RC=0，0 error
    WSL make -j8 七个 sample -> 无 error
    python tools/check_text_encoding.py -> OK: 764 text files, all strict UTF-8, no U+FFFD
    python tools/check_line_endings.py  -> line endings OK

### 注意事项

- **这是本会话唯一一个「检测输出会变」的修复。** SSD / CascadeOnet 两条线不变，
  MTCNN 四条 sample 的框数有小幅双向变化。详见上面「A/B 对拍」一节 ——
  要评估准确率变化需要标注集，本仓没有，别凭框数下结论。
- IJ.4 的 MTCNN 侧改动只把 int 减法换成 float 减法，**数值完全等价**，
  它不是上面检测数变化的原因。
- `ZQCNN_to_MNN/converter/source/ZQ_CNN_BBoxUtils.h` 是**第二份拷贝**（728 行，比主文件旧）：
  `_nms` 连 `thread_num` 形参都没有，因此也没有 2026-10-01 那轮的 OpenMP 修复；
  IJ.2 / IJ.4 / IJ.3 在它上面**逐条都在**。
  `ZQ_CNN_VideoFaceDetection_Interface.h:369-449` 还有**第三份** `_nms`（BBox106 版），
  `:384-388` 那个「`maxX` 既是 max-col 又被改写成交集宽」的复用也是主文件早修掉的形状。
  **这两份本轮没有同步** —— 它们各自有独立的调用链，同步要连各自的 sample 一起验，
  放到下一轮单独做，并在那轮补基线。


## 追加：附录 IK —— ZQ_CNN_VideoFaceDetection_Interface（六个缺陷，含一个堆越界读）

### 元发现

这个文件（965 行）此前**零门禁覆盖**：`grep -rn VideoFaceDetection tools/` 为空，
既不在 `run_audit_checks.py` 也不在 `run_sample_regression.sh`；
唯一的 sample 走 `cv::VideoCapture cap(0)`（摄像头），无头环境永远跑不到。
也就是说下面每一条都从未被任何自动检查碰过。

### 查到的真缺陷

| 编号 | 位置 | 缺陷 | 修法 |
| --- | --- | --- | --- |
| IK.1 | `:241-250` Stage-1 | `ZQ_CNN_BBox106 tmp_box;` 的 **`area` 从未被赋值**（构造函数 memset 到 0），然后 `boxes.push_back(tmp_box)` 把 area=0 的框喂进 `_nms`。而本文件自己的 `_nms` 走 **"Min"**：`IOU / __min(area1, area2)`，分母 0 -> `inter>0` 时得 **+inf**，`inf > thresh` 恒真 -> **任何与跟踪框有重叠的框都被无条件删掉**。也就是 Stage-2「全局检测」在有人脸跟踪时**形同虚设** | `tmp_box.area = cur_w * cur_h;` |
| IK.2 | `:733-737` `_filtering` | 无条件 `trace[i][0]`，而调用处只填到 `good_idx.size()`（`trace.resize(cur_box_num)` 但循环条件是 `i < cur_box_num && i < good_idx.size()`）。于是 `i >= good_idx.size()` 的是**空 vector**，`operator[](0)` 读 `_Myfirst` —— nullptr 则 SIGSEGV，残留堆指针则把 **940 字节**垃圾拷进 `results[i]`。差值 = 被 NMS 吃掉的跟踪框数 | 循环上界加 `&& i < (int)trace.size()`，分支体加 `if (cur_trace.empty()) continue;`（未被跟踪到的框原样放行，它们是 Stage-2 的结果，本就不该参与时序平滑） |
| IK.3 | `:204/:231/:574/:657` | 4 处 `GetBlobByName` 未判空。`GetBlobByName` 找不到返回 **0**，而 `Init` 对 blob 名零校验、param/model 路径全由调用方给。对照：`MTCNN_Interface.h:2012` 判了、`CascadeOnet_Interface.h:143-152` 判了、**本文件 `:726` 的 hpg 也判了** —— 只有这四处漏 | 各加 `if (x == 0) { printf(...); continue/return false; }` |
| IK.4 | `:74` Init | `thread_num` 只夹下界。而本文件的资源模型比 MTCNN 还重：`cascade_Onets` 按 thread_num 份装载**而且每份 Init 三遍**（onet_param/model 传了三次），再加 onets N 份、lnets106 N 份 = **5N 份 det3 常驻内存**。`thread_num = 10000` 就是 5 万份。对照 `MTCNN_Interface.h:129-133`（附录 II.13 刚加的 128） | 对齐加上界 128 |
| IK.5 | `:696-707` | `float transform[6];` **未初始化**，且 `CropImage_112x112_translate_scale_roll` 的 bool 返回值被丢弃。crop 失败时 transform 是栈垃圾、`task_hpg_images[0]` 留着上一帧尺寸 -> `center_and_rot` 全脏。随仓 sample 拿它算 `vir_zaxis` 再做 `pt_x = _x/_z`，`_z == 0` -> inf/NaN -> `cv::Point(inf,inf)` -> OpenCV 内部 UB | `float transform[6] = {0,...};` + 检查返回值 |
| IK.6 | `:146-149` `Find` | `results.clear()` 写在 `ConvertFromBGR` 的 **return 之后**，所以一帧转换失败时 `results` 保留**上一帧的内容**。随仓 sample 直接踩到：`if (!detector.Find(...)) { printf(...); }` 之后**无条件** `Draw(ori_im, thirdBbox106)` —— 于是在新图上画旧框 | `clear()` 提到最前面 |

### IK.1 与 IK.2 是咬合的

IK.1 让两个跟踪框互相 `inter/0 = +inf` -> 无条件互吃，于是存活数 K1 < 产出数 N1，
而 `results.size() - good_idx.size()` 正好等于 `N1 - K1`，IK.2 就触发。
**修好 IK.1 之后 IK.2 的触发概率下降，但没有消失** —— 正常 NMS 也会吃掉跟踪框。
所以两道都要修，不能因为修了上游就放过下游。

### 门禁：check_bbox_nms.py 扩到 A7~A12，文件列表加了本文件

A7 Stage-1 跟踪框必须赋 area；A8 `_filtering` 循环上界 + 空 trace 守卫；
A9 `GetBlobByName` **逐声明点**判空（窗口到下一个声明为止）；
A10 `thread_num` 上界（要守卫的**形状**，不是那个数字）；
A11 `transform[6]` 有初值 + CropImage 返回值检查；
A12 `results.clear()` 必须在 `ConvertFromBGR` 之前。

### 踩到的坑（这一轮五次，同一根因：判据写出来了但**扫不到**或**扫太松**）

1. **A9 第一版只判「文件里存在一处 `if (x == 0)`」** —— 变异测试把三处 keyPoint 判空
   全部改成 `if (false)`，门禁照样全绿，因为 hpg 那一处判空还在。
   改成**逐声明点**判、窗口到下一个声明为止。
   **这与 A39（「文件里存在一处 bgr 守卫」）是完全同一个毛病，一天之内第二次。**
2. **A10 只搜 `thread_num > 128` 这个数字** —— `if (false && thread_num > 128)` 照样匹配，
   等于没判。改成要求守卫的**形状**（`if (... thread_num > N)` 或 `__min(N, ...)`）。
3. **A9 的正则末尾多了 `\)`**：`if (hpg == 0 || hpg->GetN()*... < 9)` 的条件
   不在 0 处结束，于是 hpg 那一处被**误报成「没判空」**（假阳性）。
4. **A12 的正则括号数多了三个 `)`**（源码里只有两个），这条判据**恒假** ——
   变异测试把 `clear()` 挪回 `return` 之后也抓不到。
5. **门禁的 DEFAULT_FILES 里根本没有这个文件** —— A7~A12 写完之后扫描报「全过」，
   其实一条都没执行。**加了文件列表才真正开始跑**，然后立刻报出 A9 的假阳性。

1~4 都是「扫不到 / 扫错」，5 是「压根没扫」。同一个教训的五种表现：
**门禁全绿这件事本身需要证据，不能默认它扫到了它声称要扫的东西。**

### 实测

    python tools/check_bbox_nms.py --selfcheck -> 12 cases, all as expected（RC=0）
    python tools/check_bbox_nms.py             -> 7 文件全 OK，合计 12 项（RC=0）
    变异测试：逐条破坏 A7/A8/A9/A10/A11/A12 -> 门禁**六条全部抓到**并点名
    Linux -fsyntax-only（VideoFaceDetection + 依赖）-> RC=0
    cmake --build build_x64 --config Release --target SampleVideoFaceDetection_Interface -> RC=0，0 error
    python tools/check_text_encoding.py -> OK: 764 text files, all strict UTF-8, no U+FFFD
    python tools/check_line_endings.py  -> line endings OK

### 注意事项

- **这个 sample 没法做 A/B 对拍**：它要 `cv::VideoCapture cap(0)`（摄像头），
  无头环境跑不到；Linux 侧还缺 OpenCV 的 video 支持。
  所以 IK.1 的功能影响（Stage-2 被静默吃掉）**只有代码层面的论证，没有实测数字**。
- **`ZQ_CNN_CascadeOnet_Interface.h` 不是自包含的**：它用 `ZQ_CNN_Net`（`:113`）
  但只 include 了 `ZQ_CNN_Net_Interface.h`，所以**必须先 include `ZQ_CNN_Net.h`**
  才能编过。随仓 sample 恰好是那个顺序（`SampleVideoFaceDetection_Interface.cpp:1`），
  所以一直没人发现。写成 `-fsyntax-only` 时按自然顺序 include 就会炸。
  本轮**没有**改它（加一行 include 即可，但那是另一个头的卫生问题，单独做更干净）。
- 报告里还有几条**已验证不可达**的，没有动：`has_lnet106 == false` 时 `lnets106` 是空 vector
（看着必崩）实际被 `MTCNN_Interface::Find106:497` 的 `if (!has_lnet || !lnet_enabled) return false;`
先挡住了；`cur_key_cooldown` 的「未初始化」被 `:164` 的首帧分支保证读过至少一次。
  这两条写下来是为了以后有人改动那两处时不用重新推一遍。
- 报告提到的 `thread_num` 在本文件**完全不产生加速**（三个 vector 全部只用 `[0]`、
文件里没有任何 `#pragma omp parallel for`）—— 也就是说 IK.4 的上界 128 其实
是在给一个「1 份就够」的设计加保险。这是**设计问题不是缺陷**，
要真并行化是另一件事，本轮没做。


## 追加：附录 IL —— 把 `_nms` 的另外两份拷贝同步到主文件（6 处）

### 背景：`_nms` 一共三份

| 位置 | 状态（本轮之前） |
| --- | --- |
| `ZQCNN/ZQ_CNN_BBoxUtils.h` | 附录 IJ 已修 |
| `ZQCNN/ZQ_CNN_VideoFaceDetection_Interface.h:369-449`（BBox106 版，自带一份） | **未同步**，且还多两个问题 |
| `ZQCNN_to_MNN/converter/source/ZQ_CNN_BBoxUtils.h`（727 行，比主文件旧） | **未同步**，且还留着主文件 2026-10-01 就修掉的 OpenMP 数据竞争 |

后两份都**不在 A1~A3 那套门禁的作用范围内**（它们不走 `ZQ_CNN_BBoxUtils.h` 的 `_nms`），
所以 IJ 改完主文件之后它们并不会跟着变 —— 必须在门禁里**单独判**（A13/A14/A15）。

### 改了什么

**`ZQ_CNN_VideoFaceDetection_Interface.h`（BBox106 版 `_nms`）**

- IL.1 `order` 只挡下界 -> 补上界（同 IJ.3）。
- IL.2 交集与面积口径统一 + 分母兜底（同 IJ.2）。这一份比主文件还多一处形态问题：
  它把交集宽度**写回 `maxX`**、高度写回 `maxY`（函数级变量被当临时量复用），
  主文件 2026-10-01 那轮就是为了这个才把它们下沉成循环内局部量的。
  现在 `maxX/maxY/minX/minY/IOU` 全部是循环内局部，交集宽度另起 `inter_w/inter_h`。
  **这一点与 IK.1 是配套的**：IK.1 刚把本文件 Stage-1 的 `area` 从 0 补上，
  而 `Min` 模式的分母就是 `min(area1, area2)` —— 只修 IK.1 不修这里等于白修。

**`ZQCNN_to_MNN/converter/source/ZQ_CNN_BBoxUtils.h`**

- IL.3 四件事：
  * `if (thread_num == 1)` -> `<= 1`。`thread_num == 0` 会落进**并行**支路，
    而那里第一件事是 `ceil(box_num / thread_num)` —— **整数除零 SIGFPE**，
    紧接着 `num_threads(0)` 也非法（OpenMP 要求 >= 1）。主文件早就是 `<= 1`。
  * `IOU` / `maxX` / `maxY` / `minX` / `minY` 从**函数作用域**下沉成循环内局部。
    叠加上面的 `thread_num == 0` 落进并行支路，这段是**真会被执行到**的数据竞争。
  * 交集/面积口径统一 + 分母兜底。
  * 并行支路里的 `boundingBox.at(num)` 改成 `boundingBox[num]` —— `.at()` 越界抛异常、
    `[]` 越界是 UB，同一个 `num` 在两条支路上语义不同。
- IL.4 `it->area = (it->row2 - it->row1)*(it->col2 - it->col1);` ——
  这一份比主文件**更糟**：连 `(float)` 都没有，是纯 `int * int` 再赋给 float，
  溢出点比主文件早一步。已改成先拓宽再相减。

### 门禁：A13 / A14 / A15（15 条规则）

A13 有自己 `_nms` 的文件不得用 `thread_num == 1`；
A14 不得有函数作用域的 `IOU/maxX/maxY/minX/minY`（认「面积就地重算」这个标志，
**不认变量名** —— 主文件交集宽叫 `w`、MNN 那份叫 `inter_w`）；
A15 MNN 拷贝的 `it->area` 不得是 int 减法形态（认两种形态：`(float)(x-y)` 与纯 `x-y`）。

### 这一轮门禁自己踩的坑（又一次「扫不到 / 压根没扫」）

1. **A13/A14 的代码块被插在各文件分派的 `return` 之后** —— 永远执行不到。
   扫描报「全过」，而变异测试把 `thread_num <= 1` 改成 `== 1` 门禁**照样全绿**。
2. **A9 / A1 / A14 / has_nms 都把变量名写死了**（`float w =`、`A1_INTERSECT_PLUS1` 里的 `\sw\s=`）。
   修好之后变量改名了（`w` -> `inter_w`），这些判据当场变成**恒假**。
   凡是靠变量名识别的判据都要改成认「结构性标志」。
3. **门禁按 basename 分流，而 MNN 拷贝与主文件同名** —— 两份互相串台。
   改成传**规范化全路径**、按路径尾部判定。
   串台当场帮了个忙：它报出「MNN 拷贝的 find/end 没修」——那是真的（IJ.5 没同步过去），
   但理由是错的，不能靠这种巧合。
4. **A15 的 bad.append 是个 4 元组**，`for code, msg in bad` 直接抛 ValueError。
   门禁**确实红了**（RC=1），但崩在格式化消息上，诊断信息反而看不到。
   ——「红了」不等于「红得有用」。

1~4 都是同一件事的不同侧面：**门禁全绿/门禁变红这两个信号本身都需要证据**。
判据要能被变异测试证伪，输出要能在失败时被人读懂。

### 实测

    python tools/check_bbox_nms.py --selfcheck -> 12 cases, all as expected（RC=0）
    python tools/check_bbox_nms.py             -> 8 文件全 OK，合计 24 项（RC=0）
    变异测试：A13 / A14 / A15 逐条破坏 -> 门禁**三条全部抓到**并点名
    Linux -fsyntax-only：VideoFaceDetection + MNN BBoxUtils 两个头 -> RC=0
    python tools/check_text_encoding.py -> OK: 764 text files, all strict UTF-8, no U+FFFD
    python tools/check_line_endings.py  -> line endings OK

### 注意事项

- **MNN 拷贝仍然不在任何构建里**：主 `CMakeLists.txt` 没有 `ZQCNN_to_MNN`，
  而且它的 CMakeLists 文件名拼成了 `CMakelists.txt`（CMake 根本不找这个名字）。
  本轮对它的修改**只经过 `-fsyntax-only`**，没有任何运行时验证。
  之所以还是改了：同目录的 `ZQ_CNN_Layer.h` 上一轮同步过溢出守卫，说明这目录是
  **半维护**的；留着 `thread_num == 0` 的整数除零和 OpenMP 数据竞争不修，
  等于给「将来有人构建它」埋雷。
- **VideoFaceDetection 的 BBox106 版 `_nms` 同样没法做 A/B**：那个 sample 要摄像头。
  IL.1 / IL.2 只有代码层面的论证。
- 这两份拷贝里**其余**与主文件的差异（`overlap_count_thresh` 参数、
  `_filtering_iou` 的签名等）本轮**没有**逐一对齐 —— 目标是修**安全缺陷**，
  不是把三份文件改成完全一样。行为差异要靠 A/B 才能安全对齐，而这两份都做不了 A/B。


## 追加：附录 IM —— 头文件自包含性门禁（顺手修了一处真实的不可自包含）

### 缺陷

`ZQCNN/ZQ_CNN_CascadeOnet_Interface.h:113` 用了 `std::vector<ZQ_CNN_Net*> nets;`
（**具体**的 `ZQ_CNN_Net`），而它只 include 了 `ZQ_CNN_Net_Interface.h`
（里面只有抽象基类 `ZQ_CNN_Net_Interface`）—— 于是这个头**不是自包含的**：
按自然顺序 `#include "ZQ_CNN_CascadeOnet_Interface.h"` 就会报
`'ZQ_CNN_Net' was not declared in this scope`。

而随仓的 `SampleVideoFaceDetection_Interface.cpp` 恰好第一行是 `#include "ZQ_CNN_Net.h"`、
第三行才 include 本头 —— **顺序正好把它盖住**，所以一直没人发现。
写 `-fsyntax-only` 检查「这个头能不能单独编过」时当场就炸出来了。

修法：补一行 `#include "ZQ_CNN_Net.h"`。

### 门禁：`tools/check_header_selfcontained.py`（A27/A28）

对 `ZQCNN/` / `ZQlibFaceID/` / `ZQ_GEMM/` 下每个头生成一个**只 include 它自己**的 .cpp，
编到语法检查（不链接）。前置只有 `ZQ_CNN_CompileConfig.h` —— 它定义 `__max`/`__min`/`__int64`
这些**本项目自己的**可移植别名，是「平台前置」不是「依赖」，不算。

自测 4 例（阳性 + 阴性对照都有：自给自足 / 用了没 include 的类型 / include 了就自足 / 缺 `<vector>`）。
自测里第 4 条原来写的是「用 `__max` 而不定义」，但 `ZQ_CNN_CompileConfig.h` **本来就定义**它，
所以那条期望 rc=1 是**门禁自己写错了**，实测 rc=0 —— 换成真正缺的标准库前置。

### 首跑结果：20 个头失败，分类之后是 5 个真缺陷 + 2 个环境缺

第一版把两类混在一起报（20 个 FAIL），加上 `-I3rdparty/include/ZQlib` 之后剩 5 个，
另 2 个是本机 WSL 缺 caffe（已单列为 ENV，不判失败）。

仍然不自包含的 5 个（本轮**只报告、不修** —— 每一处都要看它到底缺什么、
在主工程里是靠谁补上的，改错会让某个 sample 编不过）：

- `ZQCNN/ZQ_CNN_MouthDetector.h`
- `ZQCNN/ZQ_CNN_PersonPose.h`
- `ZQCNN/ZQ_CNN_PersonPose2.h`
- `ZQlibFaceID/ZQ_FaceDatabaseMaker.h`
- `ZQlibFaceID/ZQ_FaceDetectorLibFaceDetect.h`

**这五个正是下一轮的审计目标**（`ZQ_CNN_PersonPose.h` / `_PersonPose2.h` / `_MouthDetector.h`
本来就在「零门禁覆盖」名单上；`_FaceCropUtils.h` 本轮**是**自包含的）。

### v66 抓到我自己引入的一处编译错误（已修）

v66 的 **D1 Windows 全量构建**失败：`ZQ_CNN_MTCNN.h(97,1): error C2017: 非法的表达式`。
那是 II.13 补 `thread_num` 上界时用 `re.subn(lambda m: NEW, ...)` 造成的 ——
lambda 形式下 ``（缩进反向引用）**不会展开**，文件里留下了 13 行字面量 `thread_num = ...`。

当时我已经发现并修掉了（`g++ -fsyntax-only` 报的 `stray '' in program`），
但 v66 是在**修复之前**启动的，它的 D1 跑在最后，正好赶上那个中间状态。

**这一条本身是 v66 起了作用**（它抓到的不是环境问题，是真的编译不过），
但也说明我之前只编了 MTCNN 相关的几个 target、不编全量 —— 
改一个被 `SamplesZQlibFaceID` 也 include 的头，就该跑全量而不是抽查。
现在 `cmake --build build_x64 --config Release`（**全量**）与 WSL `make -j8`（全量）都是 RC=0。

### 实测

    python tools/check_header_selfcontained.py --selfcheck -> selfcheck OK: 4 cases
    python tools/check_header_selfcontained.py -> 自包含 OK 39 / 失败 5 / 环境缺第三方库 2
    cmake --build build_x64 --config Release   -> **全量** RC=0，0 error
    WSL make -j8                                -> **全量** RC=0，0 error

### 注意事项

- 这道门禁**每次要跑 ~3 分钟**（每个头一次 `wsl` 调用），所以没有并进 A 组主流程，
  而是单独注册；A 组那 58 项还是 2 分钟级。
- 门禁会**把跳过的文件也报出来**（ENV 行），避免「跳过」变成「藏起来」。
  ncnn / caffe / opencv / ZQlib 这几类头本机编不过是环境问题，不是代码问题。
- 本轮**没有**修那 5 个不自包含的头。理由：它们缺的东西在主工程里是被别的头的
  include 顺序**顺手**补上的，要判断「正确的修法是补哪个 include」得看每个头的依赖链，
  而且改完必须编全量验（见上面 v66 那条教训）。留给下一轮单独做。


## 追加：附录 IM.2 / IM.3 —— 自包含性门禁自己有两个 bug，真实缺陷只剩 2 个

### 门禁的 bug 之一：`gcc | head -N` 让 gcc 收 SIGPIPE

第一版门禁把编译写成 `g++ ... 2>&1 | head -4`，然后取 `${PIPESTATUS[0]}`。
告警一多，`head` 先退出，**gcc 收到 SIGPIPE 以 141 退出** —— 而 PIPESTATUS[0] 拿到的
正是这个 141。于是「告警很多的头」被**误判成「不是自包含的」**：
`ZQ_CNN_MouthDetector.h` 和 `ZQ_FaceDatabaseMaker.h` 这两个**本来就自包含**的头就是这样被报成 FAIL 的。

改成：先把全部输出落盘、取 `$?`、最后才 `head`。

—— 这是本会话第 **7** 次「观测手段本身制造/销毁了信号」（CA.3 那条纪律的延续）。
前六次是：假路径、假文件名、被跳过的基线、SIGPIPE 型 rc、恒假判据、压根没扫。
共同点：**门禁红了不等于代码坏了，门禁绿了也不等于门禁扫到了东西**。

### 门禁的 bug 之二：环境缺与「不自包含」混在一起报

`ZQ_FaceDetectorLibFaceDetect.h` 缺的是 `facedetect-dll.h`（Windows-only 的第三方 SDK 头），
`ZQ_FaceRecognizer*MiniCaffe.h` 缺的是 `caffe/caffe.hpp` —— 都是**本机 WSL 没装**，
不是「用了某类型却没 include 它」。第一版把它们和真缺陷一起报成 FAIL，
把「环境缺」和「代码缺」混成一类。现在单列 ENV 行、不判失败，但**照样打印出来**
（避免「跳过」变成「藏起来」）。

### 真实缺陷：2 个头用了 `FLT_MAX` 却没 include `<cfloat>`

- `ZQCNN/ZQ_CNN_PersonPose.h:242`
- `ZQCNN/ZQ_CNN_PersonPose2.h:566`

`float max_weight = -FLT_MAX;`。随仓 sample 恰好在别处间接 include 了 `<cfloat>`，
所以一直编得过；单独编就报 `'FLT_MAX' was not declared in this scope`。
与 IM.1 同一族：**用到的宏/类型必须自己 include 它的定义**。

### 首跑与修完的对照

    第一版门禁首跑         : FAIL 20（含 2 个假阳性 + 15 个 -I 路径不全）
    补 -I3rdparty/include/ZQlib : FAIL 5
    修 SIGPIPE + 环境分类   : FAIL 2   <- 真实的两个
    补 <cfloat>            : **OK 56 / ENV 3**

### 实测

    python tools/check_header_selfcontained.py --selfcheck -> selfcheck OK: 4 cases
    python tools/check_header_selfcontained.py -> **自包含 OK 56 / 失败 0 / 环境缺 3**
    cmake --build build_x64 --config Release（全量）-> RC=0，0 error
    python tools/check_text_encoding.py -> OK: 765 text files, all strict UTF-8, no U+FFFD
    python tools/check_line_endings.py  -> line endings OK

### 注意事项

- **现在 56 个头全部自包含**。ENV 的 3 个（facedetect-dll.h / caffe x2）本机编不过是环境问题；
  它们在 Windows 上是能编的（`SampleFaceDetectorLibFaceDetect` 在主工程里），
  但本机 WSL 没有对应的 SDK 头，**所以门禁没能在那一侧验过它们**。
- 门禁每次跑 ~3 分钟（每个头一次 `wsl` 调用），已单独注册为 A27/A28，
  没有并进更快的那几组。


## 追加：附录 IN —— 姿态 / 嘴部 / 人脸裁剪（9 个缺陷 + 一道新门禁）

范围：`ZQ_CNN_PersonPose.h` / `ZQ_CNN_PersonPose2.h`（**逐字拷贝**）、
`ZQ_CNN_MouthDetector.h`、`ZQ_CNN_FaceCropUtils.h`（**GBK 编码**）。四个头此前**零行为门禁**。

| 编号 | 缺陷 |
| --- | --- |
| IN.1 | `MouthDetector` 的 `real_border_x/y` 可以是**负数** —— `one_face.off_x = col1`，而 MTCNN 的 `_refine_and_square_bbox` 边界钳位是**注释掉的**，脸贴边是常规输入。负值传给 `cv::Rect(Point,Point)` 会变成 `x<0` 的 ROI，`cv::Mat(image, rect)` 抛**未捕获**的 cv::Exception；同一个负值还让 `cur_box.col1 < real_border_x` 这个守卫**恒假** |
| IN.2 | `FaceCropUtils` 的 `fill_val` 形参被接受后**丢弃**，Remap 的填充值写死成 `0`；同文件 `:61` 的另一个重载传的是 `fill_val` —— 一份对一份错。可达性已核实：`ZQ_CNN_VideoFaceDetection_Interface.h:769` 明确传了 `-1`，被静默吞掉 |
| IN.3 | `PersonPose.h` 的 `points[54]` 只 `memset` 了 **51** 个 float（差第 17 个关键点）；`PersonPose2.h` 是 42/42 正确。第 17 个点没过阈值时那三个格子是**栈垃圾**，而 `num_points` 仍告诉调用方「有 18 个点」 |
| IN.4 | 两个头的成员 int 全部**未初始化**，而 `Init` 的 6 个 `return false` 都发生在 `GetInputDim` 赋值之前。调用方忽略 Init 返回值时 `Detect` 上来就除以 `pose_W` |
| IN.5 | 姿态侧 `pose_ptr = GetBlobByName(...)` 没判空，而**紧邻 100 行内的 SSD 侧判了** —— 同文件内的不对称 |
| IN.6 | `PersonPose.h` 的 `npts` 取自**调用方可控**的 public 字段 `num_points`；npts==0 时两个守卫与 0 比**恒假** -> 产出 `col1=1e9 > col2=-1e9` 的**反向框**，下一帧负宽负高进 `ConvertFromBGR`。`PersonPose2.h` 的 npts 由 half_mode 推导、永远 >= 1 |
| IN.7 | `PersonPose2.h` 的 `MapToFull` 跳过 4 个位置，其中 `full[9]` 保留了半模式的 `half[9]` 并被当成全模式的**膝盖** => 紧接着「没检到脚踝就扩框」的分支**永远走不到**，框底被截掉 |
| IN.8 | 四处 `ConvertFromBGR` + `ResizeBilinear` 返回值被丢弃（失败时 `temp_img` 停在**上一次**的尺寸）；同文件 `:107/:112/:116` 对同样的调用**全都检查了** |
| IN.9 | `PersonPose2.h` 的 `size_H*size_W*3` 是**纯 int 算术**，缺 `(__int64)` + `> 0x7FFFFFFF` 守卫；`PersonPose.h` 两处都有 |

**两份拷贝的差异汇总**（这六条都是「这份有、那份没有」或反过来）：

| 项 | PersonPose.h | PersonPose2.h |
| --- | --- | --- |
| `points[]` / memset | `float[54]` / **51** ❌ | `float[42]` / 42 ✅ |
| `size_H*size_W*3` 溢出守卫 | 有（2 处）✅ | **无** ❌ |
| `npts` 来源 | 调用方可控的 public 字段 ❌ | 由 half_mode 推导 ✅ |
| 关键点缺失时清零 | `Detect` 无 ❌；`DetectVideoSinglePerson` 有 | 两处都有 ✅ |
| 成员 int 未初始化 | 有 ❌ | 有 ❌（更多数量） |

### 门禁：`tools/check_pose_mouth.py`（A29/A30）

九个缺陷各一条判据。**IN.3 / IN.6 / IN.9 写成「两份都要有」** ——
否则 `PersonPose.h` 会因为**已经有**守卫而「通过」，正好掩盖 `PersonPose2.h` 的缺失。
自测 12 例（含阴性对照），逐条做过变异测试。

### 修的过程中自己犯的三个错（都记下来）

1. **`int ssd_C, ssd_H, ssd_W = 0;` 只初始化了 `ssd_W`。**
   C++ 里逗号声明只有**最后一个**有初值。写完 IN.4 看了一眼输出才发现 ——
   已经改成 `int ssd_C = 0, ssd_H = 0, ssd_W = 0;`。
2. **「往后看 N 个字符」的窗口第三次栽了同一种坑。**
   IN.7 的判据先用「有没有 else」、再用「else 之后 300 字符里有没有写 `other.points`」，
   结果我加的那段修复说明注释（约 500 字）又把窗口撑破了，**正确**的代码被报成不合格。
   改成**配对花括号**取块体，没有窗口。
   —— A27 的 2000、A39 的 2000、IN.7 的 300，**三次同一个错**。
   这条已经写进 `AGENTS.md` 第 26 条的对策里了：判据要认「结构性标志」。
3. **`ZQ_CNN_FaceCropUtils.h` 是 GBK 编码的。**
   用 UTF-8 读写直接 `UnicodeDecodeError`。它是上游 MFC 中文界面带来的文件，
   `check_text_encoding.py` 的白名单里写着「改成 UTF-8 会破坏 Windows 侧的中文界面，不要动」。
   门禁按 `gbk` 读它，注释也改成英文（避免再引入编码问题）。

### 实测

    python tools/check_pose_mouth.py --selfcheck -> 12 cases, all as expected（RC=0）
    python tools/check_pose_mouth.py             -> 4 文件全 OK，合计 14 项（RC=0）
    变异测试：IN.1 / IN.3 / IN.6 / IN.7 / IN.9 逐条破坏 -> 门禁逐条变红并点名
    cmake --build build_x64 --config Release（全量）-> RC=0，0 error
    WSL make -j8（全量）-> RC=0，0 error
    python tools/check_text_encoding.py -> OK: 765 text files, all strict UTF-8, no U+FFFD
    python tools/check_line_endings.py  -> line endings OK

### 注意事项

- **这四个头没有一个有 A/B 基线**：`SamplePersonPose` 系列要模型 + 视频源，
  本轮没做 A/B。IN.1 / IN.7 / IN.8 在**正常输入**下与修前**完全等价**
  （IN.1 夹到 0 之后 `real_border` 本来就 >= 0；IN.7 补的是 map_id<0 那 4 个**原本不写**的位置；IN.8 加的 `if (!...)` 在返回值 true 时是空操作），
  所以「修后没变化」是预期而不是「没验」。
- IN.3 的差异在**随仓 sample 上读不到**（`Draw14` 最大读 `13*3+2 = 41`，
  落在原来 memset 覆盖的范围内）—— 真缺陷 + 当前零覆盖。
- 报告里另有 L2~L7 七条 LOW（`ppoint` 有效性、`.names` 读不到时静默降级、
  `ssd_detector.Detect` 返回值丢弃、块内 `thresh` 遮蔽、`size()-1` 的符号回绕等），
  本轮**没有**改：它们要么不可达、要么纯可读性，改动收益低而回归成本高。


## 追加：附录 IO —— ZQ_FaceDatabaseMaker / ZQ_FaceDetectorLibFaceDetect（3 个已修 + 覆盖盲区）

范围：`ZQ_FaceDatabaseMaker.h`（1351 行）+ `ZQ_FaceDetectorLibFaceDetect.h`（225 行），
此前**零行为门禁**。

| 编号 | 缺陷 | 修法 |
| --- | --- | --- |
| IO.1 | `ErrorCode err_code;`（2 处）**未初始化**，而 `CropImage` 失败的分支会把它 push 进 `ErrorCodes`。触发：`CropImage` 返回 false，即 `_findSimilarity(5, ...)` 失败 = 检测出的 5 点退化（共线/重合）。后果：读未定值（UB），随后被 `%d` 打进 `err_log.txt`，产生**随机错误码**。同一函数里另外 3 条错误路径都设了 err_code，只有这条漏了 —— 是遗漏不是设计 | 加初值 `ERR_WARNING` |
| IO.2 | `_auto_detect_database` 的 `intptr_t lfDir;` **未初始化**，且那个 `if` 的**语句体是空的**（只剩一行注释掉的 printf）=> `_findfirst` 失败时 `lfDir == -1` 却照样走到 `_findclose(lfDir)`，对**无效句柄**调 API。跨平台对照：`#else` 的 Linux 分支**做了**保护，Windows 侧两处都没有 | 初始化为 `-1l` + `_findclose` 前加 `if (lfDir != -1l)` |
| IO.4 | `ZQ_FaceDetectorLibFaceDetect` 的 **GRAY 分支少加了 `rect_off_x`**。同一个 switch 里另外 6 个分支全都加了 —— **一份对六份错**。后果：灰度图 + `roi_min_x > 0` 时 ROI 采样整体左移，检出框和 landmark 全部错位（**不是内存越界**，读仍在界内）。仓内不可达（三个调用点全传 BGR），但这是 public API | 补上 |

### 这一轮自己犯的错（都是**补丁脚本**的错，不是被测代码的）

1. 第一版 IO.1 写了 `ErrorCode err_code = ERR_FACE_DATABASE_MAKER_OK;` ——
   **这个枚举值不存在**（枚举里只有 `ERR_WARNING` / `ERR_FATAL`）。
   写完没立刻编译，隔了几步才在 `grep` 里看到。全量构建会立刻抓，但没有。
2. 第一版 IO.2 把 `if (... == -1l)` 改成 `!= -1l` 想「让失败不进 do-while」，
   但 `else` 结构还在 —— 结果**失败时才进 do-while**，比原来更糟。
3. 第二版用「3 tab 缩进」去匹配 `_findclose(lfDir);`，
   而它是 4 tab 那行的**子串** => 两处都被改、又叠了一层守卫。
   第三次用 `.strip()` 去掉一行 tab 来「修」重复，**去错了那一行**，文件彻底乱掉。
   最后 `git checkout -- <file>` 恢复重做。

**这三次的共同点**：都在**没有立刻编译**的情况下继续叠加补丁。
正确做法是每改一处就跑一次全量构建 —— 三次加起来省下的时间远不如一次 90 秒的构建。

### 覆盖盲区（这一轮最该带走的一条）

`MakeDatabase(` / `MakeDatabaseCompact(` **零调用方**（`grep "MakeDatabase("` 全仓无命中）。
也就是说 `_make_database`、`_extract_feature_from_img`、`_extract_feature_from_box` ——
**整个检测器驱动的路径** —— 不被任何 sample 执行。
四个 `SampleFaceDatabase*` 只用 `*AlreadyCropped` 变体，恰好绕开了 `detectors[id]` 那一支。

同理，`_auto_detect_database` 的 `#else`(Linux) 分支**从不链接** ——
10 个 include 此头的 sample 全部包在 `#if defined(_WIN32)` 里。

**所以：本轮修的 IO.1 / IO.2 / IO.4 都没有运行时覆盖，只能靠源码门禁钉住。**
这一点必须与缺陷一起落进 changelog，否则下一个人会以为这条路径验过了。

### 报告里已确认干净、值得记下的几处

- `_load_feature_from_file`（`:809-846`）是 `audit_k3_20261001.md:223` 的 H20 缺陷，**已修复**：
  `fread` 返回值查、`feat_dim` 范围 `1..4096`、`ChangeSize` 后查 `pData == 0`、每条路径都 `fclose`。
  `.imgfeat` 正是不可信的磁盘文件 —— 这条最该有的守卫都在。
- `box = bbox[0]`（`:1171`）安全：每个 `FindFace` 要么 `clear()` 要么 `resize(num)`，**不会跨次累积**。
- 4 个 OpenMP 区域的 `id = omp_get_thread_num()` 索引**全部安全**：
  进并行区前把 `real_thread_num` 夹到 `min(detectors.size(), recognizers.size())`。
  副作用（调用方只传 1 个 detector + `max_thread_num=4` 会静默降到串行）**恰好挡住**了
  `ZQ_FaceDetectorLibFaceDetect::pBuffer` 的跨线程数据竞争。
- `(*handled)++` **确实**在 `#pragma omp critical` 里 —— 正好是 MTCNN 那边漏 `reduction` 的**正确写法对照**。

### 实测

    cmake --build build_x64 --config Release（全量）-> RC=0，0 error
    WSL make -j8（全量）-> RC=0，0 error
    python tools/check_text_encoding.py -> OK: 767 text files, all strict UTF-8, no U+FFFD
    python tools/check_line_endings.py  -> line endings OK

### 注意事项

- 报告里另有 A3（`-Wnarrowing`）、A5（`malloc` 的 int 乘法）、A6（固定 `0x20000` 结果缓冲区、
  容量从不校验 —— 但本仓只带了 `.lib`/`.dll` 没有源码，**无法确认 DLL 内部是否自己校验**）、
  A7（`_mkdir`/`imwrite` 返回值全不检查、失败仍返回 true）、A8（Windows 分支根目录不存在也返回 true）
  以及 B1~B6 六条 LOW，本轮**没有**改：
  A6 缺可验证的 DLL 源码，A7/A8 改起来要动多个调用点的契约，B 组要么不可达要么是卫生问题。
- 上面写的「本轮自己犯的三个错」是**补丁脚本**的错，不是被测代码的缺陷 ——
  但它们有共同教训，写在这里是因为下一轮还会用脚本改 C++。


## 追加：附录 IO 第二批 —— IO.1 统一化 + IO.4 narrowing + 新门禁 A31/A32

### 补上的两条

| 编号 | 缺陷 | 修法 |
| --- | --- | --- |
| IO.1（补） | `:171` / `:346` 那两处 `ErrorCode err_code;` 也加了初值。它们本来**每一条** `return false` 路径都赋过值（子代理已逐条核实），属于「已确认干净」；加初值是**零行为变化**的统一化 | 有初值之后门禁就不必去静态证明「每条路径都赋过值」—— 那既难又脆（加一条新分支就破），而一个初值永远安全 |
| IO.4 | `ZQ_FaceDatabaseMaker.h:1187` 的 `float center[2] = { image.cols*0.5, image.rows*0.5 };` —— `int * double` 得 double，在 braced-init-list 里窄化成 float，gcc 报 `-Wnarrowing`（HIGH 桶，`tools/warn_sweep_src.py` 会抓） | 改成 `0.5f`，数值等价、窄化消失 |

注：IO.4 一开始被我放进了 `ZQ_FaceDetectorLibFaceDetect` 那条判据里，
而它实际在 `ZQ_FaceDatabaseMaker.h` —— 变异测试立刻抓到（改了 libfacedetect 那个头，门禁不报）。
「门禁抓到了我自己的错位」正是变异测试存在的意义。

### 门禁：`tools/check_facedb_maker.py`（A31/A32，4 条规则，自测 7 例）

IO.1 所有 `ErrorCode err_code` 都必须有初值；
IO.2 `intptr_t lfDir` 必须初始化为 `-1l`，且每处 `_findclose(lfDir)` 的**上一行**
必须是 `lfDir != -1l` 守卫（按行判，不按窗口）；
IO.3 七个像素格式分支的采样指针**每个都必须带 `rect_off_x`**
（按「每个分支都带」写，不按「有没有一处带」—— 否则只修一处也会判过）；
IO.4 `float center[2] = {...}` 的 braced-init 里不得有会触发 `-Wnarrowing` 的 `int * double`。

变异测试：IO.1 / IO.2 / IO.3 / IO.4 逐条破坏 -> 门禁**四条全部抓到**并点名。

### 实测

    python tools/check_facedb_maker.py --selfcheck -> 7 cases, all as expected（RC=0）
    python tools/check_facedb_maker.py             -> 2 文件全 OK（RC=0）
    变异测试：IO.1 / IO.2 / IO.3 / IO.4 -> 门禁四条全部抓到
    cmake --build build_x64 --config Release（全量）-> RC=0，0 error
    WSL make -j8（全量）-> RC=0，0 error
    python tools/check_text_encoding.py -> OK: 767 text files, all strict UTF-8, no U+FFFD
    python tools/check_line_endings.py  -> line endings OK

### 注意事项

- 这两个头**零运行时覆盖**（见上面「覆盖盲区」一节），所以门禁是唯一防线。
  以后要验这些路径，得先给 `MakeDatabase(` 补一个调用方，或者单写一个探针 ——
  那是下一轮的事。
- IO.1 的统一化意味着「已确认干净」的那两处现在也**有**初值了。
  这不是把好代码改坏，是让「可以静态证明的性质」变多了一条。
