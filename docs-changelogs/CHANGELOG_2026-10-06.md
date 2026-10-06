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

### 试过给零覆盖路径做运行时探针，**结论是这一轮不做**（记录理由，免得下一个人重复试）

IO 那三条修复一次都没被跑到过，所以本轮尝试写一个 `MakeDatabase` 的运行时探针
（替身 detector/recognizer，不加载任何模型，因此能进 ASan+LSan 常驻回归）。
写完、编过、ASan 下跑起来之后才看清三件事：

1. **Linux 上跑的分支，生产构建根本不链接。**
   `_auto_detect_database` / `_make_database` 的图像遍历在 `#if defined(_WIN32)` / `#else`
   两侧各一份，而 **10 个 include 此头的 sample 全部包在 `#if defined(_WIN32)` 里** ——
   也就是说 Linux 探针验的是**没有生产调用方的那一份**。
   要真正覆盖，得让 sample 里有一个 `MakeDatabase(` 调用方（在 Windows 侧跑），
   或者把探针做成 Windows 的 —— 这是**设计取舍**，不是快修能解决的。
2. **WSL 其实有完整 OpenCV 3.4.13**（`/usr/local/include` + `/usr/local/lib`），
   所以不需要扩展那个 `tools/opencv_stub`（扩展了反而会让 IH 那道门禁用的桩变形）。
   这一点纠正了我原先「WSL 没装 OpenCV」的判断 —— 那个判断来自
   `check_header_selfcontained` 把缺 `caffe` / `facedetect-dll.h` 的头归到 ENV，
   我顺势以为 OpenCV 也缺。**「A 被归到 ENV」不等于「B 也一样」**，
   这跟 A39/A41 那些「有一处不等于每处都有」是同一族。
3. **Linux 分支依赖 `ent->d_type == DT_REG`**（子代理记的 B5），
   而 `/mnt/d` 这个 9p/DrvFs 挂载上 `d_type` 未必可靠 —— 探针的造数逻辑
   会在「文件确实存在」与「代码认为它是普通文件」之间分叉。

**所以**：探针文件已删除，`tools/` 下不留半成品。
IO.1 / IO.2 / IO.4 的防线仍然只有源码门禁 A31/A32（4 条规则 + 7 例自测 + 变异测试全中）。
**这一条本身就是已知缺口，写在这里是为了让下一个人知道缺口在哪、为什么现在不补。**
补它的正确形态见上面第 1 点：给 sample 加一个 `MakeDatabase(` 调用方（Windows 侧），
而不是在 Linux 上补一个验不到生产分支的探针。


## 追加：附录 IQ / IR —— SSD / CascadeOnet（6 个缺陷）+ 「语句粘连」门禁

### 一个**元结论**先说，因为它决定了本轮的做法

    diff -w -B ZQ_CNN_CascadeOnet.h ZQ_CNN_CascadeOnet_Interface.h
    两个文件的主体**只有 4 处类型替换，零逻辑差异**。

所以「这两份拷贝之间逐条找差异」的答案是 **0 条** —— 值得做的方向在别处：

| 编号 | 缺陷 | 修法 |
| --- | --- | --- |
| IQ.1 | `ZQ_CNN_NSFW.h:1` 的 include guard 写成了 `_ZQ_CNN_SSD_H_` —— 与 `ZQ_CNN_SSD.h` **完全撞名**。任一 TU 同时 include 两者，第二个整份被跳过 -> `'ZQ_CNN_NSFW' is not a member of 'ZQ'`。`#pragma once` 救不了：它按**文件**生效，阻止 NSFW.h 的是那个撞名的 guard | 改成 `_ZQ_CNN_NSFW_H_` |
| IQ.2 | `ZQ_CNN_SSD.h` 的 `bool mxnet_ssd;` **未初始化**，而 `Init` 有**两条** `return false` 排在赋值之前；调用方忽略 Init 返回值继续 `Detect` 时读的是不确定值（UB） | `= false`，对齐 `ZQ_CNN_VideoFaceDetection_Interface.h:26-34` 的既有写法 |
| IQ.3 | CascadeOnet 两份副本的 `Find(bgr_img,...)` **缺少 `ZQ_CNN_SSD.h:59` 已经有的入参守卫**。`ConvertFromBGR` 在 `ChangeSize` 之后**无条件**解引用 `bgr_pix[0..2]`：`bgr_img==nullptr` 空指针解引用；`_widthStep<_width*3` 是**读调用方图像缓冲区越界** | 照抄 SSD 那四行，不发明新的 |
| IQ.4 | 两份副本都**丢弃 `Forward` 的返回值**。`ZQ_CNN_Net::Forward` 失败时只 printf 然后 return false，**blob 内存原样保留**；于是「曾经成功过、后来失败」时读到的是**上一轮的陈旧数据**，而 `Find` 还**返回 true**。`SampleCascadeOnet_Interface.cpp:71` 传 `nIters=10`，同一批 net 连跑 10 轮 —— sample 的 `if (!Find(...)) failed;` 抓不到 | 失败时 `results.clear(); return false;` |
| IQ.5 | SSD 的 `output.clear()` 排在**七条** `return false` **之后** —— 任何一次 `Detect` 失败，调用方仍读 output 就拿到**上一次成功调用的框** | 提到函数开头，并把后面那个变成死代码的删掉 |
| IQ.6 | SSD 的 `if (show_debug_info) net.TurnOnShowDebugInfo();` **只开不开** —— 形参是每次调用的，某次传 true 之后所有 Detect 都刷屏，类里也没有 TurnOff 出口 | 改成对称的 `else net.TurnOffShowDebugInfo();` |

**A/B 对拍：SampleSSD / SampleCascadeOnet / SampleCascadeOnet_Interface 三个全部 IDENTICAL** ——
这批是零行为变化，符合预期（IQ.3/IQ.4/IQ.5/IQ.6 都是「失败路径上原来不做、现在做」，
而 sample 走的全是成功路径）。

### 门禁：A33/A34（IQ，6 条规则，自测 9 例）

A33 四个头的 include guard 必须**全局唯一**（IQ.1 的判据就是「唯一」，不是「存在」）；
A34 SSD 的 `mxnet_ssd` 有初值、`output.clear()` 在第一个 `return` 之前、调试开关有 `else`；
A35/A36 CascadeOnet 两份副本的 bgr 入参守卫与 `Forward` 返回值检查。
逐条做过变异测试，IQ.1~IQ.6 全部被抓到并点名。

### 附录 IR：「语句粘连」门禁（本轮自己造的问题）

这一轮用 Python 脚本批量改 C++ 源码，**四次**因为替换串里少一个换行把两行粘成一行：
`return false;` + `int C, H, W;`、`}` + `const ZQ_CNN_Tensor4D* prob = ...` 等。

**先纠正我自己的一个错误判断**：我以为「语句粘连」会改变语义 —— **不会**。
`}` 后接一条声明、`return false;` 后接一条声明，C++ 都解析成**两条**语句，照样编过。
本轮 4 次里**唯一真的编不过**的那次，是分割脚本把 `int C, H, W;` 的首字母吃掉变成 `nt C, H, W;` ——
那是**标识符被截断**，编译器能抓，跟粘连无关。

所以 `tools/check_stmt_joins.py`（A35/A36）的定位是**可读性 / 一致性**，不是正确性：
粘连的行在 code review 里极容易滑过去（读起来像一行），而下一轮脚本再往这行里插东西时
更容易连锁出错。

**它还必须带白名单**：仓库**原有** 8 处同类写法（三个 MTCNN 变体的 `}  void SetLimit(`、
`VideoFaceDetection` 的 `}  if (IOU > ...)`、以及 `ZQ_FaceDatabase*` 里刻意的
`score_begin[pp] = s;  s += ...` 列对齐）。不排除的话这条规则会**永远红** ——
而永远红的规则等于没有规则（AGENTS.md 第 20 条）。
白名单按 **(文件, 片段)** 记配额；第一版只按文件记，同一文件里的两条白名单互相吃掉配额，
第二条被误报成「新增」—— 又是一次「规则写出来但没验」。

自测 7 例（正常 / return 后接声明 / `}` 后接声明 / 单空格不算 / 注释行不算 /
预处理指令不算 / 行尾注释不算），变异测试确认能抓到新引入的粘连。

### 实测

    python tools/check_ssd_cascade.py --selfcheck -> 9 cases（RC=0）
    python tools/check_ssd_cascade.py             -> OK（RC=0）
    python tools/check_stmt_joins.py --selfcheck -> 7 cases（RC=0）
    python tools/check_stmt_joins.py             -> 扫了 68 个头，0 个有新引入的粘连（RC=0）
    A/B：SampleSSD / SampleCascadeOnet / SampleCascadeOnet_Interface 全部 IDENTICAL
    cmake --build build_x64 --config Release（全量）-> RC=0，0 error
    WSL make -j8（全量）-> RC=0，0 error
    python tools/check_text_encoding.py -> OK: 767 text files, all strict UTF-8, no U+FFFD

### 注意事项

- **IQ 的六条在仓内都不可达**（三个 sample 的调用点全都检查了 Init 返回值、都传 BGR、
  都用 `ori_img.step[0]`），所以它们只有源码门禁、没有运行时覆盖。
- 子代理还报了几条**结构性问题**，本轮**没有**动：
  · `ZQ_CNN_CascadeOnet_Interface.h:124` 用了**具体类** `ZQ_CNN_Net*` 而非模板形参 ——
    附录 IM.1 补 include 治的是症状不是病因，抽象泄漏原封不动；
  · `ZQ_CNN_CascadeOnet_Interface::Find` **全仓零调用点**，
    `VideoFaceDetection_Interface` 里那 `3 × thread_num` 份 `cascade_Onets` 加载后从不推理
    （`thread_num=8` 就是 24 份 det3 白常驻内存）；
  · `ZQ_CNN_SSD.h` 的 `mxnet_ssd` 开关是**死代码**（两处分支逐字节相同）。
  这三条都要动公共 API 或跨文件契约，**超出本轮范围**，单独立项更合适。

---

## 追加：IH.11 —— `EvaluationPair` 的未定值拷贝（**门禁自己在 UBSan 下抓出来的**）

### 缺陷

`ZQlibFaceID/ZQ_FaceIDPrecisionEvaluation.h` 的内部类 `EvaluationPair` **没有默认构造函数**，
于是 `_parse_lfw_list` 里两个分支的 `EvaluationPair cur_pair;` 之后
`idL` / `idR` / `flag` / `valid` 全是**未定值**；
紧接着 `pairs[i].push_back(cur_pair)` 走**隐式拷贝构造**，那一步就把未定值读了一遍。

### 是谁抓到的

`zq_lfw_eval` 门禁（附录 IH 我自己加的那道）。**ASan 那一轴完全绿** ——
拷贝未定值不是越界也不是泄漏，ASan 没有对应的检查。
**UBSan 那一轴**报得很直白：

```
ZQ_FaceIDPrecisionEvaluation.h:25:9: runtime error: load of value 37, which is not a valid value for type 'bool'
    #0 ... EvaluationPair::EvaluationPair(EvaluationPair const&) ZQ_FaceIDPrecisionEvaluation.h:25
    #4 std::vector<EvaluationPair>::push_back(...)
    #5 ZQ_FaceIDPrecisionEvaluation::_parse_lfw_list(...) :533
```

值每次都不一样（37、115……），因为它读的是栈垃圾。

### 影响面

`valid` / `flag` 在**所有**使用点之前都会被重新赋值，所以结果碰巧是对的；
但「拷贝未定值」本身就是 UB，编译器有权基于它做任何假设。

修法：给 `EvaluationPair` 一个默认构造函数，把四个 POD 成员都初始化。

### 这条记录本身的意义

它是附录 CA.3 那条纪律的**正面例子**：同一道门禁，ASan 轴全绿、UBSan 轴抓到真缺陷。
「回归全绿」这件事，**取决于你开了哪几条轴** ——
本会话的 v67 之所以在 C6 组失败，正是因为它跑了 `--ubsan-sweep`，
而 v65/v66 跑的时候这条门禁**还不存在**。

### 实测

    python tools/run_zqlib_checks.py lfw_eval            -> PASS（ASan）
    python tools/run_zqlib_checks.py --ubsan lfw_eval  -> PASS（UBSan，修前 FAILED）
    cmake --build build_x64 --config Release（全量）-> RC=0，0 error
    WSL make -j8（全量）-> RC=0，0 error
    python tools/check_text_encoding.py -> OK: 770 text files, all strict UTF-8, no U+FFFD
    python tools/check_line_endings.py  -> line endings OK

### 本轮新增门禁一览（A 组现 36 条 / 9 个工具）

| 门禁 | 规则 | 附录 |
| --- | --- | --- |
| `zq_lfw_eval_check.cpp`（B 组第 58 道） | 9 个用例 | IH |
| `check_mtcnn_setpara.py` | 21 条（A23~A43） | II |
| `check_bbox_nms.py` | 15 条（A1~A15） | IJ / IL |
| `check_pose_mouth.py` | 9 条 | IN |
| `check_facedb_maker.py` | 4 条（A31/A32） | IO |
| `check_header_selfcontained.py` | 每个头单独编（A27/A28） | IM |
| `check_ssd_cascade.py` | 6 条（A33/A34） | IQ |
| `check_stmt_joins.py` | 语句粘连（A35/A36） | IR |

每一道都配了**自测**（含阴性对照）与**变异测试**；
IR 那道额外带一份 8 处原有站点的白名单，理由写在文件头 ——
永远红的规则等于没有规则。


## 追加：附录 IR.1 / IR.2 —— NCHWC 检测线的两处「一份对 N 份错」

范围：`ZQCNN/ZQ_CNN_Forward_SSEUtils_NCHWC.cpp`（5704 行）+ `ZQ_CNN_Tensor4D_NCHWC.h`（700 行）。

### 先说**最想确认的那件事**：对齐违规

**这一族 NCHWC 没有 MTCNN 那种未对齐 `_mm256_store_ps` 问题。** 对齐契约在张量层自洽：

- `rawData` 由 `_aligned_malloc(needed + 64, 32)` 分配；
  `firstPixelData = rawData + borderH*widthStep + borderW*align`，其中 `widthStep = realW*align`。
  **NCHWC8**：`align=8`，所有偏移都是 8 float = 32 字节整数倍 -> `_mm256_load_ps/store_ps` 安全；
  **NCHWC4**：`align=4`，偏移都是 4 float = 16 字节整数倍 -> `_mm_load_ps` 安全（它只要 16）；
  **NCHWC1**：走 `malloc` + **标量** `my_mm_*`，完全不碰对齐载入。
- wrapper 里的 padding 回退 `- padW*4` / `- padW*8` 也是 4/8 float 的整数倍，不破坏对齐。
- `layers_nchwc/*_raw.h` 里**零个**硬编码的 `_mm256_` / `_mm_load` / `__m256` / `__m128`
  （grep 确认），全部走 `zq_mm_*` 宏按 align 实例化 ——
  不存在「nchwc4 代码里混进 256 位对齐载入」。
- `ChangeSize` 把 C 补齐到 `ceil(C/align)*align`，所以 `for (c=0; c<C; c+=align)` 的
  **最后一次满宽读写落在 padding lane 上、仍在缓冲内** —— 这正是 NCHWC 相对 NCHW 的结构性优势。
- 唯一的「投机读过界」（depthwise 3x3 在最后一个输出像素上）由 `ZQ_CNN_NCHWC_ALLOC_SLACK 64` 兜住，
  且三个 `ChangeSize` 都加了。

### IR.1 —— 6 个 pooling 函数漏了无条件 `return`

`MaxPooling` / `AVGPooling` 的 NCHWC1 / NCHWC4 / NCHWC8 共 6 处：

```cpp
    if (need_W <= 0 || need_H <= 0)
    {
        if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
    return;              // <- 只在 ChangeSize 失败时才 return
    }
    bool suredivided = (in_H - kernel_H) % stride_H == 0 && ...
```

**NCHW 版有**那个无条件 `return ;`（`ZQ_CNN_Forward_SSEUtils.h:1601`），NCHWC 版漏了 —— 一份对六份错。
`ChangeSize(0,0,0,0,0,0)` 是**成功**的（把 output 置空并返回 true），
所以会继续往下走到 `% stride_H`：`stride_H == 0` 时那是**整数 idiv 除零 -> SIGFPE**。

触发：`stride_H <= 0 || stride_W <= 0`。经模型文件**当前不可达**
（`ZQ_CNN_Layer_NCHWC.h:2184-2192` 的 `ReadParam` 已有守卫），
但这 6 个是 public static、外部可直接调 —— 而 NCHW 专门补的那行就是这条路径的防线。
顺带：即便 `stride > 0`，`need_H == 0`（`in_H=3, kernel_H=5, stride_H=2`）时
NCHWC 会把 output 置成 0 大小后**继续调内核**（空转、不写内存），NCHW 直接返回 —— 也是行为分叉。

### IR.2 —— packed 重载缺 `filter_N != bias_C`

同文件 **unpacked** 重载有（`InnerProductWithBias:2412` 等 4 处：`filter_C != in_C || filter_N != bias_C`），
**packed** 重载 8 个（`InnerProductWithBias` / `...PReLU` / `ConvolutionWithBias` / `...PReLU` × NCHWC4/8）
两个校验都没有。

内核按 `zq_mm_load_ps(bias + out_c)` **满宽**读 bias，而 bias 只有
`ceil(bias_C/align)*align` 个 float —— `bias_C < filter_N` 且 `bias_C % align == 0` 时，
最后一组会**读过 bias 缓冲末尾**。

**只给 4 个补**（带 bias 形参的那些）：`InnerProductWithPReLU` / `ConvolutionWithBiasPReLU` 的
packed 重载**根本没有 bias 形参**（只有 slope），补了就是编译不过 ——
第一版按函数名批量加，MSVC 报 4 个 `error C2065: 'bias': 未声明的标识符`，才发现这一点。

x86 上 Convolution 的 packed 重载被 `#if __ARM_NEON` 包着不可达，
但 **InnerProduct 的 packed 重载在 x86 上真的会被调用**（`ZQ_CNN_Layer_NCHWC.h:2299/2326` 没有 `#if`），
所以这是活的洞。

### A/B 对拍：用「只回退 IR」的树重测

第一次对比时 `mtcnn_out_v8` 与新输出有差异，但那是**基线太旧**（v8 是 IJ **之前**抓的），
不是 IR 造成的。改用「只 `git checkout` 掉 NCHWC 那一个文件、重建、再对比」：

    SampleMTCNN_NCHWC4  IDENTICAL
    SampleMTCNN         IDENTICAL
    SampleMTCNN_Interface  IDENTICAL

符合预期：IR.1 / IR.2 只改**失败路径**，而 sample 走的全是成功路径。

对比时踩的一个坑：`NORM` 里原本只有 `GF/s=` 与 `<GF>GF/s`，
而这批输出用的是 `GFLOPS=7.901` 这种形式，于是**计时噪声**整片刷出来，
看起来像「163 行全变了」。补上 `GFLOPS=` / `MUL = ... M` 之后才是真对比。
—— 这和 `tools/capture_sample_outputs.sh` 的注释里写的「不能直接 diff，计时行每次都不同」是同一件事，
只不过我第一次读那份脚本时没把它套到**自己新写的**对比上。

### 实测

    cmake --build build_x64 --config Release（全量）-> RC=0，0 error
    WSL make -j8（全量）-> RC=0，0 error
    A/B（只回退 IR 的树 vs 修复后的树）：三个 MTCNN sample 全部 IDENTICAL

### 子代理还报了几条，本轮**没有**改

| 编号 | 内容 | 为什么不动 |
| --- | --- | --- |
| F2 | `Pooling` 的 `pad` 在 NCHWC 整条链路上被丢弃（`ReadParam` 解析了但 `Forward` 的签名里没有 pad），NCHW 完整支持 `pad` + `VALID/SAME` + 非对称 padding | 是**功能分叉**不是内存问题；要改得动 `ReadParam` / `Forward` / `GetTopDim` / 内核四层。当前 NCHWC 的 pooling 模型都不带 pad（带 pad 的写成 `pad_type=` + `kernel_H`，而 NCHWC 的 `ReadParam` 认不出来 -> **加载失败**，响亮失败而不是静默算错） |
| F4 | `Permute_NCHW` 在零维输入上整数除零 | **全仓零调用点**（NCHWC 没有 Transpose 层），且 NCHW 版本逐字相同 —— 属两族共有 |
| F5 | `ConvertFromBGR` / `ConvertFromGray` 的 `align_size>=4` 分支不清零 padding lane | 只污染 padding lane，真实通道不受影响；softmax 走 `c < in_C - align` + 标量收尾，不会把 padding 算进去。属「脏但无害」 |
| F6 | `ConvolutionPrePack` 对不认识的形状「返回 true 却不产出」 | x86 上整段被 `#if __ARM_NEON` 包住；ARM 上不打包的形状恰好在 packed 版里 `return false`，被 layer 的 fallback 兜住。属**契约弱化**不是缺陷 |

这四条都写进报告了 —— 「查出来但没改」和「没查」必须能区分开（AGENTS.md 第 20 条）。


## 追加：附录 IT.1 —— NCHWC 的 `ConvertFromBGR` / `ConvertFromGray` 不清 padding lane

### 缺陷

`ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h` 的 `align_size >= 4` 分支（`ConvertFromBGR` :176、
`ConvertFromGray` :235）**没有**清零，而 `align_size == 1` 分支有
（`memset(rawData, 0, rawDataLen)`，:157 / :219），`Reset()` 也有。

`ChangeSize`（`ZQ_CNN_Tensor4D_NCHWC.cpp:596-597`）在 N/H/W/C/border **全等**时
直接 `return true`、**不清零**。于是这条路径会留住旧数据：

    同一个 blob 先是 C=4（align=4，slice=ceil(4/4)=1，imageStep=sliceStep），
    再被 ConvertFromBGR 改成 C=3（slice=ceil(3/4)=1，**imageStep 相同**）
    -> rawDataLen == needed_dst_raw_len，既不重新分配也不清零，
    **lane 3 留着上一个尺寸的旧数据**。

Gray 更明显：C=1 时 align=4，**lane 1..3 全是 padding**，不留着就全是旧数据。

### 影响面：只污染 padding lane，真实通道不受影响

`zq_cnn_softmax_nchwc_raw.h` 用 `c < in_C - align` + 标量收尾，
不会把 padding 算进 softmax。属「**脏但无害**」。

之所以还是修：它会让 UBSan / 数值对拍出现**难以解释的差异** ——
「同样的模型同样的输入，两次跑出来 lane 上的数不一样」这种问题最难查，
而成本只是一次 memset。

### A/B 对拍

    SampleMTCNN_NCHWC4    IDENTICAL
    SampleMTCNN           IDENTICAL
    SampleMTCNN_Interface IDENTICAL

符合预期：只清 padding lane，真实通道一个字节都没变。

### 补这条时踩的坑

第一版按 `if (!ChangeSize(1, _height, _width, 3, 1, 1))` 匹配，**匹配到 2 处** ——
`align_size == 1` 那个分支里也有一模一样的一行（`ConvertFromBGR:155` / `ConvertFromGray:217`），
而那两处**已经有 memset**。要是没核对就批量插，同一个函数里会出现两次 memset，
而且我写的注释会指向错误的行号。

改成「匹配到之后看**下一行是不是已经有 memset**，有就跳过」——
这跟 IR.2 那次「没 bias 形参的两个 packed 重载不该加守卫」是同一类：
**「该有的」与「不该有的」必须分开判**。

### 实测

    cmake --build build_x64 --config Release（全量）-> RC=0，0 error
    WSL make -j8（全量）-> RC=0，0 error
    A/B：三个 MTCNN sample 全部 IDENTICAL

---

## 变更：附录 IU —— NCHWC 内核用 sliceStep 冒充 imStep（两处静默算错）

### 背景：为什么这类错误能活到现在

NCHWC 的张量布局是 `[n][c][h][w]`：

| step | 走多远 |
| --- | --- |
| `widthStep` | 一个像素（w 方向，含对齐） |
| `sliceStep` | 一个**通道片**（c 方向，步长 = align） |
| `imageStep` | 走完**一张图的全部通道**（n 方向） |

遍历 batch 的那个循环推进指针时**必须**用 `imStep`。写成 `sliceStep` 会有两层遮蔽：

1. **N = 1 时**，image 循环只跑一圈，推进量乘 0，两种写法等价 —— 看不出差别。
2. **C 是 align 的整数倍时**，两个 step **数值相同**。
   `ZQ_CNN_Tensor4D_NCHWC<n>::ChangeSize` 里
   `dst_slice = ceil(dst_C/align_size)`、`dst_imStep = dst_slice * dst_sliceStep`，
   于是 `dst_slice == 1`、`imStep == sliceStep`。

项目里几乎所有模型的中间层 C 都是 align 的整数倍，于是**两个遮蔽同时成立**。
而且因为 `sliceStep <= imageStep`，写错的地址仍在缓冲区里：
**不越界、不崩、ASan/UBSan/Valgrind 一个都不报**，纯静默算错。

### IU.1 `ZQCNN/layers_nchwc/zq_cnn_pooling_nchwc_raw.h`

`zq_cnn_avgpooling_nopadding_suredivided_kernel2x2` 的 n 循环：

    -   n++, in_im_ptr += in_sliceStep, out_im_ptr += out_sliceStep)
    +   n++, in_im_ptr += in_imStep,   out_im_ptr += out_imStep)

同一个文件里 max 版本、k3x3 版本、general 版本的同名循环用的都是 `imStep`，
**只有 avg/k2x2/suredivided 这一份漏了**。

分派路径：`ZQ_CNN_Forward_SSEUtils_NCHWC::AVGPooling`
→ 条件 `suredivided = (in_H-k_H)%s_H==0 && (in_W-k_W)%s_W==0` 且 `k_H==k_W==2`
→ `zq_cnn_avgpooling_nopadding_suredivided_nchwc{1,4,8}_kernel2x2`。
即：**2x2 平均池化、整除、能放下**，NCHWC1/4/8 三个通道宽度**全中**。

### IU.2 `ZQCNN/layers_nchwc/zq_cnn_resize_nchwc_raw.h`

`zq_cnn_resize_with_safeborder` 同一个错误。同文件里的
`zq_cnn_resize_without_safeborder` 用的是 `imStep`，只有 `with` 这一份漏了。

调用链：`ZQ_CNN_Tensor4D_NCHWC<n>::Resize(...)`
→ `zq_cnn_resize_with_safeborder_nchwc{n}`，N 直接来自张量，
所以**任何 N>1 的 Resize 都受影响**。

### 门禁为什么没抓到（两个门禁各有一道独立盲区）

`tools/zq_nchwc_pool_check.cpp`：k2x2 那批用例写死 `c.C = A`（A 就是 align），
`dst_slice` 恒为 1 → 两个 step 相同 → 遮蔽 2 生效。general 那批虽然用了
`C = A + 2`，但 `N` 一直写死 1 → 遮蔽 1 生效。

`tools/zq_nchwc_resize_check.cpp`：`c.N = 1; c.C = A;` **两个都写死**，
而且 `with_safeborder` 在 cfgB 下被 `continue` 跳过，
于是它**只跑过 N=1、C=A 一种形状**。

修法是给两个门禁都补上「N >= 2 **且** C 不是 align 整数倍」的用例。

### 变异测试（先红后绿，两轮）

    zq_nchwc_pool   修前：150 个用例 / 6 个 FAIL（avg suredivided/k2x2 × nchwc1/4/8）
                         最差后向误差 4.98e-01
                    修后：150 / 150 全对
    zq_nchwc_resize 修前：18 个用例 / 3 个 FAIL（with_safeborder × nchwc1/4/8）
                         最差后向误差 9.95e-01
                    修后：18 / 18 全对

没有出现「修了反而多了 FAIL」，也没有别的入口被牵连。

### 新门禁 `tools/check_imstep_guard.py`（A40 自测 + A41 普查）

扫 `ZQCNN/layers_nchwc`、`layers_c`、`math`、`ZQ_GEMM/math` 共 119 个源文件，
凡是 `<prefix>_im_ptr += <expr>` 且 `<expr>` 里出现任何 `*_sliceStep` 就报错。

**不写死变量名**（只用 `_im_ptr` / `_sliceStep` 这两段命名约定），
也不写死前缀 —— `in_` / `out_` / `cur_` 都一样能命中，
`in_im_ptr += out_sliceStep` 这种交叉配对也拦得住。
反向不误报：推进**通道片**的 `in_slice_ptr += in_sliceStep` 是正确的，不报。

**这条门禁第一版是恒真的。** 结束符我写成 `[^;]+;`，
而这些语句绝大多数出现在 `for` 的第三个子句里、以 `)` 收尾 —— 一条都没匹配上，
门禁永远绿。是靠变异测试（把两个真实站点改回 sliceStep）当场打出来的，
已经补进 `--selfcheck`（3 正例 + 2 反例 + 1 不该报）。

### 变更文件

    ZQCNN/layers_nchwc/zq_cnn_pooling_nchwc_raw.h     IU.1
    ZQCNN/layers_nchwc/zq_cnn_resize_nchwc_raw.h      IU.2
    tools/zq_nchwc_pool_check.cpp                     补 N>=2 / C 非 align 整数倍用例
    tools/zq_nchwc_resize_check.cpp                   同上（cfgA 拆成 N=1/N=2 两组）
    tools/check_imstep_guard.py                       新门禁
    tools/run_audit_checks.py                         注册 A40 / A41

### 注意事项

- 这两处都是**静默算错**：不越界、不崩、没有任何 sanitizer 会报。
  能抓住它们的只有「形状覆盖」和「源码门禁」两件事。
- 新增门禁已进 `tools/run_audit_checks.py` 的 A 组，随每轮回归一起跑。

---

## 变更：附录 IV —— NCHWC「batch 维不变性」门禁（10 个 op 全绿的阴性结论 + 门禁自身两处洞）

### 判据不是"和手写参考比"，而是一条不变式

    out[n](N=2) 必须**逐位**等于「只把第 n 张图（N=1）单独送进同一个算子」的结果

**不需要任何参考实现**：只要 `N=2` 且 `C` 不是 align 的整数倍，
任何"batch 维步进写错"的错误都会让两侧不等。
它测的是「图与图之间有没有互相串」，而这正是 IU.1 / IU.2 那类缺陷的全部内容。

覆盖 10 个 NCHWC 算子 × {NCHWC1, NCHWC4, NCHWC8} × {C = A, C = A+2} × 2 轮
= 120 个用例。两个 C 都跑，因为"只在有鉴别力的那档红"是这类缺陷的正常形态。

### 实测：120 个用例，114 全对 / 0 有错 / 0 崩 / 6 契约不符跳过

| 算子 | 结果 |
|---|---|
| ReLU / PReLU / AddBiasPReLU / BatchNorm_b_a | 12/12 全对 |
| Softmax | 6/6 全对，**6 组 op 返回 false**（只支持部分 axis）→ 记「契约不符跳过」 |
| MaxPooling / AVGPooling | 12/12 全对（IU.1 修完之后） |
| Eltwise_Sum | 12/12 全对 |
| DepthwiseConvolution | 12/12 全对 |
| Convolution | 12/12 全对 |

**这是一条阴性结论，但它把"IU.1/IU.2 是仅有的两处"从猜测变成了实测。**
阴性结论也要落盘（AGENTS.md 第 23 条），否则下一个人会把同样的横扫重做一遍。

### 探针自己栽了两次，两次都是"形状摆错"，不是被测代码的缺陷

1. **filters 的形状摆错**。`DepthwiseConvolution` 的契约是
   `filter_C == in_C && filter_N == 1`（`ZQ_CNN_Forward_SSEUtils_NCHWC.cpp:1130`），
   我第一版摆成 `[N=C][H=kH][W=kW][C=1]`，于是 op 直接 `return false`，
   而探针把「没跑完」报成**崩溃** —— 10/12 个"崩"全是探针的错。
2. **输出张量按 `in_C` 开**。`Convolution` 的输出通道是 `filter_N` 不是 `in_C`，
   `ConvertToCompactNCHW` 写出去的是 `filter_N` 个通道，
   比对时越过实际写入的末尾、拿未初始化内存当结果，
   于是报出「一张图全错、最大差 1.2e+01」—— **那是我探针的缓冲区开小了**。

  > 两次都符合 AGENTS.md「没有证据就不要断言原因」：
  > **"崩"和"错"必须分开报**，而"形状摆不下"是第三种成因，
  > 混进崩溃桶里会把探针的 bug 说成库的缺陷。
  > 探针现在有 `ST_REJECT` 状态码，父进程单列「契约不符跳过」这一栏。

  顺带记一条**查过但不是缺陷**的：
  GEMM 版卷积把 `matrix_A_cols = kH*kW*align_C` 当成 GEMM 的 `lda`，
  我一度以为 `C % align != 0` 时 `align_C > C` 会让它读越界。
  实际每个 filter 的真实步长是 `imageStep = ceil(C/align)*kH*kW*align`，
  **与 `matrix_A_cols` 恒等**（`ceil(C/align)*align == align_C`），所以是对的。

### 门禁 `check_imstep_guard.py` 自己也有两处洞，都是这一批抓出来的

1. **裸 `im_ptr` 全漏**。规则写成 `[A-Za-z_][A-Za-z0-9_]*_im_ptr`，
   而仓库里图像级指针有 **52 处**就叫裸的 `im_ptr`（eltwise / convolution_gemm 两族）。
   改法：`\b((?:[A-Za-z_][A-Za-z0-9_]*_)?im_ptr)`，覆盖站点从 108 增到 132。
   > 是靠 `grep -oE '\bim_ptr\b'` 数出 52，跟 `[A-Za-z0-9_]+_im_ptr` 的计数**对不上**
   > 才发现的 —— **两个计数必须相加等于总量**（AGENTS.md「计数 + 标签」那条）。
2. **`--root` 少拼一层目录，于是"扫了 0 个文件"却打出一行 OK**。
   变异测试脚本先暴露了它。修法是 `--root` 明确指**仓库根**，
   并给门禁加了**零文件即失败**的守卫：
   真跑一次是 119 个，扫 0 个说明路径错了，而不是"代码干净了"
   （AGENTS.md「荒谬的数字本身就是信号」）。

### 变异测试改成**在副本上做**（不动工作区）

`tools/_mut_imstep2.py`：把真实内核头复制到临时目录当 `ZQCNN/` 的替身，
在副本上把 imStep 改回 sliceStep，验证门禁变红**并点名到行**，再恢复验证回绿。
还原放在 `finally` 里（第一版脚本异常点在 finally 之前，
**被变异的源文件留在磁盘上没还原**，紧接着那一次跑就在读半改的树）。

    基线（副本）        RC=0   OK: 119 个内核源文件
    变异 pooling        RC=1   点名 8 处（:33 :115 :183 :264 :345 :407 :473 :576）
    变异 resize         RC=1   点名 2 处（:65 :168）
    各自还原后          RC=0

### 变更文件

    tools/check_imstep_guard.py   规则放宽到裸 im_ptr、加 --root、加零文件守卫、
                                  自测补到 5 正例 + 3 反例 + 1 不该报
    tools/_batch_probe.cpp        batch 维不变性探针（10 个 op）— **尚未登记进回归**，
                                  文件名刻意不叫 zq_*_check.cpp（见下）
    tools/_mut_imstep2.py         副本式变异测试

### 注意事项

- **踩到 AGENTS.md 第 31 条（新写的那条）**：v68 正在跑 `run_zqlib_checks.py`，
  而那个脚本第 921 行是 `glob('tools/zq_*_check.cpp')` ——
  我新建的 `zq_nchwc_batch_check.cpp` **当场被当成一道新的检查项**，
  又因为还没登记 EXTRA_SOURCES 而链接失败。
  当时立刻把文件改名成 `_batch_probe.cpp`（glob 匹配不到）才没污染那一轮。
  **探针跑通、验证过之后，再改名 + 登记进回归。**

---

## 变更：附录 IV.1 —— NCHW `pad_type=SAME` 用 floor 而不是 ceil（220 个 live 层）

### 缺陷

`ZQCNN/ZQ_CNN_Layer.h` 里 `pad_type == TYPE_SAME` 的分支算 padding 时：

```cpp
int top_W = bottom_W / stride_W;     // 整数除法 = **向下取整**
int top_H = bottom_H / stride_H;
int pad_W = __max((top_W - 1)*stride_W + real_kernel_W - bottom_W, 0);
```

而 SAME 的定义（TF / Caffe / ONNX 三家一致）是

```
out     = ceil(in / stride)
pad_tot = max((out-1)*stride + real_kernel - in, 0)
```

floor 与 ceil **只在 `in % stride == 0` 时相等**，于是非整除时整层少一格：
`in=5 stride=2 k=2` 时 `top` 给 2（应 3）、`pad` 给 0（应 1）。

### 三处同形写法里只改了两处 —— 第三处**必须保持 floor**

| 站点 | 层 | 处理 |
|---|---|---|
| `:732` | `ZQ_CNN_Layer_Convolution` | **改成 ceil** |
| `:1347` | `ZQ_CNN_Layer_DepthwiseConvolution` | **改成 ceil** |
| `:4052` | `ZQ_CNN_Layer_Pooling` | **保持 floor，附理由** |

Pooling 那处不能照抄：它的输出尺寸约定是
`GetTopDim: out = ceil((in - kernel)/stride) + 1`
（`ZQ_CNN_Forward_SSEUtils::MaxPooling` 的 `need_H/need_W` 与内核的
`final_kH/final_kW` 三处一致），而不是 SAME 的 `ceil(in/stride)`。
把 `in = q*S + r` 逐段推：

| 条件 | floor 给 | ceil 给 | 有差别吗 |
|---|---|---|---|
| `r == 0` | q | q | 无 |
| `r>0 且 kernel <= r` | pad 都是 0 | pad 都是 0 | 无 |
| `r>0 且 r < kernel <= S+r` | q+1 | q+1 | 无 |
| `r>0 且 kernel > S+r` | **q** | **q+1** | **有** |

而 `q` 正是本层 VALID 的输出值（`ceil((in-k)/s)+1`）。
也就是说 **Pooling 的 floor 与它自己的约定一致，ceil 反而会破坏它**
（实例：`in=5 kernel=4 stride=2` -> floor 给 2（= VALID），ceil 给 3）。

> 这是本会话第 N 次「同一个函数里的同一段写法不保证该抄」。
> `check_imstep_guard` 那次是"两个族的同名变量含义相反"，
> 这次是"三个类的同名分支语义不同"。**照抄之前先问那个类的约定是什么。**

### 影响面：220 个 live 层，但**原生尺寸下 0 个受影响**

随仓三个模型带 `pad_type=SAME` 的层：

    Pose-zq.zqparams                 147 个   非整除 0
    det5-112-gray.zqparams            36 个   非整除 0
    headposegaze-112-gray.zqparams    37 个   非整除 0
    合计                              220 个   非整除 0

用 `tools/_padtype_impact.py` 静态传播各模型的原生输入尺寸
（Pose 192x192、另两个 112x112）得出：**没有任何 SAME 层落在非整除那一档**，
即 floor 与 ceil 取值完全相同 ⇒ **这次改动对随仓模型零影响**，
它们的输出逐位不变。换个非整除输入（PersonPose 的奇数边长）才会显形。

### 新门禁 `tools/zq_padtype_check.cpp`（576 个用例）

直接实例化层类、走 `ReadParam -> SetBottomDim -> GetTopDim` 这条**生产路径**，
把 `top_H/top_W` 与 padding 之和对上。判据分两项，两项都要对。

四类层的语义**各不相同**，期望值逐类写：

    Convolution / Depthwise : SAME out = ceil(in/stride)
    Pooling                 : 本仓库约定 out = ceil((in-k)/stride)+1（pad 恒 0）
    DeConvolution           : SAME out = in*stride（TF Conv2DTranspose）
                              VALID **未实现**，那一档不跑

    修前：576 个用例 / 有错 90（Convolution 45 + DepthwiseConvolution 45，
          全是 SAME 且 in % stride != 0；DeConvolution 144/144 对、Pooling 144/144 对）
    修后：576 个用例 / 有错 0（504 个跑、72 个是 DeConvolution VALID 不跑）

探针自己栽了三次，三次都是**期望值算错**、不是库的缺陷：

1. 对四类层用同一个 `real_k = (k-1)*d+1`，而 **Pooling 没有 dilation**
   —— 95 个"错"全是我把 dilation 用在了没有 dilation 的层上；
2. DeConvolution 的 SAME 我按 `ceil(in/stride)` 写，而 TF `Conv2DTranspose`
   的 SAME 是 `out = in*stride` —— 144 个"错"全是我抄错了定义；
3. 同一次跑法不对：`run_zqlib_checks.py` 是**从 Windows Python 驱动 WSL** 的，
   我在 WSL 里 `python3 tools/run_zqlib_checks.py`，于是 `wsl: not found`、
   子进程输出为空、门禁报 **`0/0 通过`** —— 一个字都没说。
   > **"0/0 通过"是荒谬的数字**：跑不出一个用例就不能叫通过。
   > 这与本文件「荒谬的数字本身就是信号」是同一条。

### 变异测试的教训（第 30 条的第一次真实触发）

`tools/_mut_padtype.py`（已删）要跑三轮门禁、每轮约 275s，
而后台任务有 600s 硬上限 —— **第三轮之前进程被杀**。
第一反应是查源文件有没有还原：`grep -c CEIL-REVERTED-BY-MUTATION` = 0、
`grep -c '... ceil'` = 2，**树是干净的**（`finally` 跑到了）。
`git diff` 也确认只有 4 行代码变化（两处 `top_W`/`top_H`），其余都是注释。
> 与本会话更早那次同一类：**变异脚本必须在 `finally` 里还原**，
> 而且**还原之后要 `grep -c` 确认**，不能凭"没报错"就认为还原了。
>
> 这次的红/绿证据不依赖那个脚本：**修前 90/576 错、修后 0/576 错**，
> 用的是同一个判据、同一份代码。变异脚本只是重复了一遍这件事，
> 而它 14 分钟的代价换不来比上面那两行更多的东西。

### 登记进回归

`tools/run_zqlib_checks.py` 加了 `_glob_extra()`：
按 glob 生成「编一个 TU」的命令，避免手写清单漏文件。
**"补一个报一屏 undefined reference、补两个还报一屏"就是"依赖是整族的"的信号**
（`zq_padtype` 为此连补三版：补 Forward、补 layers_c、补 `zq_avx_mathfun.c`，
第四版才发现 **128 位那份在另一个文件** `zq_sse_mathfun.c` 里）。

`_glob_extra` 第一版把 `.c` 从**源路径**里也去掉了（只该从对象名里去），
拼出 `gcc ... $R/ZQCNN/ZQCNN_layers_c_zq_cnn_addbias_32f_align_c` ——
22 条命令全错。现在带两条守卫：源文件必须真的存在；glob 一个都没匹配到就 assert。

### 变更文件

    ZQCNN/ZQ_CNN_Layer.h               IV.1（Convolution + Depthwise 改 ceil；Pooling 附不改的理由）
    tools/zq_padtype_check.cpp         新门禁
    tools/run_zqlib_checks.py          登记 zq_padtype / zq_nchwc_batch，加 _glob_extra

### 实测

    python tools/run_zqlib_checks.py zq_padtype        -> 1/1 PASS（ASan 下）
    python tools/run_zqlib_checks.py zq_nchwc_batch    -> 1/1 PASS（ASan 下）
    cmake --build build_x64 --config Release（全量）    -> RC=0
    WSL make -j8（全量）                                -> RC=0
    check_text_encoding.py -> OK 779 files / check_line_endings.py -> OK / check_stmt_joins -> OK

---

## 变更：附录 IV.5 —— v69 的 C5 变红**不是缺陷**，是基线的键含行号

### 现象

    1 CHECK GROUP(S) FAILED:
       C5 主工程 -O2 -c 优化期告警 HIGH 桶门禁

    HIGH: 基线 4 条 -> 现在 4 条
    NEW HIGH  ZQ_CNN_Forward_SSEUtils_NCHWC.cpp Wduplicated-branches :2125 :2244 :2345 :2447
    FIXED     ZQ_CNN_Forward_SSEUtils_NCHWC.cpp Wduplicated-branches :2056 :2160 :2260 :2361

**4 进 4 出、总数不变** —— 这是"整体平移"的签名。

### 核实：前后逐字相同

    基线行 2056 (旧版) vs 现在 2125
      OLD: 	else if (filter_H == 3 && filter_W == 3 && in_C <= 4)
      NEW: 	else if (filter_H == 3 && filter_W == 3 && in_C <= 4)
    （2160/2244、2260/2345、2361/2447 三对同样逐字相同）

行号整体下移 69~86 行，来自 commit `9529090`
（IR.1/IR.2 的 `return;`）在那 4 条**上方**插了代码。
而这 4 条本身是**已判定的 SIMD 分派误报**：
两个分支在 x86 上都被 `#if __ARM_NEON` 掏空
（同 AGENTS.md「常量比较在本项目基本都是刻意的 SIMD 宽度分派」那条）。

**结论：不是本次改动引入的缺陷，是门禁基线的键设计问题。**

### 修法：键从 `(文件, 标志, 文件:行:列)` 改成 `(文件, 标志, 那一行的源码文本, 第几次出现)`

行号只留给人看，**不参与比对**。

这正是 AGENTS.md「基线的键里不能放任何会随无关编辑漂移的东西」
那条**第二次**栽在同一个地方（第一次是 `(文件,行号,…)` 被无关编辑平移）。

#### 第一版键的修法本身又栽了一次

`_srctext()` 先按 `ROOT / basename` 找源文件 —— 而告警里的路径是 basename，
源文件在 `ZQCNN/` 或 `ZQ_GEMM/` 下面，**不在仓库根**。
于是 4 条警告的"源码文本"全成了空串，基线第 3 列整列为空：

    ZQ_CNN_Forward_SSEUtils_NCHWC.cpp	Wduplicated-branches		0	...:2125:7

**空键是危险的**：它对任何文件都匹配，等于把 4 条塌成"同一条出现 4 次"。
现在按 `'' / ZQCNN / ZQ_GEMM / ZQCNN/math / ZQ_GEMM/math /
ZQCNN/layers_c / ZQCNN/layers_nchwc` 依次找真实文件，
并且单测过取值：

    _srctext('ZQ_CNN_Forward_SSEUtils_NCHWC.cpp', ...:2125:7)
      -> 'else if (filter_H == 3 && filter_W == 3 && in_C <= 4)'
    _srctext('ZQ_CNN_BBox.h', ...:146:42)
      -> 'memset(this, 0, sizeof(ZQ_CNN_BBox240));'

### 实测

    --save-baseline 后基线第 3 列是真源码文本（4 条 occ = 0/1/2/3）
    --check-baseline -> HIGH: 基线 4 条 -> 现在 4 条
                        无新增、无消失。        RC=0
    变异：从基线删掉 occ=3 那一条 -> 报 1 条 NEW、RC=1（见下）

### 变更文件

    tools/warn_sweep_bounds.py          键改为源码文本；_srctext() 按多级目录找源文件；
                                        基线格式加注释说明「第 3 列刻意不是行号」
    tools/zqcnn_bounds_baseline.txt     按新格式重新生成（4 条，内容不变）

### 注意事项

- 旧格式（3 列、行号键）的基线仍然**读得进来**，但升级后第一次跑会把
  全部旧条目报成 FIXED、全部新条目报成 NEW —— **看到这种"整齐对称"的一堆，
  就是该跑 `--save-baseline` 了**。这一点写进了代码注释。

---

## 变更：附录 F2 —— NCHWC 池化**静默丢弃 pad**（形状就错一格）

### 缺陷

`ZQ_CNN_Layer_NCHWC_Pooling::ReadParam` **解析了** `pad`（写进 `pad_H`/`pad_W`），
而 `Forward` 调的

```cpp
MaxPooling(bottom, top, kernel_H, kernel_W, stride_H, stride_W, global_pool)
```

**签名里根本没有 pad** —— 那两个成员解析完就没人用了。
模型文件写 `pad 1` 会**加载成功、然后静默按无 pad 计算**。

第一版探针（拿 NCHW 当参考）的实测：

    nchwc1 MAX k=3x3 s=2x2 pad=1,1,1,1 in=7x9   形状不同 want(4,5) got(3,4)
    nchwc4 AVG ...                                    形状不同 want(4,5) got(3,4)
    nchwc8 MAX ...                                    形状不同 want(4,5) got(3,4)
    共 36 个用例：全对 12（全是 pad=0 的对照），形状不同 24

**不是数值差一点，是形状就错一格** —— 下游整条链全错。

顺带查出的同族缺口（`NCHW 有 / NCHWC 没有` 的参数键）：

| 层 | 缺 |
|---|---|
| Convolution | `pad_H_top` `pad_H_bottom` `pad_W_left` `pad_W_right` `pad_type` `same` `valid` |
| DepthwiseConvolution | 同上 |
| Pooling | 同上 + `kernel_H` `kernel_W` `stride_H` `stride_W` |

严重度分三档（实测）：

1. **NCHWC 池化的 `pad N`** —— 一声不吭、形状就错。**最糟。**
2. **NCHWC 卷积/深度卷积的 `pad_type` / 非对称 pad** —— 打一行
   `warning: unknown para`，然后按 pad=0 算。`ReadParam` 的返回条件
   （`has_num_output && has_kernelH && has_kernelW && has_bottom && has_top && has_name`）
   **不含 pad**，所以这一档是**静默算错**。
   缓解：随仓那 220 个 SAME 层全是可整除的（SAME ≡ 无 pad，见附录 IV.1 的影响面），
   所以转成 NCHWC 结果仍然一致 —— 但那是**巧合**，不是设计。
3. **NCHWC 池化的 `kernel_H/kernel_W/stride_H/stride_W`** —— 落到
   "unknown para"，`has_kernelH` 保持 false，`ReadParam` 末尾 `return false`
   ⇒ **加载失败**。这一档反而是响亮的（Pose-zq 的两行 Pooling 正是这种写法）。

### 修法

* **前向**（`ZQ_CNN_Forward_SSEUtils_NCHWC.{h,cpp}`）：六份 MaxPooling/AVGPooling
  各加 `pad_H_top/pad_H_bottom/pad_W_left/pad_W_right`（**实参带默认值 0**，
  既有调用点不用改），`input` 改成非 const 引用。
  非零 pad 时就地 `Padding`，再把窗口起点挪到 `-pad_top`。
  六份共用一个 `zq_nchwc_pool_prepare_pad()` 模板 ——
  **同一段索引算术抄六遍，抄错一份就又是一次静默算错**。
  NCHWC 的 `Padding` 只支持对称，而 SAME/VALID 经常给出非对称的 pad，
  于是补 `P = max(两侧)`；多补出来的那几行永远读不到（不等式写在代码注释里）。
* **层**（`ZQ_CNN_Layer_NCHWC.h`）：加 `pad_type` 与四个方向 pad 的解析、
  `kernel_H/kernel_W/stride_H/stride_W` 四种写法；
  `SetBottomDim` 里的 pad_type 解析**逐字照抄 NCHW 的 `ZQ_CNN_Layer_Pooling`**
  —— 包括 SAME 用 floor 那一点（附录 IV.4 已分析：那个 floor 与池化层自己的
  尺寸约定自洽，改成 ceil 反而会破坏它）。
  目的是**让 NCHWC 与 NCHW 给出同一个数**，不是"更正确"。
* `suredivided` 的判据从 `(in_H - kernel_H) % stride_H == 0`
  改成 `(need_H - 1)*stride_H + kernel_H <= in_H` —— 前者是**无 pad 时**的等价写法，
  有 pad 时不成立；后者说的是"最后一个窗口不用裁边"这件事本身。

### 关键转折：**判据不能用 NCHW 当参考，因为 NCHW 那一支自己有缺陷**

修完之后用同一份数据（4x4、值=行*4+列+1、k=2 s=2 pad=1）跑三种实现：

    输入 c0
       1   2   3   4
       5   6   7   8
       9  10  11  12
      13  14  15  16
    三种实现给出的第 2 行：
      NCHW           7   8   9     <- 窗口起点落在**数据首行**
      NCHWC（修后）  9  11  12     <- 窗口起点落在 -pad 行（= 明文定义）
      手算参考       9  11  12

NCHW 的代码里确实写了 `GetFirstPixelPtr() - pad_H_top*in_widthStep - pad_W_left*in_pixStep`，
但**实测窗口起点并没有真的退到 -pad 行** —— 代码表达的意图与实际行为不一致。
拿它当参考等于把一个缺陷固化成"标准"。

所以最终门禁 `tools/zq_nchwc_poolpad_check.cpp` 用的是**独立参考**：
按明文定义「零填充 + 池化」直接算，不经过任何被测代码。
这一族缺陷靠"和孪生实现比"是抓不到的 ——
AGENTS.md 那条「同仓的两份实现互为对照」的**适用条件是两边都对**。

### 新门禁 `tools/zq_nchwc_poolpad_check.cpp`（60 个用例）

NCHWC1/4/8 × {MAX, AVG} × {C = A, C = A+2} × 5 组 pad
（0 / 对称 / 非对称两种方向 / 不对称且不等）= 60 个用例，**逐格后向误差**。

门禁的参考实现自己也栽了两次，都是**参考错、库对**：

1. AVG 的除数写成恒定的 `kH*kW` —— 而正确的是「窗口落在补齐区里的格数」
   （最后一个窗口在补齐区右边只够 2 格时除以 2）。6 个用例红。
2. 改成按补齐区计数之后，**有效范围写成了 [0, H)**（原图）而不是
   `[-pT, H+pB)`（补齐区）—— 于是对称 padding 的 `oh=0` 少算了一格，
   18 个用例红。

> 与 AGENTS.md「一个坏测试会产出看起来很有说服力的假结论」完全同形：
> 两次的症状都是"库算错了"，而库是对的。
> **这类门禁里，参考实现本身必须先被怀疑。**

### 变异测试 —— **第一次跑，门禁没红**，那才是这一段最值钱的记录

`tools/_mut_poolpad.py`：把 `Forward` 传下去的 pad 改回全 0
（= 修之前"解析了但不使用"的行为），门禁必须变红；还原放 `finally`。

    基线（未变异）  rc=0   1/1 通过
    变异后          rc=0   1/1 通过     <-- **门禁没红**
    还原            grep 计数 2 / 0

**为什么没红**：第一版门禁只调 `MaxPooling/AVGPooling`，
压根**没经过层** —— 而缺陷恰恰在层里（层解析了 pad 却没往下传）。
门禁覆盖的是内核与 padding 的正确性，**没覆盖"参数有没有被传下去"**。

> 与 AGENTS.md「阳性对照要换一个变异位置再问一次」同源：
> 同一个变异落在**判据覆盖维度之外**时，门禁看不见。
> 而"门禁全绿"这件事本身，在这次里**没有提供任何信息** ——
> 它压根不知道那一段代码的存在。

补上第二段覆盖（驱动层：`ReadParam -> SetBottomDim -> LayerSetup -> Forward`）之后，
用例从 60 增到 120，再跑变异：

    基线（未变异）  rc=0   1/1 通过（120/120 用例）
    变异后          rc=1   FAIL (rc=1, 48 条断言失败)，**全部落在「层」那一段**，
                          例：nchwc1 层 AVG C=3 k=3x3 s=2x2 pad=1,1,1,1
                              FAIL 120/120 格不同，最大差 1.235e+04
                          （1.235e4 是我给输出预填的哨兵 -12345：
                           丢掉 pad 之后层的 top 变小，写不到的那些格保持哨兵值。）
    还原            grep 计数 2 / 0，确认成功

> 通则：**门禁要覆盖"缺陷发生在哪一层"，而不只是"结果对不对"。**
> 这一族（解析了不用、传下去丢了、传错了顺序）在数值层面都可能看不出来，
> 只有**驱动那一层**才验得到。

### 变更文件

    ZQCNN/ZQ_CNN_Forward_SSEUtils_NCHWC.h    六份 MaxPooling/AVGPooling 加 pad 形参
    ZQCNN/ZQ_CNN_Forward_SSEUtils_NCHWC.cpp  zq_nchwc_pool_prepare_pad() + 六份实现
    ZQCNN/ZQ_CNN_Layer_NCHWC.h               层：pad_type / 非对称 pad / kernel_H 四写法 /
                                             SetBottomDim / GetTopDim / Forward 传 pad
    tools/zq_nchwc_poolpad_check.cpp         新门禁
    tools/run_zqlib_checks.py                登记 zq_nchwc_poolpad

### 实测

    python tools/run_zqlib_checks.py zq_nchwc_poolpad -> 1/1 PASS（60/60 用例）

### 注意事项

- **NCHW 那一支的带 padding 池化仍然是错的**（窗口起点没退到 -pad 行）。
  本轮**没有改**它：改了会动到既有模型的数值，而随仓模型里
  池化层的 pad 全是 0（220 个 SAME 层都解析成 0），所以改它对随仓无影响 ——
  但这是**另一个独立决策**，需要单独评估，不该顺手带上。
  已在门禁文件头把这件事写清楚，避免下一个人以为 NCHW 是对的。

---

## 阴性结论：全仓再扫一遍「解析了却从不使用」的成员（附录 DF 那一族）

DF 那个池化缺陷是「ReadParam 解析了 pad，`Forward` 却从不使用」。
写一个通用探测器把这一族在全仓扫一遍，
判据是**逐个类**取成员声明，排除三类出现处（声明本身、构造初始化列表、赋值语句），
剩下的算「读」；`赋值 > 0 且读 == 0` 才报。

### 探测器自己栽了三次，三次都是「报 0 命中 / 报的东西不对」

| # | 写法 | 症状 |
|---|---|---|
| 1 | `re.match(r'\n\tclass ...')` | `readlines()` 的每行以 `\t` **开头**，`\n` 在上一行末尾 —— `match()` 永远匹配不上，**一类都没找到** |
| 2 | 类正则要求 `{` 在同一行 | ZQCNN 的 NCHW 族是 `\tclass X : public Y<T>` **换行**才 `{`，NCHWC 族才是同行 |
| 3 | 「读」的判定没排除**声明行**与**构造初始化列表** | `int pad_H;` 必然出现、`pad_H(0)` 也必然出现 —— 于是 read >= 1，**报告永远是空的** |

**三次都是靠阳性对照抓出来的**：把修之前的 `ZQ_CNN_Layer_NCHWC.h`
（`git show f52f353~1:...`）喂进去，探测器必须报出 `pad_W`。
前两次它报「没有发现」，第三次才报出来。
> 这是本会话第 N 次「扫到 0 命中时先怀疑工具」。
> 区别在于这次我**先做了阳性对照再下结论**，
> 而不是拿「0 命中」当结论写进报告。

### 扫描结果（当前树，ZQ_CNN_Layer.h + ZQ_CNN_Layer_NCHWC.h）

    ZQ_CNN_Layer                 ignore_small_value / show_debug_info / last_cost_time
    ZQ_CNN_Layer_NCHWC           ignore_small_value / last_cost_time / use_buffer
    ZQ_CNN_Layer_NCHWC_InnerProduct  kernel_H / kernel_W

**8 条，逐条核实，没有一条是新缺陷**：

1. **前 6 条是 `static` 成员，必然的假阳性** ——
   `static` 成员的读取发生在**派生类**里，探测器只看了基类自己的类体：
   `use_buffer` 读在 `ZQ_CNN_Layer.h:319`（Convolution 内），
   `ignore_small_value` 读在 `:797` / `:814`，
   `show_debug_info` 读在 `ZQ_CNN_Layer_CascadeOnet.h:43` 等处。
2. **InnerProduct 的 `kernel_H` / `kernel_W` 是"解析了但被覆盖"，不是"从不使用"** ——
   `SetBottomDim` 里 `kernel_H = bottom_H; kernel_W = bottom_W;`，
   无论文件里写的是什么都会被实际张量尺寸覆盖。
   而权重的读取长度也是按 `filters->ChangeSize(num_output, kernel_H, kernel_W, bottom_C, 0, 0)`
   来的 —— 同一个权威来源，**内部自洽**。
   NCHW 与 NCHWC 两边**行为完全一致**（两边都覆盖），所以也不存在"孪生实现分叉"。

   > 这是与 DF **不同**的一个子形态：「解析了但被权威值覆盖」**无害**，
   > 「解析了但从不使用」**有害**。区别在于被覆盖的那个值**是谁说了算**。
   > 而"两个孪生实现分叉"这条判据在这里也用不上 ——
   > 两边**同样地**错（同样地覆盖），不是一份对一份错。

### 结论

**DF 那一族在 ZQCNN 的层里已经清干净**：池化是唯一一处真缺陷，已修。
探测器本身**没有进回归** ——
它的三条已知失效模式（正则形态、类边界、声明/初始化列表判定）都太脆，
留在仓库里当一次性排查手段更合适；真要常驻必须先补自测。

### 变更文件

    无（阴性结论，只落盘）

---

## v71 全量回归：**ALL CHECKS PASSED**（最终单点）

    python tools/run_audit_checks.py --with-build --warn-sweep --src-sweep \
        --bounds-sweep --ubsan-sweep --reachability --msvc-asan
    -> ALL CHECKS PASSED   RC=0

关键组：

    D1 Windows 全量构建 (VS2022/cmake)          OK
    D2 Linux 全量构建 (gcc/wsl)                 OK
    D3 Linux sample 回归   真跑 10 / 桩 2 / 问题 0
    D4 Windows sample 回归  全部 OK（MTCNN / NCHWC4 / SSD / CascadeOnet /
                                          FaceDetectorMTCNN / MergeBNCompare /
                                          MergeBNCompareNCHWC / UnusedLayerProbe / LSTM）
    C5 主工程 -O2 -c 优化期告警 HIGH 桶          OK   <- 基线键改成源码文本之后
    C6 ZQCNN 门禁 UBSan 回归                     OK   <- 12 个 BUILD FAIL 修复之后
    A40/A41 "batch 维步进" 自测 + 普查           OK
    B  ZQlib 独立回归 x10 (ASan+LSan)   61/61 通过
        其中新登记的三道：
          zq_padtype          PASS（576 个用例）
          zq_nchwc_batch      PASS（120 个用例）
          zq_nchwc_poolpad    PASS（120 个用例）

**门禁数从 58 增到 61**，三道新门禁全部在 ASan 下跑。

### v70 -> v71 之间修掉的东西

v70 只挂了一个组（C6，12 个 BUILD FAIL），根因是
`tools/zq_net_fwd_tripwires.h` 里 6 个**手抄签名**的打桩定义
对不上附录 DF 改过的新签名 —— 一处手抄同时打掉 12 道门禁。
已按 AGENTS.md 第 33 条修好并固化该条规则。

---

## 变更：附录 DH —— NCHWC 卷积 / 深度卷积的 padding 键**静默丢弃**（先改成响亮拒载）

### 缺陷

NCHWC 这一族只支持**对称**的 `pad` / `pad_H` / `pad_W`，
而 NCHW 那一族还认 `pad_type`（SAME/VALID）与非对称的
`pad_H_top` / `pad_H_bottom` / `pad_W_left` / `pad_W_right`。

模型文件里写这些键时，NCHWC 的 `ReadParam` 原来只打一行
`warning: unknown para`，然后**按 pad=0 继续算** ——
而 `ReadParam` 的返回条件

```cpp
return has_num_output && has_kernelH && has_kernelW
     && has_bottom && has_top && has_name;
```

**不含任何 pad 标志**，所以这是**静默算错**：
`in % stride != 0` 时整层错一格。

### 为什么不直接实现，而是先让它响亮失败

真正支持要改 **21 个前向函数**的签名（9 个 Depthwise + 12 个 Convolution，
每个通道宽度 3~4 个变体），外加所有手抄这些签名的地方
（`tools/zq_net_fwd_tripwires.h` 的 6 个打桩，附录 DF 那次已经吃过一次亏）。
那是一个独立的、明显更大的工程。

**先让它响亮失败**（与附录 BD.2 对非法池化参数的处理一致）：
响亮的失败严格优于静默的错值，而且**改动面小、风险低、可回退**。
真要实现时，这道门禁会自动变红提醒"该撤掉拒载了"。

### 改法

`ZQ_CNN_Layer_NCHWC<Tensor4D>` 新增

```cpp
static bool _is_unsupported_pad_key(const char* key)
```

认 `pad_type` / `pad_H_top` / `pad_H_bottom` / `pad_W_left` / `pad_W_right` /
`same` / `valid` / `pad_type_H` / `pad_type_W` 九个键；
`ZQ_CNN_Layer_NCHWC_Convolution` 与 `_DepthwiseConvolution` 的
"unknown para" 分支在它们命中时**打印层名 + 键名 + 原因并 `return false`**。

**只改这两处**（10 个 "unknown para" 分支里各 1 处），其余 8 个保持原样 ——
它们是别的层，键集合不同（池化已经支持 pad 了，附录 DF），不能一刀切。
改的时候用"向上找最近的 `\tclass ZQ_CNN_Layer_NCHWC...`"来定位归属，
dry-run 先打印每个类各 1 处，确认唯一才 --apply。
> 第一版定位脚本用 5 行滑窗 + 预先算好的 owner 数组，
> 两处都写错了（滑窗只命中 2/10、owner 全是 None）——
> **"改错一处"的代价是改坏别的层**，所以定位必须先 dry-run 并逐类报数。

### 新门禁 `tools/zq_nchwc_padreject_check.cpp`（15 个用例）

判据是「**响亮地失败**」，不是「算对」：
走**完整 `ZQ_CNN_Net_NCHWC::LoadFrom`**（不直接实例化层 ——
"静默"这个缺陷只有在**没人报错**时才是缺陷），
要求 `LoadFrom` 返回 false **且**输出里指名了那个键。

对照组 4 个：**受支持的对称写法必须仍然能加载**。
> 没有对照组的门禁很容易把"全都拒载"也判成通过。

### 变异测试 —— 又踩了一次"变异无效"

第一版变异只加了一句 `(void)keys;`，**函数照样返回 true**，
门禁当然还是绿的。那不是"门禁没鉴别力"，是**变异根本没生效**。

改成让匹配**永不成立**（把比较结果换成 999）之后：

    基线（未变异）  rc=0  1/1 通过（15 个用例：11 拒 + 4 对照）
    变异后          rc=1  FAIL (rc=1, 11 条断言失败)
                          全部是「**加载成功了**（应当拒载）」——
                          也就是修之前那个"静默"的确切形态
    还原            assert 无 MUT 标记且 GOOD 计数回到 1

> 推论：**"变异之后门禁还是绿的"有两个完全不同的成因** ——
> 门禁没鉴别力，或者变异没生效。**先确认变异生效，再讨论门禁。**
> 本会话已经因此多花了两轮（IU 那次是 `--root` 少拼一层目录、
> F2 那次是门禁没经过出缺陷的那一层）。

### 门禁自己栽了两次（两次都是**探针的错**、不是库的错）

1. **临时模型写在 `model/` 下** —— 而门禁的 cwd 是它自己那一轮的 WDIR
   （`/tmp/zqchecks_<pid>_<ts>`），里面没有 `model/`，15 个用例全部"没跑完"，
   症状看起来像"库把受支持的写法也拒了"。
   （本文件 HV.5：相对路径必须连 cwd 一起说清。）
2. **`num_output=4` 用在 Depthwise 上** —— 深度卷积要求
   `num_output == bottom_C`（=3），于是**因为另一个原因**加载失败，
   症状同样是"受支持的写法被拒了"。
3. 判据里拿 `pad_type=SAME` 去 find 输出，而消息里写的是
   `does not support para 'pad_type'` —— 12 个用例误报成"没说清是哪个键"。

> 三次的症状**全都**指向"我们的拒载逻辑在误伤"。
> **两个不同的原因、同一种症状** —— 又一次印证"先怀疑自己那一族"。

### 变更文件

    ZQCNN/ZQ_CNN_Layer_NCHWC.h   _is_unsupported_pad_key + Convolution /
                                DepthwiseConvolution 两处拒载
    tools/zq_nchwc_padreject_check.cpp   新门禁
    tools/run_zqlib_checks.py            登记 zq_nchwc_padreject

### 实测

    python tools/run_zqlib_checks.py zq_nchwc_padreject -> 1/1 PASS
    cmake --build build_x64 --config Release -> RC=0，0 error
    WSL make -j8 -> RC=0，0 error

### 注意事项

- 随仓那 220 个 `pad_type=SAME` 层全在**可整除**的输入上（SAME ≡ 无 pad），
  所以这条拒载对**随仓模型零影响**；但那只是巧合 —— 换个输入尺寸就错。
- 拒载会让"把 Caffe 模型转成 NCHWC"这条路**暂时走不通**（那 220 层转过来会被拒）。
  这是刻意的：转换工具本来也不该静默产出错结果。
  真要支持，需要按上面说的改 21 个前向签名 —— 记在待办里。

---

## 变更：附录 DI —— MNN 转换器那份分叉头也是同一个静默丢弃（补上"第三份拷贝"）

### 先回答一个问题：DF / DH 的修复要不要同步到分叉头？

`ZQCNN_to_MNN/converter/source/` 下有 ZQCNN 的**第三份拷贝**：

    ZQ_CNN_BBox.h / ZQ_CNN_BBoxUtils.h / ZQ_CNN_CompileConfig.h
    ZQ_CNN_Forward_SSEUtils.h / ZQ_CNN_Layer.h / ZQ_CNN_Net.h / ZQ_CNN_Tensor4D.h

按 AGENTS.md 第 33 条（改一处要把所有**手抄/分叉**的地方一起列出来）逐个核过：

* **NCHWC 那两个文件（DF / DH 改的）没有副本** —— `find` 全仓只有一份，
  所以 DF / DH 的覆盖是完整的。
* **DE.1 改的 `ZQ_CNN_Layer.h` 有一份分叉**，但那份**落后于主树**：
  主树 10605 行、分叉 6345 行，`grep -c pad_type` = **0**，
  连 `ZQ_CNN_Forward_SSEUtils.h` 都只有 171 行（一个桩）。
  **也就是说分叉压根没有 pad_type 这个特性** —— 不是"漏同步"，是"那时还没有"。

### 但分叉有**同一个静默缺陷**（不同形态）

分叉的卷积 `ReadParam` 返回条件与主树一模一样：

```cpp
return has_num_output && has_kernelH && has_kernelW
     && has_bottom && has_top && has_name;      // 不含任何 pad 标志
```

而模型里写 `pad_type=SAME` / 非对称 pad 时，它落进 "unknown para"、
只打一行 warning 就**按 pad=0 继续**。

**对一个转换器来说这尤其糟：它的产物就是那张 MNN 图，没人再对一遍。**
主树那种"至少还有 sample 会用到"的下游校验，在这里完全没有。

### 改法：与 DH 同一处理（拒载），并把 C5b 的断言补上

`ZQ_CNN_Layer`（分叉）新增 `_is_unsupported_pad_key()`，
`ZQ_CNN_Layer_Convolution` 与 `_DepthwiseConvolution` 的 unknown-para 分支
命中它时**打印层名 + 键名 + 原因并 `return false`**。

C5b（`tools/probe_mnn_fork.py`）新增三条断言：
判定函数存在 + 卷积那处拒载 + 深度卷积那处拒载。
C5b 本来就是专门盯这份分叉的（顶层 CMake 没有 `add_subdirectory(ZQCNN_to_MNN)`，
转换器又要 MNN 的 `MNN_generated.h`，**那 7 个头从来没被任何编译器看过**），
它逐头 `g++ -fsyntax-only` 编一遍并断言这些守卫还在 ——
所以这次改动**当场被编过**。

### 变异测试：证明两条断言是**各自独立**的

只变异**卷积**那一处（深度卷积那处不动）：

    基线            all 7 headers compile, all guards present
    只变异卷积      GUARD MISSING 卷积的 unknown-para 分支会**拒载**（附录 DI）
                    GUARD OK     深度卷积的 unknown-para 分支同样会拒载（附录 DI）
                    rc=1

**只报一条、另一条仍 OK** —— 两条断言不是盯同一段文本。
这正是 HX 那次的教训：`_merge_bns_to_conv` 与 `_merge_bns_to_innerproduct`
的守卫文本一模一样，全文件搜的话**删掉其中一个另一个照样 OK**，实际只验了一次。
所以 C5b 的作用域机制后来支持了**类**（`^\tclass <name>`）而不仅是函数。

### 改这一份时踩到的两个坑

1. **副本是 CRLF、主树那份是 LF**（实测 6392 CRLF / 0 裸 LF）。
   第一版按 `'\n'` 拼模式，21 处 "unknown para" **一处都没匹配上**，
   报「0 命中」—— 而真实原因是行尾，不是目标形态不存在。
   改成**跟着文件本身的行尾走**、写回时也保持原样，
   免得一次编辑顺手把 6000 多行的行尾全换掉。
2. **新加的判定函数落进了 `private` 段** —— 我把它插在 `\tpublic:` **之前**，
   于是从派生类调用时报
   `error: 'static bool ZQ_CNN_Layer::_is_unsupported_pad_key(const char*)' is private within this context`
   （主树那份恰好在 `public:` 之后，所以主树编过了 —— **两份同名文件的访问级别不同**，
   又是"同仓两份实现"的一个新变体）。
   补一个 `public:` 之后 C5b 报 `0 compile failure(s)`。

### 变更文件

    ZQCNN_to_MNN/converter/source/ZQ_CNN_Layer.h   DI.1 判定函数 + 两处拒载
    tools/probe_mnn_fork.py                       C5b 新增三条断言 + 支持类作用域

### 实测

    python tools/probe_mnn_fork.py --selftest -> all 7 headers compile, all guards present
    变异测试                -> rc=1，只报卷积那一条缺失
    check_text_encoding / check_line_endings -> 全过

---

## 变更：附录 DJ —— 拒载消息的措辞被 sample 回归的 STUB 判据吃掉了

### 现象（v72 的 D4）

    SampleMergeBNCompareNCHWC STUB rc=0 1603ms
      [Layer Conv2d_0/Conv2D does not support para 'pad_type' on NCHWC
       (only symmetric pad / pad_H / pad_W are supported), layer rejected]

**这个 sample 明明加载失败了，却被报成 STUB。**
而 STUB 是**不判失败**的 —— 于是一道真失败被门禁藏了起来。

### 根因

`tools/run_sample_regression.sh` 的判据是

```sh
STUB_RE='only support|not support|not supported|only supports'
if printf '%s' "$out" | grep -qiE "$STUB_RE"; then st=STUB; ...
```

**只看"输出里有没有那句话"，不看输出有几行、也不看 rc。**
两个真桩的输出都只有**一行**
（`./SampleFaceDetectorMTCNN only support windows` / `not support in linux`），
而我那句拒载消息里恰好有 `does not support` —— 于是被归到"平台桩"那一桶。

这与附录 BR.1 是**同一个形状**：回归脚本把不该算通过的算成了通过。
只是这次不是"平台桩 rc=0 被当真跑"，而是**真失败被当成了平台桩**。

### 两处修法

1. **改措辞**（主树 + MNN 分叉各 2 处）：
   `does not support para 'X' on NCHWC (only symmetric ... are supported)`
   → `rejected para 'X' on NCHWC: this net implements symmetric ... only`
   （分叉同理）。避开 `not support` / `only support` 这两个词组。
2. **改判据**（真正的那一处）：STUB 现在要求**两条同时成立** ——
   ① 输出里有 STUB_RE 的话；② **非空行数 <= 3**。
   只满足 ① 的**按 FAIL 处理**，并把那半句和末尾几行都打出来。

   > 判据在**判之前要问"它凭什么成立"**。原来它只凭一句话就成立 ——
   > 而一句话可以出现在任何地方。

### 顺带查清的一件事：`model-face` 并不是被 DH 弄坏的

`model/model-face.zqparams` 满篇都是 `pad_H_top=1 pad_H_bottom=1 ...`
这种**非对称键的长写法**（其实是**对称**的），而且
`SampleMergeBNCompareNCHWC` 确实加载它。所以我一度以为是 DH 把它弄挂的。

翻 v71（DH 之前）的日志才发现：它**早就**在 NCHWC 上加载不了，原因是
**缺整类层**：

    MobileNetSSD_deploy   unknown layer type: Permute
    Pose-zq               unknown layer type: ReLU6
    det5-112-gray         unknown layer type: ReLU6
    headposegaze-112-gray unknown layer type: Normalize
    model-face            （同样缺层）

`pad_type` / `pad_H_top` 的 warning 是**顺带打出来的**，不是失败原因。
> 这正是 AGENTS.md 第 13 条的现场教学：
> **"我读的那几行里没出现"不等于"这不是原因"** ——
> 判失败原因要看**第一个**让它停下来的东西，而不是最后读到的那几行。
>
> 也再次说明 `model-face` 里那些 `pad_H_top=1 pad_H_bottom=1`
> **其实是对称的**、NCHWC 完全表达得了 ——
> 将来真要支持时，正确做法是"对称就接受并使用、不对称才拒载"，
> 而不是一律拒载。这条记在 DH 的待办里。

### 变更文件

    ZQCNN/ZQ_CNN_Layer_NCHWC.h                    2 处消息措辞
    ZQCNN_to_MNN/converter/source/ZQ_CNN_Layer.h 2 处消息措辞（CRLF 保持）
    tools/run_sample_regression.sh               STUB 判据加"非空行数 <= 3"

### 实测

    bash -n tools/run_sample_regression.sh -> 语法 OK
    python tools/probe_mnn_fork.py --selftest -> all 7 headers compile, all guards present
    check_text_encoding / check_line_endings -> 全过

### DJ 的实测（两条证据）

**① 新判据在真实失败上生效**（直接跑 `SampleMergeBNCompareNCHWC.exe`）：

    rc=0   非空行数=40   命中桩字样=4
    旧判据 -> STUB（不判失败）      <-- 真失败被藏起来
    新判据 -> FAIL（判失败）        <-- 正确

命中的是第 25/27/29 行：
`Layer Conv2d_0/Conv2D does not support para 'pad_type' on NCHWC ...`

**② 两个真桩确实只有一行**（`-le 3` 的阈值是量出来的，不是猜的）：

    SampleFaceDetectorMTCNN STUB rc=0 37ms  [./SampleFaceDetectorMTCNN only support windows]
    SampleCascadeOnet_Interface STUB rc=0 42ms  [not support in linux]

Linux 与 Windows 两侧都是**单行**，所以 `-le 3` 留了三行余量也不会误伤真桩。

**③ 改后的措辞不再命中那个正则**（两道防线）：

    主树  rejected para 'X' on NCHWC: this net implements symmetric pad / pad_H / pad_W only
    分叉  rejected para 'X' in the MNN converter: it implements no pad_type and no asymmetric pad

两句话里都**没有** `not support` / `only support`。
即使将来有人又写了含那几个词的错误消息，第 ② 条（非空行数）也会兜住 ——
**措辞是第一道防线，行数是第二道。**

---

## 变更：附录 DK —— DH 的「一律拒载」收得太紧，**对称的非对称写法要接受并使用**

### 起因：DJ 里那条 model-face 线索

`model/model-face.zqparams` 满篇写的是

    Convolution ... pad_H_top=1 pad_H_bottom=1 pad_W_left=1 pad_W_right=1

**这明明是对称的**（top == bottom、left == right），
而 NCHWC 的 `pad_H` / `pad_W` **完全表达得了**。
DH 把这四个键一律拒载，等于把"本来能算的模型"也挡在门外了。

### 改法

`ZQ_CNN_Layer_NCHWC<Tensor4D>` 新增 `_resolve_asym_pad()`，在
Convolution / DepthwiseConvolution 的 `ReadParam` **键循环之后**调一次：

| 四个键的形态 | 处理 |
|---|---|
| 一个都没写 | 放行（走 `pad` / `pad_H` / `pad_W` 那条路） |
| 给齐了 **且** `pad_H_top == pad_H_bottom` 且 `pad_W_left == pad_W_right` | **折成 `pad_H` / `pad_W`，正常往下算** |
| 只给了一部分 / 真的不对称 | **拒载**，并把四个值都打出来 |

`_is_unsupported_pad_key` 的名单相应缩小成
`pad_type` / `same` / `valid` / `pad_type_H` / `pad_type_W` ——
它们的语义要靠 `bottom_H` / `bottom_W` 才算得出来，而那要等 `SetBottomDim`，
**仍留在拒载名单里**（真正实现要改 21 个前向签名，单独排期）。

拒载消息也改成**点名那四个键**（`got asymmetric pad: pad_H_top=… pad_H_bottom=…`），
原来的 `top=/bottom=` 不含任何键名，门禁的"指名"判据看不到。

### 门禁 `zq_nchwc_padreject` 扩到 20 个用例

新增的**判别**用例（缺了它们，一个"看到 pad_H_top 就放行"的实现会混过去）：

    must-reject  conv pad_H_top 单独给（另一半按 0，仍不对称）
    must-reject  conv top/bottom 不等（1/0, 1/0）
    must-reject  conv top/bottom 相等但左右不等
    must-reject  dwconv top/bottom 不等
    must-accept  conv  四键对称=1   <-- model-face 的写法，DH 下会被拒，DK 下必须能加载
    must-accept  dwconv 四键对称=1
    must-accept  conv  四键对称=0

### 这一版改代码时自己栽的三次（都是同一个坑的变体）

1. **多轮改写之间行号失效**：第一版按 ctor -> mem -> close -> keys 分四轮改，
   而 close 那一轮多插了 7 行，于是后面 keys 用的（改之前算好的）行号
   整体错位 7 行，第二个类被改坏、`pad_W` 分支被复制了一份。
   > 判据：**一轮改写 = 一次下标失效**。所有替换区间要在**原始行表**上算好，
   > 再在**同一趟**里按行号从大到小应用。
   > 症状是"锚点命中了但结果不对"—— 比"没命中"更危险。
2. **行尾没探测**：`git checkout -- <file>` 在 `core.autocrlf=true` 下把工作区
   重新物化成 **CRLF**（改之前那个文件是 LF），按 LF 拼的锚点一个都不匹配。
   这与 MNN 分叉那次**是同一个坑、同一个文件**—— 我在两次里各栽了一次。
3. **`find_block` 用 `len(text)` 当行数**（那是**字符**数），
   于是拿 189 行的窗口去比 7 行的模式。

另外两处：

* `git checkout` 把我先前用 Edit 加的两个辅助函数**一起回退**了
  （只 revert 了"改坏的那次编辑"，没意识到它连带 revert 了别的），
  于是紧接着一次编译报 `there are no arguments to '_resolve_asym_pad'`。
* 修好之后仍然编不过，因为调用**没写全限定名**：
  派生类是模板、基类 `ZQ_CNN_Layer_NCHWC<Tensor4D>` 是**依赖基类**，
  而这次调用**没有任何参数依赖模板参数**，按两阶段查找必须在定义点就找得到 ——
  写裸名字直接编不过。上面那个 `_is_unsupported_pad_key` 没这个问题，
  因为它**一直是全限定的**。
  > 判据：在模板里调**依赖基类**的成员，若参数里**没有**模板相关的量，
  > 就必须写 `基类<T>::成员`，不能写裸名字。
* `check_line_endings` 报 `mixed-EOL(3243 CRLF/4 LF)` —— 4 处 Edit 插进去的行
  是 LF。用门禁自己的 `--fix` 归一。

### 实测

    python tools/run_zqlib_checks.py zq_nchwc_padreject -> 1/1 PASS（20 个用例）
    cmake --build build_x64 --config Release -> RC=0，0 error
    WSL make -j8 -> RC=0，0 error
    check_text_encoding / check_line_endings / check_stmt_joins -> 全过

### 注意事项

- `pad_type` / `same` / `valid` 仍**拒载**：它们的语义要等 `SetBottomDim`
  才知道。随仓那 220 个 SAME 层仍然只在 NCHWC 侧被拒 ——
  这是 DH 已记录的状态，本条没有改变它。

---

## 变更：附录 DL —— 某次脚本批量改写**在两个文件上各跑了两遍**

### 现象一：`ZQCNN/ZQ_CNN_PersonPose2.h`（749~570 行一带）

    // 审计修复（附录 IN.8）：返回值原来被丢弃。        <- 第 1 遍
    ... 5 行 ...
    // 审计修复（附录 IN.8）：返回值原来被丢弃。        <- 第 2 遍，一字不差
    ... 5 行 ...
    temp_img.ConvertFromBGR(...);                     <- 返回值**仍被丢弃**（原调用）
    if (!temp_img.ConvertFromBGR(...)) { ...return false; }
    if (!temp_img.ConvertFromBGR(...)) { ...return false; }   <- 同一段又来一遍

后果：同一张图被**转换三次**（第 1 次还丢弃返回值）。

**IN.8 要修的那个缺陷其实一直没修掉** —— 只是后面补了两次带检查的调用把它盖住了。
"后面那两次会重做"是**读代码推出来的**，不是量出来的，所以按原样记在这里。

### 现象二：`SamplesZQCNN/SampleMergeBNCompare/SampleMergeBNCompare.cpp`（436~443 行）

    printf("        前 10 个（未融合 -> 融合）：");
    for (int z = 0; ...) printf(" [%d %.6g->%.6g]", ...);
    printf("\n");
    printf("        前 10 个（未融合 -> 融合）：");      <- 逐字相同，又一遍
    for (int z = 0; ...) printf(" [%d %.6g->%.6g]", ...);
    printf("\n");

后果：那行会**打印两遍**。这个 sample 在双平台 sample 回归里跑着。

### 通用探测器 `tools/find_dup_blocks.py`

找「**相邻且逐字相同**的 >= 4 行代码块」—— 上面两处的形状。

    第一版：142 处 / 20 个文件
    排除「按宏展开成多份」的那一族之后：23 处 / 6 个文件

**排除名单必须写出来并说明理由**（AGENTS.md 那条：白名单不写理由，
下一个人就不知道它是"查过了"还是"忘了查"）：

    ZQCNN/layers_c/、ZQCNN/layers_nchwc/   同一份 _raw.h 被 include 2~3 次，
                                            生成的函数体在源文件里就是背靠背的
    ZQ_GEMM/math/zq_gemm_32f_align_c.c     同上
    SamplesZQCNN/example_for_very_high_gflops/、SampleMatMul/  内含大段生成代码

剩下的 23 处里，`3rdparty/` 4 处不是我们的代码；
`testImageProcessing.cpp` 11 处是它**内联了一份张量实现**（同样属于展开型）；
ZQCNN 自己的只剩上面那 2 处，**都已修**。

### 改这两处时自己栽的一次

第一版按**整段字符串**匹配，报「带检查的块 0 处」——
而 `if (!temp_img.ConvertFromBGR` 与 `ConvertFromBGR failed` 实测**各 2 处**，
**重复是真的**，只是模式里 `printf` 那行的 `\n` 转义写错了。
改成**按行号删**（删之前先核对那几行逐字相同）才做对。
> 与 DK 那条「锚点没匹配上要先怀疑模式」是同一条：
> **"模式没匹配上"和"目标不存在"在输出上一模一样。**
> 所以每次都要另找一条**独立的**证据（这里是把子串分开数）。

### 变更文件

    ZQCNN/ZQ_CNN_PersonPose2.h                        归一成一份注释 + 一次带检查的调用
    SamplesZQCNN/SampleMergeBNCompare/SampleMergeBNCompare.cpp  删掉重复的 printf 块
    tools/find_dup_blocks.py                          通用探测器（**尚未进回归**）

### 实测

    改后：if (!temp_img.ConvertFromBGR) 1 处 / 裸调用 0 处 / printf 1 处
    改后：「前 10 个（未融合」1 处
    cmake --build build_x64 --config Release -> RC=0，0 error
    WSL make -j8 -> RC=0，0 error
    check_text_encoding / check_line_endings / check_stmt_joins -> 全过

### 注意事项

探测器**没有进回归**：它只报"形状可疑"，判断还得人做，
而排除名单一旦随着代码变动而失效，恒红项会把整栏信号清零
（同 AGENTS.md 第 20 条）。留在仓库里当排查手段。

### DL 的阴性结论：全仓**没有第二处**「审计修复被改了两遍」

把 `find_dup_blocks` 的相邻重复检测加上「块内含审计标记（`审计` / `附录 XX`）」这一层，
全仓（ZQCNN / ZQ_GEMM / ZQlibFaceID / SamplesZQCNN / SamplesZQlibFaceID /
ZQCNN_to_MNN / tools / 3rdparty-ZQlib）扫一遍：

    合计 0 处「相邻重复且含审计标记」的块。

**这个 0 是有阳性对照的**（"扫到 0 命中"在本会话已经栽过太多次）：

    阳性对照（git show HEAD~2 的 PersonPose2，即 DL 修之前）
        全部相邻重复 2 处，其中含审计标记 1 处
        :550  5 行  // 审计修复 2026-10-06（附录 IN.8）：返回值原来被丢弃。
    当前 PersonPose2
        全部相邻重复 0 处

第一版的探测器**没通过这个对照**：它比的是「**连续多行彼此相同**」，
而"一段 5 行注释被复制了两遍"不是那个形状 —— 于是它对着阳性对照也报 0。
换成 `find_dup_blocks` 的「相邻两块逐字相同」才对上。
> 又一次：**"0 命中"在没有阳性对照时什么都不能证明。**

结论：DL 那类「脚本跑两遍」在本仓库**只发生了那两处**，已经都修掉。
这个扫描**不进回归**（一次 142 秒、常态 0 命中，恒绿的门禁没有价值）。

### DL 的第二个阴性结论：相邻重复里**没有**藏着一个 double-free / 重复自增

把检测缩到「相邻 2~3 行逐字相同，且块里含 `free(` / `delete` / `release` / `++` / `--`」
（重复释放、双重自增会藏在这个形状里），排除宏展开那一族之后全仓扫：

    合计 1 处，逐条看过之后是**误报**：
    ZQlibFaceID/ZQ_FaceRecognizerSphereFaceOpenCV.h:84
        *cur_pix_ptr = ori_pix_ptr[0];
        cur_pix_ptr++;
        *cur_pix_ptr = ori_pix_ptr[0];     <- 命中的是这两行
        cur_pix_ptr++;
        *cur_pix_ptr = ori_pix_ptr[0];

那是 `ZQ_PIXEL_FMT_GRAY` 分支：把**单通道**灰度广播进 3 通道的 `bgr_buffer`，
读三次 `ori_pix_ptr[0]` **正是它该做的**。判据命中的是 `cur_pix_ptr++`，
不是 `free` / `delete`。
> 顺带一条：这个形状（同一段 `x = a[i]; x++;` 背靠背出现）本来就是
> 「把一个值广播到多个位置」的常见写法，**光靠"重复 + 含 ++"判不了**，
> 必须回去看它在干什么 —— 本文件「没有证据就不要断言原因」。

所以 DL 这一族在**三种形状**（>=4 行相邻重复 / 含审计标记的相邻重复 /
含 risky 关键字的 2~3 行相邻重复）下都查过了，
真缺陷只有已修的那两处，其余是阴性结论。

### DN 的配套检查：把本会话所有消息片段丢进各门禁的启发式试一遍

DN 是**回归撞出来的**（A10 变红），代价是一整轮 v75。
所以补一次**事后**检查：把本会话新增/修改过的错误信息逐条过一遍已知的门禁启发式。

    抽到 7 条消息片段，撞上 4 条，逐条核实：

    ZQ_CNN_Layer_NCHWC.h:706  "invalid conv params: kernel "
    ZQ_CNN_Layer_NCHWC.h:1279 "invalid conv params: kernel "
    ZQ_CNN_Layer_NCHWC.h:2461 "invalid pooling params: kernel "
        -> 这三条**就该**命中 `zq_model_params` 的 REJECT_MARKS：
           它们本来就是"被守卫拒绝"的标记，门禁靠它把"模型被拒"和"没走到那步"分开。
           **不是碰撞，是刻意的耦合。**

    ZQ_CNN_Layer.h:221  '漏掉 H/W 就会造出零尺寸张量'
        -> 那是**中文注释**不是消息串，被我那个 9 行窗口的抽取器误抓。
           DIV 正则把 `H/W` 里的斜杠当成了除法 —— 同一个坑的第三次出现形态。

结论：**本会话的消息没有一处真碰撞**。

顺带核实了一件我原本担心的事：`zq_model_params` 遇到**不带 REJECT_MARK 的加载失败**
会怎么判（DH/DK 的拒载消息就不在名单里）——

    bad++;
    printf("  %-34s **FAIL**%s%s\n", base,
           why ? " 被守卫拒绝: " : " 没走到权重那步", why ? why : "");
    // 紧接着打印输出的前 3 行

即**判 FAIL、并把那几行打出来** —— 是可诊断的失败，不是静默放过。
所以拒载消息**不需要**去凑 REJECT_MARKS 的词。
（凑上去反而会变成"被守卫拒绝：<某个不相关的标记>"，更难定位。）

> 第三次「`/` 被当成除法 / 词组被当成桩」，三次形态不同：
> 真除法、消息里的斜杠、注释里的 `H/W`。
> **判据对**任何 `/` 都敏感，而 `/` 在中文技术写作里很常见 ——
> 所以写带路径/比例的注释之前，先知道 A10 会报它。

---

## v76 全量回归：**ALL CHECKS PASSED**（RC=0，确认 DN 修好）

    python tools/run_audit_checks.py --with-build --warn-sweep --src-sweep \
        --bounds-sweep --ubsan-sweep --reachability --msvc-asan
    -> ALL CHECKS PASSED   RC=0

    D1/D2 双平台全量构建                OK
    D3/D4 sample 回归（Linux+Windows）  OK，0 问题
    C3/C4/C5/C5/C5b/C7/C10~C16        OK
    C6 ZQCNN 门禁 UBSan 回归   62/62 通过
    B  ZQlib 独立回归 x10      OK
    A10 除以模型参数           OK   <- 附录 DN 改措辞之后转绿
    A40/A41 batch 维步进       OK

顺带确认 sample 的分类也对了：

    SampleMergeBNCompareNCHWC OK rc=0 1580ms  42 行输出   <- 不再被误判成 STUB
    SampleFaceDetectorMTCNN    STUB rc=0 43ms  [only support windows]   <- 真桩仍然是 STUB
    SampleCascadeOnet_Interface STUB rc=0 42ms  [not support in linux]  <- 同上

**这一轮把本会话的十一项（AZ / DE.1 / DE.5 / DF / DG / DH / DI / DJ / DK / DL / DN）
全部在一次干净的回归里确认。**

---

## 变更：附录 DO —— NCHW 池化带 padding 时**用了 Padding 之前的步长**（360/432 个用例错）

### 缺陷

`ZQ_CNN_Forward_SSEUtils::MaxPooling` / `AVGPooling` 在函数开头读

```cpp
int in_pixStep  = input.GetPixelStep();
int in_widthStep = input.GetWidthStep();
int in_sliceStep = input.GetSliceStep();
```

之后才进 padding 分支，而 **`Padding` 会重建张量**（`realW` 变了）：

```cpp
input.Padding(pad_W_left, pad_W_right, pad_H_top, pad_H_bottom, 0);
const float* in_data = input.GetFirstPixelPtr()
                     - pad_H_top*in_widthStep - pad_W_left*in_pixStep;   // 旧步长
_maxpooling(..., in_pixStep, in_widthStep, in_sliceStep, ...);          // 也是旧步长
```

实测（4x4 / C=4 / pad=1，装置见下）：

    == Padding 之前 ==   pixelStep=4  widthStep=16  sliceStep=64   borderW=0
    == Padding 之后 ==   pixelStep=4  widthStep=24  sliceStep=144  borderW=1

于是**指针偏移**与**内核用的行步长**不是同一套几何，窗口落到错的位置。
凡是 `pad != 0` 都受影响（而 `pad == 0` 完全不受影响 —— 这一条正是它躲过所有回归的原因）。

### 新门禁 `tools/zq_nchw_poolpad_check.cpp`（432 个用例）

判据：NCHW 池化在 `pad != 0` 时必须等于「零填充后按 kernel/stride 池化」。
覆盖 k{2,3} x s{1,2,3} x pad{0 / 对称 / 非对称 / 不对称且不等} x C{4,6}。

    修之前   72/432 全对，**360 个红**（全在 pad != 0 上）
    修之后  432/432 全对

### 这一段我自己栽了两次，都值得记

**① 门禁的参考写错了，差点去"修"一个不存在的 bug。**

第一版参考把「原图坐标」和「补齐区坐标」混着用：

```cpp
const int ih = oh * c.s - c.pT + kh;   // 以为这是补齐区下标
...
in[((size_t)ch * c.H + (ih - c.pT)) * c.W + (iw - c.pL)]   // 又减了一次 pT
```

于是参考给出 `0 0 0 / 0 6 8 / 0 14 16` 这种一眼就不对的东西，
而库给 `1 3 4 / 9 11 12 / 13 15 16` —— **库是对的**。
并排打印才看明白：零填充语义下窗口起点 `-1` 映射到补齐区下标 `0`，
**pad 只该减一次**。

这也**更正了附录 DF**：DF 里据"库给 7 8 9、参考给 9 11 12"判定
"NCHW 带 padding 池化是错的"—— **那条结论是错的**，错在参考。
（`7 8 9` 是**修之前**的陈旧步长算出来的，那才是真缺陷。）

**② 变异测试无效，导致我把修好的代码撤了。**

撤之前我跑 `tools/_mut_do.py` 想量"修之前错多少"，结果报"修之前也 432/432 全对"。
查下去：那个变异只把三行里的

    const int pad_pixStep = input.GetPixelStep();   ->   in_pixStep

**另外两行（`pad_widthStep` / `pad_sliceStep`）没换** ——
于是变成了"像素步长用旧的、行步长用新的"这个**两边都不沾**的状态，
而它恰好也全对。**不是"没有 bug"，是"变异没生效"。**
把三行一起退回（真正的修前状态）才是 72/432。

> 本会话第 N 次同一条：**"变异之后门禁还是绿的"有两个成因** ——
> 门禁没鉴别力，或者变异没生效。**先确认变异生效，再讨论门禁。**
> 而"确认变异生效"最省事的办法是**看它有没有把该改的都改了**
> （数一数替换处数），不是看它跑出来什么。

顺带：撤销那一步自己也犯了 DK 那条"多轮改写之间行号失效"——
替换模式没把 `input.Padding(...)` 那一行包进去，撤回去的时候它被复制成了两份。
`check_stmt_joins` 抓到之后手工去掉了（`input.Padding(pad_W_left` 现在仍是 2 处）。

### 变更文件

    ZQCNN/ZQ_CNN_Forward_SSEUtils.h   MaxPooling / AVGPooling 各一处：
                                        Padding 之后重读 pixStep/widthStep/sliceStep
                                        + 一段"曾经误判它有 bug"的说明
    tools/zq_nchw_poolpad_check.cpp    新门禁
    tools/run_zqlib_checks.py         登记 zq_nchw_poolpad（门禁总数 62 -> 63）

### 实测

    python tools/run_zqlib_checks.py zq_nchw_poolpad -> 1/1 PASS
    cmake --build build_x64 --config Release -> RC=0，0 error
    WSL make -j8 -> RC=0，0 error
    check_text_encoding / check_line_endings / check_stmt_joins -> 全过

### 影响面

随仓 27 个模型的池化层 pad **全为 0**（220 个 SAME 层都解析成 0），
所以这个修复对**随仓 sample 的输出零影响** ——
它修的是一条"换个输入尺寸 / 换个模型就会踩"的路径。

---

## v77 全量回归：**ALL CHECKS PASSED**（RC=0，确认 DO）

    python tools/run_audit_checks.py --with-build --warn-sweep --src-sweep \
        --bounds-sweep --ubsan-sweep --reachability --msvc-asan
    -> ALL CHECKS PASSED   RC=0

    D1/D2 双平台全量构建                 OK
    D3/D4 sample 回归（Linux + Windows） OK，0 问题
    C3/C4/C5/C5/C5b/C7/C10~C16           OK
    C6 ZQCNN 门禁 UBSan 回归   63/63 通过（含新增的 zq_nchw_poolpad）
    B  ZQlib 独立回归 x10      OK
    A10 / A40 / A41               OK

门禁总数 **58 -> 63**，本会话新增五道：
    zq_nchwc_batch     NCHWC 的 batch 维不变性（10 个算子 x 3 个宽度）
    zq_padtype         NCHW pad_type 的输出尺寸与 padding 之和（576 个用例）
    zq_nchwc_poolpad   NCHWC 池化带 padding（120 个用例，两段覆盖）
    zq_nchwc_padreject NCHWC 表达不了的 padding 键必须拒载（20 个用例）
    zq_nchw_poolpad    NCHW 池化带 padding（432 个用例）

### 本会话的净结果

修掉 12 项（AZ / DE.1 / DE.5 / DF / DG / DH / DI / DJ / DK / DL / DN / DO），
更正 1 条自己写错的结论（DF 里"NCHW 带 padding 池化错位"——错在参考把 pad 减了两次），
补了 3 条阴性结论（DG 的"解析了却从不使用"全仓已清干净 / 两次扫描的误报），
AGENTS.md 增补第 33~35 条。

**留在待办里、本轮刻意没做的**：
    NCHWC 真正实现 `pad_type` / `same` / `valid` —— 要改 21 个前向函数的签名
    （9 个 Depthwise + 12 个 Convolution），外加所有手抄这些签名的地方。
    现在这些键在 NCHWC 侧是**响亮拒载**（附录 DH/DK），不会静默算错；
    随仓 220 个 SAME 层只在**可整除**的输入上跑（SAME 等于无 pad），
    所以这条待办对现有 sample 零影响 —— 但它是"换个输入尺寸就会踩"的路径。
---

## 新增：平台分支上的 MSVC 专有拼写门禁（附录 IJ）

### 变更文件

    新增 tools/probe_platform_divergence.py   两部分探针 + --selftest（15 个用例）
    改   tools/run_audit_checks.py             接入 C17 / C17b（快组）

### 问题

`fopen_s` / `sprintf_s` / `sscanf_s` / `_mkdir` / `Sleep` 这些在 Linux/gcc 上
**根本不存在**。两边构建都过**不代表没有** —— 只有被 `#if defined(_WIN32)`
圈住才安全。

先手工核实过 ZQlibFaceID：那些 `fopen_s` 看着像没圈住，实际
`ZQ_FaceDatabase.h:1083` 是规规矩矩的 `#if defined(_WIN32) / #else`。Linux 上
不带任何垫片编 `ZQ_FaceDatabase.h` 也确实通过，所以**没有真缺陷**。

### 实测结果

一方代码（ZQCNN / ZQlibFaceID / ZQ_GEMM / Samples* / model）：

    未圈住的 MSVC 专有拼写     0 处
    全树 `#if _WIN32 ... #else` 区域      7 个
    其中常量不对称的             1 个

唯一那一个是 `model/benchncnn.cpp:167` 的 `Sleep(10 * 1000)` vs `sleep(10)` ——
毫秒对秒，**是对的**。所以常量那一半**只报不判**，只有未圈住那一半判红。

### 注意事项

这道门禁是**先坏后修**的，过程值得记：

1. 第一版要求"每一层外层 `#if` 都得是 win32 判定"，于是每个带 include guard
   的头整份被判成未圈住，报出 **109 处假阳性**。
2. 改成"存在某一层把代码限制在 win32 就算圈住"之后，仍有真实漏洞：在只被
   include guard 包住的头里植入一个真 `fopen_s`，门禁**照样绿**。
3. 原因是空栈/中性层的语义写反了 —— include guard 本身**不构成**保护。

结论就是 AGENTS.md 那条：**门禁在变异后仍然绿，先怀疑门禁没鉴别力**。
这次判据是"在真实文件里做变异、并数命中数"，光看自测不够 ——
自测的 include-guard 用例里同时含 win32 层，正好盖住了这个 bug。---

## 新增：GEMM 调度的正确性覆盖从来没跑过（附录 IK）

### 变更文件

    改 tools/run_zqlib_checks.py       zq_gemm_shape 移出 SLOW，进默认通道
    改 tools/run_audit_checks.py       新增 --with-slow（透传给 run_zqlib_checks.py）
    改 tools/zq_gemm_shape_check.cpp   改掉两句失效的"现在是坏的 / 在 SKIP 里"

### 问题

扫「声称当前状态已坏的注释」时撞到 `zq_gemm_shape_check.cpp` 尾部两句，
都不成立：「崩溃/结果错的形状现在就是坏的，所以这个测试当前应当是红的」
是附录 BO 修 K-align fallback **之前**的状态；「门禁里它是 SKIP」也不对 ——
附录 BP 已经移出 SKIP，现在 `SKIP = {}` 是空的，它挂的是另一个集合 SLOW。

顺着"它为什么不跑"查下去发现问题更严重：**全仓没有任何地方传过
`--with-slow`** —— run_audit_checks.py 不带、脚本不带、AGENTS.md 里写的
"全量回归"命令也不带。于是 SLOW 里 8 个测试**历史上一次都没被自动执行过**。

其中 5 个正好是 GEMM 调度器 `zq_gemm_32f_AnoTrans_Btrans_auto` 的全部调用点
（zq_nchw_conv / zq_nchwc_conv / zq_nchwc_conv8 / zq_nchwc_ip / zq_innerproduct，
外加形状安全图 zq_gemm_shape）。也就是说用户这一轮反复要求"和 MKL 对标"的
那个调度器，其**正确性**在默认通道里一条用例都没有。

### 实测结果

    zq_gemm_shape 实跑三次，全部 PASS，耗时 150s / 150s / 150s
    第三次已**不带** --with-slow，即验证改动生效

3240 个网格用例 + 14 个 production 形状全部通过，零崩溃零错值 ——
附录 BO 的 K-align fallback 有效。SLOW 给它写的排除理由
"zq_gemm_32f_align_c.c 单个 >5 分钟"经实测是**假的**。

### 注意事项

150 秒换 GEMM 调度唯一的形状安全图，划算，已进默认通道。
剩下 7 个仍昂贵：它们各自把**同一个 TU** 单独编一遍到**不同文件名**
（编译命令/头路径/sanitizer 档位全同），整轮里 `zq_gemm_32f_align_c.c`
被编 6 次以上。方向是**编一次共享 .o** —— `zq_nchwc_conv` / `zq_nchwc_conv8`
已经在这么干了，推广过去能让 `--with-slow` 便宜 5~6 倍。

在那之前 **AGENTS.md 里的"全量回归"命令要加 `--with-slow` 才真正覆盖 GEMM 调度**。

**教训**：一条自称"这里是坏的"的注释长期错，往往不是因为没人写，
而是因为**验证它的那个测试根本没跑**。测试不跑 = 状态注释必然腐烂。---

## 变更：EXTRA_SOURCES 去重 —— 同一个 TU 一轮里编 6 次以上（附录 IL）

### 变更文件

    改 tools/run_zqlib_checks.py

### 问题

接附录 IK。`SLOW` 挡人的理由是「zq_gemm_32f_align_c.c 单个 >5 分钟」，
但**贵不是因为文件大，是因为它被反复编译**。同一个 TU 被十几条命令各编一遍，
编译命令/头路径/sanitizer 档位**全同**，只有输出文件名不同
（zq_gemm_align.o / zq_shape_gemm_align.o / zq_nchwcip_gemm_align.o / ...）。
asm 与 auto 两个 TU 同样。

### 实测结果

    编译次数  239 -> 82，减少 66%（157 次重复编译被 cp 取代）
    python tools/run_zqlib_checks.py --with-slow   71/71 通过，ELAPSED=1059s

顺带确认一条阴性结论：附录 IK 说那 5 个 GEMM 调度调用点测试从未自动跑过，
现在全部 PASS，调度器没有潜伏缺陷。

**没有做改动前的整轮耗时基线** —— 早先那次 933s 有 27 个测试在链接期就失败了，
比现在快纯粹因为提前退出，不能拿来比。能确切说的是编译**次数**。

### 注意事项

**共享的判据是「整条命令逐字相同」，不是「同一个源文件」。** 第一版按源文件名
共享，结果 `zq_gemm_shape` 那条**没有** `$SAN`、别家有，于是把一份没插桩的
`.o` cp 给了 sanitizer 门禁—— 照样全绿，但越界读一个元素也没人报，
正是 2026-10-02 栽过的那一跤。现在键取「抹掉 -o 名字后的整条命令」，
旗标进键：`zq_gemm_32f_align_c` 因此是**两个**对象（一个插桩一个不插桩）。

**第二版自己砸了整轮 44/71**：第一次那条被**改名**成规范名，于是第一个用到它的
门禁链接时找不到自己的 `.o`，27 个 BUILD FAIL 全是链接期找不到文件 ——
DY.8 那个「错误出现在错误的地方」。改成「第一次原样发 + 额外 cp 一份规范名」。

静态验证比跑一遍更早发现问题：直接生成改动前后两份脚本，比对产出的文件名集合，
**丢失 0 个、新增 82 个**。这个检查值得当常规动作。

留了一个疑点没动：`zq_gemm_shape` 的编译命令里**没有** `$SAN`，而它测的正是
内核越界读 —— 内核没插桩，ASan 抓不到「读超一两个元素」那种越界，
而附录 BO 的 padK 越界恰好就是这种。补 `$SAN` 会让这个 TU 的编译慢到不可接受
（单编超过 10 分钟），只记下来。---

## 变更：GEMM 形状安全图补 $SAN —— 实测买到的是确定性（附录 IM）

### 变更文件

    改 tools/run_zqlib_checks.py    zq_gemm_shape 的三条 EXTRA_SOURCES 补 $SAN

### 问题（比上一轮记的更大）

上一轮我记成「zq_gemm_shape 一个测试缺 $SAN」。查全量后发现是 **6 个**：
zq_gemm_shape / zq_innerproduct / zq_nchw_conv / zq_nchwc_conv /
zq_nchwc_conv8 / zq_nchwc_ip —— 这 6 个 tag 一共 30 条 EXTRA_SOURCES 命令，
**没有一条带 $SAN**，即它们测的 conv / innerproduct / GEMM 内核
**从来没被插桩过**，ASan 在这 6 个测试里只覆盖了测试自己的代码。

zq_gemm_shape 存在的意义恰恰是抓「读超了一点点」（附录 BO 的 padK 越界），
而 ASan 是靠编译期给 load/store 插检查的 —— 没插桩的代码里，
越界读只有落到未映射页才会 SEGV，同一页之内完全静默。

### 实测结果

在 fallback 里植入一处越界读（`k < K` -> `k <= K`，断言只出现 1 处），
同一份被测代码跑两种构建：

    插桩(ASan)   ok 2171 / 崩溃 1083 / 结果错 0     确定性报出
    不插桩       ok 1330 / 崩溃  997 / 结果错 927   也报，但靠垃圾值

**结论和我预设的不一样**：不插桩**也报出来了**，所以「不插桩就是瞎的」是错的
（这句话我一度写进注释，已改掉）。真正的差别是机制 —— 插桩版在第一个越界就
abort、3254 个用例确定性全报出；不插桩版读到的地方是不是未映射页、
读到的数够不够大，取决于那块内存当时恰好是什么，**换一次运行就可能不报**。

所以买到的是**确定性**，不是「从看不见到看得见」。保留它的理由就在这里。

### 代价（如实记）

    整轮 --with-slow   1059s -> 1751s  (+692s / +65%)，71/71 通过
    单跑 zq_gemm_shape  150s ->  698s，PASS

贵在 ASan 对 3240 个 fork 子进程的**运行时**开销，不在编译。
顺带纠正上一轮的猜测：「补 $SAN 在去重之后几乎免费」在**单跑**时是错的 ——
去重省掉的是编译，省不掉运行时插桩开销。

### 注意事项

插桩之后 zq_gemm_shape 仍然 PASS，即形状安全图本身无潜伏缺陷，
上面那些数全是变异版的结果。变异已还原并核对（git diff 为空）。

另外 5 个 tag 的 30 条 EXTRA_SOURCES 同样缺 $SAN，这轮**没动**：
它们在 SLOW 里不是默认通道；补上要重新验一整轮 --with-slow（30 分钟起步）；
代价模型已量出来，可以先估再决定。---

## 变更：GEMM 那 5 个 tag 也补 $SAN，并列清剩下 18 个（附录 IN）

### 变更文件

    改 tools/run_zqlib_checks.py    zq_innerproduct / zq_nchw_conv /
                                     zq_nchwc_conv / zq_nchwc_conv8 / zq_nchwc_ip
                                     共 27 条 EXTRA_SOURCES 补 $SAN

### 实测结果

    python tools/run_zqlib_checks.py --with-slow    71/71 通过，ELAPSED=1594s

对照上一轮（只插了 zq_gemm_shape）的 1751s，没有变慢。开销主要落在
zq_gemm_shape 那 3240 个 fork 子进程上，这 5 个 tag 的用例数少得多。

**阴性结论**：插桩后 71/71 全绿，**没有冒出潜伏缺陷** ——
conv / innerproduct / GEMM 内核在 ASan 下干净。

补完之后，EXTRA_SOURCES 里编译 zq_gemm_32f_align_c.c 的 **11 个 tag 全部带
$SAN**，即附录 IK 点名的「GEMM 调度的全部调用点」加形状安全图，一个不落。

### 注意事项

改这份文件时脚本连栽两次，都是脚本自己的问题，已写进报告：

1. 按整条字面量替换 —— EXTRA_SOURCES 里一个编译命令在源码上是**两行隐式
   拼接**，逻辑上一条、源码里不连续，`count` 恒为 0；
2. 改全局替换更糟 —— 全文件 64 处编译行不带 $SAN，我要改的只有 27 处，
   全局替换会顺手改掉另外 37 处**不属于这 5 个 tag** 的命令。

**判据：改某一个 tag 的 EXTRA_SOURCES，只能按 tag 块定位后在块内逐行替换，
绝不能按字面量全局替换**（同一个编译前缀被几十个 tag 共用）。最终做法是按
`'tag': [` 切块、块内逐行替换，并断言改动行数 == 27。

**更大的范围**：全量列表后发现仍有 **18 个 tag** 的编译行不带 $SAN
（zq_bns / zq_eltwise / zq_lrn / zq_nchw_act / zq_nchw_depthwise /
zq_nchw_lstm / zq_nchw_reduction / zq_nchw_resize / zq_nchw_scalop /
zq_nchw_sqrtnrm / zq_nchwc_act / zq_nchwc_bn / zq_nchwc_depthwise /
zq_nchwc_elt_relu / zq_nchwc_pool / zq_nchwc_resize / zq_nchwc_softmax），
它们测的 NCHW/NCHWC 各层实现 TU 同样没被插桩。这轮**没动**，表已列好，
下一轮照着做即可。