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
