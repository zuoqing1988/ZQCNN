# build with cmake

CMake 是本项目唯一的构建入口（仓库中不再保留 Visual Studio 的 .sln/.vcxproj，
那些文件停留在 VS2015/v140 且文件列表早已与源码脱节）。

## windows

Take Visual Studio 2022 as example:

```shell
cmake -S . -B build_x64 -G"Visual Studio 17 2022" -A x64
cmake --build build_x64 --config Release -j8
```

产物在 `cmake-out-win32-x64/release/Release/`，其中 `data/` 与 `model/` 为指向仓库根目录的目录联接（junction）。
如果本机没有可用的 OpenCV，CMake 会自动回退到 `3rdparty/opencv` 里的预编译库。

## linux

默认走自带的 ZQ_GEMM，**不需要**任何 BLAS：

```shell
mkdir cmake-build-release && cd cmake-build-release
cmake ..
make -j4
```

**用 OpenBLAS**：`add cmake flag: -DBLAS_TYPE=openblas`，并把**对应你架构**的
OpenBLAS 放进 `3rdparty/lib`。

> ⚠️ **仓库随附的 `3rdparty/lib/libopenblas.{so,a}` 是 ARM(32 位) 的**
> （它是给下面 `SIMD_ARCH_TYPE=arm` 那条路径准备的）。
> 在 x86 上直接 `-DBLAS_TYPE=openblas` 会在 **configure 阶段**失败，
> 并明确告诉你"这个库是 ARM(32 位) 的，而当前构建要的是 x86-64"。
> 这条检查是 2026-10-03 加的（见 `audit_k3_20261001.md` 附录 GL）——
> 在那之前同样的命令会一路绿灯编过去，实际什么也没切换。

**用 MKL**：`add cmake flag: -DBLAS_TYPE=mkl`，并把 MKL 放进 `3rdparty/lib`。
仓库随附的只有 Windows 的 `3rdparty/lib/mklml.lib`；Linux 需要自己下载
[mklml_lnx](https://github.com/intel/mkl-dnn/releases/download/v0.17.2/mklml_lnx_2019.0.1.20181227.tgz)。

> 这三个开关此前是**空操作**，且是两层各自独立地废掉的：
> 头文件无条件 `#define ZQ_CNN_USE_BLAS_GEMM 0` 把命令行的 `-D` 按了回去，
> 而 CMake 又把 `openblas` 链成了 `mklml`、UNIX 分支干脆不链 BLAS。
> 2026-10-03 全部修好，并加了门禁 `tools/check_blas_config.py`（回归里的 C9 组）
> 逐组断言这 8 种配置下宏的**取值**——"编得过"不算数，**值对**才算。

## arm

**32bit**
```shell
mkdir cmake-build-release && cd cmake-build-release
cmake .. -DSIMD_ARCH_TYPE=arm
make SampleMatMulNEON
make SampleMTCNN
make SampleSphereFaceNet
```

**64bit**
```shell
mkdir cmake-build-release && cd cmake-build-release
cmake .. -DSIMD_ARCH_TYPE=arm64
make SampleMatMulNEON
make SampleMTCNN
make SampleSphereFaceNet
```

**use OpenBLAS**

add cmake flag: -DBLAS_TYPE=openblas
（ARM 上随附的那份 `3rdparty/lib/libopenblas.*` 可以直接用。）

**运行时自动选路（先试 ZQ_GEMM，失败再回落 OpenBLAS）**

add cmake flag: `-DBLAS_TYPE=openblas_zq_gemm`

> 只在 `SIMD_ARCH_TYPE=arm/arm64` 上有意义：自动选路的派发点只存在于
> `__ARM_NEON` 分支（`#if __ARM_NEON && ZQ_CNN_USE_ZQ_GEMM && ZQ_CNN_USE_BLAS_GEMM`，
> `ZQCNN/layers_c/` 与 `layers_nchwc/` 各 8 处）。x86 上给这个值会在
> configure 阶段直接报错，而不是像以前那样被**静默忽略**。

