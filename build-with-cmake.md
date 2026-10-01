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

If you are using 3rdparty blas libraries, please download [mklmk_lnx](https://github.com/intel/mkl-dnn/releases/download/v0.17.2/mklml_lnx_2019.0.1.20181227.tgz) or [openblas](https://www.openblas.net/) to `3rdparty/lib`. Then run as following:

```shell
mkdir cmake-build-release && cd cmake-build-release
cmake .. 
make -j4
```

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


