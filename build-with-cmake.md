# build with cmake

## windows

Take Vsiual Studio 2015 as example:

```shell
mkdir build_x64 && cd build_x64
cmake .. -G"Visual Studio 14 Win64"
```

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


