#!/usr/bin/env bash 
export ANDROID_NDK=
export OPENCV_SDK=
mkdir -p build-android-armv7
pushd build-android-armv7
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DOpenCV_DIR=$OPENCV_SDK/sdk/native/jni/ -DANDROID_ARM_NEON=ON -DSIMD_ARCH_TYPE=arm -DANDROID_PLATFORM=android-24 ..
make -j4 ZQCNN
#make install
popd

