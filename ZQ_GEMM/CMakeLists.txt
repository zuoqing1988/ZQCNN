
include_directories(${ZQCNN_INCLUDE_DIRS})

file(GLOB ZQ_GEMM_SRC ${CMAKE_CURRENT_LIST_DIR}/math/*.c)

# MSVC 的 x64 目标不支持函数体内联汇编 (__asm {} 会报 C4235), Windows 下
# 手写汇编内核改由独立的 MASM 文件提供 (ml64), 指令序列与 GCC 侧的内联
# __asm__ 版一一对应。详见 zq_gemm_32f_align_c_asm_msvc.asm。
if(WIN32)
    enable_language(ASM_MASM)
    list(APPEND ZQ_GEMM_SRC ${CMAKE_CURRENT_LIST_DIR}/math/zq_gemm_32f_align_c_asm_msvc.asm)
endif()

add_library(ZQ_GEMM ${ZQ_GEMM_SRC})
