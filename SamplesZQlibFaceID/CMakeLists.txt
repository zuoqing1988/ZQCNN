MACRO(SUBDIRLIST result curdir)
FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
SET(dirlist "")
FOREACH(child ${children})
  IF(IS_DIRECTORY ${curdir}/${child})
    LIST(APPEND dirlist ${child})
  ENDIF()
ENDFOREACH()
SET(${result} ${dirlist})
ENDMACRO()

find_package(OpenCV REQUIRED)
if(NOT OPENCV_FOUND)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv/build/include)
    set(OpenCV_LIBS ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv/build/x64/vc14/lib/opencv_world342.lib)
endif()
include_directories(${ZQCNN_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_LIST_DIR}/../ZQlibFaceID
                    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/include/ZQlib
                    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/include/mini-caffe
                    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/include/libfacedetection)
link_directories(${ZQCNN_LIBRARY_DIR})

set(thirdparty_libs caffe libfacedetect-x64)

SUBDIRLIST(SAMPLE_SUBDIRS ${CMAKE_CURRENT_LIST_DIR})
foreach(SAMPLE_SUBDIR ${SAMPLE_SUBDIRS})
    message(STATUS "add sample ${SAMPLE_SUBDIR}")
    file(GLOB sample_src ${CMAKE_CURRENT_LIST_DIR}/${SAMPLE_SUBDIR}/*.cpp
                        ${CMAKE_CURRENT_LIST_DIR}/${SAMPLE_SUBDIR}/*.c)
    add_executable(${SAMPLE_SUBDIR} ${sample_src})
    target_link_libraries(${SAMPLE_SUBDIR} ZQCNN ${OpenCV_LIBS} ${thirdparty_libs})
endforeach()