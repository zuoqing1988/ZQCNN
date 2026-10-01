if(WIN32)
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

  find_package(OpenCV QUIET)
  if(NOT OpenCV_FOUND)
    # Fallback to the bundled Windows binaries
    if(WIN32 AND EXISTS ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv/build/x64/vc14/lib/opencv_world342.lib)
      include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv/build/include)
      set(OpenCV_LIBS ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv/build/x64/vc14/lib/opencv_world342.lib)
      set(OpenCV_FOUND TRUE)
      message(STATUS "OpenCV not found by find_package, using bundled 3rdparty/opencv")
    endif()
  endif()
  if(NOT OpenCV_FOUND)
    message(WARNING "OpenCV not found: skipping SamplesZQlibFaceID (core ZQCNN library is unaffected)")
    return()
  endif()
  
  find_package(OpenMP REQUIRED)
  if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  endif()

  include_directories(${ZQCNN_INCLUDE_DIRS})
  include_directories(${CMAKE_CURRENT_LIST_DIR}/../ZQlibFaceID
                    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/include/ZQlib
                    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/include/mini-caffe
                    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/include/libfacedetection
                    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/include/SeetaFaceEngine/FaceIdentification/include)
  link_directories(${ZQCNN_LIBRARY_DIR})

  set(thirdparty_libs caffe libfacedetect-x64 SeetaFaceEngine_FaceIdentification ncnn)

  SUBDIRLIST(SAMPLE_SUBDIRS ${CMAKE_CURRENT_LIST_DIR})
  foreach(SAMPLE_SUBDIR ${SAMPLE_SUBDIRS})
    message(STATUS "add sample ${SAMPLE_SUBDIR}")
    file(GLOB sample_src ${CMAKE_CURRENT_LIST_DIR}/${SAMPLE_SUBDIR}/*.cpp
                        ${CMAKE_CURRENT_LIST_DIR}/${SAMPLE_SUBDIR}/*.c)
    add_executable(${SAMPLE_SUBDIR} ${sample_src})
    target_link_libraries(${SAMPLE_SUBDIR} ZQCNN ${OpenCV_LIBS} ${thirdparty_libs})
  endforeach()

else() #NOT WIN32

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

  find_package(OpenCV QUIET)
  if(NOT OpenCV_FOUND)
    # Fallback to the bundled Windows binaries
    if(WIN32 AND EXISTS ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv/build/x64/vc14/lib/opencv_world342.lib)
      include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv/build/include)
      set(OpenCV_LIBS ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv/build/x64/vc14/lib/opencv_world342.lib)
      set(OpenCV_FOUND TRUE)
      message(STATUS "OpenCV not found by find_package, using bundled 3rdparty/opencv")
    endif()
  endif()
  if(NOT OpenCV_FOUND)
    message(WARNING "OpenCV not found: skipping SamplesZQlibFaceID (core ZQCNN library is unaffected)")
    return()
  endif()
  
  find_package(OpenMP REQUIRED)
  if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  endif()
  
  include_directories(${ZQCNN_INCLUDE_DIRS})
  include_directories(${CMAKE_CURRENT_LIST_DIR}/../ZQlibFaceID
                    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/include
                    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/include/ZQlib
                    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/include/mini-caffe
                    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/include/libfacedetection)
  link_directories(${ZQCNN_LIBRARY_DIR})

  #set(thirdparty_libs caffe libfacedetect-x64)
  set(thirdparty_libs ncnn pthread)

  SUBDIRLIST(SAMPLE_SUBDIRS ${CMAKE_CURRENT_LIST_DIR})
  # Linux 下没有 mini-caffe / SeetaFace / libfacedetection 的预编译库，跳过依赖它们的示例
  list(FILTER SAMPLE_SUBDIRS EXCLUDE REGEX "MiniCaffe|SeetaFace|LibFaceDetect|CropImagesFor(ArcFace|SphereFace)Fast")
  foreach(SAMPLE_SUBDIR ${SAMPLE_SUBDIRS})
    message(STATUS "add sample ${SAMPLE_SUBDIR}")
    file(GLOB sample_src ${CMAKE_CURRENT_LIST_DIR}/${SAMPLE_SUBDIR}/*.cpp
                        ${CMAKE_CURRENT_LIST_DIR}/${SAMPLE_SUBDIR}/*.c)
    add_executable(${SAMPLE_SUBDIR} ${sample_src})
    target_link_libraries(${SAMPLE_SUBDIR} ZQCNN ${OpenCV_LIBS} ${thirdparty_libs})
  endforeach()
endif()
