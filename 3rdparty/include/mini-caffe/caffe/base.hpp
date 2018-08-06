#ifndef CAFFE_BASE_HPP_
#define CAFFE_BASE_HPP_

#include <string>
#include <vector>
#include <memory>

#include <caffe/logging.hpp>

#ifdef _MSC_VER
#ifdef CAFFE_EXPORTS
#define CAFFE_API __declspec(dllexport)
#else
#define CAFFE_API __declspec(dllimport)
#endif
#else
#define CAFFE_API
#endif

#ifdef _MSC_VER
#pragma warning(disable:4251)
#endif

// Convert macro to string
#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname)            \
private:                                              \
  classname(const classname&) = delete;               \
  classname(classname&&) = delete;                    \
  classname& operator=(const classname&) = delete;    \
  classname& operator=(classname&&) = delete

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"
#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."

namespace caffe {

// Common functions and classes from std that caffe often uses.
using std::vector;
using std::string;
using std::shared_ptr;

typedef float real_t;

enum DeviceMode {
  CPU, GPU
};

/*!
 * \brief gpu avariable
 * \return true if gpu available
 */
CAFFE_API bool GPUAvailable();
/*!
 * \brief set caffe mode
 * \param mode GPU or CPU
 * \param device GPU device id, -1 for CPU
 */
CAFFE_API void SetMode(DeviceMode mode, int device);

//// ThreadLocal Memory Pool API

struct MemPoolState {
  int gpu_mem;  // gpu memory, calculate on all device memory used by this thread
  int cpu_mem;  // cpu memory
  int unused_gpu_mem;  // not used gpu memory
  int unused_cpu_mem;  // not used cpu memory
};
/*! \brief get memory usage in current thread */
CAFFE_API MemPoolState MemPoolGetState();
/*! \brief clear unused memory pool in current thread */
CAFFE_API void MemPoolClear();

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
