#ifndef CAFFE_C_API_H_
#define CAFFE_C_API_H_

#ifdef _MSC_VER
#ifdef CAFFE_EXPORTS
#define CAFFE_API __declspec(dllexport)
#else
#define CAFFE_API __declspec(dllimport)
#endif
#else
#define CAFFE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef float real_t;

typedef void *BlobHandle;
typedef void *NetHandle;

// Blob API

/*! \brief get blob num */
CAFFE_API int CaffeBlobNum(BlobHandle blob);
/*! \brief get blob channels */
CAFFE_API int CaffeBlobChannels(BlobHandle blob);
/*! \brief get blob height */
CAFFE_API int CaffeBlobHeight(BlobHandle blob);
/*! \brief get blob width */
CAFFE_API int CaffeBlobWidth(BlobHandle blob);
/*! \brief get blob data */
CAFFE_API real_t *CaffeBlobData(BlobHandle blob);
/*! \brief get blob count */
CAFFE_API int CaffeBlobCount(BlobHandle blob);
/*!
 * \brief reshape blob
 * \note  this may change blob data pointer
 */
CAFFE_API int CaffeBlobReshape(BlobHandle blob, int shape_size, int* shape);
/*! \brief get blob shape */
CAFFE_API int CaffeBlobShape(BlobHandle blob, int* shape_size, int** shape);

// Net API

/*!
 * \brief create network
 * \param net_path path to network prototxt file
 * \param model_path path to network caffemodel file
 * \param net output NetHandle
 * \return return code, 0 for success, -1 for failed
 */
CAFFE_API int CaffeNetCreate(const char *net_path,
                             const char *model_path,
                             NetHandle *net);
/*!
 * \brief create network from internal buffer
 * \param net_buffer binary buffer for prototxt
 * \param nb_len net buffer length
 * \param model_buffer binary model buffer for caffemodel
 * \param mb_len model buffer length
 * \param net output handle
 * \param return code, 0 for success, -1 for failed
 */
CAFFE_API int CaffeNetCreateFromBuffer(const char *net_buffer, int nb_len,
                                       const char *model_buffer, int mb_len,
                                       NetHandle *net);
/*! \brief destroy network */
CAFFE_API int CaffeNetDestroy(NetHandle net);
/*!
 * \brief mark internal blob as output
 * \param net net handle
 * \param name blob name
 */
CAFFE_API int CaffeNetMarkOutput(NetHandle net, const char *name);
/*!
 * \brief forward network
 * \note  fill network input blobs before calling this function
 */
CAFFE_API int CaffeNetForward(NetHandle net);
/*!
 * \brief get network internal blob by name
 * \param net NetHandle
 * \param name blob name
 * \param blob BlobHandle
 * \return return code
 */
CAFFE_API int CaffeNetGetBlob(NetHandle net,
                              const char *name,
                              BlobHandle *blob);
/*!
 * \brief list all network internal data buffer
 * \param net net handle
 * \param n number of blobs
 * \param names list of string to blob names
 * \param blobs list of BlobHandle
 */
CAFFE_API int CaffeNetListBlob(NetHandle net,
                               int *n,
                               const char ***names,
                               BlobHandle **blobs);
/*!
 * \brief list all network parameters
 * \param net net handle
 * \param n number of params
 * \param names list of string to param names,
                same layer has same name prefix
 * \param params list of BlobHandle
 */
CAFFE_API int CaffeNetListParam(NetHandle net,
                                int *n,
                                const char ***names,
                                BlobHandle **params);

// Profiler, don't enable Profiler in multi-thread Env

/*!
 * \brief enable profiler
 */
CAFFE_API int CaffeProfilerEnable();
/*!
 * \brief disable profiler
 */
CAFFE_API int CaffeProfilerDisable();
/*!
 * \brief open a scope on profiler
 * \param name scope name
 */
CAFFE_API int CaffeProfilerScopeStart(const char *name);
/*!
 * \brief close a scope
 */
CAFFE_API int CaffeProfilerScopeEnd();
/*!
 * \brief dump profile data to file
 * \param fn file name or path
 */
CAFFE_API int CaffeProfilerDump(const char *fn);

// Helper

/*!
 * \brief gpu avariable
 * \return 1 for gpu available, 0 for not available
 */
CAFFE_API int CaffeGPUAvailable();
/*!
 * \brief set caffe mode
 * \param mode 1 for GPU, 0 for CPU
 * \param device GPU device id, -1 for CPU
 */
CAFFE_API int CaffeSetMode(int mode, int device);
/*!
 * \brief return last API error info
 * \note  this function is thread safe
 */
CAFFE_API const char *CaffeGetLastError();
/*!
 * \brief clear unused memory in threaded memory pool
 */
CAFFE_API int CaffeMemoryPoolClear();

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // CAFFE_C_API_H_
