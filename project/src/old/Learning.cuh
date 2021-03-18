#pragma once
#include <cuda_runtime.h>
#include <cudaDefs.h>
//#include <cudnn.h>
#include <tensorflow/c/c_api.h>/*
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/lib/io/path.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/public/session.h>*/
//#include <torch/library.h>
#include "Settings.h"
#include <iostream>
#include <cassert>

#define BATCH_SIZE ITEM_SIZE*2+5

__global__ void prepareBatch(
    unsigned char* __restrict__ lines,
    const unsigned int  lines_lenght,
    RESULT_TYPE* __restrict__ results,
    unsigned char* inputs,
    unsigned char* labels);


static TF_Buffer* read_tf_buffer_from_file(const char* file);

/**
 * A Wrapper for the C API status object.
 */
class CStatus {
public:
    TF_Status* ptr;
    CStatus() {
        ptr = TF_NewStatus();
    }

    /**
     * Dump the current error message.
     */
    void dump_error()const {
        std::cerr << "TF status error: " << TF_Message(ptr) << std::endl;
    }

    /**
     * Return a boolean indicating whether there was a failure condition.
     * @return
     */
    inline bool failure()const {
        return TF_GetCode(ptr) != TF_OK;
    }

    ~CStatus() {
        if (ptr)TF_DeleteStatus(ptr);
    }
};

namespace detail {
    template<class T>
    class TFObjDeallocator;

    template<>
    struct TFObjDeallocator<TF_Status> { static void run(TF_Status* obj) { TF_DeleteStatus(obj); } };

    template<>
    struct TFObjDeallocator<TF_Graph> { static void run(TF_Graph* obj) { TF_DeleteGraph(obj); } };

    template<>
    struct TFObjDeallocator<TF_Tensor> { static void run(TF_Tensor* obj) { TF_DeleteTensor(obj); } };

    template<>
    struct TFObjDeallocator<TF_SessionOptions> { static void run(TF_SessionOptions* obj) { TF_DeleteSessionOptions(obj); } };

    template<>
    struct TFObjDeallocator<TF_Buffer> { static void run(TF_Buffer* obj) { TF_DeleteBuffer(obj); } };

    template<>
    struct TFObjDeallocator<TF_ImportGraphDefOptions> {
        static void run(TF_ImportGraphDefOptions* obj) { TF_DeleteImportGraphDefOptions(obj); }
    };

    template<>
    struct TFObjDeallocator<TF_Session> {
        static void run(TF_Session* obj) {
            CStatus status;
            TF_DeleteSession(obj, status.ptr);
            if (status.failure()) {
                status.dump_error();
            }
        }
    };
}

template<class T> struct TFObjDeleter {
    void operator()(T* ptr) const {
        detail::TFObjDeallocator<T>::run(ptr);
    }
};

template<class T> struct TFObjMeta {
    typedef std::unique_ptr<T, TFObjDeleter<T>> UniquePtr;
};

template<class T> typename TFObjMeta<T>::UniquePtr tf_obj_unique_ptr(T* obj) {
    typename TFObjMeta<T>::UniquePtr ptr(obj);
    return ptr;
}

class MySession {
public:
    typename TFObjMeta<TF_Graph>::UniquePtr graph;
    typename TFObjMeta<TF_Session>::UniquePtr session;

    TF_Output inputs, outputs;
};

/**
 * Load a GraphDef from a provided file.
 * @param filename: The file containing the protobuf encoded GraphDef
 * @param input_name: The name of the input placeholder
 * @param output_name: The name of the output tensor
 * @return
 */
MySession* my_model_load(const char* filename, const char* input_name, const char* output_name);

/**
 * Deallocator for TF_NewTensor data.
 * @tparam T
 * @param data
 * @param length
 * @param arg
 */
template<class T> static void cpp_void_array_deallocator(void* data, size_t length, void* arg) {
}

/**
 * Deallocator for TF_NewTensor data.
 * @tparam T
 * @param data
 * @param length
 */
template<class T> static void cpp_void_deallocator(void* data, size_t length) {
    
}

/**
 * Read the entire content of a file and return it as a TF_Buffer.
 * @param file: The file to be loaded.
 * @return
 */
static TF_Buffer* read_tf_buffer_from_file(const char* file) {
    std::ifstream t(file, std::ifstream::binary);
    std::string str((std::istreambuf_iterator<char>(t)),
                     std::istreambuf_iterator<char>());

    TF_Buffer* buf = TF_NewBuffer();
    buf->data = str.c_str();
    buf->length = str.size();
    buf->data_deallocator = cpp_void_deallocator<char>;

    return buf;
}

#define MY_TENSOR_SHAPE_MAX_DIM 16
struct TensorShape {
    int64_t values[MY_TENSOR_SHAPE_MAX_DIM];
    int dim;

    int64_t size()const {
        assert(dim >= 0);
        int64_t v = 1;
        for (int i = 0; i < dim; i++)v *= values[i];
        return v;
    }
};