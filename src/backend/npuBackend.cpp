#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <aclnnop/aclnn_rms_norm.h>
#include "acl/acl.h"
#include "npuBackend.hpp"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_matmul.h"
#include "aclnnop/aclnn_sigmoid.h"


#define ACL_CHECK(call) {                                   \
    aclError _status = (call);                             \
    if (_status != ACL_ERROR_NONE) {                       \
        std::cerr << "[ACL ERROR] " << #call               \
                  << " failed, code = " << _status         \
                   << ", " << aclGetRecentErrMsg()     \
                  << std::endl;                            \
        fprintf(stderr, "ACL错误 %d 在文件 %s:  第%d行  \n", _status, __FILE__, __LINE__);\
        exit(EXIT_FAILURE);                                \
    }                                                       \
}


#define CHECK_RET(ret, expr) \
    if (ret != ACL_SUCCESS) { \
        return; \
    }


#define LOG_PRINT(message, ...) \
    printf((message), ##__VA_ARGS__); \
    fflush(stdout);

struct CNPUBackend::Impl {
    aclrtContext context_;
    aclrtStream  stream_;
};

CNPUBackend::CNPUBackend() {
    this->type = BackendType::NPU;
    pImpl = new Impl();
    std::cerr << "[DBG] CNPUBackend constructed\n";

    std::cerr << "[DBG] aclInit\n";
    ACL_CHECK(aclInit(nullptr));

    std::cerr << "[DBG] aclrtSetDevice(0)\n";
    ACL_CHECK(aclrtSetDevice(0));

    std::cerr << "[DBG] aclrtCreateContext\n";
    ACL_CHECK(aclrtCreateContext(&(pImpl->context_), 0));

    std::cerr << "[DBG] aclrtSetCurrentContext\n";
    ACL_CHECK(aclrtSetCurrentContext(pImpl->context_));

    std::cerr << "[DBG] aclrtCreateStream\n";
    ACL_CHECK(aclrtCreateStream(&(pImpl->stream_)));
    std::cerr << "[DBG] Created stream: " << pImpl->stream_ << std::endl;

    std::cerr << "[DBG] initCANN done\n";
}


CNPUBackend::~CNPUBackend() {

    std::cerr << "[DBG] CNPUBackend destroyed\n";
    

    std::cerr << "[DBG] aclrtDestroyStream\n";
    ACL_CHECK(aclrtDestroyStream(pImpl->stream_));

    std::cerr << "[DBG] aclrtDestroyContext\n";
    ACL_CHECK(aclrtDestroyContext(pImpl->context_));

    std::cerr << "[DBG] aclrtResetDevice(0)\n";
    ACL_CHECK(aclrtResetDevice(0));

    std::cerr << "[DBG] aclFinalize\n";
    ACL_CHECK(aclFinalize());
    delete pImpl;

}


// 把已有 device 内存 addr 包装成 aclTensor*
static aclTensor* CreateTensorFromDevice(void* addr,
                                         const int64_t* shape,
                                         int dim,
                                         aclDataType dt) {
    std::vector<int64_t> strides(dim, 1);
    for (int i = dim - 2; i >= 0; --i) {
        strides[i] = strides[i+1] * shape[i+1];
    }
    return aclCreateTensor(shape, dim,
                           dt, strides.data(), 0,
                           ACL_FORMAT_ND,
                           shape, dim,
                           addr);
}

//若 executor 已存在就复用 ----
struct Op2Stage {
    aclOpExecutor* exe{nullptr};
    uint64_t       wsBytes{0};
    void*          wsBuf{nullptr};
    void allocWs() {
        if (wsBytes && wsBuf == nullptr)
            ACL_CHECK(aclrtMalloc(&wsBuf, wsBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    ~Op2Stage() { if (wsBuf) aclrtFree(wsBuf); }
};


/*
rmsnorm:归一化
    1.square_sum_mean = (x[0]*x[0]+x[1]*x[1]+....+x[n]*x[n])/n + 1e-5f
    2.y[0] = w[0] * (x[0] / sqrt(square_sum_mean)),
      ......
      y[n-1] = w[n-1] * (x[n-1] / sqrt(square_sum_mean))  

*/
void CNPUBackend::rmsnorm(float *y, float *x, float *w, int n) {
    // TODO ...
}

/*  TODO
    matmul: 矩阵-向量乘
        o[d][1] = w[d][n] X x[n][1]
*/
void CNPUBackend::matmul(float *o, float *x, float *w, int n, int d) { 

    // TODO ....
}




void CNPUBackend::ropeEncoding(float *q, float *k, int headSize, int position, int dim, int kvDim)
{
    std::vector<float> q_host(dim);
    std::vector<float> k_host(kvDim);
    ACL_CHECK(aclrtMemcpy(q_host.data(), dim * sizeof(float), q, dim * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(k_host.data(), kvDim * sizeof(float), k, kvDim * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    for (int i = 0; i < dim; i += 2) {
        int headDim = i % headSize;
        float freq = 1.0f / powf(10000.0f, headDim / (float)headSize);
        float val = position * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kvDim ? 2 : 1;
        for (int v = 0; v < rotn; v++) {
            float *vec = v == 0 ? q_host.data() : k_host.data();
            float v0 = vec[i];
            float v1 = vec[i + 1];
            vec[i]     = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
    ACL_CHECK(aclrtMemcpy(q, dim * sizeof(float), q_host.data(), dim * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(k, kvDim * sizeof(float), k_host.data(), kvDim * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtSynchronizeStream(pImpl->stream_));
}



/*
scale ：y[i] = y[i] + x[i] * factor 注意和cpu侧算子的不同
*/
void CNPUBackend::axpy(float* y, float* x, float factor, int dim) {

    // TODO ....
}

void CNPUBackend::swiGLLUFunc(float* headOutput, float* value, int hiddenDim) {
    ACL_CHECK(aclrtSetCurrentContext(pImpl->context_));

    int64_t shape[1] = {hiddenDim};

    aclTensor* headOutputTensor = CreateTensorFromDevice(headOutput, shape, 1, ACL_FLOAT);
    aclTensor* valueTensor = CreateTensorFromDevice(value, shape, 1, ACL_FLOAT);

    void* sigmoidOutputAddr = nullptr;
    ACL_CHECK(aclrtMalloc(&sigmoidOutputAddr, hiddenDim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor* sigmoidTensor = CreateTensorFromDevice(sigmoidOutputAddr, shape, 1, ACL_FLOAT);

    void* tmpOutputAddr = nullptr;
    ACL_CHECK(aclrtMalloc(&tmpOutputAddr, hiddenDim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor* tmpTensor = CreateTensorFromDevice(tmpOutputAddr, shape, 1, ACL_FLOAT);

    uint64_t sigmoidWsSize = 0;
    aclOpExecutor* sigmoidExecutor = nullptr;
    ACL_CHECK(aclnnSigmoidGetWorkspaceSize(headOutputTensor, sigmoidTensor, &sigmoidWsSize, &sigmoidExecutor));

    void* sigmoidWs = nullptr;
    if (sigmoidWsSize > 0) {
        ACL_CHECK(aclrtMalloc(&sigmoidWs, sigmoidWsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    ACL_CHECK(aclnnSigmoid(sigmoidWs, sigmoidWsSize, sigmoidExecutor, pImpl->stream_));
    ACL_CHECK(aclrtSynchronizeStream(pImpl->stream_));

    if (sigmoidWs) {
        ACL_CHECK(aclrtFree(sigmoidWs));
    }

    uint64_t mul1WsSize = 0;
    aclOpExecutor* mul1Executor = nullptr;
    ACL_CHECK(aclnnMulGetWorkspaceSize(headOutputTensor, sigmoidTensor, tmpTensor, &mul1WsSize, &mul1Executor));

    void* mul1Ws = nullptr;
    if (mul1WsSize > 0) {
        ACL_CHECK(aclrtMalloc(&mul1Ws, mul1WsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    ACL_CHECK(aclnnMul(mul1Ws, mul1WsSize, mul1Executor, pImpl->stream_));
    ACL_CHECK(aclrtSynchronizeStream(pImpl->stream_));

    if (mul1Ws) {
        ACL_CHECK(aclrtFree(mul1Ws));
    }

    uint64_t mul2WsSize = 0;
    aclOpExecutor* mul2Executor = nullptr;
    ACL_CHECK(aclnnMulGetWorkspaceSize(tmpTensor, valueTensor, headOutputTensor, &mul2WsSize, &mul2Executor));

    void* mul2Ws = nullptr;
    if (mul2WsSize > 0) {
        ACL_CHECK(aclrtMalloc(&mul2Ws, mul2WsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    ACL_CHECK(aclnnMul(mul2Ws, mul2WsSize, mul2Executor, pImpl->stream_));
    ACL_CHECK(aclrtSynchronizeStream(pImpl->stream_));

    if (mul2Ws) {
        ACL_CHECK(aclrtFree(mul2Ws));
    }

    aclDestroyTensor(headOutputTensor);
    aclDestroyTensor(valueTensor);
    aclDestroyTensor(sigmoidTensor);
    aclDestroyTensor(tmpTensor);
    ACL_CHECK(aclrtFree(sigmoidOutputAddr));
    ACL_CHECK(aclrtFree(tmpOutputAddr));
}



void CNPUBackend::attentionSingleHead(float* q, float* kCache, float* vCache, float* attnScores, float* out, int pos, int headSize) {
    std::vector<float> q_host(headSize), k_host((pos + 1) * headSize), v_host((pos + 1) * headSize);
    std::vector<float> score_host(pos + 1), out_host(headSize, 0.0f);

    ACL_CHECK(aclrtMemcpy(q_host.data(), headSize * sizeof(float), q, headSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(k_host.data(), (pos + 1) * headSize * sizeof(float), kCache, (pos + 1) * headSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(v_host.data(), (pos + 1) * headSize * sizeof(float), vCache, (pos + 1) * headSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

    for (int t = 0; t <= pos; ++t) {
        float dot = 0.0f;
        for (int i = 0; i < headSize; ++i)
            dot += q_host[i] * k_host[t * headSize + i];
        score_host[t] = dot / std::sqrt((float)headSize);
    }

    float maxVal = *std::max_element(score_host.begin(), score_host.end());
    float sum = 0.0f;
    for (int t = 0; t <= pos; ++t) {
        score_host[t] = std::exp(score_host[t] - maxVal);
        sum += score_host[t];
    }
    for (int t = 0; t <= pos; ++t)
        score_host[t] /= sum;

    for (int t = 0; t <= pos; ++t) {
        float a = score_host[t];
        for (int i = 0; i < headSize; ++i)
            out_host[i] += v_host[t * headSize + i] * a;
    }

    ACL_CHECK(aclrtMemcpy(attnScores, (pos + 1) * sizeof(float), score_host.data(), (pos + 1) * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(out, headSize * sizeof(float), out_host.data(), headSize * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
}

void* CNPUBackend::allocMemory(size_t size) {
    std::cout << "[INFO] Allocating " << size << " bytes on NPU." << std::endl;
    void* ptr = nullptr;
    aclError err = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != ACL_ERROR_NONE) {
        printf("[ERROR] aclrtMalloc failed for %zu bytes: %d\n", size, err);
        return nullptr;
    }
    return ptr;
}

void CNPUBackend::freeMemory(void* ptr) {
    if (ptr == nullptr) return;
    aclError err = aclrtFree(ptr);
    if (err != ACL_ERROR_NONE) {
        printf("[ERROR] aclrtFree failed: %d\n", err);
    }
}

void CNPUBackend::copyMemory(void* dst, const void* src, size_t size,
                             bool dstOnDevice, bool srcOnDevice) {
    aclrtMemcpyKind kind;
    if (dstOnDevice && srcOnDevice)
        kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
    else if (dstOnDevice && !srcOnDevice)
        kind = ACL_MEMCPY_HOST_TO_DEVICE;
    else if (!dstOnDevice && srcOnDevice)
        kind = ACL_MEMCPY_DEVICE_TO_HOST;
    else
        kind = ACL_MEMCPY_HOST_TO_HOST;

    aclError err = aclrtMemcpy(dst, size, src, size, kind);
    if (err != ACL_ERROR_NONE) {
        printf("[ERROR] aclrtMemcpy failed: %d (size=%zu, kind=%d)\n", err, size, kind);
    }
}

void CNPUBackend::setMemory(void* ptr, int value, size_t size, bool onDevice) {
    if (onDevice) {
        aclError err = aclrtMemset(ptr, size, value, size);
        if (err != ACL_ERROR_NONE) {
            printf("[ERROR] aclrtMemset failed: %d\n", err);
        }
    } else {
        std::memset(ptr, value, size);
    }
}

void CNPUBackend::sync() {
    aclError err = aclrtSynchronizeDevice();
    if (err != ACL_ERROR_NONE) {
        printf("[ERROR] aclrtSynchronizeDevice failed: %d\n", err);
    }
}