#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <aclnnop/aclnn_rms_norm.h>
#include "acl/acl.h"
#include "npuBackend.hpp"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_matmul.h"
#include "aclnnop/aclnn_sigmoid.h"
#include "aclnnop/aclnn_softmax.h"


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

    struct RopeCacheEntry {
        void* cosAddr{nullptr};
        void* sinAddr{nullptr};
        int pairCount{0};
    };
    std::unordered_map<uint64_t, RopeCacheEntry> ropeCache_;
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
    
    for (auto& item : pImpl->ropeCache_) {
        if (item.second.cosAddr != nullptr) {
            ACL_CHECK(aclrtFree(item.second.cosAddr));
        }
        if (item.second.sinAddr != nullptr) {
            ACL_CHECK(aclrtFree(item.second.sinAddr));
        }
    }
    pImpl->ropeCache_.clear();

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

static aclTensor* CreateTensorFromDeviceWithStrides(void* addr,
                                                    const int64_t* viewShape,
                                                    const int64_t* strides,
                                                    const int64_t* storageShape,
                                                    int dim,
                                                    aclDataType dt) {
    return aclCreateTensor(viewShape, dim,
                           dt, strides, 0,
                           ACL_FORMAT_ND,
                           storageShape, dim,
                           addr);
}

template <typename RunFunc>
static void RunAclnnTwoStage(uint64_t workspaceSize,
                             aclOpExecutor* executor,
                             aclrtStream stream,
                             RunFunc runFunc) {
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    ACL_CHECK(runFunc(workspaceAddr, workspaceSize, executor, stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    if (workspaceAddr != nullptr) {
        ACL_CHECK(aclrtFree(workspaceAddr));
    }
}

static void RunAclnnMulTensor(aclTensor* lhs,
                              aclTensor* rhs,
                              aclTensor* out,
                              aclrtStream stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ACL_CHECK(aclnnMulGetWorkspaceSize(lhs, rhs, out, &workspaceSize, &executor));
    RunAclnnTwoStage(workspaceSize, executor, stream, aclnnMul);
}

static void RunAclnnAddTensor(aclTensor* lhs,
                              aclTensor* rhs,
                              aclScalar* alpha,
                              aclTensor* out,
                              aclrtStream stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ACL_CHECK(aclnnAddGetWorkspaceSize(lhs, rhs, alpha, out, &workspaceSize, &executor));
    RunAclnnTwoStage(workspaceSize, executor, stream, aclnnAdd);
}

static void RunAclnnInplaceMuls(aclTensor* tensor,
                                aclScalar* scalar,
                                aclrtStream stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ACL_CHECK(aclnnInplaceMulsGetWorkspaceSize(tensor, scalar, &workspaceSize, &executor));
    RunAclnnTwoStage(workspaceSize, executor, stream, aclnnInplaceMuls);
}

static void RunAclnnMatmulTensor(aclTensor* lhs,
                                 aclTensor* rhs,
                                 aclTensor* out,
                                 aclrtStream stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    constexpr int8_t cubeMathType = 1;
    ACL_CHECK(aclnnMatmulGetWorkspaceSize(lhs, rhs, out, cubeMathType,
                                          &workspaceSize, &executor));
    RunAclnnTwoStage(workspaceSize, executor, stream, aclnnMatmul);
}

static void RunAclnnSoftmaxTensor(aclTensor* input,
                                  int64_t dim,
                                  aclTensor* out,
                                  aclrtStream stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ACL_CHECK(aclnnSoftmaxGetWorkspaceSize(input, dim, out, &workspaceSize, &executor));
    RunAclnnTwoStage(workspaceSize, executor, stream, aclnnSoftmax);
}

static uint64_t MakeRopeCacheKey(int headSize, int position, int pairCount) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(headSize)) << 40) |
           (static_cast<uint64_t>(static_cast<uint32_t>(position)) << 16) |
           static_cast<uint64_t>(static_cast<uint16_t>(pairCount));
}

static CNPUBackend::Impl::RopeCacheEntry& GetRopeCacheEntry(CNPUBackend::Impl* impl,
                                                            int headSize,
                                                            int position,
                                                            int pairCount) {
    const uint64_t key = MakeRopeCacheKey(headSize, position, pairCount);
    auto it = impl->ropeCache_.find(key);
    if (it != impl->ropeCache_.end()) {
        return it->second;
    }

    CNPUBackend::Impl::RopeCacheEntry entry;
    entry.pairCount = pairCount;

    std::vector<float> cosHost(pairCount);
    std::vector<float> sinHost(pairCount);
    for (int pairIdx = 0; pairIdx < pairCount; ++pairIdx) {
        const int i = pairIdx * 2;
        const int headDim = i % headSize;
        const float freq = 1.0f / powf(10000.0f, headDim / static_cast<float>(headSize));
        const float val = position * freq;
        cosHost[pairIdx] = cosf(val);
        sinHost[pairIdx] = sinf(val);
    }

    const size_t bytes = pairCount * sizeof(float);
    ACL_CHECK(aclrtMalloc(&entry.cosAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&entry.sinAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(entry.cosAddr, bytes, cosHost.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(entry.sinAddr, bytes, sinHost.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    auto inserted = impl->ropeCache_.emplace(key, entry);
    return inserted.first->second;
}

static void ApplyRopeVector(float* vec,
                            int pairCount,
                            void* cosAddr,
                            void* sinAddr,
                            aclrtStream stream) {
    if (pairCount <= 0) {
        return;
    }

    int64_t pairShape[1] = {pairCount};
    int64_t stride2[1] = {2};
    int64_t evenStorageShape[1] = {2 * pairCount};
    int64_t oddStorageShape[1] = {2 * pairCount - 1};

    aclTensor* evenTensor = CreateTensorFromDeviceWithStrides(vec, pairShape, stride2,
                                                              evenStorageShape, 1, ACL_FLOAT);
    aclTensor* oddTensor = CreateTensorFromDeviceWithStrides(vec + 1, pairShape, stride2,
                                                             oddStorageShape, 1, ACL_FLOAT);
    aclTensor* cosTensor = CreateTensorFromDevice(cosAddr, pairShape, 1, ACL_FLOAT);
    aclTensor* sinTensor = CreateTensorFromDevice(sinAddr, pairShape, 1, ACL_FLOAT);

    void* evenCosAddr = nullptr;
    void* oddSinAddr = nullptr;
    void* evenSinAddr = nullptr;
    void* oddCosAddr = nullptr;
    const size_t bytes = pairCount * sizeof(float);
    ACL_CHECK(aclrtMalloc(&evenCosAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&oddSinAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&evenSinAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&oddCosAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));

    aclTensor* evenCosTensor = CreateTensorFromDevice(evenCosAddr, pairShape, 1, ACL_FLOAT);
    aclTensor* oddSinTensor = CreateTensorFromDevice(oddSinAddr, pairShape, 1, ACL_FLOAT);
    aclTensor* evenSinTensor = CreateTensorFromDevice(evenSinAddr, pairShape, 1, ACL_FLOAT);
    aclTensor* oddCosTensor = CreateTensorFromDevice(oddCosAddr, pairShape, 1, ACL_FLOAT);

    RunAclnnMulTensor(evenTensor, cosTensor, evenCosTensor, stream);
    RunAclnnMulTensor(oddTensor, sinTensor, oddSinTensor, stream);
    RunAclnnMulTensor(evenTensor, sinTensor, evenSinTensor, stream);
    RunAclnnMulTensor(oddTensor, cosTensor, oddCosTensor, stream);

    const float minusOneValue = -1.0f;
    const float plusOneValue = 1.0f;
    aclScalar* minusOne = aclCreateScalar(&minusOneValue, ACL_FLOAT);
    aclScalar* plusOne = aclCreateScalar(&plusOneValue, ACL_FLOAT);

    RunAclnnAddTensor(evenCosTensor, oddSinTensor, minusOne, evenTensor, stream);
    RunAclnnAddTensor(evenSinTensor, oddCosTensor, plusOne, oddTensor, stream);

    aclDestroyScalar(minusOne);
    aclDestroyScalar(plusOne);
    aclDestroyTensor(evenTensor);
    aclDestroyTensor(oddTensor);
    aclDestroyTensor(cosTensor);
    aclDestroyTensor(sinTensor);
    aclDestroyTensor(evenCosTensor);
    aclDestroyTensor(oddSinTensor);
    aclDestroyTensor(evenSinTensor);
    aclDestroyTensor(oddCosTensor);
    ACL_CHECK(aclrtFree(evenCosAddr));
    ACL_CHECK(aclrtFree(oddSinAddr));
    ACL_CHECK(aclrtFree(evenSinAddr));
    ACL_CHECK(aclrtFree(oddCosAddr));
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
    ACL_CHECK(aclrtSetCurrentContext(pImpl->context_));

    int64_t xShape[2] = {1, n};
    int64_t wShape[1] = {n};
    int64_t rstdShape[1] = {1};

    aclTensor* xTensor = CreateTensorFromDevice(x, xShape, 2, ACL_FLOAT);
    aclTensor* wTensor = CreateTensorFromDevice(w, wShape, 1, ACL_FLOAT);
    aclTensor* yTensor = CreateTensorFromDevice(y, xShape, 2, ACL_FLOAT);

    void* rstdAddr = nullptr;
    ACL_CHECK(aclrtMalloc(&rstdAddr, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor* rstdTensor = CreateTensorFromDevice(rstdAddr, rstdShape, 1, ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    constexpr double epsilon = 1e-5;
    ACL_CHECK(aclnnRmsNormGetWorkspaceSize(xTensor, wTensor, epsilon, yTensor, rstdTensor,
                                           &workspaceSize, &executor));
    RunAclnnTwoStage(workspaceSize, executor, pImpl->stream_, aclnnRmsNorm);

    aclDestroyTensor(xTensor);
    aclDestroyTensor(wTensor);
    aclDestroyTensor(yTensor);
    aclDestroyTensor(rstdTensor);
    ACL_CHECK(aclrtFree(rstdAddr));
}

/*  TODO
    matmul: 矩阵-向量乘
        o[d][1] = w[d][n] X x[n][1]
*/
void CNPUBackend::matmul(float *o, float *x, float *w, int n, int d) { 
    ACL_CHECK(aclrtSetCurrentContext(pImpl->context_));

    int64_t xShape[2] = {1, n};
    int64_t wViewShape[2] = {n, d};
    int64_t wStrides[2] = {1, n};
    int64_t wStorageShape[2] = {d, n};
    int64_t outShape[2] = {1, d};

    aclTensor* xTensor = CreateTensorFromDevice(x, xShape, 2, ACL_FLOAT);
    aclTensor* wTensor = CreateTensorFromDeviceWithStrides(w, wViewShape, wStrides,
                                                           wStorageShape, 2, ACL_FLOAT);
    aclTensor* outTensor = CreateTensorFromDevice(o, outShape, 2, ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    constexpr int8_t cubeMathType = 1;
    ACL_CHECK(aclnnMatmulGetWorkspaceSize(xTensor, wTensor, outTensor, cubeMathType,
                                          &workspaceSize, &executor));
    RunAclnnTwoStage(workspaceSize, executor, pImpl->stream_, aclnnMatmul);

    aclDestroyTensor(xTensor);
    aclDestroyTensor(wTensor);
    aclDestroyTensor(outTensor);
}




void CNPUBackend::ropeEncoding(float *q, float *k, int headSize, int position, int dim, int kvDim)
{
    ACL_CHECK(aclrtSetCurrentContext(pImpl->context_));

    const int qPairs = dim / 2;
    const int kPairs = kvDim / 2;
    auto& cache = GetRopeCacheEntry(pImpl, headSize, position, qPairs);

    ApplyRopeVector(q, qPairs, cache.cosAddr, cache.sinAddr, pImpl->stream_);
    ApplyRopeVector(k, kPairs, cache.cosAddr, cache.sinAddr, pImpl->stream_);

    ACL_CHECK(aclrtSynchronizeStream(pImpl->stream_));
}



/*
scale ：y[i] = y[i] + x[i] * factor 注意和cpu侧算子的不同
*/
void CNPUBackend::axpy(float* y, float* x, float factor, int dim) {
    ACL_CHECK(aclrtSetCurrentContext(pImpl->context_));

    int64_t shape[1] = {dim};
    aclTensor* yTensor = CreateTensorFromDevice(y, shape, 1, ACL_FLOAT);
    aclTensor* xTensor = CreateTensorFromDevice(x, shape, 1, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&factor, ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ACL_CHECK(aclnnInplaceAddGetWorkspaceSize(yTensor, xTensor, alpha,
                                              &workspaceSize, &executor));
    RunAclnnTwoStage(workspaceSize, executor, pImpl->stream_, aclnnInplaceAdd);

    aclDestroyTensor(yTensor);
    aclDestroyTensor(xTensor);
    aclDestroyScalar(alpha);
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
    ACL_CHECK(aclrtSetCurrentContext(pImpl->context_));

    const int seqLen = pos + 1;
    int64_t qShape[2] = {1, headSize};
    int64_t scoreShape[2] = {1, seqLen};
    int64_t kViewShape[2] = {headSize, seqLen};
    int64_t kStrides[2] = {1, headSize};
    int64_t kStorageShape[2] = {seqLen, headSize};
    int64_t vShape[2] = {seqLen, headSize};
    int64_t outShape[2] = {1, headSize};

    void* rawScoreAddr = nullptr;
    ACL_CHECK(aclrtMalloc(&rawScoreAddr, seqLen * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));

    aclTensor* qTensor = CreateTensorFromDevice(q, qShape, 2, ACL_FLOAT);
    aclTensor* kTensor = CreateTensorFromDeviceWithStrides(kCache, kViewShape, kStrides,
                                                           kStorageShape, 2, ACL_FLOAT);
    aclTensor* rawScoreTensor = CreateTensorFromDevice(rawScoreAddr, scoreShape, 2, ACL_FLOAT);
    aclTensor* scoreTensor = CreateTensorFromDevice(attnScores, scoreShape, 2, ACL_FLOAT);
    aclTensor* vTensor = CreateTensorFromDevice(vCache, vShape, 2, ACL_FLOAT);
    aclTensor* outTensor = CreateTensorFromDevice(out, outShape, 2, ACL_FLOAT);

    RunAclnnMatmulTensor(qTensor, kTensor, rawScoreTensor, pImpl->stream_);

    const float scaleValue = 1.0f / std::sqrt(static_cast<float>(headSize));
    aclScalar* scale = aclCreateScalar(&scaleValue, ACL_FLOAT);
    RunAclnnInplaceMuls(rawScoreTensor, scale, pImpl->stream_);
    aclDestroyScalar(scale);

    RunAclnnSoftmaxTensor(rawScoreTensor, 1, scoreTensor, pImpl->stream_);
    RunAclnnMatmulTensor(scoreTensor, vTensor, outTensor, pImpl->stream_);

    aclDestroyTensor(qTensor);
    aclDestroyTensor(kTensor);
    aclDestroyTensor(rawScoreTensor);
    aclDestroyTensor(scoreTensor);
    aclDestroyTensor(vTensor);
    aclDestroyTensor(outTensor);
    ACL_CHECK(aclrtFree(rawScoreAddr));
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
