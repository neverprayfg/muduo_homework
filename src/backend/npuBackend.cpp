#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <aclnnop/aclnn_rms_norm.h>
#include "acl/acl.h"
#include "npuBackend.hpp"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_matmul.h"
#include "aclnnop/aclnn_sigmoid.h"
#include "aclnnop/aclnn_softmax.h"
#include "aclnnop/aclnn_rotary_position_embedding.h"


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

    struct ReusableBuffer {
        void* addr{nullptr};
        size_t bytes{0};
    };

    struct RopeCacheEntry {
        void* cosAddr{nullptr};
        void* sinAddr{nullptr};
        int elemCount{0};
    };

    ReusableBuffer workspace_;
    std::vector<ReusableBuffer> tempBuffers_;
    std::unordered_map<int, ReusableBuffer> attentionScaleCache_;
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

    if (pImpl->workspace_.addr != nullptr) {
        ACL_CHECK(aclrtFree(pImpl->workspace_.addr));
    }
    for (auto& buffer : pImpl->tempBuffers_) {
        if (buffer.addr != nullptr) {
            ACL_CHECK(aclrtFree(buffer.addr));
        }
    }
    pImpl->tempBuffers_.clear();
    for (auto& item : pImpl->attentionScaleCache_) {
        if (item.second.addr != nullptr) {
            ACL_CHECK(aclrtFree(item.second.addr));
        }
    }
    pImpl->attentionScaleCache_.clear();

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

static void* GetWorkspace(CNPUBackend::Impl* impl, size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    if (impl->workspace_.bytes < bytes) {
        if (impl->workspace_.addr != nullptr) {
            ACL_CHECK(aclrtFree(impl->workspace_.addr));
        }
        ACL_CHECK(aclrtMalloc(&impl->workspace_.addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        impl->workspace_.bytes = bytes;
    }
    return impl->workspace_.addr;
}

static void* GetTempBuffer(CNPUBackend::Impl* impl, size_t slot, size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    if (impl->tempBuffers_.size() <= slot) {
        impl->tempBuffers_.resize(slot + 1);
    }

    auto& buffer = impl->tempBuffers_[slot];
    if (buffer.bytes < bytes) {
        if (buffer.addr != nullptr) {
            ACL_CHECK(aclrtFree(buffer.addr));
        }
        ACL_CHECK(aclrtMalloc(&buffer.addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        buffer.bytes = bytes;
    }
    return buffer.addr;
}

static void* GetAttentionScale(CNPUBackend::Impl* impl, int headSize) {
    auto it = impl->attentionScaleCache_.find(headSize);
    if (it != impl->attentionScaleCache_.end()) {
        return it->second.addr;
    }

    CNPUBackend::Impl::ReusableBuffer buffer;
    buffer.bytes = sizeof(float);
    const float scaleValue = 1.0f / std::sqrt(static_cast<float>(headSize));
    ACL_CHECK(aclrtMalloc(&buffer.addr, buffer.bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(buffer.addr, buffer.bytes, &scaleValue, buffer.bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    auto inserted = impl->attentionScaleCache_.emplace(headSize, buffer);
    return inserted.first->second.addr;
}

template <typename RunFunc>
static void RunAclnnTwoStage(CNPUBackend::Impl* impl,
                             uint64_t workspaceSize,
                             aclOpExecutor* executor,
                             aclrtStream stream,
                             RunFunc runFunc) {
    void* workspaceAddr = GetWorkspace(impl, workspaceSize);

    ACL_CHECK(runFunc(workspaceAddr, workspaceSize, executor, stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
}

static void RunAclnnMulTensor(aclTensor* lhs,
                              aclTensor* rhs,
                              aclTensor* out,
                              CNPUBackend::Impl* impl,
                              aclrtStream stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ACL_CHECK(aclnnMulGetWorkspaceSize(lhs, rhs, out, &workspaceSize, &executor));
    RunAclnnTwoStage(impl, workspaceSize, executor, stream, aclnnMul);
}

static void RunAclnnSigmoidTensor(aclTensor* input,
                                  aclTensor* out,
                                  CNPUBackend::Impl* impl,
                                  aclrtStream stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ACL_CHECK(aclnnSigmoidGetWorkspaceSize(input, out, &workspaceSize, &executor));
    RunAclnnTwoStage(impl, workspaceSize, executor, stream, aclnnSigmoid);
}

static void RunAclnnMatmulTensor(aclTensor* lhs,
                                 aclTensor* rhs,
                                 aclTensor* out,
                                 CNPUBackend::Impl* impl,
                                 aclrtStream stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    constexpr int8_t cubeMathType = 1;
    ACL_CHECK(aclnnMatmulGetWorkspaceSize(lhs, rhs, out, cubeMathType,
                                          &workspaceSize, &executor));
    RunAclnnTwoStage(impl, workspaceSize, executor, stream, aclnnMatmul);
}

static void RunAclnnSoftmaxTensor(aclTensor* input,
                                  int64_t dim,
                                  aclTensor* out,
                                  CNPUBackend::Impl* impl,
                                  aclrtStream stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ACL_CHECK(aclnnSoftmaxGetWorkspaceSize(input, dim, out, &workspaceSize, &executor));
    RunAclnnTwoStage(impl, workspaceSize, executor, stream, aclnnSoftmax);
}

static void RunAclnnRotaryPositionEmbedding(aclTensor* x,
                                            aclTensor* cos,
                                            aclTensor* sin,
                                            aclTensor* out,
                                            CNPUBackend::Impl* impl,
                                            aclrtStream stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    constexpr int64_t mode = 1; // interleave
    ACL_CHECK(aclnnRotaryPositionEmbeddingGetWorkspaceSize(x, cos, sin, mode, out,
                                                           &workspaceSize, &executor));
    RunAclnnTwoStage(impl, workspaceSize, executor, stream, aclnnRotaryPositionEmbedding);
}

static uint64_t MakeRopeCacheKey(int headSize, int position, int elemCount) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(headSize)) << 40) |
           (static_cast<uint64_t>(static_cast<uint32_t>(position)) << 16) |
           static_cast<uint64_t>(static_cast<uint16_t>(elemCount));
}

static CNPUBackend::Impl::RopeCacheEntry& GetRopeCacheEntry(CNPUBackend::Impl* impl,
                                                            int headSize,
                                                            int position,
                                                            int elemCount) {
    const uint64_t key = MakeRopeCacheKey(headSize, position, elemCount);
    auto it = impl->ropeCache_.find(key);
    if (it != impl->ropeCache_.end()) {
        return it->second;
    }

    CNPUBackend::Impl::RopeCacheEntry entry;
    entry.elemCount = elemCount;

    std::vector<float> cosHost(elemCount);
    std::vector<float> sinHost(elemCount);
    for (int i = 0; i < elemCount; i += 2) {
        const int headDim = i % headSize;
        const float freq = 1.0f / powf(10000.0f, headDim / static_cast<float>(headSize));
        const float val = position * freq;
        cosHost[i] = cosf(val);
        sinHost[i] = sinf(val);
        if (i + 1 < elemCount) {
            cosHost[i + 1] = cosHost[i];
            sinHost[i + 1] = sinHost[i];
        }
    }

    const size_t bytes = elemCount * sizeof(float);
    ACL_CHECK(aclrtMalloc(&entry.cosAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&entry.sinAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(entry.cosAddr, bytes, cosHost.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(entry.sinAddr, bytes, sinHost.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    auto inserted = impl->ropeCache_.emplace(key, entry);
    return inserted.first->second;
}

static void ApplyRopeVector(float* vec,
                            int elemCount,
                            int headSize,
                            void* cosAddr,
                            void* sinAddr,
                            CNPUBackend::Impl* impl,
                            aclrtStream stream) {
    if (elemCount <= 0 || headSize <= 0) {
        return;
    }

    const int heads = elemCount / headSize;
    int64_t xShape[4] = {1, heads, 1, headSize};
    int64_t freqShape[4] = {1, 1, 1, headSize};
    aclTensor* xTensor = CreateTensorFromDevice(vec, xShape, 4, ACL_FLOAT);
    aclTensor* cosTensor = CreateTensorFromDevice(cosAddr, freqShape, 4, ACL_FLOAT);
    aclTensor* sinTensor = CreateTensorFromDevice(sinAddr, freqShape, 4, ACL_FLOAT);

    const size_t bytes = elemCount * sizeof(float);
    void* outAddr = GetTempBuffer(impl, 1, bytes);
    aclTensor* outTensor = CreateTensorFromDevice(outAddr, xShape, 4, ACL_FLOAT);

    RunAclnnRotaryPositionEmbedding(xTensor, cosTensor, sinTensor, outTensor, impl, stream);
    ACL_CHECK(aclrtMemcpy(vec, bytes, outAddr, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE));

    aclDestroyTensor(xTensor);
    aclDestroyTensor(cosTensor);
    aclDestroyTensor(sinTensor);
    aclDestroyTensor(outTensor);
}

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

    void* rstdAddr = GetTempBuffer(pImpl, 0, sizeof(float));
    aclTensor* rstdTensor = CreateTensorFromDevice(rstdAddr, rstdShape, 1, ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    constexpr double epsilon = 1e-5;
    ACL_CHECK(aclnnRmsNormGetWorkspaceSize(xTensor, wTensor, epsilon, yTensor, rstdTensor,
                                           &workspaceSize, &executor));
    RunAclnnTwoStage(pImpl, workspaceSize, executor, pImpl->stream_, aclnnRmsNorm);

    aclDestroyTensor(xTensor);
    aclDestroyTensor(wTensor);
    aclDestroyTensor(yTensor);
    aclDestroyTensor(rstdTensor);
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
    RunAclnnTwoStage(pImpl, workspaceSize, executor, pImpl->stream_, aclnnMatmul);

    aclDestroyTensor(xTensor);
    aclDestroyTensor(wTensor);
    aclDestroyTensor(outTensor);
}




void CNPUBackend::ropeEncoding(float *q, float *k, int headSize, int position, int dim, int kvDim)
{
    ACL_CHECK(aclrtSetCurrentContext(pImpl->context_));

    auto& cache = GetRopeCacheEntry(pImpl, headSize, position, headSize);

    ApplyRopeVector(q, dim, headSize, cache.cosAddr, cache.sinAddr, pImpl, pImpl->stream_);
    ApplyRopeVector(k, kvDim, headSize, cache.cosAddr, cache.sinAddr, pImpl, pImpl->stream_);

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
    RunAclnnTwoStage(pImpl, workspaceSize, executor, pImpl->stream_, aclnnInplaceAdd);

    aclDestroyTensor(yTensor);
    aclDestroyTensor(xTensor);
    aclDestroyScalar(alpha);
}

void CNPUBackend::swiGLLUFunc(float* headOutput, float* value, int hiddenDim) {
    ACL_CHECK(aclrtSetCurrentContext(pImpl->context_));

    int64_t shape[1] = {hiddenDim};

    aclTensor* headOutputTensor = CreateTensorFromDevice(headOutput, shape, 1, ACL_FLOAT);
    aclTensor* valueTensor = CreateTensorFromDevice(value, shape, 1, ACL_FLOAT);

    void* sigmoidOutputAddr = GetTempBuffer(pImpl, 2, hiddenDim * sizeof(float));
    aclTensor* sigmoidTensor = CreateTensorFromDevice(sigmoidOutputAddr, shape, 1, ACL_FLOAT);

    void* tmpOutputAddr = GetTempBuffer(pImpl, 3, hiddenDim * sizeof(float));
    aclTensor* tmpTensor = CreateTensorFromDevice(tmpOutputAddr, shape, 1, ACL_FLOAT);

    RunAclnnSigmoidTensor(headOutputTensor, sigmoidTensor, pImpl, pImpl->stream_);
    RunAclnnMulTensor(headOutputTensor, sigmoidTensor, tmpTensor, pImpl, pImpl->stream_);
    RunAclnnMulTensor(tmpTensor, valueTensor, headOutputTensor, pImpl, pImpl->stream_);

    aclDestroyTensor(headOutputTensor);
    aclDestroyTensor(valueTensor);
    aclDestroyTensor(sigmoidTensor);
    aclDestroyTensor(tmpTensor);
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

    void* rawScoreAddr = GetTempBuffer(pImpl, 4, seqLen * sizeof(float));
    void* scaledScoreAddr = GetTempBuffer(pImpl, 5, seqLen * sizeof(float));
    void* scaleAddr = GetAttentionScale(pImpl, headSize);

    aclTensor* qTensor = CreateTensorFromDevice(q, qShape, 2, ACL_FLOAT);
    aclTensor* kTensor = CreateTensorFromDeviceWithStrides(kCache, kViewShape, kStrides,
                                                           kStorageShape, 2, ACL_FLOAT);
    aclTensor* rawScoreTensor = CreateTensorFromDevice(rawScoreAddr, scoreShape, 2, ACL_FLOAT);
    aclTensor* scaledScoreTensor = CreateTensorFromDevice(scaledScoreAddr, scoreShape, 2, ACL_FLOAT);
    int64_t scaleShape[2] = {1, 1};
    aclTensor* scaleTensor = CreateTensorFromDevice(scaleAddr, scaleShape, 2, ACL_FLOAT);
    aclTensor* scoreTensor = CreateTensorFromDevice(attnScores, scoreShape, 2, ACL_FLOAT);
    aclTensor* vTensor = CreateTensorFromDevice(vCache, vShape, 2, ACL_FLOAT);
    aclTensor* outTensor = CreateTensorFromDevice(out, outShape, 2, ACL_FLOAT);

    RunAclnnMatmulTensor(qTensor, kTensor, rawScoreTensor, pImpl, pImpl->stream_);
    RunAclnnMulTensor(rawScoreTensor, scaleTensor, scaledScoreTensor, pImpl, pImpl->stream_);
    RunAclnnSoftmaxTensor(scaledScoreTensor, 1, scoreTensor, pImpl, pImpl->stream_);
    RunAclnnMatmulTensor(scoreTensor, vTensor, outTensor, pImpl, pImpl->stream_);

    aclDestroyTensor(qTensor);
    aclDestroyTensor(kTensor);
    aclDestroyTensor(rawScoreTensor);
    aclDestroyTensor(scaledScoreTensor);
    aclDestroyTensor(scaleTensor);
    aclDestroyTensor(scoreTensor);
    aclDestroyTensor(vTensor);
    aclDestroyTensor(outTensor);
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
