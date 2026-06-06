#include "npuRunstate.hpp"
#include "acl/acl.h"

CNPURunState::CNPURunState() {
}

CNPURunState::~CNPURunState() {
    deallocateMemory();
}

void CNPURunState::allocateMemory(CModelConfig* config) {
    int dim = config->dim;
    int kvDim = (config->dim * config->numKvHeads) / config->numHeads; // 简化写法
    int ffDim = config->feedForwardDim;
    int seqLen = config->maxSeqLen;
    int layers = config->numLayers;
    int vocab = config->vocabSize;
    int numHeads = config->numHeads;

    size_t totalMemoryAllocated = 0;  // 用来追踪总分配的内存大小

    // 分配 NPU 显存，使用 aclrtMalloc
    aclError ret = aclrtMalloc((void**)&currentActivation, dim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR:] NPU memory allocation failed for currentActivation!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "[DEBUG] currentActivation allocated at address: " << currentActivation << std::endl;
    totalMemoryAllocated += dim * sizeof(float);
    std::cout << "[DEBUG] Allocated memory for currentActivation: " << dim * sizeof(float) << " bytes" << std::endl;

    ret = aclrtMalloc((void**)&branchActivation, dim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR:] NPU memory allocation failed for branchActivation!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "[DEBUG] branchActivation allocated at address: " << branchActivation << std::endl;
    totalMemoryAllocated += dim * sizeof(float);
    std::cout << "[DEBUG] Allocated memory for branchActivation: " << dim * sizeof(float) << " bytes" << std::endl;

    ret = aclrtMalloc((void**)&extraBuffer, dim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR:] NPU memory allocation failed for extraBuffer!" << std::endl;
        exit(EXIT_FAILURE);
    }
    totalMemoryAllocated += dim * sizeof(float);
    std::cout << "[DEBUG] Allocated memory for extraBuffer: " << dim * sizeof(float) << " bytes" << std::endl;

    ret = aclrtMalloc((void**)&hiddenBuffer, ffDim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR:] NPU memory allocation failed for hiddenBuffer!" << std::endl;
        exit(EXIT_FAILURE);
    }
    totalMemoryAllocated += ffDim * sizeof(float);
    std::cout << "[DEBUG] Allocated memory for hiddenBuffer: " << ffDim * sizeof(float) << " bytes" << std::endl;

    ret = aclrtMalloc((void**)&extraHiddenBuffer, ffDim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR:] NPU memory allocation failed for extraHiddenBuffer!" << std::endl;
        exit(EXIT_FAILURE);
    }
    totalMemoryAllocated += ffDim * sizeof(float);
    std::cout << "[DEBUG] Allocated memory for extraHiddenBuffer: " << ffDim * sizeof(float) << " bytes" << std::endl;

    ret = aclrtMalloc((void**)&q, dim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR:] NPU memory allocation failed for q!" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cerr << "[ALLOC] state->q alloc at " << q << std::endl;
    totalMemoryAllocated += dim * sizeof(float);
    std::cout << "[DEBUG] Allocated memory for q: " << dim * sizeof(float) << " bytes" << std::endl;

    ret = aclrtMalloc((void**)&k, kvDim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR:] NPU memory allocation failed for k!" << std::endl;
        exit(EXIT_FAILURE);
    }
    totalMemoryAllocated += kvDim * sizeof(float);
    std::cout << "[DEBUG] Allocated memory for k: " << kvDim * sizeof(float) << " bytes" << std::endl;

    ret = aclrtMalloc((void**)&v, kvDim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR:] NPU memory allocation failed for v!" << std::endl;
        exit(EXIT_FAILURE);
    }
    totalMemoryAllocated += kvDim * sizeof(float);
    std::cout << "[DEBUG] Allocated memory for v: " << kvDim * sizeof(float) << " bytes" << std::endl;

    ret = aclrtMalloc((void**)&attentionScores, numHeads * seqLen * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR:] NPU memory allocation failed for attentionScores!" << std::endl;
        exit(EXIT_FAILURE);
    }
    totalMemoryAllocated += numHeads * seqLen * sizeof(float);
    std::cout << "[DEBUG] Allocated memory for attentionScores: " << numHeads * seqLen * sizeof(float) << " bytes" << std::endl;

    ret = aclrtMalloc((void**)&keyCache, layers * seqLen * kvDim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR:] NPU memory allocation failed for keyCache!" << std::endl;
        exit(EXIT_FAILURE);
    }
    totalMemoryAllocated += layers * seqLen * kvDim * sizeof(float);
    std::cout << "[DEBUG] Allocated memory for keyCache: " << layers * seqLen * kvDim * sizeof(float) << " bytes" << std::endl;

    ret = aclrtMalloc((void**)&valueCache, layers * seqLen * kvDim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR:] NPU memory allocation failed for valueCache!" << std::endl;
        exit(EXIT_FAILURE);
    }
    totalMemoryAllocated += layers * seqLen * kvDim * sizeof(float);
    std::cout << "[DEBUG] Allocated memory for valueCache: " << layers * seqLen * kvDim * sizeof(float) << " bytes" << std::endl;

    ret = aclrtMalloc((void**)&logits_gpu, vocab * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR:] NPU memory allocation failed for logits_gpu!" << std::endl;
        exit(EXIT_FAILURE);
    }
    totalMemoryAllocated += vocab * sizeof(float);
    std::cout << "[DEBUG] Allocated memory for logits_gpu: " << vocab * sizeof(float) << " bytes" << std::endl;

    logits = new float[config->vocabSize]();  // 为logits分配CPU内存
    totalMemoryAllocated += config->vocabSize * sizeof(float);
    std::cout << "[DEBUG] Allocated memory for logits (CPU): " << config->vocabSize * sizeof(float) << " bytes" << std::endl;

    // 打印总分配的内存大小
    std::cout << "[DEBUG] Total memory allocated: " << totalMemoryAllocated << " bytes" << std::endl;

    // 检查内存分配是否成功
    if (!currentActivation || !branchActivation || !extraBuffer || !hiddenBuffer || !extraHiddenBuffer ||
        !q || !k || !v || !attentionScores || !logits || !keyCache || !valueCache) {
        std::cerr << "[ERROR:] NPU memory allocation failed!" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CNPURunState::deallocateMemory() {
    // 释放 NPU 显存，使用 aclrtFree
    aclrtFree(currentActivation);
    aclrtFree(branchActivation);
    aclrtFree(extraBuffer);
    aclrtFree(hiddenBuffer);
    aclrtFree(extraHiddenBuffer);
    aclrtFree(q);
    aclrtFree(k);
    aclrtFree(v);
    aclrtFree(attentionScores);
    aclrtFree(keyCache);
    aclrtFree(valueCache);
    aclrtFree(logits_gpu);

    delete[] logits;  // 释放CPU内存
    currentActivation = nullptr;
    branchActivation = nullptr;
    extraBuffer = nullptr;
    hiddenBuffer = nullptr;
    q = nullptr;
    k = nullptr;
    v = nullptr;
    attentionScores = nullptr;
    keyCache = nullptr;
    valueCache = nullptr;
    logits_gpu = nullptr;
    logits = nullptr;
}
