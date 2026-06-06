#ifndef RUNSTATE_HPP
#define RUNSTATE_HPP

#include "../model/modelConfig.hpp"

class CRunState {
public:

    float* currentActivation;  // 当前时间步的激活值(dim)
    float* branchActivation;   // 残差分支内部当前时间步的激活值(dim)
    float* extraBuffer;        // 额外的缓冲区(dim)
    float* hiddenBuffer;       // FFN 中隐藏维度的缓冲区(dim)
    float* extraHiddenBuffer;  // FFN 中额外隐藏维度的缓冲区(dim)

    //注意力机制
    float* q;              // (dim)
    float* k;              // (dim)
    float* v;              // (dim)
    float* attentionScores;    // (numHeads, seqLen)
    float* logits; 
    float *logits_gpu;            

    //KVCache
    float* keyCache;           // KCache (layer, seqLen, dim)
    float* valueCache;         // VCache (layer, seqLen, dim)

    CRunState();
    ~CRunState();

    virtual void allocateMemory(CModelConfig* config);
    virtual void deallocateMemory();
};

#endif 