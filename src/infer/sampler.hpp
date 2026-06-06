#ifndef SAMPLER_HPP
#define SAMPLER_HPP

#include "../backend/backend.hpp"
#include "../common.hpp"
struct Candidate {
    float probability;
    int tokenIndex;
};

class CSampler{
public:
    int vocabSize;
    Candidate* candidates; // 在 top-p 采样中使用的缓冲区
    float temperature;
    float topP;
    unsigned long long rngState;
    CSampler();
    ~CSampler();
    void initializeSampler(int vocabSize, float temperature, float topP, unsigned long long rngSeed);
    void freeSampler();
    int sample(float* logits,CBackend *backend);
};

#endif