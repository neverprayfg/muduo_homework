#include "runState.hpp"
#include <iostream>

CRunState::CRunState()
    : currentActivation(nullptr), branchActivation(nullptr), extraBuffer(nullptr),
      hiddenBuffer(nullptr), extraHiddenBuffer(nullptr), q(nullptr), k(nullptr),
      v(nullptr), attentionScores(nullptr), logits(nullptr), logits_gpu(nullptr),
      keyCache(nullptr), valueCache(nullptr) {}


CRunState::~CRunState() {
}

void CRunState::allocateMemory(CModelConfig* config) {
    // exit(1);
    int kvDim = (config->dim * config->numKvHeads) / config->numKvHeads;

    currentActivation = new float[config->dim]();
    branchActivation = new float[config->dim]();
    extraBuffer = new float[config->dim]();
    hiddenBuffer = new float[config->feedForwardDim]();
    extraHiddenBuffer = new float[config->feedForwardDim]();
    q = new float[config->dim]();
    keyCache = new float[config->numLayers * config->maxSeqLen * kvDim]();
    valueCache = new float[config->numLayers * config->maxSeqLen * kvDim]();
    attentionScores = new float[config->numHeads * config->maxSeqLen]();
    logits = new float[config->vocabSize]();  
    logits_gpu = new float[config->vocabSize]();
    if (!currentActivation || !branchActivation || !extraBuffer || !hiddenBuffer || !extraHiddenBuffer ||
        !q || !keyCache || !valueCache || !attentionScores || !logits) {
        std::cerr << "[ERROR:] Memory allocation failed!" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CRunState::deallocateMemory() {
    delete[] currentActivation;
    delete[] branchActivation;
    delete[] extraBuffer;
    delete[] hiddenBuffer;
    delete[] extraHiddenBuffer;
    delete[] q;
    delete[] attentionScores;
    delete[] logits; 
    delete[] logits_gpu;
    delete[] keyCache;
    delete[] valueCache;
}
