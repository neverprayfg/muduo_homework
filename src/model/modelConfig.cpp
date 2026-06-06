#include "modelConfig.hpp"
CModelConfig::CModelConfig() {

}

CModelConfig::CModelConfig(int dim, int feedForwardDim, int numLayers, int numHeads, 
    int numKvHeads, int vocabSize, int maxSeqLen) {
    this->dim = dim;
    this->feedForwardDim = feedForwardDim;
    this->numLayers = numLayers;
    this->numHeads = numHeads;
    this->numKvHeads = numKvHeads;
    this->vocabSize = vocabSize;
    this->maxSeqLen = maxSeqLen;
}

CModelConfig::~CModelConfig() {

}