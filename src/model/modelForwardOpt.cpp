#include "modelForwardOpt.hpp"
#ifdef USE_NPU
#include "../backend/npuBackend.hpp"

float* CModelForwardOpt::forward(int token, int pos) {
    
    CModelConfig* config = &this->config;
    float* inputVec = state->currentActivation;
    const int embeddingDim = config->dim;
    const int kvDim = (config->dim * config->numKvHeads) / config->numHeads;
    const int kvHeadMultiplier = config->numHeads / config->numKvHeads;
    const int headSize = embeddingDim / config->numHeads;
    const int ffnHiddenDim = config->feedForwardDim;

    float* tokenEmbedding = w.tokenEmbeddingTable + token * embeddingDim;
    backend->copyMemory(inputVec, tokenEmbedding,
                    embeddingDim * sizeof(float),
                    true,  // dstOnDevice
                    true  // srcOnDevice
    );
    CNPUBackend* npuBackend = static_cast<CNPUBackend*>(backend);
    backend->rmsnorm(state->branchActivation, inputVec, w.rmsAttWeight, embeddingDim);
    float* logitsInput = inputVec;

    for (uint64_t layer = 0; layer < config->numLayers; ++layer) {
        const int kvCacheOffset = layer * config->maxSeqLen * kvDim;

        backend->matmul(state->q, state->branchActivation, w.wq + layer * embeddingDim * embeddingDim, embeddingDim, embeddingDim);
        backend->matmul(state->k, state->branchActivation, w.wk + layer * embeddingDim * kvDim, embeddingDim, kvDim);
        backend->matmul(state->v, state->branchActivation, w.wv + layer * embeddingDim * kvDim, embeddingDim, kvDim);
        backend->ropeEncoding(state->q, state->k, headSize, pos, embeddingDim, kvDim);

        int kvHeadDim = kvDim / config->numKvHeads;
        for (int kvHeadIdx = 0; kvHeadIdx < config->numKvHeads; ++kvHeadIdx) {
            float* dst_k = state->keyCache
                         + kvCacheOffset
                         + kvHeadIdx * config->maxSeqLen * kvHeadDim
                         + pos * kvHeadDim;

            float* src_k = state->k + kvHeadIdx * kvHeadDim;

            backend->copyMemory(dst_k, src_k,
                                kvHeadDim * sizeof(float),
                                true,  // dstOnDevice
                                true   // srcOnDevice
            );
            float* dst_v = state->valueCache
                         + kvCacheOffset
                         + kvHeadIdx * config->maxSeqLen * kvHeadDim
                         + pos * kvHeadDim;

            float* src_v = state->v + kvHeadIdx * kvHeadDim;
            backend->copyMemory(dst_v, src_v,
                    kvHeadDim * sizeof(float),
                    true,  // dstOnDevice
                    true   // srcOnDevice
            );
        }

        for (int headIdx = 0; headIdx < config->numHeads; ++headIdx) {
            float* query = state->q + headIdx * headSize;
            int kvHeadIndex = headIdx / kvHeadMultiplier;

            float* k = state->keyCache
                     + kvCacheOffset
                     + kvHeadIndex * config->maxSeqLen * kvHeadDim;

            float* v = state->valueCache
                     + kvCacheOffset
                     + kvHeadIndex * config->maxSeqLen * kvHeadDim;

            float* out = state->branchActivation + headIdx * headSize;
            float* scores = state->attentionScores + headIdx * config->maxSeqLen;

            backend->attentionSingleHead(query, k, v, scores, out, pos, headSize);
        }

        backend->matmul(state->extraBuffer, state->branchActivation, w.wo + layer * embeddingDim * embeddingDim, embeddingDim, embeddingDim);
        npuBackend->addRmsNorm(state->branchActivation, inputVec,
                               inputVec, state->extraBuffer,
                               w.rmsFfnWeight + layer * embeddingDim,
                               embeddingDim);

        backend->matmul(state->hiddenBuffer, state->branchActivation, w.w1 + layer * embeddingDim * ffnHiddenDim, embeddingDim, ffnHiddenDim);
        backend->matmul(state->extraHiddenBuffer, state->branchActivation, w.w3 + layer * embeddingDim * ffnHiddenDim, embeddingDim, ffnHiddenDim);

        backend->swiGLLUFunc(state->hiddenBuffer, state->extraHiddenBuffer, ffnHiddenDim);
        backend->matmul(state->extraBuffer, state->hiddenBuffer, w.w2 + layer * ffnHiddenDim * embeddingDim, ffnHiddenDim, embeddingDim);
        if (layer + 1 < config->numLayers) {
            npuBackend->addRmsNorm(state->branchActivation, inputVec,
                                   inputVec, state->extraBuffer,
                                   w.rmsAttWeight + (layer + 1) * embeddingDim,
                                   embeddingDim);
        } else {
            npuBackend->addRmsNorm(state->branchActivation, inputVec,
                                   inputVec, state->extraBuffer,
                                   w.rmsFinalWeight,
                                   embeddingDim);
            logitsInput = state->branchActivation;
        }
    }

    backend->matmul(state->logits_gpu, logitsInput, w.wcls, embeddingDim, config->vocabSize);
    backend->copyMemory(state->logits, state->logits_gpu,
                        config->vocabSize * sizeof(float),
                        false, // dstOnDevice = Host
                        true   // srcOnDevice = Device
    );
    return state->logits;
}
#else
float* CModelForwardOpt::forward(int token, int pos) {
    CModelConfig* config = &this->config;
    // CRunState* state = this->state;

    float* inputVec = state->currentActivation;
    const int embeddingDim = config->dim;
    const int kvDim = (config->dim * config->numKvHeads) / config->numHeads;
    const int kvHeadMultiplier = config->numHeads / config->numKvHeads;
    const int headSize = embeddingDim / config->numHeads;
    const int ffnHiddenDim = config->feedForwardDim;

    float* tokenEmbedding = w.tokenEmbeddingTable + token * embeddingDim;
    std::memcpy(inputVec, tokenEmbedding, embeddingDim * sizeof(float));    
    for (uint64_t layer = 0; layer < config->numLayers; ++layer) {

        backend->rmsnorm(state->branchActivation, inputVec, w.rmsAttWeight + layer * embeddingDim, embeddingDim);

        const int kvCacheOffset = layer * config->maxSeqLen * kvDim;
        state->k = state->keyCache + kvCacheOffset + pos * kvDim;
        state->v = state->valueCache + kvCacheOffset + pos * kvDim;

        backend->matmul(state->q, state->branchActivation, w.wq + layer * embeddingDim * embeddingDim, embeddingDim, embeddingDim);

        backend->matmul(state->k, state->branchActivation, w.wk + layer * embeddingDim * kvDim, embeddingDim, kvDim);

        backend->matmul(state->v, state->branchActivation, w.wv + layer * embeddingDim * kvDim, embeddingDim, kvDim);

        backend->ropeEncoding(state->q, state->k, headSize, pos, embeddingDim, kvDim);

        for (int headIdx = 0; headIdx < config->numHeads; ++headIdx) {
            float* query = state->q + headIdx * headSize;
            float* attentionScores = state->attentionScores + headIdx * config->maxSeqLen;
            
            backend->gemvQkSeq(query,state->keyCache+kvCacheOffset+(headIdx / kvHeadMultiplier) * headSize,attentionScores, pos, kvDim, headSize);

            backend->softmax(attentionScores, pos + 1);

            float* headOutput = state->branchActivation + headIdx * headSize;

            std::memset(headOutput, 0, headSize * sizeof(float));
            backend->weightedV(headOutput, state->valueCache + kvCacheOffset + (headIdx / kvHeadMultiplier) * headSize, attentionScores, pos, kvDim, headSize);
        }

        backend->matmul(state->extraBuffer, state->branchActivation, w.wo + layer * embeddingDim * embeddingDim, embeddingDim, embeddingDim);

        backend->axpy(inputVec, state->extraBuffer, 1.f, embeddingDim);

        backend->rmsnorm(state->branchActivation, inputVec, w.rmsFfnWeight + layer * embeddingDim, embeddingDim);

        backend->matmul(state->hiddenBuffer, state->branchActivation, w.w1 + layer * embeddingDim * ffnHiddenDim, embeddingDim, ffnHiddenDim);
       
        backend->matmul(state->extraHiddenBuffer, state->branchActivation, w.w3 + layer * embeddingDim * ffnHiddenDim, embeddingDim, ffnHiddenDim);
       
        backend->swiGLLUFunc(state->hiddenBuffer, state->extraHiddenBuffer, ffnHiddenDim);

        backend->matmul(state->branchActivation, state->hiddenBuffer, w.w2 + layer * ffnHiddenDim * embeddingDim, ffnHiddenDim, embeddingDim);

        backend->axpy(inputVec, state->branchActivation, 1.f, embeddingDim);

    }
    backend->rmsnorm(inputVec, inputVec, w.rmsFinalWeight, embeddingDim);
    backend->matmul(state->logits, inputVec, w.wcls, embeddingDim, config->vocabSize);
    return state->logits;
}

#endif
