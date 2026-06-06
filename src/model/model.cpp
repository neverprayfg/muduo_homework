#include "model.hpp"

CModel::~CModel() {
    freeModel();
}

int compareTokens(const void *a, const void *b) {
    return strcmp(static_cast<const CTokenIndex*>(a)->token, static_cast<const CTokenIndex*>(b)->token);
}

int getTokenIndex(const char* str, CTokenIndex* vocabSortedList, int vocabSize) {
    CTokenIndex tok{ str };
    CTokenIndex* res = static_cast<CTokenIndex*>(bsearch(&tok, vocabSortedList, vocabSize, sizeof(CTokenIndex), compareTokens));
    return res != nullptr ? res->id : -1;
}
void CModel::encode(CTokenizer* tokenizer, std::string text, int8_t bos, int8_t eos, int* tokens, int* numTokens) {
    if (text.empty()) {
        std::cerr<<"[ERROR:] Text input is empty and cannot be processed.\n"<<std::endl;
        exit(EXIT_FAILURE);
    }
    if (tokenizer == nullptr || tokens == nullptr || numTokens == nullptr) {
        std::cerr<<"[ERROR:] Invalid input arguments detected. Ensure tokenizer, tokens, and numTokens are properly initialized.\n"<<std::endl;
        exit(EXIT_FAILURE);
    }
    if (tokenizer->vocabSortedList == nullptr) {
        tokenizer->vocabSortedList = static_cast<CTokenIndex*>(malloc(tokenizer->vocabSize * sizeof(CTokenIndex)));
        for (int i = 0; i < tokenizer->vocabSize; i++) {
            tokenizer->vocabSortedList[i].token = tokenizer->vocab[i];
            tokenizer->vocabSortedList[i].id = i;
        }
        qsort(tokenizer->vocabSortedList, tokenizer->vocabSize, sizeof(CTokenIndex), compareTokens);
    }
    
    char* strBuffer = static_cast<char*>(malloc((tokenizer->maxTokenLength * 2 + 1 + 2) * sizeof(char)));
    size_t strLen = 0;

    *numTokens = 0;

    if (bos) tokens[(*numTokens)++] = 1;
    
    if (text[0] != '\0') {
        int dummyPrefix = getTokenIndex(" ", tokenizer->vocabSortedList, tokenizer->vocabSize);
        if (dummyPrefix != -1) {
            tokens[(*numTokens)++] = dummyPrefix;
        }
    }

    for (size_t i = 0; i < text.size(); ++i) {
        char currentChar = text[i];
        if ((currentChar & 0xC0) != 0x80) {
            strLen = 0;
        }

        strBuffer[strLen++] = currentChar;
        strBuffer[strLen] = '\0';

        if (i + 1 < text.size() && (text[i + 1] & 0xC0) == 0x80 && strLen < 4) {
            continue;
        }

        int id = getTokenIndex(strBuffer, tokenizer->vocabSortedList, tokenizer->vocabSize);

        if (id != -1) {
            tokens[(*numTokens)++] = id;
        } else {

            // +3: <unk>, <s>, </s>
            for (size_t i = 0; i < strLen; i++) {
                tokens[(*numTokens)++] = static_cast<unsigned char>(strBuffer[i]) + 3;
            }
        }
        strLen = 0;
    }

    while (true) {
        float bestScore = -1e10;
        int bestId = -1;
        int bestIdx = -1;

        for (int i = 0; i < (*numTokens - 1); i++) {
            if (tokens[i] < 0 || tokens[i] >= tokenizer->vocabSize ||
                tokens[i + 1] < 0 || tokens[i + 1] >= tokenizer->vocabSize) {
                continue;
            }

            snprintf(strBuffer,
                     tokenizer->maxTokenLength * 2 + 1 + 2,
                     "%s%s",
                     tokenizer->vocab[tokens[i]],
                     tokenizer->vocab[tokens[i + 1]]);
            int id = getTokenIndex(strBuffer, tokenizer->vocabSortedList, tokenizer->vocabSize);
            if (id != -1 && tokenizer->vocabScores[id] > bestScore) {
                bestScore = tokenizer->vocabScores[id];
                bestId = id;
                bestIdx = i;
            }
        }

        if (bestIdx == -1) {
            break; 
        }

        tokens[bestIdx] = bestId;
        for (int i = bestIdx + 1; i < (*numTokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*numTokens)--; 
    }

    if (eos) tokens[(*numTokens)++] = 2;

    free(strBuffer);
}

char* CModel::decode(CTokenizer* tokenizer, int previousToken, int token) {
    char* piece = tokenizer->vocab[token];

    if (previousToken == 1 && piece[0] == ' ') {
        piece++;  }
    unsigned char byteVal;
    if (sscanf(piece, "<0x%02hhX>", &byteVal) == 1) {
        piece = reinterpret_cast<char*>(tokenizer->bytePieces) + byteVal * 2;
    }
    return piece;  
}


void CModel::mapWeightsToMemory(CModelConfig* modelConfig, float* ptr, int sharedWeights){
    const int headSize = modelConfig->dim / modelConfig->numHeads;
    const uint64_t numLayers = modelConfig->numLayers;
    float* currentPtr = ptr; 


    w.tokenEmbeddingTable = currentPtr;
    currentPtr += modelConfig->vocabSize * modelConfig->dim;


    w.rmsAttWeight = currentPtr;
    currentPtr += numLayers * modelConfig->dim;
    
    w.wq = currentPtr;
    currentPtr += numLayers * modelConfig->dim * (modelConfig->numHeads * headSize);
    
    w.wk = currentPtr;
    currentPtr += numLayers * modelConfig->dim * (modelConfig->numKvHeads * headSize);
    
    w.wv = currentPtr;
    currentPtr += numLayers * modelConfig->dim * (modelConfig->numKvHeads * headSize);
    
    w.wo = currentPtr;
    currentPtr += numLayers * (modelConfig->numHeads * headSize) * modelConfig->dim;

    w.rmsFfnWeight = currentPtr;
    currentPtr += numLayers * modelConfig->dim;
    
    w.w1 = currentPtr;
    currentPtr += numLayers * modelConfig->dim * modelConfig->feedForwardDim;
    
    w.w2 = currentPtr;
    currentPtr += numLayers * modelConfig->feedForwardDim * modelConfig->dim;
    
    w.w3 = currentPtr;
    currentPtr += numLayers * modelConfig->dim * modelConfig->feedForwardDim;


    w.rmsFinalWeight = currentPtr;
    currentPtr += modelConfig->dim;


    currentPtr += modelConfig->maxSeqLen * headSize / 2; 
    currentPtr += modelConfig->maxSeqLen * headSize / 2; 


    w.wcls = sharedWeights ? w.tokenEmbeddingTable : currentPtr;
}


void CModel::load(const std::string& checkpointPath, CModelConfig* modelConfig, int* fileDescriptor, float** data, ssize_t* totalFileSize) {
    

    std::ifstream fileStream(checkpointPath, std::ios::binary | std::ios::ate);
    if (!fileStream) {
        std::cerr<<"[ERROR:] Unable to open checkpoint file: " << checkpointPath <<std::endl;
    }


    *totalFileSize = fileStream.tellg();
    fileStream.seekg(0, std::ios::beg);


    fileStream.read(reinterpret_cast<char*>(modelConfig), sizeof(CModelConfig));
    if (!fileStream) {
        std::cerr<< "[ERROR:] Unable to read model config"<<std::endl;
    }


    bool sharedWeights = modelConfig->vocabSize > 0;
    modelConfig->vocabSize = std::abs(modelConfig->vocabSize);

    fileStream.close();


    *fileDescriptor = open(checkpointPath.c_str(), O_RDONLY);
    if (*fileDescriptor == -1) {
         std::cerr<< "[ERROR:] Unable to open file descriptor for: " << checkpointPath<<std::endl;
    }


    *data = static_cast<float*>(mmap(nullptr, *totalFileSize, PROT_READ, MAP_PRIVATE, *fileDescriptor, 0));
    if (*data == MAP_FAILED) {
         std::cerr<< "[ERROR:] Unable to memory-map the file: " << checkpointPath<<std::endl;
    }

    constexpr uint64_t configSizeInFloats = sizeof(CModelConfig) / sizeof(float);
    //float* weightsPtr = *data + configSizeInFloats;

    float* weightsPtr;
    size_t weights_size = *totalFileSize - sizeof(CModelConfig);

    if(backend->type == NPU ){ //断点
        weightsPtr = (float*)backend->allocMemory(weights_size);
        backend->copyMemory(
            weightsPtr,
            *data + sizeof(CModelConfig) / sizeof(float),
            weights_size,
            /*dstOnDevice=*/true,
            /*srcOnDevice=*/false
        );
    }
    else{
        weightsPtr = (float*)malloc(weights_size); 
        memcpy(weightsPtr, *data + sizeof(CModelConfig)/sizeof(float), weights_size);
    }
    mapWeightsToMemory(modelConfig, weightsPtr, sharedWeights);
}


void CModel::initializeModel(const std::string checkpointPath,CBackend *backend,CRunState *state){ 
    this->backend = backend;
    this->state = state;
    load(checkpointPath, &config, &fd, &data, &fileSize);
    this->state->allocateMemory(&config);
}


void CModel::freeModel() {

    if (data != MAP_FAILED) {
        munmap(data, fileSize);
    }
    
    if (fd != -1) {
        close(fd);
    }
    this->state->deallocateMemory();
}


float* CModel::forward(int token, int pos) {
    CModelConfig* config = &this->config;
    // CRunState* state = this->state;

    float* inputVec = state->currentActivation;
    const int embeddingDim = config->dim;
    const int kvDim = (config->dim * config->numKvHeads) / config->numHeads;
    const int kvHeadMultiplier = config->numHeads / config->numKvHeads;
    const int headSize = embeddingDim / config->numHeads;
    const int ffnHiddenDim = config->feedForwardDim;

    float* tokenEmbedding = w.tokenEmbeddingTable + token * embeddingDim;
    if(backend->type == NPU){
        backend->copyMemory(
        inputVec,
        tokenEmbedding,
        embeddingDim * sizeof(*inputVec),
        /*dstOnDevice=*/true,
        /*srcOnDevice=*/true
    );
}
    else
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
            if(backend->type == NPU){
                backend->setMemory(headOutput, 0, headSize * sizeof(float), /*onDevice=*/true);
            }
            else
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


    if(backend->type == NPU){
        backend->matmul(state->logits_gpu, inputVec, w.wcls, embeddingDim, config->vocabSize);
        backend->copyMemory(
                state->logits, 
                state->logits_gpu,
                config->vocabSize * sizeof(float),
                /*dstOnDevice=*/false,
                /*srcOnDevice=*/true
            );
    } 
    else
        backend->matmul(state->logits, inputVec, w.wcls, embeddingDim, config->vocabSize);
    return state->logits;
}
