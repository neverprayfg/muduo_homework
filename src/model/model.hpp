#ifndef MODEL_HPP
#define MODEL_HPP

#include "tokenizer.hpp"
#include "modelConfig.hpp"
#include "../infer/runState.hpp"
#include "../backend/backend.hpp"
#include "../common.hpp"

class CModel{
public:

    CModelConfig config; 
    CRunState* state; 
    CBackend* backend;  
  
    int fd;             
    float* data;         
    ssize_t fileSize;    

    struct weights {
        float* tokenEmbeddingTable;
        float* rmsAttWeight;         // 注意力层的 RMS 权重
        float* rmsFfnWeight;         // 前馈网络层的 RMS 权重
        float* wq;
        float* wk;
        float* wv;
        float* wo;
        float* w1;
        float* w2;
        float* w3;
        float* rmsFinalWeight;
        float* wcls;
    } w; // 模型权重信息
    CModel() = default; 
    ~CModel() ; 
    
    virtual void load(const std::string& checkpointPath, CModelConfig* modelConfig, int* fileDescriptor, float** data, ssize_t* totalFileSize) ; 
    virtual void encode(CTokenizer* t, std::string text, int8_t bos, int8_t eos, int *tokens, int *nTokens);
    virtual float* forward(int token, int pos);
    virtual char* decode(CTokenizer* t, int prevToken, int token);
    virtual void initializeModel(const std::string checkpointPath,CBackend *backend,CRunState *state);
    virtual void mapWeightsToMemory(CModelConfig* modelConfig, float* ptr, int sharedWeights);
    virtual void freeModel();
};


#endif