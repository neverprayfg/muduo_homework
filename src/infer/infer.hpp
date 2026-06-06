#ifndef INFER_HPP
#define INFER_HPP

#include "../model/model.hpp"
#include "../model/transformer.hpp"
#include "../backend/backend.hpp"
#include "../model/modelForwardOpt.hpp"
#ifdef USE_NPU
#include "../backend/npuBackend.hpp"
#include "npuRunstate.hpp"
#endif
#include "../util.hpp"
#include "sampler.hpp"

#include "../common.hpp"

class CInfer{
    private:
        enum ModelType mt;
        enum BackendType bt;

        CBackend *backend;
        CRunState *state;
        CModel *model;
        CTokenizer *tokenizer;
        CSampler *sampler;

        int maxSeqLen;
        float temperature;   
        float topp;          
        int steps;
        unsigned long long rngSeed;

    public:
        CInfer();
        ~CInfer();
        
        void build(std::string modelPath, std::string tknzrPath, ModelType mt, BackendType bt, bool deviceOpt);
        std::tuple<std::string, int, long> generate(std::string prompt);
};

#endif