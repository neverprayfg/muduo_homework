// src/backend/backendnpu.hpp
#ifndef BACKENDNPU_HPP
#define BACKENDNPU_HPP

#include "backend.hpp"

/*
#include <acl/acl.h>
#include <iostream>
*/
// 全局 ACL 句柄
// aclrtContext context = nullptr;
// aclrtStream  stream  = nullptr;


class CNPUBackend : public CBackend {
public:

    CNPUBackend();
    ~CNPUBackend();

    void matmul(float* xout, float* x, float* w, int n, int d);                                                     //NPU
    void rmsnorm(float* o, float* x, float* weight, int size);                                                      //NPU
    void addRmsNorm(float* y, float* xOut, float* x1, float* x2, float* weight, int size);                          //NPU
    void axpy(float *y, float *x, float factor, int dim);                                                           //NPU
    void swiGLLUFunc(float *hb, float *hb2, int hiddenDim);                                                         //NPU
    //void gemvQkSeq(float *q, float *key, float *att, int pos, int kvDim, int headSize) override;
    //void softmax(float* x, int size) override;
    //void weightedV(float *xb, float *value, float *att, int pos, int kvDim, int headSize) override;
    void attentionSingleHead(float* q,float* k,float* v,float* attnScores,float* out,int pos,int headSize);          //NPU
    void ropeEncoding(float *q, float *k, int headSize, int position, int dim, int kvDim);                           //NPU
    void* allocMemory(size_t size);
    void freeMemory(void* ptr);
    void copyMemory(void* dst, const void* src, size_t size,
                    bool dstOnDevice, bool srcOnDevice);
    void setMemory(void* ptr, int value, size_t size, bool onDevice);
    void sync();

    struct Impl;
    Impl* pImpl;

};

#endif // BACKENDNPU_HPP
