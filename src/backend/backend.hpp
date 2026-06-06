#ifndef BACKEND_HPP
#define BACKEND_HPP
#include "../common.hpp"
#include "../util.hpp"
#include "../model/modelConfig.hpp"
#include "../infer/runState.hpp"
class CBackend {
    public:
       BackendType type = CPU; 
       virtual void softmax(float* x, int size);
       virtual  void matmul(float* xout, float* x, float* w, int n, int d);
       virtual  void rmsnorm(float* o, float* x, float* weight, int size);
       virtual void axpy(float *y, float *x, float factor, int dim);
       virtual  void dot(float *y, float *x1, float*x2, int dim);
       virtual  void ropeEncoding(float *q, float *k, int headSize, int position, int dim, int kvDim);
       virtual  void gemvQkSeq(float *q, float *key, float *att, int pos, int kvDim, int headSize);
       virtual  void weightedV(float *xb, float *value, float *att, int pos, int kvDim, int headSize);
       virtual  void swiGLLUFunc(float *hb, float *hb2, int hiddenDim);

       virtual void qkvMatmul(float* q_out, float* k_out, float* v_out, const float* x, const float* w_q, const float* w_k, const float* w_v, int d_in, int d_q, int d_k, int d_v);
       virtual void multiHeadAttention(int pos, CModelConfig* p, CRunState* s, int kv_dim, int kv_mul, int head_size, int loff);
       virtual void gemvAxpyFused(float *out, const float *in, const float *w, int n, int d, float alpha, float beta);  
       virtual void attentionSingleHead(float* q,float* k,float* v,float* attnScores,float* out,int pos,int headSize);       
       virtual void* allocMemory(size_t size);
       virtual void freeMemory(void* ptr);
       virtual void copyMemory(void* dst, const void* src, size_t size,
                                bool dstOnDevice, bool srcOnDevice);
       virtual void setMemory(void* ptr, int value, size_t size, bool onDevice);
       virtual void sync();
       CBackend();
        virtual ~CBackend() ;
        
    };

#endif
