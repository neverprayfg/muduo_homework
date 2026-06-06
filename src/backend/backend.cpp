
#include "backend.hpp"

CBackend::CBackend() {};

CBackend::~CBackend() {};

/*  TODO
    softmax: 将实数向量转换为概率分布
    (1) sum = e^(x[0]) + e^(x[1]) + ... + e^(x[n-1])
    (2) x[0] = (e^x[0]) / sum,
        x[1] = (e^x[1]) / sum,
        ......,
        x[n-1] = (e^x[n-1]) / sum
*/
void CBackend::softmax(float *x, int n)
{
    float maxVal = x[0];
    for (int i = 1; i < n; ++i)
    {
        if (x[i] > maxVal)
            maxVal = x[i]; // 防止数值溢出：减去最大值
    }

    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        x[i] = exp(x[i] - maxVal);
        sum += x[i];
    }

    for (int i = 0; i < n; ++i)
    {
        x[i] /= sum;
    }
}

/*  TODO
    matmul: 矩阵-向量乘
        o[d][1] = w[d][n] X x[n][1]
*/
void CBackend::matmul(float *o, float *x, float *w, int n, int d)
{
  // TODO....
}

/*  TODO
    rmsnorm: 归一化
    (1) squareSum = (x[0]*x[0] + x[1]*x[1] + ... + x[n-1]*x[n-1])/n + 1e-5f
    (2) y[0] = w[0] * (x[0] / sqrt(squareSum)),
        ......,
        y[n-1] = w[n-1] * (x[n-1] / sqrt(squareSum))
*/
void CBackend::rmsnorm(float *y, float *x, float *w, int n)
{
    // TODO....
}

/*  TODO
    axpy: 标量和向量相乘
    y[0] = x[0] * factor,
    y[1] = x[1] * factor,
    ......,
    y[dim-1] = x[dim-1] * factor
*/
void CBackend::axpy(float *y, float *x, float factor, int dim)
{
    // TODO....
}

/*  TODO
    dot: 向量点积
    *y += x1[0]*x2[0] + x1[1]*x2[1] + ... + x1[dim-1]*x2[dim-1]
*/
void CBackend::dot(float *y, float *x1, float *x2, int dim)
{
    for (int i = 0; i < dim; ++i)
    {
        *y += x1[i] * x2[i];
    }
}

void CBackend::ropeEncoding(float *q, float *k, int headSize, int position, int dim, int kvDim)
{
    for (int i = 0; i < dim; i += 2)
    {
        int headDim = i % headSize;
        float freq = 1.0f / powf(10000.0f, headDim / (float)headSize);
        float val = position * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kvDim ? 2 : 1;
        for (int v = 0; v < rotn; v++)
        {
            float *vec = v == 0 ? q : k;
            float v0 = vec[i];
            float v1 = vec[i + 1];
            vec[i] = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}
void CBackend::gemvQkSeq(float *q, float *key, float *attentionScores, int pos, int kvDim, int headSize)
{
    for (int timestep = 0; timestep <= pos; timestep++)
    {
        float *k = key + timestep * kvDim;
        float score = 0.0f;
        dot(&score, q, k, headSize);
        score /= sqrtf(headSize);
        attentionScores[timestep] = score;
    }
}

void CBackend::weightedV(float *headOutput, float *value, float *attentionScores, int pos, int kvDim, int headSize)
{
    for (int t = 0; t <= pos; t++)
    {
        float *v = value + t * kvDim;
        float a = attentionScores[t];
        axpy(headOutput, v, a, headSize);
    }
}
void CBackend::swiGLLUFunc(float *headOutput, float *value, int hiddenDim)
{
    for (int i = 0; i < hiddenDim; i++)
    {
        float val = headOutput[i];
        val *= (1.0f / (1.0f + expf(-val)));
        val *= value[i];
        headOutput[i] = val;
    }
}
void CBackend::qkvMatmul(
    float* q_out, float* k_out, float* v_out,     // 输出地址
    const float* x,                                // 输入激活 x
    const float* w_q, const float* w_k, const float* w_v, // 权重
    int d_in, int d_q, int d_k, int d_v            // 各维度
) {
}
void CBackend::multiHeadAttention(int pos, CModelConfig* p, CRunState* s, int kvDim, int kvMul, int head_size, int loff) {
}
void CBackend::gemvAxpyFused(float *out, const float *in, const float *w, int n, int d, float alpha, float beta){
    
}
void CBackend:: attentionSingleHead(float* q,float* k,float* v,float* attnScores,float* out,int pos,int headSize){
}
void* CBackend::allocMemory(size_t size){
        return std::malloc(size);
    }
    
void CBackend::freeMemory(void* ptr){
        std::free(ptr);
    }

void CBackend::copyMemory(void* dst, const void* src, size_t size,
                bool dstOnDevice, bool srcOnDevice){
        std::memcpy(dst, src, size);
    }
void CBackend::setMemory(void* ptr, int value, size_t size, bool onDevice){
        std::memset(ptr, value, size);
    }
void CBackend::sync() {
    return;
}