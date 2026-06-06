#ifndef COMMON_HPP
#define COMMON_HPP

// ========= C++ 标准库 =========
#include <vector>
#include <cstring>  
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream> 
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <tuple>
#include <iomanip>  
#include <iostream>

// ========= DCU(HIP) =========
#ifdef USE_DCU
  #include <hip/hip_runtime.h>
  #include <hipcub/hipcub.hpp>
  #include <hipblas.h>
#endif

// // ========= CUDA =========
// #ifdef USE_CUDA
//   #ifdef __CUDACC__   // 仅在 NVCC 编译 .cu 文件时生效
//     #include <cub/cub.cuh>
//     #include <cuda_runtime.h>
//     #include <cublas_v2.h>
//   #endif
// #endif

// ========= Ascend =========
#ifdef USE_NPU
  #include <acl/acl.h>
  // #include <aclnnop/aclnn_rope_with_sin_cos_cache.h>
#endif

// ========= 后端统一接口 =========
// #ifdef USE_CUDA
//   #include "backend/cudaBackend.hpp"
// #elif defined(USE_DCU)
//   #include "backend/dcuBackend.hpp"
// #elif defined(USE_ASCEND)
//   #include "backend/ascendBackend.hpp"
// #else
//   #include "backend/backend.hpp"
// #endif

#endif // COMMON_HPP