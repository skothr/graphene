#include "cuda-vbo.cuh"

#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>

#include "field.cuh"
#include "fluid.cuh"
#include "vector-operators.h"
#include "cuda-tools.cuh"

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16
// #define BLOCKDIM_Z 8

template<typename T> __global__ void fillVLines_k(FluidField<T> src, CudaVBO dst, FluidParams<T> cp)
{
  unsigned long ix = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned long iy = blockIdx.y*blockDim.y + threadIdx.y;
  // unsigned long iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y)
    {
      unsigned long i = src.idx(ix, iy, 0);
      dst[i] = Vertex{ float3{(float)ix, (float)iy, (float)0}, float4{1.0f, 1.0f, 1.0f, 1.0f} };
    }
}

// wrapppers
template<typename T>
void fillVLines(FluidField<T> &src, CudaVBO &dst, FluidParams<T> &cp)
{
  if(src.size.x > 0 && src.size.y > 0 && dst.size >= src.size.x*src.size.y)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y); //, BLOCKDIM_Z);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y)); // , (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      dst.map();
      fillVLines_k <T> <<<grid, threads>>>(src, dst, cp);
      dst.unmap();
    }
  else { std::cout << "====> WARNING: skipped fillVLines --> " << src.size << " / " << dst.size << "\n"; }
}

// template instantiation
template void fillVLines<float>(FluidField<float> &src, CudaVBO &dst, FluidParams<float> &cp);
