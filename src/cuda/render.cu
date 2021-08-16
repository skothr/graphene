#include "render.cuh"

#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>

#include "physics.h"
#include "raytrace.cuh"
#include "vector-operators.h"
#include "cuda-tools.cuh"
#include "cuda-vbo.h"
#include "mathParser.hpp"

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16


//// RENDERING ////

template<typename T>
__global__ void renderFieldEM_k(EMField<T> src, CudaTexture dst, EmRenderParams rp)
{
  long long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long long iy = blockIdx.y*blockDim.y + threadIdx.y;
  if(ix < dst.size.x && iy < dst.size.y)
    {
      int ti = ix + iy*dst.size.x;
      int2 fp = int2{(int)(ix*(src.size.x/(T)dst.size.x)),  // scale texture index to field index
                     (int)(iy*(src.size.y/(T)dst.size.y))};
      float bScale = 1.0f; //float(src.size.z-1)/(rp.numLayers2D+1);
      float4 color = float4{0.0f, 0.0f, 0.0f, 0.0f};
      for(int i = max(0, min(src.size.z-1, rp.numLayers2D-1)); i >= 0; i--)
        {
         int fi = src.idx(fp.x, fp.y, i);
         T qLen = (src.Q[fi].x - src.Q[fi].y); T eLen = length(src.E[fi]); T bLen = length(src.B[fi]);
         float4 col = rp.brightness*rp.opacity*(qLen*rp.Qmult*rp.Qcol + eLen*rp.Emult*rp.Ecol + bLen*rp.Bmult*rp.Bcol);
         col.x *= bScale; col.y *= bScale; col.z *= bScale;
         fluidBlend(color, col, rp);
         //if(color.x >= 1.0f || color.y >= 1.0f || color.z >= 1.0f) { break; }
        }
      // float a = color.w;
      // color += float4{BG_COLOR.x, BG_COLOR.y, BG_COLOR.z, 0.0} * BG_COLOR.w*(1-a*rp.brightness);
      // color.w += BG_COLOR.w*(1-color.w)*(rp.opacity);
      dst[ti] += float4{ max(0.0f, min(1.0f, color.x)), max(0.0f, min(1.0f, color.y)), max(0.0f, min(1.0f, color.z)), 1.0f };
    }
}

template<typename T>
__global__ void renderFieldMat_k(Field<Material<T>> src, CudaTexture dst, EmRenderParams rp)
{
  long long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long long iy = blockIdx.y*blockDim.y + threadIdx.y;
  if(ix < dst.size.x && iy < dst.size.y)
    {
      int ti = ix + iy*dst.size.x;
      int2 fp = int2{(int)(ix*(src.size.x/(T)dst.size.x)),  // scale texture index to field index
                     (int)(iy*(src.size.y/(T)dst.size.y))};
      float bScale = 1.0f; //float(src.size.z-1)/(rp.numLayers2D+1);
      float4 color = float4{0.0f, 0.0f, 0.0f, 0.0f};
      for(int i = min(src.size.z-1, rp.numLayers2D-1); i >= 0; i--)
        {
         int fi = src.idx(fp.x, fp.y, i);
         Material<T> mat = src[fi];
         float4 col = (mat.vacuum() ? float4{0.0f, 0.0f, 0.0f, 1.0f} :
                       rp.brightness*rp.opacity*(mat.permittivity*rp.epMult*rp.epCol +
                                                 mat.permeability*rp.muMult*rp.muCol +
                                                 mat.conductivity*rp.sigMult*rp.sigCol));
         col.x *= bScale; col.y *= bScale; col.z *= bScale;
         fluidBlend(color, col, rp);
         //if(color.x >= 1.0f || color.y >= 1.0f || color.z >= 1.0f) { break; }
        }
      // float a = color.w;
      // color += float4{BG_COLOR.x, BG_COLOR.y, BG_COLOR.z, 0.0} * BG_COLOR.w*(1-a*rp.brightness);
      // color.w += BG_COLOR.w*(1-color.w)*(rp.opacity);
      dst[ti] += float4{ max(0.0f, min(1.0f, color.x)), max(0.0f, min(1.0f, color.y)), max(0.0f, min(1.0f, color.z)), 1.0f };
    }
}


template<typename T>
__global__ void rtRenderFieldEM_k(EMField<T> src, CudaTexture dst, CameraDesc<double> cam, EmRenderParams rp, FieldParams<T> cp, double2 aspect)
{
  long long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long long iy = blockIdx.y*blockDim.y + threadIdx.y;
  if(ix < dst.size.x && iy < dst.size.y)
    {
      Ray<double> ray = cam.castRay(double2{ix/double(dst.size.x), iy/double(dst.size.y)}, aspect);
      float4 color = rayTraceField(src, ray, rp, cp);
      
      long long ti = ix + iy*dst.size.x;
      dst.dData[ti] = (color.w < 0.0f ? float4{0.0f, 0.0f, 0.0f, 1.0f} : color);
    }
}
template<typename T>
__global__ void rtRenderFieldMat_k(EMField<T> src, CudaTexture dst, CameraDesc<double> cam, EmRenderParams rp, FieldParams<T> cp, double2 aspect)
{
  long long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long long iy = blockIdx.y*blockDim.y + threadIdx.y;
  if(ix < dst.size.x && iy < dst.size.y)
    {
      Ray<double> ray = cam.castRay(double2{ix/double(dst.size.x), iy/double(dst.size.y)}, aspect);
      float4 color = rayTraceField(src, ray, rp, cp);
      
      long long ti = ix + iy*dst.size.x;
      dst.dData[ti] = (color.w < 0.0f ? float4{0.0f, 0.0f, 0.0f, 1.0f} : float4{color.x, color.y, color.z, 1.0f});
    }
}

// wrappers
template<typename T>
void renderFieldEM(EMField<T> &src, CudaTexture &dst, const EmRenderParams &rp)
{
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y)); // 2D -- thread texture pixels
      bool mapped = dst.mapped;
      if(!mapped) { dst.map(); }
      renderFieldEM_k<<<grid, threads>>>(src, dst, rp); cudaDeviceSynchronize(); getLastCudaError("====> ERROR: renderFieldEM_k failed!");
      if(!mapped) { dst.unmap(); }
    }
  else { std::cout << "Skipped EMField render --> " << src.size << " / " << dst.size << " \n"; }
}
template<typename T>
void renderFieldMat(Field<Material<T>> &src, CudaTexture &dst, const EmRenderParams &rp)
{
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y)); // 2D -- thread texture pixels
      bool mapped = dst.mapped;
      if(!mapped) { dst.map(); }
      renderFieldMat_k<<<grid, threads>>>(src, dst, rp); cudaDeviceSynchronize(); getLastCudaError("====> ERROR: renderFieldMat_k failed!");
      if(!mapped) { dst.unmap(); }
    }
  else { std::cout << "Skipped EMField render --> " << src.size << " / " << dst.size << " \n"; }
}

template<typename T>
void raytraceFieldEM(EMField<T> &src, CudaTexture &dst, const Camera<double> &camera, const EmRenderParams &rp, const FieldParams<T> &cp, const Vec2d &aspect)
{
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y)); // 2D -- thread texture pixels
      bool mapped = dst.mapped;
      if(!mapped) { dst.map(); }
      rtRenderFieldEM_k<<<grid, threads>>>(src, dst, camera.desc, rp, cp, double2{aspect.x, aspect.y});
      cudaDeviceSynchronize(); getLastCudaError("====> ERROR: raytraceFieldEM_k failed!");
      if(!mapped) { dst.unmap(); }
    }
  else { std::cout << "Skipped EMField render (RT) --> " << src.size << " / " << dst.size << " \n"; }
}

template<typename T>
void raytraceFieldMat(EMField<T> &src, CudaTexture &dst, const Camera<double> &camera, const EmRenderParams &rp, const FieldParams<T> &cp, const Vec2d &aspect)
{
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y)); // 2D -- thread texture pixels
      bool mapped = dst.mapped;
      if(!mapped) { dst.map(); }
      rtRenderFieldMat_k<<<grid, threads>>>(src, dst, camera.desc, rp, cp, double2{aspect.x, aspect.y});
      cudaDeviceSynchronize(); getLastCudaError("====> ERROR: raytraceFieldEM_k failed!");
      if(!mapped) { dst.unmap(); }
    }
  else { std::cout << "Skipped EMField render (RT) --> " << src.size << " / " << dst.size << " \n"; }
}

// template instantiation
template void renderFieldEM   <float>(EMField<float>          &src, CudaTexture &dst, const EmRenderParams &rp);
template void renderFieldMat  <float>(Field<Material<float>> &src, CudaTexture &dst, const EmRenderParams &rp);
template void raytraceFieldEM <float>(EMField<float> &src, CudaTexture &dst, const Camera<double> &camera,
                                      const EmRenderParams &rp, const FieldParams<float> &cp, const Vec2d &aspect);
template void raytraceFieldMat<float>(EMField<float> &src, CudaTexture &dst, const Camera<double> &camera,
                                      const EmRenderParams &rp, const FieldParams<float> &cp, const Vec2d &aspect);
