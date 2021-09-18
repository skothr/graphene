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
#include "draw.cuh"

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

#define SIG_HIGHLIGHT_COLOR float4{0.5, 1.0, 0.5, 0.1}
#define MAT_HIGHLIGHT_COLOR float4{1.0, 0.5, 0.5, 0.1}

//// RENDERING ////

template<typename T>
__global__ void renderFieldEM_k(EMField<T> src, CudaTexture dst, RenderParams<T> rp, FieldParams<T> cp)
{
  typedef typename DimType<T,3>::VEC_T VT3;
  typedef typename DimType<T,4>::VEC_T VT4;
  long long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long long iy = blockIdx.y*blockDim.y + threadIdx.y;
  if(ix < dst.size.x && iy < dst.size.y)
    {
      int ti = ix + iy*dst.size.x;
      int2 fp = int2{(int)(ix*(src.size.x/(T)dst.size.x)),  // scale texture index to field index
                     (int)(iy*(src.size.y/(T)dst.size.y))};
      
      VT4 color = VT4{0.0, 0.0, 0.0, 0.0};
      for(int iz = max(0, min(src.size.z-1, rp.zRange.y)); iz >= rp.zRange.x; iz--)
        {
          int fi = src.idx(fp.x, fp.y, iz);
          //T qLen = (src.Q[fi].x - src.Q[fi].y);
          T eLen = length(src.E[fi]); T bLen = length(src.B[fi]);
          VT4 col = rp.brightness*rp.opacity*(//qLen*rp.Qmult*rp.Qcol +
                                              eLen*rp.Emult*rp.Ecol +
                                              bLen*rp.Bmult*rp.Bcol);

          VT3 pCell = VT3{(T)fp.x, (T)fp.y, (T)iz}; VT3 pSrc = rp.penPos;
          VT3 diff; VT3 diff0; VT3 diff1;  VT3 dist_2; VT3 dist0_2; VT3 dist1_2;
          if(rp.sigPenHighlight &&
             penOverlaps2 (pCell, pSrc, diff, diff0, diff1, dist_2, dist0_2, dist1_2, (Pen<T>*)&rp.sigPen, cp, 0.0f) &&
             !penOverlaps2(pCell, pSrc, diff, diff0, diff1, dist_2, dist0_2, dist1_2, (Pen<T>*)&rp.sigPen, cp, -1.0f)) { col += SIG_HIGHLIGHT_COLOR; }
          if(rp.matPenHighlight &&
             penOverlaps2 (pCell, pSrc, diff, diff0, diff1, dist_2, dist0_2, dist1_2, (Pen<T>*)&rp.matPen, cp, 0.0f) &&
             !penOverlaps2(pCell, pSrc, diff, diff0, diff1, dist_2, dist0_2, dist1_2, (Pen<T>*)&rp.matPen, cp, -1.0f)) { col += MAT_HIGHLIGHT_COLOR; }
          
          fluidBlend(color, col, rp);
          if(color.x >= 1 || color.y >= 1 || color.z >= 1) { break; }
        }
      // blend with background color
      T a = color.w;
      color += VT4{BG_COLOR.x, BG_COLOR.y, BG_COLOR.z, 0.0} * BG_COLOR.w*(1-a*rp.brightness);
      color.w += BG_COLOR.w*(1-color.w)*(rp.opacity);
      dst[ti] += float4{ max(0.0f, min(1.0f, (float)color.x)), max(0.0f, min(1.0f, (float)color.y)), max(0.0f, min(1.0f, (float)color.z)), 1.0f };
    }
}

template<typename T>
__global__ void renderFieldMat_k(Field<Material<T>> src, CudaTexture dst, RenderParams<T> rp, FieldParams<T> cp)
{
  typedef typename DimType<T,3>::VEC_T VT3;
  typedef typename DimType<T,4>::VEC_T VT4;
  long long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long long iy = blockIdx.y*blockDim.y + threadIdx.y;
  if(ix < dst.size.x && iy < dst.size.y)
    {
      int ti = ix + iy*dst.size.x;
      int2 fp = int2{(int)(ix*(src.size.x/(T)dst.size.x)),  // scale texture index to field index
                     (int)(iy*(src.size.y/(T)dst.size.y))};
      
      VT4 color = VT4{0.0f, 0.0f, 0.0f, 0.0f};
      for(int iz = min(src.size.z-1, rp.zRange.y); iz >= rp.zRange.x; iz--)
        {
          int fi = src.idx(fp.x, fp.y, iz);
          Material<T> mat = src[fi];
          VT4 col = (mat.vacuum() ? VT4{0.0f, 0.0f, 0.0f, 1.0f} :
                     rp.brightness*rp.opacity*(mat.permittivity*rp.epMult*rp.epCol +
                                               mat.permeability*rp.muMult*rp.muCol +
                                               mat.conductivity*rp.sigMult*rp.sigCol));
         
          VT3 pCell = VT3{(T)fp.x, (T)fp.y, (T)iz}; VT3 pSrc = rp.penPos;
          VT3 diff; VT3 diff0; VT3 diff1;  VT3 dist_2; VT3 dist0_2; VT3 dist1_2;
          if(rp.sigPenHighlight &&
             penOverlaps2 (pCell, pSrc, diff, diff0, diff1, dist_2, dist0_2, dist1_2, (Pen<T>*)&rp.sigPen, cp, 0.0f) &&
             !penOverlaps2(pCell, pSrc, diff, diff0, diff1, dist_2, dist0_2, dist1_2, (Pen<T>*)&rp.sigPen, cp, -1.0f)) { col += SIG_HIGHLIGHT_COLOR; }
          if(rp.matPenHighlight &&
             penOverlaps2 (pCell, pSrc, diff, diff0, diff1, dist_2, dist0_2, dist1_2, (Pen<T>*)&rp.matPen, cp, 0.0f) &&
             !penOverlaps2(pCell, pSrc, diff, diff0, diff1, dist_2, dist0_2, dist1_2, (Pen<T>*)&rp.matPen, cp, -1.0f)) { col += MAT_HIGHLIGHT_COLOR; }

         fluidBlend(color, col, rp);
         if(color.x >= 1.0f || color.y >= 1.0f || color.z >= 1.0f) { break; }
        }
      // blend with background color
      T a = color.w;
      color += VT4{BG_COLOR.x, BG_COLOR.y, BG_COLOR.z, 0.0} * BG_COLOR.w*(1-a*rp.brightness);
      color.w += BG_COLOR.w*(1-color.w)*(rp.opacity);
      dst[ti] += float4{ max(0.0f, min(1.0f, (float)color.x)), max(0.0f, min(1.0f, (float)color.y)), max(0.0f, min(1.0f, (float)color.z)), 1.0f };
    }
}


template<typename T>
__global__ void rtRenderFieldEM_k(EMField<T> src, CudaTexture dst, CameraDesc<T> cam, RenderParams<T> rp, FieldParams<T> cp,
                                  typename DimType<T, 2>::VEC_T aspect)
{
  typedef typename DimType<T,2>::VEC_T VT2;
  typedef typename DimType<T,4>::VEC_T VT4;
  long long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long long iy = blockIdx.y*blockDim.y + threadIdx.y;
  if(ix < dst.size.x && iy < dst.size.y)
    {
      Ray<T> ray = cam.castRay(VT2{ix/(T)dst.size.x, iy/(T)dst.size.y}, aspect);
      VT4 color = rayTraceField(src, ray, rp, cp);
      
      long long ti = ix + iy*dst.size.x;
      //dst[ti] = (color.w < 0.0f ? VT4{0.0f, 0.0f, 0.0f, 1.0f} : float4{(float)color.x, (float)color.y, (float)color.z, 1.0f);
      dst[ti] += float4{ max(0.0f, min(1.0f, (float)color.x)), max(0.0f, min(1.0f, (float)color.y)), max(0.0f, min(1.0f, (float)color.z)), 1.0f };
    }
}
template<typename T>
__global__ void rtRenderFieldMat_k(EMField<T> src, CudaTexture dst, CameraDesc<T> cam, RenderParams<T> rp, FieldParams<T> cp,
                                   typename DimType<T, 2>::VEC_T aspect)
{
  typedef typename DimType<T,2>::VEC_T VT2;
  typedef typename DimType<T,4>::VEC_T VT4;
  long long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long long iy = blockIdx.y*blockDim.y + threadIdx.y;
  if(ix < dst.size.x && iy < dst.size.y)
    {
      Ray<T> ray = cam.castRay(VT2{ix/(T)dst.size.x, iy/(T)dst.size.y}, aspect);
      VT4 color = rayTraceField(src, ray, rp, cp);
      
      long long ti = ix + iy*dst.size.x;
      // dst[ti] = (color.w < 0.0f ? float4{0.0f, 0.0f, 0.0f, 1.0f} : float4{(float)color.x, (float)color.y, (float)color.z, 1.0f});
      dst[ti] += float4{ max(0.0f, min(1.0f, (float)color.x)), max(0.0f, min(1.0f, (float)color.y)), max(0.0f, min(1.0f, (float)color.z)), 1.0f };
    }
}

// wrappers
template<typename T>
void renderFieldEM(EMField<T> &src, CudaTexture &dst, const RenderParams<T> &rp, const FieldParams<T> &cp)
{
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y)); // 2D -- thread texture pixels
      bool mapped = dst.mapped;
      if(!mapped) { dst.map(); }
      renderFieldEM_k<<<grid, threads>>>(src, dst, rp, cp);
      // cudaDeviceSynchronize(); getLastCudaError("====> ERROR: renderFieldEM_k failed!");
      if(!mapped) { dst.unmap(); }
    }
  else { std::cout << "Skipped EMField render --> " << src.size << " / " << dst.size << " \n"; }
}
template<typename T>
void renderFieldMat(Field<Material<T>> &src, CudaTexture &dst, const RenderParams<T> &rp, const FieldParams<T> &cp)
{
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y)); // 2D -- thread texture pixels
      bool mapped = dst.mapped;
      if(!mapped) { dst.map(); }
      renderFieldMat_k<<<grid, threads>>>(src, dst, rp, cp);
      // cudaDeviceSynchronize(); getLastCudaError("====> ERROR: renderFieldMat_k failed!");
      if(!mapped) { dst.unmap(); }
    }
  else { std::cout << "Skipped EMField render --> " << src.size << " / " << dst.size << " \n"; }
}

template<typename T>
void raytraceFieldEM(EMField<T> &src, CudaTexture &dst, const Camera<T> &camera, const RenderParams<T> &rp, const FieldParams<T> &cp, 
                     const Vector<T, 2> &aspect)
{
  typedef typename DimType<T,2>::VEC_T VT2;
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y)); // 2D -- thread texture pixels
      bool mapped = dst.mapped;
      if(!mapped) { dst.map(); }
      rtRenderFieldEM_k<<<grid, threads>>>(src, dst, camera.desc, rp, cp, VT2{aspect.x, aspect.y});
      // cudaDeviceSynchronize(); getLastCudaError("====> ERROR: raytraceFieldEM_k failed!");
      if(!mapped) { dst.unmap(); }
    }
  else { std::cout << "Skipped EMField render (RT) --> " << src.size << " / " << dst.size << " \n"; }
}

template<typename T>
void raytraceFieldMat(EMField<T> &src, CudaTexture &dst, const Camera<T> &camera, const RenderParams<T> &rp, const FieldParams<T> &cp,
                      const Vector<T, 2> &aspect)
{
  typedef typename DimType<T,2>::VEC_T VT2;
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y)); // 2D -- thread texture pixels
      bool mapped = dst.mapped;
      if(!mapped) { dst.map(); }
      rtRenderFieldMat_k<<<grid, threads>>>(src, dst, camera.desc, rp, cp, VT2{aspect.x, aspect.y});
      // cudaDeviceSynchronize(); getLastCudaError("====> ERROR: raytraceFieldEM_k failed!");
      if(!mapped) { dst.unmap(); }
    }
  else { std::cout << "Skipped EMField render (RT) --> " << src.size << " / " << dst.size << " \n"; }
}

// template instantiation
template void renderFieldEM   <float>(EMField<float>         &src, CudaTexture &dst, const RenderParams<float> &rp, const FieldParams<float> &cp);
template void renderFieldMat  <float>(Field<Material<float>> &src, CudaTexture &dst, const RenderParams<float> &rp, const FieldParams<float> &cp);
template void raytraceFieldEM <float>(EMField<float>         &src, CudaTexture &dst, const Camera<float> &camera,
                                      const RenderParams<float> &rp, const FieldParams<float> &cp, const Vec2f &aspect);
template void raytraceFieldMat<float>(EMField<float>         &src, CudaTexture &dst, const Camera<float> &camera,
                                      const RenderParams<float> &rp, const FieldParams<float> &cp, const Vec2f &aspect);
