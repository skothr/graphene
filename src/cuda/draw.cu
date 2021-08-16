#include "draw.hpp"

#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>

#include "physics.h"
#include "raytrace.cuh"
#include "vector-operators.h"
#include "cuda-tools.cuh"
#include "mathParser.hpp"

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16
#define BLOCKDIM_Z 1



// TODO: better overlap detection
// inline __device__ float3 pointInCircle(const float3 cp, const float3 lp, float cr) { }
// // returns intersections
// template<typename T> inline __device__ typename Dims<T, 4>::VECTOR_T
// lineIntersectCircle(T cRad, const typename Dims<T, 2>::VECTOR_T cp, const typename Dims<T, 2>::VECTOR_T lp1, const typename Dims<T, 2>::VECTOR_T lp2)
// {
//   using VT2 = typename Dims<T, 2>::VECTOR_T;
//   using VT2 = typename Dims<T, 4>::VECTOR_T;
//   VT2 lDiff = lp2 - lp1;
//   T lDist2  = lDiff.x*lDiff.x + lDiff.y*lDiff.y;
//   T lD      = lp1.x*lp2.y - lp2.x*lp1.y;
//   T discrim = cRad*cRad*lDist_2 - lD*lD;Q
//   if(discrim <= 0) { return 0.0f; }
// }



//// ADD SIGNAL ////

// add field containing signals
template<typename T> __global__ void addSignal_k(EMField<T> signal, EMField<T> dst, FieldParams<T> cp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      int i = dst.idx(ix, iy, iz);
      dst.Q[i]   += signal.Q[i];   // * cp.u.dt;  // NOTE: should be multiplied by FPS, not dt, for FPS invariance (?)
      dst.QPV[i] += signal.QPV[i]; // * cp.u.dt;  //      --> (dL?)
      dst.QNV[i] += signal.QNV[i]; // * cp.u.dt;
      dst.E[i]   += signal.E[i];   // * cp.u.dt;
      dst.B[i]   += signal.B[i];   // * cp.u.dt;
    }
}

// draw in signals based on pen location and parameters
template<typename T> __global__ void addSignal_k(typename DimType<T, 3>::VECTOR_T pSrc, EMField<T> dst, SignalPen<T> pen, FieldParams<T> cp)
{
  typedef typename DimType<T, 2>::VECTOR_T VT2;
  typedef typename DimType<T, 3>::VECTOR_T VT3;
  long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long iy = blockIdx.y*blockDim.y + threadIdx.y;
  long iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      unsigned long long i = dst.idx(ix, iy, iz);
      VT3 pCell = VT3{(T)ix,(T)iy,(T)iz} + VT3{0.5, 0.5, 0.5};
      if(pen.cellAlign) { pCell = floor(pCell); pSrc = floor(pSrc); }
      
      VT3 diff  = (pCell - pSrc);// + (pen.cellAlign ? 1.0f : 0.0f);
      T   dist2 = dot(diff, diff);

      T rad = pen.radius;///sum(cp.u.dL/3);
      //T rMax = pen.radius + 1/(T)sqrt(2.0); // circle radius plus maximum possible intersection radius from cell (center to corner)
      //if(dist2 <= rMax*rMax)
      if((!pen.square && dist2 <= rad*rad) || (pen.square && abs(diff.x) <= rad+0.5f &&
                                                             abs(diff.y) <= rad+0.5f &&
                                               abs(diff.z) <= rad+0.5f))
        {
          T dist = sqrt(dist2);
          VT3 n  = (dist == 0 ? diff : diff/dist);

          T rMult   = (dist  != 0.0f ? 1.0f / dist  : 1.0f);
          T r2Mult  = (dist2 != 0.0f ? 1.0f / dist2 : 1.0f);
          T cosMult = cos(2.0f*M_PI*pen.frequency*cp.t);
          T sinMult = sin(2.0f*M_PI*pen.frequency*cp.t);
          T tMult   = atan2(n.y, n.x);

          T   QoptMult   =   pen.mult*((pen.Qopt   & IDX_R   ? rMult   : 1)*(pen.Qopt   & IDX_R2  ? r2Mult  : 1)*(pen.Qopt   & IDX_T ? tMult : 1) *
                                       (pen.Qopt   & IDX_COS ? cosMult : 1)*(pen.Qopt   & IDX_SIN ? sinMult : 1));
          VT3 QPVoptMult = n*pen.mult*((pen.QPVopt & IDX_R   ? rMult   : 1)*(pen.QPVopt & IDX_R2  ? r2Mult  : 1)*(pen.QPVopt & IDX_T ? tMult : 1) *
                                       (pen.QPVopt & IDX_COS ? cosMult : 1)*(pen.QPVopt & IDX_SIN ? sinMult : 1));
          VT3 QNVoptMult = n*pen.mult*((pen.QNVopt & IDX_R   ? rMult   : 1)*(pen.QNVopt & IDX_R2  ? r2Mult  : 1)*(pen.QNVopt & IDX_T ? tMult : 1) *
                                       (pen.QNVopt & IDX_COS ? cosMult : 1)*(pen.QNVopt & IDX_SIN ? sinMult : 1));
          T   EoptMult   =   pen.mult*((pen.Eopt   & IDX_R   ? rMult   : 1)*(pen.Eopt   & IDX_R2  ? r2Mult  : 1)*(pen.Eopt   & IDX_T ? tMult : 1) *
                                       (pen.Eopt   & IDX_COS ? cosMult : 1)*(pen.Eopt   & IDX_SIN ? sinMult : 1));
          T   BoptMult   =   pen.mult*((pen.Bopt   & IDX_R   ? rMult   : 1)*(pen.Bopt   & IDX_R2  ? r2Mult  : 1)*(pen.Bopt   & IDX_T ? tMult : 1) *
                                       (pen.Bopt   & IDX_COS ? cosMult : 1)*(pen.Bopt   & IDX_SIN ? sinMult : 1));

          dst.Q  [i] += pen.Qmult   * QoptMult;                 // * cp.u.dt;  // NOTE: should be multiplied by FPS, not dt, for FPS invariance (?)
          dst.QPV[i] += pen.QPVmult * QPVoptMult; // / cp.u.dL; // * cp.u.dt;  //      --> (dL?)
          dst.QNV[i] += pen.QNVmult * QNVoptMult; // / cp.u.dL; // * cp.u.dt;
          dst.E  [i] += pen.Emult   * EoptMult;   // / cp.u.dL; // * cp.u.dt;
          dst.B  [i] += pen.Bmult   * BoptMult;   // / cp.u.dL; // * cp.u.dt;
        }
    }
}

// wrappers
template<typename T> void addSignal(EMField<T> &signal, EMField<T> &dst, const FieldParams<T> &cp)
{
  if(dst.size.x > 0 && dst.size.y > 0 && signal.size == dst.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      addSignal_k<<<grid, threads>>>(signal, dst, cp);
      cudaDeviceSynchronize(); getLastCudaError("====> ERROR: addSignal_k failed!");
    }
  else { std::cout << "==> WARNING: Skipped addSignal(srcField) (" << signal.size << " / " << dst.size << ")\n"; }
}
template<typename T> void addSignal(const typename DimType<T,3>::VECTOR_T &pSrc, EMField<T> &dst, const SignalPen<T> &pen, const FieldParams<T> &cp)
{
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      addSignal_k<<<grid, threads>>>(pSrc, dst, pen, cp);
      cudaDeviceSynchronize(); getLastCudaError("====> ERROR: addSignal_k failed!");
    }
  else { std::cout << "==> WARNING: Skipped addSignal(srcPoint) (" << dst.size << ")\n"; }
}

// template instantiation
template void addSignal<float>(EMField<float> &signal, EMField<float> &dst, const FieldParams<float> &cp);
template void addSignal<float>(const float3 &pSrc, EMField<float> &dst, const SignalPen<float> &pen, const FieldParams<float> &cp);

















//// ADD MATERIAL ////
template<typename T> __global__ void addMaterial_k(typename DimType<T, 3>::VECTOR_T pSrc, EMField<T> dst, MaterialPen<T> pen, FieldParams<T> cp)
{
  typedef typename DimType<T, 2>::VECTOR_T VT2;
  typedef typename DimType<T, 3>::VECTOR_T VT3;
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      int i = dst.idx(ix, iy, iz);
      //T rMax = pen.radius + 1/(T)sqrt(2.0); // circle radius plus maximum possible intersection radius from cell (center to corner)

      //T   rad = pen.radius;// / sum(cp.cu.dL/3);
      VT3 pCell = VT3{(T)ix+0.5f, (T)iy+0.5f, (T)iz+0.5f};
      VT3 diff  = pCell - pSrc;
      T   dist2 = dot(diff, diff);
      if((!pen.square && dist2 <= pen.radius*pen.radius) || (pen.square && (abs(diff.x) <= pen.radius+0.5f &&
                                                                            abs(diff.y) <= pen.radius+0.5f &&
                                                                            abs(diff.z) <= pen.radius+0.5f)))
        { dst.mat[i] = Material<T>(pen.mult*pen.permittivity, pen.mult*pen.permeability, pen.mult*pen.conductivity, pen.vacuum); }
    }
}

// wrapper functions
template<typename T>
void addMaterial(const typename DimType<T,3>::VECTOR_T &pSrc, EMField<T> &dst, const MaterialPen<T> &pen, const FieldParams<T> &cp)
{
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      addMaterial_k<<<grid, threads>>>(pSrc, dst, pen, cp);
      cudaDeviceSynchronize(); getLastCudaError("====> ERROR: addMaterial_k failed!");
    }
  else { std::cout << "==> WARNING: Skipped addMaterial(srcPoint) (" << dst.size << ")\n"; }
}

// template instantiation
template void addMaterial<float>(const float3 &pSrc, EMField<float> &dst, const MaterialPen<float> &pen, const FieldParams<float> &cp);

