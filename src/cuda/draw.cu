#include "draw.cuh"

#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>

#include "fluid.cuh"
#include "physics.h"
#include "raytrace.cuh"
#include "vector-operators.h"
#include "cuda-tools.cuh"
#include "mathParser.hpp"

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8
#define BLOCKDIM_Z 8


//// ADD SIGNAL ////

// add signal to field 
template<typename T>
__global__ void addSignal_k(Field<T> signal, Field<T> dst, FieldParams<T> cp, T mult)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      int i = dst.idx(ix, iy, iz);
      dst[i] += signal[i]*mult;
    }
}
template<typename T>
__global__ void addSignal_k(Field<typename DimType<T, 3>::VEC_T> signal, Field<typename DimType<T, 3>::VEC_T> dst, FieldParams<T> cp, T mult)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      int i = dst.idx(ix, iy, iz);
      dst[i] += signal[i]*mult;
    }
}

// add field containing signals
template<typename T>
__global__ void addSignal_k(EMField<T> signal, EMField<T> dst, FieldParams<T> cp, T mult)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      int i = dst.idx(ix, iy, iz);
      dst.Qn[i] += signal.Qn[i] * mult;
      dst.Qp[i] += signal.Qp[i] * mult;
      dst.Qv[i] += signal.Qv[i] * mult;
      dst.E[i]  += signal.E[i]  * mult;
      dst.B[i]  += signal.B[i]  * mult;
    }
}
// add field containing signals
template<typename T>
__global__ void addSignal_k(FluidField<T> signal, FluidField<T> dst, FluidParams<T> cp, T mult)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      int i = dst.idx(ix, iy, iz);
      dst.v[i]   += signal.v[i]   * mult;
      dst.p[i]   += signal.p[i]   * mult;
      dst.div[i] += signal.div[i] * mult;
      dst.Qn[i]  += signal.Qn[i]  * mult;
      dst.Qp[i]  += signal.Qp[i]  * mult;
      dst.Qv[i]  += signal.Qv[i]  * mult;
      dst.E[i]   += signal.E[i]   * mult;
      dst.B[i]   += signal.B[i]   * mult;
    }
}


// draw in signals based on pen location and parameters
template<typename T>
__global__ void addSignal_k(typename DimType<T, 3>::VEC_T mpos, EMField<T> dst, SignalPen<T> pen, FieldParams<T> cp, T mult)
{
  typedef typename DimType<T, 3>::VEC_T VT3;
  unsigned long ix = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned long iy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned long iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      VT3 pCell = VT3{(T)ix+0.5f, (T)iy+0.5f, (T)iz+0.5f};
      VT3 diff; VT3 dist2; // output by penOverlaps()

      // if(penOverlaps(pCell, mpos, diff, dist2, &pen, cp, 0.0f))
      T overlap = penOverlap3(pCell, mpos, diff, dist2, &pen, cp, 0.0f);
      if(overlap > 0.0f)
        {
          T dist2Mag = length(dist2);  dist2Mag = (dist2Mag == 0.0f || isnan(dist2Mag)) ? 1.0f : dist2Mag;
          T distMag  = sqrt(dist2Mag);  distMag = (distMag  == 0.0f || isnan(distMag))  ? 1.0f : distMag;
          VT3 n      = normalize(diff);       n = ((isnan(n) || isinf(n)) ? VT3{1.0f, 1.0f, 1.0f} : n);
          
          VT3 radialMult = (pen.radial ? n : VT3{1,1,1});
          T   gaussMult  = exp(-dist2Mag/(dot(pen.radius0,pen.radius0)*2.0)); // TODO: gaussian multiplier?
          
          T rMult   = (distMag > 0.0f ? 1.0f/distMag : 1.0f);
          T r2Mult  = (dist2Mag >= 1.0f ? 1.0f/dist2Mag : 1.0f);
          T cosMult = cos(2.0f*M_PI*pen.frequency*(cp.t-pen.startTime));
          T sinMult = sin(2.0f*M_PI*pen.frequency*(cp.t-pen.startTime));
          T tMult   = atan2(n.y, n.x);
          
          T speed = (pen.speed ? max(1.0f, pen.speedMult*length(pen.mouseSpeed)) : 1.0f);
          // VT3 mvec  = ((pen.speed && speed >= 1.0f) ? pen.mouseSpeed : VT3{1.0f, 1.0f, 1.0f}); // TODO: mouse move direction

          T QnMult = pen.mult*((pen.pQn.multR   ? rMult   : 1)*(pen.pQn.multR_2 ? r2Mult  : 1)*(pen.pQn.multT ? tMult : 1) *
                               (pen.pQn.multCos ? cosMult : 1)*(pen.pQn.multSin ? sinMult : 1));
          T QpMult = pen.mult*((pen.pQp.multR   ? rMult   : 1)*(pen.pQp.multR_2 ? r2Mult  : 1)*(pen.pQp.multT ? tMult : 1) *
                               (pen.pQp.multCos ? cosMult : 1)*(pen.pQp.multSin ? sinMult : 1));
          T QvMult = pen.mult*((pen.pQv.multR   ? rMult   : 1)*(pen.pQv.multR_2 ? r2Mult  : 1)*(pen.pQv.multT ? tMult : 1) *
                               (pen.pQv.multCos ? cosMult : 1)*(pen.pQv.multSin ? sinMult : 1));
          T EMult  = pen.mult*((pen.pE.multR    ? rMult   : 1)*(pen.pE.multR_2  ? r2Mult  : 1)*(pen.pE.multT  ? tMult : 1) *
                               (pen.pE.multCos  ? cosMult : 1)*(pen.pE.multSin  ? sinMult : 1));
          T BMult  = pen.mult*((pen.pB.multR    ? rMult   : 1)*(pen.pB.multR_2  ? r2Mult  : 1)*(pen.pB.multT  ? tMult : 1) *
                               (pen.pB.multCos  ? cosMult : 1)*(pen.pB.multSin  ? sinMult : 1));

          unsigned long i = dst.idx(ix, iy, iz);
          dst.Qn[i] += speed * pen.pQn.base * QnMult * gaussMult * overlap * mult;
          dst.Qp[i] += speed * pen.pQp.base * QpMult * gaussMult * overlap * mult;
          dst.Qv[i] += speed * pen.pQv.base * QvMult * gaussMult * overlap * mult * radialMult;
          dst.E[i]  += speed * pen.pE.base  * EMult  * gaussMult * overlap * mult * radialMult;
          dst.B[i]  += speed * pen.pB.base  * BMult  * gaussMult * overlap * mult * radialMult;
        }
    }
}

// draw in signals based on pen location and parameters
template<typename T>
__global__ void addSignal_k(typename DimType<T, 3>::VEC_T mpos, FluidField<T> dst, SignalPen<T> pen, FluidParams<T> cp, T mult)
{
  typedef typename DimType<T, 3>::VEC_T VT3;
  unsigned long ix = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned long iy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned long iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      VT3 pCell = VT3{(T)ix+0.5f, (T)iy+0.5f, (T)iz+0.5f};
      VT3 diff; VT3 dist2; // output by penOverlaps()
      T overlap = penOverlap3(pCell, mpos, diff, dist2, &pen, cp, 0.0f);
      if(overlap > 0.0f) //if(penOverlaps(pCell, mpos, diff, dist2, &pen, cp, 0.0f))
        {
          T   dist2Mag = length(dist2);   dist2Mag = (dist2Mag == 0.0f || isnan(dist2Mag)) ? 1.0f : dist2Mag;
          T   distMag  = sqrt(dist2Mag);  distMag  = (distMag  == 0.0f || isnan(distMag))  ? 1.0f : distMag;
          VT3 n        = normalize(diff); n = ((isnan(n) || isinf(n)) ? VT3{1.0f, 1.0f, 1.0f} : n);
          
          VT3 radialMult = (pen.radial ? n : VT3{1,1,1});
          T   gaussMult  = exp(-dist2Mag/(dot(pen.radius0,pen.radius0)*2.0)); // TODO: gaussian?

          T rMult   = (distMag > 0.0f ? 1.0f/distMag : 1.0f);
          T r2Mult  = (dist2Mag >= 1.0f ? 1.0f/dist2Mag : 1.0f);
          T cosMult = cos(2.0f*M_PI*pen.frequency*(cp.t-pen.startTime));
          T sinMult = sin(2.0f*M_PI*pen.frequency*(cp.t-pen.startTime));
          T tMult   = atan2(n.y, n.x);
          
          T speed = (pen.speed ? max(1.0f, pen.speedMult*length(pen.mouseSpeed)) : 1.0f);
          // VT3 mvec  = ((pen.speed && speed >= 1.0f) ? pen.mouseSpeed : VT3{1.0f, 1.0f, 1.0f}); // TODO: mouse move direction
          
          T VMult  = pen.mult*((pen.pV.multR    ? rMult   : 1)*(pen.pV.multR_2  ? r2Mult  : 1)*(pen.pV.multT  ? tMult : 1) *
                               (pen.pV.multCos  ? cosMult : 1)*(pen.pV.multSin  ? sinMult : 1));
          T PMult  = pen.mult*((pen.pP.multR    ? rMult   : 1)*(pen.pP.multR_2  ? r2Mult  : 1)*(pen.pP.multT  ? tMult : 1) *
                               (pen.pP.multCos  ? cosMult : 1)*(pen.pP.multSin  ? sinMult : 1));
          T QnMult = pen.mult*((pen.pQn.multR   ? rMult   : 1)*(pen.pQn.multR_2 ? r2Mult  : 1)*(pen.pQn.multT ? tMult : 1) *
                               (pen.pQn.multCos ? cosMult : 1)*(pen.pQn.multSin ? sinMult : 1));
          T QpMult = pen.mult*((pen.pQp.multR   ? rMult   : 1)*(pen.pQp.multR_2 ? r2Mult  : 1)*(pen.pQp.multT ? tMult : 1) *
                               (pen.pQp.multCos ? cosMult : 1)*(pen.pQp.multSin ? sinMult : 1));
          T QvMult = pen.mult*((pen.pQv.multR   ? rMult   : 1)*(pen.pQv.multR_2 ? r2Mult  : 1)*(pen.pQv.multT ? tMult : 1) *
                               (pen.pQv.multCos ? cosMult : 1)*(pen.pQv.multSin ? sinMult : 1));
          T EMult  = pen.mult*((pen.pE.multR    ? rMult   : 1)*(pen.pE.multR_2  ? r2Mult  : 1)*(pen.pE.multT  ? tMult : 1) *
                               (pen.pE.multCos  ? cosMult : 1)*(pen.pE.multSin  ? sinMult : 1));
          T BMult  = pen.mult*((pen.pB.multR    ? rMult   : 1)*(pen.pB.multR_2  ? r2Mult  : 1)*(pen.pB.multT  ? tMult : 1) *
                               (pen.pB.multCos  ? cosMult : 1)*(pen.pB.multSin  ? sinMult : 1));
          
          unsigned long i = dst.idx(ix, iy, iz);
          dst.v[i]  += speed * pen.pV.base  * VMult  * gaussMult * overlap * mult * radialMult;
          dst.p[i]  += speed * pen.pP.base  * PMult  * gaussMult * overlap * mult;
          dst.Qn[i] += speed * pen.pQn.base * QnMult * gaussMult * overlap * mult;
          dst.Qp[i] += speed * pen.pQp.base * QpMult * gaussMult * overlap * mult;
          dst.Qv[i] += speed * pen.pQv.base * QvMult * gaussMult * overlap * mult * radialMult;
          dst.E[i]  += speed * pen.pE.base  * EMult  * gaussMult * overlap * mult * radialMult;
          dst.B[i]  += speed * pen.pB.base  * BMult  * gaussMult * overlap * mult * radialMult;
        }
    }
}

// draw in signals based on pen location and parameters
template<typename T>
__global__ void addSignal_k(typename DimType<T, 3>::VEC_T mpos,
                            Field<typename DimType<T, 3>::VEC_T> dstV, Field<T> dstP,
                            Field<T> dstQn, Field<T> dstQp,            Field<typename DimType<T, 3>::VEC_T> dstQv,
                            Field<typename DimType<T, 3>::VEC_T> dstE, Field<typename DimType<T, 3>::VEC_T> dstB,
                            SignalPen<T> pen, FluidParams<T> cp, T mult)
{
  typedef typename DimType<T, 3>::VEC_T VT3;
  unsigned long ix = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned long iy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned long iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dstE.size.x && iy < dstE.size.y && iz < dstE.size.z &&
     ix < dstB.size.x && iy < dstB.size.y && iz < dstB.size.z)
    {
      VT3 pCell = VT3{(T)ix+0.5f, (T)iy+0.5f, (T)iz+0.5f};
      VT3 diff; VT3 dist2; // output by penOverlaps()

      // if(penOverlaps(pCell, mpos, diff, dist2, &pen, cp, 0.0f))
      T overlap = penOverlap3(pCell, mpos, diff, dist2, &pen, cp, 0.0f);
      if(overlap > 0.0f)
        {
          T   dist2Mag = length(dist2);   dist2Mag = (dist2Mag == 0.0f || isnan(dist2Mag)) ? 1.0f : dist2Mag;
          T   distMag  = sqrt(dist2Mag);  distMag  = (distMag  == 0.0f || isnan(distMag))  ? 1.0f : distMag;
          VT3 n        = normalize(diff); n        = ((isnan(n) || isinf(n)) ? VT3{1.0f, 1.0f, 1.0f} : n);
          
          VT3 radialMult = (pen.radial ? n : VT3{1,1,1});
          T   gaussMult  = exp(-dist2Mag/(dot(pen.radius0,pen.radius0)*2.0)); // TODO: gaussian?

          T rMult   = (distMag > 0.0f ? 1.0f/distMag : 1.0f);
          T r2Mult  = (dist2Mag >= 1.0f ? 1.0f/dist2Mag : 1.0f);
          T cosMult = cos(2.0f*M_PI*pen.frequency*(cp.t-pen.startTime));
          T sinMult = sin(2.0f*M_PI*pen.frequency*(cp.t-pen.startTime));
          T tMult   = atan2(n.y, n.x);

          T speed = (pen.speed ? max(1.0f, pen.speedMult*length(pen.mouseSpeed)) : 1.0f);
          // VT3 mvec  = ((pen.speed && speed >= 1.0f) ? pen.mouseSpeed : VT3{1.0f, 1.0f, 1.0f}); // TODO: mouse move direction
          
          T VMult  = pen.mult*((pen.pV.multR    ? rMult   : 1)*(pen.pV.multR_2  ? r2Mult  : 1)*(pen.pV.multR_2  ? tMult : 1) *
                               (pen.pV.multCos  ? cosMult : 1)*(pen.pV.multSin  ? sinMult : 1));
          T PMult  = pen.mult*((pen.pP.multR    ? rMult   : 1)*(pen.pP.multR_2  ? r2Mult  : 1)*(pen.pP.multR_2 ? tMult : 1) *
                               (pen.pP.multCos  ? cosMult : 1)*(pen.pP.multSin  ? sinMult : 1));
          T QnMult = pen.mult*((pen.pQn.multR   ? rMult   : 1)*(pen.pQn.multR_2 ? r2Mult  : 1)*(pen.pQn.multT   ? tMult : 1) *
                               (pen.pQn.multCos ? cosMult : 1)*(pen.pQn.multSin ? sinMult : 1));
          T QpMult = pen.mult*((pen.pQp.multR   ? rMult   : 1)*(pen.pQp.multR_2 ? r2Mult  : 1)*(pen.pQp.multT   ? tMult : 1) *
                               (pen.pQp.multCos ? cosMult : 1)*(pen.pQp.multSin ? sinMult : 1));
          T QvMult = pen.mult*((pen.pQv.multR   ? rMult   : 1)*(pen.pQv.multR_2 ? r2Mult  : 1)*(pen.pQv.multT   ? tMult : 1) *
                               (pen.pQv.multCos ? cosMult : 1)*(pen.pQv.multSin ? sinMult : 1));
          T EMult  = pen.mult*((pen.pE.multR    ? rMult   : 1)*(pen.pE.multR_2  ? r2Mult  : 1)*(pen.pE.multR_2  ? tMult : 1) *
                               (pen.pE.multCos  ? cosMult : 1)*(pen.pE.multSin  ? sinMult : 1));
          T BMult  = pen.mult*((pen.pB.multR    ? rMult   : 1)*(pen.pB.multR_2  ? r2Mult  : 1)*(pen.pB.multR_2  ? tMult : 1) *
                               (pen.pB.multCos  ? cosMult : 1)*(pen.pB.multSin  ? sinMult : 1));
          
          unsigned long i = dstE.idx(ix, iy, iz);
          dstV[i]  += speed * pen.pV.base  * VMult  * gaussMult * overlap * mult * radialMult;
          dstP[i]  += speed * pen.pP.base  * PMult  * gaussMult * overlap * mult;
          dstQn[i] += speed * pen.pQn.base * QnMult * gaussMult * overlap * mult;
          dstQp[i] += speed * pen.pQp.base * QpMult * gaussMult * overlap * mult;
          dstQv[i] += speed * pen.pQv.base * QvMult * gaussMult * overlap * mult * radialMult;
          dstE[i]  += speed * pen.pE.base  * EMult  * gaussMult * overlap * mult * radialMult;
          dstB[i]  += speed * pen.pB.base  * BMult  * gaussMult * overlap * mult * radialMult;
        }
    }
}

// wrappers

// Field<VT3>
template<typename T> void addSignal(Field<typename DimType<T, 3>::VEC_T> &signal, Field<typename DimType<T, 3>::VEC_T> &dst, const FieldParams<T> &cp, T mult)
{
  if(dst.size.x > 0 && dst.size.y > 0 && dst.size.z > 0 && signal.size == dst.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      addSignal_k<<<grid, threads>>>(signal, dst, cp, mult);
    }
  else { std::cout << "==> WARNING: Skipped addSignal(source Field<VT3>) (" << signal.size << " / " << dst.size << ")\n"; }
}
template<typename T> void addSignal(Field<typename DimType<T, 3>::VEC_T> &signal, Field<typename DimType<T, 3>::VEC_T> &dst, const FluidParams<T> &cp, T mult)
{
  if(dst.size.x > 0 && dst.size.y > 0 && dst.size.z > 0 && signal.size == dst.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      addSignal_k<<<grid, threads>>>(signal, dst, FieldParams<T>(cp), mult);
    }
  else { std::cout << "==> WARNING: Skipped addSignal(source Field<VT3>) (" << signal.size << " / " << dst.size << ")\n"; }
}

// Field<T>
template<typename T> void addSignal(Field<T> &signal, Field<T> &dst, const FieldParams<T> &cp, T mult)
{
  if(dst.size.x > 0 && dst.size.y > 0 && dst.size.z > 0 && signal.size == dst.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      addSignal_k<<<grid, threads>>>(signal, dst, cp, mult);
    }
  else { std::cout << "==> WARNING: Skipped addSignal(source Field<VT3>) (" << signal.size << " / " << dst.size << ")\n"; }
}
template<typename T> void addSignal(Field<T> &signal, Field<T> &dst, const FluidParams<T> &cp, T mult)
{
  if(dst.size.x > 0 && dst.size.y > 0 && dst.size.z > 0 && signal.size == dst.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      addSignal_k<<<grid, threads>>>(signal, dst, FieldParams<T>(cp), mult);
    }
  else { std::cout << "==> WARNING: Skipped addSignal(source Field<VT3>) (" << signal.size << " / " << dst.size << ")\n"; }
}


// EMField
template<typename T> void addSignal(EMField<T> &signal, EMField<T> &dst, const FieldParams<T> &cp, T mult)
{
  if(dst.size.x > 0 && dst.size.y > 0 && dst.size.z > 0 && signal.size == dst.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      addSignal_k<<<grid, threads>>>(signal, dst, cp, mult);
    }
  else { std::cout << "==> WARNING: Skipped addSignal(source EMField) (" << signal.size << " / " << dst.size << ")\n"; }
}
// FluidField
template<typename T> void addSignal(FluidField<T> &signal, FluidField<T> &dst, const FluidParams<T> &cp, T mult)
{
  if(dst.size.x > 0 && dst.size.y > 0 && dst.size.z > 0 && signal.size == dst.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      addSignal_k<<<grid, threads>>>(signal, dst, cp, mult);
    }
  else { std::cout << "==> WARNING: Skipped addSignal(source FluidField) (" << signal.size << " / " << dst.size << ")\n"; }
}

// Field<VT3>
template<typename T> void addSignal(const typename DimType<T, 3>::VEC_T &mpos,
                                    Field<typename DimType<T, 3>::VEC_T> &dstV, Field<T> &dstP,
                                    Field<T> &dstQn, Field<T> &dstQp, Field<typename DimType<T, 3>::VEC_T> &dstQv,
                                    Field<typename DimType<T, 3>::VEC_T> &dstE, Field<typename DimType<T, 3>::VEC_T> &dstB,
                                    const SignalPen<T> &pen, const FluidParams<T> &cp, T mult)
{
  if(dstE.size.x > 0 && dstE.size.y > 0 && dstE.size.z > 0 &&
     dstB.size.x > 0 && dstB.size.y > 0 && dstB.size.z > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(dstE.size.x/(float)BLOCKDIM_X),
                (int)ceil(dstE.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dstE.size.z/(float)BLOCKDIM_Z));
      addSignal_k<<<grid, threads>>>(mpos, dstV, dstP, dstQn, dstQp, dstQv, dstE, dstB, pen, cp, mult);
    }
  else { std::cout << "==> WARNING: Skipped addSignal(source point Field<VT3>) (E: " << dstE.size << " / B: " << dstB.size << ")\n"; }
}

// EMField
template<typename T> void addSignal(const typename DimType<T, 3>::VEC_T &mpos, EMField<T> &dst,
                                    const SignalPen<T> &pen, const FieldParams<T> &cp, T mult)
{
  if(dst.size.x > 0 && dst.size.y > 0 && dst.size.z > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      addSignal_k<<<grid, threads>>>(mpos, dst, pen, cp, mult);
    }
  else { std::cout << "==> WARNING: Skipped addSignal(source point) (" << dst.size << ")\n"; }
}

// FluidField
template<typename T> void addSignal(const typename DimType<T, 3>::VEC_T &mpos, FluidField<T> &dst,
                                    const SignalPen<T> &pen, const FluidParams<T> &cp, T mult)
{
  if(dst.size.x > 0 && dst.size.y > 0 && dst.size.z > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      addSignal_k<<<grid, threads>>>(mpos, dst, pen, cp, mult);
    }
  else { std::cout << "==> WARNING: Skipped addSignal(source point) (" << dst.size << ")\n"; }
}

// template instantiation
template void addSignal<float>(Field<float>   &signal, Field<float>   &dst,  const FieldParams<float> &cp, float mult);
template void addSignal<float>(Field<float>   &signal, Field<float>   &dst,  const FluidParams<float> &cp, float mult);
template void addSignal<float>(Field<float3>  &signal, Field<float3>  &dst,  const FieldParams<float> &cp, float mult);
template void addSignal<float>(Field<float3>  &signal, Field<float3>  &dst,  const FluidParams<float> &cp, float mult);
template void addSignal<float>(EMField<float> &signal, EMField<float> &dst,  const FieldParams<float> &cp, float mult);
template void addSignal<float>(FluidField<float> &signal, FluidField<float> &dst,  const FluidParams<float> &cp, float mult);
template void addSignal<float>(const float3 &mpos,   Field<float3> &dstV,  Field<float>  &dstP,
                               Field<float>  &dstQn, Field<float>  &dstQp, Field<float3> &dstQv, Field<float3> &dstE,  Field<float3> &dstB,
                               const SignalPen<float> &pen, const FluidParams<float> &cp, float mult);
template void addSignal<float>(const float3 &mpos, EMField   <float> &dst, const SignalPen<float> &pen, const FieldParams<float> &cp, float mult);
template void addSignal<float>(const float3 &mpos, FluidField<float> &dst, const SignalPen<float> &pen, const FluidParams<float> &cp, float mult);





//// DECAY SIGNAL ////

// decay input signals (prevent stuck cells)
template<typename T>
__global__ void decaySignal_k(Field<T> src, FieldParams<T> cp)
{
  using VT3 = typename DimType<T, 3>::VEC_T;
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      int i = src.idx(ix, iy, iz);
      src[i] *= pow(cp.decay, cp.u.dt); // (decay)^(dt)
    }
}
template<typename T>
__global__ void decaySignal_k(Field<typename DimType<T, 3>::VEC_T> src, FieldParams<T> cp)
{
  using VT3 = typename DimType<T, 3>::VEC_T;
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      int i = src.idx(ix, iy, iz);
      src[i] *= pow(cp.decay, cp.u.dt); // (decay)^(dt)
    }
}

// wrapper
template<typename T> void decaySignal(Field<T> &src, FieldParams<T> &cp)
{
  if(src.size.x > 0 && src.size.y > 0 && src.size.z > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      decaySignal_k<<<grid, threads>>>(src, cp);
    }
  else { std::cout << "==> WARNING: Skipped decaySignal (" << src.size << ")\n"; }
}
template<typename T> void decaySignal(Field<typename DimType<T, 3>::VEC_T> &src, FieldParams<T> &cp)
{
  if(src.size.x > 0 && src.size.y > 0 && src.size.z > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      decaySignal_k<<<grid, threads>>>(src, cp);
    }
  else { std::cout << "==> WARNING: Skipped decaySignal (" << src.size << ")\n"; }
}

// template instantiation
template void decaySignal<float>(Field<float3> &src, FieldParams<float> &cp);
template void decaySignal<float>(Field<float>  &src, FieldParams<float> &cp);






//// ADD MATERIAL ////
template<typename T> __global__ void addMaterial_k(typename DimType<T, 3>::VEC_T mpos, EMField<T> dst, MaterialPen<T> pen, FieldParams<T> cp)
{
  typedef typename DimType<T, 3>::VEC_T VT3;
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      VT3 pCell = VT3{(T)ix+0.5f, (T)iy+0.5f, (T)iz+0.5f};
      VT3 diff; VT3 dist2; // output by penOverlaps()
      T overlap = penOverlap3(pCell, mpos, diff, dist2, &pen, cp, 0.0f);
      if(overlap > 0.0f) //if(penOverlaps(pCell, mpos, diff, dist2, &pen, cp, 0.0f))
        {
          int i = dst.idx(ix, iy, iz);
          dst.mat[i] = Material<T>(pen.mult*pen.mat.permittivity,
                                   pen.mult*pen.mat.permeability,
                                   pen.mult*pen.mat.conductivity,
                                   pen.mat.vacuum());
        }
    }
}

// wrapper functions
template<typename T>
void addMaterial(const typename DimType<T,3>::VEC_T &mpos, EMField<T> &dst, const MaterialPen<T> &pen, const FieldParams<T> &cp)
{
  if(dst.size.x > 0 && dst.size.y > 0 && dst.size.z > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      addMaterial_k<<<grid, threads>>>(mpos, dst, pen, cp);
    }
  else { std::cout << "==> WARNING: Skipped addMaterial(srcPoint) (" << dst.size << ")\n"; }
}



// template instantiation
template void addMaterial<float>(const float3 &mpos,     EMField<float> &dst, const MaterialPen<float> &pen, const FieldParams<float> &cp);

