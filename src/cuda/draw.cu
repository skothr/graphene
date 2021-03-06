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
__global__ void addSignal_k(Field<T> signal, Field<T> dst, FieldParams<T> cp, T mult=1.0)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      int i = dst.idx(ix, iy, iz);
      dst[i] += signal[i]*cp.u.dt*mult;
    }
}
template<typename T>
__global__ void addSignal_k(Field<typename DimType<T, 3>::VEC_T> signal, Field<typename DimType<T, 3>::VEC_T> dst, FieldParams<T> cp, T mult=1.0)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      int i = dst.idx(ix, iy, iz);
      dst[i] += signal[i]*cp.u.dt*mult;
    }
}

// add field containing signals
template<typename T>
__global__ void addSignal_k(EMField<T> signal, EMField<T> dst, FieldParams<T> cp, T mult=1.0)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      int i = dst.idx(ix, iy, iz);
      dst.Qn[i] += signal.Qn[i] * cp.u.dt*mult;
      dst.Qp[i] += signal.Qp[i] * cp.u.dt*mult;
      dst.Qv[i] += signal.Qv[i] * cp.u.dt*mult;
      dst.E[i]  += signal.E[i]  * cp.u.dt*mult;
      dst.B[i]  += signal.B[i]  * cp.u.dt*mult;
    }
}
// add field containing signals
template<typename T>
__global__ void addSignal_k(FluidField<T> signal, FluidField<T> dst, FluidParams<T> cp, T mult=1.0)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      int i = dst.idx(ix, iy, iz);
      dst.v[i]   += signal.v[i]   * cp.u.dt*mult;
      dst.p[i]   += signal.p[i]   * cp.u.dt*mult;
      dst.div[i] += signal.div[i] * cp.u.dt*mult;
      dst.Qn[i]  += signal.Qn[i]  * cp.u.dt*mult;
      dst.Qp[i]  += signal.Qp[i]  * cp.u.dt*mult;
      dst.Qv[i]  += signal.Qv[i]  * cp.u.dt*mult;
      dst.E[i]   += signal.E[i]   * cp.u.dt*mult;
      dst.B[i]   += signal.B[i]   * cp.u.dt*mult;
    }
}


////////////////////////////////////////////////////////////////
// NOTE: figure out why X appears to negative (...?)
////////////////////////////////////////////////////////////////

// draw in signals based on pen location and parameters
template<typename T>
__global__ void addSignal_k(typename DimType<T, 3>::VEC_T mpos, EMField<T> dst, SignalPen<T> pen, FieldParams<T> cp)
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
      if(overlap > 0.0f) //penOverlaps(pCell, mpos, diff, dist2, &pen, cp, 0.0f))
        {
          T dist2Mag = length(dist2);   dist2Mag = (dist2Mag == 0.0f || isnan(dist2Mag)) ? 1.0f : dist2Mag;
          T distMag  = sqrt(dist2Mag);  distMag  = (distMag  == 0.0f || isnan(distMag))  ? 1.0f : distMag;
          VT3 n      = normalize(diff); n        = ((isnan(n) || isinf(n)) ? VT3{1.0f, 1.0f, 1.0f} : n);

          T rMult   = (distMag > 0.0f ? 1.0f/distMag : 1.0f);
          T r2Mult  = (dist2Mag >= 1.0f ? 1.0f/dist2Mag : 1.0f);
          T cosMult = cos(2.0f*M_PI*pen.frequency*(cp.t-pen.startTime));
          T sinMult = sin(2.0f*M_PI*pen.frequency*(cp.t-pen.startTime));
          T tMult   = atan2(n.y, n.x);
          
          T speed = (pen.speed ? max(1.0f, pen.speedMult*length(pen.mouseSpeed)) : 1.0f);

          // TODO: gaussian multiplier?
          T gaussMult = 1.0f;//exp(-dist2Mag/(2*pen->radius0*pen->radius0));
          
          T QnOptMult = pen.mult*((pen.Qnopt & IDX_R   ? rMult   : 1)*(pen.Qnopt & IDX_R2  ? r2Mult  : 1)*(pen.Qnopt & IDX_T ? tMult : 1) *
                                  (pen.Qnopt & IDX_COS ? cosMult : 1)*(pen.Qnopt & IDX_SIN ? sinMult : 1));
          T QpOptMult = pen.mult*((pen.Qpopt & IDX_R   ? rMult   : 1)*(pen.Qpopt & IDX_R2  ? r2Mult  : 1)*(pen.Qpopt & IDX_T ? tMult : 1) *
                                  (pen.Qpopt & IDX_COS ? cosMult : 1)*(pen.Qpopt & IDX_SIN ? sinMult : 1));
          T QvOptMult = pen.mult*((pen.Qvopt & IDX_R   ? rMult   : 1)*(pen.Qvopt & IDX_R2  ? r2Mult  : 1)*(pen.Qvopt & IDX_T ? tMult : 1) *
                                  (pen.Qvopt & IDX_COS ? cosMult : 1)*(pen.Qvopt & IDX_SIN ? sinMult : 1));
          T EoptMult  = pen.mult*((pen.Eopt  & IDX_R   ? rMult   : 1)*(pen.Eopt  & IDX_R2  ? r2Mult  : 1)*(pen.Eopt  & IDX_T ? tMult : 1) *
                                  (pen.Eopt  & IDX_COS ? cosMult : 1)*(pen.Eopt  & IDX_SIN ? sinMult : 1));
          T BoptMult  = pen.mult*((pen.Bopt  & IDX_R   ? rMult   : 1)*(pen.Bopt  & IDX_R2  ? r2Mult  : 1)*(pen.Bopt  & IDX_T ? tMult : 1) *
                                  (pen.Bopt  & IDX_COS ? cosMult : 1)*(pen.Bopt  & IDX_SIN ? sinMult : 1));

          // n.x *= -1; // ?
          unsigned long i = dst.idx(ix, iy, iz);
          dst.Qn[i] += speed * pen.Qnmult * QnOptMult * gaussMult * cp.u.dt * overlap;
          dst.Qp[i] += speed * pen.Qpmult * QpOptMult * gaussMult * cp.u.dt * overlap;
          dst.Qv[i] += (pen.radial ? n : VT3{1,1,1}) * speed * pen.Qvmult * QvOptMult * gaussMult * cp.u.dt * overlap;
          dst.E[i]  += (pen.radial ? n : VT3{1,1,1}) * speed * pen.Emult  * EoptMult  * gaussMult * cp.u.dt * overlap;
          dst.B[i]  += (pen.radial ? n : VT3{1,1,1}) * speed * pen.Bmult  * BoptMult  * gaussMult * cp.u.dt * overlap;
        }
    }
}
// draw in signals based on pen location and parameters
template<typename T>
__global__ void addSignal_k(typename DimType<T, 3>::VEC_T mpos, FluidField<T> dst, SignalPen<T> pen, FluidParams<T> cp)
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
          T dist2Mag = length(dist2);  dist2Mag = (dist2Mag == 0.0f || isnan(dist2Mag)) ? 1.0f : dist2Mag;
          T distMag  = sqrt(dist2Mag);  distMag = (distMag  == 0.0f || isnan(distMag))  ? 1.0f : distMag;
          VT3 n = normalize(diff);            n = ((isnan(n) || isinf(n)) ? VT3{1.0f, 1.0f, 1.0f} : n);

          T rMult   = (distMag > 0.0f ? 1.0f/distMag : 1.0f);
          T r2Mult  = (dist2Mag >= 1.0f ? 1.0f/dist2Mag : 1.0f);
          T cosMult = cos(2.0f*M_PI*pen.frequency*(cp.t-pen.startTime));
          T sinMult = sin(2.0f*M_PI*pen.frequency*(cp.t-pen.startTime));
          T tMult   = atan2(n.y, n.x);
          
          T speed = (pen.speed ? max(1.0f, pen.speedMult*length(pen.mouseSpeed)) : 1.0f);

          // TODO: gaussian multiplier?
          T gaussMult = 1.0f;//exp(-dist2Mag/(2*pen->radius0*pen->radius0));
          
          T VOptMult  = pen.mult*((pen.Vopt  & IDX_R   ? rMult   : 1)*(pen.Vopt  & IDX_R2  ? r2Mult  : 1)*(pen.Vopt  & IDX_T ? tMult : 1) *
                                  (pen.Vopt  & IDX_COS ? cosMult : 1)*(pen.Vopt  & IDX_SIN ? sinMult : 1));
          T POptMult  = pen.mult*((pen.Popt  & IDX_R   ? rMult   : 1)*(pen.Popt  & IDX_R2  ? r2Mult  : 1)*(pen.Popt  & IDX_T ? tMult : 1) *
                                  (pen.Popt  & IDX_COS ? cosMult : 1)*(pen.Popt  & IDX_SIN ? sinMult : 1));
          T QnOptMult = pen.mult*((pen.Qnopt & IDX_R   ? rMult   : 1)*(pen.Qnopt & IDX_R2  ? r2Mult  : 1)*(pen.Qnopt & IDX_T ? tMult : 1) *
                                  (pen.Qnopt & IDX_COS ? cosMult : 1)*(pen.Qnopt & IDX_SIN ? sinMult : 1));
          T QpOptMult = pen.mult*((pen.Qpopt & IDX_R   ? rMult   : 1)*(pen.Qpopt & IDX_R2  ? r2Mult  : 1)*(pen.Qpopt & IDX_T ? tMult : 1) *
                                  (pen.Qpopt & IDX_COS ? cosMult : 1)*(pen.Qpopt & IDX_SIN ? sinMult : 1));
          T QvOptMult = pen.mult*((pen.Qvopt & IDX_R   ? rMult   : 1)*(pen.Qvopt & IDX_R2  ? r2Mult  : 1)*(pen.Qvopt & IDX_T ? tMult : 1) *
                                  (pen.Qvopt & IDX_COS ? cosMult : 1)*(pen.Qvopt & IDX_SIN ? sinMult : 1));
          T EoptMult  = pen.mult*((pen.Eopt  & IDX_R   ? rMult   : 1)*(pen.Eopt  & IDX_R2  ? r2Mult  : 1)*(pen.Eopt  & IDX_T ? tMult : 1) *
                                  (pen.Eopt  & IDX_COS ? cosMult : 1)*(pen.Eopt  & IDX_SIN ? sinMult : 1));
          T BoptMult  = pen.mult*((pen.Bopt  & IDX_R   ? rMult   : 1)*(pen.Bopt  & IDX_R2  ? r2Mult  : 1)*(pen.Bopt  & IDX_T ? tMult : 1) *
                                  (pen.Bopt  & IDX_COS ? cosMult : 1)*(pen.Bopt  & IDX_SIN ? sinMult : 1));

          // n.x *= -1; // ?
          unsigned long i = dst.idx(ix, iy, iz);
          dst.v[i]  += (pen.radial ? n : VT3{1,1,1})  * speed * pen.Vmult  * VOptMult  * gaussMult * cp.u.dt * overlap;
          dst.p[i]  += speed * pen.Pmult  * POptMult  * gaussMult * cp.u.dt * overlap;
          dst.Qn[i] += speed * pen.Qnmult * QnOptMult * gaussMult * cp.u.dt * overlap;
          dst.Qp[i] += speed * pen.Qpmult * QpOptMult * gaussMult * cp.u.dt * overlap;
          dst.Qv[i] += (pen.radial ? n : VT3{1,1,1}) * speed * pen.Qvmult * QvOptMult * gaussMult * cp.u.dt * overlap;
          dst.E[i]  += (pen.radial ? n : VT3{1,1,1}) * speed * pen.Emult  * EoptMult  * gaussMult * cp.u.dt * overlap;
          dst.B[i]  += (pen.radial ? n : VT3{1,1,1}) * speed * pen.Bmult  * BoptMult  * gaussMult * cp.u.dt * overlap;
        }
    }
}
// draw in signals based on pen location and parameters
template<typename T>
__global__ void addSignal_k(typename DimType<T, 3>::VEC_T mpos,
                            Field<typename DimType<T, 3>::VEC_T> dstV, Field<T> dstP,
                            Field<T> dstQn, Field<T> dstQp,            Field<typename DimType<T, 3>::VEC_T> dstQv,
                            Field<typename DimType<T, 3>::VEC_T> dstE, Field<typename DimType<T, 3>::VEC_T> dstB,
                            SignalPen<T> pen, FluidParams<T> cp)
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
      T overlap = penOverlap3(pCell, mpos, diff, dist2, &pen, cp, 0.0f);
      if(overlap > 0.0f) //if(penOverlaps(pCell, mpos, diff, dist2, &pen, cp, 0.0f))
        {
          T dist2Mag = length(dist2);  dist2Mag = (dist2Mag == 0.0f || isnan(dist2Mag)) ? 1.0f : dist2Mag;
          T distMag  = sqrt(dist2Mag); distMag  = (distMag  == 0.0f || isnan(distMag))  ? 1.0f : distMag;
          VT3 n = normalize(diff);     n        = ((isnan(n) || isinf(n)) ? VT3{1.0f, 1.0f, 1.0f} : n);

          T rMult   = (distMag > 0.0f ? 1.0f/distMag : 1.0f);
          T r2Mult  = (dist2Mag >= 1.0f ? 1.0f/dist2Mag : 1.0f);
          T cosMult = cos(2.0f*M_PI*pen.frequency*(cp.t-pen.startTime));
          T sinMult = sin(2.0f*M_PI*pen.frequency*(cp.t-pen.startTime));
          T tMult   = atan2(n.y, n.x);

          T speed = (pen.speedMult ? max(1.0f, pen.speedMult*length(pen.mouseSpeed)) : 1.0f);
          //VT3 mvec  = ((pen.speed && speed >= 1.0f) ? pen.mouseSpeed : VT3{1.0f, 1.0f, 1.0f});

          // TODO: gaussian multiplier?
          T gaussMult = 1.0f;//exp(-dist2Mag/(2*pen->radius0*pen->radius0));
          
          T VOptMult  = pen.mult*((pen.Vopt  & IDX_R   ? rMult   : 1)*(pen.Vopt  & IDX_R2  ? r2Mult  : 1)*(pen.Vopt  & IDX_T ? tMult : 1) *
                                  (pen.Vopt  & IDX_COS ? cosMult : 1)*(pen.Vopt  & IDX_SIN ? sinMult : 1));
          T POptMult  = pen.mult*((pen.Popt  & IDX_R   ? rMult   : 1)*(pen.Popt  & IDX_R2  ? r2Mult  : 1)*(pen.Popt  & IDX_T ? tMult : 1) *
                                  (pen.Popt  & IDX_COS ? cosMult : 1)*(pen.Popt  & IDX_SIN ? sinMult : 1));
          T QnOptMult = pen.mult*((pen.Qnopt & IDX_R   ? rMult   : 1)*(pen.Qnopt & IDX_R2  ? r2Mult  : 1)*(pen.Qnopt & IDX_T ? tMult : 1) *
                                  (pen.Qnopt & IDX_COS ? cosMult : 1)*(pen.Qnopt & IDX_SIN ? sinMult : 1));
          T QpOptMult = pen.mult*((pen.Qpopt & IDX_R   ? rMult   : 1)*(pen.Qpopt & IDX_R2  ? r2Mult  : 1)*(pen.Qpopt & IDX_T ? tMult : 1) *
                                  (pen.Qpopt & IDX_COS ? cosMult : 1)*(pen.Qpopt & IDX_SIN ? sinMult : 1));
          T QvOptMult = pen.mult*((pen.Qvopt & IDX_R   ? rMult   : 1)*(pen.Qvopt & IDX_R2  ? r2Mult  : 1)*(pen.Qvopt & IDX_T ? tMult : 1) *
                                  (pen.Qvopt & IDX_COS ? cosMult : 1)*(pen.Qvopt & IDX_SIN ? sinMult : 1));
          T EoptMult  = pen.mult*((pen.Eopt  & IDX_R   ? rMult   : 1)*(pen.Eopt  & IDX_R2  ? r2Mult  : 1)*(pen.Eopt  & IDX_T ? tMult : 1) *
                                  (pen.Eopt  & IDX_COS ? cosMult : 1)*(pen.Eopt  & IDX_SIN ? sinMult : 1));
          T BoptMult  = pen.mult*((pen.Bopt  & IDX_R   ? rMult   : 1)*(pen.Bopt  & IDX_R2  ? r2Mult  : 1)*(pen.Bopt  & IDX_T ? tMult : 1) *
                                  (pen.Bopt  & IDX_COS ? cosMult : 1)*(pen.Bopt  & IDX_SIN ? sinMult : 1));

          // n.x *= -1; // ?
          unsigned long i = dstE.idx(ix, iy, iz);
          dstV[i]  += (pen.radial ? n : VT3{1,1,1})  * speed * pen.Vmult  * VOptMult  * gaussMult * cp.u.dt * overlap;
          dstP[i]  += speed * pen.Pmult  * POptMult  * gaussMult * cp.u.dt * overlap;
          dstQn[i] += speed * pen.Qnmult * QnOptMult * gaussMult * cp.u.dt * overlap;
          dstQp[i] += speed * pen.Qpmult * QpOptMult * gaussMult * cp.u.dt * overlap;
          dstQv[i] += (pen.radial ? n : VT3{1,1,1})  * speed * pen.Qvmult * QvOptMult * gaussMult * cp.u.dt * overlap;
          dstE[i]  += (pen.radial ? n : VT3{1,1,1})  * speed * pen.Emult  * EoptMult  * gaussMult * cp.u.dt * overlap;
          dstB[i]  += (pen.radial ? n : VT3{1,1,1})  * speed * pen.Bmult  * BoptMult  * gaussMult * cp.u.dt * overlap;
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
                                    const SignalPen<T> &pen, const FluidParams<T> &cp)
{
  if(dstE.size.x > 0 && dstE.size.y > 0 && dstE.size.z > 0 &&
     dstB.size.x > 0 && dstB.size.y > 0 && dstB.size.z > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(dstE.size.x/(float)BLOCKDIM_X),
                (int)ceil(dstE.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dstE.size.z/(float)BLOCKDIM_Z));
      addSignal_k<<<grid, threads>>>(mpos, dstV, dstP, dstQn, dstQp, dstQv, dstE, dstB, pen, cp);
    }
  else { std::cout << "==> WARNING: Skipped addSignal(source point Field<VT3>) (E: " << dstE.size << " / B: " << dstB.size << ")\n"; }
}
// EMField
template<typename T> void addSignal(const typename DimType<T, 3>::VEC_T &mpos, EMField<T> &dst,
                                    const SignalPen<T> &pen, const FieldParams<T> &cp)
{
  if(dst.size.x > 0 && dst.size.y > 0 && dst.size.z > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      addSignal_k<<<grid, threads>>>(mpos, dst, pen, cp);
    }
  else { std::cout << "==> WARNING: Skipped addSignal(source point) (" << dst.size << ")\n"; }
}
// FluidField
template<typename T> void addSignal(const typename DimType<T, 3>::VEC_T &mpos, FluidField<T> &dst,
                                    const SignalPen<T> &pen, const FluidParams<T> &cp)
{
  if(dst.size.x > 0 && dst.size.y > 0 && dst.size.z > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      addSignal_k<<<grid, threads>>>(mpos, dst, pen, cp);
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
                               Field<float>  &dstQn, Field<float>  &dstQp, Field<float3> &dstQv,
                               Field<float3> &dstE,  Field<float3> &dstB,
                               const SignalPen<float> &pen, const FluidParams<float> &cp);
template void addSignal<float>(const float3 &mpos, EMField<float> &dst,
                               const SignalPen<float> &pen, const FieldParams<float> &cp);
template void addSignal<float>(const float3 &mpos, FluidField<float> &dst,
                               const SignalPen<float> &pen, const FluidParams<float> &cp);





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
          dst.mat[i] = Material<T>(pen.mult*pen.material.permittivity,
                                   pen.mult*pen.material.permeability,
                                   pen.mult*pen.material.conductivity,
                                   pen.material.vacuum());
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

