#include "sim.cuh"
#include "sim.hpp"
using namespace grph;

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>

#include "states.h"
#include "params.h"
#include "fill.cuh"
#include "cuda-tools.cuh"
#include "physics.h"

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

#define FORWARD_EULER 1


////////////////////////////////////////////////////////////////////////////////////////////////
//// kernels
////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ bool slipPlane(typename Dims<T>::SIZE_T p, const SimParams<T> &params)
{
  typedef T VT;
  using     ST = typename Dims<T>::BASE;
  using     IT = typename Dims<T>::SIZE_T;

  return (((params.fp.edgePX == EDGE_NOSLIP || params.fp.edgePX == EDGE_SLIP) && p.x == params.fp.fluidSize.x-1) ||
          ((params.fp.edgeNX == EDGE_NOSLIP || params.fp.edgeNX == EDGE_SLIP) && p.x == 0) ||
          ((params.fp.edgePY == EDGE_NOSLIP || params.fp.edgePY == EDGE_SLIP) && p.y == params.fp.fluidSize.y-1) ||
          ((params.fp.edgeNY == EDGE_NOSLIP || params.fp.edgeNY == EDGE_SLIP) && p.y == 0));
}

template<typename T>
__global__ void advect_k(FluidState<T> src, FluidState<T> dst, SimParams<T> params)
{
  typedef T VT;
  using     ST = typename Dims<T>::BASE;
  using     IT = typename Dims<T>::SIZE_T;
  
  int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  IT  ip   = makeI<IT>(ix, iy, iz);
  IT  size = src.size;
  if(ip < size)
    {
      int i  = src.idx(ip);
      VT  p  = VT{ST(ix), ST(iy)};
      ST  dt = params.dt;

      CellState<T> s = src[i];
      
      if(slipPlane(ip, params))
        { s.v = makeV<VT>(0); } //s.qpv = makeV<VT>(0); s.qnv = makeV<VT>(0); } // zero velocity on edges (assumes wall is static)

      if(slipPlane(ip-IT{1,0}, params)) { s.v.x = 0; s.qpv.x =  abs(s.qpv.x); s.qnv.x =  abs(s.qnv.x); }
      if(slipPlane(ip+IT{1,0}, params)) { s.v.x = 0; s.qpv.x = -abs(s.qpv.x); s.qnv.x = -abs(s.qnv.x); }
      if(slipPlane(ip-IT{0,1}, params)) { s.v.y = 0; s.qpv.y =  abs(s.qpv.y); s.qnv.y =  abs(s.qnv.y); }
      if(slipPlane(ip+IT{0,1}, params)) { s.v.y = 0; s.qpv.y = -abs(s.qpv.y); s.qnv.y = -abs(s.qnv.y); }
      // if(slipPlane(ip-IT{1,0}, params)) { s.v.x = 0; s.qpv.x = 0.0f; s.qnv.x = 0.0f; }
      // if(slipPlane(ip+IT{1,0}, params)) { s.v.x = 0; s.qpv.x = 0.0f; s.qnv.x = 0.0f; }
      // if(slipPlane(ip-IT{0,1}, params)) { s.v.y = 0; s.qpv.y = 0.0f; s.qnv.y = 0.0f; }
      // if(slipPlane(ip+IT{0,1}, params)) { s.v.y = 0; s.qpv.y = 0.0f; s.qnv.y = 0.0f; }

      // apply charge force to velocity
      s.v += params.qvfMult*(s.qpv + s.qnv)*dt;
      
      // s.v.x   = (s.v.x   < 0 ? -1.0f : 1.0f)*min(abs(s.v.x),   16.0f);
      // s.v.y   = (s.v.y   < 0 ? -1.0f : 1.0f)*min(abs(s.v.y),   16.0f);
      // s.qpv.x = (s.qpv.x < 0 ? -1.0f : 1.0f)*min(abs(s.qpv.x), 16.0f);
      // s.qpv.y = (s.qpv.y < 0 ? -1.0f : 1.0f)*min(abs(s.qpv.y), 16.0f);
      // s.qnv.x = (s.qnv.x < 0 ? -1.0f : 1.0f)*min(abs(s.qnv.x), 16.0f);
      // s.qnv.y = (s.qnv.y < 0 ? -1.0f : 1.0f)*min(abs(s.qnv.y), 16.0f);

      // check for invalid values
      if(isnan(s.v.x)   || isinf(s.v.x))   { s.v.x   = 0.0; } if(isnan(s.v.y) || isinf(s.v.y))     { s.v.y  = 0.0; }
      if(isnan(s.d)     || isinf(s.d))     { s.d     = 0.0; }
      if(isnan(s.p)     || isinf(s.p))     { s.p     = 0.0; }
      if(isnan(s.div)   || isinf(s.div))   { s.div   = 0.0; }
      if(isnan(s.qn)    || isinf(s.qn))    { s.qn    = 0.0; }
      if(isnan(s.qp)    || isinf(s.qp))    { s.qp    = 0.0; }
      if(isnan(s.qnv.x) || isinf(s.qnv.x)) { s.qnv.x = 0.0; } if(isnan(s.qnv.y) || isinf(s.qnv.y)) { s.qnv.y = 0.0; }
      if(isnan(s.qpv.x) || isinf(s.qpv.x)) { s.qpv.x = 0.0; } if(isnan(s.qpv.y) || isinf(s.qpv.y)) { s.qpv.y = 0.0; }
      if(isnan(s.E.x)   || isinf(s.E.x))   { s.E.x   = 0.0; } if(isnan(s.E.y)   || isinf(s.E.y))   { s.E.y  = 0.0; }
      if(isnan(s.B.x)   || isinf(s.B.x))   { s.B.x   = 0.0; } if(isnan(s.B.y)   || isinf(s.B.y))   { s.B.y  = 0.0; }
      
      // use forward Euler method
      VT p2    = integrateForwardEuler(src.v, p, s.v, dt);
      // add actively to next point in texture
      int4   tiX   = texPutIX   (p2, params);
      int4   tiY   = texPutIY   (p2, params);
      float4 mults = texPutMults(p2);
      IT     p00   = IT{tiX.x, tiY.x}; IT p10 = IT{tiX.y, tiY.y};
      IT     p01   = IT{tiX.z, tiY.z}; IT p11 = IT{tiX.w, tiY.w};

      //__device__ void texAtomicAdd(float *tex, float val, int2 p, const SimSrc.Params<float2> &src.params);
      // scale value by grid overlap and store in each location
      // v
      texAtomicAdd(dst.v,   s.v*mults.x,   p00, params); texAtomicAdd(dst.v,   s.v*mults.z,   p01, params);
      texAtomicAdd(dst.v,   s.v*mults.y,   p10, params); texAtomicAdd(dst.v,   s.v*mults.w,   p11, params);
      // d                                 
      texAtomicAdd(dst.d,   s.d*mults.x,   p00, params); texAtomicAdd(dst.d,   s.d*mults.z,   p01, params);
      texAtomicAdd(dst.d,   s.d*mults.y,   p10, params); texAtomicAdd(dst.d,   s.d*mults.w,   p11, params);
      // p                                 
      texAtomicAdd(dst.p,   s.p*mults.x,   p00, params); texAtomicAdd(dst.p,   s.p*mults.z,   p01, params);
      texAtomicAdd(dst.p,   s.p*mults.y,   p10, params); texAtomicAdd(dst.p,   s.p*mults.w,   p11, params);
      // qn                                
      texAtomicAdd(dst.qn,  s.qn*mults.x,  p00, params); texAtomicAdd(dst.qn,  s.qn*mults.z,  p01, params);
      texAtomicAdd(dst.qn,  s.qn*mults.y,  p10, params); texAtomicAdd(dst.qn,  s.qn*mults.w,  p11, params);
      // qp                                
      texAtomicAdd(dst.qp,  s.qp*mults.x,  p00, params); texAtomicAdd(dst.qp,  s.qp*mults.z,  p01, params);
      texAtomicAdd(dst.qp,  s.qp*mults.y,  p10, params); texAtomicAdd(dst.qp,  s.qp*mults.w,  p11, params);
      // qnv
      texAtomicAdd(dst.qnv, s.qnv*mults.x, p00, params); texAtomicAdd(dst.qnv, s.qnv*mults.z, p01, params);
      texAtomicAdd(dst.qnv, s.qnv*mults.y, p10, params); texAtomicAdd(dst.qnv, s.qnv*mults.w, p11, params);
      // qpv
      texAtomicAdd(dst.qpv, s.qpv*mults.x, p00, params); texAtomicAdd(dst.qpv, s.qpv*mults.z, p01, params);
      texAtomicAdd(dst.qpv, s.qpv*mults.y, p10, params); texAtomicAdd(dst.qpv, s.qpv*mults.w, p11, params);
      // E
      texAtomicAdd(dst.E,   s.E*mults.x,   p00, params); texAtomicAdd(dst.E,   s.E*mults.z,   p01, params);
      texAtomicAdd(dst.E,   s.E*mults.y,   p10, params); texAtomicAdd(dst.E,   s.E*mults.w,   p11, params);
      // B
      texAtomicAdd(dst.B,   s.B*mults.x,   p00, params); texAtomicAdd(dst.B,   s.B*mults.z,   p01, params);
      texAtomicAdd(dst.B,   s.B*mults.y,   p10, params); texAtomicAdd(dst.B,   s.B*mults.w,   p11, params);
    }
}



template<typename T>
__global__ void pressurePre_k(FluidState<T> src, FluidState<T> dst, SimParams<T> params)
{
  typedef T VT;
  using     ST = typename Dims<T>::BASE;
  using     IT = typename Dims<T>::SIZE_T;
  
  int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  IT  ip   = makeI<IT>(ix, iy, iz);
  IT  size = src.size;
  if(ip < size)
    {
      int i = src.idx(ip);
      VT h  = 1.0/makeV<VT>(size);
      
      IT p00 = applyBounds(ip,   src.size, params); // current cell
      IT pn1 = applyBounds(ip-1, src.size, params); // cell - (1,1)
      IT pp1 = applyBounds(ip+1, src.size, params); // cell + (1,1)
      
      // calculate divergence
      if(p00 >= 0 && p00 < size &&
         pp1 >= 0 && pp1 < size &&
         pn1 >= 0 && pn1 < size)
        {
          dst.div[i] = -0.5f*(h.x*(src.v[p00.y*size.x + pp1.x].x - src.v[p00.y*size.x + pn1.x].x) +
                              h.y*(src.v[pp1.y*size.x + p00.x].y - src.v[pn1.y*size.x + p00.x].y));
        }
      //src.div[i] = dst.div[i];
    }
}

template<typename T>
__global__ void pressureIter_k(FluidState<T> src, FluidState<T> dst, SimParams<T> params)
{
  typedef T VT;
  using     ST = typename Dims<T>::BASE;
  using     IT = typename Dims<T>::SIZE_T;
  
  int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  IT  ip   = makeI<IT>(ix, iy, iz);
  IT  size = src.size;
  if(ip < size)
    {
      int i = src.idx(ip);
      IT p00 = applyBounds(ip,   src.size, params); // current cell
      IT pp1 = applyBounds(ip+1, src.size, params); // cell + (1,1)
      IT pn1 = applyBounds(ip-1, src.size, params); // cell - (1,1)

      // iterate --> update pressure
      if(p00 >= 0 && pp1 >= 0 && pn1 >= 0)
        {
          dst.p[i] = (src.p[p00.y*size.x + pn1.x] + src.p[p00.y*size.x + pp1.x] +
                      src.p[pn1.y*size.x + p00.x] + src.p[pp1.y*size.x + p00.x] +
                      src.div[i]) / 4.0f;
        }
    }
}

template<typename T>
__global__ void pressurePost_k(FluidState<T> src, FluidState<T> dst, SimParams<T> params)
{
  typedef T VT;
  using     ST = typename Dims<T>::BASE;
  using     IT = typename Dims<T>::SIZE_T;
  
  int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  IT  ip   = makeI<IT>(ix, iy, iz);
  IT  size = src.size;
  if(ip < size)
    {
      int i = src.idx(ip);
      VT  h = 1.0/makeV<VT>(size);
      
      IT p00 = applyBounds(ip,   src.size, params); // current cell
      IT pp1 = applyBounds(ip+1, src.size, params); // cell + (1,1)
      IT pn1 = applyBounds(ip-1, src.size, params); // cell - (1,1)
      
      if(p00 >= 0 && p00 < size &&
         pp1 >= 0 && pp1 < size &&
         pn1 >= 0 && pn1 < size)
        { // apply pressure to velocity
          dst.v[i].x -= 0.5f*(src.p[p00.y*size.x + pp1.x] - src.p[p00.y*size.x + pn1.x]) / h.x;
          dst.v[i].y -= 0.5f*(src.p[pp1.y*size.x + p00.x] - src.p[pn1.y*size.x + p00.x]) / h.y;
        }
    }
}


// fluid ions self-interaction via electrostatic Coulomb forces
template<typename T>
__global__ void emCalc_k(FluidState<T> src, SimParams<T> params)
{
  typedef T VT;
  using     ST = typename Dims<T>::BASE;
  using     IT = typename Dims<T>::SIZE_T;
  
  int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  IT  ip   = makeI<IT>(ix, iy, iz);
  IT  size = src.size;
  if(ip < size)
    {
      int i = src.idx(ip);
      VT  F = makeV<VT>(0);
      VT  p = makeV<VT>(ip);
      CellState<T> s = src[i];
      
      // add force from adjacent cells with charge
      int emRad = params.emRad;
      for(int x = -emRad; x <= emRad; x++)
        for(int y = -emRad; y <= emRad; y++)
          {
            IT dpi    = makeI<IT>(x, y, 0); // TODO: z
            VT dp     = makeV<VT>(dpi) / makeV<VT>(params.fp.fluidSize) * src.params.simSize + src.params.simPos;
            //dp /= GRAPHENE_CARBON_DIST;
            ST dist_2 = dot(dp, dp);
            ST dTest  = sqrt(dist_2);
            if(dist_2 > 0.0f && dTest <= emRad)
             {
               IT p2 = applyBounds(ip + dpi, src.size, params);
               if(p2.x >= 0 && p2.y >= 0 && p2.x < size.x && p2.y < size.y)
                 {
                   int i2 = src.idx(p2);
                   ST  q2 = src.qp[i2] - src.qn[i2]; // charge at other point
                   F += coulombForce(1.0f, q2, -dp); // q0 applied after loop (separately for qp and qn)
                 }
              }
          }
      
      // VT pforce =  F * (bCount > 0 ? (sCount/(float)bCount) : 1.0f) * params.emMult;
      VT pforce =  F * params.emMult*params.dt;

      s.E = pforce*(s.qp - s.qn);
      s.qpv += s.qp*pforce*params.dt;
      s.qnv += -s.qn*pforce*params.dt;

      // s.qpv.x = (s.qpv.x < 0 ? -1 : 1)*min(abs(s.qpv.x), 1.0f/params.dt);
      // s.qpv.y = (s.qpv.y < 0 ? -1 : 1)*min(abs(s.qpv.y), 1.0f/params.dt);
      // s.qnv.x = (s.qnv.x < 0 ? -1 : 1)*min(abs(s.qnv.x), 1.0f/params.dt);
      // s.qnv.y = (s.qnv.y < 0 ? -1 : 1)*min(abs(s.qnv.y), 1.0f/params.dt);

      // s.qpv = normalize(s.qpv)*min(length(s.qpv), 1.0f/params.dt);
      // s.qnv = normalize(s.qnv)*min(length(s.qnv), 1.0f/params.dt);
      
      src.qpv[i] = s.qpv;
      src.qnv[i] = s.qnv;
      
      // // use forward Euler method to advect charge current
      // VT p2p = integrateForwardEuler(src.qpv, p, s.qpv, params.dt);
      // VT p2n = integrateForwardEuler(src.qnv, p, s.qnv, params.dt);

      // // add actively to next point in texture

      // // positive charge
      // int4   tiX   = texPutIX   (p2p, params);
      // int4   tiY   = texPutIY   (p2p, params);
      // float4 mults = texPutMults(p2p);
      // IT     p00   = int2{tiX.x, tiY.x}; IT p10 = int2{tiX.y, tiY.y};
      // IT     p01   = int2{tiX.z, tiY.z}; IT p11 = int2{tiX.w, tiY.w};
      // texAtomicAdd(dst.qp,  s.qp*mults.x,  p00, params); texAtomicAdd(dst.qp,  s.qp*mults.z,  p01, params);
      // texAtomicAdd(dst.qp,  s.qp*mults.y,  p10, params); texAtomicAdd(dst.qp,  s.qp*mults.w,  p11, params);
      // texAtomicAdd(dst.qpv, s.qpv*mults.x, p00, params); texAtomicAdd(dst.qpv, s.qpv*mults.z, p01, params);
      // texAtomicAdd(dst.qpv, s.qpv*mults.y, p10, params); texAtomicAdd(dst.qpv, s.qpv*mults.w, p11, params);
      
      // // negative charge
      // tiX   = texPutIX   (p2n, params);
      // tiY   = texPutIY   (p2n, params);
      // mults = texPutMults(p2n);
      // p00   = int2{tiX.x, tiY.x};  p10 = int2{tiX.y, tiY.y};
      // p01   = int2{tiX.z, tiY.z};  p11 = int2{tiX.w, tiY.w};
      // texAtomicAdd(dst.qn,  s.qn*mults.x,  p00, params); texAtomicAdd(dst.qn,  s.qn*mults.z,  p01, params);
      // texAtomicAdd(dst.qn,  s.qn*mults.y,  p10, params); texAtomicAdd(dst.qn,  s.qn*mults.w,  p11, params);
      // texAtomicAdd(dst.qnv, s.qnv*mults.x, p00, params); texAtomicAdd(dst.qnv, s.qnv*mults.z, p01, params);
      // texAtomicAdd(dst.qnv, s.qnv*mults.y, p10, params); texAtomicAdd(dst.qnv, s.qnv*mults.w, p11, params);
      
      // dst.v[i]   = s.v;
      // dst.d[i]   = s.d;
      // dst.p[i]   = s.p;
      // dst.div[i] = s.div;   
      // dst.E[i]   = s.E;
      // dst.B[i]   = s.B;
    }
}


// fluid ions self-interaction via electrostatic Coulomb forces
template<typename T>
__global__ void emAdvect_k(FluidState<T> src, FluidState<T> dst, SimParams<T> params)
{
  typedef T VT;
  using     ST = typename Dims<T>::BASE;
  using     IT = typename Dims<T>::SIZE_T;
  int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  IT  ip   = makeI<IT>(ix, iy, iz);
  IT  size = src.size;
  if(ip < size)
    {
      int i = src.idx(ip);
      VT  p = makeV<VT>(ip);
      CellState<T> s = src[i];
      
      // use forward Euler method to advect charge current
      VT p2p = integrateForwardEuler(src.qpv, p, s.qpv, params.dt);
      VT p2n = integrateForwardEuler(src.qnv, p, s.qnv, params.dt);
      
      // positive charge
      int4   tiX   = texPutIX   (p2p, params);
      int4   tiY   = texPutIY   (p2p, params);
      float4 mults = texPutMults(p2p);
      IT     p00   = IT{tiX.x, tiY.x}; IT p10 = IT{tiX.y, tiY.y};
      IT     p01   = IT{tiX.z, tiY.z}; IT p11 = IT{tiX.w, tiY.w};
      texAtomicAdd(dst.qp,  s.qp*mults.x,  p00, params); texAtomicAdd(dst.qp,  s.qp*mults.z,  p01, params);
      texAtomicAdd(dst.qp,  s.qp*mults.y,  p10, params); texAtomicAdd(dst.qp,  s.qp*mults.w,  p11, params);
      texAtomicAdd(dst.qpv, s.qpv*mults.x, p00, params); texAtomicAdd(dst.qpv, s.qpv*mults.z, p01, params);
      texAtomicAdd(dst.qpv, s.qpv*mults.y, p10, params); texAtomicAdd(dst.qpv, s.qpv*mults.w, p11, params);
      
      // negative charge
      tiX   = texPutIX   (p2n, params);
      tiY   = texPutIY   (p2n, params);
      mults = texPutMults(p2n);
      p00   = IT{tiX.x, tiY.x};  p10 = IT{tiX.y, tiY.y};
      p01   = IT{tiX.z, tiY.z};  p11 = IT{tiX.w, tiY.w};
      texAtomicAdd(dst.qn,  s.qn*mults.x,  p00, params); texAtomicAdd(dst.qn,  s.qn*mults.z,  p01, params);
      texAtomicAdd(dst.qn,  s.qn*mults.y,  p10, params); texAtomicAdd(dst.qn,  s.qn*mults.w,  p11, params);
      texAtomicAdd(dst.qnv, s.qnv*mults.x, p00, params); texAtomicAdd(dst.qnv, s.qnv*mults.z, p01, params);
      texAtomicAdd(dst.qnv, s.qnv*mults.y, p10, params); texAtomicAdd(dst.qnv, s.qnv*mults.w, p11, params);
      
      dst.v[i]   = s.v;
      dst.d[i]   = s.d;
      dst.p[i]   = s.p;
      dst.div[i] = s.div;   
      dst.E[i]   = s.E;
      dst.B[i]   = s.B;
    }
}










#define SAMPLE_LINEAR 1
#define SAMPLE_POINT  0

template<typename T>
__global__ void render_k(FluidState<T> src, CudaTex<T> tex, SimParams<T> params)
{
  typedef T VT;
  using     ST = typename Dims<T>::BASE;
  using     IT = typename Dims<T>::SIZE_T;
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  //int iz = blockIdx.z*blockDim.z + threadIdx.z;
  IT  ip = makeI<IT>(ix, iy, 0);
  if(ip >= 0 && ip < params.fp.texSize)
    {
      VT tp    = makeV<VT>(ip);
      VT tSize = makeV<VT>(IT{tex.size.x, tex.size.y});
      VT fSize = makeV<VT>(src.size);
      
      VT fp = ((tp + 0.5)/tSize) * fSize - 0.5;
      
      CellState<T> s;
#if   SAMPLE_LINEAR // linearly interpolated sampling (bilinear)
      VT fp0   = floor(fp); // lower index
      VT fp1   = fp0 + 1;   // upper index
      VT alpha = fp - fp0;  // fractional offset
      IT bp00 = applyBounds(makeV<IT>(fp0), src.size, params);
      IT bp11 = applyBounds(makeV<IT>(fp1), src.size, params);

      if(bp00 >= 0 && bp00 < src.size && bp11 >= 0 && bp11 < src.size)
        {
          IT bp01 = bp00; bp01.x = bp11.x; // x + 1
          IT bp10 = bp00; bp10.y = bp11.y; // y + 1
          
          s = lerp(lerp(src[src.idx(bp00)], src[src.idx(bp01)], alpha.x),
                   lerp(src[src.idx(bp10)], src[src.idx(bp11)], alpha.x), alpha.y);
        }
      else
        {
          int ti = iy*tex.size.x + ix;
          tex.data[ti] = float4{1.0f, 0.0f, 1.0f, 1.0f};
          return;
        }
      
#elif SAMPLE_POINT  // integer point sampling (render true cell areas)
      VT fp0 = floor(fp);  // lower index
      IT bp  = applyBounds(makeV<IT>(fp0), src.size, params);
      if(bp >= 0 && bp < src.size)
        { s = src[src.idx(ip)]; }
      else
        {
          int ti = iy*tex.size.x + ix;
          tex.data[ti] = float4{1.0f, 0.0f, 1.0f, 1.0f};
          return;
        }
#endif

      float4 color = float4{0.0f, 0.0f, 0.0f, 0.0f};
      
      ST vLen  = length(s.v);
      VT vn    = (vLen != 0.0f ? normalize(s.v) : makeV<VT>(0.0f));
      ST nq    = s.qn;
      ST pq    = s.qp;
      ST q     = s.qp - s.qn;
      VT qv    = s.qpv - s.qnv;
      ST qvLen = length(qv);
      VT qvn   = (qvLen != 0.0f ? normalize(qv) : makeV<VT>(0.0f));
      
      ST Emag  = length(s.E);
      ST Bmag  = length(s.B);

      vn = abs(vn);
      
      color += s.v.x * params.render.getParamMult(FLUID_RENDER_VX);
      color += s.v.y * params.render.getParamMult(FLUID_RENDER_VY);
      //color += s.v.z * params.render.getParamMult(FLUID_RENDER_VZ);
      color +=  vLen * params.render.getParamMult(FLUID_RENDER_VMAG);
      color +=  vn.x * params.render.getParamMult(FLUID_RENDER_NVX);
      color +=  vn.y * params.render.getParamMult(FLUID_RENDER_NVY);
      //color +=  vn.z * params.render.getParamMult(FLUID_RENDER_NVZ);
      color += s.div * params.render.getParamMult(FLUID_RENDER_DIV);
      color +=   s.d * params.render.getParamMult(FLUID_RENDER_D);
      color +=   s.p * params.render.getParamMult(FLUID_RENDER_P);
      color +=     q * params.render.getParamMult(FLUID_RENDER_Q);
      color +=    nq * params.render.getParamMult(FLUID_RENDER_NQ);
      color +=    pq * params.render.getParamMult(FLUID_RENDER_PQ);
      color +=  qv.x * params.render.getParamMult(FLUID_RENDER_QVX);
      color +=  qv.y * params.render.getParamMult(FLUID_RENDER_QVY);
      //color +=  qv.z * params.render.getParamMult(FLUID_RENDER_QVZ);
      color += qvLen * params.render.getParamMult(FLUID_RENDER_QVMAG);
      color += qvn.x * params.render.getParamMult(FLUID_RENDER_NQVX);
      color += qvn.y * params.render.getParamMult(FLUID_RENDER_NQVY);
      //color += qvn.z * params.render.getParamMult(FLUID_RENDER_NQVZ);
      color +=  Emag * params.render.getParamMult(FLUID_RENDER_E);
      color +=  Bmag * params.render.getParamMult(FLUID_RENDER_B);      

      color = float4{ max(0.0f, min(color.x, 1.0f)), max(0.0f, min(color.y, 1.0f)),
                      max(0.0f, min(color.z, 1.0f)), max(0.0f, min(color.w, 1.0f)) };
      
      int ti = iy*tex.size.x + ix;
      tex.data[ti] = float4{color.x, color.y, color.z, 1.0f};
    }
}










//// HOST INTERFACE TEMPLATES ////

template<typename T>
void fieldAdvect(FluidState<T> src, FluidState<T> dst, SimParams<T> params)
{
  if(src.size > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y));
#if FORWARD_EULER
      // set to zero for forward euler method -- kernel will re-add contents
      fieldClear_k<<<grid, threads>>>(dst);
#endif
      advect_k <<<grid, threads>>>(src, dst, params);
    }
}

//// PRESSURE STEPS ////
// (assume error checking is handled elsewhere)
template<typename T>
void fieldPressurePre(FluidState<T> src, FluidState<T> dst, SimParams<T> params)
{
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
  dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X), (int)ceil(src.size.y/(float)BLOCKDIM_Y));
  pressurePre_k<<<grid, threads>>>(src, dst, params);
}
template<typename T>
void fieldPressureIter(FluidState<T> src, FluidState<T> dst, SimParams<T> params)
{
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
  dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X), (int)ceil(src.size.y/(float)BLOCKDIM_Y));
  pressureIter_k<<<grid, threads>>>(src, dst, params);
}
template<typename T>
void fieldPressurePost(FluidState<T> src, FluidState<T> dst, SimParams<T> params)
{
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
  dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X), (int)ceil(src.size.y/(float)BLOCKDIM_Y));
  pressurePost_k<<<grid, threads>>>(src, dst, params);
}

// (combined steps)
static FluidBase *g_temp1p = nullptr;
static FluidBase *g_temp2p = nullptr;
template<typename T>
void fieldPressure(FluidState<T> src, FluidState<T> dst, SimParams<T> params)
{
  if(src.size > 0 && dst.size == src.size)
    {
      FluidState<T> *temp1 = static_cast<FluidState<T>*>(g_temp1p);
      FluidState<T> *temp2 = static_cast<FluidState<T>*>(g_temp2p);
      
      if(!temp1)
        {
          temp1    = new FluidState<T>();
          g_temp1p = static_cast<FluidBase*>(temp1);
        }
      if(!temp2)
        {
          temp2    = new FluidState<T>();
          g_temp2p = static_cast<FluidBase*>(temp2);
        }
      // if(!temp1p) { temp1p = new FluidState<T>(); } if(!temp2p) { temp2p = new FluidState<T>(); }
      temp1->create(src.size); temp2->create(src.size);
      if(params.pIter > 0)
        {
          // PRESSURE INIT
          src.copyTo(*temp1); src.copyTo(*temp2); src.copyTo(dst);
          fieldPressurePre(src, *temp1, params); // src --> temp1
          if(params.debug) { cudaDeviceSynchronize(); } getLastCudaError("Pre-Pressure step failed!");

          // PRESSURE ITERATION
          for(int i = 0; i < params.pIter; i++)
            {
              fieldPressureIter(*temp1, *temp2, params); std::swap(temp1, temp2); // temp1 --> temp2 (--> temp1)
              if(params.debug) { cudaDeviceSynchronize(); }
              getLastCudaError(("Pressure Iteration step failed! (" + std::to_string(i) + "/" + std::to_string(params.pIter) + ")").c_str());
            }
          if(params.pIter % 2 == 0) { std::swap(temp1, temp2); } // ?
      
          // PRESSURE FINALIZE
          fieldPressurePost(*temp1, dst, params); // temp1 --> dst
          if(params.debug) { cudaDeviceSynchronize(); } getLastCudaError("Post-Pressure step failed!");
        }
      else// copy src to dst
        { src.copyTo(dst); }
    }
}
//////////////////

template<typename T>
void fieldEM(FluidState<T> src, FluidState<T> dst, SimParams<T> params)
{
  if(src.size > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y));
      
#if FORWARD_EULER
      // set to zero for forward euler method -- kernel will re-add contents
      fieldClear_k<<<grid, threads>>>(dst);
#endif // FORWARD_EULER
      
      emCalc_k<<<grid, threads>>>(src, params);
      getLastCudaError("====> ERROR: emCalc_k failed!");
      emAdvect_k<<<grid, threads>>>(src, dst, params);
      getLastCudaError("====> ERROR: emAdvect_k failed!");
    }
}


static FluidBase *g_temp1 = nullptr;
static FluidBase *g_temp2 = nullptr;

template<typename T>
void fieldStep(FluidState<T> &src, FluidState<T> &dst, GrapheneState<T> &gsrc, SimParams<T> &params)
{
  if(src.size > 0 && dst.size == src.size)
    {
      FluidState<T> *temp1 = static_cast<FluidState<T>*>(g_temp1);
      FluidState<T> *temp2 = static_cast<FluidState<T>*>(g_temp2);

      // initialize temp storage
      if(!temp1)
        {
          temp1   = new FluidState<T>();
          g_temp1 = static_cast<FluidBase*>(temp1);
        }
      if(!temp2)
        {
          temp2   = new FluidState<T>();
          g_temp2 = static_cast<FluidBase*>(temp2);
        }
      temp1->create(src.size);
      temp2->create(src.size);
      src.copyTo(*temp1);
      
      // // PRESSURE SOLVE (1)
      if(params.runPressure1)
        {
          fieldPressure(*temp1, *temp2, params); std::swap(temp1, temp2);
          if(params.debug) { cudaDeviceSynchronize(); } getLastCudaError("Pressure failed!");
        }
      // ADVECT
      if(params.runAdvect)
        {
          fieldAdvect(*temp1, *temp2, params);   std::swap(temp1, temp2);
          if(params.debug) { cudaDeviceSynchronize(); } getLastCudaError("Advect failed!");
        }
      // PRESSURE SOLVE (2)
      if(params.runPressure2)
        {
          fieldPressure(*temp1, *temp2, params); std::swap(temp1, temp2);
          if(params.debug) { cudaDeviceSynchronize(); } getLastCudaError("Pressure failed!");
        }
      // EM FORCES
      if(params.runEM)
        {
          fieldEM(*temp1, *temp2, params);       std::swap(temp1, temp2);
          if(params.debug) { cudaDeviceSynchronize(); } getLastCudaError("EMforce failed!");
        }

      // GRAPHENE FORCES
      if(params.runGraphene)
        {
          grapheneForce(*temp1, *temp2, gsrc, params); std::swap(temp1, temp2);
          if(params.debug) { cudaDeviceSynchronize(); } getLastCudaError("grapheneForce failed!");
        }
      
      temp1->copyTo(dst);
    }
}


template<typename T>
void fieldRender(FluidState<T> &src, CudaTex<T> &tex, SimParams<T> &params)
{
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
  dim3 grid((int)ceil(tex.size.x/(float)BLOCKDIM_X),
            (int)ceil(tex.size.y/(float)BLOCKDIM_Y));
  float4 *texData = tex.map();
  render_k<<<grid, threads>>>(src, tex, params);
  cudaDeviceSynchronize(); getLastCudaError("====> ERROR: render_k failed!");
  tex.unmap();
}



// template instantiations
template void fieldStep  <float2>(FluidState   <float2> &src,  FluidState   <float2> &dst, GrapheneState<float2> &gsrc, SimParams<float2> &params);
template void fieldRender<float2>(FluidState   <float2> &src,  CudaTex      <float2> &tex, SimParams<float2> &params);


template void fieldStep  <float3>(FluidState   <float3> &src,  FluidState   <float3> &dst, GrapheneState<float3> &gsrc, SimParams<float3> &params);
template void fieldRender<float3>(FluidState   <float3> &src,  CudaTex      <float3> &tex, SimParams<float3> &params);


//// 3D ////
// template void fieldStep<float3> (FluidState<float3> &src, FluidState<float3> &dst, Graphene<float3> &graphene, SimParams<float3> &params);
// template void fieldRender<float3>(FluidState<float3> &src, CudaTex<float3> &tex, SimParams<float3> &params);
