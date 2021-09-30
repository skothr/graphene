#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>

#include "cuda-tools.cuh"
#include "fluid.cuh"
#include "physics.h"

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8
#define BLOCKDIM_Z 8

#define FORWARD_EULER 1


////////////////////////////////////////////////////////////////////////////////////////////////
//// kernels
////////////////////////////////////////////////////////////////////////////////////////////////


template<typename T>
__global__ void advect_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> params)
{
  typedef typename DimType<T, 3>::VEC_T VT3;
  typedef typename Dim<VT3>::SIZE_T IT;
  
  int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  IT  ip   = makeI<IT>(ix, iy, iz);
  IT  size = src.size;
  if(ip.x < size.x && ip.y < size.y && ip.z < size.z)
    {
      int i  = src.idx(ip);
      VT3 p  = VT3{T(ix), T(iy), T(iz)};
      T   dt = params.u.dt;

      VT3 v0 = src.v[i];
      T   p0 = src.p[i];
      T   d0 = src.div[i];
      T   Qn = src.Qn[i];
      T   Qp = src.Qp[i];
      VT3 Qv = src.Qv[i];
      VT3 E0 = src.E[i];
      VT3 B0 = src.B[i];
      Material<T> mat = src.mat[i];
      
      if(slipPlane(ip, params)) { v0 = makeV<VT3>(0); } // zero velocity on edges (assumes wall is static)
      if(slipPlane(ip-IT{1,0,0}, params)) { v0.x = 0; } // abs(v0.x); }
      if(slipPlane(ip+IT{1,0,0}, params)) { v0.x = 0; } //-abs(v0.x); }
      if(slipPlane(ip-IT{0,1,0}, params)) { v0.y = 0; } // abs(v0.y); }
      if(slipPlane(ip+IT{0,1,0}, params)) { v0.y = 0; } //-abs(v0.y); }
      if(slipPlane(ip-IT{0,0,1}, params)) { v0.z = 0; } // abs(v0.z); }
      if(slipPlane(ip+IT{0,0,1}, params)) { v0.z = 0; } //-abs(v0.z); }
      
      // apply charge force to velocity
      //s.v += params.qvfMult*s.qv*dt;
      
      // v0.x = (v0.x < 0 ? -1.0f : 1.0f)*min(abs(v0.x), 16.0f);
      // v0.y = (v0.y < 0 ? -1.0f : 1.0f)*min(abs(v0.y), 16.0f);
      // Qv.x = (Qv.x < 0 ? -1.0f : 1.0f)*min(abs(Qv.x), 16.0f);
      // Qv.y = (Qv.y < 0 ? -1.0f : 1.0f)*min(abs(Qv.y), 16.0f);

      // check for invalid values
      if(isnan(v0.x) || isinf(v0.x)) { v0.x = 0.0; } if(isnan(v0.y) || isinf(v0.y)) { v0.y = 0.0; } if(isnan(v0.z) || isinf(v0.z)) { v0.z = 0.0; }
      if(isnan(p0)   || isinf(p0))   { p0   = 0.0; } if(isnan(d0)   || isinf(d0))   { d0   = 0.0; }
      
      // use forward Euler method
      VT3 p2    = integrateForwardEuler(src.v.dData, p, v0, dt);
      // add actively to next point in texture
      int4   tiX    = texPutIX(p2, params);
      int4   tiY    = texPutIY(p2, params);
      int4   tiZ    = texPutIZ(p2, params);
      float4 mults0 = texPutMults0<float>(p2);
      float4 mults1 = texPutMults1<float>(p2);
      IT     p000   = IT{tiX.x, tiY.x, tiZ.x}; IT p100 = IT{tiX.y, tiY.y, tiZ.x};
      IT     p010   = IT{tiX.z, tiY.z, tiZ.x}; IT p110 = IT{tiX.w, tiY.w, tiZ.x};
      IT     p001   = IT{tiX.x, tiY.x, tiZ.z}; IT p101 = IT{tiX.y, tiY.y, tiZ.z};
      IT     p011   = IT{tiX.z, tiY.z, tiZ.z}; IT p111 = IT{tiX.w, tiY.w, tiZ.z};

      //__device__ void texAtomicAdd(float *tex, float val, int2 p, const SimSrc.Params<float2> &src.params);
      // scale value by grid overlap and store in each location
      // v
      texAtomicAdd(dst.v.dData,   v0*mults0.x,  p000, params); texAtomicAdd(dst.v.dData,   v0*mults0.z,  p010, params);
      texAtomicAdd(dst.v.dData,   v0*mults0.y,  p100, params); texAtomicAdd(dst.v.dData,   v0*mults0.w,  p110, params);
      texAtomicAdd(dst.v.dData,   v0*mults1.x,  p001, params); texAtomicAdd(dst.v.dData,   v0*mults1.z,  p011, params);
      texAtomicAdd(dst.v.dData,   v0*mults1.y,  p101, params); texAtomicAdd(dst.v.dData,   v0*mults1.w,  p111, params);
      // // p
      // texAtomicAdd(dst.p.dData,   p0*mults0.x,  p000, params); texAtomicAdd(dst.p.dData,   p0*mults0.z,  p010, params);
      // texAtomicAdd(dst.p.dData,   p0*mults0.y,  p100, params); texAtomicAdd(dst.p.dData,   p0*mults0.w,  p110, params);
      // texAtomicAdd(dst.p.dData,   p0*mults1.x,  p001, params); texAtomicAdd(dst.p.dData,   p0*mults1.z,  p011, params);
      // texAtomicAdd(dst.p.dData,   p0*mults1.y,  p101, params); texAtomicAdd(dst.p.dData,   p0*mults1.w,  p111, params);
      // Qn
      texAtomicAdd(dst.Qn.dData,  Qn*mults0.x,  p000, params); texAtomicAdd(dst.Qn.dData,  Qn*mults0.z,  p010, params);
      texAtomicAdd(dst.Qn.dData,  Qn*mults0.y,  p100, params); texAtomicAdd(dst.Qn.dData,  Qn*mults0.w,  p110, params);
      texAtomicAdd(dst.Qn.dData,  Qn*mults1.x,  p001, params); texAtomicAdd(dst.Qn.dData,  Qn*mults1.z,  p011, params);
      texAtomicAdd(dst.Qn.dData,  Qn*mults1.y,  p101, params); texAtomicAdd(dst.Qn.dData,  Qn*mults1.w,  p111, params);
      // Qp
      texAtomicAdd(dst.Qp.dData,  Qp*mults0.x,  p000, params); texAtomicAdd(dst.Qp.dData,  Qp*mults0.z,  p010, params);
      texAtomicAdd(dst.Qp.dData,  Qp*mults0.y,  p100, params); texAtomicAdd(dst.Qp.dData,  Qp*mults0.w,  p110, params);
      texAtomicAdd(dst.Qp.dData,  Qp*mults1.x,  p001, params); texAtomicAdd(dst.Qp.dData,  Qp*mults1.z,  p011, params);
      texAtomicAdd(dst.Qp.dData,  Qp*mults1.y,  p101, params); texAtomicAdd(dst.Qp.dData,  Qp*mults1.w,  p111, params);
      // Qv
      texAtomicAdd(dst.Qv.dData,  Qv*mults0.x,  p000, params); texAtomicAdd(dst.Qv.dData,  Qv*mults0.z,  p010, params);
      texAtomicAdd(dst.Qv.dData,  Qv*mults0.y,  p100, params); texAtomicAdd(dst.Qv.dData,  Qv*mults0.w,  p110, params);
      texAtomicAdd(dst.Qv.dData,  Qv*mults1.x,  p001, params); texAtomicAdd(dst.Qv.dData,  Qv*mults1.z,  p011, params);
      texAtomicAdd(dst.Qv.dData,  Qv*mults1.y,  p101, params); texAtomicAdd(dst.Qv.dData,  Qv*mults1.w,  p111, params);
      // div
      dst.div[i] = d0;
      // E
      dst.E[i]   = E0;
      // B
      dst.B[i]   = B0;
      // mat
      dst.mat[i] = mat;
    }
}



template<typename T>
__global__ void pressurePre_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> params)
{
  typedef typename DimType<T, 3>::VEC_T VT3;
  typedef typename Dim<VT3>::SIZE_T IT;
  
  int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  IT  ip   = makeI<IT>(ix, iy, iz);
  IT  size = src.size;
  if(ip.x < size.x && ip.y < size.y && ip.z < size.z)
    {
      int i = src.idx(ip);
      VT3 h = 1.0/makeV<VT3>(size);
      
      IT p00 = applyBounds(ip,   src.size, params); // current cell
      IT pn1 = applyBounds(ip-1, src.size, params); // cell - (1,1)
      IT pp1 = applyBounds(ip+1, src.size, params); // cell + (1,1)
      
      // calculate divergence
      if(p00 >= 0 && p00 < size &&
         pp1 >= 0 && pp1 < size &&
         pn1 >= 0 && pn1 < size)
        {
          dst.div[i] = -0.33f*(h.x*(src.v[src.idx(pp1.x, p00.y, p00.z)].x - src.v[src.idx(pn1.x, p00.y, p00.z)].x) +
                               h.y*(src.v[src.idx(p00.x, pp1.y, p00.z)].y - src.v[src.idx(p00.x, pn1.y, p00.z)].y) +
                               h.z*(src.v[src.idx(p00.x, p00.y, pp1.z)].z - src.v[src.idx(p00.x, p00.y, pn1.z)].z));
        }
      dst.p[i] = 0;
    }
}

template<typename T>
__global__ void pressureIter_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> params)
{
  typedef typename DimType<T, 3>::VEC_T VT3;
  typedef typename Dim<VT3>::SIZE_T IT;
  
  int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  IT  ip   = makeI<IT>(ix, iy, iz);
  IT  size = src.size;
  if(ip.x < size.x && ip.y < size.y && ip.z < size.z)
    {
      int i = src.idx(ip);
      IT p00 = applyBounds(ip,   src.size, params); // current cell
      IT pp1 = applyBounds(ip+1, src.size, params); // cell + (1,1,1)
      IT pn1 = applyBounds(ip-1, src.size, params); // cell - (1,1,1)

      // iterate --> update pressure
      if(p00 >= 0 && pp1 >= 0 && pn1 >= 0)
        {
          dst.p[i] = (src.p[src.idx(pn1.x, p00.y, p00.z)] + src.p[src.idx(pp1.x, p00.y, p00.z)] +
                      src.p[src.idx(p00.x, pn1.y, p00.z)] + src.p[src.idx(p00.x, pp1.y, p00.z)] +
                      src.p[src.idx(p00.x, p00.y, pn1.z)] + src.p[src.idx(p00.x, p00.y, pp1.z)] +
                      src.div[i]) / 6.0f;
        }
    }
}

template<typename T>
__global__ void pressurePost_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> params)
{
  typedef typename DimType<T, 3>::VEC_T VT3;
  using IT = typename Dim<VT3>::SIZE_T;
  
  int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  IT  ip   = makeI<IT>(ix, iy, iz);
  IT  size = src.size;
  if(ip.x < size.x && ip.y < size.y && ip.z < size.z)
    {
      int i = src.idx(ip);
      VT3 h = 1.0/makeV<VT3>(size);
      
      IT p00 = applyBounds(ip,   src.size, params); // current cell
      IT pp1 = applyBounds(ip+1, src.size, params); // cell + (1,1,1)
      IT pn1 = applyBounds(ip-1, src.size, params); // cell - (1,1,1)
      
      if(p00 >= 0 && p00 < size &&
         pp1 >= 0 && pp1 < size &&
         pn1 >= 0 && pn1 < size)
        { // apply pressure to velocity
          dst.v[i].x -= 0.33f*(src.p[src.idx(pp1.x, p00.y, p00.z)] - src.p[src.idx(pn1.x, p00.y, p00.z)]) / h.x;
          dst.v[i].y -= 0.33f*(src.p[src.idx(p00.x, pp1.y, p00.z)] - src.p[src.idx(p00.x, pn1.y, p00.z)]) / h.y;
          dst.v[i].z -= 0.33f*(src.p[src.idx(p00.x, p00.y, pp1.z)] - src.p[src.idx(p00.x, p00.y, pn1.z)]) / h.z;
        }
    }
}


// fluid ions self-interaction via electrostatic Coulomb forces
template<typename T>
__global__ void emCalc_k(FluidField<T> dst, FluidParams<T> params)
{
  typedef typename DimType<T, 3>::VEC_T VT3;
  typedef typename Dim<VT3>::SIZE_T IT;
  
  int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  IT  ip   = makeI<IT>(ix, iy, iz);
  IT  size = dst.size;
  if(ip.x < size.x && ip.y < size.y && ip.z < size.z)
    {
      int i = dst.idx(ip);
      VT3 F = makeV<VT3>(0);

      // VT3 v  = dst.v[i];
      // T   p  = dst.p[i];
      // T   d  = dst.div[i];
      // T   Qn = dst.Qn[i];
      // T   Qp = dst.Qp[i];
      // VT3 Qv = dst.Qv[i];
      // VT3 E  = dst.E[i];
      // VT3 B  = dst.B[i];
      // Material<T> mat = dst.mat[i];
      
      // // add force from adjacent cells with charge
      // int emRad = params.emRad;
      // for(int x = -emRad; x <= emRad; x++)
      //   for(int y = -emRad; y <= emRad; y++)
      //     {
      //       IT  dpi  = makeI<I3T>(x, y, 0); // TODO: z
      //       VT3 dp   = makeV<VT3>(dpi)/makeV<VT3>(params.fs) + params.fp;
      //       //dp /= GRAPHENE_CARBON_DIST;
      //       T dist_2 = dot(dp, dp);
      //       T dTest  = sqrt(dist_2);
      //       if(dist_2 > 0.0f && dTest <= emRad)
      //        {
      //          IT p2 = applyBounds(ip + dpi, src.size, params);
      //          if(p2.x >= 0 && p2.y >= 0 && p2.x < size.x && p2.y < size.y)
      //            {
      //              int i2 = src.idx(p2);
      //              T  q2 = src.Qp[i2] - src.Qn[i2]; // charge at other point
      //              F += coulombForce(1.0f, q2, -dp); // q0 applied after loop (separately for qp and qn)
      //            }
      //         }
      //     }
      
      // // VT3 pforce =  F * (bCount > 0 ? (sCount/(float)bCount) : 1.0f) * params.emMult;
      // VT3 pforce =  F * params.emMult*params.dt;

      // E   = pforce*(Qp - Qn);
      // Qv += Qp*pforce*params.dt;

      // s.qpv.x = (s.qpv.x < 0 ? -1 : 1)*min(abs(s.qpv.x), 1.0f/params.dt);
      // s.qpv.y = (s.qpv.y < 0 ? -1 : 1)*min(abs(s.qpv.y), 1.0f/params.dt);
      // s.qnv.x = (s.qnv.x < 0 ? -1 : 1)*min(abs(s.qnv.x), 1.0f/params.dt);
      // s.qnv.y = (s.qnv.y < 0 ? -1 : 1)*min(abs(s.qnv.y), 1.0f/params.dt);

      // s.qpv = normalize(s.qpv)*min(length(s.qpv), 1.0f/params.dt);
      // s.qnv = normalize(s.qnv)*min(length(s.qnv), 1.0f/params.dt);
      
      // src.Qv[i] = Qv;
      
      // // use forward Euler method to advect charge current
      // VT3 p2p = integrateForwardEuler(src.qpv, p, s.qpv, params.dt);
      // VT3 p2n = integrateForwardEuler(src.qnv, p, s.qnv, params.dt);

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
      
      // dst.v[i]   = v;
      // dst.p[i]   = p;
      // dst.div[i] = d;   
      // dst.Qn[i]  = Qn;  
      // dst.Qp[i]  = Qp;  
      // dst.Qv[i]  = Qv;
      // dst.E[i]   = E;
      // dst.B[i]   = B;
      // dst.mat[i] = mat;
    }
}


// fluid ions self-interaction via electrostatic Coulomb forces
template<typename T>
__global__ void emAdvect_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> params)
{
  typedef typename DimType<T, 3>::VEC_T VT3;
  typedef typename Dim<VT3>::SIZE_T IT;
  int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  IT  ip   = makeI<IT>(ix, iy, iz);
  IT  size = src.size;
  if(ip.x < size.x && ip.y < size.y && ip.z < size.z)
    {
      int i = src.idx(ip);
      VT3 p = makeV<VT3>(ip);
      
      VT3 v0 = src.v[i];
      T   p0 = src.p[i];
      T   d0 = src.div[i];
      T   Qn = src.Qn[i];
      T   Qp = src.Qp[i];
      VT3 Qv = src.Qv[i];
      VT3 E0 = src.E[i];
      VT3 B0 = src.B[i];
      Material<T> mat = src.mat[i];
      
      // use forward Euler method to advect charge current
      VT3 p2p = integrateForwardEuler(src.Qv.dData, p, Qv, params.u.dt);
      VT3 p2n = integrateForwardEuler(src.Qv.dData, p, Qv, params.u.dt);
      
      // positive charge
      int4   tiX    = texPutIX(p2p, params);
      int4   tiY    = texPutIY(p2p, params);
      int4   tiZ    = texPutIZ(p2p, params);
      float4 mults0 = texPutMults0<float>(p2p);
      float4 mults1 = texPutMults1<float>(p2p);
      IT    p000   = IT{tiX.x, tiY.x, tiZ.x}; IT p100 = IT{tiX.y, tiY.y, tiZ.x};
      IT    p010   = IT{tiX.z, tiY.z, tiZ.x}; IT p110 = IT{tiX.w, tiY.w, tiZ.x};
      IT    p001   = IT{tiX.x, tiY.x, tiZ.z}; IT p101 = IT{tiX.y, tiY.y, tiZ.z};
      IT    p011   = IT{tiX.z, tiY.z, tiZ.z}; IT p111 = IT{tiX.w, tiY.w, tiZ.z};
      texAtomicAdd(dst.Qp.dData, Qp*mults0.x, p000, params); texAtomicAdd(dst.Qp.dData, Qp*mults0.z, p010, params);
      texAtomicAdd(dst.Qp.dData, Qp*mults0.y, p100, params); texAtomicAdd(dst.Qp.dData, Qp*mults0.w, p110, params);
      texAtomicAdd(dst.Qp.dData, Qp*mults1.x, p001, params); texAtomicAdd(dst.Qp.dData, Qp*mults1.z, p011, params);
      texAtomicAdd(dst.Qp.dData, Qp*mults1.y, p101, params); texAtomicAdd(dst.Qp.dData, Qp*mults1.w, p111, params);
      texAtomicAdd(dst.Qv.dData, Qv*mults0.x, p000, params); texAtomicAdd(dst.Qv.dData, Qv*mults0.z, p010, params);
      texAtomicAdd(dst.Qv.dData, Qv*mults0.y, p100, params); texAtomicAdd(dst.Qv.dData, Qv*mults0.w, p110, params);
      texAtomicAdd(dst.Qv.dData, Qv*mults1.x, p001, params); texAtomicAdd(dst.Qv.dData, Qv*mults1.z, p011, params);
      texAtomicAdd(dst.Qv.dData, Qv*mults1.y, p101, params); texAtomicAdd(dst.Qv.dData, Qv*mults1.w, p111, params);
      
      // // negative charge
      tiX   = texPutIX   (p2n, params);
      tiY   = texPutIY   (p2n, params);
      mults0 = texPutMults0<float>(p2p);
      mults1 = texPutMults1<float>(p2p);
      p000   = IT{tiX.x, tiY.x, tiZ.x};  p100 = IT{tiX.y, tiY.y, tiZ.x};
      p010   = IT{tiX.z, tiY.z, tiZ.x};  p110 = IT{tiX.w, tiY.w, tiZ.x};
      p001   = IT{tiX.x, tiY.x, tiZ.z};  p101 = IT{tiX.y, tiY.y, tiZ.z};
      p011   = IT{tiX.z, tiY.z, tiZ.z};  p111 = IT{tiX.w, tiY.w, tiZ.z};
      texAtomicAdd(dst.Qn.dData, Qn*mults0.x, p000, params); texAtomicAdd(dst.Qn.dData, Qn*mults0.z, p010, params);
      texAtomicAdd(dst.Qn.dData, Qn*mults0.y, p100, params); texAtomicAdd(dst.Qn.dData, Qn*mults0.w, p110, params);
      texAtomicAdd(dst.Qn.dData, Qn*mults1.x, p001, params); texAtomicAdd(dst.Qn.dData, Qn*mults1.z, p011, params);
      texAtomicAdd(dst.Qn.dData, Qn*mults1.y, p101, params); texAtomicAdd(dst.Qn.dData, Qn*mults1.w, p111, params);
      
      dst.v[i]   = v0;
      dst.p[i]   = p0;
      dst.div[i] = d0;
      dst.E[i]   = E0;
      dst.B[i]   = B0;
      dst.mat[i] = mat;
    }
}










// #define SAMPLE_LINEAR 1
// #define SAMPLE_POINT  0

// template<typename T>
// __global__ void render_k(FluidField<T> src, CudaTex<T> tex, FluidParams<T> params)
// {
//   typedef typename DimType<T, 3>::VEC_T VT3;
//   using IT = typename Dim<VT3>::SIZE_T;
  
//   int ix = blockIdx.x*blockDim.x + threadIdx.x;
//   int iy = blockIdx.y*blockDim.y + threadIdx.y;
//   //int iz = blockIdx.z*blockDim.z + threadIdx.z;
//   IT  ip = makeI<IT>(ix, iy, 0);
//   if(ip >= 0 && ip < params.fp.texSize)
//     {
//       VT3 tp    = makeV<VT3>(ip);
//       VT3 tSize = makeV<VT3>(IT{tex.size.x, tex.size.y});
//       VT3 fSize = makeV<VT3>(src.size);
      
//       VT3 fp = ((tp + 0.5)/tSize) * fSize - 0.5;
      
//       //CellState<T> s;
// #if   SAMPLE_LINEAR // linearly interpolated sampling (bilinear)
//       VT fp0   = floor(fp); // lower index
//       VT fp1   = fp0 + 1;   // upper index
//       VT alpha = fp - fp0;  // fractional offset
//       IT bp00 = applyBounds(makeV<IT>(fp0), src.size, params);
//       IT bp11 = applyBounds(makeV<IT>(fp1), src.size, params);

//       if(bp00 >= 0 && bp00 < src.size && bp11 >= 0 && bp11 < src.size)
//         {
//           IT bp01 = bp00; bp01.x = bp11.x; // x + 1
//           IT bp10 = bp00; bp10.y = bp11.y; // y + 1
          
//           s = lerp(lerp(src[src.idx(bp00)], src[src.idx(bp01)], alpha.x),
//                    lerp(src[src.idx(bp10)], src[src.idx(bp11)], alpha.x), alpha.y);
//         }
//       else
//         {
//           int ti = iy*tex.size.x + ix;
//           tex.data[ti] = float4{1.0f, 0.0f, 1.0f, 1.0f};
//           return;
//         }
      
// #elif SAMPLE_POINT  // integer point sampling (render true cell areas)
//       VT fp0 = floor(fp);  // lower index
//       IT bp  = applyBounds(makeV<IT>(fp0), src.size, params);
//       if(bp >= 0 && bp < src.size)
//         { s = src[src.idx(ip)]; }
//       else
//         {
//           int ti = iy*tex.size.x + ix;
//           tex.data[ti] = float4{1.0f, 0.0f, 1.0f, 1.0f};
//           return;
//         }
// #endif

//       float4 color = float4{0.0f, 0.0f, 0.0f, 0.0f};
      
//       T  vLen  = length(s.v);
//       VT vn    = (vLen != 0.0f ? normalize(s.v) : makeV<VT>(0.0f));
//       T  nq    = s.qn;
//       T  pq    = s.qp;
//       T  q     = s.qp - s.qn;
//       VT qv    = s.qpv - s.qnv;
//       T  qvLen = length(qv);
//       VT qvn   = (qvLen != 0.0f ? normalize(qv) : makeV<VT>(0.0f));
      
//       T Emag  = length(s.E);
//       T Bmag  = length(s.B);

//       vn = abs(vn);
      
//       color += s.v.x * params.render.getParamMult(FLUID_RENDER_VX);
//       color += s.v.y * params.render.getParamMult(FLUID_RENDER_VY);
//       //color += s.v.z * params.render.getParamMult(FLUID_RENDER_VZ);
//       color +=  vLen * params.render.getParamMult(FLUID_RENDER_VMAG);
//       color +=  vn.x * params.render.getParamMult(FLUID_RENDER_NVX);
//       color +=  vn.y * params.render.getParamMult(FLUID_RENDER_NVY);
//       //color +=  vn.z * params.render.getParamMult(FLUID_RENDER_NVZ);
//       color += s.div * params.render.getParamMult(FLUID_RENDER_DIV);
//       color +=   s.d * params.render.getParamMult(FLUID_RENDER_D);
//       color +=   s.p * params.render.getParamMult(FLUID_RENDER_P);
//       color +=     q * params.render.getParamMult(FLUID_RENDER_Q);
//       color +=    nq * params.render.getParamMult(FLUID_RENDER_NQ);
//       color +=    pq * params.render.getParamMult(FLUID_RENDER_PQ);
//       color +=  qv.x * params.render.getParamMult(FLUID_RENDER_QVX);
//       color +=  qv.y * params.render.getParamMult(FLUID_RENDER_QVY);
//       //color +=  qv.z * params.render.getParamMult(FLUID_RENDER_QVZ);
//       color += qvLen * params.render.getParamMult(FLUID_RENDER_QVMAG);
//       color += qvn.x * params.render.getParamMult(FLUID_RENDER_NQVX);
//       color += qvn.y * params.render.getParamMult(FLUID_RENDER_NQVY);
//       //color += qvn.z * params.render.getParamMult(FLUID_RENDER_NQVZ);
//       color +=  Emag * params.render.getParamMult(FLUID_RENDER_E);
//       color +=  Bmag * params.render.getParamMult(FLUID_RENDER_B);      

//       color = float4{ max(0.0f, min(color.x, 1.0f)), max(0.0f, min(color.y, 1.0f)),
//                       max(0.0f, min(color.z, 1.0f)), max(0.0f, min(color.w, 1.0f)) };
      
//       int ti = iy*tex.size.x + ix;
//       tex.data[ti] = float4{color.x, color.y, color.z, 1.0f};
//     }
// }










//// HOST INTERFACE TEMPLATES ////

template<typename T>
void fluidAdvect(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> params)
{
  if(src.size > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
#if FORWARD_EULER
      // set to zero for forward euler method -- kernel will re-add contents
      dst.v.clear();
      // dst.p.clear();
      dst.Qn.clear();
      dst.Qp.clear();
      dst.Qv.clear();
#endif
      advect_k<<<grid, threads>>>(src, dst, params);
    }
}

//// PRESSURE STEPS ////
// (assume error checking is handled elsewhere)
template<typename T>
void fluidPressurePre(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> params)
{
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
  dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
            (int)ceil(src.size.y/(float)BLOCKDIM_Y),
            (int)ceil(src.size.z/(float)BLOCKDIM_Z));
  pressurePre_k<<<grid, threads>>>(src, dst, params);
}
template<typename T>
void fluidPressureIter(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> params)
{
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
  dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
            (int)ceil(src.size.y/(float)BLOCKDIM_Y),
            (int)ceil(src.size.z/(float)BLOCKDIM_Z));
  pressureIter_k<<<grid, threads>>>(src, dst, params);
}
template<typename T>
void fluidPressurePost(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> params)
{
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
  dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
            (int)ceil(src.size.y/(float)BLOCKDIM_Y),
            (int)ceil(src.size.z/(float)BLOCKDIM_Z));
  pressurePost_k<<<grid, threads>>>(src, dst, params);
}

// (combined steps)
template<typename T>
void fluidPressure(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> params, int iter)
{
  if(src.size > 0 && dst.size == src.size)
    {
      src.copyTo(dst);
      FluidField<T> *temp1 = &src;
      FluidField<T> *temp2 = &dst;
      if(iter > 0)
        {
          // PRESSURE INIT
          fluidPressurePre(*temp1, *temp2, params); std::swap(temp1, temp2); // temp1 --> temp2 (--> temp1)
          // PRESSURE ITERATION
          for(int i = 0; i < iter; i++)
            {
              fluidPressureIter(*temp1, *temp2, params); std::swap(temp1, temp2); // temp1 --> temp2 (--> temp1)
              getLastCudaError(("Pressure Iteration step failed! (" + std::to_string(i) + "/" + std::to_string(iter) + ")").c_str());
            }
          // PRESSURE FINALIZE
          fluidPressurePost(*temp1, *temp2, params);  std::swap(temp1, temp2); // temp1 --> temp2 (--> temp1)
          if(temp1 != &dst) { temp1->v.copyTo(dst.v); temp1->p.copyTo(dst.p); temp1->div.copyTo(dst.div); }
        }
    }
}
//////////////////

template<typename T>
void fluidEM(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> params)
{
  if(src.size > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      
#if FORWARD_EULER
      // set to zero for forward euler method -- kernel will re-add contents
      dst.clear();
#endif // FORWARD_EULER
      
      emCalc_k<<<grid, threads>>>(src, params);
      getLastCudaError("====> ERROR: emCalc_k failed!");
      emAdvect_k<<<grid, threads>>>(src, dst, params);
      getLastCudaError("====> ERROR: emAdvect_k failed!");
    }
}



template<typename T>
void fluidStep(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> &params)
{
  if(src.size > 0 && dst.size == src.size)
    {
      FluidField<T> *temp1 = &src;
      FluidField<T> *temp2 = &dst;
      
      // // PRESSURE SOLVE (1)
      if(params.updateP1)       { fluidPressure(*temp1, *temp2, params, params.pIter1); std::swap(temp1, temp2); }
      // ADVECT
      if(params.updateAdvect)   { fluidAdvect(*temp1, *temp2, params);                  std::swap(temp1, temp2); }
      // PRESSURE SOLVE (2)
      if(params.updateP2)       { fluidPressure(*temp1, *temp2, params, params.pIter2); std::swap(temp1, temp2); }
      // // EM FORCES
      // if(params.runEM)       { fluidEM(*temp1, *temp2, params);                      std::swap(temp1, temp2); }
      // // GRAPHENE FORCES
      // if(params.runGraphene) { grapheneForce(*temp1, *temp2, gsrc, params);          std::swap(temp1, temp2); }

      if(temp1 != &dst) { temp1->copyTo(dst); }
    }
}


// template<typename T>
// void fieldRender(FluidField<T> &src, CudaTex<T> &tex, FluidParams<T> &params)
// {
//   dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
//   dim3 grid((int)ceil(tex.size.x/(float)BLOCKDIM_X),
//             (int)ceil(tex.size.y/(float)BLOCKDIM_Y),
//             (int)ceil(tex.size.z/(float)BLOCKDIM_Z));
//   float4 *texData = tex.map();
//   render_k<<<grid, threads>>>(src, tex, params);
//   tex.unmap();
// }



// template instantiations
template void fluidStep  <float>(FluidField<float> &src, FluidField<float> &dst, FluidParams<float> &params);
// template void fieldRender<float>(FluidField<float> &src, CudaTex   <float> &tex, FluidParams<float> &params);


//// 3D ////
// template void fieldStep<float3> (FluidField<float3> &src, FluidField<float3> &dst, Graphene<float3> &graphene, FluidParams<float3> &params);
// template void fieldRender<float3>(FluidField<float3> &src, CudaTex<float3> &tex, FluidParams<float3> &params);
