#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>

#include "cuda-tools.cuh"
#include "field-operators.cuh"
#include "fluid.cuh"
#include "physics.h"

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8
#define BLOCKDIM_Z 8


////////////////////////////////////////////////////////////////////////////////////////////////
//// kernels
////////////////////////////////////////////////////////////////////////////////////////////////


//// ADVECTION ////

template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT=typename Dim<VT3>::SIZE_T>
__global__ void advect_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> fp)
{
  int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  IT  ip   = makeI<IT>(ix, iy, iz);
  IT  size = src.size;
  if(ip.x < size.x && ip.y < size.y && ip.z < size.z)
    {
      int i  = src.idx(ip);
      VT3 p  = VT3{T(ix), T(iy), T(iz)};
      T   dt = fp.u.dt;
      
      VT3 v0  = src.v[i];
      T   p0  = src.p[i];
      T   d0  = src.div[i];
      T   Qn  = src.Qn[i];
      T   Qp  = src.Qp[i];
      VT3 Qnv = src.Qnv[i];
      VT3 Qpv = src.Qpv[i];
      VT3 E0  = src.E[i];
      VT3 B0  = src.B[i];
      Material<T> mat = src.mat[i];
      
      // limit magnitude (removes some energy from the system if enabled)
      if(fp.limitV) { v0 = normalize(v0)*min(length(v0), fp.maxV); }
      
      // check for invalid values
      if(!isvalid(v0.x) || !isvalid(v0.x)) { v0.x = 0.0; } if(!isvalid(v0.y)) { v0.y = 0.0; } if(!isvalid(v0.z)) { v0.z = 0.0; }
      
      // use forward Euler method to advect
      VT3 p2 = integrateForwardEuler(src.v.dData, p, v0, dt);

      putTex2DD(dst.v.dData,   v0,  p2, fp);
      putTex2DD(dst.Qn.dData,  Qn,  p2, fp);
      putTex2DD(dst.Qp.dData,  Qp,  p2, fp);
      putTex2DD(dst.Qnv.dData, Qnv, p2, fp);
      putTex2DD(dst.Qpv.dData, Qpv, p2, fp);

      
      // //p2 = (VT3{(max(0.0f, min(p2.x, float(src.size.x-1))));
      // //          (max(0.0f, min(p2.y, float(src.size.y-1)))),
      // //          (max(0.0f, min(p2.z, float(src.size.z-1)))) });
      // int4   tiX = texPutIX(p2, fp); int4 tiY = texPutIY(p2, fp); int4 tiZ = texPutIZ(p2, fp);
      // float4 mults0 = texPutMults0<float>(p2); float4 mults1 = texPutMults1<float>(p2);
      // IT p000 = IT{tiX.x, tiY.x, tiZ.x}; IT p100 = IT{tiX.y, tiY.y, tiZ.x};
      // IT p010 = IT{tiX.z, tiY.z, tiZ.x}; IT p110 = IT{tiX.w, tiY.w, tiZ.x};
      // IT p001 = IT{tiX.x, tiY.x, tiZ.z}; IT p101 = IT{tiX.y, tiY.y, tiZ.z};
      // IT p011 = IT{tiX.z, tiY.z, tiZ.z}; IT p111 = IT{tiX.w, tiY.w, tiZ.z};

      // // scale value by grid overlap and store in each location
      // // v
      // texAtomicAdd(dst.v.dData,    v0*mults0.x,   p000, fp); texAtomicAdd(dst.v.dData,    v0*mults0.z,   p010, fp);
      // texAtomicAdd(dst.v.dData,    v0*mults0.y,   p100, fp); texAtomicAdd(dst.v.dData,    v0*mults0.w,   p110, fp);
      // texAtomicAdd(dst.v.dData,    v0*mults1.x,   p001, fp); texAtomicAdd(dst.v.dData,    v0*mults1.z,   p011, fp);
      // texAtomicAdd(dst.v.dData,    v0*mults1.y,   p101, fp); texAtomicAdd(dst.v.dData,    v0*mults1.w,   p111, fp);
      // // // p
      // // texAtomicAdd(dst.p.dData,    p0*mults0.x,   p000, fp); texAtomicAdd(dst.p.dData,    p0*mults0.z,   p010, fp);
      // // texAtomicAdd(dst.p.dData,    p0*mults0.y,   p100, fp); texAtomicAdd(dst.p.dData,    p0*mults0.w,   p110, fp);
      // // texAtomicAdd(dst.p.dData,    p0*mults1.x,   p001, fp); texAtomicAdd(dst.p.dData,    p0*mults1.z,   p011, fp);
      // // texAtomicAdd(dst.p.dData,    p0*mults1.y,   p101, fp); texAtomicAdd(dst.p.dData,    p0*mults1.w,   p111, fp);
      // // Qn                                                                                              
      // texAtomicAdd(dst.Qn.dData,   Qn*mults0.x,   p000, fp); texAtomicAdd(dst.Qn.dData,   Qn*mults0.z,   p010, fp);
      // texAtomicAdd(dst.Qn.dData,   Qn*mults0.y,   p100, fp); texAtomicAdd(dst.Qn.dData,   Qn*mults0.w,   p110, fp);
      // texAtomicAdd(dst.Qn.dData,   Qn*mults1.x,   p001, fp); texAtomicAdd(dst.Qn.dData,   Qn*mults1.z,   p011, fp);
      // texAtomicAdd(dst.Qn.dData,   Qn*mults1.y,   p101, fp); texAtomicAdd(dst.Qn.dData,   Qn*mults1.w,   p111, fp);
      // // Qp                                                                                              
      // texAtomicAdd(dst.Qp.dData,   Qp*mults0.x,   p000, fp); texAtomicAdd(dst.Qp.dData,   Qp*mults0.z,   p010, fp);
      // texAtomicAdd(dst.Qp.dData,   Qp*mults0.y,   p100, fp); texAtomicAdd(dst.Qp.dData,   Qp*mults0.w,   p110, fp);
      // texAtomicAdd(dst.Qp.dData,   Qp*mults1.x,   p001, fp); texAtomicAdd(dst.Qp.dData,   Qp*mults1.z,   p011, fp);
      // texAtomicAdd(dst.Qp.dData,   Qp*mults1.y,   p101, fp); texAtomicAdd(dst.Qp.dData,   Qp*mults1.w,   p111, fp);
      // // Qv
      // texAtomicAdd(dst.Qnv.dData,  Qnv*mults0.x,  p000, fp); texAtomicAdd(dst.Qnv.dData,  Qnv*mults0.z,  p010, fp);
      // texAtomicAdd(dst.Qnv.dData,  Qnv*mults0.y,  p100, fp); texAtomicAdd(dst.Qnv.dData,  Qnv*mults0.w,  p110, fp);
      // texAtomicAdd(dst.Qnv.dData,  Qnv*mults1.x,  p001, fp); texAtomicAdd(dst.Qnv.dData,  Qnv*mults1.z,  p011, fp);
      // texAtomicAdd(dst.Qnv.dData,  Qnv*mults1.y,  p101, fp); texAtomicAdd(dst.Qnv.dData,  Qnv*mults1.w,  p111, fp);
      // texAtomicAdd(dst.Qpv.dData,  Qpv*mults0.x,  p000, fp); texAtomicAdd(dst.Qpv.dData,  Qpv*mults0.z,  p010, fp);
      // texAtomicAdd(dst.Qpv.dData,  Qpv*mults0.y,  p100, fp); texAtomicAdd(dst.Qpv.dData,  Qpv*mults0.w,  p110, fp);
      // texAtomicAdd(dst.Qpv.dData,  Qpv*mults1.x,  p001, fp); texAtomicAdd(dst.Qpv.dData,  Qpv*mults1.z,  p011, fp);
      // texAtomicAdd(dst.Qpv.dData,  Qpv*mults1.y,  p101, fp); texAtomicAdd(dst.Qpv.dData,  Qpv*mults1.w,  p111, fp);

      // // // E
      // // texAtomicAdd(dst.E.dData,    E0*mults0.x,   p000, fp); texAtomicAdd(dst.E.dData,    E0*mults0.z,   p010, fp);
      // // texAtomicAdd(dst.E.dData,    E0*mults0.y,   p100, fp); texAtomicAdd(dst.E.dData,    E0*mults0.w,   p110, fp);
      // // texAtomicAdd(dst.E.dData,    E0*mults1.x,   p001, fp); texAtomicAdd(dst.E.dData,    E0*mults1.z,   p011, fp);
      // // texAtomicAdd(dst.E.dData,    E0*mults1.y,   p101, fp); texAtomicAdd(dst.E.dData,    E0*mults1.w,   p111, fp);
      // // // B
      // // texAtomicAdd(dst.B.dData,    B0*mults0.x,   p000, fp); texAtomicAdd(dst.B.dData,    B0*mults0.z,   p010, fp);
      // // texAtomicAdd(dst.B.dData,    B0*mults0.y,   p100, fp); texAtomicAdd(dst.B.dData,    B0*mults0.w,   p110, fp);
      // // texAtomicAdd(dst.B.dData,    B0*mults1.x,   p001, fp); texAtomicAdd(dst.B.dData,    B0*mults1.z,   p011, fp);
      // // texAtomicAdd(dst.B.dData,    B0*mults1.y,   p101, fp); texAtomicAdd(dst.B.dData,    B0*mults1.w,   p111, fp);
      // // // mat
      // // texAtomicAdd(dst.mat.dData,  mat*mults0.x,  p000, fp); texAtomicAdd(dst.mat.dData,  mat*mults0.z,  p010, fp);
      // // texAtomicAdd(dst.mat.dData,  mat*mults0.y,  p100, fp); texAtomicAdd(dst.mat.dData,  mat*mults0.w,  p110, fp);
      // // texAtomicAdd(dst.mat.dData,  mat*mults1.x,  p001, fp); texAtomicAdd(dst.mat.dData,  mat*mults1.z,  p011, fp);
      // // texAtomicAdd(dst.mat.dData,  mat*mults1.y,  p101, fp); texAtomicAdd(dst.mat.dData,  mat*mults1.w,  p111, fp);

      // (unchanged) // TODO: segment fields to avoid unnecessary copying
      // dst.Qnv[i] = Qnv; dst.Qpv[i] = Qpv;
      dst.E[i]   = E0;
      dst.B[i]   = B0;
      dst.mat[i] = mat;
      dst.div[i] = d0;
    }
}




//// BOUNDARIES (velocity) ////

template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT=typename Dim<VT3>::SIZE_T>
__global__ void velocityBounds_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> fp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT ip = makeI<IT>(ix, iy, iz); int i = src.idx(ip);
      IT p0 = applyBounds(ip, src.size, fp);

      // apply boundaries
      VT3 v = VT3{0,0,0};
      // int3 count = int3{0,0,0};
      int vpx = 0; int vpy = 0; int vpz = 0;
      if(p0.x == 0 && p0.x < src.size.x && fp.edgeNX == EDGE_FREESLIP)     { vpx++; }  // { v.x -= src.v[src.idx(p0+IT{1,0,0})].x; count.x++; } //
      if(p0.y == 0 && p0.y < src.size.y && fp.edgeNY == EDGE_FREESLIP)     { vpy++; }  // { v.y -= src.v[src.idx(p0+IT{0,1,0})].y; count.y++; } //
      if(p0.z == 0 && p0.z < src.size.z && fp.edgeNZ == EDGE_FREESLIP)     { vpz++; }  // { v.z -= src.v[src.idx(p0+IT{0,0,1})].z; count.z++; } //
      if(p0.x == src.size.x-1 && p0.x >= 0 && fp.edgePX == EDGE_FREESLIP)  { vpx--; }  // { v.x -= src.v[src.idx(p0-IT{1,0,0})].x; count.x++; } //
      if(p0.y == src.size.y-1 && p0.y >= 0 && fp.edgePY == EDGE_FREESLIP)  { vpy--; }  // { v.y -= src.v[src.idx(p0-IT{0,1,0})].y; count.y++; } //
      if(p0.z == src.size.z-1 && p0.z >= 0 && fp.edgePZ == EDGE_FREESLIP)  { vpz--; }  // { v.z -= src.v[src.idx(p0-IT{0,0,1})].z; count.z++; } //
      // v = (VT3{(count.x == 0 ? src.v[i].x : v.x/(T)count.x),
      //          (count.y == 0 ? src.v[i].y : v.y/(T)count.y),
      //          (count.z == 0 ? src.v[i].z : v.z/(T)count.z)});
      v = (VT3{vpx==0 ? src.v[i].x : -src.v[src.idx(ip+IT{vpx,0,0})].x,
               vpy==0 ? src.v[i].y : -src.v[src.idx(ip+IT{0,vpy,0})].y,
               vpz==0 ? src.v[i].z : -src.v[src.idx(ip+IT{0,0,vpz})].z});

      // if(ip.x == 1 || ip.x == src.size.x-2) { v.x = 0.0; }
      // else if(ip.y == 1 || ip.y == src.size.y-2) { v.y = 0.0; }
      // else if(ip.z == 1 || ip.z == src.size.z-2) { v.z = 0.0; }
      
      dst.v[i] = v;
      dst.p[i] = src.p[i];
      dst.div[i] = src.div[i];
    }
}




//// EXTERNAL FORCES (e.g. gravity) ////

template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT=typename Dim<VT3>::SIZE_T>
__global__ void externalForces_k(FluidField<T> src, FluidParams<T> fp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      int i  = src.idx(makeI<IT>(ix, iy, iz));
      src.v[i] += fp.gravity * fp.u.dt;
    }
}







//// VISCOSITY ////

template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT=typename Dim<VT3>::SIZE_T>
__global__ void viscosityIter_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> fp)
{
  int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT  ip = makeI<IT>(ix, iy, iz); int i = src.idx(ip);
      IT p0 = applyBounds(ip,   src.size, fp); // current cell
      IT pN = applyBounds(ip-1, src.size, fp); // cell - (1,1)
      IT pP = applyBounds(ip+1, src.size, fp); // cell + (1,1)
      
      VT3 v = src.v[i];
      if(p0 >= 0 && pP >= 0 && pN >= 0)
        {
          // jacobi parameters --> see: https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
          T alpha = fp.u.dL*fp.u.dL*fp.u.dL/(fp.u.dt*fp.viscosity); // α = dx³/(ν*dt)
          T beta  = 1.0 / (6.0 + alpha);                            // β = dx³/(ν*dt)
          v -= jacobi(src.v, p0, pP, pN, alpha, beta);
        }
      dst.v[i] = (isvalid(v) ? v : src.v[i]);
    }
}

  
//// PRESSURE ////

template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT=typename Dim<VT3>::SIZE_T>
__global__ void pressurePre_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> fp)
{
  int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT ip = makeI<IT>(ix, iy, iz); int i = src.idx(ip);
      IT p0 = applyBounds(ip,   src.size, fp); // current cell
      IT pN = applyBounds(ip-1, src.size, fp); // cell - (1,1)
      IT pP = applyBounds(ip+1, src.size, fp); // cell + (1,1)
      // calculate divergence
      T p = src.p[i];
      T d = 0;
      // if(p0 >= 0 && pP >= 0 && pN >= 0)
      //   {
          // d = divergence(src.v, p00, pp1, pn1, fp.u.dL) / fp.density;
          
          d = ((((pP.x < 0 || pP.x >= src.size.x) ? 0 : src.v[src.idx(pP.x, p0.y, p0.z)].x) -
                ((pN.x < 0 || pN.x >= src.size.x) ? 0 : src.v[src.idx(pN.x, p0.y, p0.z)].x)) +
               (((pP.y < 0 || pP.y >= src.size.y) ? 0 : src.v[src.idx(p0.x, pP.y, p0.z)].y) -
                ((pN.y < 0 || pN.y >= src.size.y) ? 0 : src.v[src.idx(p0.x, pN.y, p0.z)].y)) +
               (((pP.z < 0 || pP.z >= src.size.z) ? 0 : src.v[src.idx(p0.x, p0.y, pP.z)].z) -
                ((pN.z < 0 || pN.z >= src.size.z) ? 0 : src.v[src.idx(p0.x, p0.y, pN.z)].z))) / (2.0f * fp.u.dL);
        // }
      // else
      //   {
      //     // // boundaries
      //     // if(p0.x == 0            && (fp.edgeNX == EDGE_FREESLIP)) { p = src.p[src.idx(p0+IT{1,0,0})]; }
      //     // if(p0.y == 0            && (fp.edgeNY == EDGE_FREESLIP)) { p = src.p[src.idx(p0+IT{0,1,0})]; }
      //     // if(p0.z == 0            && (fp.edgeNZ == EDGE_FREESLIP)) { p = src.p[src.idx(p0+IT{0,0,1})]; }
      //     // if(p0.x == src.size.x-1 && (fp.edgePX == EDGE_FREESLIP)) { p = src.p[src.idx(p0-IT{1,0,0})]; }
      //     // if(p0.y == src.size.y-1 && (fp.edgePY == EDGE_FREESLIP)) { p = src.p[src.idx(p0-IT{0,1,0})]; }
      //     // if(p0.z == src.size.z-1 && (fp.edgePZ == EDGE_FREESLIP)) { p = src.p[src.idx(p0-IT{0,0,1})]; }
      //     if(p0.x == 0            && (fp.edgeNX == EDGE_FREESLIP)) {  }
      //     if(p0.y == 0            && (fp.edgeNY == EDGE_FREESLIP)) {  }
      //     if(p0.z == 0            && (fp.edgeNZ == EDGE_FREESLIP)) {  }
      //     if(p0.x == src.size.x-1 && (fp.edgePX == EDGE_FREESLIP)) {  }
      //     if(p0.y == src.size.y-1 && (fp.edgePY == EDGE_FREESLIP)) {  }
      //     if(p0.z == src.size.z-1 && (fp.edgePZ == EDGE_FREESLIP)) {  }

      dst.v[i]   = src.v[i];
      dst.p[i]   = (isvalid(p) ? p : 0);
      dst.div[i] = (isvalid(d) ? d : 0);
    }
}


template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT=typename Dim<VT3>::SIZE_T>
__global__ void pressureIter_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> fp)
{
  int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT  ip = makeI<IT>(ix, iy, iz); int i = src.idx(ip);
      IT p0 = applyBounds(ip,   src.size, fp); // current cell
      IT pP = applyBounds(ip+1, src.size, fp); // cell + (1,1,1)
      IT pN = applyBounds(ip-1, src.size, fp); // cell - (1,1,1)
      
      // iterate --> solve pressure poisson equation (Jacobi method)
      // T p = 0; T d = src.div[i];
      // if(p0 >= 0 && pP >= 0 && pN >= 0)
      //   {
      //     // jacobi parameters --> see: https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
      //     // T alpha = -fp.u.dL*fp.u.dL; // dL²
      //     // T beta  = src.div[i] / 6.0;
      //     // p = jacobi(src.p, p00, pp1, pn1, alpha, beta, fp.u.dL, fp.u.dt);
      //     dst.p[i] = (src.p[src.idx(pN.x, p0.y, p0.z)] + src.p[src.idx(pP.x, p0.y, p0.z)] +
      //                 src.p[src.idx(p0.x, pN.y, p0.z)] + src.p[src.idx(p0.x, pP.y, p0.z)] +
      //                 src.p[src.idx(p0.x, p0.y, pN.z)] + src.p[src.idx(p0.x, p0.y, pP.z)] - d) / 6.0f;
      //   }
      
      T p = 0; T d = src.div[i];
      int count = 0;
      if(p0 >= 0 && (pP >= 0 || pN >= 0))
        {
          if(pP.x >= 1 && pP.x < src.size.x-1) { p += src.p[src.idx(pP.x, p0.y, p0.z)]; count++; }
          if(pP.y >= 1 && pP.y < src.size.y-1) { p += src.p[src.idx(p0.x, pP.y, p0.z)]; count++; }
          if(pP.z >= 1 && pP.z < src.size.z-1) { p += src.p[src.idx(p0.x, p0.y, pP.z)]; count++; }
          if(pN.x >= 1 && pN.x < src.size.x-1) { p += src.p[src.idx(pN.x, p0.y, p0.z)]; count++; }
          if(pN.y >= 1 && pN.y < src.size.y-1) { p += src.p[src.idx(p0.x, pN.y, p0.z)]; count++; }
          if(pN.z >= 1 && pN.z < src.size.z-1) { p += src.p[src.idx(p0.x, p0.y, pN.z)]; count++; }
          p = count > 0 ? ((p - d)/(T)count) : p;
        }
      dst.p[i]   = (isvalid(p) ? p : 0);
      dst.div[i] = (isvalid(d) ? d : 0);
    }
}

template<typename T,
         typename VT3 = typename DimType<T, 3>::VEC_T,
         typename IT  = typename Dim<VT3>::SIZE_T>
__global__ void pressurePost_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> fp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT  ip = makeI<IT>(ix, iy, iz);
      int i  = src.idx(ip);
      IT p0 = applyBounds(ip,   src.size, fp); // current cell
      IT pP = applyBounds(ip+1, src.size, fp); // cell + (1,1,1)
      IT pN = applyBounds(ip-1, src.size, fp); // cell - (1,1,1)
      
      // // subtract gradient of pressure from velocity
      // if(p0 >= 0 && pP >= 0 && pN >= 0)
      //   {
      //     // VT3 h = 1.0/makeV<VT3>(src.size);
      //     dst.v[i] += (VT3{(src.p[src.idx(pN.x, p0.y, p0.z)] - src.p[src.idx(pP.x, p0.y, p0.z)]),
      //                      (src.p[src.idx(p0.x, pN.y, p0.z)] - src.p[src.idx(p0.x, pP.y, p0.z)]),
      //                      (src.p[src.idx(p0.x, p0.y, pN.z)] - src.p[src.idx(p0.x, p0.y, pP.z)])}) / (2.0 * fp.density);
      //     // dst.v[i] -= gradient(dst.p, p0, pP, pN, fp.u.dL);
      //   }
      
      VT3 v  = src.v[i];
      VT3 dv = VT3{0,0,0};
      VT3 dL = VT3{2,2,2}*fp.u.dL;
      if(// p0 >= 0 || 
         pP >= 0 || pN >= 0)
        {
          if(pP.x < src.size.x) { dv.x += src.p[src.idx(pP.x, p0.y, p0.z)]; } // else { dv.x += v.x; dL.x /= 2.0; }
          if(pP.y < src.size.y) { dv.y += src.p[src.idx(p0.x, pP.y, p0.z)]; } // else { dv.y += v.y; dL.y /= 2.0; }
          if(pP.z < src.size.z) { dv.z += src.p[src.idx(p0.x, p0.y, pP.z)]; } // else { dv.z += v.z; dL.z /= 2.0; }
          if(pN.x >= 0)         { dv.x -= src.p[src.idx(pN.x, p0.y, p0.z)]; } // else { dv.x -= v.x; dL.x /= 2.0; }
          if(pN.y >= 0)         { dv.y -= src.p[src.idx(p0.x, pN.y, p0.z)]; } // else { dv.y -= v.y; dL.y /= 2.0; }
          if(pN.z >= 0)         { dv.z -= src.p[src.idx(p0.x, p0.y, pN.z)]; } // else { dv.z -= v.z; dL.z /= 2.0; }
          v -= dv/(dL*fp.density);
        }
      // else
      //   { // edge cell -- apply boundaries
      //     v  = VT3{0,0,0};
      //     int3 count = int3{0,0,0};
      //     if(p0.x == 0 && p0.x < src.size.x-1 && fp.edgeNX == EDGE_FREESLIP) { v.x -= src.v[src.idx(p0+IT{1,0,0})].x; count.x++; }
      //     if(p0.y == 0 && p0.y < src.size.y-1 && fp.edgeNY == EDGE_FREESLIP) { v.y -= src.v[src.idx(p0+IT{0,1,0})].y; count.y++; }
      //     if(p0.z == 0 && p0.z < src.size.z-1 && fp.edgeNZ == EDGE_FREESLIP) { v.z -= src.v[src.idx(p0+IT{0,0,1})].z; count.z++; }
      //     if(p0.x == src.size.x-1 && p0.x > 0 && fp.edgePX == EDGE_FREESLIP) { v.x -= src.v[src.idx(p0-IT{1,0,0})].x; count.x++; }
      //     if(p0.y == src.size.y-1 && p0.y > 0 && fp.edgePY == EDGE_FREESLIP) { v.y -= src.v[src.idx(p0-IT{0,1,0})].y; count.y++; }
      //     if(p0.z == src.size.z-1 && p0.z > 0 && fp.edgePZ == EDGE_FREESLIP) { v.z -= src.v[src.idx(p0-IT{0,0,1})].z; count.z++; }
      //     dst.v[i] = (VT3{((count.x == 0) ? 0 : v.x/(T)count.x),
      //                     ((count.y == 0) ? 0 : v.y/(T)count.y),
      //                     ((count.z == 0) ? 0 : v.z/(T)count.z)});
      //   }

      dst.v[i]   = v;
      dst.p[i]   = src.p[i];
      dst.div[i] = src.div[i];
    }
}

template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT=typename Dim<VT3>::SIZE_T>
__global__ void pressureBounds_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> fp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT  ip = makeI<IT>(ix, iy, iz);
      int i  = src.idx(ip);
      IT p0 = applyBounds(ip, src.size, fp); // current cell

      // boundaries
      T p = 0; int count = 0;
      // int ppx = 0; int ppy = 0; int ppz = 0;
      if(p0.x == 0 && p0.x < src.size.x    && fp.edgeNX == EDGE_FREESLIP) { p += src.p[src.idx(p0+IT{1,0,0})]; count++; }//ppx++; } // 
      if(p0.y == 0 && p0.y < src.size.y    && fp.edgeNY == EDGE_FREESLIP) { p += src.p[src.idx(p0+IT{0,1,0})]; count++; }//ppy++; } // 
      if(p0.z == 0 && p0.z < src.size.z    && fp.edgeNZ == EDGE_FREESLIP) { p += src.p[src.idx(p0+IT{0,0,1})]; count++; }//ppz++; } // 
      if(p0.x == src.size.x-1 && p0.x >= 0 && fp.edgePX == EDGE_FREESLIP) { p += src.p[src.idx(p0-IT{1,0,0})]; count++; }//ppx++; } // 
      if(p0.y == src.size.y-1 && p0.y >= 0 && fp.edgePY == EDGE_FREESLIP) { p += src.p[src.idx(p0-IT{0,1,0})]; count++; }//ppy++; } // 
      if(p0.z == src.size.z-1 && p0.z >= 0 && fp.edgePZ == EDGE_FREESLIP) { p += src.p[src.idx(p0-IT{0,0,1})]; count++; }//ppz++; } // 
      dst.p[i]   = (count == 0 ? src.p[i] : p/(T)count);
      //dst.p[i]   = src.p[src.idx(ip+IT{ppx, ppy, ppz})]; //
      dst.v[i]   = src.v[i];
      dst.div[i] = src.div[i];
    }
}


// wrappers

//// ADVECTION ////
template<typename T> void fluidAdvect(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> fp)
{
  if(src.size > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      // set to zero for forward euler method -- kernel will re-add contents
      dst.v.clear(); //dst.p.clear(); 
      dst.Qn.clear(); dst.Qp.clear(); dst.E.clear(); dst.B.clear(); dst.Qnv.clear(); dst.Qpv.clear();// dst.mat.clear();
      advect_k<<<grid, threads>>>(src, dst, fp);
      // dst.copyTo(src);
      velocityBounds_k<<<grid, threads>>>(dst, dst, fp);
    }
}

//// EXTERNAL FORCES (e.g. gravity) ////
////// NOTE: in-place
template<typename T> void fluidExternalForces(FluidField<T> &src, FluidParams<T> fp)
{
  if(src.size > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      externalForces_k<<<grid, threads>>>(src, fp);
    }
}

//// VISCOSITY ////
template<typename T> void fluidViscosity(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> fp, int iter)
{
  if(src.size > 0 && dst.size == src.size)
    {
      src.copyTo(dst);
      if(iter > 0)
        {
          dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
          dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                    (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                    (int)ceil(src.size.z/(float)BLOCKDIM_Z));
          
          FluidField<T> *temp1 = &src; FluidField<T> *temp2 = &dst;
          for(int i = 0; i < iter; i++)
            { viscosityIter_k<<<grid, threads>>>(*temp1, *temp2, fp); std::swap(temp1, temp2); }
          
          if(temp1 != &dst) { src.v.copyTo(dst.v); } // make sure dst contains final state
        }
    }
}

//// PRESSURE ////
template<typename T> void fluidPressure(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> fp, int iter)
{
  if(src.size > 0 && dst.size == src.size)
    {
      src.copyTo(dst);
      if(iter > 0)
        {
          dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
          dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                    (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                    (int)ceil(src.size.z/(float)BLOCKDIM_Z));
          
          FluidField<T> *temp1 = &src; FluidField<T> *temp2 = &dst;
          // pre
          pressurePre_k<<<grid, threads>>>(*temp1, *temp2, fp);      std::swap(temp1, temp2);
          // iteration
          for(int i = 0; i < iter; i++)
            { pressureIter_k<<<grid, threads>>>(*temp1, *temp2, fp); std::swap(temp1, temp2); }
          // post
          pressurePost_k<<<grid, threads>>>(*temp1, *temp2, fp);     std::swap(temp1, temp2);

          // apply pressure bounds
          pressureBounds_k<<<grid, threads>>>(*temp1, *temp2, fp);   std::swap(temp1, temp2);
          
          // // apply velocity bounds
          // velocityBounds_k<<<grid, threads>>>(*temp1, *temp2, fp);   std::swap(temp1, temp2);
          
          if(temp1 != &dst) // make sure final result is in dst
            { temp1->v.copyTo(dst.v); temp1->p.copyTo(dst.p); temp1->div.copyTo(dst.div); }
        }
    }
}




template void fluidAdvect        <float>(FluidField<float> &src, FluidField<float> &dst, FluidParams<float> fp);
template void fluidExternalForces<float>(FluidField<float> &src, FluidParams<float> fp);
template void fluidViscosity     <float>(FluidField<float> &src, FluidField<float> &dst, FluidParams<float> fp, int iter);
template void fluidPressure      <float>(FluidField<float> &src, FluidField<float> &dst, FluidParams<float> fp, int iter);
