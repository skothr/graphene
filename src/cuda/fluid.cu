#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>

#include "cuda-tools.cuh"
#include "vector-calc.cuh"
#include "fluid.cuh"
#include "physics.h"

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8
#define BLOCKDIM_Z 8


////////////////////////////////////////////////////////////////////////////////////////////////
//// kernels
////////////////////////////////////////////////////////////////////////////////////////////////


//// ADVECTION ////

template<typename T, typename VT3=typename cuda_vec<T, 3>::VT, typename IT=typename cuda_vec<VT3>::IT>
__global__ void advect_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> fp)
{
  const int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  const int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  const int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  const IT  ip   = makeV<IT>(ix, iy, iz);
  if(ip.x < src.size.x && ip.y < src.size.y && ip.z < src.size.z)
    {
      const int i  = src.idx(ip);
      const VT3 p  = VT3{T(ix), T(iy), T(iz)};
      VT3 p2; // (sample point for integration)
      
      VT3 v = src.v.dData[i];
      T Qn; T Qp; VT3 Qnv; VT3 Qpv; //VT3 E; VT3 B;
      if(isImplicit(fp.vIntegration))
        { //// integrate for implicit methods (sample value from previous time point)
          if(fp.vIntegration == INTEGRATION_BACKWARD_EULER) { p2 = integrateBackwardEuler(src.v.dData, p, v, fp.u.dt); }
          v   = tex2DD(src.v.dData,   p2, fp);
          Qn  = tex2DD(src.Qn.dData,  p2, fp);
          Qp  = tex2DD(src.Qp.dData,  p2, fp);
          Qnv = tex2DD(src.Qnv.dData, p2, fp);
          Qpv = tex2DD(src.Qpv.dData, p2, fp);
        }
      else
        {
          Qn  = src.Qn[i];
          Qp  = src.Qp[i];
          Qnv = src.Qnv[i];
          Qpv = src.Qpv[i];
        }
      
      // check for invalid values
      if(!isvalid(v.x) || !isvalid(v.x)) { v.x = 0.0; } if(!isvalid(v.y)) { v.y = 0.0; } if(!isvalid(v.z)) { v.z = 0.0; }
      // limit magnitude (removes some energy from the system if enabled)
      if(fp.limitV) { v = normalize(v)*min(length(v), fp.maxV); }
      // check for invalid values
      if(!isvalid(v.x) || !isvalid(v.x)) { v.x = 0.0; } if(!isvalid(v.y)) { v.y = 0.0; } if(!isvalid(v.z)) { v.z = 0.0; }
      
      //// integrate
      if(fp.vIntegration == INTEGRATION_BACKWARD_EULER)
        { //// write final values for implicit methods (current cell)
          dst.v[i]   = v;
          dst.Qn[i]  = Qn;
          dst.Qp[i]  = Qp;
          dst.Qnv[i] = Qnv;
          dst.Qpv[i] = Qpv;
        }
      else
        { //// integrate for explicit methods (write atomically to advected position)
          if(fp.vIntegration == INTEGRATION_FORWARD_EULER) { p2 = integrateForwardEuler(src.v.dData, p, v, fp.u.dt); }
          else if(fp.vIntegration == INTEGRATION_RK4)      { p2 = integrateRK4(src.v, p, fp); }
          putTex2DD(dst.v.dData,   v,   p2, fp);
          putTex2DD(dst.Qn.dData,  Qn,  p2, fp);
          putTex2DD(dst.Qp.dData,  Qp,  p2, fp);
          putTex2DD(dst.Qnv.dData, Qnv, p2, fp);
          putTex2DD(dst.Qpv.dData, Qpv, p2, fp);
        }
    }
}




//// BOUNDARIES (velocity) ////

template<typename T, typename VT3=typename cuda_vec<T, 3>::VT, typename IT=typename cuda_vec<VT3>::IT>
__global__ void velocityBounds_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> fp)
{
  const int ix = blockIdx.x*blockDim.x + threadIdx.x;
  const int iy = blockIdx.y*blockDim.y + threadIdx.y;
  const int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      const IT ip = makeV<IT>(ix, iy, iz); const int i = src.idx(ip);
      const IT p0 = applyBounds(ip,   src.size, fp); // current cell
      const IT pN = applyBounds(ip-1, src.size, fp); // cell - (1,1)
      const IT pP = applyBounds(ip+1, src.size, fp); // cell + (1,1)
      
      // apply boundaries
      IT sp = makeV<IT>(0);
      if(p0.x == 0 && p0.x  < src.size.x-1 && fp.edgeNX == BOUND_SLIP) { sp.x++; }
      if(p0.y == 0 && p0.y  < src.size.y-1 && fp.edgeNY == BOUND_SLIP) { sp.y++; }
      if(p0.z == 0 && p0.z  < src.size.z-1 && fp.edgeNZ == BOUND_SLIP) { sp.z++; }
      if(p0.x >= 0 && p0.x == src.size.x-1 && fp.edgePX == BOUND_SLIP) { sp.x--; }
      if(p0.y >= 0 && p0.y == src.size.y-1 && fp.edgePY == BOUND_SLIP) { sp.y--; }
      if(p0.z >= 0 && p0.z == src.size.z-1 && fp.edgePZ == BOUND_SLIP) { sp.z--; }
      const VT3 v0 = src.v[i];
      dst.v[i] = (VT3{(sp.x == 0 ? v0.x : -src.v[src.idx(ip+IT{sp.x, 0, 0})].x),
                      (sp.y == 0 ? v0.y : -src.v[src.idx(ip+IT{0, sp.y, 0})].y),
                      (sp.z == 0 ? v0.z : -src.v[src.idx(ip+IT{0, 0, sp.z})].z)});
    }
}




//// EXTERNAL FORCES (e.g. gravity) ////

template<typename T, typename VT3=typename cuda_vec<T, 3>::VT, typename IT=typename cuda_vec<VT3>::IT>
__global__ void externalForces_k(FluidField<T> src, FluidParams<T> fp)
{
  const int ix = blockIdx.x*blockDim.x + threadIdx.x;
  const int iy = blockIdx.y*blockDim.y + threadIdx.y;
  const int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      const int i = src.idx(makeV<IT>(ix, iy, iz));
      src.v[i] += fp.gravity * fp.u.dt;
    }
}







//// VISCOSITY ////

template<typename T, typename VT3=typename cuda_vec<T, 3>::VT, typename IT=typename cuda_vec<VT3>::IT>
__global__ void viscosityIter_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> fp)
{
  const int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  const int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  const int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      const IT ip = makeV<IT>(ix, iy, iz); const int i = src.idx(ip);
      const IT p0 = applyBounds(ip,   src.size, fp); // current cell
      const IT pN = applyBounds(ip-1, src.size, fp); // cell - (1,1)
      const IT pP = applyBounds(ip+1, src.size, fp); // cell + (1,1)
      
      VT3 v = src.v[i];
      if(pP >= 0 && pN >= 0)
        { // jacobi parameters --> see: https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
          const T alpha = fp.u.dL*fp.u.dL*fp.u.dL/(fp.u.dt*fp.viscosity); // α = dx*dy*dz/(η*dt)
          const T beta  = 1.0 / (6.0 + alpha);                            // β = 1/(6+α)
          v = jacobi(src.v, p0, pP, pN, alpha, beta);
        }
      dst.v[i] = (isvalid(v) ? v : src.v[i]);
    }
}

  
//// PRESSURE ////

template<typename T, typename VT3=typename cuda_vec<T, 3>::VT, typename IT=typename cuda_vec<VT3>::IT>
__global__ void pressurePre_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> fp)
{
  const int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  const int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  const int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      const IT ip = makeV<IT>(ix, iy, iz); const int i = src.idx(ip);
      const IT p0 = applyBounds(ip,   src.size, fp); // current cell
      const IT pN = applyBounds(ip-1, src.size, fp); // cell - (1,1)
      const IT pP = applyBounds(ip+1, src.size, fp); // cell + (1,1)
      
      // calculate divergence
      // const T div = divergence(src.v, p00, pp1, pn1, fp.u.dL) / fp.density;
      
      const T div = ((((pP.x >= 0 && pP.x < src.size.x) ? src.v[src.idx(pP.x, p0.y, p0.z)].x : 0) -
                      ((pN.x >= 0 && pN.x < src.size.x) ? src.v[src.idx(pN.x, p0.y, p0.z)].x : 0)) +
                     (((pP.y >= 0 && pP.y < src.size.y) ? src.v[src.idx(p0.x, pP.y, p0.z)].y : 0) -
                      ((pN.y >= 0 && pN.y < src.size.y) ? src.v[src.idx(p0.x, pN.y, p0.z)].y : 0)) +
                     (((pP.z >= 0 && pP.z < src.size.z) ? src.v[src.idx(p0.x, p0.y, pP.z)].z : 0) -
                      ((pN.z >= 0 && pN.z < src.size.z) ? src.v[src.idx(p0.x, p0.y, pN.z)].z : 0))) / (2.0*fp.u.dL);
      const T pressure = (fp.clearPressure ? (T)0 : src.p[i]);

      dst.v[i]   = src.v[i];
      dst.p[i]   = (isvalid(pressure) ? pressure : 0);
      dst.div[i] = (isvalid(div)      ? div      : 0);
    }
}


template<typename T, typename VT3=typename cuda_vec<T, 3>::VT, typename IT=typename cuda_vec<VT3>::IT>
__global__ void pressureIter_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> fp)
{
  const int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  const int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  const int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      const IT ip = makeV<IT>(ix, iy, iz); const int i = src.idx(ip);
      const IT p0 = applyBounds(ip,   src.size, fp); // current cell
      const IT pP = applyBounds(ip+1, src.size, fp); // cell + (1,1,1)
      const IT pN = applyBounds(ip-1, src.size, fp); // cell - (1,1,1)
      
      // iterate --> solve pressure poisson equation (Jacobi method)
      // T p = 0; T d = src.div[i];
      // if(pP >= 0 && pN >= 0)
      //   {
      //     // jacobi parameters --> see: https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
      //     // const T alpha = fp.u.dL*fp.u.dL*fp.u.dL/(fp.u.dt*fp.density); // α = dx*dy*dz/(ρ*dt)
      //     // const T beta  = 1.0 / (6.0 + alpha);                          // β = 1/(6+α)
      //     // p = jacobi(src.p, p00, pp1, pn1, alpha, beta, fp.u.dL, fp.u.dt);
      //   }
      
      T pressure = 0.0; T count = 0.0;
      if((pP.x > 0 && pP.x < src.size.x-1) || (pP.x >= 0 && fp.edgePX != BOUND_NOSLIP)) { pressure += src.p[src.idx(pP.x, p0.y, p0.z)]; count += 1.0; }
      if((pP.y > 0 && pP.y < src.size.y-1) || (pP.y >= 0 && fp.edgePY != BOUND_NOSLIP)) { pressure += src.p[src.idx(p0.x, pP.y, p0.z)]; count += 1.0; }
      if((pP.z > 0 && pP.z < src.size.z-1) || (pP.z >= 0 && fp.edgePZ != BOUND_NOSLIP)) { pressure += src.p[src.idx(p0.x, p0.y, pP.z)]; count += 1.0; }
      if((pN.x > 0 && pN.x < src.size.x-1) || (pN.x >= 0 && fp.edgeNX != BOUND_NOSLIP)) { pressure += src.p[src.idx(pN.x, p0.y, p0.z)]; count += 1.0; }
      if((pN.y > 0 && pN.y < src.size.y-1) || (pN.y >= 0 && fp.edgeNY != BOUND_NOSLIP)) { pressure += src.p[src.idx(p0.x, pN.y, p0.z)]; count += 1.0; }
      if((pN.z > 0 && pN.z < src.size.z-1) || (pN.z >= 0 && fp.edgeNZ != BOUND_NOSLIP)) { pressure += src.p[src.idx(p0.x, p0.y, pN.z)]; count += 1.0; }
      
      const T div = src.div[i];
      pressure   = (count > 0 ? (pressure/count - div/6.0) : pressure);
      dst.p[i]   = (isvalid(pressure) ? pressure : 0);
      dst.div[i] = (isvalid(div)      ? div      : 0);
    }
}

template<typename T, typename VT3 = typename cuda_vec<T, 3>::VT, typename IT  = typename cuda_vec<VT3>::IT>
__global__ void pressurePost_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> fp)
{
  const int ix = blockIdx.x*blockDim.x + threadIdx.x;
  const int iy = blockIdx.y*blockDim.y + threadIdx.y;
  const int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      const IT  ip = makeV<IT>(ix, iy, iz);
      const int i  = src.idx(ip);
      const IT p0 = applyBounds(ip,   src.size, fp); // current cell
      const IT pP = applyBounds(ip+1, src.size, fp); // cell + (1,1,1)
      const IT pN = applyBounds(ip-1, src.size, fp); // cell - (1,1,1)
      
      // subtract gradient of pressure from velocity
      
      // // VT3 h = 1.0/makeV<VT3>(src.size);
      // dst.v[i] += (VT3{(src.p[src.idx(pN.x, p0.y, p0.z)] - src.p[src.idx(pP.x, p0.y, p0.z)]),
      //                  (src.p[src.idx(p0.x, pN.y, p0.z)] - src.p[src.idx(p0.x, pP.y, p0.z)]),
      //                  (src.p[src.idx(p0.x, p0.y, pN.z)] - src.p[src.idx(p0.x, p0.y, pP.z)])}) / (2.0 * fp.density);
      // // dst.v[i] -= gradient(dst.p, p0, pP, pN, fp.u.dL);
      
      VT3 dv = makeV<VT3>(0);
      if(pP.x >= 0 && pP.x < src.size.x) { dv.x += src.p[src.idx(pP.x, p0.y, p0.z)]; }
      if(pP.y >= 0 && pP.y < src.size.y) { dv.y += src.p[src.idx(p0.x, pP.y, p0.z)]; }
      if(pP.z >= 0 && pP.z < src.size.z) { dv.z += src.p[src.idx(p0.x, p0.y, pP.z)]; }
      if(pN.x >= 0 && pN.x < src.size.x) { dv.x -= src.p[src.idx(pN.x, p0.y, p0.z)]; }
      if(pN.y >= 0 && pN.y < src.size.y) { dv.y -= src.p[src.idx(p0.x, pN.y, p0.z)]; }
      if(pN.z >= 0 && pN.z < src.size.z) { dv.z -= src.p[src.idx(p0.x, p0.y, pN.z)]; }

      const VT3 dL = makeV<VT3>(fp.u.dL);
      
      dst.v[i]   = src.v[i] - dv/(fp.density*2.0*dL);
      dst.p[i]   = src.p[i];
      dst.div[i] = src.div[i];
    }
}

template<typename T, typename VT3=typename cuda_vec<T, 3>::VT, typename IT=typename cuda_vec<VT3>::IT>
__global__ void pressureBounds_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> fp)
{
  const int ix = blockIdx.x*blockDim.x + threadIdx.x;
  const int iy = blockIdx.y*blockDim.y + threadIdx.y;
  const int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      const IT ip = makeV<IT>(ix, iy, iz);
      const int i = src.idx(ip);
      const IT p0 = applyBounds(ip, src.size, fp); // current cell
      const IT pP = applyBounds(ip+1, src.size, fp); // cell + (1,1,1)
      const IT pN = applyBounds(ip-1, src.size, fp); // cell - (1,1,1)
      
      // boundaries
      T pressure = 0; //src.p[i];
      if(((p0.x == 0 || p0.x  < src.size.x)   && (fp.edgeNX == BOUND_VOID)) ||
         ((p0.y == 0 || p0.y  < src.size.y)   && (fp.edgeNY == BOUND_VOID)) ||
         ((p0.z == 0 || p0.z  < src.size.z)   && (fp.edgeNZ == BOUND_VOID)) ||
         ((p0.x  < 0 || p0.x == src.size.x-1) && (fp.edgePX == BOUND_VOID)) ||
         ((p0.y  < 0 || p0.y == src.size.y-1) && (fp.edgePY == BOUND_VOID)) ||
         ((p0.z  < 0 || p0.z == src.size.z-1) && (fp.edgePZ == BOUND_VOID)))
        { pressure = 0.0; }
      else
        {
          // int ppx = 0; int ppy = 0; int ppz = 0;
          IT sp = makeV<IT>(0);
          if(p0.x == 0 && p0.x  < src.size.x-1 && fp.edgeNX == BOUND_SLIP) { sp.x++; }
          if(p0.y == 0 && p0.y  < src.size.y-1 && fp.edgeNY == BOUND_SLIP) { sp.y++; }
          if(p0.z == 0 && p0.z  < src.size.z-1 && fp.edgeNZ == BOUND_SLIP) { sp.z++; }
          if(p0.x  > 0 && p0.x == src.size.x-2 && fp.edgePX == BOUND_SLIP) { sp.x--; }
          if(p0.y  > 0 && p0.y == src.size.y-2 && fp.edgePY == BOUND_SLIP) { sp.y--; }
          if(p0.z  > 0 && p0.z == src.size.z-2 && fp.edgePZ == BOUND_SLIP) { sp.z--; }

          // if((pP.x > 0 && pP.x < src.size.x-2) || (pP.x >= 0 && fp.edgePX != BOUND_NOSLIP)) { sp.x = pP.x; } // check min bounds (sample +inside)
          // if((pP.y > 0 && pP.y < src.size.y-2) || (pP.y >= 0 && fp.edgePY != BOUND_NOSLIP)) { sp.y = pP.y; }
          // if((pP.z > 0 && pP.z < src.size.z-2) || (pP.z >= 0 && fp.edgePZ != BOUND_NOSLIP)) { sp.z = pP.z; }
          // if((pN.x > 0)                        || (pN.x >= 0 && fp.edgeNX != BOUND_NOSLIP)) { sp.x = pN.x; } // check max bounds (sample -inside)
          // if((pN.y > 0)                        || (pN.y >= 0 && fp.edgeNY != BOUND_NOSLIP)) { sp.y = pN.y; }
          // if((pN.z > 0)                        || (pN.z >= 0 && fp.edgeNZ != BOUND_NOSLIP)) { sp.z = pN.z; }
          
          // if(p0.x == 0            && pP.x >= 0 && fp.edgeNX == BOUND_SLIP)
          // if(p0.y == 0            && pP.y >= 0 && fp.edgeNY == BOUND_SLIP)
          // if(p0.z == 0            && pP.z >= 0 && fp.edgeNZ == BOUND_SLIP)
          // if(p0.x == src.size.x-1 && pN.x >= 0 && fp.edgePX == BOUND_SLIP)
          // if(p0.y == src.size.y-1 && pN.y >= 0 && fp.edgePY == BOUND_SLIP)
          // if(p0.z == src.size.z-1 && pN.z >= 0 && fp.edgePZ == BOUND_SLIP)
  
          pressure = src.p[src.idx(ip+sp)]; // ip+IT{ppx, ppy, ppz})];
        }      
      dst.p[i]   = pressure;
      dst.v[i]   = src.v[i];
      dst.div[i] = src.div[i];
    }
}


// wrappers

template<typename T> void fluidAdvect(FluidField<T> &src, FluidField<T> &dst, const FluidParams<T> &fp)
{
  if(src.size > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      
      if(!isImplicit(fp.vIntegration))
        { // clear advected dst fields if using explitic integration method (written to atomically)
          dst.v.clear();
          dst.Qn.clear();
          dst.Qp.clear();
          dst.Qnv.clear();
          dst.Qpv.clear();
          // dst.E.clear();
          // dst.B.clear();
          
          getLastCudaError("fluidAdvect() ==> clear modified fields");
        }
      
      // copy unmodified fields
      src.p.copyTo(dst.p);
      src.mat.copyTo(dst.mat);
      src.E.copyTo(dst.E);
      src.B.copyTo(dst.B);
      src.Bp.copyTo(dst.Bp);
      
      advect_k<<<grid, threads>>>(src, dst, fp);
      getLastCudaError("fluidAdvect() ==> kernel");
      
      dst.v.copyTo(src.v); // TODO: just pass v field instead of copying (?)
      getLastCudaError("fluidAdvect() ==> copy dst-->src");
      
      velocityBounds_k<<<grid, threads>>>(src, dst, fp);
      getLastCudaError("fluidAdvect() ==> v bounds");
    }
  getLastCudaError("fluidAdvect()");
}

template<typename T> void fluidExtForces(FluidField<T> &src, const FluidParams<T> &fp)
{
  if(src.size > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      externalForces_k<<<grid, threads>>>(src, fp);
    }
  getLastCudaError("fluidExtForces()");
}

template<typename T> void fluidViscosity(FluidField<T> &src, FluidField<T> &dst, const FluidParams<T> &fp, int iter)
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
          
          // iteration
          for(int i = 0; i < iter; i++)
            {
              viscosityIter_k<<<grid, threads>>>(*temp1, *temp2, fp); std::swap(temp1, temp2);
              getLastCudaError(("fluidViscosity() ==> iter " + std::to_string(i)).c_str());
            }
          
          // apply velocity bounds
          velocityBounds_k<<<grid, threads>>>(*temp1, *temp2, fp);   std::swap(temp1, temp2);
          getLastCudaError("fluidViscosity() ==> v bounds");
          
          if(temp1 != &dst) { temp1->v.copyTo(dst.v); } // make sure dst contains final state
          getLastCudaError("fluidViscosity() ==> final copy");
        }
    }
  getLastCudaError("fluidViscosity()");
}

//// PRESSURE ////
template<typename T> void fluidPressure(FluidField<T> &src, FluidField<T> &dst, const FluidParams<T> &fp, int iter)
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
          getLastCudaError("fluidPressure() ==> pre");
          // iteration
          for(int i = 0; i < iter; i++)
            {
              pressureIter_k<<<grid, threads>>>(*temp1, *temp2, fp); std::swap(temp1, temp2);
              getLastCudaError(("fluidPressure() ==> iter " + std::to_string(i)).c_str());
            }
          // post
          pressurePost_k<<<grid, threads>>>(*temp1, *temp2, fp);     std::swap(temp1, temp2);
          getLastCudaError("fluidPressure() ==> post");

          // apply pressure bounds
          pressureBounds_k<<<grid, threads>>>(*temp1, *temp2, fp);   std::swap(temp1, temp2);
          getLastCudaError("fluidPressure() ==> p bounds");
          
          // apply velocity bounds
          velocityBounds_k<<<grid, threads>>>(*temp1, *temp2, fp);   std::swap(temp1, temp2);
          getLastCudaError("fluidPressure() ==> v bounds");
          
          if(temp1 != &dst) // make sure final result is in dst
            { temp1->v.copyTo(dst.v); temp1->p.copyTo(dst.p); temp1->div.copyTo(dst.div); }
          getLastCudaError("fluidPressure() ==> final copy");
        }
    }
  getLastCudaError("fluidPressure()");
}


// template instantiation
template void fluidAdvect   <float>(FluidField<float> &src, FluidField<float> &dst, const FluidParams<float> &fp);
template void fluidViscosity<float>(FluidField<float> &src, FluidField<float> &dst, const FluidParams<float> &fp, int iter);
template void fluidPressure <float>(FluidField<float> &src, FluidField<float> &dst, const FluidParams<float> &fp, int iter);
template void fluidExtForces<float>(FluidField<float> &src,                         const FluidParams<float> &fp);
