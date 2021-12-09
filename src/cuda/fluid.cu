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
      VT3 p2; // (sample point for integration)
      
      VT3 v = src.v.dData[i];
      // T p0;
      T Qn; T Qp; VT3 Qnv; VT3 Qpv; //VT3 E; VT3 B;
      if(isImplicit(fp.vIntegration))
        { //// integrate for implicit methods (sample value from previous time point)
          if(fp.vIntegration == INTEGRATION_BACKWARD_EULER) { p2 = integrateBackwardEuler(src.v.dData, p, v, fp.u.dt); }
          v   = tex2DD(src.v.dData,   p2, fp);
          Qn  = tex2DD(src.Qn.dData,  p2, fp);
          Qp  = tex2DD(src.Qp.dData,  p2, fp);
          Qnv = tex2DD(src.Qnv.dData, p2, fp);
          Qpv = tex2DD(src.Qpv.dData, p2, fp);
          // E   = tex2DD(src.E.dData,   p2, fp);
          // B   = tex2DD(src.B.dData,   p2, fp);
        }
      else
        {
          Qn  = src.Qn[i];
          Qp  = src.Qp[i];
          Qnv = src.Qnv[i];
          Qpv = src.Qpv[i];
          // E = src.E[i];
          // B = src.B[i];
        }
      
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
          // dst.E[i]   = E;
          // dst.B[i]   = B;
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
          // putTex2DD(dst.E.dData,   E,   p2, fp);
          // putTex2DD(dst.B.dData,   B,   p2, fp);
        }
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
      IT p0 = ip; //applyBounds(ip, src.size, fp);

      // apply boundaries
      IT vp = IT{0,0,0};
      if(p0.x == 0 && p0.x  < src.size.x-1 && fp.edgeNX == EDGE_FREESLIP) { vp.x++; }
      if(p0.x >= 0 && p0.x == src.size.x-1 && fp.edgePX == EDGE_FREESLIP) { vp.x--; }
      if(p0.y == 0 && p0.y  < src.size.y-1 && fp.edgeNY == EDGE_FREESLIP) { vp.y++; }
      if(p0.y >= 0 && p0.y == src.size.y-1 && fp.edgePY == EDGE_FREESLIP) { vp.y--; }
      if(p0.z == 0 && p0.z  < src.size.z-1 && fp.edgeNZ == EDGE_FREESLIP) { vp.z++; }
      if(p0.z >= 0 && p0.z == src.size.z-1 && fp.edgePZ == EDGE_FREESLIP) { vp.z--; }
      dst.v[i] = (VT3{(vp.x==0 ? src.v[i].x : -src.v[src.idx(ip+IT{vp.x,0,0})].x),
                      (vp.y==0 ? src.v[i].y : -src.v[src.idx(ip+IT{0,vp.y,0})].y),
                      (vp.z==0 ? src.v[i].z : -src.v[src.idx(ip+IT{0,0,vp.z})].z)});
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
      IT ip = makeI<IT>(ix, iy, iz); int i = src.idx(ip);
      IT p0 = ip; // applyBounds(ip,   src.size, fp); // current cell
      IT pN = applyBounds(ip-1, src.size, fp); // cell - (1,1)
      IT pP = applyBounds(ip+1, src.size, fp); // cell + (1,1)
      
      VT3 v = src.v[i];
      if(pP >= 0 && pN >= 0)
        { // jacobi parameters --> see: https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
          T alpha = fp.u.dL*fp.u.dL*fp.u.dL/(fp.u.dt*fp.viscosity); // α = dx³/(*dt)
          T beta  = 1.0 / (6.0 + alpha);                            // β = dx³/(ν*dt)
          v = jacobi(src.v, p0, pP, pN, alpha, beta);
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
      IT p0 = ip; //applyBounds(ip,   src.size, fp); // current cell
      IT pN = applyBounds(ip-1, src.size, fp); // cell - (1,1)
      IT pP = applyBounds(ip+1, src.size, fp); // cell + (1,1)
      // calculate divergence
      T p = src.p[i];
      T d = 0;
      // d = divergence(src.v, p00, pp1, pn1, fp.u.dL) / fp.density;
      d = ((((pP.x >= 0 && pP.x < src.size.x) ? src.v[src.idx(pP.x, p0.y, p0.z)].x : 0) -
            ((pN.x >= 0 && pN.x < src.size.x) ? src.v[src.idx(pN.x, p0.y, p0.z)].x : 0)) +
           (((pP.y >= 0 && pP.y < src.size.y) ? src.v[src.idx(p0.x, pP.y, p0.z)].y : 0) -
            ((pN.y >= 0 && pN.y < src.size.y) ? src.v[src.idx(p0.x, pN.y, p0.z)].y : 0)) +
           (((pP.z >= 0 && pP.z < src.size.z) ? src.v[src.idx(p0.x, p0.y, pP.z)].z : 0) -
            ((pN.z >= 0 && pN.z < src.size.z) ? src.v[src.idx(p0.x, p0.y, pN.z)].z : 0))) / (2.0f * fp.u.dL);

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
      IT p0 = ip; // current cell
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
      
      T p = 0;
      T d = src.div[i];
      int count = 0;
      if((pP.x > 0 && pP.x < src.size.x-1) || (pP.x >= 0 && fp.edgePX != EDGE_NOSLIP)) { p += src.p[src.idx(pP.x, p0.y, p0.z)]; count++; }
      if((pP.y > 0 && pP.y < src.size.y-1) || (pP.y >= 0 && fp.edgePY != EDGE_NOSLIP)) { p += src.p[src.idx(p0.x, pP.y, p0.z)]; count++; }
      if((pP.z > 0 && pP.z < src.size.z-1) || (pP.z >= 0 && fp.edgePZ != EDGE_NOSLIP)) { p += src.p[src.idx(p0.x, p0.y, pP.z)]; count++; }
      if((pN.x > 0 && pN.x < src.size.x-1) || (pN.x >= 0 && fp.edgeNX != EDGE_NOSLIP)) { p += src.p[src.idx(pN.x, p0.y, p0.z)]; count++; }
      if((pN.y > 0 && pN.y < src.size.y-1) || (pN.y >= 0 && fp.edgeNY != EDGE_NOSLIP)) { p += src.p[src.idx(p0.x, pN.y, p0.z)]; count++; }
      if((pN.z > 0 && pN.z < src.size.z-1) || (pN.z >= 0 && fp.edgeNZ != EDGE_NOSLIP)) { p += src.p[src.idx(p0.x, p0.y, pN.z)]; count++; }
      p = count > 0 ? ((p/(T)count - d/6.0)) : p;
      dst.p[i]   = (isvalid(p) ? p : 0);
      dst.div[i] = (isvalid(d) ? d : 0);
    }
}

template<typename T, typename VT3 = typename DimType<T, 3>::VEC_T, typename IT  = typename Dim<VT3>::SIZE_T>
__global__ void pressurePost_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> fp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT  ip = makeI<IT>(ix, iy, iz);
      int i  = src.idx(ip);
      IT p0 = ip; //applyBounds(ip,   src.size, fp); // current cell
      IT pP = applyBounds(ip+1, src.size, fp); // cell + (1,1,1)
      IT pN = applyBounds(ip-1, src.size, fp); // cell - (1,1,1)
      
      // subtract gradient of pressure from velocity
      VT3 v  = src.v[i];
      VT3 dv = VT3{0,0,0};
      VT3 dL = VT3{2,2,2}*fp.u.dL;
      if(pP.x >= 0 && pP.x < src.size.x) { dv.x += src.p[src.idx(pP.x, p0.y, p0.z)]; }
      if(pP.y >= 0 && pP.y < src.size.y) { dv.y += src.p[src.idx(p0.x, pP.y, p0.z)]; }
      if(pP.z >= 0 && pP.z < src.size.z) { dv.z += src.p[src.idx(p0.x, p0.y, pP.z)]; }
      if(pN.x >= 0 && pN.x < src.size.x) { dv.x -= src.p[src.idx(pN.x, p0.y, p0.z)]; }
      if(pN.y >= 0 && pN.y < src.size.y) { dv.y -= src.p[src.idx(p0.x, pN.y, p0.z)]; }
      if(pN.z >= 0 && pN.z < src.size.z) { dv.z -= src.p[src.idx(p0.x, p0.y, pN.z)]; }
      // // VT3 h = 1.0/makeV<VT3>(src.size);
      // dst.v[i] += (VT3{(src.p[src.idx(pN.x, p0.y, p0.z)] - src.p[src.idx(pP.x, p0.y, p0.z)]),
      //                  (src.p[src.idx(p0.x, pN.y, p0.z)] - src.p[src.idx(p0.x, pP.y, p0.z)]),
      //                  (src.p[src.idx(p0.x, p0.y, pN.z)] - src.p[src.idx(p0.x, p0.y, pP.z)])}) / (2.0 * fp.density);
      // // dst.v[i] -= gradient(dst.p, p0, pP, pN, fp.u.dL);

      dst.v[i]   = v - dv/(dL*fp.density);
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
      IT  p0 = applyBounds(ip, src.size, fp); // current cell

      // boundaries
      T p = 0;
      if(((p0.x == 0 || p0.x  < src.size.x)   && (fp.edgeNX == EDGE_VOID)) ||
         ((p0.y == 0 || p0.y  < src.size.y)   && (fp.edgeNY == EDGE_VOID)) ||
         ((p0.z == 0 || p0.z  < src.size.z)   && (fp.edgeNZ == EDGE_VOID)) ||
         ((p0.x  < 0 || p0.x == src.size.x-1) && (fp.edgePX == EDGE_VOID)) ||
         ((p0.y  < 0 || p0.y == src.size.y-1) && (fp.edgePY == EDGE_VOID)) ||
         ((p0.z  < 0 || p0.z == src.size.z-1) && (fp.edgePZ == EDGE_VOID)))
        { p = 0.0f; }
      else
        {
          int ppx = 0; int ppy = 0; int ppz = 0;
          if(p0.x == 0 && p0.x  < src.size.x-1 && fp.edgeNX == EDGE_FREESLIP) { ppx++; }
          if(p0.y == 0 && p0.y  < src.size.y-1 && fp.edgeNY == EDGE_FREESLIP) { ppy++; }
          if(p0.z == 0 && p0.z  < src.size.z-1 && fp.edgeNZ == EDGE_FREESLIP) { ppz++; }
          if(p0.x  > 0 && p0.x == src.size.x-2 && fp.edgePX == EDGE_FREESLIP) { ppx--; }
          if(p0.y  > 0 && p0.y == src.size.y-2 && fp.edgePY == EDGE_FREESLIP) { ppy--; }
          if(p0.z  > 0 && p0.z == src.size.z-2 && fp.edgePZ == EDGE_FREESLIP) { ppz--; }
          p = src.p[src.idx(ip+IT{ppx, ppy, ppz})];
        }      
      dst.p[i]   = p;
      dst.v[i]   = src.v[i];
      dst.div[i] = src.div[i];
    }
}


// wrappers

template<typename T> void fluidAdvect(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> fp)
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
  getLastCudaError("fluidExternalForces()");
}

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
template void fluidAdvect        <float>(FluidField<float> &src, FluidField<float> &dst, FluidParams<float> fp);
template void fluidExternalForces<float>(FluidField<float> &src, FluidParams<float> fp);
template void fluidViscosity     <float>(FluidField<float> &src, FluidField<float> &dst, FluidParams<float> fp, int iter);
template void fluidPressure      <float>(FluidField<float> &src, FluidField<float> &dst, FluidParams<float> fp, int iter);
