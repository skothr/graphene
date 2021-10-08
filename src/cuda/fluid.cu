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
      
      // zero perpendicular velocity at edges (assumes wall is static)
      v0 = VT3{ ((slipPlane(ip-IT{1,0,0}, params) || slipPlane(ip+IT{1,0,0}, params)) ? 0 : v0.x),
                ((slipPlane(ip-IT{0,1,0}, params) || slipPlane(ip+IT{0,1,0}, params)) ? 0 : v0.y),
                ((slipPlane(ip-IT{0,0,1}, params) || slipPlane(ip+IT{0,0,1}, params)) ? 0 : v0.z) };

      // TODO: apply viscosity

      // // limit magnitude
      const T vMax = 100.0f;
      v0.x = (v0.x < 0 ? -1.0f : 1.0f)*min(abs(v0.x), vMax);
      v0.y = (v0.y < 0 ? -1.0f : 1.0f)*min(abs(v0.y), vMax);
      v0.z = (v0.z < 0 ? -1.0f : 1.0f)*min(abs(v0.z), vMax);

      // check for invalid values
      if(isnan(v0.x) || isinf(v0.x)) { v0.x = 0.0; }
      if(isnan(v0.y) || isinf(v0.y)) { v0.y = 0.0; }
      if(isnan(v0.z) || isinf(v0.z)) { v0.z = 0.0; }
      
      // use forward Euler method to advect
      VT3    p2     = integrateForwardEuler(src.v.dData, p, v0, dt);
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
      T p = 0; T d = 0;
      if(p00 >= 0 && pp1 >= 0 && pn1 >= 0)
        {
          d = -(h.x*(src.v[src.idx(pp1.x, p00.y, p00.z)].x - src.v[src.idx(pn1.x, p00.y, p00.z)].x) +
                h.y*(src.v[src.idx(p00.x, pp1.y, p00.z)].y - src.v[src.idx(p00.x, pn1.y, p00.z)].y) +
                h.z*(src.v[src.idx(p00.x, p00.y, pp1.z)].z - src.v[src.idx(p00.x, p00.y, pn1.z)].z)) / 3.0f;
          p = src.p[i];
        }
      dst.p[i]   = (isnan(p) || isinf(p)) ? 0 : p;
      dst.div[i] = (isnan(d) || isinf(d)) ? 0 : d;
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
      
      if(p00 >= 0 && pp1 >= 0 && pn1 >= 0)
        { // apply pressure to velocity
          dst.v[i].x += (src.p[src.idx(pn1.x, p00.y, p00.z)] - src.p[src.idx(pp1.x, p00.y, p00.z)]) / 3.0f / h.x;
          dst.v[i].y += (src.p[src.idx(p00.x, pn1.y, p00.z)] - src.p[src.idx(p00.x, pp1.y, p00.z)]) / 3.0f / h.y;
          dst.v[i].z += (src.p[src.idx(p00.x, p00.y, pn1.z)] - src.p[src.idx(p00.x, p00.y, pp1.z)]) / 3.0f / h.z;
        } else { dst.v[i] = src.v[i]; }
      T p = src.p[i];   dst.p[i]   = (isnan(p) || isinf(p)) ? 0 : p;
      T d = src.div[i]; dst.div[i] = (isnan(d) || isinf(d)) ? 0 : d;
    }
}


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
      if(iter > 0)
        {
          FluidField<T> *temp1 = &src;
          FluidField<T> *temp2 = &dst;
          // PRESSURE INIT (copies 
          fluidPressurePre(*temp1, *temp2, params); std::swap(temp1, temp2);        // temp1 --> temp2 --> temp1

          // PRESSURE ITERATION
          for(int i = 0; i < iter; i++)
            { fluidPressureIter(*temp1, *temp2, params); std::swap(temp1, temp2); } // temp1 --> temp2 --> temp1
          if(temp1 == &dst) { std::swap(temp1, temp2); } //temp2 = temp1; } // in-place post
          
          // PRESSURE FINALIZE
          fluidPressurePost(*temp1, *temp2, params);                                // temp1 --> temp2 (either dst-->dst, or src-->dst)
          //if(temp1 != &dst) { temp1->v.copyTo(dst.v); temp1->p.copyTo(dst.p); temp1->div.copyTo(dst.div); }
          //if(temp1 != &dst) { temp1->copyTo(dst); }
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
      
      // PRESSURE SOLVE (1)
      if(params.updateP1)       { fluidPressure(*temp1, *temp2, params, params.pIter1); std::swap(temp1, temp2); }
      // ADVECT
      if(params.updateAdvect)   { fluidAdvect  (*temp1, *temp2, params);                std::swap(temp1, temp2); }
      // PRESSURE SOLVE (2)
      if(params.updateP2)       { fluidPressure(*temp1, *temp2, params, params.pIter2); std::swap(temp1, temp2); }

      if(temp1 != &dst) { temp1->copyTo(dst); }
    }
}

// template instantiations
template void fluidStep  <float>(FluidField<float> &src, FluidField<float> &dst, FluidParams<float> &params);

