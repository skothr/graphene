#ifndef CUDA_TOOLS_CUH
#define CUDA_TOOLS_CUH

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <string>

#include "vector-operators.h"
#include "fluid.cuh"

// TEMP(?) --> TODO: use Field::idx()
#define IDX_T unsigned long
__device__ inline unsigned long idx(const int3 &p, const int3 &sz) { return (IDX_T)p.x + (IDX_T)sz.x*((IDX_T)p.y + ((IDX_T)sz.y*(IDX_T)p.z)); }
#undef IDX_T


////////////////////////////////////////////////////////////////////////////////////////////////
//// kernel tools
////////////////////////////////////////////////////////////////////////////////////////////////

#define MAX_VALUE 1e18
template<typename T> __device__ bool isvalid(const T &v)
{ return (!isnan(v) && !isinf(v) && v <= MAX_VALUE); }
template<>           __device__ inline bool isvalid<float3>(const float3 &v)
{ return (!isnan(v) && !isinf(v) && v.x <= MAX_VALUE && v.y <= MAX_VALUE && v.z <= MAX_VALUE); }

template<typename T>
__device__ bool slipPlane(const int3 &p, const FluidParams<T> &params)
{
  return (((params.edgePX == EDGE_FREESLIP || params.edgePX == EDGE_NOSLIP) && p.x == params.fs.x-1) ||
          ((params.edgeNX == EDGE_FREESLIP || params.edgeNX == EDGE_NOSLIP) && p.x == 0) ||
          ((params.edgePY == EDGE_FREESLIP || params.edgePY == EDGE_NOSLIP) && p.y == params.fs.y-1) ||
          ((params.edgeNY == EDGE_FREESLIP || params.edgeNY == EDGE_NOSLIP) && p.y == 0) ||
          ((params.edgePZ == EDGE_FREESLIP || params.edgePZ == EDGE_NOSLIP) && p.z == params.fs.z-1) ||
          ((params.edgeNZ == EDGE_FREESLIP || params.edgeNZ == EDGE_NOSLIP) && p.z == 0));
}

// basic linear interpolation
template<typename T>
__device__ T lerp(T x0, T x1, float alpha) { return x1*alpha + x0*(1-alpha); }

template<typename T=float>
__device__ typename DimType<int, 3>::VEC_T applyBounds(typename DimType<int, 3>::VEC_T p, typename DimType<int, 3>::VEC_T s,
                                                       const FluidParams<typename Dim<T>::BASE_T> &params)
{
  if(p.x < 0)
    {
      if     (params.edgeNX == EDGE_WRAP)  { p.x = s.x - ((-p.x) % s.x); }
      else if(params.edgeNX == EDGE_VOID)  { p.x = -1;  }
      else if(params.edgeNX == EDGE_NOSLIP || params.edgeNX == EDGE_FREESLIP) { p.x = 0; }
    }
  if(p.y < 0)
    {
      if     (params.edgeNY == EDGE_WRAP)  { p.y = s.y - ((-p.y) % s.y); }
      else if(params.edgeNY == EDGE_VOID)  { p.y = -1; }
      else if(params.edgeNY == EDGE_NOSLIP || params.edgeNY == EDGE_FREESLIP) { p.y = 0; }
    }
  if(p.z < 0)
    {
      if     (params.edgeNZ == EDGE_WRAP)  { p.z = s.z - ((-p.z) % s.z); }
      else if(params.edgeNZ == EDGE_VOID)  { p.z = -1; }
      else if(params.edgeNZ == EDGE_NOSLIP || params.edgeNZ == EDGE_FREESLIP) { p.z = 0; }
    }

  if(p.x > s.x-1)
    {
      if     (params.edgePX == EDGE_WRAP)  { p.x = (p.x % s.x); }
      else if(params.edgePX == EDGE_VOID)  { p.x = -1; }
      else if(params.edgePX == EDGE_NOSLIP || params.edgePX == EDGE_FREESLIP) { p.x = s.x-1; }
    }
  if(p.y > s.y-1)
    {
      if     (params.edgePY == EDGE_WRAP)  { p.y = (p.y % s.y); }
      else if(params.edgePY == EDGE_VOID)  { p.y = -1; }
      else if(params.edgePY == EDGE_NOSLIP || params.edgePY == EDGE_FREESLIP) { p.y = s.y-1; }
    }
  if(p.z > s.z-1)
    {
      if     (params.edgePZ == EDGE_WRAP)  { p.z = (p.z % s.z); }
      else if(params.edgePZ == EDGE_VOID)  { p.z = -1; }
      else if(params.edgePZ == EDGE_NOSLIP || params.edgePZ == EDGE_FREESLIP) { p.z = s.z-1; }
    }
  return p;
}

// get/put data at single integer index
template<typename T>
__device__ T texGet(T *tex, typename DimType<int, 3>::VEC_T p, typename DimType<int, 3>::VEC_T s, const FluidParams<typename Dim<T>::BASE_T> &params)
{
  typedef typename DimType<int, 3>::VEC_T IT;
  IT p0 = applyBounds(p, s, params);
  if(p0 >= 0) { return tex[idx(p0, s)]; }
  else        { return T(); }
}
template<typename T>
__device__ void texPut(T *tex, T val, typename DimType<int, 3>::VEC_T p, const FluidParams<typename Dim<T>::BASE_T> &params)
{
  typedef typename DimType<int, 3>::VEC_T IT;  
  IT s  = params.fs;
  IT p0 = applyBounds(p, s, params);
  if(p0 >= 0) { tex[idx(p0, s)] = val; }
}



template<typename T, typename BASE=typename Dim<T>::BASE_T, typename VT3=typename DimType<BASE, 3>::VEC_T, typename IT3=typename DimType<int, 3>::VEC_T>
__device__ T tex2DD(T *tex, VT3 p, const FluidParams<BASE> &params)
{
  IT3 p0 = IT3{int(floor(p.x)), int(floor(p.y)), int(floor(p.y))}; // integer position
  VT3 fp = abs(VT3{p.x-p0.x, p.y-p0.y, p.z-p0.z});                 // fractional position
  IT3 s  = params.fs;
  
  IT3 p000 = applyBounds(p0,   s, params);
  IT3 p111 = applyBounds(p0+1, s, params);
  if(p000 >= IT3{0,0,0} && p000 < s && p111 >= IT3{0,0,0} && p111 < s)
    {
      return lerp<T>(lerp<T>(lerp<T>(texGet(tex, IT3{p000.x, p000.y, p000.z}, s, params),
                                     texGet(tex, IT3{p111.x, p000.y, p000.z}, s, params), fp.x),
                             lerp<T>(texGet(tex, IT3{p000.x, p111.y, p000.z}, s, params),
                                     texGet(tex, IT3{p111.x, p111.y, p000.z}, s, params), fp.x), fp.y),
                     lerp<T>(lerp<T>(texGet(tex, IT3{p000.x, p000.y, p111.z}, s, params),
                                     texGet(tex, IT3{p111.x, p000.y, p111.z}, s, params), fp.x),
                             lerp<T>(texGet(tex, IT3{p000.x, p111.y, p111.z}, s, params),
                                     texGet(tex, IT3{p111.x, p111.y, p111.z}, s, params), fp.x), fp.y),
                     fp.z);
    }
  else { return T(); }
}





// atomically get/put data at single integer index
template<typename T, int N=3, typename IT3=typename DimType<int, 3>::VEC_T>
__device__ inline void texAtomicAdd(float *tex, float val, const IT3 &p, const FluidParams<T> &params)
{
  IT3 s  = params.fs;
  IT3 p0 = applyBounds(p, s, params);
  if(p0 >= 0 && p0 < s) { atomicAdd(&tex[idx(p0, s)], val); }
}
template<typename T, int N=3, typename IT3=typename DimType<int, 3>::VEC_T>
__device__ inline void texAtomicAdd(float2 *tex, float2 val, const IT3 &p, const FluidParams<T> &params)
{
  IT3 s  = params.fs;
  IT3 p0 = applyBounds(p, s, params);
  if(p0 >= 0 && p0 < s)
    { atomicAdd(&tex[idx(p0, s)].x, val.x); atomicAdd(&tex[idx(p0, s)].y, val.y); }
}
template<typename T, int N=3, typename IT3=typename DimType<int, 3>::VEC_T>
__device__ inline void texAtomicAdd(float3 *tex, float3 val, const IT3 &p, const FluidParams<T> &params)
{
  IT3 s  = params.fs;
  IT3 p0 = applyBounds(p, s, params);
  if(p0 >= 0 && p0 < s)
    { atomicAdd(&tex[idx(p0, s)].x, val.x); atomicAdd(&tex[idx(p0, s)].y, val.y); atomicAdd(&tex[idx(p0, s)].z, val.z); }
}
template<typename T, int N=3, typename IT3=typename DimType<int, 3>::VEC_T, typename ITN=typename DimType<int, N>::VEC_T>
__device__ inline void texAtomicAdd(Material<T> *tex, const Material<T> &val, const ITN &p, const FluidParams<T> &params)
{
  IT3 s  = params.fs;
  IT3 p0 = applyBounds(p, s, params);
  if(p0 >= 0 && p0 < s)
    { atomicAdd(&tex[idx(p0, s)].ep,  val.ep); atomicAdd(&tex[idx(p0, s)].mu,  val.mu); atomicAdd(&tex[idx(p0, s)].sig, val.sig); }
}








template<typename T, typename BASE=typename Dim<T>::BASE_T, typename IT3=typename Dim<T>::SIZE_T>
__device__ int4 texPutIX(T p, const FluidParams<BASE> &params)
{
  IT3 s  = params.fs;
  IT3 p0  = IT3{int(floor(p.x)), int(floor(p.y)), int(floor(p.z))};   // integer position
  IT3 p000 = applyBounds(p0,   s, params);
  IT3 p111 = applyBounds(p0+1, s, params);
  return int4{ p000.x, p111.x, p000.x, p111.x }; // ORDER --> { X00, X10, X01, X11 }
}
template<typename T, typename BASE=typename Dim<T>::BASE_T, typename IT3=typename Dim<T>::SIZE_T>
__device__ int4 texPutIY(T p, const FluidParams<BASE> &params)
{
  IT3 s  = params.fs;
  IT3 p0  = IT3{int(floor(p.x)), int(floor(p.y)), int(floor(p.z))};   // integer position
  IT3 p000 = applyBounds(p0,   s, params);
  IT3 p111 = applyBounds(p0+1, s, params);
  return int4{ p000.y, p000.y, p111.y, p111.y }; // ORDER --> { Y00, Y10,  Y01, Y11 }
}
template<typename T, typename BASE=typename Dim<T>::BASE_T, typename IT3=typename Dim<T>::SIZE_T>
__device__ int4 texPutIZ(T p, const FluidParams<BASE> &params)
{
  IT3 s  = params.fs;
  IT3 p0  = IT3{int(floor(p.x)), int(floor(p.y)), int(floor(p.z))};   // integer position
  IT3 p000 = applyBounds(p0,   s, params);
  IT3 p111 = applyBounds(p0+1, s, params);
  return int4{ p000.z, p000.z, p111.z, p111.z }; // ORDER --> { Z00, Z10,  Z01, Z11 }
}
template<typename T=float, typename VT3=typename DimType<T, 3>::VEC_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ float4 texPutMults0(const VT3 &p)
{
  IT3 p0 = IT3{int(floor(p.x)), int(floor(p.y)), int(floor(p.z))}; // integer position
  VT3 fp = abs(p - makeV<VT3>(p0));                                // fractional position
  T m00 = (-fp.x + 1.0f) * (-fp.y + 1.0f);
  T m10 = (fp.x) * (-fp.y + 1.0f);
  T m01 = (-fp.x + 1.0f) * (fp.y);
  T m11 = (fp.x) * (fp.y);
  return float4{ m00*(1.0f-fp.z), m10*(1.0f-fp.z), m01*(1.0f-fp.z), m11*(1.0f-fp.z) };
}

template<typename T=float, typename VT3=typename DimType<T, 3>::VEC_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ float4 texPutMults1(const VT3 &p)
{
  IT3 p0 = IT3{int(floor(p.x)), int(floor(p.y)), int(floor(p.z))};         // integer position
  VT3 fp = abs(p - makeV<VT3>(p0));// fractional position
  T m00 = (-fp.x + 1.0f) * (-fp.y + 1.0f);
  T m10 = (fp.x) * (-fp.y + 1.0f);
  T m01 = (-fp.x + 1.0f) * (fp.y);
  T m11 = (fp.x) * (fp.y);
  return float4{ m00*fp.z, m10*fp.z, m01*fp.z, m11*fp.z };
}

// NOTE: atomic operation
template<typename T, typename BASE=typename Dim<T>::BASE_T, typename VT3=typename DimType<BASE, 3>::VEC_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ void putTex2DD(T *tex, T val, VT3 p, const FluidParams<BASE> &params)
{
  IT3 p0 = IT3{int(floor(p.x)), int(floor(p.y)), int(floor(p.z))}; // integer position
  VT3 fp = abs(VT3{p.x-p0.x, p.y-p0.y, p.z-p0.z});                 // fractional position
  IT3 s  = params.fs;
  p0 = applyBounds(p0+1, s, params);
  
  IT3 p000 = applyBounds(p0,   s, params);
  IT3 p111 = applyBounds(p0+1, s, params);
  if(p000 >= IT3{0,0,0} && p000 < s && p111 >= IT3{0,0,0} && p111 < s)
    {
      float4 mults0 = texPutMults0(p);
      float4 mults1 = texPutMults1(p);
      int4 tiX = texPutIX(p, params);
      int4 tiY = texPutIY(p, params);
      int4 tiZ = texPutIZ(p, params);
      p000     = IT3{ tiX.x, tiY.x, tiZ.x }; IT3 p110 = IT3{ tiX.w, tiY.w, tiZ.x };
      IT3 p010 = IT3{ tiX.z, tiY.z, tiZ.x }; IT3 p100 = IT3{ tiX.y, tiY.y, tiZ.x };
      IT3 p001 = IT3{ tiX.x, tiY.x, tiZ.z };     p111 = IT3{ tiX.w, tiY.w, tiZ.z };
      IT3 p011 = IT3{ tiX.z, tiY.z, tiZ.z }; IT3 p101 = IT3{ tiX.y, tiY.y, tiZ.z };
  
      // scale value by grid overlap and store in each location
      texAtomicAdd(tex, val*mults0.x, p000, params); texAtomicAdd(tex, val*mults0.y, p100, params);
      texAtomicAdd(tex, val*mults0.z, p010, params); texAtomicAdd(tex, val*mults0.w, p110, params);
      texAtomicAdd(tex, val*mults1.x, p001, params); texAtomicAdd(tex, val*mults1.y, p101, params);
      texAtomicAdd(tex, val*mults1.z, p011, params); texAtomicAdd(tex, val*mults1.w, p111, params);
    }
}






template<typename T, typename VT>
__device__ VT integrateBackwardEuler(VT *data, const VT &p, const VT &v, T dt)
{ // trace velocity backward (IMPLICIT)
  return p - v*dt;
}
template<typename T, typename VT>
__device__ VT integrateForwardEuler(VT *data, const VT &p, const VT &v, T dt)
{ // trace velocity forward (EXPLICIT)
  return p + v*dt;
}

//// RK4 ////
template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT=DimType<int, 3>::VEC_T>
__device__ VT3 integrateRK4(const Field<VT3> &data, const VT3 &p, const FluidParams<T> &cp)
{
  IT  ip0 = makeV<IT>(p);
  VT3 k1 = cp.u.dt*data[data.idx(ip0)];
  
  VT3 p1  = p + k1/2.0;
  IT  ip1 = applyBounds(makeV<IT>(p1), data.size, cp);
  VT3 k2  = cp.u.dt*data[data.idx(ip1)];

  VT3 p2  = p + k2/2.0;
  IT  ip2 = applyBounds(makeV<IT>(p2), data.size, cp);
  VT3 k3  = cp.u.dt*data[data.idx(ip2)];
  
  VT3 p3  = p + k3;
  IT  ip3 = applyBounds(makeV<IT>(p3), data.size, cp);
  VT3 k4  = cp.u.dt*data[data.idx(ip3)];
  
  return p + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0;
}


// // maximization reduction
// // template<typename T, unsigned int blockSize> __global__ void fieldMax_k(CudaField<T> fieldIn, CudaField<T> fieldOut, unsigned int n);
// class CudaFieldBase;
// template<typename T> class CudaField;
// template<typename T> float fieldMax(CudaFieldBase *field, CudaFieldBase *fieldOut, CudaField<float> *dst);
// template<typename T> float fieldNorm(CudaFieldBase *field, CudaFieldBase *fieldOut, CudaField<float> *dst);




#endif // CUDA_TOOLS_CUH
