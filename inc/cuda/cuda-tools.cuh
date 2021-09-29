#ifndef CUDA_TOOLS_CUH
#define CUDA_TOOLS_CUH

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <string>

#include "vector-operators.h"
#include "fluid.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
//// kernel tools
////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ bool slipPlane(const int3 &p, const FluidParams<T> &params)
{
  return (((params.edgePX == EDGE_NOSLIP || params.edgePX == EDGE_SLIP) && p.x == params.fs.x-1) ||
          ((params.edgeNX == EDGE_NOSLIP || params.edgeNX == EDGE_SLIP) && p.x == 0) ||
          ((params.edgePY == EDGE_NOSLIP || params.edgePY == EDGE_SLIP) && p.y == params.fs.y-1) ||
          ((params.edgeNY == EDGE_NOSLIP || params.edgeNY == EDGE_SLIP) && p.y == 0) ||
          ((params.edgePZ == EDGE_NOSLIP || params.edgePZ == EDGE_SLIP) && p.z == params.fs.z-1) ||
          ((params.edgeNZ == EDGE_NOSLIP || params.edgeNZ == EDGE_SLIP) && p.z == 0));
}

// basic linear interpolation
template<typename T>
__host__ __device__ T lerp(T x0, T x1, float alpha) { return x1*alpha + x0*(1-alpha); }

template<typename T=float>
__device__ typename DimType<int, 3>::VEC_T applyBounds(typename DimType<int, 3>::VEC_T p, typename DimType<int, 3>::VEC_T s,
                                                       const FluidParams<typename Dim<T>::BASE_T> &params)
{
  if(p.x < 0)
    {
      if     (params.edgeNX == EDGE_WRAP)  { p.x = s.x - ((-p.x) % s.x); }
      else if(params.edgeNX == EDGE_VOID)  { p.x = -1;  }
      else if(params.edgeNX == EDGE_SLIP || params.edgeNX == EDGE_NOSLIP) { p.x = 0; }
    }
  if(p.y < 0)
    {
      if     (params.edgeNY == EDGE_WRAP)  { p.y = s.y - ((-p.y) % s.y); }
      else if(params.edgeNY == EDGE_VOID)  { p.y = -1; }
      else if(params.edgeNY == EDGE_SLIP || params.edgeNY == EDGE_NOSLIP) { p.y = 0; }
    }
  if(p.z < 0)
    {
      if     (params.edgeNZ == EDGE_WRAP)  { p.z = s.z - ((-p.z) % s.z); }
      else if(params.edgeNZ == EDGE_VOID)  { p.z = -1; }
      else if(params.edgeNZ == EDGE_SLIP || params.edgeNZ == EDGE_NOSLIP) { p.z = 0; }
    }

  if(p.x > s.x-1)
    {
      if     (params.edgePX == EDGE_WRAP)  { p.x = (p.x % s.x); }
      else if(params.edgePX == EDGE_VOID)  { p.x = -1; }
      else if(params.edgePX == EDGE_SLIP || params.edgePX == EDGE_NOSLIP) { p.x = s.x-1; }
    }
  if(p.y > s.y-1)
    {
      if     (params.edgePY == EDGE_WRAP)  { p.y = (p.y % s.y); }
      else if(params.edgePY == EDGE_VOID)  { p.y = -1; }
      else if(params.edgePY == EDGE_SLIP || params.edgePY == EDGE_NOSLIP) { p.y = s.y-1; }
    }
  if(p.z > s.z-1)
    {
      if     (params.edgePZ == EDGE_WRAP)  { p.z = (p.z % s.z); }
      else if(params.edgePZ == EDGE_VOID)  { p.z = -1; }
      else if(params.edgePZ == EDGE_SLIP || params.edgePZ == EDGE_NOSLIP) { p.z = s.z-1; }
    }
  return p;
}

// get/put data at single integer index
template<typename T>
__device__ T texGet(T *tex, typename DimType<int, 3>::VEC_T p, typename DimType<int, 3>::VEC_T s, const FluidParams<typename Dim<T>::BASE_T> &params)
{
  typedef typename DimType<int, 3>::VEC_T IT;
  IT p0 = applyBounds(p, s, params);
  if(p0 >= 0 && p0 < s) { return tex[p0.x + s.x*(p0.y + (s.y*p0.z))]; }
  else                  { return T(); }
}
template<typename T>
__device__ void texPut(T *tex, T val, typename DimType<int, 3>::VEC_T p, const FluidParams<typename Dim<T>::BASE_T> &params)
{
  typedef typename DimType<int, 3>::VEC_T IT;  
  IT s  = params.fs;
  IT p0 = applyBounds(p, s, params);
  if(p0.x >= 0 && p0.x < s.x && p0.y >= 0 && p0.y < s.y && p0.z >= 0 && p0.z < s.z) { tex[p0.x + s.x*(p0.y + (s.y*p0.z))] = val; }
}

// atomically get/put data at single integer index
// template<typename T>
// __device__ void texAtomicAdd(T *tex, T val, typename Dim<T>::SIZE_T p, const FluidParams<float2> &params);

// template<>
template<typename T, int N=3>
__device__ inline void texAtomicAdd(float *tex, float val, const typename DimType<int, N>::VEC_T &p, const FluidParams<T> &params)
{
  typedef typename DimType<int, 3>::VEC_T IT;
  IT s  = params.fs;
  IT p0 = applyBounds(p, s, params);
  
  if(p0.x >= 0 && p0.x < s.x && p0.y >= 0 && p0.y < s.y && p0.z >= 0 && p0.z < s.z)
    { atomicAdd(&tex[p0.x + s.x*(p0.y + (s.y*p0.z))], val); }
}
template<typename T, int N=3>
__device__ inline void texAtomicAdd(float2 *tex, float2 val, const typename DimType<int, N>::VEC_T &p, const FluidParams<T> &params)
{
  typedef typename DimType<int, 3>::VEC_T IT;
  IT s  = params.fs;
  IT p0 = applyBounds(p, s, params);
  
  if(p0.x >= 0 && p0.x < s.x && p0.y >= 0 && p0.y < s.y && p0.z >= 0 && p0.z < s.z)
    {
      atomicAdd(&tex[p0.x + s.x*(p0.y + (s.y*p0.z))].x, val.x);
      atomicAdd(&tex[p0.x + s.x*(p0.y + (s.y*p0.z))].y, val.y);
    }
}
template<typename T, int N=3>
__device__ inline void texAtomicAdd(float3 *tex, float3 val, const typename DimType<int, N>::VEC_T &p, const FluidParams<T> &params)
{
  typedef typename DimType<int, 3>::VEC_T IT;
  IT s  = params.fs;
  IT p0 = applyBounds(p, s, params);
  
  if(p0.x >= 0 && p0.x < s.x && p0.y >= 0 && p0.y < s.y && p0.z >= 0 && p0.z < s.z)
    {
      atomicAdd(&tex[p0.x + s.x*(p0.y + (s.y*p0.z))].x, val.x);
      atomicAdd(&tex[p0.x + s.x*(p0.y + (s.y*p0.z))].y, val.y);
      atomicAdd(&tex[p0.x + s.x*(p0.y + (s.y*p0.z))].z, val.z);
    }
}

template<typename T>
__device__ T tex2DD(T *tex, typename DimType<typename Dim<T>::BASE_T, 3>::VEC_T p, const FluidParams<typename Dim<T>::BASE_T> &params)
{
  typedef typename DimType<typename Dim<T>::BASE_T, 3>::VEC_T VT;
  typedef typename DimType<int, 3>::VEC_T                     IT;
  IT p0 = IT{int(floor(p.x)), int(floor(p.y)), int(floor(p.y))}; // integer position
  VT fp = abs(VT{p.x-p0.x, p.y-p0.y, p.z-p0.z});        // fractional position
  IT s  = params.fs;

  IT p000 = applyBounds(p0,   s, params);
  IT p111 = applyBounds(p0+1, s, params);

  if(p000.x < 0 || p000.y < 0 || p000.z < 0 || p111.x < 0 || p111.y < 0 || p111.z < 0) { return T(); }
  
  T result = lerp<T>(lerp<T>(lerp<T>(texGet(tex, IT{p000.x, p000.y, p000.z}, s, params), texGet(tex, IT{p111.x, p000.y, p000.z}, s, params), fp.x),
                             lerp<T>(texGet(tex, IT{p000.x, p111.y, p000.z}, s, params), texGet(tex, IT{p111.x, p111.y, p000.z}, s, params), fp.x), fp.y),
                     lerp<T>(lerp<T>(texGet(tex, IT{p000.x, p000.y, p111.z}, s, params), texGet(tex, IT{p111.x, p000.y, p111.z}, s, params), fp.x),
                             lerp<T>(texGet(tex, IT{p000.x, p111.y, p111.z}, s, params), texGet(tex, IT{p111.x, p111.y, p111.z}, s, params), fp.x), fp.y),
                     fp.z);
  return result;
}

template<typename T>
__device__ int4 texPutIX(T p, const FluidParams<typename Dim<T>::BASE_T> &params)
{
  typedef typename Dim<T>::SIZE_T IT;
  IT s  = params.fs;
  IT p0  = IT{int(floor(p.x)), int(floor(p.y)), int(floor(p.z))};   // integer position
  IT p000 = applyBounds(p0,   s, params);
  IT p111 = applyBounds(p0+1, s, params);
  return int4{ p000.x, p111.x, p000.x, p111.x }; // ORDER --> { X00, X10, X01, X11 }
}
template<typename T>
__device__ int4 texPutIY(T p, const FluidParams<typename Dim<T>::BASE_T> &params)
{
  typedef typename Dim<T>::SIZE_T IT;
  IT s  = params.fs;
  IT p0  = IT  {int(floor(p.x)), int(floor(p.y)), int(floor(p.z))};   // integer position
  IT p000 = applyBounds(p0,   s, params);
  IT p111 = applyBounds(p0+1, s, params);
  return int4{ p000.y, p000.y, p111.y, p111.y }; // ORDER --> { Y00, Y10,  Y01, Y11 }
}
template<typename T>
__device__ int4 texPutIZ(T p, const FluidParams<typename Dim<T>::BASE_T> &params)
{
  typedef typename Dim<T>::SIZE_T IT;
  IT s  = params.fs;
  IT p0  = IT  {int(floor(p.x)), int(floor(p.y)), int(floor(p.z))};   // integer position
  IT p000 = applyBounds(p0,   s, params);
  IT p111 = applyBounds(p0+1, s, params);
  return int4{ p000.z, p000.z, p111.z, p111.z }; // ORDER --> { Z00, Z10,  Z01, Z11 }
}
template<typename T=float>
__device__ float4 texPutMults0(const typename DimType<T, 3>::VEC_T &p)
{
  typedef typename DimType<T, 3>::VEC_T VT;
  typedef typename Dim<VT>::SIZE_T IT;
  IT p0 = IT{int(floor(p.x)), int(floor(p.y)), int(floor(p.z))}; // integer position
  VT fp = abs(VT{p.x-p0.x, p.y-p0.y, p.z-p0.z});                 // fractional position
  T m00 = (-fp.x + 1.0f) * (-fp.y + 1.0f);
  T m10 = (fp.x) * (-fp.y + 1.0f);
  T m01 = (-fp.x + 1.0f) * (fp.y);
  T m11 = (fp.x) * (fp.y);
  return float4{ m00*(1.0f-fp.z), m10*(1.0f-fp.z), m01*(1.0f-fp.z), m11*(1.0f-fp.z) };
}

template<typename T=float>
__device__ float4 texPutMults1(const typename DimType<T, 3>::VEC_T &p)
{
  typedef typename DimType<T, 3>::VEC_T VT;
  typedef typename Dim<VT>::SIZE_T IT;
  IT p0 = IT{int(floor(p.x)), int(floor(p.y)), int(floor(p.z))}; // integer position
  VT fp = abs(VT{p.x-p0.x, p.y-p0.y, p.z-p0.z});                 // fractional position
  T m00 = (-fp.x + 1.0f) * (-fp.y + 1.0f);
  T m10 = (fp.x) * (-fp.y + 1.0f);
  T m01 = (-fp.x + 1.0f) * (fp.y);
  T m11 = (fp.x) * (fp.y);
  return float4{ m00*fp.z, m10*fp.z, m01*fp.z, m11*fp.z };
}

// NOTE: atomic operation
template<typename T>
__device__ void putTex2DD(T *tex, T val, T p, const FluidParams<typename Dim<T>::BASE_T> &params)
{
  typedef T VT;
  typedef typename Dim<T>::SIZE_T IT;
  IT p0 = IT{int(floor(p.x)), int(floor(p.y)), int(floor(p.z))}; // integer position
  VT fp = abs(VT{p.x-p0.x, p.y-p0.y, p.z-p0.z});                 // fractional position
  IT s  = params.fs;
  p0 = applyBounds(p0+1, s, params);
  
  IT p000 = applyBounds(p0,   s, params);
  IT p111 = applyBounds(p0+1, s, params);
  float4 mults0 = texPutMults0(p);
  float4 mults1 = texPutMults1(p);
  float4 tiX = texPutIX(p, params);
  float4 tiY = texPutIY(p, params);
  float4 tiZ = texPutIZ(p, params);
  p000    = IT{ tiX.x, tiY.x, tiZ.x }; IT p110 = IT{ tiX.w, tiY.w, tiZ.x };
  IT p010 = IT{ tiX.z, tiY.z, tiZ.x }; IT p100 = IT{ tiX.y, tiY.y, tiZ.x };
  IT p001 = IT{ tiX.x, tiY.x, tiZ.z };    p111 = IT{ tiX.w, tiY.w, tiZ.z };
  IT p011 = IT{ tiX.z, tiY.z, tiZ.z }; IT p101 = IT{ tiX.y, tiY.y, tiZ.z };
  if(p000.x < 0 || p111.x < 0 || p000.y < 0 || p111.y < 0 || p000.z < 0 || p111.z < 0) { return; }

  // scale value by grid overlap and store in each location
  texAtomicAdd(tex, val*mults0.x, p000, params); texAtomicAdd(tex, val*mults0.y, p100, params);
  texAtomicAdd(tex, val*mults0.z, p010, params); texAtomicAdd(tex, val*mults0.w, p110, params);
  texAtomicAdd(tex, val*mults1.x, p001, params); texAtomicAdd(tex, val*mults1.y, p101, params);
  texAtomicAdd(tex, val*mults1.z, p011, params); texAtomicAdd(tex, val*mults1.w, p111, params);
}



template<typename SCALAR_T, typename VECTOR_T>
__device__ VECTOR_T integrateReverseEuler(VECTOR_T *data, const VECTOR_T &p, const VECTOR_T &v, SCALAR_T dt)
{ // trace velocity backward
  return p - v*dt;
}

template<typename SCALAR_T, typename VECTOR_T>
__device__ VECTOR_T integrateForwardEuler(VECTOR_T *data, const VECTOR_T &p, const VECTOR_T &v, SCALAR_T dt)
{ // trace velocity forward
  return p + v*dt;
}




// //
// //// 2D ////
// //

// // currently implements WRAP addressing
// template<typename T>
// __device__ T texGet(T *tex, int x, int y, int w, int h)
// {
//   if(x < 0) { x = 0; } else if(x >= w) { x = w-1; }
//   if(y < 0) { y = 0; } else if(y >= h) { y = h-1; }
//   return tex[y*w + x];
// }
// // currently implements WRAP addressing
// template<typename T>
// __device__ void texPut(T *tex, T val, int x, int y, int w, int h)
// {
//   if(x < 0) { x = 0; } if(x >= w) { x = w-1; }
//   if(y < 0) { y = 0; } if(y >= h) { y = h-1; }
//   tex[y*w + x] = val;
// }
// template<typename T>
// __device__ T tex2DD(T *tex, float x, float y, int w, int h)
// {
//   x -= 0.5f; y -= 0.5f;
//   int2    p = int2  {int(floor(x)), int(floor(y))}; // integer position
//   float2 fp = float2{x-p.x, y-p.y};                 // fractional position
  
//   if(p.x > 0 && p.x < w-1 && p.y > 0 && p.y < h-1)
//     {
//       T result = lerp<T>(lerp<T>(texGet(tex, p.x, p.y,   w, h), texGet(tex, p.x+1, p.y,   w, h), fp.x),
//                          lerp<T>(texGet(tex, p.x, p.y+1, w, h), texGet(tex, p.x+1, p.y+1, w, h), fp.x), fp.y);
//       return T(isnan(result) ? T{0.0} : result);
//     }
//   else { return T{0.0f}; }
// }



// //
// //// 3D ////
// //

// // currently implements WRAP addressing
// template<typename T>
// __device__ T texGet3(T *tex, int x, int y, int z, int w, int h, int d)
// {
//   if(x < 0) { x = 0; } else if(x >= w) { x = w-1; }
//   if(y < 0) { y = 0; } else if(y >= h) { y = h-1; }
//   if(z < 0) { z = 0; } else if(z >= d) { z = d-1; }
//   return tex[x + w*(y + h*(z))];
// }
// // currently implements WRAP addressing
// template<typename T>
// __device__ void texPut3(T *tex, T val, int x, int y, int z, int w, int h, int d)
// {
//   if(x < 0) { x = 0; } if(x >= w) { x = w-1; }
//   if(y < 0) { y = 0; } if(y >= h) { y = h-1; }
//   if(z < 0) { z = 0; } if(z >= d) { z = d-1; }
//   tex[x + w*(y + h*(z))] = val;
// }
// template<typename T>
// __device__ T tex3DD(T *tex, float x, float y, float z, int w, int h, int d)
// {
//   x -= 0.5f; y -= 0.5f; z -= 0.5f;
//   int3    p = int3  {int(floor(x)), int(floor(y)), int(floor(z))}; // integer position
//   float3 fp = float3{x-p.x, y-p.y, z-p.z};                         // fractional position
  
//   if(p.x > 0 && p.x < w-1 && p.y > 0 && p.y < h-1 && p.z > 0 && p.z < d-1)
//     {
//       T resultz1 = lerp<T>(lerp<T>(texGet3(tex, p.x, p.y,   p.z,   w, h, d), texGet3(tex, p.x+1, p.y,   p.z,   w, h, d), fp.x),
//                            lerp<T>(texGet3(tex, p.x, p.y+1, p.z,   w, h, d), texGet3(tex, p.x+1, p.y+1, p.z,   w, h, d), fp.x), fp.y);
//       T resultz2 = lerp<T>(lerp<T>(texGet3(tex, p.x, p.y,   p.z+1, w, h, d), texGet3(tex, p.x+1, p.y,   p.z+1, w, h, d), fp.x),
//                            lerp<T>(texGet3(tex, p.x, p.y+1, p.z+1, w, h, d), texGet3(tex, p.x+1, p.y+1, p.z+1, w, h, d), fp.x), fp.y);
//       T result = lerp<T>(resultz1, resultz2, fp.z);
//       return T(isnan(result) ? T{0.0} : result);
//     }
//   else { return T{0.0f}; }
// }



// // maximization reduction
// // template<typename T, unsigned int blockSize> __global__ void fieldMax_k(CudaField<T> fieldIn, CudaField<T> fieldOut, unsigned int n);
// class CudaFieldBase;
// template<typename T> class CudaField;
// template<typename T> float fieldMax(CudaFieldBase *field, CudaFieldBase *fieldOut, CudaField<float> *dst);
// template<typename T> float fieldNorm(CudaFieldBase *field, CudaFieldBase *fieldOut, CudaField<float> *dst);




#endif // CUDA_TOOLS_CUH
