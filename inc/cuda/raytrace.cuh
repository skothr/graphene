#ifndef RAYTRACE_CUH
#define RAYTRACE_CUH

#include <cuda_runtime.h>
#include "field.hpp"
#include "vector-operators.h"
#include "raytrace.h"

#define TOL 0.0001 // tolerance/epsilon to make sure ray fully intersects
#define BG_COLOR      float4{0.1f, 0.1f, 0.1f, 1.0f} // color of background behind field
#define FAIL_COLOR    float4{1.0f, 0.0f, 1.0f, 1.0f} // color returned on failure/error

#ifdef ENABLE_CUDA

// vector type shorthand for declarations
template<typename T> using VT2_t = typename DimType<T,2>::VECTOR_T;
template<typename T> using VT3_t = typename DimType<T,3>::VECTOR_T;

template<typename T>
__device__ T planeIntersect(const VT3_t<T> &p, const VT3_t<T> &n, const Ray<T> &ray)
{
  T denom = dot(n, ray.dir);
  T t = -1.0;
  if(abs(denom) > TOL) { t = dot((p - ray.pos), n) / denom; }
  return t; // no intersection if t < 0
}

// render 3D --> raytrace field
//  - field assumed to be size (1,1,1) in 3D space
//  - return value < 0 means ray missed, value == 0 means ray started inside cube
// returns {tmin, tmax}
template<typename T>
__device__ VT2_t<T> cubeIntersect(const VT3_t<T> &pos, const VT3_t<T> &size, const Ray<T> &ray)
{
  typedef VT2_t<T> VT2; typedef VT3_t<T> VT3;
  T tnx = (pos.x - ray.pos.x)          / ray.dir.x;
  T tpx = (pos.x - ray.pos.x + size.x) / ray.dir.x;
  T tny = (pos.y - ray.pos.y)          / ray.dir.y;
  T tpy = (pos.y - ray.pos.y + size.y) / ray.dir.y;
  T tnz = (pos.z - ray.pos.z)          / ray.dir.z;
  T tpz = (pos.z - ray.pos.z + size.z) / ray.dir.z;
  T tmin = max(max(min(tnx, tpx), min(tny, tpy)), min(tnz, tpz));
  T tmax = min(min(max(tnx, tpx), max(tny, tpy)), max(tnz, tpz));
  return (tmin < 0 ? VT2_t<T>{0,0} : (tmin > tmax) ? VT2_t<T>{-1.0,-1.0} : VT2_t<T>{tmin, tmax});
}

template<typename T>
__device__ float4 renderCell(ChargeField<T> &src, long long i, const EmRenderParams &rp)
{
  typedef VT2_t<T> VT2; typedef VT3_t<T> VT3;
  T qLen = (src.Q[i].x - src.Q[i].y);
  T eLen = length(src.E[i]);
  T bLen = length(src.B[i]);
  float4 color = rp.brightness*rp.opacity*(qLen*rp.Qmult*rp.Qcol + eLen*rp.Emult*rp.Ecol + bLen*rp.Bmult*rp.Bcol);
  //color.w *= (qLen+eLen+bLen);
  return color;
}

// (to prevent infinite loops to some extent)
#define MAX_ITER 16
#define MAX_ITER2 4

__device__ float4& fluidBlend(float4 &rayColor, const float4 &cellColor, const EmRenderParams &rp)
{
  float a = rayColor.w;
  rayColor += float4{cellColor.x, cellColor.y, cellColor.z, 0.0} * cellColor.w*(1-a)*rp.brightness;
  rayColor.w += cellColor.w*(1-rayColor.w)*(rp.opacity);
  return rayColor;
}

// render 3D --> raytrace field
template<typename T>
__device__ float4 rayTraceField(ChargeField<T> &src, const Ray<double> &ray, const EmRenderParams &rp, const ChargeParams &cp)
{
  //typedef VT2_t<double> VT2; typedef VT3_t<double> VT3;
  typedef double2 VT2; typedef double3 VT3;

  VT3 cs = VT3{cp.cs.x, cp.cs.y, cp.cs.z};
  VT3 fs = VT3{(double)src.size.x, (double)src.size.y, (double)src.size.z}; // field size vector
  
  VT3 fPos  = -cs*fs/2.0;
  VT3 fSize =  cs*fs;
  
  VT2 tp = cubeIntersect(fPos, fSize, ray); // { tmin, tmax }
  T t = tp.x; // tmin
  if(t >= -TOL)
    {
      VT3 wp = ray.pos + ray.dir*(t+TOL); // world-space pos of primary intersection
      VT3 fp = (wp - fPos) / fSize * fs;
      // cube marching
      float4 color = float4{0.0f, 0.0f, 0.0f, 0.0f};
      while((t < tp.y || tp.x == 0))// && color.x < 1 && color.y < 1 && color.z < 1)
        {
          if(fp.x < 0 || fp.x >= src.size.x || fp.y < 0 || fp.y >= src.size.y || fp.z < 0 || fp.z >= src.size.z) { break; }
          unsigned long long i = src.idx((int)fp.x, (int)fp.y, (int)fp.z);

          fluidBlend(color, renderCell(src, i, rp), rp);
          
          VT3 fp2 = fp;
          while((int)(fp2.x) == (int)(fp.x) && (int)(fp2.y) == (int)(fp.y) && (int)(fp2.z) == (int)(fp.z) &&
                (t <= tp.y || tp.x == (T)0))
            {
              VT3 pi    = VT3{(T)(ray.dir.x < 0 ? ceil(fp2.x) : floor(fp2.x)), // fractional distance past current grid index along ray trajectory
                              (T)(ray.dir.y < 0 ? ceil(fp2.y) : floor(fp2.y)),
                              (T)(ray.dir.z < 0 ? ceil(fp2.z) : floor(fp2.z)) };
              VT3 dSign = VT3{(T)(ray.dir.x < 0 ? -1 : 1),
                              (T)(ray.dir.y < 0 ? -1 : 1),
                              (T)(ray.dir.z < 0 ? -1 : 1)};
              VT3 step  = (pi + dSign) - fp2; // distance to next grid index in each dimension
              step = fmod(step, VT3{(T)1, (T)1, (T)1});  // (TODO: necessary?)
              step = abs(step/fs)*fSize; // convert to world coordinates
              step = abs(step/ray.dir);  // project distance onto ray
              t += min(step.x, min(step.y, step.z)) + 0.001;
              VT3 wp2 = ray.pos + ray.dir*(t+TOL);
              fp2 = (wp2 - fPos) / fSize * fs;
            }
          fp = fp2;
        }
      float a = color.w;
      color += float4{BG_COLOR.x, BG_COLOR.y, BG_COLOR.z, 0.0} * BG_COLOR.w*(1-a*rp.brightness);
      color.w += BG_COLOR.w*(1-color.w)*(rp.opacity);
      return float4{color.x, color.y, color.z, 1.0f};
    }
  else { return BG_COLOR; }
}


// TODO: generalize cube marching
// __device__ VT2 cubeMarch(const VT3 &fPos, const VT3 &fSize, const VT3 &rPos, const VT3 &rDir)
// {
//   //float4 color = float4{0.0f, 0.0f, 0.0f, 1.0f};
//   VT tnx = (fPos.x - rPos.x)           / rDir.x;
//   VT tpx = (fPos.x - rPos.x + fSize.x) / rDir.x;
//   VT tny = (fPos.y - rPos.y)           / rDir.y;
//   VT tpy = (fPos.y - rPos.y + fSize.y) / rDir.y;
//   VT tnz = (fPos.z - rPos.z)           / rDir.z;
//   VT tpz = (fPos.z - rPos.z + fSize.z) / rDir.z;  
//   VT tmin = max(max(min(tnx, tpx), min(tny, tpy)), min(tnz, tpz));
//   VT tmax = min(min(max(tnx, tpx), max(tny, tpy)), max(tnz, tpz));
//   if(tmin > tmax) { return VT2{-1.0, -1.0}; } // no intersection
//   else            { return VT2{tmin, tmax}; }
// }












// OLD (?)
// // render 3D --> raytrace field
// //  - field assumed to be size (1,1,1) in 3D space
// __device__ float4 rayTraceFluid(HyperFluid<3> &src, float3 fPos, float3 rPos, float3 rDir)
// {
//   //float4 color = float4{0.0f, 0.0f, 0.0f, 1.0f};
//   float tnx = (fPos.x - rPos.x)     / rDir.x;
//   float tpx = (fPos.x - rPos.x + 1) / rDir.x;
//   float tny = (fPos.y - rPos.y)     / rDir.y;
//   float tpy = (fPos.y - rPos.y + 1) / rDir.y;
//   float tnz = (fPos.z - rPos.z)     / rDir.z;
//   float tpz = (fPos.z - rPos.z + 1) / rDir.z;
//   float tmin = min(min(min(tnx, tpx), min(tny, tpy)), min(tnz, tpz));
//   float tmax = max(max(max(tnx, tpx), max(tny, tpy)), max(tnz, tpz));
//   if(tmin > tmax) // no intersection
//     { return float4{0.0f, 0.0f, 0.0f, -1.0f}; }
//   else
//     { return float4{1.0f, 1.0f, 1.0f, 1.0f}; }
// }
// template instantiation
//template __device__ float4 rayTraceField<float>(const ChargeField<float> &src, const Ray<float> &ray, const EmRenderParams &rp, const ChargeParams &cp);


#endif // ENABLE_CUDA

#endif // RAYTRACE_CUH
