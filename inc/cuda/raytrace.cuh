#ifndef RAYTRACE_CUH
#define RAYTRACE_CUH

#include <cuda_runtime.h>
#include "field.hpp"
#include "vector-operators.h"
#include "raytrace.h"

#define BG_COLOR      float4{0.1f, 0.1f, 0.1f, 1.0f} // color of background behind field
#define FAIL_COLOR    float4{1.0f, 0.0f, 1.0f, 1.0f} // color returned on failure/error

#ifdef ENABLE_CUDA

// vector type shorthand for declarations
template<typename T> using VT2_t = typename DimType<T,2>::VECTOR_T;
template<typename T> using VT3_t = typename DimType<T,3>::VECTOR_T;

// template<typename T>
// __device__ T planeIntersect(const VT3_t<T> &p, const VT3_t<T> &n, const Ray<T> &ray)
// {
//   T denom = dot(n, ray.dir);
//   T t = -1.0;
//   if(abs(denom) > TOL) { t = dot((p - ray.pos), n) / denom; }
//   return t; // no intersection if t < 0
// }

// // render 3D --> raytrace field
// //  - field assumed to be size (1,1,1) in 3D space
// //  - return value < 0 means ray missed, value == 0 means ray started inside cube
// // returns {tmin, tmax}
// template<typename T>
// __device__ VT2_t<T> cubeIntersect(const VT3_t<T> &pos, const VT3_t<T> &size, const Ray<T> &ray)
// {
//   typedef VT2_t<T> VT2; typedef VT3_t<T> VT3;
//   T tnx = (pos.x - ray.pos.x)          / ray.dir.x;
//   T tpx = (pos.x - ray.pos.x + size.x) / ray.dir.x;
//   T tny = (pos.y - ray.pos.y)          / ray.dir.y;
//   T tpy = (pos.y - ray.pos.y + size.y) / ray.dir.y;
//   T tnz = (pos.z - ray.pos.z)          / ray.dir.z;
//   T tpz = (pos.z - ray.pos.z + size.z) / ray.dir.z;
//   T tmin = max(max(min(tnx, tpx), min(tny, tpy)), min(tnz, tpz));
//   T tmax = min(min(max(tnx, tpx), max(tny, tpy)), max(tnz, tpz));
//   return (tmin < 0 ? VT2_t<T>{0,0} : (tmin > tmax) ? VT2_t<T>{-1.0,-1.0} : VT2_t<T>{tmin, tmax});
// }

template<typename T>
__device__ float4 renderCell(EMField<T> &src, unsigned long long i, const EmRenderParams &rp)
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

__device__ inline float4& fluidBlend(float4 &rayColor, const float4 &cellColor, const EmRenderParams &rp)
{
  float a = rayColor.w;
  rayColor += float4{cellColor.x, cellColor.y, cellColor.z, 0.0} * cellColor.w*(1-a)*rp.brightness;
  rayColor.w += cellColor.w*(1-rayColor.w)*(rp.opacity);
  return rayColor;
}

// render 3D --> raytrace field
template<typename T>
__device__ float4 rayTraceField(EMField<T> &src, const Ray<double> &ray, const EmRenderParams &rp, const FieldParams<T> &cp)
{
  //typedef VT2_t<double> VT2; typedef VT3_t<double> VT3;
  typedef double2 VT2; typedef double3 VT3;

  VT3 cs = VT3{cp.u.dL, cp.u.dL, cp.u.dL};
  VT3 fs = VT3{(T)src.size.x, (T)src.size.y, (T)src.size.z}; // field size vector
  VT3 fp = VT3{(T)cp.fp.x,    (T)cp.fp.y,    (T)cp.fp.z};    // field position vector
  
  //VT3 fPos  = VT3{0.0, 0.0, 0.0};
  VT3 fPos  = cs*fp;
  VT3 fSize = cs*fs;
  T maxDim = max(fs); // minimum size dimension
  
  VT2 tp = cubeIntersect(fPos, fSize, ray); // { tmin, tmax }
  T t = tp.x; // tmin
  if(t >= -TOL)
    {
      VT3 wp = ray.pos + ray.dir*(t+TOL); // world-space pos of primary intersection
      VT3 fp = (wp - fPos) / fSize * fs;
      // cube marching
      float4 color = float4{0.0f, 0.0f, 0.0f, 0.0f};
      int iterations = 0;
      while((t < tp.y || tp.x == (T)0)  &&
            //color.x >= 0.0 && color.y >= 0.0 && color.z >= 0.0 && color.w >= 0.0 &&
            //color.x <= 1.0 && color.y <= 1.0 && color.z <= 1.0 && color.w <= 1.0/rp.opacity &&
            iterations < maxDim)
        {
          if(fp.x < 0 || fp.x >= src.size.x || fp.y < 0 || fp.y >= src.size.y || fp.z < 0 || fp.z >= src.size.z) { break; }
          unsigned long long i = src.idx((unsigned long long)fp.x, (unsigned long long)fp.y, (unsigned long long)fp.z);
          iterations++;
          
          fluidBlend(color, renderCell(src, i, rp), rp);
          
          VT3 fp2 = fp;
          int iterations2 = 0;
          while((T)floor(fp2.x) == (T)floor(fp.x) &&
                (T)floor(fp2.y) == (T)floor(fp.y) &&
                (T)floor(fp2.z) == (T)floor(fp.z) &&
                (t <= tp.y || tp.x == (T)0) && iterations2 < MAX_ITER)
            {
              iterations2++;
              
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
          // if((T)floor(fp2.x) == (T)floor(fp.x) && (T)floor(fp2.y) == (T)floor(fp.y) && (T)floor(fp2.z) == (T)floor(fp.z)) { break; }
          fp = fp2;
        }
      float a = color.w;
      color += float4{BG_COLOR.x, BG_COLOR.y, BG_COLOR.z, 0.0} * BG_COLOR.w*(1-a*rp.brightness);
      color.w += BG_COLOR.w*(1-color.w)*(rp.opacity);
      return float4{min(1.0f, color.x), min(1.0f, color.y), min(1.0f, color.z), 1.0f};
    }
  else { return BG_COLOR; }
}


#endif // ENABLE_CUDA

#endif // RAYTRACE_CUH
