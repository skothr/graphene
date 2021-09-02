#ifndef RAYTRACE_CUH
#define RAYTRACE_CUH

#include <cuda_runtime.h>
#include "field.cuh"
#include "vector-operators.h"
#include "raytrace.h"
#include "render.cuh"

#define BG_COLOR      VT4{0.08, 0.08, 0.08, 1.0} // color of background behind field
#define FAIL_COLOR    VT4{1.0, 0.0, 1.0, 1.0} // color returned on failure/error

#define SIG_HIGHLIGHT_COLOR float4{0.5, 1.0, 0.5, 0.1}
#define MAT_HIGHLIGHT_COLOR float4{1.0, 0.5, 0.5, 0.1}

#ifdef ENABLE_CUDA

template<typename T>
__device__ typename DimType<T, 4>::VECTOR_T renderCell(EMField<T> &src, unsigned long long i, const RenderParams<T> &rp)
{
  typedef typename DimType<T,2>::VECTOR_T VT2;
  typedef typename DimType<T,3>::VECTOR_T VT3;
  typedef typename DimType<T,4>::VECTOR_T VT4;
  T qLen = (src.Q[i].x - src.Q[i].y);
  T eLen = length(src.E[i]);
  T bLen = length(src.B[i]);
  VT4 color = rp.brightness*rp.opacity*(qLen*rp.Qmult*rp.Qcol + eLen*rp.Emult*rp.Ecol + bLen*rp.Bmult*rp.Bcol);
  //color.w *= (qLen+eLen+bLen);
  return color;
}

// (to prevent infinite loops to some extent)
#define MAX_ITER 16
#define MAX_ITER2 4

template<typename T>
__device__ inline typename DimType<T, 4>::VECTOR_T& fluidBlend(typename DimType<T, 4>::VECTOR_T &rayColor,
                                                               const typename DimType<T, 4>::VECTOR_T &cellColor,
                                                               const RenderParams<T> &rp)
{
  typedef typename DimType<T, 4>::VECTOR_T VT4;
  T a = rayColor.w;
  rayColor += VT4{cellColor.x, cellColor.y, cellColor.z, 0.0} * cellColor.w*(1-a)*rp.brightness;
  rayColor.w += cellColor.w*(1-rayColor.w)*(rp.opacity);
  return rayColor;
}

// render 3D --> raytrace field
template<typename T>
__device__ typename DimType<T,4>::VECTOR_T rayTraceField(EMField<T> &src, Ray<T> ray, const RenderParams<T> &rp, const FieldParams<T> &cp)
{
  typedef typename DimType<T,2>::VECTOR_T VT2;
  typedef typename DimType<T,3>::VECTOR_T VT3;
  typedef typename DimType<T,4>::VECTOR_T VT4;

  const int zl = rp.zRange.y - rp.zRange.x + 1; // number of z layers to render

  ray.pos /= cp.u.dL;
  
  const T   tol = TOL;
  const VT3 fPos  = VT3{(T)cp.fp.x,    (T)cp.fp.y,    (T)(cp.fp.z + rp.zRange.x)};
  const VT3 fSize = VT3{(T)src.size.x, (T)src.size.y, (T)min(src.size.z, zl)};
  const T maxDim = dot(fSize, fSize); // maximum allowed steps (extra room)
  
  const VT2 tp = cubeIntersect(fPos, fSize, ray); // { tmin, tmax }
  T t = tp.x; // tmin
  if(t >= 0.0)
    {
      VT3 pSrc = rp.penPos; VT3 diff; VT3 diff0; VT3 diff1;  VT3 dist_2; VT3 dist0_2; VT3 dist1_2; // for pen highlighting

      VT3 wp = ray.pos + ray.dir*(t+tol); // world-space pos of primary intersection
      VT3 fp = (wp - fPos);// / cs;
      fp = floor(fp);
      
      // cube marching
      VT4 color = VT4{0.0, 0.0, 0.0, 0.0};
      int iterations = 0;
      while((t < tp.y || (tp.x == (T)0))&&// && fp.z >= rp.zRange.x-1 && fp.z <= rp.zRange.y+1)) &&
            (!rp.surfaces || (color.x < 1.0 && color.y < 1.0 && color.z < 1.0)) &&
            color.w < 1.0 && iterations < maxDim)
        {
          if(fp.x < 0 || fp.x >= src.size.x ||
             fp.y < 0 || fp.y >= src.size.y ||
             fp.z < 0 || fp.z+rp.zRange.x >= src.size.z) { break; } // || fp.z+rp.zRange.x > rp.zRange.y+1) { break; }
          unsigned long long i = src.idx((unsigned long long)fp.x, (unsigned long long)fp.y, (unsigned long long)(fp.z+rp.zRange.x));

          // base cell color
          float4 col = renderCell(src, i, rp);

          // highlight if active pen
          VT3 pCell = fp; pCell.z += rp.zRange.x;
          if(rp.sigPenHighlight &&
             penOverlaps2 (pCell, pSrc, diff, diff0, diff1, dist_2, dist0_2, dist1_2, (Pen<T>*)&rp.sigPen, cp, 0.0f) &&
             !penOverlaps2(pCell, pSrc, diff, diff0, diff1, dist_2, dist0_2, dist1_2, (Pen<T>*)&rp.sigPen, cp, -1.0f)) { col += SIG_HIGHLIGHT_COLOR; }
          if(rp.matPenHighlight &&
             penOverlaps2 (pCell, pSrc, diff, diff0, diff1, dist_2, dist0_2, dist1_2, (Pen<T>*)&rp.matPen, cp, 0.0f) &&
             !penOverlaps2(pCell, pSrc, diff, diff0, diff1, dist_2, dist0_2, dist1_2, (Pen<T>*)&rp.matPen, cp, -1.0f)) { col += MAT_HIGHLIGHT_COLOR; }

          // calculate next grid intersection
          T tLast = t;
          VT3 fp2 = fp;
          int iterations2 = 0;
          bool same = ((T)floor(fp2.x) == (T)floor(fp.x) && (T)floor(fp2.y) == (T)floor(fp.y) && (T)floor(fp2.z) == (T)floor(fp.z));
          while(same &&
                (t < tp.y || (tp.x == (T)0 && fp2.z >= rp.zRange.x-1 && fp2.z+rp.zRange.x <= rp.zRange.y)) &&
                iterations2 < MAX_ITER)
            {
              VT3 pi    = VT3{(T)(ray.dir.x < 0 ? ceil(fp2.x) : floor(fp2.x)), // fractional distance past current grid index along ray trajectory
                              (T)(ray.dir.y < 0 ? ceil(fp2.y) : floor(fp2.y)),
                              (T)(ray.dir.z < 0 ? ceil(fp2.z) : floor(fp2.z)) };
              VT3 dSign = VT3{(T)(ray.dir.x < 0 ? -1 : 1),
                              (T)(ray.dir.y < 0 ? -1 : 1),
                              (T)(ray.dir.z < 0 ? -1 : 1)};
              VT3 step  = (pi + dSign) - fp2; // distance to next grid index in each dimension
              step = abs(step/ray.dir);  // project distance onto ray
              t += min(step.x, min(step.y, step.z)) + 1.0*tol;
              wp = ray.pos + ray.dir*(t+tol);
              fp2 = (wp - fPos);
              same = ((T)floor(fp2.x) == (T)floor(fp.x) && (T)floor(fp2.y) == (T)floor(fp.y) && (T)floor(fp2.z) == (T)floor(fp.z));
              iterations2++;
            }
          if(same) { break; }
          
          // scale blending based on distance travelled to next intersection
          col.w *= (t-tLast);///sum(cs);
          fluidBlend(color, col, rp);
          
          fp = fp2;
          iterations++;
        }
      // blend with background color
      T a = color.w;
      color += VT4{BG_COLOR.x, BG_COLOR.y, BG_COLOR.z, 0.0} * BG_COLOR.w*(1-a*rp.brightness);
      //color.w += BG_COLOR.w*(1-color.w)*(rp.opacity);
      return VT4{min(1.0f, color.x), min(1.0f, color.y), min(1.0f, color.z), 1.0f};
    }
  else { return BG_COLOR; }
}


#endif // ENABLE_CUDA

#endif // RAYTRACE_CUH
