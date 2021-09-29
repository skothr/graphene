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

#ifdef __NVCC__

// render a single cell with simple main color options (for speed)
template<typename T>
__device__ typename DimType<T, 4>::VEC_T renderCellSimple(FluidField<T> &src, unsigned long i, const RenderParams<T> &rp, const FluidParams<T> &cp)
{
  typedef typename DimType<T,4>::VEC_T VT4;
  
  VT4 color = rp.fBrightness*rp.fOpacity*(length(src.v[i])      * rp.getFinalColor(FLUID_RENDER_V)   +
                                          src.p[i]              * rp.getFinalColor(FLUID_RENDER_P)   +
                                          src.div[i]            * rp.getFinalColor(FLUID_RENDER_D));
  color  += rp.emBrightness*rp.emOpacity*((src.Qp[i]-src.Qn[i]) * rp.getFinalColor(FLUID_RENDER_Q)    +
                                          length(src.Qv[i])     * rp.getFinalColor(FLUID_RENDER_QV)   +
                                          length(src.E[i])      * rp.getFinalColor(FLUID_RENDER_E)    +
                                          length(src.B[i])      * rp.getFinalColor(FLUID_RENDER_B));
  auto M = src.mat[i];
  if(!M.vacuum())
    {
      color += rp.matBrightness*rp.matOpacity*(M.ep      * rp.getFinalColor(FLUID_RENDER_EP)  +
                                               M.mu      * rp.getFinalColor(FLUID_RENDER_MU)  +
                                               M.sig     * rp.getFinalColor(FLUID_RENDER_SIG));
    }
  return color;
}




// render a single cell with extended options
template<typename T>
__device__ typename DimType<T, 4>::VEC_T renderCellAll(FluidField<T> &src, unsigned long i, const RenderParams<T> &rp, const FluidParams<T> &cp)
{
  typedef typename DimType<T,3>::VEC_T VT3;
  typedef typename DimType<T,4>::VEC_T VT4;

  static const VT3 z3{0.0, 0.0, 0.0};
  
  const VT3 v   = src.v[i];
  const T   p   = src.p[i];
  const T   d   = src.div[i];
  const T   Qn  = src.Qn[i];
  const T   Qp  = src.Qp[i];
  const VT3 Qv  = src.Qv[i];
  const VT3 E   = src.E[i];
  const VT3 B   = src.B[i];
  
  const VT3 vp  =  max(v,  z3);
  const VT3 vn  = -min(v,  z3);
  const VT3 Qvp =  max(Qv, z3);
  const VT3 Qvn = -min(Qv, z3);
  const T   Q   = (Qp - Qn);
  
  const T vLen  = length(v);
  const T QvLen = length(Qv);
  const T ELen  = length(src.E[i]);
  const T BLen  = length(src.B[i]);
  
  VT4 color = rp.fBrightness*rp.fOpacity*(v.x   * rp.getFinalColor(FLUID_RENDER_VX)  +
                                          v.y   * rp.getFinalColor(FLUID_RENDER_VY)  +
                                          v.z   * rp.getFinalColor(FLUID_RENDER_VZ)  +
                                          vLen  * rp.getFinalColor(FLUID_RENDER_V)   +
                                          vp.x  * rp.getFinalColor(FLUID_RENDER_PVX) +
                                          vp.y  * rp.getFinalColor(FLUID_RENDER_PVY) +
                                          vp.z  * rp.getFinalColor(FLUID_RENDER_PVZ) +
                                          vn.x  * rp.getFinalColor(FLUID_RENDER_NVX) +
                                          vn.y  * rp.getFinalColor(FLUID_RENDER_NVY) +
                                          vn.z  * rp.getFinalColor(FLUID_RENDER_NVZ) +
                                          
                                          p     * rp.getFinalColor(FLUID_RENDER_P)   +
                                          d     * rp.getFinalColor(FLUID_RENDER_D)
                                          );
  
  color += rp.emBrightness*rp.emOpacity*( Q     * rp.getFinalColor(FLUID_RENDER_Q)    +
                                          Qn    * rp.getFinalColor(FLUID_RENDER_QN)   +
                                          Qp    * rp.getFinalColor(FLUID_RENDER_QP)   +
                                          
                                          Qv.x  * rp.getFinalColor(FLUID_RENDER_QVX)  +
                                          Qv.y  * rp.getFinalColor(FLUID_RENDER_QVY)  +
                                          Qv.z  * rp.getFinalColor(FLUID_RENDER_QVZ)  +
                                          QvLen * rp.getFinalColor(FLUID_RENDER_QV)   +
                                          Qvp.x * rp.getFinalColor(FLUID_RENDER_QPVX) +
                                          Qvp.y * rp.getFinalColor(FLUID_RENDER_QPVY) +
                                          Qvp.z * rp.getFinalColor(FLUID_RENDER_QPVZ) +
                                          Qvn.x * rp.getFinalColor(FLUID_RENDER_QNVX) +
                                          Qvn.y * rp.getFinalColor(FLUID_RENDER_QNVY) +
                                          Qvn.z * rp.getFinalColor(FLUID_RENDER_QNVZ) +
                                          
                                          E.x   * rp.getFinalColor(FLUID_RENDER_EX)   +
                                          E.y   * rp.getFinalColor(FLUID_RENDER_EY)   +
                                          E.z   * rp.getFinalColor(FLUID_RENDER_EZ)   +
                                          ELen  * rp.getFinalColor(FLUID_RENDER_E)    +
                                          B.x   * rp.getFinalColor(FLUID_RENDER_BX)   +
                                          B.y   * rp.getFinalColor(FLUID_RENDER_BY)   +
                                          B.z   * rp.getFinalColor(FLUID_RENDER_BZ)   +
                                          BLen  * rp.getFinalColor(FLUID_RENDER_B)
                                          );
  auto M = src.mat[i];
  if(!M.vacuum())
    {
      color += rp.matBrightness*rp.matOpacity*(M.ep      * rp.getFinalColor(FLUID_RENDER_EP)  +
                                               M.mu      * rp.getFinalColor(FLUID_RENDER_MU)  +
                                               M.sig     * rp.getFinalColor(FLUID_RENDER_SIG) +
                                               M.n(cp.u) * rp.getFinalColor(FLUID_RENDER_N)
                                               );
    }
  return color;
}




// blend fluid colors from one cell to the next
template<typename T>
__device__ inline typename DimType<T, 4>::VEC_T& fluidBlend(typename DimType<T, 4>::VEC_T &rayColor,
                                                            const typename DimType<T, 4>::VEC_T &cellColor,
                                                            const RenderParams<T> &rp)
{
  typedef typename DimType<T, 4>::VEC_T VT4;
  T a = rayColor.w;
  rayColor += VT4{cellColor.x, cellColor.y, cellColor.z, 0.0} * cellColor.w*(1-a)*rp.emBrightness;
  rayColor.w += cellColor.w*(1-rayColor.w)*(rp.emOpacity);
  return rayColor;
}



// (to prevent infinite loops to some extent)
#define MAX_ITER  16
#define MAX_ITER2  4

// render 3D --> raytrace field
template<typename T>
__device__ typename DimType<T,4>::VEC_T rayTraceField(FluidField<T> &src, Ray<T> ray, const RenderParams<T> &rp, const FluidParams<T> &cp)
{
  typedef typename DimType<T,2>::VEC_T VT2;
  typedef typename DimType<T,3>::VEC_T VT3;
  typedef typename DimType<T,4>::VEC_T VT4;

  const T   tol = TOL;
  const VT3 fPos    = VT3{(T)cp.fp.x,    (T)cp.fp.y,    (T)(cp.fp.z + rp.zRange.x) + tol};
  const VT3 fSize   = VT3{(T)src.size.x, (T)src.size.y, (T)min(src.size.z, rp.zRange.y-rp.zRange.x+1) - tol};
  const T   maxDim  = dot(fSize, fSize); // maximum allowed steps (extra room)
  
  const VT2 tp = cubeIntersect(fPos, fSize, ray); // { tmin, tmax }
  T t = tp.x; // tmin
  if(t >= 0.0)
    {
      VT3 pSrc = rp.penPos; VT3 diff; VT3 diff0; VT3 diff1;  VT3 dist_2; VT3 dist0_2; VT3 dist1_2; // for pen highlighting
      VT3 wp = ray.pos + ray.dir*(t+tol); // world-space pos of primary intersection
      VT3 fp = floor(wp - fPos);
      
      // cube marching
      VT4 color = VT4{0.0, 0.0, 0.0, 0.0};
      int iterations = 0;
      while((t < tp.y || (tp.x == (T)0)) &&
            (!rp.surfaces || (color.x < 1.0 && color.y < 1.0 && color.z < 1.0)) && color.w < 1.0 &&
            iterations < maxDim)
        {
          if(fp.x < 0 || fp.x >= src.size.x || // double check index range
             fp.y < 0 || fp.y >= src.size.y ||
             fp.z < 0 || fp.z+rp.zRange.x >= src.size.z) { break; }
          unsigned long i = src.idx((unsigned long)fp.x, (unsigned long)fp.y, (unsigned long)(fp.z+rp.zRange.x));

          // base cell color
          float4 col = (rp.simple ? renderCellSimple(src, i, rp, cp) : renderCellAll(src, i, rp, cp));

          // highlight if active pen
          VT3 pCell = fp; pCell.z += rp.zRange.x;
          if(rp.sigPenHighlight &&
             penOverlap3 (pCell, pSrc, diff, diff0, diff1, dist_2, dist0_2, dist1_2, (Pen<T>*)&rp.sigPen, cp, 0.0f) &&
             !penOverlap3(pCell, pSrc, diff, diff0, diff1, dist_2, dist0_2, dist1_2, (Pen<T>*)&rp.sigPen, cp, -1.0f)) { col += SIG_HIGHLIGHT_COLOR; }
          if(rp.matPenHighlight &&
             penOverlap3 (pCell, pSrc, diff, diff0, diff1, dist_2, dist0_2, dist1_2, (Pen<T>*)&rp.matPen, cp, 0.0f) &&
             !penOverlap3(pCell, pSrc, diff, diff0, diff1, dist_2, dist0_2, dist1_2, (Pen<T>*)&rp.matPen, cp, -1.0f)) { col += MAT_HIGHLIGHT_COLOR; }

          // calculate next grid intersection
          T tLast = t;
          VT3 fp2 = fp;
          int iterations2 = 0;
          bool same = false;
          while(same=((T)floor(fp2.x) == (T)floor(fp.x) && (T)floor(fp2.y) == (T)floor(fp.y) && (T)floor(fp2.z) == (T)floor(fp.z)) &&
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
              step = abs(step/ray.dir);       // project distance onto ray
              t   += min(step) + tol;
              
              wp   = ray.pos + ray.dir*(t+tol);
              fp2  = (wp - fPos);
              iterations2++;
            }
          if(same) { break; }
          
          // scale blending based on distance travelled to next intersection
          col.w *= (t - tLast);
          fluidBlend(color, col, rp);
          fp = fp2; iterations++;
        }
      // blend with background color
      T a = color.w;
      color += VT4{BG_COLOR.x, BG_COLOR.y, BG_COLOR.z, 0.0} * BG_COLOR.w*(1-a*rp.emBrightness);
      return VT4{min(1.0f, color.x), min(1.0f, color.y), min(1.0f, color.z), 1.0f};
    }
  else { return BG_COLOR; }
}


#endif // __NVCC__

#endif // RAYTRACE_CUH
