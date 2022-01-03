#ifndef DRAW_CUH
#define DRAW_CUH

#include <cuda.h>

#include "vector-operators.h"
#include "raytrace.h"
#include "material.h"
#include "units.cuh"
#include "drawPens.hpp"

//// CUDA ////
// forward declarations
template<typename T> class  Field;
template<typename T> class  EMField;
template<typename T> struct FieldParams;
template<typename T> struct FluidParams;
template<typename T> class  FluidField;


// overlap helpers
template<typename T, typename VT3=typename cuda_vec<T, 3>::VT>
__device__ inline bool penOverlaps(VT3 &pCell, VT3 &mpos, VT3 &diff,  VT3 &dist_2, const Pen<T> *pen, const FieldParams<T> &cp, T radOffset)
{
  const VT3 rMult = pen->sizeMult*pen->xyzMult;
  
  if(pen->cellAlign) { pCell = floor(pCell); mpos = floor(mpos); }
  VT3 diff0   = pCell-(mpos + rMult*pen->rDist/2);
  VT3 dist0_2 = diff0*diff0;
  VT3 diff1   = pCell-(mpos - rMult*pen->rDist/2);
  VT3 dist1_2 = diff1*diff1;
  
  diff   = (diff0   + diff1)   / 2.0f; // set return values
  dist_2 = (dist0_2 + dist1_2) / 2.0f;

  VT3 r0 = rMult*pen->radius0 + radOffset;
  VT3 r1 = rMult*pen->radius1 + radOffset;
  return ((!pen->square &&                         (dist0_2.x/(r0.x*r0.x) + dist0_2.y/(r0.y*r0.y) + dist0_2.z/(r0.z*r0.z) <= 1.0f) &&
           (r1.x <= 0 || r1.y <= 0 || r1.z <= 0 || (dist1_2.x/(r1.x*r1.x) + dist1_2.y/(r1.y*r1.y) + dist1_2.z/(r1.z*r1.z) <= 1.0f))) ||
          ( pen->square &&                         (abs(diff0.x) <= r0.x+0.5f && abs(diff0.y) <= r0.y+0.5f && abs(diff0.z) <= r0.z+0.5f) &&
           (r1.x <= 0 || r1.y <= 0 || r1.z <= 0 || (abs(diff1.x) <= r1.x+0.5f && abs(diff1.y) <= r1.y+0.5f && abs(diff1.z) <= r1.z+0.5f))));
}

template<typename T, typename VT3=typename cuda_vec<T, 3>::VT>
__device__ inline bool penOverlaps2(VT3 &pCell,  VT3 &mpos, VT3 &diff,   VT3 &diff0,   VT3 &diff1, VT3 &dist_2, VT3 &dist0_2, VT3 &dist1_2,
                                    const Pen<T> *pen, const FieldParams<T> &cp, T radOffset)
{
  const VT3 rMult = pen->sizeMult*pen->xyzMult;
  
  if(pen->cellAlign) { pCell = floor(pCell); mpos = floor(mpos); }
  diff0  = pCell-(mpos + rMult*pen->rDist/2);
  dist0_2 = diff0*diff0;
  diff1  = pCell-(mpos - rMult*pen->rDist/2);
  dist1_2 = diff1*diff1;
  
  diff   = (diff0   + diff1)   / 2.0f; // set return values
  dist_2 = (dist0_2 + dist1_2) / 2.0f;
  
  VT3 r0 = rMult*pen->radius0 + radOffset;
  VT3 r1 = rMult*pen->radius1 + radOffset;
  return ((!pen->square &&                         (dist0_2.x/(r0.x*r0.x) + dist0_2.y/(r0.y*r0.y) + dist0_2.z/(r0.z*r0.z) <= 1.0f) &&
           (r1.x <= 0 || r1.y <= 0 || r1.z <= 0 || (dist1_2.x/(r1.x*r1.x) + dist1_2.y/(r1.y*r1.y) + dist1_2.z/(r1.z*r1.z) <= 1.0f))) ||
          ( pen->square &&                         (abs(diff0.x) <= r0.x+0.5f && abs(diff0.y) <= r0.y+0.5f && abs(diff0.z) <= r0.z+0.5f) &&
           (r1.x <= 0 || r1.y <= 0 || r1.z <= 0 || (abs(diff1.x) <= r1.x+0.5f && abs(diff1.y) <= r1.y+0.5f && abs(diff1.z) <= r1.z+0.5f))));
}

// returns true if point with given distances are at the edge of the pen (hollow shell)
template<typename T, typename VT3=typename cuda_vec<T, 3>::VT>
__device__ inline bool penBorder(VT3 &diff0, VT3 &dist1, const Pen<T> *pen, T width)
{ return (abs(pen->radius0 - diff0) <= width) || (pen->radius1 > 0.0 && abs(pen->radius1 - dist1) <= width); }

// returns true if point with given offsets are at the edge of each axis plane cross-section
template<typename T, typename VT3=typename cuda_vec<T, 3>::VT>
__device__ inline bool penAxisBorder(const VT3 &mpos, const VT3 &diff0, const VT3 &diff1, const Pen<T> *pen, T width)
{
  VT3 dist0XY = VT3{diff0.x, diff0.y, 0.0f};
  VT3 dist0YZ = VT3{diff0.x, 0.0f, diff0.z};
  VT3 dist0ZX = VT3{0.0f, diff0.y, diff0.z};
  VT3 dist1XY = VT3{diff1.x, diff1.y, 0.0f};
  VT3 dist1YZ = VT3{diff1.x, 0.0f, diff1.z};
  VT3 dist1ZX = VT3{0.0f, diff1.y, diff1.z};

  T w = width*length(pen->sizeMult*pen->xyzMult);
  return (penBorder(dist0XY, dist1XY, pen, w) || penBorder(dist0YZ, dist1YZ, pen, w) || penBorder(dist0ZX, dist1ZX, pen, w));
}


// overlap detection (add overlapping components of sphere)
// inline __device__ float3 pointInSphere(const float3 &p, const float3 &cp, float cr)
// { float3 d = p - cp; return dot(d, d) <= cr*cr;  }

// returns intersections
template<typename T, typename VT3=typename cuda_vec<T, 3>::VT>
inline __device__ bool lineIntersectSphere(const VT3 &p1, const VT3 &p2, const VT3 &cp, T cr, 
                                           VT3 &i1, VT3 &i2, // (OUT) cube-sphere intersection points
                                           int &np) // (OUT) number of cube points contained in sphere
{
  VT3 ldiff = p2 - p1;
  T ldist_2 = dot(ldiff, ldiff);
  T lD      = p1.x*p2.y - p2.x*p1.y;
  T discrim = ldist_2*cr*cr - lD*lD;
  if(discrim <= 0) { np = 0; return false; } // no overlap
}


template<typename T, typename VT3=typename cuda_vec<T, 3>::VT>
__device__ inline T penOverlap3(VT3 &pCell, VT3 &mpos, VT3 &diff, VT3 &dist_2, const Pen<T> *pen, const FieldParams<T> &cp, T radOffset)
{
  const VT3 rMult = pen->sizeMult*pen->xyzMult;
  const T   cellDiag = cp.u.dL * (T)(sqrt(3.0)/2.0);
  
  if(pen->cellAlign) { pCell = floor(pCell); mpos = floor(mpos); }
  VT3 diff0   = pCell-(mpos + rMult*pen->rDist/2);
  VT3 dist0_2 = diff0*diff0;
  VT3 diff1   = pCell-(mpos - rMult*pen->rDist/2);
  VT3 dist1_2 = diff1*diff1;
  
  diff   = (diff0   + diff1)   / 2.0f; // set return values
  dist_2 = (dist0_2 + dist1_2) / 2.0f;

  VT3 r0 = rMult*pen->radius0 + radOffset;
  VT3 r1 = rMult*pen->radius1 + radOffset;
  
  VT3 rMax0 = r0 + cellDiag; VT3 rMax1 = r1 + cellDiag; // sphere radius plus maximum possible intersection radius from cell (center to corner)
  if(dist0_2 <= rMax0*rMax0)
    {
      T amount = 1.0;
      if(pen->square)
        {
          // rMax0 += 0.5; rMax1 += 0.5; // extra offset to adjust pixel center (?)
          VT3 dv0  = VT3{abs(diff0.x)-rMax0.x, abs(diff0.y)-rMax0.y, abs(diff0.z)-rMax0.z};
          if (dv0.x > 0 || dv0.y > 0 || dv0.z > 0) { return 0.0; } // too far
          else if(min(dv0) <= 1.0)                 { return 1.0; } // completely inside
          else                                     { dv0 /= rMax0; amount *= max((T)0, (dv0.x*dv0.y*dv0.z)); } // *cp.u.dL*cp.u.dL*cp.u.dL/rMax0; }
          
          if(r1.x > 0 && r1.y > 0 && r1.z > 0)
            { // check inside other sphere (intersection)
              VT3 dv1 = VT3{abs(diff1.x)-rMax1.x, abs(diff1.y)-rMax1.y, abs(diff1.z)-rMax1.z} / rMax1;
              if (dv1.x > 0 || dv1.y > 0 || dv1.z > 0) { return 0.0; } // too far
              else if(min(dv1) <= 1.0)                 { return 1.0; } // completely inside
              else                                     { dv1 /= rMax1; amount *= max((T)0, (dv1.x*dv1.y*dv1.z)); } // *cp.u.dL*cp.u.dL*cp.u.dL/rMax1; }
            }
        }
      else
        {
          T   ellipse0  = dist0_2.x/(rMax0.x*rMax0.x) + dist0_2.y/(rMax0.y*rMax0.y) + dist0_2.z/(rMax0.z*rMax0.z); // > 1.0 if too far (ellipsoid)
          VT3 dv0       = rMax0 - abs(diff0);
          if (ellipse0 >= 1)                            { return 0.0; } // too far
          else if(dot(dv0, dv0) <= cellDiag*cellDiag*4) { return 1.0; } // completely inside
          else                                          { amount *= 1.0 - ellipse0; }

          if(r1.x > 0 && r1.y > 0 && r1.z > 0)
            { // check inside other sphere (intersection)
              T   ellipse1 = dist1_2.x/(rMax1.x*rMax1.x) + dist1_2.y/(rMax1.y*rMax1.y) + dist1_2.z/(rMax1.z*rMax1.z); // > 1.0 if too far (ellipsoid)
              VT3 dv1      = rMax1 - abs(diff1);
              
              if (ellipse1 > 1)                             { return 0.0; } // too far
              else if(dot(dv1, dv1) <= cellDiag*cellDiag*4) { return 1.0; } // completely inside
              else                                          { amount *= 1.0 - ellipse1; }
            }
        }
      return amount;
    }
  else { return 0.0; }
}

template<typename T, typename VT3=typename cuda_vec<T, 3>::VT>
__device__ inline T penOverlap3(VT3 &pCell,  VT3 &mpos, VT3 &diff, VT3 &diff0, VT3 &diff1, VT3 &dist_2, VT3 &dist0_2, VT3 &dist1_2,
                                const Pen<T> *pen, const FieldParams<T> &cp, T radOffset)
{
  const VT3 rMult = pen->sizeMult*pen->xyzMult;
  const T   cellDiag = cp.u.dL * (T)(sqrt(3.0)/2.0);

  if(pen->cellAlign) { pCell = floor(pCell); mpos = floor(mpos); }
  diff0  = pCell-(mpos + rMult*pen->rDist/2);
  dist0_2 = diff0*diff0;
  diff1  = pCell-(mpos - rMult*pen->rDist/2);
  dist1_2 = diff1*diff1;
  diff   = (diff0   + diff1)   / 2.0f; // set return values
  dist_2 = (dist0_2 + dist1_2) / 2.0f;
  
  VT3 r0 = rMult*pen->radius0 + radOffset;
  VT3 r1 = rMult*pen->radius1 + radOffset;
  VT3 rMax0 = r0 + cellDiag; VT3 rMax1 = r1 + cellDiag; // sphere radius plus maximum possible intersection radius from cell (center to corner)
  if(dist0_2 <= rMax0*rMax0)
    {
      T amount = 1.0;
      if(pen->square)
        {
          // rMax0 += 0.5; rMax1 += 0.5; // extra offset to adjust pixel center (?)
          VT3 dv0  = VT3{abs(diff0.x)-rMax0.x, abs(diff0.y)-rMax0.y, abs(diff0.z)-rMax0.z};
          if (dv0.x > 0 || dv0.y > 0 || dv0.z > 0) { return 0.0; } // too far
          else if(min(dv0/rMax0) <= 1.0/sqrt(3.0)) { return 1.0; } // completely inside
          else                                     { dv0 /= rMax0; amount *= max((T)0, (dv0.x*dv0.y*dv0.z)); } // *cp.u.dL*cp.u.dL*cp.u.dL/rMax0; }
          
          if(r1.x > 0 && r1.y > 0 && r1.z > 0)
            { // check inside other sphere (intersection)
              VT3 dv1 = VT3{abs(diff1.x)-rMax1.x, abs(diff1.y)-rMax1.y, abs(diff1.z)-rMax1.z} / rMax1;
              if (dv1.x > 0 || dv1.y > 0 || dv1.z > 0) { return 0.0; } // too far
              else if(min(dv1/rMax1) <= 1.0/sqrt(3.0)) { return 1.0; } // completely inside
              else                                     { dv1 /= rMax1; amount *= max((T)0, (dv1.x*dv1.y*dv1.z)); } // *cp.u.dL*cp.u.dL*cp.u.dL/rMax1; }
            }
        }
      else
        {
          T   ellipse0  = dist0_2.x/(rMax0.x*rMax0.x) + dist0_2.y/(rMax0.y*rMax0.y) + dist0_2.z/(rMax0.z*rMax0.z); // > 1.0 if too far (ellipsoid)
          VT3 dv0       = rMax0 - abs(diff0);
          if (ellipse0 >= 1)                            { return 0.0; } // too far
          else if(dot(dv0, dv0) <= cellDiag*cellDiag*4) { return 1.0; } // completely inside
          else                                          { amount *= 1.0 - ellipse0; }

          if(r1.x > 0 && r1.y > 0 && r1.z > 0)
            { // check inside other sphere (intersection)
              T   ellipse1 = dist1_2.x/(rMax1.x*rMax1.x) + dist1_2.y/(rMax1.y*rMax1.y) + dist1_2.z/(rMax1.z*rMax1.z); // > 1.0 if too far (ellipsoid)
              VT3 dv1      = rMax1 - abs(diff1);
              
              if (ellipse1 > 1)                             { return 0.0; } // too far
              else if(dot(dv1, dv1) <= cellDiag*cellDiag*4) { return 1.0; } // completely inside
              else                                          { amount *= 1.0 - ellipse1; }
            }
        }
      return amount;
    }
  else { return 0.0; }
      
}

// add signal from source field
template<typename T> void addSignal(Field<T> &signal, Field<T> &dst, const FieldParams<T> &cp, T mult=1.0);
template<typename T> void addSignal(Field<T> &signal, Field<T> &dst, const FluidParams<T> &cp, T mult=1.0);
template<typename T, typename VT3=typename cuda_vec<T, 3>::VT>
void addSignal(Field<VT3> &signal, Field<VT3> &dst, const FieldParams<T> &cp, T mult=1.0);
template<typename T, typename VT3=typename cuda_vec<T, 3>::VT>
void addSignal(Field<VT3> &signal, Field<VT3> &dst, const FluidParams<T> &cp, T mult=1.0);
template<typename T>
void addSignal(EMField<T> &signal, EMField<T> &dst, const FieldParams<T> &cp, T mult=1.0);
template<typename T>
void addSignal(FluidField<T> &signal, FluidField<T> &dst, const FluidParams<T> &cp, T mult=1.0);

// add signal from mouse position/pen
template<typename T, typename VT3=typename cuda_vec<T, 3>::VT>
void addSignal(const VT3 &mpos, Field<VT3> &dstV, Field<T> &dstP, Field<T> &dstQn, Field<T> &dstQp,
               Field<VT3> &dstQnv, Field<VT3> &dstQpv, Field<VT3> &dstE, Field<VT3> &dstB,
               const SignalPen<T> &pen, const FluidParams<T> &cp, T mult=1.0);
template<typename T, typename VT3=typename cuda_vec<T, 3>::VT>
void addSignal(const VT3 &mpos, EMField<T> &dst, const SignalPen<T> &pen, const FieldParams<T> &cp, T mult=1.0); // EMField
template<typename T, typename VT3=typename cuda_vec<T, 3>::VT>
void addSignal(const VT3 &mpos, FluidField<T> &dst, const SignalPen<T> &pen, const FluidParams<T> &cp, T mult=1.0); // FluidField

// signal decay
template<typename T>
void decaySignal(Field<T> &src, FieldParams<T> &cp);
template<typename T, typename VT3=typename cuda_vec<T, 3>::VT>
void decaySignal(Field<VT3> &src, FieldParams<T> &cp);

// add material from mouse position/pen
template<typename T, typename VT3=typename cuda_vec<T, 3>::VT>
void addMaterial(const VT3 &mpos, EMField<T> &dst, const MaterialPen<T> &pen, const FieldParams<T> &cp);




#endif // DRAW_CUH
