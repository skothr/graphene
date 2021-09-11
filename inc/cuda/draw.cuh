#ifndef DRAW_CUH
#define DRAW_CUH

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include "vector-operators.h"
#include "raytrace.h"
#include "material.h"
#include "units.hpp"


enum PenType { PEN_NONE, PEN_SIGNAL, PEN_MATERIAL};

template<typename T>
struct Pen
{
  using VT3 = typename DimType<T, 3>::VECTOR_T;
  bool active     = true;
  bool cellAlign  = false; // snap offset to center of cell
  bool square     = false; // draw with square pen
  int  depth      = 0;     // depth of pen center from view surface
  VT3  radius0    = VT3{20.0, 20.0, 20.0};  // base pen size in fluid cells
  VT3  radius1    = VT3{ 0.0,  0.0,  0.0};  // if x,y,z > 0, pen shape will be the intersection of circles radius0 and radius1
  VT3  rDist      = VT3{ 0.0,  0.0,  0.0};  // positional difference between intersecting circles
  T    mult       = 1.0;                // multiplier (signal/material)
  T    sizeMult   = 1.0;                // multiplier (pen size)
  VT3  xyzMult    = VT3{1.0, 1.0, 1.0}; // multiplier (pen size, each dimension)
  virtual PenType type() const { return PEN_NONE; }
};

enum // bit flags for applying multipliers to field values
  {
   IDX_NONE = 0x00,
   IDX_R    = 0x01, // scale signal by 1/lenth(r)   at each point
   IDX_R2   = 0x02, // scale signal by 1/length(r)^2 at each point
   IDX_T    = 0x04, // scale signal by theta   at each point
   IDX_SIN  = 0x08, // scale signal by sin(2*pi*t*frequency) at each point
   IDX_COS  = 0x10  // scale signal by cos(2*pi*t*frequency) at each point
  };
// for drawing signal in with mouse
template<typename T>
struct SignalPen : public Pen<T>
{
  virtual PenType type() const override { return PEN_SIGNAL; }
  using VT2 = typename DimType<T, 2>::VECTOR_T;
  using VT3 = typename DimType<T, 3>::VECTOR_T;
  
  T frequency  = 0.1;  // Hz(1/t in sim time) for sin/cos mult flags
  T wavelength = 10.0; // in dL units (cells per period)
  
  // base field values to add
  T   QPmult = 0;
  T   QNmult = 0;
  VT3 QVmult = VT3{0.0,  0.0,  0.0};
  VT3 Emult  = VT3{0.0,  0.0,  1.0};
  VT3 Bmult  = VT3{0.0,  0.0, -1.0};
  // parameteric option flags (IDX_*)
  int Qopt; int QVopt; int Eopt; int Bopt;
  
  SignalPen() : Qopt(IDX_NONE), QVopt(IDX_NONE), Eopt(IDX_SIN), Bopt(IDX_COS) { this->mult = 10.0; }
  
};

template<typename T>
struct MaterialPen : public Pen<T>
{
  virtual PenType type() const override { return PEN_MATERIAL; }
  
  Material<T> material;
  MaterialPen() : material(2.4, 1.3, 0.001, false) { }
};




//// CUDA ////
// forward declarations
template<typename T> struct EMField;
template<typename T> struct FieldParams;

// add signal from source field
template<typename T> void addSignal  (EMField<T> &signal, EMField<T> &dst, const FieldParams<T> &cp);
// add signal from mouse position/pen
template<typename T> void addSignal  (const typename DimType<T, 3>::VECTOR_T &pSrc, EMField<T> &dst, const SignalPen<T>   &pen, const FieldParams<T> &cp);
template<typename T> void addMaterial(const typename DimType<T, 3>::VECTOR_T &pSrc, EMField<T> &dst, const MaterialPen<T> &pen, const FieldParams<T> &cp);


template<typename T>
__device__ inline bool penOverlaps(typename DimType<T, 3>::VECTOR_T &pCell, typename DimType<T, 3>::VECTOR_T &pSrc,
                                   typename DimType<T, 3>::VECTOR_T &diff, typename DimType<T, 3>::VECTOR_T &dist_2, const Pen<T> *pen,
                                   const FieldParams<T> &cp, T radOffset)
{
  typedef typename DimType<T, 3>::VECTOR_T VT3;
  
  if(pen->cellAlign) { pCell = floor(pCell); pSrc = floor(pSrc); }

  VT3 diff0  = ((pSrc + (pen->sizeMult*pen->xyzMult)*pen->rDist/T(2))-pCell);
  VT3 dist0_2 = diff0*diff0;
  VT3 diff1  = ((pSrc - (pen->sizeMult*pen->xyzMult)*pen->rDist/T(2))-pCell);
  VT3 dist1_2 = diff1*diff1;
  diff = diff0;
  dist_2 = dist0_2;

  VT3 r0 = pen->radius0 * (pen->sizeMult*pen->xyzMult) + radOffset;
  VT3 r1 = pen->radius1 * (pen->sizeMult*pen->xyzMult) + radOffset;
  return ((!pen->square &&                         (dist0_2.x/(r0.x*r0.x) + dist0_2.y/(r0.y*r0.y) + dist0_2.z/(r0.z*r0.z) <= 1.0f) &&
           (r1.x <= 0 || r1.y <= 0 || r1.z <= 0 || (dist1_2.x/(r1.x*r1.x) + dist1_2.y/(r1.y*r1.y) + dist1_2.z/(r1.z*r1.z) <= 1.0f))) ||
          ( pen->square &&                         (abs(diff0.x) <= r0.x+0.5f && abs(diff0.y) <= r0.y+0.5f && abs(diff0.z) <= r0.z+0.5f) &&
           (r1.x <= 0 || r1.y <= 0 || r1.z <= 0 || (abs(diff1.x) <= r1.x+0.5f && abs(diff1.y) <= r1.y+0.5f && abs(diff1.z) <= r1.z+0.5f))));
}
template<typename T>
__device__ inline bool penOverlaps2(typename DimType<T, 3>::VECTOR_T &pCell,  typename DimType<T, 3>::VECTOR_T &pSrc,
                                    typename DimType<T, 3>::VECTOR_T &diff,   typename DimType<T, 3>::VECTOR_T &diff0,   typename DimType<T, 3>::VECTOR_T &diff1,
                                    typename DimType<T, 3>::VECTOR_T &dist_2, typename DimType<T, 3>::VECTOR_T &dist0_2, typename DimType<T, 3>::VECTOR_T &dist1_2,
                                    const Pen<T> *pen, const FieldParams<T> &cp, T radOffset)
{
  typedef typename DimType<T, 3>::VECTOR_T VT3;
  
  if(pen->cellAlign) { pCell = floor(pCell); pSrc = floor(pSrc); }
  diff0  = ((pSrc + (pen->sizeMult*pen->xyzMult)*pen->rDist/T(2))-pCell);
  dist0_2 = diff0*diff0;
  diff1  = ((pSrc - (pen->sizeMult*pen->xyzMult)*pen->rDist/T(2))-pCell);
  dist1_2 = diff1*diff1;
  diff = diff0;
  dist_2 = dist0_2;
  
  VT3 r0 = pen->radius0 * (pen->sizeMult*pen->xyzMult) + radOffset;
  VT3 r1 = pen->radius1 * (pen->sizeMult*pen->xyzMult) + radOffset;
  return ((!pen->square &&                         (dist0_2.x/(r0.x*r0.x) + dist0_2.y/(r0.y*r0.y) + dist0_2.z/(r0.z*r0.z) <= 1.0f) &&
           (r1.x <= 0 || r1.y <= 0 || r1.z <= 0 || (dist1_2.x/(r1.x*r1.x) + dist1_2.y/(r1.y*r1.y) + dist1_2.z/(r1.z*r1.z) <= 1.0f))) ||
          ( pen->square &&                         (abs(diff0.x) <= r0.x+0.5f && abs(diff0.y) <= r0.y+0.5f && abs(diff0.z) <= r0.z+0.5f) &&
           (r1.x <= 0 || r1.y <= 0 || r1.z <= 0 || (abs(diff1.x) <= r1.x+0.5f && abs(diff1.y) <= r1.y+0.5f && abs(diff1.z) <= r1.z+0.5f))));
}

// returns true if point with given distances are at the edge of the pen (hollow shell)
template<typename T>
__device__ inline bool penBorder(typename DimType<T, 3>::VECTOR_T &diff0, typename DimType<T, 3>::VECTOR_T &dist1, const Pen<T> *pen, T width)
{ return (abs(pen->radius0 - diff0) <= width) || (pen->radius1 > 0.0 && abs(pen->radius1 - dist1) <= width); }

// returns true if point with given offsets are at the edge of each axis plane cross-section
template<typename T>
__device__ inline bool penAxisBorder(const typename DimType<T, 3>::VECTOR_T &pSrc, const typename DimType<T, 3>::VECTOR_T &diff0,
                                     const typename DimType<T, 3>::VECTOR_T &diff1, const Pen<T> *pen, T width)
{
  typedef typename DimType<T, 3>::VECTOR_T VT3;
  VT3 dist0XY = VT3{diff0.x, diff0.y, 0.0f};
  VT3 dist0YZ = VT3{diff0.x, 0.0f, diff0.z};
  VT3 dist0ZX = VT3{0.0f, diff0.y, diff0.z};
  VT3 dist1XY = VT3{diff1.x, diff1.y, 0.0f};
  VT3 dist1YZ = VT3{diff1.x, 0.0f, diff1.z};
  VT3 dist1ZX = VT3{0.0f, diff1.y, diff1.z};

  T w = width*min(pen->sizeMult*pen->xyzMult);
  return (penBorder(dist0XY, dist1XY, pen, w) || penBorder(dist0YZ, dist1YZ, pen, w) || penBorder(dist0ZX, dist1ZX, pen, w));
}

#endif // DRAW_CUH
