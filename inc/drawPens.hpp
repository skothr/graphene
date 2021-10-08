#ifndef DRAW_PENS_H
#define DRAW_PENS_H

#include "vector-operators.h"
#include "raytrace.h"
#include "material.h"
#include "drawPens.hpp"

#include <array>
#include <vector>


// defines a "pen" used to add to a fiel 
template<typename T>
struct Pen
{
  using VT3 = typename DimType<T, 3>::VEC_T;
  bool active    = true;  // pen always active
  bool cellAlign = false; // snap offset to center of cell
  bool square    = false; // draw with square pen
  bool radial    = false; // multiply by normal (from pen center)
  bool speed     = true;  // scale by mouse speed
  
  int depth      = 0;                     // depth of pen center from view surface
  VT3 radius0    = VT3{10.0, 10.0, 10.0}; // base pen size in fluid cells
  VT3 radius1    = VT3{ 0.0,  0.0,  0.0}; // if x,y,z > 0, pen shape will be the intersection of spheres 0/1
  VT3 rDist      = VT3{ 0.0,  0.0,  0.0}; // positional difference between intersecting spheres
  T   mult       = 1.0;                   // multiplier (amplitude if signal)
  T   sizeMult   = 1.0;                   // multiplier (pen size)
  VT3 xyzMult    = VT3{1.0, 1.0, 1.0};    // multiplier (pen size, each dimension)
  T   speedMult  = 1.0;                   // multiplier for mouse speed

  T startTime  = -1.0; // time of initial mouse click (< 0 if inactive)
  T mouseSpeed =  0.0; // current mouse speed

  virtual ~Pen() = default;
};


// derived -- signal
template<typename T>
struct SigFieldParams
{
  T base;
  union
  {
    struct
    {
      bool multR;   // multiplies signal by 1/r
      bool multR_2; // multiplies signal by 1/r^2
      bool multT;   // multiplies signal by theta
      // bool multT_2; // multiplies signal by theta^2
      bool multSin; // multiplies signal by sin(2*pi*f*t) [t => simTime, f => frequency]
      bool multCos; // multiplies signal by cos(2*pi*f*t) [t => simTime, f => frequency]
    };
    bool mods[5] {false, false, false, false, false};
#ifndef __NVCC__
    std::array<bool, 5> modArr;
#endif // __NVCC__
  };
  SigFieldParams(const T &m) : base(m) { }
};

// for drawing signal in with mouse
template<typename T>
struct SignalPen : public Pen<T>
{
  using VT3 = typename DimType<T, 3>::VEC_T;
  T wavelength = 12.0;           // in dL units (cells per period)
  T frequency  = 1.0/wavelength; // Hz(c/t in sim time) for sin/cos mult flags (c --> speed in vacuum = vMat.c())
  SigFieldParams<VT3> pV;
  SigFieldParams<T>   pP;
  SigFieldParams<T>   pQn;
  SigFieldParams<T>   pQp;
  SigFieldParams<VT3> pQv;
  SigFieldParams<VT3> pE;
  SigFieldParams<VT3> pB;
  SignalPen() // TODO: set defaults somewhere
    : pV(VT3{0,0,0}), pP(T(0)), pQn(T(0)), pQp(T(0)),
      pQv(VT3{0,0,0}), pE(VT3{10,10,10}), pB(VT3{0,0,0}) { this->mult = 20.0f; }
  ~SignalPen() = default;
  // old non-zero defaults:
  // VT3 Emult = VT3{10.0, 10.0, 10.0}; // X/Y/Z vector multipliers
  // int Eopt  = IDX_SIN; int Bopt = IDX_SIN;
};

template<typename T>
struct MaterialPen : public Pen<T>
{
  Material<T> mat = Material<T>(2.4, 2.0, 0.0001, false);
  MaterialPen()  = default;
  ~MaterialPen() = default;
};


#endif // DRAW_PENS_H
