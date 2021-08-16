#ifndef DRAW_HPP
#define DRAW_HPP


#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include "vector-operators.h"
#include "raytrace.h"
#include "material.h"
#include "units.hpp"

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
struct SignalPen
{
  using VT2 = typename DimType<T, 2>::VECTOR_T;
  using VT3 = typename DimType<T, 3>::VECTOR_T;

  bool active    = true;
  bool square    = false; // square pen
  bool cellAlign = false; // align to cells
  T    radius    = 10.0;  // pen size in fluid cells
  T    mult      = 1.0;   // signal multiplier
  T    frequency = 0.2;   // Hz(1/t in sim time) for sin/cos mult flags

  
  VT2 Qmult   = VT2{0.0, 0.0};       // base field values to add
  VT3 QPVmult = VT3{0.0, 0.0,  0.0};
  VT3 QNVmult = VT3{0.0, 0.0,  0.0};
  VT3 Emult   = VT3{0.0, 0.0,  1.0};
  VT3 Bmult   = VT3{0.0, 0.0, -1.0};
  
  int Qopt; int QPVopt; int QNVopt;  // parameteric modification options
  int Eopt; int Bopt;
  
  SignalPen() : Qopt(IDX_NONE), QPVopt(IDX_NONE), QNVopt(IDX_NONE), Eopt(IDX_SIN), Bopt(IDX_COS) { }
};

template<typename T>
struct MaterialPen
{
  bool active       = true;
  bool square       = false;
  bool vacuum       = false; // (eraser)
  T    radius       = 10.0;  // pen in fluid cells
  T    mult         = 1.0;   // multiplier
  T    permittivity = 1.0;   // vacuum permittivity (E)
  T    permeability = 1.0;   // vacuum permeability (B)
  T    conductivity = 0.0;   // vacuum conductivity (Q)
};








// forward declarations
template<typename T> struct EMField;
template<typename T> struct FieldParams;


// add signal from source field
template<typename T> void addSignal  (EMField<T> &signal, EMField<T> &dst, const FieldParams<T> &cp);
// add signal from mouse position/pen
template<typename T> void addSignal  (const typename DimType<T, 3>::VECTOR_T &pSrc, EMField<T> &dst, const SignalPen<T>   &pen, const FieldParams<T> &cp);
template<typename T> void addMaterial(const typename DimType<T, 3>::VECTOR_T &pSrc, EMField<T> &dst, const MaterialPen<T> &pen, const FieldParams<T> &cp);

#endif // DRAW_HPP
