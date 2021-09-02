#ifndef RENDER_CUH
#define RENDER_CUH

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include "field.cuh"
#include "vector-operators.h"
#include "cuda-tools.h"
#include "physics.h"
#include "raytrace.h"
#include "material.h"
#include "units.hpp"


//// RENDER PARAMS
template<typename T>
struct RenderParams
{
  typedef typename DimType<T,2>::VECTOR_T VT2;
  typedef typename DimType<T,3>::VECTOR_T VT3;
  typedef typename DimType<T,4>::VECTOR_T VT4;
  // base colors for field components
  VT4 Qcol  = VT4{1.0, 0.0, 0.0, 1.0}; // R
  VT4 Ecol  = VT4{0.0, 1.0, 0.0, 1.0}; // G
  VT4 Bcol  = VT4{0.0, 0.0, 1.0, 1.0}; // B
  T  Qmult  = 0.2; T Emult   = 0.2; T Bmult   = 0.2; // additional multipliers
  // base colors for materials
  VT4 epCol   = VT4{1.0, 0.0, 0.0, 1.0}; // R
  VT4 muCol   = VT4{0.0, 1.0, 0.0, 1.0}; // G
  VT4 sigCol  = VT4{0.0, 0.0, 1.0, 1.0}; // B
  T  epMult = 0.2; T muMult  = 0.2; T sigMult = 0.2; // additional multipliers

  // pen outlines, if active
  SignalPen<T>   sigPen;
  MaterialPen<T> matPen;
  bool matOutline = true;   // outline materials (inner border)
  bool sigPenHighlight = false; // if true, highlight sigPen area of effect
  bool matPenHighlight = false; // if true, highlight matPen area of effect
  VT3  penPos;              // pen location
  
  int2 zRange = int2{0, 0}; // blends layers from highest to lowest (front to back / top down)
  T opacity    = 0.05;
  T brightness = 2.0;
  bool surfaces = true; // if true, quits marching if any color component >= 1 (interesting, brings out surface features)
};


// forward declarations
typedef void* ImTextureID;

// field rendering
template<typename T> void renderFieldEM (EMField<T> &src,         CudaTexture &dst, const RenderParams<T> &rp, const FieldParams<T> &cp);
template<typename T> void renderFieldMat(Field<Material<T>> &src, CudaTexture &dst, const RenderParams<T> &rp, const FieldParams<T> &cp);
// ray marching for 3D
template<typename T> void raytraceFieldEM (EMField<T> &src, CudaTexture &dst, const Camera<T> &camera,
                                           const RenderParams<T> &rp, const FieldParams<T> &cp, const Vector<T, 2> &aspect);
template<typename T> void raytraceFieldMat(EMField<T> &src, CudaTexture &dst, const Camera<T> &camera,
                                           const RenderParams<T> &rp, const FieldParams<T> &cp, const Vector<T, 2> &aspect);

#endif // RENDER_CUH
