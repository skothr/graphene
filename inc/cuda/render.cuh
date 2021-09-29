#ifndef RENDER_CUH
#define RENDER_CUH

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include "tools.hpp"
#include "field.cuh"
#include "cuda-texture.cuh"
#include "vector-operators.h"
#include "cuda-tools.h"
#include "physics.h"
#include "raytrace.h"
#include "material.h"
#include "units.hpp"


enum RenderFlags : long long
  {
   FLUID_RENDER_NONE  = 0,
   
   // fluid velocity
   FLUID_RENDER_VX    = (1LL<<0), //  abs(v.x)
   FLUID_RENDER_VY    = (1LL<<1), //  abs(v.y)
   FLUID_RENDER_VZ    = (1LL<<2), //  abs(v.z)
   FLUID_RENDER_V     = (1LL<<3), // |v|
   FLUID_RENDER_PVX   = (1LL<<4), //  max(0, v.x)
   FLUID_RENDER_PVY   = (1LL<<5), //  max(0, v.y)
   FLUID_RENDER_PVZ   = (1LL<<6), //  max(0, v.z)
   FLUID_RENDER_NVX   = (1LL<<7), // -min(0, v.x)
   FLUID_RENDER_NVY   = (1LL<<8), // -min(0, v.y)
   FLUID_RENDER_NVZ   = (1LL<<9), // -min(0, v.z)
   //   pressure/divergence   
   FLUID_RENDER_P     = (1LL<<10), // p
   FLUID_RENDER_D     = (1LL<<11), // div
   
   // charge density
   FLUID_RENDER_QN    = (1LL<<12), // Qn (-)
   FLUID_RENDER_QP    = (1LL<<13), // Qp (+)
   FLUID_RENDER_Q     = (1LL<<14), // Qp-Qn
   // charge velocity
   FLUID_RENDER_QVX   = (1LL<<15), //  abs(Qv.x)
   FLUID_RENDER_QVY   = (1LL<<16), //  abs(Qv.y)
   FLUID_RENDER_QVZ   = (1LL<<17), //  abs(Qv.z)
   FLUID_RENDER_QV    = (1LL<<18), // |Qv|
   FLUID_RENDER_QPVX  = (1LL<<19), //  max(0, Qv.x)
   FLUID_RENDER_QPVY  = (1LL<<20), //  max(0, Qv.y)
   FLUID_RENDER_QPVZ  = (1LL<<21), //  max(0, Qv.z)
   FLUID_RENDER_QNVX  = (1LL<<22), // -min(0, Qv.x)
   FLUID_RENDER_QNVY  = (1LL<<23), // -min(0, Qv.y)
   FLUID_RENDER_QNVZ  = (1LL<<24), // -min(0, Qv.z)
   // E / B
   FLUID_RENDER_EX    = (1LL<<25), //  abs(E.x)
   FLUID_RENDER_EY    = (1LL<<26), //  abs(E.x)
   FLUID_RENDER_EZ    = (1LL<<27), //  abs(E.x)
   FLUID_RENDER_E     = (1LL<<28), // |E|
   FLUID_RENDER_BX    = (1LL<<29), //  abs(E.x)
   FLUID_RENDER_BY    = (1LL<<30), //  abs(E.x)
   FLUID_RENDER_BZ    = (1LL<<31), //  abs(E.x)
   FLUID_RENDER_B     = (1LL<<32), // |B|

   // materials
   FLUID_RENDER_EP    = (1LL<<33), //  abs(mat.e)
   FLUID_RENDER_MU    = (1LL<<34), //  abs(mat.m)
   FLUID_RENDER_SIG   = (1LL<<35), //  abs(mat.s)
   FLUID_RENDER_N     = (1LL<<36), //  index of refraction

   FLUID_RENDER_FOFFSET   = FLUID_RENDER_VX,
   FLUID_RENDER_EMOFFSET  = FLUID_RENDER_QN,   
   FLUID_RENDER_MATOFFSET = FLUID_RENDER_EP,
  };
#define RENDER_FLAG_COUNT 37
ENUM_FLAG_OPERATORS_LL(RenderFlags)

inline constexpr const char* renderFlagName(RenderFlags f)
{
  if     (f & FLUID_RENDER_VX  ) { return "Vx";  }
  else if(f & FLUID_RENDER_VY  ) { return "Vy";  }
  else if(f & FLUID_RENDER_VZ  ) { return "Vz";  }
  else if(f & FLUID_RENDER_V   ) { return "V";   }
  else if(f & FLUID_RENDER_PVX ) { return "+Vx"; }
  else if(f & FLUID_RENDER_PVY ) { return "+Vy"; }
  else if(f & FLUID_RENDER_PVZ ) { return "+Vz"; }
  else if(f & FLUID_RENDER_NVX ) { return "-Vx"; }
  else if(f & FLUID_RENDER_NVY ) { return "-Vy"; }
  else if(f & FLUID_RENDER_NVZ ) { return "-Vz"; }
 
  else if(f & FLUID_RENDER_P   ) { return "P";  }
  else if(f & FLUID_RENDER_D   ) { return "D";  }
 
  else if(f & FLUID_RENDER_QN  ) { return "Q-"; }
  else if(f & FLUID_RENDER_QP  ) { return "Q+"; }
  else if(f & FLUID_RENDER_Q   ) { return "Q";  }
 
  else if(f & FLUID_RENDER_QVX ) { return "QVx";  }
  else if(f & FLUID_RENDER_QVY ) { return "QVy";  }
  else if(f & FLUID_RENDER_QVZ ) { return "QVz";  }
  else if(f & FLUID_RENDER_QV  ) { return "QV";   }
  else if(f & FLUID_RENDER_QPVX) { return "+QVx"; }
  else if(f & FLUID_RENDER_QPVY) { return "+QVy"; }
  else if(f & FLUID_RENDER_QPVZ) { return "+QVz"; }
  else if(f & FLUID_RENDER_QNVX) { return "-QVx"; }
  else if(f & FLUID_RENDER_QNVY) { return "-QVy"; }
  else if(f & FLUID_RENDER_QNVZ) { return "-QVz"; }
 
  else if(f & FLUID_RENDER_EX  ) { return "Ex"; }
  else if(f & FLUID_RENDER_EY  ) { return "Ey"; }
  else if(f & FLUID_RENDER_EZ  ) { return "Ez"; }
  else if(f & FLUID_RENDER_E   ) { return "E"; }
  else if(f & FLUID_RENDER_BX  ) { return "Bx"; }
  else if(f & FLUID_RENDER_BY  ) { return "By"; }
  else if(f & FLUID_RENDER_BZ  ) { return "Bz"; }
  else if(f & FLUID_RENDER_B   ) { return "B"; }
 
  else if(f & FLUID_RENDER_EP  ) { return "ε"; }
  else if(f & FLUID_RENDER_MU  ) { return "μ"; }
  else if(f & FLUID_RENDER_SIG ) { return "σ"; }
  else if(f & FLUID_RENDER_N   ) { return "n"; }
  else                           { return "<INVALID>"; }
}

//// RENDER PARAMS
template<typename T>
struct RenderParams
{
  typedef typename DimType<T,3>::VEC_T VT3;
  typedef typename DimType<T,4>::VEC_T VT4;

  bool simple = true; // if true only handles main components for faster rendering -- v, Q, E, B, mat
  static const RenderFlags MAIN = (FLUID_RENDER_V  |
                                   FLUID_RENDER_QN | FLUID_RENDER_QP | FLUID_RENDER_Q | 
                                   FLUID_RENDER_E  | FLUID_RENDER_B  |
                                   FLUID_RENDER_EP | FLUID_RENDER_MU | FLUID_RENDER_SIG);
  
  VT4  rColors[RENDER_FLAG_COUNT]  = { VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, // Vxyz, V
                                       VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0},                          // PV
                                       VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0},                          // NV
                                       VT4{0.0, 1.0, 0.0, 1.0}, VT4{0.0, 0.0, 1.0, 1.0},                                                   // P, D
                                       VT4{1.0, 0.0, 1.0, 1.0}, VT4{1.0, 1.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0},                          // Q
                                       VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, // QVxyz, QV
                                       VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0},                          // QPV
                                       VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0},                          // QNV
                                       VT4{0.0, 1.0, 0.0, 1.0}, VT4{0.0, 1.0, 0.0, 1.0}, VT4{0.0, 1.0, 0.0, 1.0}, VT4{0.0, 1.0, 0.0, 1.0}, // Exyz, E
                                       VT4{0.0, 0.0, 1.0, 1.0}, VT4{0.0, 0.0, 1.0, 1.0}, VT4{0.0, 0.0, 1.0, 1.0}, VT4{0.0, 0.0, 1.0, 1.0}, // Bxyz, B
                                       VT4{1.0, 0.0, 0.0, 1.0}, VT4{0.0, 1.0, 0.0, 1.0}, VT4{0.0, 0.0, 0.0, 1.0}, VT4{0.0, 0.0, 1.0, 1.0}, // mat, mat.n()
  };
  T    rMults[RENDER_FLAG_COUNT]   = { 1.0,   1.0,  1.0,  1.0, // Vxyz, V
                                       1.0,   1.0,  1.0,       // PV
                                       1.0,   1.0,  1.0,       // NV
                                       1.0,   1.0,             // P, D
                                      10.0,  10.0,  1.0,       // Q
                                       1.0,   1.0,  1.0,  1.0, // QVxyz, QV
                                       1.0,   1.0,  1.0,       // QPV
                                       1.0,   1.0,  1.0,       // QNV
                                       1.0,   1.0,  1.0,  1.0, // Exyz, E
                                       1.0,   1.0,  1.0,  1.0, // Bxyz, B
                                       1.0,   1.0,  1.0,  1.0, // mat, mat.n()
  };
  bool rToggles[RENDER_FLAG_COUNT] = { false, false, false, false, // Vxyz, V
                                       false, false, false,        // PV
                                       false, false, false,        // NV
                                       false, false,               // P, D
                                       true,  true,  false,        // Q
                                       false, false, false, false, // QVxyz, QV
                                       false, false, false,        // QPV
                                       false, false, false,        // QNV
                                       false, false, false, true,  // Exyz, E
                                       false, false, false, true,  // Bxyz, B
                                       true, true, true, false,    // mat, mat.n()
  };
  
  // pen outlines, if active
  SignalPen<T>   sigPen;
  MaterialPen<T> matPen;
  bool matOutline      = true;  // outline materials (inner border)
  bool sigPenHighlight = false; // if true, highlight sigPen area of effect
  bool matPenHighlight = false; // if true, highlight matPen area of effect
  VT3  penPos;              // pen location
  
  int2 zRange = int2{0, 0}; // blends layers from highest to lowest (front to back / top down)
  T fOpacity      = 0.03;
  T fBrightness   = 1.00;
  T emOpacity     = 0.03;
  T emBrightness  = 1.00;
  T matOpacity    = 0.03;
  T matBrightness = 1.00;
  bool surfaces = true; // if true, quits marching if any color component >= 1 (interesting, brings out surface features)

  typedef long long IT;
  const VT4*  getColor (RenderFlags f) const { IT i = (IT)log2((float)f); return (i < 0 ? nullptr : &rColors[i]);  }
  const T*    getMult  (RenderFlags f) const { IT i = (IT)log2((float)f); return (i < 0 ? nullptr : &rMults[i]);   }
  const bool* getToggle(RenderFlags f) const { IT i = (IT)log2((float)f); return (i < 0 ? nullptr : &rToggles[i]); }
  VT4*  getColor (RenderFlags f)             { IT i = (IT)log2((float)f); return (i < 0 ? nullptr : &rColors[i]);  }
  T*    getMult  (RenderFlags f)             { IT i = (IT)log2((float)f); return (i < 0 ? nullptr : &rMults[i]);   }
  bool* getToggle(RenderFlags f)             { IT i = (IT)log2((float)f); return (i < 0 ? nullptr : &rToggles[i]); }
  __host__ __device__ VT4 getFinalColor(RenderFlags f) const
  {
    const IT i = (IT)log2((float)f);
    return ((i < 0 || !rToggles[i]) ? VT4{0.0, 0.0, 0.0, 0.0} : rMults[i]*rColors[i]);
  }
};

// forward declarations
typedef void* ImTextureID;

// field rendering
template<typename T> void renderFieldEM (FluidField<T>      &src, CudaTexture &dst, const RenderParams<T> &rp, const FluidParams<T> &cp);
template<typename T> void renderFieldMat(Field<Material<T>> &src, CudaTexture &dst, const RenderParams<T> &rp, const FluidParams<T> &cp);
// ray marching for 3D
template<typename T> void raytraceFieldEM (FluidField<T> &src, CudaTexture &dst, const Camera<T> &camera,
                                           const RenderParams<T> &rp, const FluidParams<T> &cp, const Vector<T, 2> &aspect);
template<typename T> void raytraceFieldMat(FluidField<T> &src, CudaTexture &dst, const Camera<T> &camera,
                                           const RenderParams<T> &rp, const FluidParams<T> &cp, const Vector<T, 2> &aspect);

#endif // RENDER_CUH
