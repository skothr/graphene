#ifndef RENDER_CUH
#define RENDER_CUH

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include "vector-operators.h"
#include "cuda-tools.h"
#include "tools.hpp"
#include "units.cuh"
#include "field.cuh"
#include "cuda-texture.cuh"
#include "camera.hpp"
#include "physics.h"
#include "raytrace.h"
#include "material.h"

// forward declarations
template<typename T> class Camera;



enum RenderFlags : long long
  {
   FLUID_RENDER_NONE  = 0,

   // fluid velocity
   FLUID_RENDER_V     = (1LL<<0),  // ||v||
   FLUID_RENDER_VX    = (1LL<<1),  // v.x
   FLUID_RENDER_VY    = (1LL<<2),  // v.y
   FLUID_RENDER_VZ    = (1LL<<3),  // v.z
   FLUID_RENDER_VXMAG = (1LL<<4),  // abs(v.x)
   FLUID_RENDER_VYMAG = (1LL<<5),  // abs(v.y)
   FLUID_RENDER_VZMAG = (1LL<<6),  // abs(v.z)
   FLUID_RENDER_VXP   = (1LL<<7),  // max( v.x, 0)
   FLUID_RENDER_VYP   = (1LL<<8),  // max( v.y, 0)
   FLUID_RENDER_VZP   = (1LL<<9), // max( v.z, 0)
   FLUID_RENDER_VXN   = (1LL<<10), // max(-v.x, 0)
   FLUID_RENDER_VYN   = (1LL<<11), // max(-v.y, 0)
   FLUID_RENDER_VZN   = (1LL<<12), // max(-v.z, 0)
   // fluid pressure
   FLUID_RENDER_P     = (1LL<<13), // p
   FLUID_RENDER_PMAG  = (1LL<<14), // ||p||
   FLUID_RENDER_PP    = (1LL<<15), // p+
   FLUID_RENDER_PN    = (1LL<<16), // p-

   // charge density
   FLUID_RENDER_Q     = (1LL<<17), // (Q+)-(Q-)
   FLUID_RENDER_QMAG  = (1LL<<18), // |Q|
   FLUID_RENDER_QP    = (1LL<<19), // Q+
   FLUID_RENDER_QN    = (1LL<<20), // Q-
   // charge velocity
   FLUID_RENDER_QV    = (1LL<<21), // ||Qv||
   FLUID_RENDER_QVX   = (1LL<<22), // Qv.x
   FLUID_RENDER_QVY   = (1LL<<23), // Qv.y
   FLUID_RENDER_QVZ   = (1LL<<24), // Qv.z
   FLUID_RENDER_QVXMAG= (1LL<<25), // abs(Qv.x)
   FLUID_RENDER_QVYMAG= (1LL<<26), // abs(Qv.y)
   FLUID_RENDER_QVZMAG= (1LL<<27), // abs(Qv.z)
   FLUID_RENDER_QVXP  = (1LL<<28), // max( Qv.x, 0)
   FLUID_RENDER_QVYP  = (1LL<<29), // max( Qv.y, 0)
   FLUID_RENDER_QVZP  = (1LL<<30), // max( Qv.z, 0)
   FLUID_RENDER_QVXN  = (1LL<<31), // max(-Qv.x, 0)
   FLUID_RENDER_QVYN  = (1LL<<32), // max(-Qv.y, 0)
   FLUID_RENDER_QVZN  = (1LL<<33), // max(-Qv.z, 0)
   // E / B
   FLUID_RENDER_E     = (1LL<<34), // ||E||
   FLUID_RENDER_EX    = (1LL<<35), // E.x
   FLUID_RENDER_EY    = (1LL<<36), // E.y
   FLUID_RENDER_EZ    = (1LL<<37), // E.z
   FLUID_RENDER_EXMAG = (1LL<<38), // abs(E.x)
   FLUID_RENDER_EYMAG = (1LL<<39), // abs(E.y)
   FLUID_RENDER_EZMAG = (1LL<<40), // abs(E.z)
   FLUID_RENDER_EXP   = (1LL<<41), // max( E.x, 0)
   FLUID_RENDER_EYP   = (1LL<<42), // max( E.y, 0)
   FLUID_RENDER_EZP   = (1LL<<43), // max( E.z, 0)
   FLUID_RENDER_EXN   = (1LL<<44), // max(-E.x, 0)
   FLUID_RENDER_EYN   = (1LL<<45), // max(-E.y, 0)
   FLUID_RENDER_EZN   = (1LL<<46), // max(-E.z, 0)
   FLUID_RENDER_B     = (1LL<<47), // ||B||
   FLUID_RENDER_BX    = (1LL<<48), // B.x
   FLUID_RENDER_BY    = (1LL<<49), // B.y
   FLUID_RENDER_BZ    = (1LL<<50), // B.z
   FLUID_RENDER_BXMAG = (1LL<<51), // abs(B.x)
   FLUID_RENDER_BYMAG = (1LL<<52), // abs(B.y)
   FLUID_RENDER_BZMAG = (1LL<<53), // abs(B.z)
   FLUID_RENDER_BXP   = (1LL<<54), // max( B.x, 0)
   FLUID_RENDER_BYP   = (1LL<<55), // max( B.y, 0)
   FLUID_RENDER_BZP   = (1LL<<56), // max( B.z, 0)
   FLUID_RENDER_BXN   = (1LL<<57), // max(-B.x, 0)
   FLUID_RENDER_BYN   = (1LL<<58), // max(-B.y, 0)
   FLUID_RENDER_BZN   = (1LL<<59), // max(-B.z, 0)

   // materials
   FLUID_RENDER_EP    = (1LL<<60), //  abs(mat.e)
   FLUID_RENDER_MU    = (1LL<<61), //  abs(mat.m)
   FLUID_RENDER_SIG   = (1LL<<62), //  abs(mat.s)
   FLUID_RENDER_N     = (1LL<<63), //  index of refraction
   
   FLUID_RENDER_FOFFSET   = FLUID_RENDER_V,
   FLUID_RENDER_EMOFFSET  = FLUID_RENDER_Q,
   FLUID_RENDER_MATOFFSET = FLUID_RENDER_EP,
  };
#define RENDER_FLAG_COUNT 64
ENUM_FLAG_OPERATORS_LL(RenderFlags)


inline constexpr const char* renderFlagName(RenderFlags f)
{
  if     (f & FLUID_RENDER_V     ) { return "||V||"; }
  else if(f & FLUID_RENDER_VX    ) { return  "Vx";   }
  else if(f & FLUID_RENDER_VY    ) { return  "Vy";   }
  else if(f & FLUID_RENDER_VZ    ) { return  "Vz";   }
  else if(f & FLUID_RENDER_VXMAG ) { return "|Vx|";  }
  else if(f & FLUID_RENDER_VYMAG ) { return "|Vy|";  }
  else if(f & FLUID_RENDER_VZMAG ) { return "|Vz|";  }
  else if(f & FLUID_RENDER_VXP   ) { return "+Vx";   }
  else if(f & FLUID_RENDER_VYP   ) { return "+Vy";   }
  else if(f & FLUID_RENDER_VZP   ) { return "+Vz";   }
  else if(f & FLUID_RENDER_VXN   ) { return "-Vx";   }
  else if(f & FLUID_RENDER_VYN   ) { return "-Vy";   }
  else if(f & FLUID_RENDER_VZN   ) { return "-Vz";   }

  else if(f & FLUID_RENDER_P     ) { return  "P";    }
  else if(f & FLUID_RENDER_PMAG  ) { return "|P|";   }
  else if(f & FLUID_RENDER_PP    ) { return  "P+";   }
  else if(f & FLUID_RENDER_PN    ) { return  "P-";   }

  else if(f & FLUID_RENDER_Q     ) { return  "Q";    }
  else if(f & FLUID_RENDER_QMAG  ) { return "|Q|";   }
  else if(f & FLUID_RENDER_QP    ) { return  "Q+";   }
  else if(f & FLUID_RENDER_QN    ) { return  "Q-";   }

  else if(f & FLUID_RENDER_QV    ) { return "||QV||";}
  else if(f & FLUID_RENDER_QVX   ) { return  "QVx";  }
  else if(f & FLUID_RENDER_QVY   ) { return  "QVy";  }
  else if(f & FLUID_RENDER_QVZ   ) { return  "QVz";  }
  else if(f & FLUID_RENDER_QVXMAG) { return "|QVx|"; }
  else if(f & FLUID_RENDER_QVYMAG) { return "|QVy|"; }
  else if(f & FLUID_RENDER_QVZMAG) { return "|QVz|"; }
  else if(f & FLUID_RENDER_QVXP  ) { return "+QVx";  }
  else if(f & FLUID_RENDER_QVYP  ) { return "+QVy";  }
  else if(f & FLUID_RENDER_QVZP  ) { return "+QVz";  }
  else if(f & FLUID_RENDER_QVXN  ) { return "-QVx";  }
  else if(f & FLUID_RENDER_QVYN  ) { return "-QVy";  }
  else if(f & FLUID_RENDER_QVZN  ) { return "-QVz";  }

  else if(f & FLUID_RENDER_E     ) { return "||E||"; }
  else if(f & FLUID_RENDER_EX    ) { return  "Ex";   }
  else if(f & FLUID_RENDER_EY    ) { return  "Ey";   }
  else if(f & FLUID_RENDER_EZ    ) { return  "Ez";   }
  else if(f & FLUID_RENDER_EXMAG ) { return "|Ex|";  }
  else if(f & FLUID_RENDER_EYMAG ) { return "|Ey|";  }
  else if(f & FLUID_RENDER_EZMAG ) { return "|Ez|";  }
  else if(f & FLUID_RENDER_EXP   ) { return "+Ex";   }
  else if(f & FLUID_RENDER_EYP   ) { return "+Ey";   }
  else if(f & FLUID_RENDER_EZP   ) { return "+Ez";   }
  else if(f & FLUID_RENDER_EXN   ) { return "-Ex";   }
  else if(f & FLUID_RENDER_EYN   ) { return "-Ey";   }
  else if(f & FLUID_RENDER_EZN   ) { return "-Ez";   }

  else if(f & FLUID_RENDER_B     ) { return "||B||"; }
  else if(f & FLUID_RENDER_BX    ) { return  "Bx";   }
  else if(f & FLUID_RENDER_BY    ) { return  "By";   }
  else if(f & FLUID_RENDER_BZ    ) { return  "Bz";   }
  else if(f & FLUID_RENDER_BXMAG ) { return "|Bx|";  }
  else if(f & FLUID_RENDER_BYMAG ) { return "|By|";  }
  else if(f & FLUID_RENDER_BZMAG ) { return "|Bz|";  }
  else if(f & FLUID_RENDER_BXP   ) { return "+Bx";   }
  else if(f & FLUID_RENDER_BYP   ) { return "+By";   }
  else if(f & FLUID_RENDER_BZP   ) { return "+Bz";   }
  else if(f & FLUID_RENDER_BXN   ) { return "-Bx";   }
  else if(f & FLUID_RENDER_BYN   ) { return "-By";   }
  else if(f & FLUID_RENDER_BZN   ) { return "-Bz";   }

  else if(f & FLUID_RENDER_EP    ) { return "ε";     }
  else if(f & FLUID_RENDER_MU    ) { return "μ";     }
  else if(f & FLUID_RENDER_SIG   ) { return "σ";     }
  else if(f & FLUID_RENDER_N     ) { return "n";     }
  else                             { return "<INVALID>"; }
}


inline constexpr const char* renderFlagGroupName(RenderFlags f)
{
  if     (f & FLUID_RENDER_V     ) { return "Velocity"; }
  else if(f & FLUID_RENDER_P     ) { return "Pressure"; }
  else if(f & FLUID_RENDER_Q     ) { return "Charge";   }
  else if(f & FLUID_RENDER_QV    ) { return "Charge Velocity"; }
  else if(f & FLUID_RENDER_E     ) { return "Electric Field";  }
  else if(f & FLUID_RENDER_B     ) { return "Magnetic Field";  }
  else                             { return "<INVALID>"; }
}

//// RENDER PARAMS
template<typename T>
struct RenderParams
{
  typedef typename DimType<T,3>::VEC_T VT3;
  typedef typename DimType<T,4>::VEC_T VT4;

  bool simple = true; // if true only handles main components for faster rendering -- v, p, Q, E, B, mat
  static const RenderFlags MAIN = (FLUID_RENDER_V  | FLUID_RENDER_P  |
                                   FLUID_RENDER_QN | FLUID_RENDER_QP | FLUID_RENDER_Q | FLUID_RENDER_QV |
                                   FLUID_RENDER_E  | FLUID_RENDER_B  |
                                   FLUID_RENDER_EP | FLUID_RENDER_MU | FLUID_RENDER_SIG);

  VT4  rColors[RENDER_FLAG_COUNT]  = { VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, // V,VX,VY,VZ
                                       VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0},                          // VMAG
                                       VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0},                          // VP
                                       VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0},                          // VN
                                       VT4{0.0, 1.0, 0.0, 1.0}, VT4{0.0, 1.0, 0.0, 1.0}, VT4{0.0, 1.0, 0.0, 1.0}, VT4{0.0, 1.0, 0.0, 1.0}, // P,PMAG,PP,PN
                                       VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 1.0, 0.0, 1.0}, VT4{1.0, 0.0, 1.0, 1.0}, // Q,QMAG,QP,QN
                                       VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, // QV,QVX,QVY,QVZ
                                       VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0},                          // QVMAG
                                       VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0},                          // QPV
                                       VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0}, VT4{1.0, 0.0, 0.0, 1.0},                          // QNV
                                       VT4{0.0, 1.0, 0.0, 1.0}, VT4{0.0, 1.0, 0.0, 1.0}, VT4{0.0, 1.0, 0.0, 1.0}, VT4{0.0, 1.0, 0.0, 1.0}, // E,EX,EY,EZ
                                       VT4{0.0, 1.0, 0.0, 1.0}, VT4{0.0, 1.0, 0.0, 1.0}, VT4{0.0, 1.0, 0.0, 1.0},                          // EMAG
                                       VT4{0.0, 1.0, 0.0, 1.0}, VT4{0.0, 1.0, 0.0, 1.0}, VT4{0.0, 1.0, 0.0, 1.0},                          // EP
                                       VT4{0.0, 1.0, 0.0, 1.0}, VT4{0.0, 1.0, 0.0, 1.0}, VT4{0.0, 1.0, 0.0, 1.0},                          // EN
                                       VT4{0.0, 0.0, 1.0, 1.0}, VT4{0.0, 0.0, 1.0, 1.0}, VT4{0.0, 0.0, 1.0, 1.0}, VT4{0.0, 0.0, 1.0, 1.0}, // B,BX,BY,BZ
                                       VT4{0.0, 0.0, 1.0, 1.0}, VT4{0.0, 0.0, 1.0, 1.0}, VT4{0.0, 0.0, 1.0, 1.0},                          // BMAG
                                       VT4{0.0, 0.0, 1.0, 1.0}, VT4{0.0, 0.0, 1.0, 1.0}, VT4{0.0, 0.0, 1.0, 1.0},                          // BP
                                       VT4{0.0, 0.0, 1.0, 1.0}, VT4{0.0, 0.0, 1.0, 1.0}, VT4{0.0, 0.0, 1.0, 1.0},                          // BN
                                       VT4{1.0, 0.0, 0.0, 1.0}, VT4{0.0, 1.0, 0.0, 1.0}, VT4{0.0, 0.0, 0.0, 1.0}, VT4{0.0, 0.0, 1.0, 1.0}, // ε,μ,σ,mat.n()
  };
  T    rMults[RENDER_FLAG_COUNT]   = { 1.0,   1.0,  1.0,   1.0, // V,VX,VY,VZ
                                       1.0,   1.0,  1.0,        // VMAG
                                       1.0,   1.0,  1.0,        // VP
                                       1.0,   1.0,  1.0,        // VN
                                       1.0,   1.0,  1.0,   1.0, // P,PMAG,PP,PN
                                       1.0,   1.0,  10.0, 10.0, // Q,QMAG,QP,QN
                                       1.0,   1.0,  1.0,   1.0, // QV,QVX,QVY,QVZ
                                       1.0,   1.0,  1.0,        // QVMAG
                                       1.0,   1.0,  1.0,        // QVP
                                       1.0,   1.0,  1.0,        // QVN
                                       1.0,   1.0,  1.0,   1.0, // E,EX,EY,EZ
                                       1.0,   1.0,  1.0,        // EMAG
                                       1.0,   1.0,  1.0,        // EP
                                       1.0,   1.0,  1.0,        // EN
                                       1.0,   1.0,  1.0,   1.0, // B,BX,BY,BZ
                                       1.0,   1.0,  1.0,        // BMAG
                                       1.0,   1.0,  1.0,        // BP
                                       1.0,   1.0,  1.0,        // BN
                                       1.0,   1.0,  1.0,   1.0  // ε,μ,σ,mat.n()
  };
  bool rToggles[RENDER_FLAG_COUNT] = { false, false, false, false, // V,VX,VY,VZ
                                       false, false, false,        // VMAG
                                       false, false, false,        // VP
                                       false, false, false,        // VN
                                       false, false, false, false, // P,PMAG,PP,PN
                                       false, false, true,  true,  // Q,QMAG,QP,QN
                                       false, false, false, false, // QV,QVX,QVY,QVZ
                                       false, false, false,        // QMAG
                                       false, false, false,        // QVP
                                       false, false, false,        // QVN
                                       true,  false, false, false, // E,EX,EY,EZ
                                       false, false, false,        // EMAG
                                       false, false, false,        // EP
                                       false, false, false,        // EN
                                       true,  false, false, false, // B,BX,BY,BZ
                                       false, false, false,        // BMAG
                                       false, false, false,        // BP
                                       false, false, false,        // BN
                                       true,  true,  false, false, // ε,μ,σ,mat.n()
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

  typedef unsigned long long IT;
  const VT4*  getColor (RenderFlags f) const { IT i = log2((double)f); return (i >= RENDER_FLAG_COUNT ? nullptr : &rColors[i]);  }
  const T*    getMult  (RenderFlags f) const { IT i = log2((double)f); return (i >= RENDER_FLAG_COUNT ? nullptr : &rMults[i]);   }
  const bool* getToggle(RenderFlags f) const { IT i = log2((double)f); return (i >= RENDER_FLAG_COUNT ? nullptr : &rToggles[i]); }
  VT4*  getColor (RenderFlags f)             { IT i = log2((double)f); return (i >= RENDER_FLAG_COUNT ? nullptr : &rColors[i]);  }
  T*    getMult  (RenderFlags f)             { IT i = log2((double)f); return (i >= RENDER_FLAG_COUNT ? nullptr : &rMults[i]);   }
  bool* getToggle(RenderFlags f)             { IT i = log2((double)f); return (i >= RENDER_FLAG_COUNT ? nullptr : &rToggles[i]); }
  __host__ __device__ VT4 getFinalColor(RenderFlags f) const
  {
    const IT i = (IT)log2((float)f);
    return ((i >= RENDER_FLAG_COUNT || !rToggles[i]) ? VT4{0.0, 0.0, 0.0, 0.0} : rMults[i]*rColors[i]);
  }
};




// render a single cell with simple main color options (for speed)
template<typename T>
__device__ typename DimType<T, 4>::VEC_T renderCellSimple(FluidField<T> &src, unsigned long i, const RenderParams<T> &rp, const FluidParams<T> &cp)
{
  typedef typename DimType<T,4>::VEC_T VT4;

  VT4 color = rp.fBrightness*rp.fOpacity*(length(src.v[i])      * rp.getFinalColor(FLUID_RENDER_V)   +
                                          src.p[i]              * rp.getFinalColor(FLUID_RENDER_P));
  color  += rp.emBrightness*rp.emOpacity*(src.Qn[i]             * rp.getFinalColor(FLUID_RENDER_QN)  +
                                          src.Qp[i]             * rp.getFinalColor(FLUID_RENDER_QP)  +
                                          (src.Qp[i]-src.Qn[i]) * rp.getFinalColor(FLUID_RENDER_Q)   +
                                          length(src.Qpv[i]-src.Qnv[i]) * rp.getFinalColor(FLUID_RENDER_QV)  +
                                          length(src.E[i])      * rp.getFinalColor(FLUID_RENDER_E)   +
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

  const VT3 v   = src.v[i];
  const T   p   = src.p[i];
  const T   Qn  = src.Qn[i];
  const T   Qp  = src.Qp[i];
  const VT3 Qv  = src.Qpv[i]+src.Qnv[i];
  const VT3 E   = src.E[i];
  const VT3 B   = src.B[i];

  const T vLen  = length(v);
  const T QvLen = length(Qv);
  const T ELen  = length(src.E[i]);
  const T BLen  = length(src.B[i]);

  const VT3 vp  = max( v,  0.0f);
  const VT3 vn  = max(-v,  0.0f);
  const T   pp  = max( p,  0.0f);
  const T   pn  = max(-p,  0.0f);
  const VT3 Qvp = max( Qv, 0.0f);
  const VT3 Qvn = max(-Qv, 0.0f);
  const T   Q   = (Qp - Qn);
  const VT3 Ep  = VT3{max( E.x, 0.0f), max( E.y, 0.0f), max( E.z, 0.0f)};
  const VT3 En  = VT3{max(-E.x, 0.0f), max(-E.y, 0.0f), max(-E.z, 0.0f)};
  const VT3 Bp  = VT3{max( B.x, 0.0f), max( B.y, 0.0f), max( B.z, 0.0f)};
  const VT3 Bn  = VT3{max(-B.x, 0.0f), max(-B.y, 0.0f), max(-B.z, 0.0f)};

  VT4 color = rp.fBrightness*rp.fOpacity*(vLen     * rp.getFinalColor(FLUID_RENDER_V)       +
                                          v.x      * rp.getFinalColor(FLUID_RENDER_VX)      +
                                          v.y      * rp.getFinalColor(FLUID_RENDER_VY)      +
                                          v.z      * rp.getFinalColor(FLUID_RENDER_VZ)      +
                                          abs(v.x) * rp.getFinalColor(FLUID_RENDER_VXMAG)   +
                                          abs(v.y) * rp.getFinalColor(FLUID_RENDER_VYMAG)   +
                                          abs(v.z) * rp.getFinalColor(FLUID_RENDER_VZMAG)   +
                                          vp.x     * rp.getFinalColor(FLUID_RENDER_VXP)     +
                                          vp.y     * rp.getFinalColor(FLUID_RENDER_VYP)     +
                                          vp.z     * rp.getFinalColor(FLUID_RENDER_VZP)     +
                                          vn.x     * rp.getFinalColor(FLUID_RENDER_VXN)     +
                                          vn.y     * rp.getFinalColor(FLUID_RENDER_VYN)     +
                                          vn.z     * rp.getFinalColor(FLUID_RENDER_VZN)     +

                                          p        * rp.getFinalColor(FLUID_RENDER_P)       +
                                          abs(p)   * rp.getFinalColor(FLUID_RENDER_PMAG)    +
                                          pp       * rp.getFinalColor(FLUID_RENDER_PP)      +
                                          pn       * rp.getFinalColor(FLUID_RENDER_PN) );

  color += rp.emBrightness*rp.emOpacity*( Q        * rp.getFinalColor(FLUID_RENDER_Q)       +
                                          abs(Q)   * rp.getFinalColor(FLUID_RENDER_QMAG)    +
                                          Qp       * rp.getFinalColor(FLUID_RENDER_QP)      +
                                          Qn       * rp.getFinalColor(FLUID_RENDER_QN)      +

                                          QvLen     * rp.getFinalColor(FLUID_RENDER_QV)     +
                                          Qv.x      * rp.getFinalColor(FLUID_RENDER_QVX)    +
                                          Qv.y      * rp.getFinalColor(FLUID_RENDER_QVY)    +
                                          Qv.z      * rp.getFinalColor(FLUID_RENDER_QVZ)    +
                                          abs(Qv.x) * rp.getFinalColor(FLUID_RENDER_QVXMAG) +
                                          abs(Qv.y) * rp.getFinalColor(FLUID_RENDER_QVYMAG) +
                                          abs(Qv.z) * rp.getFinalColor(FLUID_RENDER_QVZMAG) +
                                          Qvp.x     * rp.getFinalColor(FLUID_RENDER_QVXP)   +
                                          Qvp.y     * rp.getFinalColor(FLUID_RENDER_QVYP)   +
                                          Qvp.z     * rp.getFinalColor(FLUID_RENDER_QVZP)   +
                                          Qvn.x     * rp.getFinalColor(FLUID_RENDER_QVXN)   +
                                          Qvn.y     * rp.getFinalColor(FLUID_RENDER_QVYN)   +
                                          Qvn.z     * rp.getFinalColor(FLUID_RENDER_QVZN)   +

                                          ELen      * rp.getFinalColor(FLUID_RENDER_E)      +
                                          E.x       * rp.getFinalColor(FLUID_RENDER_EX)     +
                                          E.y       * rp.getFinalColor(FLUID_RENDER_EY)     +
                                          E.z       * rp.getFinalColor(FLUID_RENDER_EZ)     +
                                          abs(E.x)  * rp.getFinalColor(FLUID_RENDER_EXMAG)  +
                                          abs(E.y)  * rp.getFinalColor(FLUID_RENDER_EYMAG)  +
                                          abs(E.z)  * rp.getFinalColor(FLUID_RENDER_EZMAG)  +
                                          Ep.x      * rp.getFinalColor(FLUID_RENDER_EXP)    +
                                          Ep.y      * rp.getFinalColor(FLUID_RENDER_EYP)    +
                                          Ep.z      * rp.getFinalColor(FLUID_RENDER_EZP)    +
                                          En.x      * rp.getFinalColor(FLUID_RENDER_EXN)    +
                                          En.y      * rp.getFinalColor(FLUID_RENDER_EYN)    +
                                          En.z      * rp.getFinalColor(FLUID_RENDER_EZN)    +
                                          BLen      * rp.getFinalColor(FLUID_RENDER_B)      +
                                          B.x       * rp.getFinalColor(FLUID_RENDER_BX)     +
                                          B.y       * rp.getFinalColor(FLUID_RENDER_BY)     +
                                          B.z       * rp.getFinalColor(FLUID_RENDER_BZ)     +
                                          abs(B.x)  * rp.getFinalColor(FLUID_RENDER_BXMAG)  +
                                          abs(B.y)  * rp.getFinalColor(FLUID_RENDER_BYMAG)  +
                                          abs(B.z)  * rp.getFinalColor(FLUID_RENDER_BZMAG)  +
                                          Bp.x      * rp.getFinalColor(FLUID_RENDER_BXP)    +
                                          Bp.y      * rp.getFinalColor(FLUID_RENDER_BYP)    +
                                          Bp.z      * rp.getFinalColor(FLUID_RENDER_BZP)    +
                                          Bn.x      * rp.getFinalColor(FLUID_RENDER_BXN)    +
                                          Bn.y      * rp.getFinalColor(FLUID_RENDER_BYN)    +
                                          Bn.z      * rp.getFinalColor(FLUID_RENDER_BZN) );
  auto M = src.mat[i];
  if(!M.vacuum())
    {
      color += rp.matBrightness*rp.matOpacity*(M.ep      * rp.getFinalColor(FLUID_RENDER_EP)  +
                                               M.mu      * rp.getFinalColor(FLUID_RENDER_MU)  +
                                               M.sig     * rp.getFinalColor(FLUID_RENDER_SIG) +
                                               M.n(cp.u) * rp.getFinalColor(FLUID_RENDER_N) );
    }
  return color;
}


// blend fluid colors from one cell to the next (along ray)
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





// field rendering
template<typename T> void renderFieldEM (FluidField<T>      &src, CudaTexture &dst, const RenderParams<T> &rp, const FluidParams<T> &cp);
template<typename T> void renderFieldMat(Field<Material<T>> &src, CudaTexture &dst, const RenderParams<T> &rp, const FluidParams<T> &cp);
// ray marching for 3D
template<typename T> void raytraceFieldEM (FluidField<T> &src, CudaTexture &dst, const Camera<T> &camera, const RenderParams<T> &rp, const FluidParams<T> &cp);
template<typename T> void raytraceFieldMat(FluidField<T> &src, CudaTexture &dst, const Camera<T> &camera, const RenderParams<T> &rp, const FluidParams<T> &cp);

#endif // RENDER_CUH
