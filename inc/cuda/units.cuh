#ifndef UNITS_CUH
#define UNITS_CUH

#include "physics.h"
#include "material.h"

template<typename T>
struct Units
{
  // discretization
  T dt = 0.20; // TIME   (field update timestep)
  T dL = 1.0;  // LENGTH (field cell size)
  // NOTE: dL/dt > ~2 usually explodes
  
  // EM
  T e  = 1.0;       // elementary charge
  T a  = 1.0/137.0; // fine structure constant
  T e0 = 1.0;       // electric constant / permittivity of free space    (E)
  T u0 = 1.0;       // magnetic constant / permeability of free space    (B)
  T s0 = 0.0;       // conductivity of free space (may just be abstract) (Q?)

  __host__ __device__ Material<T> vacuum() const { return Material<T>(e0, u0, s0, true); }

  // derive
  __host__ __device__ T c() const { return 1/(T)sqrt(e0*u0); }     // speed of light in a vacuum
  __host__ __device__ T h() const { return u0*e*e*c() / (2.0*a); } // Planck's constant
};



#endif // UNITS_CUH
