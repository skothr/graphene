#ifndef MATERIAL_H
#define MATERIAL_H

#include <cuda.h>
#include <iostream>

#ifndef __NVCC__

#endif // __NVCC__

// #include "units.hpp"


// ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσΤτΥυΦφΧχΨψΩω

// forward declarations
template<typename T> struct Units;

//// EM MATERIAL PROPERTIES ////
template<typename T>
struct Material
{
  bool nonVacuum = false; // if false, use user-defined vacuum properties in ChargeParams instead (defined this way so memset(material, 0) clears it to vacuum) 
  union { T permittivity = 1.0f; T epsilon; T ep;  }; // ε  epsilon -- electric permittivity (E)
  union { T permeability = 1.0f; T mu;             }; // μ  mu      -- magnetic permeability (B)
  union { T conductivity = 0.0f; T sigma;   T sig; }; // σ  sigma   -- material conductivity (Q) (?)

  Material() = default;
  __host__ __device__ Material(T permit, T permeab, T conduct, bool vacuum_=true) // *NOTE*: logic reversed within class (see member declaration)
    : permittivity(permit), permeability(permeab), conductivity(conduct), nonVacuum(!vacuum_) { }

  __host__ __device__ bool vacuum() const      { return !nonVacuum; } // *NOTE*: logic reversed within class (see nonVacuum member declaration)
  __host__            void setVacuum(bool vac) { nonVacuum = !vac;  } // *NOTE*: logic reversed within class (see nonVacuum member declaration)

  __host__ __device__ T c(const Units<T> &u) const { return T(1)/(T)sqrt(u.e0*u.u0); } // 1 / sqrt(ε₀μ₀)        speed of light within this material
  __host__ __device__ T n(const Units<T> &u) const { return sqrt((ep*mu)*c(u));      } // sqrt(εμ) / sqrt(ε₀μ₀) index of refraction given vacuum parameters
  
  struct Blend { T alpha; T beta; };

  // NOTE: part of Yee's method
  //   --> E(t) = alphaE*E(t-1) + betaE*dE/dt
  __host__ __device__ Blend getBlendE(T dt, T dL) const
  { // E1 = ((1 - dt*(σ/(2μ))) / (1 + dt*(σ/(2μ))))*E0 + dt*(dL / (μ + dt*(σ/2)))*dE/dt
    
    T cE = dt * sigma / (2.0*mu);   T dE = T(1) / (T(1) + cE);
    
    T alphaE = dE * (T(1) - cE);
    T betaE  = (dt/dL) * (dE/mu);
    return Blend{ alphaE, betaE };
  }
  
  //   --> B(t) = alphaB*B(t-1) + betaB*dB/dt  
  __host__ __device__ Blend getBlendB(T dt, T dL) const
  { // B1 = ((1 - dt[σ/(2ε)]) / (1 + dt[σ/(2ε)]))*E0 + dt*(dL / (ε + dt[σ/2]))*dE/dt
    T cB     = dt * conductivity / (2.0*permittivity);
    T dB     = T(1) / (T(1) + cB);
    T alphaB = dB * (T(1) - cB);
    T betaB  = dB * dt / (dL*permittivity);
    return Blend{ alphaB, betaB };
  }
};
template<typename T> inline std::ostream& operator<<(std::ostream &os, const Material<T> &mat)
{
  os << "Material" << (mat.vacuum() ? " VACUUM(" : "") << "<ε=" << mat.epsilon <<  "|μ=" << mat.mu << "|σ=" << mat.sigma << ">)";
  return os;
}

#endif // MATERIAL_H
