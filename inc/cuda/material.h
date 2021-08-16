#ifndef MATERIAL_H
#define MATERIAL_H


#include <cuda.h>
#include <iostream>


//// EM MATERIAL PROPERTIES ////
template<typename T>
struct Material
{
  bool nonVacuum = false; // if false, use user-defined vacuum properties in ChargeParams instead (defined this way so memset(material, 0) clears it to vacuum) 
  union { T permittivity = 1.0f; T epsilon; }; // epsilon -- electric permittivity (E)
  union { T permeability = 1.0f; T mu;      }; // mu      -- magnetic permeability (B)
  union { T conductivity = 0.0f; T sigma;   }; // sigma   -- material conductivity (Q)

  __host__ __device__ Material() { }
  __host__ __device__ Material(T permit, T permeab, T conduct, bool vacuum_=true)
    : permittivity(permit), permeability(permeab), conductivity(conduct), nonVacuum(!vacuum_) { }

  __host__ __device__ bool vacuum() const { return !nonVacuum; }
  
  // NOTE:
  //   E(t) = alphaE*E(t-1) + betaE*dE/dt
  //   B(t) = alphaB*B(t-1) + betaB*dB/dt
  struct Factors { T alphaE; T betaE; T alphaB; T betaB; };
  __host__ __device__ Factors getFactors(T dt, T cellSize) const
  {
    // E --  (1 / (1 + dt[s/(2mu)])) /
    T cE     = dt * conductivity/(2*permeability);
    T dE     = 1  / (1 + cE);
    T alphaE = dE * (1 - cE);
    T betaE  = dt/cellSize * dE/permeability;
    // B
    T cB     = dt * conductivity/(2*permittivity);
    T dB     = 1  / (1 + cB);
    T alphaB = dB * (1 - cB);
    T betaB  = dt/cellSize * dB/permittivity;
    return Factors{ alphaE, betaE, alphaB, betaB };
  }
};
template<typename T> inline std::ostream& operator<<(std::ostream &os, const Material<T> &mat)
{
  if(mat.vacuum()) { os << "Material<vacuum>"; }
  else             { os << "Material<ep=" << mat.permittivity <<  "|mu=" << mat.permeability << "|sig=" << mat.conductivity << ">"; }
  return os;
}

#endif // MATERIAL_H
