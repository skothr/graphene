#ifndef MATERIAL_H
#define MATERIAL_H

#include <cuda.h>
#include <iostream>

// ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσΤτΥυΦφΧχΨψΩω

// forward declarations
template<typename T> struct Units;

//// EM MATERIAL PROPERTIES ////
template<typename T>
struct Material
{
  bool nonVacuum = false; // if false, use user-defined vacuum properties in ChargeParams instead (defined this way so memset(material, 0) clears it to vacuum)
  union { T permittivity = 1.0f; T epsilon; T ep;  }; // ε -- electric permittivity (E)
  union { T permeability = 1.0f;            T mu;  }; // μ -- magnetic permeability (B)
  union { T conductivity = 0.0f; T sigma;   T sig; }; // σ -- material conductivity (Q?)

  Material() = default;
  __host__ __device__ Material(T permit, T permeab, T conduct, bool vacuum_=true) // *NOTE*: logic reversed within class (see member declaration)
    : permittivity(permit), permeability(permeab), conductivity(conduct), nonVacuum(!vacuum_) { }
  __host__ __device__ Material(const Material &other)
    : permittivity(other.ep), permeability(other.mu), conductivity(other.sig),
      nonVacuum(other.nonVacuum && (other.ep*other.mu > 0.0f)) { }
  // __host__ __device__ ~Material() = default;

  __host__ __device__ bool vacuum() const      { return !nonVacuum; } // *NOTE*: logic reversed within class (see bool nonVacuum)
  __host__            void setVacuum(bool vac) { nonVacuum = !vac;  }

  __host__ __device__ T c(const Units<T> &u) const  { return 1 / (T)sqrt(u.e0*u.u0); } // 1 / sqrt(ε₀μ₀)        speed of light within this material
  __host__ __device__ T n(const Units<T> &u) const  { return sqrt((ep*mu))*c(u);     } // sqrt(εμ) / sqrt(ε₀μ₀) index of refraction given vacuum parameters
  
  struct Blend { T alpha; T beta; };

  // NOTE: part of Yee's method
  //   --> E(t) = alphaE*E(t-1) + betaE*dE/dt
  __host__ __device__ Blend getBlendE(T dt, T dL) const
  { // E1 = ((1 - dt*(σ/(2μ))) / (1 + dt*(σ/(2μ))))*E0 + dt*(dL / (μ + dt*(σ/2)))*dE/dt
    T C = dt*sigma / (2.0*mu);
    T D = 1 / (1+C);
    return Blend{ D*(1-C), (dt/dL)*(D/mu) };
  }
  
  //   --> B(t) = alphaB*B(t-1) + betaB*dB/dt  
  __host__ __device__ Blend getBlendB(T dt, T dL) const
  { // B1 = ((1 - dt[σ/(2ε)]) / (1 + dt[σ/(2ε)]))*E0 + dt*(dL / (ε + dt[σ/2]))*dE/dt
    T C = dt*sigma / (2.0*epsilon);
    T D = 1 / (1+C);
    return Blend{ D*(1-C), (dt/dL)*(D/epsilon) };
  }


  __host__ __device__ Material<T>& operator+=(const T &v) { ep += v; mu += v; return *this; }
  __host__ __device__ Material<T>& operator-=(const T &v) { ep -= v; mu -= v; return *this; }
  __host__ __device__ Material<T>& operator*=(const T &v) { ep *= v; mu *= v; return *this; }
  __host__ __device__ Material<T>& operator/=(const T &v) { ep /= v; mu /= v; return *this; }
  
  __host__ __device__ Material<T> operator+(const T &v) const { Material<T> result(*this); result += v; return result; }
  __host__ __device__ Material<T> operator-(const T &v) const { Material<T> result(*this); result -= v; return result; }
  __host__ __device__ Material<T> operator*(const T &v) const { Material<T> result(*this); result *= v; return result; }
  __host__ __device__ Material<T> operator/(const T &v) const { Material<T> result(*this); result /= v; return result; }

  
};
template<typename T> inline std::ostream& operator<<(std::ostream &os, const Material<T> &mat)
{
  os << "Material" << (mat.vacuum() ? " VACUUM(" : "") << "<ε=" << mat.epsilon <<  "|μ=" << mat.mu << "|σ=" << mat.sigma << ">)";
  return os;
}

#endif // MATERIAL_H
