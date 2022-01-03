#ifndef PHYSICS_H
#define PHYSICS_H

#include <cuda_runtime.h>
#include <cmath>
#include "vector-operators.h"


// unit conversions
#define ANGSTROM_PER_M 1e10 // Ao  -->  m     ( (Ao / ANGSTROM_PER_M) ==> m
#define NM_PER_M       1e9  // nm  -->  m     ( (nm / NM_PER_M)       ==> m
  
#define GRAPHENE_CARBON_DIST (1.42 * NM_PER_M)      // (m)
#define ELECTRON_ION_DIST    (4.0 * ANGSTROM_PER_M) // (m)


// UNITS //
// 
// J  ==> (kg*m^2) / s^2
// N  ==> (kg*m) / s&2
// 
// 
  
//--//-//--//-//--//-//--//-//=//
//.// FUNDAMENTAL CONSTANTS //.//
//--//-//--//-//--//-//--//-//=//
#define ELEMENTARY_CHARGE            1.602176634e-19      // | 1.6 | -19 |  [  C            ] (EXACT)
#define CONTANT_OF_GRAVIATION        6.6743015e-11        // | 6.6 | -11 |  [  m^3/(kg*s^2) ]
#define PLANCK_CONSTANT              6.62607015e-34       // | 6.6 | -34 |  [  J/Hz         ] (EXACT)
#define SPEED_OF_LIGHT               2.99792458e8         // | 2.9 |   8 |  [  m/s          ] (EXACT)
#define VACUUM_ELECTRIC_PERMITTIVITY 8.8541878128e-12     // | 8.8 | -12 |  [  F/m          ]
#define VACUUM_MAGNETIC_PERMEABILITY 1.25663706212e-6     // | 1.2 |  -6 |  [  N/(A^2)      ]
#define ELECTRON_MASS                9.1093837015e-31     // | 9.1 | -31 |  [  kg           ]
#define FINE_STRUCTURE_CONSTANT      7.2973525693e-3      // | 7.2 |  -3 |  [               ]
#define JOSEPHSON_CONSTANT           4.835978484e14       // | 4.8 | -14 |  [  Hz/V         ]
#define RYDBERG_CONSTANT             1.0973731568160e7    // | 1.0 |   7 |  [  1/m          ]
#define VON_KLITZING_CONSTANT        2.581280745e4        // | 2.5 |   4 |  [  Ω            ]
  
// abbreviated constants
#define M_e     ELEMENTARY_CHARGE
#define M_G     CONTANT_OF_GRAVIATION
#define M_h     PLANCK_CONSTANT
#define M_c     SPEED_OF_LIGHT
#define M_ep0   VACUUM_ELECTRIC_PERMITTIVITY
#define M_mu0   VACUUM_MAGNETIC_PERMEABILITY
#define M_me    ELECTRON_MASS
#define M_alpha FINE_STRUCTURE_CONSTANT
#define M_Kj    JOSEPHSON_CONSTANT
#define M_Rinf  RYDBERG_CONSTANT
#define M_Rk    VON_KLITZING_CONSTANT
// Derived:
#define M_Ke    1.0 / (4.0*M_PI*M_ep0)    // Coulomb's constant






//// UNIT CONVERSION

/////////////////////////////////////////////////////////////////////////////////////////////
// calculates current density J given charge density and velocity
//  ==> NOTE:   p = q / (volume of cell) (?)
//  qd --> (rho) charge density  (C/m^3)
//  v  --> velocity              (m/s)
/////////////////////////////////////////////////////////////////////////////////////////////
__host__ __device__ inline float3  currentDensity(float  qd, const float3  &v) { return v*qd; }
__host__ __device__ inline double3 currentDensity(double qd, const double3 &v) { return v*qd; }



//////////////////////
//// CALCULATIONS ////
//////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
// calculates Coulomb force on charge q0 given another charge q1 and vector n between them
//  q0 --> charge being affected                         (C <=> A*s)
//  q1 --> other charge                                  (C <=> A*s)
//  n  --> vector from other charge to q0 --> (p0 - p1)  (m)
/////////////////////////////////////////////////////////////////////////////////////////////
template<typename VT=float3, typename ST=float> __device__ VT coulombForce(ST q0, ST q1, const VT &n);

template<> __device__ inline float3  coulombForce<float3,  float> (float  q0, float  q1, const float3  &n)
{ float  dist = length(n); return (dist != 0.0f ? (((float)M_Ke)*(q0*q1*n) / (dist*dist*dist)) : float3{0.0f, 0.0f, 0.0f}); }

template<> __device__ inline double3 coulombForce<double3, double>(double q0, double q1, const double3 &n)
{ double dist = length(n); return (dist != 0.0f ? (M_Ke*(q0*q1*n) / (dist*dist*dist)) : double3{0.0f, 0.0f, 0.0f}); }

// (normalized units)
template<typename VT=float3, typename ST=float> __device__ VT coulombForceN(ST q0, ST q1, const VT &n);
template<> __device__ inline float3  coulombForceN<float3,  float> (float  q0, float  q1, const float3  &n)
{ float  dist = length(n); return (dist != 0.0f ? ((q0*q1*n) / (dist*dist*dist)) : float3{0.0f, 0.0f, 0.0f}); }

template<> __device__ inline double3 coulombForceN<double3, double>(double q0, double q1, const double3 &n)
{ double dist = length(n); return (dist != 0.0f ? ((q0*q1*n) / (dist*dist*dist)) : double3{0.0f, 0.0f, 0.0f}); }


/////////////////////////////////////////////////////////////////////////////////////////////
// calculates Lorentz force F on a charge q (single particle) given charge velocity and vectors E and B at charge position
//  q  --> charge being affected  (C)
//  v  --> charge velocity        (m/s)
//  E  --> electric field vector  (V/m <=> N/C)
//  B  --> magnetic field vector  (T   <=> N/m/A <=> N*A/m <=> kg/(s^2)/A)
/////////////////////////////////////////////////////////////////////////////////////////////
__host__ __device__ inline float3  lorentzForce_q(float  q, const float3  &v, const float3  &E, const float3  &B)
{ return q*(E + cross(v, B)); }
__host__ __device__ inline double3 lorentzForce_q(double q, const double3 &v, const double3 &E, const double3 &B)
{ return q*(E + cross(v, B)); }

/////////////////////////////////////////////////////////////////////////////////////////////
// calculates Lorentz force denisty f on a field cell given charge density, charge velocity, and vectors E and B at cell
//  ==> NOTE:   F = integral(integral(integral(f*dV)))
//              J = p*v
//  qd --> (rho) charge density   (C/m^3)
//  v  --> charge velocity        (m/s)
//  E  --> electric field vector  (V/m <=> N/C)
//  B  --> magnetic field vector  (T   <=> N/m/A <=> N*A/m <=> kg/(s^2)/A)
/////////////////////////////////////////////////////////////////////////////////////////////
__host__ __device__ inline float3  lorentzForce_p(float  qd, const float3  &v, const float3  &E, const float3  &B)
{ return qd*E + cross(currentDensity(qd,v), B); }
__host__ __device__ inline double3 lorentzForce_p(double qd, const double3 &v, const double3 &E, const double3 &B)
{ return qd*E + cross(currentDensity(qd,v), B); }



/////////////////////////////
//// MAXWELL's EQUATIONS ////
/////////////////////////////



/////////////////////
//// GAUSS'S LAW ////
/////////////////////////////////////////////////////////////////////////////////////////////
// calculates divergence of vector E at a field cell given charge density
//  qd --> (rho) charge density at cell (total charge per unit volume)            (C/m^3)
/////////////////////////////////////////////////////////////////////////////////////////////
__host__ __device__ inline float  gausssLaw_divE(float  qd) { return (qd / (float)M_ep0); }
__host__ __device__ inline double gausssLaw_divE(double qd) { return (qd / M_ep0); }
/////////////////////////////////////////////////////////////////////////////////////////////
// calculates charge density at a field cell given gradient of vector E (converts to divergence)
//  gE  --> gradient of electric field vector at cell (dE/dx, dE/dy, dE/dz)  (V/m <=> N/C)
/////////////////////////////////////////////////////////////////////////////////////////////
__host__ __device__ inline float  gausssLaw_qd(const float3  &gE) { return ((gE.x + gE.y + gE.z) * (float)M_ep0); }
__host__ __device__ inline double gausssLaw_qd(const double3 &gE) { return ((gE.x + gE.y + gE.z) * M_ep0); }



///////////////////////
//// AMPERE'S LAW ////
/////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////
//// FARADAY'S LAW ////
/////////////////////////////////////////////////////////////////////////////////////////////
// calculates updated magentic field B(p,t) given previous state, dt
//  ==> timestep dt, cell dimensions cs, and previous state t-1:
//      { B(p), E(p), E(p + <cs.x,0,0>), E(p + <0,cs.y,0>), E(p + <0,0,cs.z>) }(t-1)  ==>  { B(p) }(t)
//
//  B0  --> B(p, t-1)              previous magnetic field vector at cell            (T   <=> N/m/A <=> N*A/m <=> kg/(s^2)/A)
//  E0  --> E(p, t-1)              previous electric field vector at cell            (V/m <=> N/C)
//  Ex1 --> E(p+<cx,0,0>, t-1)     previous electric field vector at next cell (+X)  (V/m <=> N/C)
//  Ey1 --> E(p+<0,cy,0>, t-1)     previous electric field vector at next cell (+Y)  (V/m <=> N/C)
//  Ez1 --> E(p+<0,0,cz>, t-1)     previous electric field vector at next cell (+Z)  (V/m <=> N/C)
//  cs  --> size of single field cell                                                (m)
//  dt  --> physics timestep                                                         (s)
/////////////////////////////////////////////////////////////////////////////////////////////
__host__ __device__ inline float3  faradaysLaw_B(const float3  &B0, const float3  &E0,
                                                 const float3  &Ex1, const float3  &Ey1, const float3  &Ez1,
                                                 const float3  &cs, float  dt)
{
  return B0 + dt*float3 {  (Ez1.y - E0.y)/cs.z - (Ey1.z - E0.z)/cs.y,
                           (Ex1.z - E0.z)/cs.x - (Ez1.x - E0.x)/cs.z,
                           (Ey1.x - E0.x)/cs.y - (Ex1.y - E0.y)/cs.x };
}
__host__ __device__ inline double3 faradaysLaw_B(const double3 &B0, const double3 &E0,
                                                 const double3 &Ex1, const double3 &Ey1, const double3 &Ez1,
                                                 const double3 &cs, double dt)
{ 
  return B0 + dt*double3{  (Ez1.y - E0.y)/cs.z - (Ey1.z - E0.z)/cs.y,
                           (Ex1.z - E0.z)/cs.x - (Ez1.x - E0.x)/cs.z,
                           (Ey1.x - E0.x)/cs.y - (Ex1.y - E0.y)/cs.x };
}


#endif // PHYSICS_H
