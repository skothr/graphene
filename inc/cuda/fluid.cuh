#ifndef FLUID_CUH
#define FLUID_CUH

#include "em.cuh"

template<typename T>
struct FluidParams : public EMParams<T>
{
  // boundary conditions for each edge of field
  BoundType edgeNX = BOUND_WRAP; BoundType edgePX = BOUND_WRAP;
  BoundType edgeNY = BOUND_SLIP; BoundType edgePY = BOUND_SLIP;
  BoundType edgeNZ = BOUND_SLIP; BoundType edgePZ = BOUND_SLIP;
  
  IntegrationType vIntegration = INTEGRATION_RK4; // fluid velocity integration
  
  T density   = 1.5; // fluid density   (constant)
  T viscosity = 1.0; // fluid viscosity (constant)
  
  int pIter1 = 111; // pressure iterations for pre-advection step
  int pIter2 = 111; // pressure iterations for post-advection step
  int vIter  = 111; // viscosity iterations

  bool clearPressure = false;          // if true, clear pressure to zero each step before projection
  bool limitV = false; T maxV = 64.0f; // limit fluid velocity magnitude (clip magnitude to value)
};

//// FLUID FIELD ////
template<typename T>
class FluidField : public EMField<T>
{
  typedef typename cuda_vec<T, 3>::VT VT3;
public:
  Field<VT3> v;   // fluid velocity
  Field<T>   p;   // fluid pressure
  Field<T>   div; // fluid divergence
  
  FluidField() { this->FIELDS.push_back(&v); this->FIELDS.push_back(&p); this->FIELDS.push_back(&div); }
  FluidField(const FluidField &other) = default;
  FluidField& operator=(const FluidField &other) = default;
};

template<typename T> void fluidAdvect   (FluidField<T> &src, FluidField<T> &dst, const FluidParams<T> &fp);
template<typename T> void fluidViscosity(FluidField<T> &src, FluidField<T> &dst, const FluidParams<T> &fp, int iter);
template<typename T> void fluidPressure (FluidField<T> &src, FluidField<T> &dst, const FluidParams<T> &fp, int iter);
template<typename T> void fluidExtForces(FluidField<T> &src,                     const FluidParams<T> &fp);

#endif // FLUID_CUH
