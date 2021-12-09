#ifndef FLUID_CUH
#define FLUID_CUH

#include "em.cuh"

template<typename T>
struct FluidParams : public EMParams<T>
{
  // boundary conditions for each edge of field
  EdgeType edgeNX = EDGE_WRAP;     EdgeType edgePX = EDGE_WRAP;
  EdgeType edgeNY = EDGE_FREESLIP; EdgeType edgePY = EDGE_FREESLIP;
  EdgeType edgeNZ = EDGE_FREESLIP; EdgeType edgePZ = EDGE_FREESLIP;
  
  IntegrationType vIntegration = INTEGRATION_RK4; // fluid velocity integration
  
  T density   = 1.5; // fluid density   (constant)
  T viscosity = 1.0; // fluid viscosity (constant)
  
  int pIter1 = 111; // pressure iterations for pre-advection step
  int pIter2 = 111; // pressure iterations for post-advection step
  int vIter  = 111; // viscosity iterations
  
  bool limitV = false; T maxV = 64.0f; // limit fluid velocity magnitude (clipped)
};

//// FLUID FIELD ////
template<typename T>
class FluidField : public EMField<T>
{
  typedef typename DimType<T, 3>::VEC_T VT3;
public:
  Field<VT3> v; // fluid velocity
  Field<T>   p; // fluid pressure
  Field<T> div; // fluid divergence
  
  FluidField() { this->FIELDS.push_back(&v); this->FIELDS.push_back(&p); this->FIELDS.push_back(&div); }
};


template<typename T> void fluidAdvect        (FluidField<T> &src, FluidField<T> &dst, FluidParams<T> fp);
template<typename T> void fluidViscosity     (FluidField<T> &src, FluidField<T> &dst, FluidParams<T> fp, int iter);
template<typename T> void fluidPressure      (FluidField<T> &src, FluidField<T> &dst, FluidParams<T> fp, int iter);
template<typename T> void fluidExternalForces(FluidField<T> &src, FluidParams<T> fp);

#endif // FLUID_CUH
