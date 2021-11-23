#ifndef FLUID_CUH
#define FLUID_CUH

#include "em.cuh"

enum EdgeType  // boundary conditions
  {
    EDGE_IGNORE = -1,
    EDGE_WRAP   =  0, // boundaries wrap to opposite side (2D --> torus?)
    EDGE_VOID,        // forces and material properties pass through boundary (lost)
    EDGE_FREESLIP,    // layer of material at boundary can move parallel to boundary edge
    EDGE_NOSLIP,      // layer of material at boundary sticks to surface (vNormal = 0)
  };

enum IntegrationType  // integration for advection
  {
    INTEGRATION_INVALID = -1,
    INTEGRATION_FORWARD_EULER = 0,
    INTEGRATION_REVERSE_EULER = 1,
    // TODO: RK4 (, etc.?)
  };


template<typename T>
struct FluidParams : public FieldParams<T>
{
  EdgeType edgeNX = EDGE_FREESLIP; EdgeType edgePX = EDGE_FREESLIP; // edge behavior
  EdgeType edgeNY = EDGE_WRAP; EdgeType edgePY = EDGE_WRAP;
  EdgeType edgeNZ = EDGE_WRAP; EdgeType edgePZ = EDGE_WRAP;
  
  IntegrationType vIntegration = INTEGRATION_FORWARD_EULER; // velocity integration
  
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
