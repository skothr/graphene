#ifndef FLUID_CUH
#define FLUID_CUH

#include "maxwell.cuh"

enum EdgeType 
  { // types of boundaries
   EDGE_IGNORE = -1,
   EDGE_WRAP   =  0, // boundaries wrap to opposite side (2D --> torus?)
   EDGE_VOID   =  1, // forces and material properties pass through boundary (lost)
   EDGE_NOSLIP =  2, // layer of material at boundary sticks to surface (velocity = 0)
   EDGE_SLIP   =  3, // layer of material at boundary can move parallel to boundary edge (? not implemented)
  };


template<typename T>
struct FluidParams : public FieldParams<T>
{
  EdgeType edgeNX = EDGE_WRAP;   EdgeType edgePX = EDGE_WRAP;
  EdgeType edgeNY = EDGE_NOSLIP; EdgeType edgePY = EDGE_NOSLIP;
  EdgeType edgeNZ = EDGE_VOID;   EdgeType edgePZ = EDGE_VOID;

  bool updateP1     = true;
  bool updateAdvect = true;
  bool updateP2     = true;

  int pIter1 = 111;
  int pIter2 = 111;
};

//// FLUID FIELD ////
template<typename T>
class FluidField : public EMField<T>
{
public:
  typedef typename DimType<T, 3>::VEC_T VT3;
  
  Field<VT3> v; // fluid velocity
  Field<T>   p; // fluid pressure
  Field<T> div; // fluid divergence

  FluidField() { this->FIELDS.push_back(&v); this->FIELDS.push_back(&p); this->FIELDS.push_back(&div); }
};


template<typename T> void fluidStep(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> &params);


#endif // FLUID_CUH
