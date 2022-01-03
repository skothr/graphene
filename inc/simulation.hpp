#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include <vector>
#include "field.hpp"
#include "field.cuh"
#include "fluid.cuh"
#include "em.cuh"

template<typename T>
struct SimInfo
{
  T   t     = 0.0f; // simulation time passed since initial state
  T   fps   = 0.0f; // render fps
  int frame = 0;    // number of frames rendered since initial state
  int uStep = 0;    // keeps track of microsteps betweeen frames
};

template<typename T>
class Simulation
{
private:
  std::vector<FieldBase*> mFieldsSrc;
  std::vector<FieldBase*> mFieldsDst;
  FluidField<T> *mTempState = nullptr; // temp state (avoids destroying input source state)

  SimInfo<T> mInfo;
  
public:
  Simulation() { }

  bool create()  { return true; }
  void destroy() { }
  
  void addField(FieldBase *f) { mFieldsSrc.push_back(f); mFieldsDst.push_back(f->makeCopy()); }
  void removeField(int i)     { mFieldsSrc.erase(mFieldsSrc.begin()+i); mFieldsDst.erase(mFieldsDst.begin()+i); }

  void step(FluidField<T> *src, FluidField<T> *dst, const FluidParams<T> &cp, const FieldInterface<T> *fui);
};

template<typename T>
void Simulation<T>::step(FluidField<T> *src, FluidField<T> *dst, const FluidParams<T> &cp, const FieldInterface<T> *fui)
{
  // // step simulation
  // if(DESTROY_LAST_STATE) { std::swap(temp, src); } // overwrite source state
  // else                   { src->copyTo(*temp); }   // don't overwrite previous state (use temp)

  // //// INPUTS
  // // (NOTE: remove added sources to avoid persistent lumps building up)
  // addSignal(*mInputV,   src->v,   cp, cp.u.dt); // add V input signal
  // addSignal(*mInputP,   src->p,   cp, cp.u.dt); // add P input signal
  // addSignal(*mInputQn,  src->Qn,  cp, cp.u.dt); // add Qn input signal
  // addSignal(*mInputQp,  src->Qp,  cp, cp.u.dt); // add Qp input signal
  // addSignal(*mInputQnv, src->Qnv, cp, cp.u.dt); // add Qnv input signal
  // addSignal(*mInputQpv, src->Qpv, cp, cp.u.dt); // add Qpv input signal
  // addSignal(*mInputE,   src->E,   cp, cp.u.dt); // add E input signal
  // addSignal(*mInputB,   src->B,   cp, cp.u.dt); // add B input signal

  //// EM STEP (Q/Qv/E/B)
  if(fui->updateEM)
    {
      if(fui->updateCoulomb) { updateCoulomb (*src, *dst, cp);              std::swap(src, dst); } // ∇·E = Q/ε₀
      if(fui->updateQ)       { updateCharge  (*src, *dst, cp);              std::swap(src, dst); } // Q,Qv --> Q (advect Q within fluid)
      if(fui->updateE)       { updateElectric(*src, *dst, cp);              std::swap(src, dst); } // δE/δt = (∇×B - J)
      if(fui->updateB)       { updateMagnetic(*src, *dst, cp);              std::swap(src, dst); } // δB/δt = -(∇×E)
      if(fui->updateDivB)    { updateDivB    (*src, *dst, cp, cp.divBIter); std::swap(src, dst); } // ∇·B = 0
    }
  //// FLUID STEP (V/P)
  if(fui->updateFluid) // V, P
    {
      if(fui->updateP1)      { fluidPressure (*src, *dst, cp, cp.pIter1);   std::swap(src, dst); } // PRESSURE SOLVE (1)
      if(fui->updateAdvect)  { fluidAdvect   (*src, *dst, cp);              std::swap(src, dst); } // ADVECT
      if(fui->updateVisc)    { fluidViscosity(*src, *dst, cp, cp.vIter);    std::swap(src, dst); } // VISCOSITY SOLVE
      if(fui->applyGravity)  { fluidExtForces(*src,  cp); }                                        // EXTERNAL FORCES (in-place)
      if(fui->updateP2)      { fluidPressure (*src, *dst, cp, cp.pIter2);   std::swap(src, dst); } // PRESSURE SOLVE (2)
    }
  
  // std::swap(src, dst); // (un-)swap final result back into dst
  // if(!DESTROY_LAST_STATE) { std::swap(mTempState, src); } // use other state as new temp (pointer changes if number of steps is odd)}
  // else                    { std::swap((FluidField<T>*&)mStates.back(), src); }
  // mStates.pop_front(); mStates.push_back(dst);

  // // increment time/frame info
  // mInfo.t += mUnits.dt;
  // mInfo.uStep++;
  // if(mInfo.uStep >= mParams.uSteps) { mInfo.frame++; mInfo.uStep = 0; mNewSimFrame = true; }
}




#endif // SIMULATION_HPP
