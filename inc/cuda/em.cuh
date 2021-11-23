#ifndef MAXWELL_CUH
#define MAXWELL_CUH

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include "cuda-tools.h"
#include "vector-operators.h"
#include "physics.h"
#include "field.cuh"
// #include "raytrace.h"
#include "material.h"
#include "units.cuh"
#include "draw.cuh"


//// EM FIELD ////
template<typename T>
class EMField : public FieldBase
{
  typedef typename DimType<T, 3>::VEC_T VT3;
public:
  Field<T>   Qn;  // - charge
  Field<T>   Qp;  // + charge
  Field<VT3> Qnv; // - charge velocity
  Field<VT3> Qpv; // + charge velocity
  Field<VT3> E;   // electric field
  Field<VT3> B;   // magnetic field
  Field<Material<T>> mat; // material

  Field<VT3> gradQ; // gradient of Q
  Field<T>   divE;  // divergence of E
  Field<T>   divB;  // divergence of B
  Field<T>   Ep;    // E presure (?)
  Field<T>   Bp;    // B pressure (to remove divergence) --> ?
  
  std::vector<FieldBase*> FIELDS {{&Qp, &Qn, &Qnv, &Qpv, &E, &B, &mat, &gradQ, &divE, &divB, &Bp}};
  
  virtual bool create(int3 sz) override;
  virtual void destroy() override;
  virtual void pullData() override;
  virtual void pushData() override;
  virtual void clear()    override;
  virtual bool allocated() const override;
  virtual void pullData(unsigned long i0, unsigned long i1) override;
  void copyTo(EMField<T> &other) const;
};

template<typename T>
bool EMField<T>::create(int3 sz)
{
  if(sz > int3{0,0,0} && size != sz)
    {
      destroy();
      for(auto f : FIELDS) { if(!f->create(sz)) { return false; } }
      this->size = sz; return true;
    }
  else { return false; }
}
template<typename T> void EMField<T>::destroy()         { for(auto f : FIELDS) { f->destroy();  } this->size = int3{0, 0, 0};  }
template<typename T> void EMField<T>::pullData()        { for(auto f : FIELDS) { f->pullData(); } }
template<typename T> void EMField<T>::pushData()        { for(auto f : FIELDS) { f->pushData(); } }
template<typename T> bool EMField<T>::allocated() const { for(auto f : FIELDS) { if(!f->allocated()) { return false; } } return true; }
template<typename T> void EMField<T>::clear()           { for(auto f : FIELDS) { f->clear(); } }

template<typename T> void EMField<T>::pullData(unsigned long i0, unsigned long i1)
{ for(int i = 0; i < FIELDS.size(); i++) { FIELDS[i]->pullData(i0, i1); } }

template<typename T>
void EMField<T>::copyTo(EMField<T> &other) const
{
  typedef typename DimType<T, 2>::VEC_T VT2;
  if(allocated())
    {
      if(other.size != this->size) { other.create(this->size); }
      for(int i = 0; i < FIELDS.size(); i++)
        {
          Field<VT3> *f3 = reinterpret_cast<Field<VT3>*>(FIELDS[i]);
          if(f3) { f3->copyTo(*reinterpret_cast<Field<VT3>*>(other.FIELDS[i]));  }
          else
            {
              Field<VT2> *f2 = reinterpret_cast<Field<VT2>*>(FIELDS[i]);
              if(f2) { f2->copyTo(*reinterpret_cast<Field<VT2>*>(other.FIELDS[i])); }
              else
                {
                  Field<Material<T>> *fM = reinterpret_cast<Field<Material<T>>*>(FIELDS[i]);
                  if(fM) { fM->copyTo(*reinterpret_cast<Field<Material<T>>*>(other.FIELDS[i]));} 
                  else { std::cout << "====> WARNING(EMField::copyTo): Failed to cast field type! (VT2/VT3/Material)\n"; }
                }
            }
        }
    }
  else { std::cout << "====> WARNING(EMField::copyTo): Field not allocated!\n"; }
}



template<typename T> class CudaExpression; // forward declaration

// maxwell physics update
template<typename T> void updateCharge   (FluidField<T> &src, FluidField<T> &dst, FluidParams<T> &cp);
template<typename T> void updateElectric (FluidField<T> &src, FluidField<T> &dst, FluidParams<T> &cp);
template<typename T> void updateMagnetic (FluidField<T> &src, FluidField<T> &dst, FluidParams<T> &cp);

// calculate E from Coulomb forces
template<typename T> void updateCoulomb  (FluidField<T> &src, FluidField<T> &dst, FluidParams<T> &cp);
// calculate E from divergence of charge
template<typename T> void chargePotential(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> &cp, int iter);
// remove divergence from magnetic field
template<typename T> void magneticCurl   (FluidField<T> &src, FluidField<T> &dst, FluidParams<T> &cp, int iter);



#endif // MAXWELL_CUH
