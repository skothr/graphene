#ifndef MAXWELL_CUH
#define MAXWELL_CUH

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include "cuda-tools.h"
#include "vector-operators.h"
#include "physics.h"
#include "field.cuh"
#include "raytrace.h"
#include "material.h"
#include "units.hpp"
#include "draw.cuh"


//// EM FIELD ////
template<typename T>
class EMField : public FieldBase
{
public:
  using VT2 = typename DimType<T, 2>::VEC_T;
  using VT3 = typename DimType<T, 3>::VEC_T;
  
  Field<T>   Qn; // - charge field
  Field<T>   Qp; // + charge field
  Field<VT3> Qv; // charge velocity field
  Field<VT3> E;  // electric field
  Field<VT3> B;  // magnetic field
  Field<Material<T>> mat; // material field
  
  std::vector<FieldBase*> FIELDS {{&Qp, &Qn, &Qv, &E, &B, &mat}};
  
  virtual bool create(int3 sz) override;
  virtual void destroy() override;
  virtual void pullData() override;
  virtual void pushData() override;
  virtual void clear()    override;
  void copyTo(EMField<T> &other) const;
  
  virtual bool allocated() const override;
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

template<typename T>
void EMField<T>::copyTo(EMField<T> &other) const
{
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
                  else { std::cout << "====> WARNING(EMField::copyTo()): Failed to cast field type! (VT2/VT3/Material)\n"; }
                }
            }
        }
    }
  else { std::cout << "====> WARNING(EMField::copyTo()): Field not allocated!\n"; }
}


template<typename T> class CudaExpression; // forward declaration

// maxwell physics update
template<typename T> void updateCharge  (FluidField<T> &src, FluidField<T> &dst, FluidParams<T> &cp);
template<typename T> void updateElectric(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> &cp);
template<typename T> void updateMagnetic(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> &cp);

#endif // MAXWELL_CUH
