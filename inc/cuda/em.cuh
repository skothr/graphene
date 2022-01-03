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



template<typename T>
struct EMParams : public FieldParams<T>
{
  int  rCoulomb     = 11;          // effective radius of Coulomb force
  T    coulombMult  = 1.0;         // Coulomb force multiplier
  T    coulombBlend = 0.5;         // blend D/E
  int  divBIter     = 11;          // magnetic "pressure" iterations (to remove divergence)
  
  IntegrationType qIntegration = INTEGRATION_RK4; // charge velocity integration
};

//// EM FIELD ////
template<typename T>
class EMField : public FieldBase
{
  typedef typename cuda_vec<T, 3>::VT VT3;
public:
  Field<T>   Qn;  // - charge
  Field<T>   Qp;  // + charge
  Field<VT3> Qnv; // - charge velocity
  Field<VT3> Qpv; // + charge velocity
  Field<VT3> E;   // electric field
  Field<VT3> B;   // magnetic field
  Field<Material<T>> mat; // EM material
  
  Field<T>   divB; // divergence of B
  Field<T>   Bp;   // B pressure (to remove divergence)

  FieldBounds bounds; // boundary behavior for inter-cell faces (BOUND_WRAP treated same as BOUND_VOID for internal faces)

  // list of sub-field pointers for looping
  std::vector<FieldBase*> FIELDS {{&Qp, &Qn, &Qnv, &Qpv, &E, &B, &mat, &divB, &Bp, &bounds}};

  EMField() = default;
  EMField(const EMField &other) = default;
  EMField& operator=(const EMField &other) = default;
  
  virtual bool create(int3 sz) override;
  virtual void destroy() override;
  virtual void pullData() override;
  virtual void pushData() override;
  virtual void clear()    override;
  virtual bool allocated() const override;
  virtual void pullData(unsigned long i0, unsigned long i1) override;
  void copyTo(EMField<T> &other) const;
  virtual FieldBase* makeCopy() const override;
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
  typedef typename cuda_vec<T, 2>::VT VT2;
  if(allocated())
    {
      if(other.size != this->size) { other.create(this->size); }
      for(int i = 0; i < FIELDS.size(); i++)
        { // check if 3D vector field
          Field<VT3> *f3 = reinterpret_cast<Field<VT3>*>(FIELDS[i]);
          if(f3) { f3->copyTo(*reinterpret_cast<Field<VT3>*>(other.FIELDS[i])); }
          else
            { // check if scalar field
              Field<T> *f1 = reinterpret_cast<Field<T>*>(FIELDS[i]);
              if(f1) { f1->copyTo(*reinterpret_cast<Field<T>*>(other.FIELDS[i])); }
              else
                { // check if boundary field
                  FieldBounds *fb = reinterpret_cast<FieldBounds*>(FIELDS[i]);
                  if(fb) { fb->copyTo(*reinterpret_cast<FieldBounds*>(other.FIELDS[i])); }
                  else
                    { // check if Material field
                      Field<Material<T>> *fM = reinterpret_cast<Field<Material<T>>*>(FIELDS[i]);
                      if(fM) { fM->copyTo(*reinterpret_cast<Field<Material<T>>*>(other.FIELDS[i]));} 
                      else // unknown type
                        { std::cout << "====> WARNING(EMField::copyTo): Failed to cast field type! (VT2/VT3/Material)\n"; }
                    }
                }
            }
        }
    }
  else { std::cout << "====> WARNING(EMField::copyTo): Field not allocated!\n"; }
}

template<typename T>
FieldBase* EMField<T>::makeCopy() const
{
  EMField<T> *cf = nullptr;
  if(allocated())
    {
      cf = new EMField<T>();
      if(cf->create(size))
        {
          this->copyTo(*cf);
          getLastCudaError("EMField::makeCopy() --> cudaMemcpy\n");
        }
    }
  else { std::cout << "======> WARNING(EMField::makeCopy): Field not allocated!\n"; }
  return cf;
}



template<typename T> class CudaExpression; // forward declaration

// maxwell physics update
template<typename T> void updateCharge   (FluidField<T> &src, FluidField<T> &dst, const FluidParams<T> &cp);
template<typename T> void updateElectric (FluidField<T> &src, FluidField<T> &dst, const FluidParams<T> &cp);
template<typename T> void updateMagnetic (FluidField<T> &src, FluidField<T> &dst, const FluidParams<T> &cp);

// calculate E from Coulomb forces
template<typename T> void updateCoulomb  (FluidField<T> &src, FluidField<T> &dst, const FluidParams<T> &cp);
// calculate E from divergence of charge
template<typename T> void chargePotential(FluidField<T> &src, FluidField<T> &dst, const FluidParams<T> &cp, int iter);
// remove divergence from magnetic field
template<typename T> void updateDivB     (FluidField<T> &src, FluidField<T> &dst, const FluidParams<T> &cp, int iter);



#endif // MAXWELL_CUH
