#ifndef FIELD_CUH
#define FIELD_CUH

#include "cuda-tools.h"
#include "vector-operators.h"
#include "physics.h"
#include "raytrace.h"
#include "material.h"
#include "units.hpp"
#include "draw.cuh"

//// FIELD PARAMS ////
template<typename T>
struct FieldParams
{
  typedef typename DimType<T, 3>::VEC_T VT3;
  Units<T> u; // units
  T t = 0.0f; // current simulation time
  
  VT3  fp;                     // field position
  int3 fs = int3{256, 256, 8}; // number of cells in charge field

  T    decay   = 0.10f;    // source decay
  bool reflect = false;    // reflective boundaries
};


//// FIELD BASE: base class for fields (data type defined in child class) ////
class FieldBase
{
public:
  int3 size = int3{0,0,0};
  unsigned long numCells = 0;
  unsigned long typeSize = 0;
  unsigned long dataSize = 0; // TODO: uncoalesced memory (?)
  // data indexing -- should work for any dimensionality <= 3D
  __host__ __device__ unsigned long idx(unsigned long ix, unsigned long iy=0, unsigned long iz=0) const 
  { return ix + size.x*(iy + (size.y*iz)); }
  __host__ __device__ unsigned long idx(const int3  &p) const { return idx((unsigned long)p.x, (unsigned long)p.y, (unsigned long)p.z); }
  __host__ __device__ unsigned long idx(const Vec3i &p) const { return idx((unsigned long)p.x, (unsigned long)p.y, (unsigned long)p.z); }
  __host__ __device__ unsigned long idx(const Vec3f &p) const { return idx((unsigned long)p.x, (unsigned long)p.y, (unsigned long)p.z); }
  
  virtual bool allocated() const { return (gCudaInitialized && size.x > 0 && size.y > 0 && size.z > 0 && dataSize > 0); }

  // pure virtual functions
  virtual bool create(int3 sz) = 0;
  virtual void destroy()  = 0;
  virtual void pullData() = 0;
  virtual void pushData() = 0;
  virtual void clear()    = 0;
};


//// FIELD -- handles host/device data ////
template<typename T>
class Field : public FieldBase
{
public:
  T *hData   = nullptr;
  T *dData   = nullptr;

  __device__ const T& operator[](unsigned long i) const { return dData[i]; }
  __device__       T& operator[](unsigned long i)       { return dData[i]; }
  
  virtual bool create(int3 sz) override;
  virtual void destroy()  override;
  virtual void pullData() override;
  virtual void pushData() override;
  virtual void clear()    override
  {
    if(allocated())
      {
        cudaMemset(dData, 0, dataSize);
        if(hData) { memset(hData, 0, dataSize); }
      }
  }
  void copyTo(Field<T> &other) const;
  
  virtual bool allocated() const override;
};

template<typename T>
bool Field<T>::create(int3 sz)
{
  initCudaDevice();
  if(gCudaInitialized && min(sz) > 0)
    {
      if(sz == this->size) { return true; } //std::cout << "Field already allocated (" << size << ")\n"; return true; }
      else { destroy(); }
      unsigned long nCells = (unsigned long)sz.x*(unsigned long)sz.y*(unsigned long)sz.z;
      unsigned long tSize  = sizeof(T);
      unsigned long dSize  = tSize*nCells;
      if(nCells <= 0) { std::cout << "====> ERROR: Could not create Field with size " << sz << "\n"; return false; }
      else            { std::cout << "Creating Field ("  << sz << ")... --> data: " << nCells << "*" << tSize << " ==> " << dSize << "\n"; }

      // init device data
      int err = cudaMalloc((void**)&dData, dSize);
      if(err) { std::cout << "====> ERROR: Failed to allocated memory for field!\n"; return false; }
      err = cudaMemset(dData, 0, dSize);
      if(err) { std::cout << "====> ERROR: Failed to initialize memory for field!\n"; cudaFree(dData); dData = nullptr; return false; }
      
      getLastCudaError("Field::create(sz)");
      size = sz; numCells = nCells; typeSize = tSize; dataSize = dSize;
      return true;
    }
  else
    {
      if(!gCudaInitialized) { std::cout << "====> WARNING(Field::create()): CUDA device not initialized!\n"; }
      if(min(sz) <= 0)      { std::cout << "====> WARNING(Field::create()): zero size! " << size << " / " << sz << "\n"; }
      return false;
    }
}

template<typename T>
void Field<T>::destroy()
{
  if(allocated())
    {
      std::cout << "Destroying Field (" << size << ")... ";
      if(dData) { cudaFree(dData); dData = nullptr; }
      if(hData) { free(hData);     hData = nullptr; }
      size = int3{0, 0, 0}; numCells = 0; dataSize = 0;
      getLastCudaError("Field::destroy()");
      std::cout << "DONE\n";
    }
}
template<typename T>
bool Field<T>::allocated() const { return (FieldBase::allocated() && dData); }
template<typename T>
void Field<T>::pullData()
{
  if(allocated())
    {
      if(!hData) { hData = (T*)malloc(dataSize); }
      cudaMemcpy(hData, dData, dataSize, cudaMemcpyDeviceToHost); getLastCudaError("Field::pullData()\n");
    }
  else            { std::cout << "====> WARNING(Field::pullData()): Field not allocated!\n"; }
}
template<typename T>
void Field<T>::pushData()
{
  if(allocated() && hData) { cudaMemcpy(dData, hData, dataSize, cudaMemcpyHostToDevice); getLastCudaError("Field::pushData()\n"); }
  else                     { std::cout << "====> WARNING(Field::pushData()): Field not allocated!\n"; }
}
template<typename T>
void Field<T>::copyTo(Field<T> &other) const
{
  if(allocated() && other.allocated())
    { cudaMemcpy(other.dData, dData, dataSize, cudaMemcpyDeviceToDevice); getLastCudaError("Field::copyTo()\n"); }
  else
    { std::cout << "====> WARNING(Field::copyTo()): Field not allocated!\n"; }
}


// forward declarations
template<typename T> class CudaExpression;

// in field.cu
template<typename T> void fillFieldMaterial(Field<Material<T>> &dst, CudaExpression<T> *dExprEp, CudaExpression<T> *dExprMu, CudaExpression<T> *dExprSig);
template<typename T> void fillFieldValue   (Field<T> &dst, const T & val);
template<typename T> void fillField        (Field<T> &dst, CudaExpression<T> *dExpr);
template<typename T> void fillFieldChannel (Field<T> &dst, CudaExpression<typename Dim<T>::BASE_T> *dExpr, int channel);


#endif // FIELD_CUH
