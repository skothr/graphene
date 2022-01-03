#ifndef FIELD_CUH
#define FIELD_CUH

#include <iomanip>

#include "cuda-tools.h"
#include "vector-operators.h"
#include "physics.h"
#include "material.h"
#include "units.cuh"
#include "draw.cuh"

enum BoundType  // boundary conditions
  {
    BOUND_VOID = 0, // no boundary (forces/etc. pass normally, discarded if passing edge of field)
    BOUND_WRAP,     // boundaries at edge of field wrap to opposite side (hypertorus manifold?)
    BOUND_SLIP,     // layer of material at boundary can move parallel to boundary edge
    BOUND_NOSLIP,   // layer of material at boundary sticks to surface (vNormal = 0) [bounce off/reflect?]
  };
enum IntegrationType  // integration for advection step(s)
  {
    INTEGRATION_FORWARD_EULER =  0,
    INTEGRATION_BACKWARD_EULER, // (TODO: fix)
    INTEGRATION_RK4,
    // TODO: others?
    INTEGRATION_COUNT,
  };
__host__ __device__ inline bool isImplicit(IntegrationType it) { return (it == INTEGRATION_BACKWARD_EULER); }

#ifndef __NVCC__
#include <vector>
#include <string>
static const std::vector<std::string> g_edgeNames       {{ "Void", "Wrap", "Slip", "No Slip" }};
static const std::vector<std::string> g_integrationNames{{ "Forward Euler", "Backward Euler", "RK4" }};
#endif // __NVCC__



//// FIELD PARAMS ////
template<typename T>
struct FieldParams
{
  typedef typename cuda_vec<T, 3>::VT VT3;
  VT3  fp;                     // field position
  int3 fs = int3{256, 256, 8}; // size of field (number of cells)

  Units<T> u;      // field units
  T t     = 0.0f;  // current simulation time  
  T decay = 0.10f; // source decay
  VT3 gravity = VT3{0,0,0}; // force of gravity
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
  __host__ __device__ unsigned long idx(unsigned long ix, unsigned long iy=0, unsigned long iz=0) const { return ix + size.x*(iy + (size.y*iz)); }
  __host__ __device__ unsigned long idx(const int3  &p) const { return idx((unsigned long)p.x, (unsigned long)p.y, (unsigned long)p.z); }
  __host__ __device__ unsigned long idx(const Vec3i &p) const { return idx((unsigned long)p.x, (unsigned long)p.y, (unsigned long)p.z); }
  __host__ __device__ unsigned long idx(const Vec3f &p) const { return idx((unsigned long)p.x, (unsigned long)p.y, (unsigned long)p.z); }
    
  virtual bool create(int3 sz) = 0;
  virtual void destroy()  = 0;
  virtual void pullData() = 0;
  virtual void pushData() = 0;
  virtual void clear()    = 0;
  virtual void pullData(unsigned long i0, unsigned long i1) { }
  virtual bool allocated() const { return (gCudaInitialized && size.x > 0 && size.y > 0 && size.z > 0); }

  virtual FieldBase* makeCopy() const = 0;
};


// byte index and bit offset for a boolean field -- handles boolean compression (bitwise)
struct BitBool
{
  unsigned char mask  = 0u;
  unsigned char *byte = nullptr;

  BitBool& operator=(const BitBool &b) = default;
  BitBool& operator=(bool b) { *byte = ((*byte) & ~mask) + ((*byte) | (b ? mask : 0)); return *this; }

  operator bool() const { return ((*byte) & mask); }
};



//// FIELD -- handles host/device data ////
template<typename T>
class Field : public FieldBase
{
public:
  T *hData   = nullptr;
  T *dData   = nullptr;

#ifdef __NVCC__
  // (T != bool)
  template<typename T_=T, typename std::enable_if<!std::is_same<bool, T_>::value>::type* = nullptr>
  __device__ const T& operator[](unsigned long i) const { return dData[i]; }
  template<typename T_=T, typename std::enable_if<!std::is_same<bool, T_>::value>::type* = nullptr>
  __device__       T& operator[](unsigned long i)       { return dData[i]; }
  // (T == bool)
  template<typename T_=T, typename std::enable_if< std::is_same<T_, bool>::value>::type* = nullptr>
  __device__  BitBool operator[](unsigned long i)
  {
    const int byte = i/8; const int offset = i%8;
    return BitBool{(0x1 << offset), (unsigned char*)&dData[i]};
  }
#else
  // (T != bool)
  template<typename T_=T, typename std::enable_if<!std::is_same<bool, T_>::value>::type* = nullptr>
  __host__ const T& operator[](unsigned long i) const { return hData[i]; }
  template<typename T_=T, typename std::enable_if<!std::is_same<bool, T_>::value>::type* = nullptr>
  __host__       T& operator[](unsigned long i)       { return hData[i]; }
  // (T == bool)
  template<typename T_=T, typename std::enable_if< std::is_same<T_, bool>::value>::type* = nullptr>
  __host__  BitBool operator[](unsigned long i)
  {
    const int byte = i/8; const int offset = i%8;
    return BitBool{(0x1 << offset), (unsigned char*)&hData[i]};
  }
#endif
  
  Field()  = default;
  ~Field() = default;
  Field(const Field &other) = default;
  Field& operator=(const Field &other) = default;
  
  virtual bool create(int3 sz) override;
  virtual void destroy()  override;
  virtual void pullData() override;
  virtual void pushData() override;
  virtual void clear()    override;
  virtual bool allocated() const override;
  virtual void pullData(unsigned long i0, unsigned long i1) override;
  void copyTo(Field<T> &other) const;
  virtual FieldBase* makeCopy() const override;
};

template<typename T>
bool Field<T>::create(int3 sz)
{
  initCudaDevice();
  if(gCudaInitialized && min(sz) > 0)
    {
      if(sz == this->size) { return true; }
      else { destroy(); }
      
      const unsigned long nCells = (unsigned long)sz.x*(unsigned long)sz.y*(unsigned long)sz.z;
      const unsigned long tSize  = sizeof(T);
      unsigned long       dSize  = tSize*nCells;
      if constexpr (std::is_same<T, bool>::value) { dSize = std::max(1ul, dSize/8ul); } // bool --> stored bitwise
      
      if(nCells <= 0) { std::cout << "======> ERROR: Could not create Field with size " << sz << "\n"; return false; }
      else
        {
          std::cout << "==== Creating Field     " << std::setw(18) << ("("+to_string(sz)+")") << " |  data: " << std::setw(8) << std::right << nCells << " * "
                    << std::setw(4) << std::left << tSize << " --> " << std::setw(10) << std::right << dSize << "... ";
        }
      
      // init device data
      int err = cudaMalloc((void**)&dData, dSize);
      if(err) { std::cout << "\n======> ERROR: Failed to allocate memory for field!\n"; return false; }
      err = cudaMemset(dData, 0, dSize);
      if(err) { std::cout << "\n======> ERROR: Failed to initialize memory for field!\n"; cudaFree(dData); dData = nullptr; return false; }
      
      getLastCudaError("Field::create(sz)");
      size = sz; numCells = nCells; typeSize = tSize; dataSize = dSize;
      std::cout << " (done)\n";
      return true;
    }
  else
    {
      if(!gCudaInitialized) { std::cout << "======> WARNING(Field::create): CUDA device not initialized!\n"; }
      if(min(sz) <= 0)      { std::cout << "======> WARNING(Field::create): zero size! " << size << " / " << sz << "\n"; }
      return false;
    }
}

template<typename T>
void Field<T>::destroy()
{
  if(allocated())
    {
      std::cout << "==== Destroying Field (" << size << ")... ";
      if(dData) { cudaFree(dData); dData = nullptr; }
      if(hData) { free(hData);     hData = nullptr; }
      size = int3{0, 0, 0}; numCells = 0; dataSize = 0;
      getLastCudaError("Field::destroy");
      std::cout << " (done)\n";
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
  else { std::cout << "======> WARNING(Field::pullData): Field not allocated!\n"; }
}

template<typename T>
void Field<T>::pullData(unsigned long i0, unsigned long i1)
{
  if(allocated())
    {
      if(!hData) { hData = (T*)malloc(dataSize); }
      
      unsigned long byte0 = i0; unsigned long byte1 = i1;
      if constexpr (std::is_same<T, bool>::value) { byte0 /= 8; byte1 /= 8; } // bool --> stored bitwise

      cudaMemcpy(hData+byte0, dData+byte0, (byte1-byte0)*typeSize, cudaMemcpyDeviceToHost);
      getLastCudaError(("Field::pullData(" + std::to_string(i0) + ":" + std::to_string(i1) + " / " + std::to_string(dataSize) + ")\n").c_str());
    }
  else { std::cout << "======> WARNING(Field::pullData): Field not allocated!\n"; }
}

template<typename T>
void Field<T>::pushData()
{
  if(allocated() && hData) { cudaMemcpy(dData, hData, dataSize, cudaMemcpyHostToDevice); getLastCudaError("Field::pushData()\n"); }
  else                     { std::cout << "======> WARNING(Field::pushData): Field not allocated!\n"; }
}

template<typename T>
void Field<T>::clear()
{
  if(allocated())
    {
      cudaMemset(dData, 0, dataSize);
      if(hData) { memset(hData, 0, dataSize); }
    }
}

template<typename T>
void Field<T>::copyTo(Field<T> &other) const
{
  if(allocated() && other.allocated())
    { cudaMemcpy(other.dData, dData, dataSize, cudaMemcpyDeviceToDevice); getLastCudaError("Field::copyTo()\n"); }
  else
    { std::cout << "======> WARNING(Field::copyTo): Field not allocated!\n"; }
}

template<typename T>
FieldBase* Field<T>::makeCopy() const
{
  Field<T> *cf = nullptr;
  if(allocated())
    {
      cf = new Field<T>();
      if(cf->create(size))
        { this->copyTo(*cf); getLastCudaError("Field::makeCopy() --> cudaMemcpy\n"); }
    }
  else
    { std::cout << "======> WARNING(Field::makeCopy): Field not allocated!\n"; }
  return cf;
}



// handles boundaries between each cell and field edges
//  NOTE: encapsulates extra element in each dimension -- pass unmodified sizes and indices
class FieldBounds : public FieldBase
{
public:
  Field<unsigned char> bx; // boundaries planes along X axis (each parallel to YZ plane)
  Field<unsigned char> by; // boundaries planes along Y axis (each parallel to ZX plane)
  Field<unsigned char> bz; // boundaries planes along Z axis (each parallel to XY plane)

#ifdef __NVCC__
  __device__ BoundType nx(const int3 &p) { return static_cast<BoundType>(bx[bx.idx(p)]); }             // lower -X bound of cell at p (p.x + 0)
  __device__ BoundType ny(const int3 &p) { return static_cast<BoundType>(by[by.idx(p)]); }             // lower -Y bound of cell at p (p.y + 0)
  __device__ BoundType nz(const int3 &p) { return static_cast<BoundType>(bz[bz.idx(p)]); }             // lower -Z bound of cell at p (p.z + 0)
  __device__ BoundType px(const int3 &p) { return static_cast<BoundType>(bx[bx.idx(p+int3{1,0,0})]); } // upper +X bound of cell at p (p.x + 1)
  __device__ BoundType py(const int3 &p) { return static_cast<BoundType>(by[by.idx(p+int3{0,1,0})]); } // upper +Y bound of cell at p (p.y + 1)
  __device__ BoundType pz(const int3 &p) { return static_cast<BoundType>(bz[bz.idx(p+int3{0,0,1})]); } // upper +Z bound of cell at p (p.z + 1)
#else
  __host__   BoundType nx(const int3 &p) { return static_cast<BoundType>(bx[bx.idx(p)]); }             // lower -X bound of cell at p (p.x + 0)
  __host__   BoundType ny(const int3 &p) { return static_cast<BoundType>(by[by.idx(p)]); }             // lower -Y bound of cell at p (p.y + 0)
  __host__   BoundType nz(const int3 &p) { return static_cast<BoundType>(bz[bz.idx(p)]); }             // lower -Z bound of cell at p (p.z + 0)
  __host__   BoundType px(const int3 &p) { return static_cast<BoundType>(bx[bx.idx(p+int3{1,0,0})]); } // upper +X bound of cell at p (p.x + 1)
  __host__   BoundType py(const int3 &p) { return static_cast<BoundType>(by[by.idx(p+int3{0,1,0})]); } // upper +Y bound of cell at p (p.y + 1)
  __host__   BoundType pz(const int3 &p) { return static_cast<BoundType>(bz[bz.idx(p+int3{0,0,1})]); } // upper +Z bound of cell at p (p.z + 1)
#endif // __NVCC__
  
  FieldBounds() = default;
  virtual bool create(int3 sz) override;
  virtual void destroy()   override;
  virtual void pullData()  override;
  virtual void pushData()  override;
  virtual void clear()     override;
  virtual bool allocated() const override;
  virtual void pullData(unsigned long i0, unsigned long i1) override;
  void copyTo(FieldBounds &other) const;
  virtual FieldBase* makeCopy() const override;
};

inline bool FieldBounds::create(int3 sz)
{
  if(sz > int3{0,0,0} && size != sz)
    {
      destroy();
      sz += 1; // boundaries between cells -- 1 extra in each dimension
      if(!bx.create(sz+int3{1,0,0})) { return false; }
      if(!by.create(sz+int3{0,1,0})) { bx.destroy(); return false; }
      if(!bz.create(sz+int3{0,0,1})) { bx.destroy(); by.destroy(); return false; }
      this->size     = sz;
      return true;
    }
  else { return false; }
}

inline void FieldBounds::destroy()         { bx.destroy();  by.destroy();  bz.destroy(); this->size = int3{0,0,0}; }
inline void FieldBounds::pullData()        { bx.pullData(); by.pullData(); bz.pullData(); }
inline void FieldBounds::pushData()        { bx.pushData(); by.pushData(); bz.pushData(); }
inline bool FieldBounds::allocated() const { return (bx.allocated() && by.allocated() && bz.allocated()); }
inline void FieldBounds::clear()           { bx.clear(); by.clear(); bz.clear(); }
inline void FieldBounds::pullData(unsigned long i0, unsigned long i1)
{ bx.pullData(i0, i1); by.pullData(i0, i1); bz.pullData(i0, i1); }

inline void FieldBounds::copyTo(FieldBounds &other) const
{
  if(allocated())
    {
      if(other.size != this->size) { other.create(this->size); }
      bx.copyTo(other.bx); by.copyTo(other.by); bz.copyTo(other.bz);
    }
  else { std::cout << "====> WARNING(FieldBounds::copyTo): Field not allocated!\n"; }
}


inline FieldBase* FieldBounds::makeCopy() const
{
  FieldBounds *cf = nullptr;
  if(allocated())
    {
      cf = new FieldBounds();
      if(cf->create(size)) { this->copyTo(*cf); getLastCudaError("FieldBounds::makeCopy() --> cudaMemcpy\n"); }
    }
  else { std::cout << "======> WARNING(FieldBounds::makeCopy): Field not allocated!\n"; }
  return cf;
}











// forward declarations
template<typename T> class CudaExpression;

// in field.cu
template<typename T> void fillFieldMaterial(Field<Material<T>> &dst, CudaExpression<T> *dExprEp, CudaExpression<T> *dExprMu, CudaExpression<T> *dExprSig);
template<typename T> void fillFieldValue   (Field<T> &dst, const T & val);
template<typename T> void fillField        (Field<T> &dst, CudaExpression<T> *dExpr);
template<typename T> void fillFieldChannel (Field<T> &dst, CudaExpression<typename cuda_vec<T>::BASE> *dExpr, int channel);


#endif // FIELD_CUH
