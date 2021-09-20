#ifndef FIELD_CUH
#define FIELD_CUH

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include "cuda-tools.h"
#include "vector-operators.h"
#include "physics.h"
#include "raytrace.h"
#include "material.h"
#include "units.hpp"
#include "draw.cuh"

// forward declarations
typedef void* ImTextureID;


//// FIELD PARAMS ////
template<typename T>
struct FieldParams
{
  typedef typename DimType<T, 3>::VEC_T VT3;
  Units<T>  u;          // units
  T t = 0.0f;           // current simulation time
  T    decay   = 0.92f; // source decay
  bool reflect = false; // reflective boundaries
  VT3 fp;               // field position
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
      if(sz == this->size) { std::cout << "Field already allocated (" << size << ")\n"; return true; }
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
      std::cout << "Destroying Field...\n";
      if(dData) { cudaFree(dData); dData = nullptr; }
      if(hData) { free(hData);     hData = nullptr; }
      size = int3{0, 0, 0}; numCells = 0; dataSize = 0;
      getLastCudaError("Field::destroy()");
      std::cout << "  (DONE)\n";
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




//// EM FIELD ////
template<typename T>
class EMField : public FieldBase
{
public:
  using VT2 = typename DimType<T, 2>::VEC_T;
  using VT3 = typename DimType<T, 3>::VEC_T;
  
  Field<VT3> E;   // electric field
  Field<VT3> B;   // magnetic field
  Field<Material<T>> mat; // material field
  
  const std::vector<FieldBase*> FIELDS {{// &Q, &QPV, &QNV,
                                         &E, &B, &mat}};
  
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



//// CUDA TEXTURE ////
// --> contains texture data (special case of 2D field)
struct CudaTexture : public Field<float4>
{
  cudaGraphicsResource *mPboResource = nullptr;
  GLuint glTex  = 0;
  GLuint glPbo  = 0;
  bool mapped = false;
  bool bound  = false;

  __device__ const float4& operator[](unsigned int i) const { return dData[i]; }
  __device__       float4& operator[](unsigned int i)       { return dData[i]; }
  
  virtual bool create(int3 sz) override;
  virtual void destroy()   override;
  virtual bool allocated() const override { return (gCudaInitialized && size.x > 0 && size.y > 0 && dataSize > 0 && typeSize > 0 && mPboResource); }
  virtual void pullData()  override       { map(); Field<float4>::pullData(); unmap(); }
  virtual void pushData()  override       { map(); Field<float4>::pushData(); unmap(); }
  void copyTo(CudaTexture &other);
  
  // GL interop
  void    initGL(int3 sz); // initialize CUDA/opengl interop
  void    bind();    // bind for use with opengl
  void    release(); // unbind
  float4* map();     // texture data mapping to device pointer for rendering via CUDA kernel
  void    unmap();   // texture data unmapping
  ImTextureID* texId() const { return reinterpret_cast<ImTextureID*>(glTex); }

  virtual void clear() override { if(allocated() && map()) { cudaMemset(dData, 0, dataSize); unmap(); } }
};

inline void CudaTexture::copyTo(CudaTexture &other)
{
  other.create(this->size);
  if(this->allocated() && other.create(this->size) && other.allocated())
    {
      if(map() && other.map())
        {
          Field<float4>::copyTo(other);
          unmap(); other.unmap();
          getLastCudaError("CudaTexture::copyTo()\n"); }
    }
  else { std::cout << "====> WARNING(CudaTexture::copyTo()): Field not allocated!\n"; }
}

inline bool CudaTexture::create(int3 sz)
{
  initCudaDevice();
  if(gCudaInitialized && min(sz) > 0)
    {
      sz.z = 1; // 2D only
      if(sz == this->size) { std::cout << "Texture already allocated (" << size << ")\n"; return true; }
      else { destroy(); }
      unsigned long nCells = (unsigned long)sz.x*(unsigned long)sz.y;
      unsigned long tSize  = (unsigned long)sizeof(float4);
      unsigned long dSize  = tSize*nCells;
      if(nCells <= 0) { std::cout << "====> ERROR: Could not create CudaTexture with size " << sz << "\n"; return false; }
      else            { std::cout << "Creating Cuda Texture ("  << sz << ")... --> data: " << nCells << "*" << tSize << " ==> " << dSize << "\n"; }

      // init device data
      int err = cudaMalloc((void**)&dData, (unsigned long)dSize);
      if(err) { std::cout << "====> ERROR: Failed to allocated memory for field!\n"; return false; }
      err = cudaMemset(dData, 0, dSize);
      if(err) { std::cout << "====> ERROR: Failed to initialize memory for field!\n"; cudaFree(dData); dData = nullptr; return false; }
      
      initGL(sz);
      
      getLastCudaError("CudaTexture::create(sz)");
      size = sz; numCells = nCells; typeSize = tSize; dataSize = dSize;
      return true;
    }
  if(!gCudaInitialized) { std::cout << "====> WARNING(CudaTexture::create()): CUDA device not initialized!\n"; }
  if(min(sz) <= 0)      { std::cout << "====> WARNING(CudaTexture::create()): zero size! " << size << "\n"; }
  if(!mPboResource)     { std::cout << "====> WARNING(CudaTexture::create()): PBO resource not initialized!\n"; }
  return false;
}

inline void CudaTexture::initGL(int3 sz)
{
  // delete old buffers
  if(glTex > 0)    { glDeleteTextures(1, &glTex); glTex = 0; }
  if(glPbo > 0)    { glDeleteBuffers(1, &glPbo);  glPbo = 0; }
  if(mPboResource)
    {
      cudaGraphicsUnregisterResource(mPboResource); mPboResource = nullptr;
      getLastCudaError("CudaTexture-->cudaGraphicsUnregisterResource()\n");
    }
  // 2D OpenGL texture
  glGenTextures(1, &glTex); glBindTexture(GL_TEXTURE_2D, glTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, sz.x, sz.y, 0, GL_RGBA, GL_FLOAT, 0);
  glGenBuffers(1, &glPbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, glPbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, sz.x*sz.y*sizeof(float4), 0, GL_STREAM_COPY);
  // CUDA resource for interop
  cudaGraphicsGLRegisterBuffer(&mPboResource, glPbo, cudaGraphicsMapFlagsWriteDiscard);
  getLastCudaError("CudaTexture-->cudaGraphicsRegisterResource()\n");
  if(!mPboResource) { std::cout << "====> ERROR(CudaTexture::initGL()): mPboResource NULL --> failed to register!\n"; }
  
  glBindTexture(GL_TEXTURE_2D, 0);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  getLastCudaError("CudaTexture::initGL()\n");
}

inline void CudaTexture::destroy()
{
  if(allocated())
    {
      std::cout << "Destroying Cuda Texture...\n";
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
      if(hData)        { free(hData); hData = nullptr; }
      if(mPboResource) { cudaGraphicsUnregisterResource(mPboResource); mPboResource = nullptr; }
      if(glPbo > 0)    { glDeleteBuffers(1,  &glPbo); glPbo = 0; }
      if(glTex > 0)    { glDeleteTextures(1, &glTex); glTex = 0; }
      getLastCudaError("CudaTexture::destroy()");
      std::cout << "  (DONE)\n";
    }
  size = int3{0, 0, 0};  numCells = 0;  dataSize = 0;
}

inline void CudaTexture::bind()
{
  if(allocated() && !bound)
    {
      // load texture from pbos
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, glPbo);
      // load texture from pbos
      glBindTexture(GL_TEXTURE_2D, glTex);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, size.x, size.y, GL_RGBA, GL_FLOAT, 0);
      bound = true;
    }
  if(!gCudaInitialized)          { std::cout << "====> WARNING(CudaTexture::bind()): CUDA device not initialized!\n"; }
  if(size.x == 0 || size.y == 0) { std::cout << "====> WARNING(CudaTexture::bind()): zero size! " << size << "\n";    }
  if(!mPboResource)              { std::cout << "====> WARNING(CudaTexture::bind()): PBO resource not initialized!\n";   }
}

inline void CudaTexture::release()
{
  if(allocated() && bound)
    {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
      glBindTexture(GL_TEXTURE_2D, 0);
      bound = false;
    }
  if(!gCudaInitialized)          { std::cout << "====> WARNING(CudaTexture::release()): CUDA device not initialized!\n";  }
  if(size.x == 0 || size.y == 0) { std::cout << "====> WARNING(CudaTexture::release()): zero size! " << size << "\n";     }
  if(!mPboResource)              { std::cout << "====> WARNING(CudaTexture::release()): PBO resource not initialized!\n"; }
}

inline float4* CudaTexture::map()
{
  if(!mPboResource)
    {
      std::cout << "====> WARNING(CudaTexture::map()): PBO resource not initialized!\n";
      //create(size); // try to initialize
      return nullptr;
    }  
  if(allocated())
    {
      if(!mapped)
        {
          int err = cudaGraphicsMapResources(1, &mPboResource, 0);
          if(err == CUDA_ERROR_ALREADY_MAPPED)
            {
              std::cout << "====> WARNING: CudaTexture already mapped! (" << err << ") --> cudaGraphicsMapResources\n";
              getLastCudaError(("CudaTexture::map() --> " + std::to_string(err) + "\n").c_str());
              return nullptr;
            }
          else if(err)
            {
              std::cout << "====> WARNING: CudaTexture failed to map! (" << err << ") --> cudaGraphicsMapResources\n";
              getLastCudaError(("CudaTexture::map() --> " + std::to_string(err) + "\n").c_str());
              mapped = false; dData = nullptr; return nullptr;
            }
          size_t nbytes;
          err = cudaGraphicsResourceGetMappedPointer((void**)&dData, &nbytes, mPboResource);
          if(err == CUDA_ERROR_ALREADY_MAPPED)
            {
              std::cout << "====> WARNING: CudaTexture already mapped! (" << err << ") --> (cudaGraphicsResourceGetMappedPointer)\n";
              getLastCudaError(("CudaTexture::map() --> " + std::to_string(err) + "\n").c_str());
              mapped = true; return nullptr;
            }
          else if(err)
            {
              std::cout << "====> WARNING: CudaTexture failed to map! (" << err << ") --> (cudaGraphicsResourceGetMappedPointer)\n";
              getLastCudaError(("CudaTexture::map() --> " + std::to_string(err) + "\n").c_str());
              mapped = false; dData = nullptr; return nullptr;
            }
          mapped = true; getLastCudaError(("CudaTexture::map() -->" + std::to_string(err) + "\n").c_str());
        }
      else { std::cout << "====> WARNING: CudaTexture::map() called on mapped texture!\n"; }      
      return dData;
    }
  if(!gCudaInitialized)          { std::cout << "====> WARNING(CudaTexture::map()): CUDA device not initialized!\n";  }
  if(size.x == 0 || size.y == 0) { std::cout << "====> WARNING(CudaTexture::map()): zero size! " << size << "\n";     }
  if(!mPboResource)              { std::cout << "====> WARNING(CudaTexture::map()): PBO resource not initialized!\n"; }
  return nullptr;
}

inline void CudaTexture::unmap()
{
  if(gCudaInitialized && size.x > 0 && size.y > 0 && mPboResource)
    {
      if(mapped || dData)
        {
          int err = cudaGraphicsUnmapResources(1, &mPboResource, 0);
          if(err)
            {
              std::cout << "====> WARNING: CudaTexture failed to unmap (" << err << ")! --> cudaGraphicsUnmapResources\n";
              getLastCudaError("CudaTexture::unmap()\n");
              dData = nullptr; mapped = false; return;
            }
          else { getLastCudaError("CudaTexture::unmap()\n"); }
        }
      else { std::cout << "====> WARNING: CudaTexture::unmap() called on unmapped texture!\n"; }
    }
  if(!gCudaInitialized) { std::cout << "====> WARNING(CudaTexture::unmap()): CUDA device not initialized!\n";  }
  if(min(size) <= 0)    { std::cout << "====> WARNING(CudaTexture::unmap()): zero size! " << size << "\n";     }
  if(!mPboResource)     { std::cout << "====> WARNING(CudaTexture::unmap()): PBO resource not initialized!\n"; }
  dData = nullptr; mapped = false;
}



template<typename T> class CudaExpression; // forward declaration

// in field.cu
template<typename T> void fillFieldMaterial(Field<Material<T>> &dst, CudaExpression<T> *dExprEp, CudaExpression<T> *dExprMu, CudaExpression<T> *dExprSig);
template<typename T> void fillFieldValue   (Field<T> &dst, const T & val);
template<typename T> void fillField        (Field<T> &dst, CudaExpression<T> *dExpr);
template<typename T> void fillFieldChannel (Field<T> &dst, CudaExpression<typename Dim<T>::BASE_T> *dExpr, int channel);
// physics update
template<typename T> void updateCharge     (EMField<T> &src, EMField<T> &dst, FieldParams<T> &cp);
template<typename T> void updateElectric   (EMField<T> &src, EMField<T> &dst, FieldParams<T> &cp);
template<typename T> void updateMagnetic   (EMField<T> &src, EMField<T> &dst, FieldParams<T> &cp);


#endif // FIELD_CUH
