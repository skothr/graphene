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
  typedef typename DimType<T, 3>::VECTOR_T VT3;
  Units<T>  u;          // units
  T t = 0.0f;           // current simulation time
  VT3 fp;               // field position
  bool reflect = false; // reflective boundaries
};



//// FIELD BASE: base class for fields (data type defined in child class) ////
class FieldBase
{
public:
  int3 size;
  unsigned long long numCells = 0;
  unsigned long long typeSize = 0;
  unsigned long long dataSize = 0; // TODO: uncoalesced memory
  // data indexing -- should work for any dimensionality <= 3D
  __host__ __device__ unsigned long long idx(unsigned long long ix, unsigned long long iy=0, unsigned long long iz=0) const 
  { return ix + size.x*(iy + (size.y*iz)); }
  __host__ __device__ unsigned long long idx(const int3  &p) const { return idx(p.x, p.y, p.z); }
  __host__ __device__ unsigned long long idx(const Vec3i &p) const { return idx(p.x, p.y, p.z); }
  __host__ __device__ unsigned long long idx(const Vec3f &p) const { return idx(p.x, p.y, p.z); }
  
  virtual bool allocated() const { return (gCudaInitialized && size.x > 0 && size.y > 0 && size.z > 0 && dataSize > 0); }
  virtual std::string hDataStr(const Vec3i &pos) const { return ""; }

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

  __device__ const T& operator[](unsigned long long i) const { return dData[i]; }
  __device__       T& operator[](unsigned long long i)       { return dData[i]; }
  
  virtual bool create(int3 sz) override;
  virtual void destroy()  override;
  virtual void pullData() override;
  virtual void pushData() override;
  virtual void clear()    override { if(allocated()) { cudaMemset(dData, 0, dataSize); memset(hData, 0, dataSize); } }
  void copyTo(Field<T> &other);
  
  virtual bool allocated() const override;
  virtual std::string hDataStr(const Vec3i &pos) const override;
};

template<typename T>
bool Field<T>::create(int3 sz)
{
  initCudaDevice();
  if(gCudaInitialized && min(sz) > 0)
    {
      if(sz == this->size) { std::cout << "Field already allocated (" << size << ")\n"; return true; }
      else { destroy(); }
      unsigned long long nCells = (unsigned long long)sz.x*(unsigned long long)sz.y*(unsigned long long)sz.z;
      unsigned long long tSize  = sizeof(T);
      unsigned long long dSize  = tSize*nCells;
      if(nCells <= 0) { std::cout << "====> ERROR: Could not create Field with size " << sz << "\n"; return false; }
      else            { std::cout << "Creating Hyper-field ("  << sz << ")... --> data: " << nCells << "*" << tSize << " ==> " << dSize << "\n"; }

      // init device data
      int err = cudaMalloc((void**)&dData, dSize);
      if(err) { std::cout << "====> ERROR: Failed to allocated memory for field!\n"; return false; }
      err = cudaMemset(dData, 0, dSize);
      if(err) { std::cout << "====> ERROR: Failed to initialize memory for field!\n"; cudaFree(dData); dData = nullptr; return false; }
      // init host data
      hData = (T*)malloc(dSize);
      memset(hData, 0, dSize);
      
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
bool Field<T>::allocated() const { return (FieldBase::allocated() && dData && hData); }
template<typename T>
void Field<T>::pullData()
{
  if(allocated()) { cudaMemcpy(hData, dData, dataSize, cudaMemcpyDeviceToHost); getLastCudaError("Field::pullData()\n"); }
  else            { std::cout << "====> WARNING(Field::pullData()): Field not allocated!\n"; }
}
template<typename T>
void Field<T>::pushData()
{
  if(allocated()) { cudaMemcpy(dData, hData, dataSize, cudaMemcpyHostToDevice); getLastCudaError("Field::pushData()\n"); }
  else            { std::cout << "====> WARNING(Field::pushData()): Field not allocated!\n"; }
}
template<typename T>
void Field<T>::copyTo(Field<T> &other)
{
  if(allocated() && other.allocated())
    { cudaMemcpy(other.dData, dData, dataSize, cudaMemcpyDeviceToDevice); getLastCudaError("Field::copyTo()\n"); }
  else
    { std::cout << "====> WARNING(Field::copyTo()): Field not allocated!\n"; }
}
template<typename T>
std::string Field<T>::hDataStr(const Vec3i &p) const { std::stringstream ss; ss << hData[idx(p)]; return ss.str(); }










//// EM FIELD ////
template<typename T>
class EMField : public FieldBase
{
public:
  using VT2 = typename DimType<T, 2>::VECTOR_T;
  using VT3 = typename DimType<T, 3>::VECTOR_T;
  using ST = T; // scalar type, provided
  
  Field<VT2> Q;   // Q   = {q+, q-} (NOTE: Q.y represents density of negative particles, so sign should be positive)
  Field<VT3> QPV; // velocity of positive charge <-- **** idea: +/- charge velocity just inverted, these should be velocity over +/- cell borders ****
  Field<VT3> QNV; // velocity of negative charge
  Field<VT3> E;   // electric field
  Field<VT3> B;   // magnetic field
  Field<Material<T>> mat; // material field
  
  const std::vector<FieldBase*> FIELDS {{&Q, &QPV, &QNV, &E, &B, &mat}}; //, &AB}};
  
  virtual bool create(int3 sz) override;
  virtual void destroy() override;
  virtual void pullData() override;
  virtual void pushData() override;
  virtual void clear()    override;
  void copyTo(EMField<T> &other);
  
  virtual bool allocated() const override;
  virtual std::string hDataStr(const Vec3i &pos) const override;  
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
template<typename T> void EMField<T>::destroy()  { for(auto f : FIELDS) { f->destroy(); } this->size = int3{0, 0, 0};  }
template<typename T> void EMField<T>::pullData() { for(auto f : FIELDS) { f->pullData(); } }
template<typename T> void EMField<T>::pushData() { for(auto f : FIELDS) { f->pushData(); } }
template<typename T> bool EMField<T>::allocated() const { for(auto f : FIELDS) { if(!f->allocated()) { return false; } } return true; }
template<typename T> std::string EMField<T>::hDataStr(const Vec3i &p) const { return Q.hDataStr(p); }
template<typename T> void EMField<T>::clear()    { for(auto f : FIELDS) { f->clear(); } }

template<typename T>
void EMField<T>::copyTo(EMField<T> &other)
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
                }
            }
        }
    }
  else { std::cout << "====> WARNING(EMField::copyTo()): Field not allocated!\n"; }
}



//// CUDA TEXTURE ////
// --> contains texture data for a field
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
      unsigned long long nCells = (unsigned long long)sz.x*(unsigned long long)sz.y;
      unsigned long long tSize  = (unsigned long long)sizeof(float4);
      unsigned long long dSize  = tSize*nCells;
      if(nCells <= 0) { std::cout << "====> ERROR: Could not create CudaTexture with size " << sz << "\n"; return false; }
      else            { std::cout << "Creating Cuda Texture ("  << sz << ")... --> data: " << nCells << "*" << tSize << " ==> " << dSize << "\n"; }

      // init device data
      int err = cudaMalloc((void**)&dData, (unsigned long long)dSize);
      if(err) { std::cout << "====> ERROR: Failed to allocated memory for field!\n"; return false; }
      err = cudaMemset(dData, 0, dSize);
      if(err) { std::cout << "====> ERROR: Failed to initialize memory for field!\n"; cudaFree(dData); dData = nullptr; return false; }
      // init host data
      hData = (float4*)malloc(dSize);
      memset(hData, 0, dSize);
      
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
      std::cout << "Destroying CudaTexture...\n";
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
template<typename T> void fieldFillValue   (Field<T> &dst, const T & val);
template<typename T> void fieldFill        (Field<T> &dst, CudaExpression<T> *dExpr);
template<typename T> void fieldFillChannel (Field<T> &dst, CudaExpression<typename Dims<T>::BASE> *dExpr, int channel);
// physics update
template<typename T> void updateCharge     (EMField<T> &src, EMField<T> &dst, FieldParams<T> &cp);
template<typename T> void updateElectric   (EMField<T> &src, EMField<T> &dst, FieldParams<T> &cp);
template<typename T> void updateMagnetic   (EMField<T> &src, EMField<T> &dst, FieldParams<T> &cp);


#endif // FIELD_CUH
