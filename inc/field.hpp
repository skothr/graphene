#ifndef FIELD_HPP
#define FIELD_HPP

#include <string>
#include <sstream>
#include <iostream>

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include "vector-operators.h"
#include "cuda-tools.h"
#include "physics.h"
#include "raytrace.h"

// forward declarations
typedef void* ImTextureID;


//// EM MATERIAL PROPERTIES ////
template<typename T>
struct Material
{
  bool nonVacuum = false; // if false, use user-defined vacuum properties in ChargeParams instead (defined this way so memset(material, 0) clears it to vacuum) 
  union { T permittivity = 1.0f; T epsilon; }; // epsilon -- electric permittivity (E)
  union { T permeability = 1.0f; T mu;      }; // mu      -- magnetic permeability (B)
  union { T conductivity = 0.0f; T sigma;   }; // sigma   -- material conductivity (Q)

  __host__ __device__ Material() { }
  __host__ __device__ Material(T permit, T permeab, T conduct, bool vacuum_=true)
    : permittivity(permit), permeability(permeab), conductivity(conduct), nonVacuum(!vacuum_) { }

  __host__ __device__ bool vacuum() const { return !nonVacuum; }
  
  // NOTE:
  //   E(t) = alphaE*E(t-1) + betaE*dE/dt
  //   B(t) = alphaB*B(t-1) + betaB*dB/dt
  struct Factors { T alphaE; T betaE; T alphaB; T betaB; };
  __host__ __device__ Factors getFactors(T dt, T cellSize) const
  {
    // E --  (1 / (1 + dt[s/(2mu)])) /
    T cE     = dt * conductivity/(2*permeability);
    T dE     = 1  / (1 + cE);
    T alphaE = dE * (1 - cE);
    T betaE  = dt/cellSize * dE/permeability;
    // B
    T cB     = dt * conductivity/(2*permittivity);
    T dB     = 1  / (1 + cB);
    T alphaB = dB * (1 - cB);
    T betaB  = dt/cellSize * dB/permittivity;
    return Factors{ alphaE, betaE, alphaB, betaB };
  }
};
template<typename T> inline std::ostream& operator<<(std::ostream &os, const Material<T> &mat)
{
  if(mat.vacuum()) { os << "Material<vacuum>"; }
  else             { os << "Material<ep=" << mat.permittivity <<  "|mu=" << mat.permeability << "|sig=" << mat.conductivity << ">"; }
  return os;
}





//// FIELD PARAMS BASE CLASS ////
struct FieldParamsBase { };

#define SIZE_MULT 1.0e-13f

// //// HYPER FIELD PARAMS ////
// struct FieldParams : public FieldParamsBase
// {
//   float3 physPos   = float3{0.0f, 0.0f, 0.0f}; // (m) 3D location of cell with index 0 
//   float3 physSize  = float3{1.0f, 1.0f, 1.0f}; // (m) 3D size of field (min index to max index)
//   float  texMult   = 1.0f; // ratio of texture resolution to field resolution
//   float  dt        = 0.1f; // frame timestep
//   float  gravity   = 0.0f; // force of gravity

//   __host__ __device__ inline float3 cellSize(const int3 &fSize) const // returns physical cell size vector
//   { return physSize / float3{(float)fSize.x, (float)fSize.y, (float)fSize.z}; }
// };

//// HYPER FIELD BASE: base class for hyper-dimensional fields (dtype/dimension defined in child class) ////
//template<int N>
class FieldBase
{
public:
  int3 size;
  //FieldParams params;
  unsigned long long numCells = 0;
  unsigned long long typeSize = 0;
  unsigned long long dataSize = 0;

  __host__ __device__ unsigned long long idx(int ix, int iy=0, int iz=0) const // data indexing -- should work for any dimensionality <= 3D
  { return (unsigned long long)ix + (unsigned long long)size.x*((unsigned long long)iy + ((unsigned long long)size.y*(unsigned long long)iz)); }
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




//// HYPER FIELD -- handles host/device data ////
template<typename T>
class Field : public FieldBase
{
public:
  T *hData   = nullptr;
  T *dData   = nullptr;

  //__device__ const T& operator[](unsigned int i) const { return dData[i]; }
  __device__ T& operator[](long long i) { return dData[i]; }
  
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
      unsigned long long tSize  = (unsigned long long)sizeof(T);
      unsigned long long dSize  = tSize*nCells;
      if(nCells <= 0)
        { std::cout << "====> ERROR: Could not create Field with size " << sz << "\n"; return false; }
      else
        { std::cout << "Creating Hyper-field ("  << sz << ")... --> data: " << nCells << "*" << tSize << " ==> " << dSize << "\n"; }

      // init device data
      int err = cudaMalloc((void**)&dData, (unsigned long long)dSize);
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

  void copyTo(CudaTexture &other)
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
  
  // GL interop
  void initGL(int3 sz);  // initialize CUDA-->opengl interop
  void bind();    // bind for use with opengl
  void release(); // unbind
  float4* map();   // texture data mapping to device pointer for rendering via CUDA kernel
  void  unmap();  // texture data unmapping
  ImTextureID* texId() const { return reinterpret_cast<ImTextureID*>(glTex); }

  virtual void clear() override { if(allocated() && map()) { cudaMemset(dData, 0, dataSize); unmap(); } }
};

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
  
  // OpenGL texture
  glGenTextures(1, &glTex); glBindTexture(GL_TEXTURE_2D, glTex);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, sz.x, sz.y, 0, GL_RGBA, GL_FLOAT, 0);
  glGenBuffers(1, &glPbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, glPbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, sz.x*sz.y*sizeof(float4), 0, GL_STREAM_COPY);

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
          mapped = true;
          //if(nbytes == 0) { return nullptr; } // TODO: check?
          getLastCudaError(("CudaTexture::map() -->" + std::to_string(err) + "\n").c_str());
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























//// EM FIELD ////
template<typename T>
class ChargeField : public FieldBase
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
  //Field<VT2> AB; // alpha/beta values from material field
  
  const std::vector<FieldBase*> FIELDS {{&Q, &QPV, &QNV, &E, &B, &mat}}; //, &AB}};
  
  virtual bool create(int3 sz) override;
  virtual void destroy() override;
  virtual void pullData() override;
  virtual void pushData() override;
  virtual void clear()    override;
  void copyTo(ChargeField<T> &other);
  
  virtual bool allocated() const override;
  virtual std::string hDataStr(const Vec3i &pos) const override;  
};

template<typename T>
bool ChargeField<T>::create(int3 sz)
{
  if(sz > int3{0,0,0} && size != sz)
    {
      destroy();
      for(auto f : FIELDS) { if(!f->create(sz)) { return false; } }
      this->size = sz;
      return true;
    }
  else { return false; }
}
template<typename T> void ChargeField<T>::destroy()  { for(auto f : FIELDS) { f->destroy(); } this->size = int3{0, 0, 0};  }
template<typename T> void ChargeField<T>::pullData() { for(auto f : FIELDS) { f->pullData(); } }
template<typename T> void ChargeField<T>::pushData() { for(auto f : FIELDS) { f->pushData(); } }
template<typename T> bool ChargeField<T>::allocated() const { for(auto f : FIELDS) { if(!f->allocated()) { return false; } } return true; }
template<typename T> std::string ChargeField<T>::hDataStr(const Vec3i &p) const { return Q.hDataStr(p); }
template<typename T> void ChargeField<T>::clear()    { for(auto f : FIELDS) { f->clear(); } }

template<typename T>
void ChargeField<T>::copyTo(ChargeField<T> &other)
{
  if(allocated())
    {
      if(other.size != this->size) { other.create(this->size); }
      for(int i = 0; i < FIELDS.size(); i++)
        {
          Field<VT3> *f3 = reinterpret_cast<Field<VT3>*>(FIELDS[i]);
          if(f3)  { f3->copyTo(*reinterpret_cast<Field<VT3>*>(other.FIELDS[i]));  }
          else
            {
              Field<VT2> *f2 = reinterpret_cast<Field<VT2>*>(FIELDS[i]);
              if(f2)  { f2->copyTo(*reinterpret_cast<Field<VT2>*>(other.FIELDS[i])); }
              else
                {
                  Field<Material<T>> *fM = reinterpret_cast<Field<Material<T>>*>(FIELDS[i]);
                  if(fM) { fM->copyTo(*reinterpret_cast<Field<Material<T>>*>(other.FIELDS[i]));} 
                }
            }
        }
    }
  else { std::cout << "====> WARNING(ChargeField::copyTo()): Field not allocated!\n"; }
}

//// FIELD PARAMS
struct ChargeParams
{
  bool   boundReflect = false;
  float  t            = 0.0f;                     // time
  float  dt           = 0.01f;                    // timestep
  float3 cs           = float3{0.1f, 0.1f, 0.1f}; // cell size
  float3 bs           = float3{0.0f, 0.0f, 0.0f}; // bound size
  Material<float> material; // default vacuum material
  ChargeParams(bool reflect, float dt_, float3 cs_, float3 bs_)
    : boundReflect(reflect), dt(dt_), cs(cs_), bs(bs_) { }
};

//// RENDER PARAMS
struct EmRenderParams
{
  // field components
  float4 Qcol  = float4{1.0f, 0.0f, 0.0f, 1.0f};
  float4 Ecol  = float4{0.0f, 1.0f, 0.0f, 1.0f};
  float4 Bcol  = float4{0.0f, 0.0f, 1.0f, 1.0f};
  float  Qmult = 0.2f;
  float  Emult = 0.2f;
  float  Bmult = 0.2f;
  // materials
  float4 epCol   = float4{1.0f, 0.0f, 0.0f, 1.0f};
  float4 muCol   = float4{0.0f, 1.0f, 0.0f, 1.0f};
  float4 sigCol  = float4{0.0f, 0.0f, 1.0f, 1.0f};
  float  epMult  = 0.2f;
  float  muMult  = 0.2f;
  float  sigMult = 0.2f;
  // 2D parameters
  int   numLayers2D = 1; // blends layers from numLayers to 0 (top-down)
  // 3D parameters
  float opacity    = 0.1f;
  float brightness = 2.0f;
};


enum // bit flags for applying multipliers to field values
  {
   IDX_NONE = 0x00,
   IDX_R    = 0x01, // scale signal by 1/lenth(r)   at each point
   IDX_R2   = 0x02, // scale signal by 1/length(r)^2 at each point
   IDX_T    = 0x04, // scale signal by theta   at each point
   IDX_SIN  = 0x08, // scale signal by sin(2*pi*t*frequency) at each point
   IDX_COS  = 0x10 // scale signal by cos(2*pi*t*frequency) at each point
  };
// for drawing signal in with mouse
template<typename T>
struct SignalPen
{
  using VT2 = typename DimType<T, 2>::VECTOR_T;
  using VT3 = typename DimType<T, 3>::VECTOR_T;

  bool active    = true;
  bool square    = false; // square pen
  bool cellAlign = false; // align to cells
  T    radius    = 10.0;  // pen size in fluid cells
  T    mult      = 1.0;   // signal multiplier
  T    frequency = 0.4;   // Hz(1/t in sim time) for sin/cos mult flags

  // base field values to add
  VT2 Q   = VT2{0.0, 0.0};
  VT3 QPV = VT3{0.0, 0.0,  0.0};
  VT3 QNV = VT3{0.0, 0.0,  0.0};
  VT3 E   = VT3{0.0, 0.0,  1.0};
  VT3 B   = VT3{0.0, 0.0, -1.0};
  
  // int tMult   = IDX_NONE;
  // int rMult   = IDX_NONE;
  // int r2Mult  = IDX_NONE;
  // int sinMult = IDX_NONE;
  // int cosMult = IDX_NONE;

  int Qopt    = IDX_NONE;
  int QPVopt  = IDX_NONE;
  int QNVopt  = IDX_NONE;
  int Eopt    = IDX_NONE;
  int Bopt    = IDX_NONE;
  
  SignalPen() : Eopt(IDX_SIN), Bopt(IDX_COS) { }
};

template<typename T>
struct MaterialPen
{
  bool active       = true;
  bool square       = false;
  bool vacuum       = false; // (eraser)
  T    radius       = 10.0;   // pen in fluid cells
  T    mult         = 1.0;   // multiplier
  T    permittivity = 1.0;   // vacuum permittivity (E)
  T    permeability = 1.0;   // vacuum permeability (B)
  T    conductivity = 0.0;   // vacuum conductivity (Q)
};



template<typename T> class CudaExpression; // forward declaration

// in field.cu
template<typename T> void fieldFillValue  (Field<T> &dst, const T & val);
template<typename T> void fieldFill       (Field<T> &dst, CudaExpression<T> *dExpr);
template<typename T> void fieldFillChannel(Field<T> &dst, CudaExpression<typename Dims<T>::BASE> *dExpr, int channel);
// add signal from source field
template<typename T> void addSignal        (ChargeField<T> &signal, ChargeField<T> &dst, const ChargeParams &cp);
// add signal from mouse position/pen
template<typename T> void addSignal  (const typename DimType<T, 3>::VECTOR_T &pSrc, ChargeField<T> &dst, const SignalPen<T> &pen, const ChargeParams &cp);
template<typename T> void addMaterial(const typename DimType<T, 3>::VECTOR_T &pSrc, ChargeField<T> &dst, const MaterialPen<T> &pen, const ChargeParams &cp);

// physics update
template<typename T> void updateCharge     (ChargeField<T> &src, ChargeField<T> &dst, ChargeParams &cp);
template<typename T> void updateElectric   (ChargeField<T> &src, ChargeField<T> &dst, ChargeParams &cp);
template<typename T> void updateMagnetic   (ChargeField<T> &src, ChargeField<T> &dst, ChargeParams &cp);

// rendering
template<typename T> void renderFieldEM (ChargeField<T> &src,          CudaTexture &dst, const EmRenderParams &rp);
template<typename T> void renderFieldMat(Field<Material<T>> &src, CudaTexture &dst, const EmRenderParams &rp);
// ray marching
template<typename T> void raytraceFieldEM (ChargeField<T> &src, CudaTexture &dst, const Camera<double> &camera,
                                           const EmRenderParams &rp, const ChargeParams &cp, double aspect);
template<typename T> void raytraceFieldMat(ChargeField<T> &src, CudaTexture &dst, const Camera<double> &camera,
                                           const EmRenderParams &rp, const ChargeParams &cp, double aspect);

#endif // FIELD_HPP
