#ifndef CUDA_TEXTURE_CUH
#define CUDA_TEXTURE_CUH

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include "field.cuh"

typedef void* ImTextureID;

//// CUDA TEXTURE ////
// --> contains texture data (special case of 2D field)
struct CudaTexture : public Field<float4>
{
protected:
  virtual void pullData(unsigned long i0, unsigned long i1) { }
  
public:
  
  cudaGraphicsResource *mPboResource = nullptr;
  GLuint glTex = 0;
  GLuint glPbo = 0;
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
          getLastCudaError("CudaTexture::copyTo\n"); }
    }
  else { std::cout << "====> WARNING(CudaTexture::copyTo): Field not allocated!\n"; }
}

inline bool CudaTexture::create(int3 sz)
{
  sz.z = 1; // 2D(x,y) only
  if(Field<float4>::create(sz))
    {
      initGL(sz);
      getLastCudaError(("CudaTexture::create(<"+std::to_string(sz.x)+", "+std::to_string(sz.y)+", "+std::to_string(sz.z)+">)").c_str());
      return true;
    }
  if(!gCudaInitialized) { std::cout << "====> WARNING(CudaTexture::create): CUDA device not initialized!\n"; }
  if(min(sz) <= 0)      { std::cout << "====> WARNING(CudaTexture::create): zero size! " << size << "\n"; }
  if(!mPboResource)     { std::cout << "====> WARNING(CudaTexture::create): PBO resource not initialized!\n"; }
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
  if(!mPboResource) { std::cout << "====> ERROR(CudaTexture::initGL): mPboResource NULL --> failed to register!\n"; }
  
  glBindTexture(GL_TEXTURE_2D, 0);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  getLastCudaError("CudaTexture::initGL()\n");
}

inline void CudaTexture::destroy()
{
  if(allocated())
    {
      std::cout << "==== Destroying Cuda Texture... ";
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
      if(hData)        { free(hData); hData = nullptr; }
      if(mPboResource) { cudaGraphicsUnregisterResource(mPboResource); mPboResource = nullptr; }
      if(glPbo > 0)    { glDeleteBuffers(1,  &glPbo); glPbo = 0; }
      if(glTex > 0)    { glDeleteTextures(1, &glTex); glTex = 0; }
      getLastCudaError("CudaTexture::destroy()");
      std::cout << " (done)\n";
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
  if(!gCudaInitialized)          { std::cout << "====> WARNING(CudaTexture::bind): CUDA device not initialized!\n"; }
  if(size.x == 0 || size.y == 0) { std::cout << "====> WARNING(CudaTexture::bind): zero size! " << size << "\n";    }
  if(!mPboResource)              { std::cout << "====> WARNING(CudaTexture::bind): PBO resource not initialized!\n";   }
}

inline void CudaTexture::release()
{
  if(allocated() && bound)
    {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
      glBindTexture(GL_TEXTURE_2D, 0);
      bound = false;
    }
  if(!gCudaInitialized)          { std::cout << "====> WARNING(CudaTexture::release): CUDA device not initialized!\n";  }
  if(size.x == 0 || size.y == 0) { std::cout << "====> WARNING(CudaTexture::release): zero size! " << size << "\n";     }
  if(!mPboResource)              { std::cout << "====> WARNING(CudaTexture::release): PBO resource not initialized!\n"; }
}

inline float4* CudaTexture::map()
{
  if(!mPboResource)
    {
      std::cout << "====> WARNING(CudaTexture::map): PBO resource not initialized!\n";
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
  if(!gCudaInitialized)          { std::cout << "====> WARNING(CudaTexture::map): CUDA device not initialized!\n";  }
  if(size.x == 0 || size.y == 0) { std::cout << "====> WARNING(CudaTexture::map): zero size! " << size << "\n";     }
  if(!mPboResource)              { std::cout << "====> WARNING(CudaTexture::map): PBO resource not initialized!\n"; }
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
  if(!gCudaInitialized) { std::cout << "====> WARNING(CudaTexture::unmap): CUDA device not initialized!\n";  }
  if(min(size) <= 0)    { std::cout << "====> WARNING(CudaTexture::unmap): zero size! " << size << "\n";     }
  if(!mPboResource)     { std::cout << "====> WARNING(CudaTexture::unmap): PBO resource not initialized!\n"; }
  dData = nullptr; mapped = false;
}





#endif // CUDA_TEXTURE_CUH
