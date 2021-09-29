#ifndef CUDA_VBO_CUH
#define CUDA_VBO_CUH

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
#include "material.h"
#include "units.hpp"

// X/Y/Z position and R/G/B/A color for a vertex 
//struct Vertex { float4 pos; }; //float4 color; };
typedef float4 Vertex;

//// CUDA VBO ////
// --> contains vertex data for drawing with opengl
struct CudaVBO
{
  unsigned long size = 0; // buffer size (number of vertices)
  bool   mapped = false;
  bool   bound  = false;
  GLuint glVbo  = 0;
  Vertex *dData = nullptr;
  cudaGraphicsResource *cuVbo = nullptr;

  __device__ const Vertex& operator[](unsigned int i) const { return dData[i]; }
  __device__       Vertex& operator[](unsigned int i)       { return dData[i]; }
  
  bool create(unsigned long sz);
  void destroy();
  
  bool allocated() const { return (gCudaInitialized && size > 0 && glVbo); }

  void copyTo(CudaVBO &other)
  {
    other.create(this->size);
    if(this->allocated() && other.create(this->size) && other.allocated())
      {
        if(map() && other.map())
          {
            cudaMemcpy(other.dData, dData, size*sizeof(Vertex), cudaMemcpyDeviceToDevice);
            unmap(); other.unmap();
            getLastCudaError("CudaVBO::copyTo()\n"); }
          }
    else { std::cout << "====> WARNING(CudaVBO::copyTo()): Buffer not allocated!\n"; }
  }
  
  // GL interop
  void initGL(unsigned long sz);  // initialize CUDA-->opengl interop
  void bind();    // bind for use with opengl
  void release(); // unbind
  void draw();    // draw vertices to screen (NOTE: probably need a shader, etc. attached)
  Vertex* map();  // texture data mapping to device pointer for rendering via CUDA kernel
  void  unmap();  // texture data unmapping
  void clear() { if(allocated() && map()) { cudaMemset(dData, 0, size*sizeof(Vertex)); unmap(); } }
};

inline bool CudaVBO::create(unsigned long sz)
{
  initCudaDevice();
  if(gCudaInitialized)
    {
      if(sz == this->size) { return true; } ///std::cout << "Buffer already allocated (" << size << ")\n"; return true; }
      else { destroy(); }
      std::cout << "Creating Cuda Texture ("  << sz << ")... --> data: " << sz << "*" << sizeof(Vertex) << " ==> " << sz*sizeof(Vertex) << "\n";

      // init device data
      int err = cudaMalloc((void**)&dData, sz*sizeof(Vertex));
      if(err) { std::cout << "====> ERROR: Failed to allocated memory for field!\n"; return false; }
      err = cudaMemset(dData, 0, sz*sizeof(Vertex));
      if(err) { std::cout << "====> ERROR: Failed to initialize memory for field!\n"; cudaFree(dData); dData = nullptr; return false; }
      // init VBO
      initGL(sz); getLastCudaError("CudaVBO::create(sz)");
      size = sz;
      return true;
    }
  if(!gCudaInitialized) { std::cout << "====> WARNING(CudaVBO::create()): CUDA device not initialized!\n"; }
  if(sz == 0)           { std::cout << "====> WARNING(CudaVBO::create()): zero size! " << sz << "\n"; }
  if(!glVbo)            { std::cout << "====> WARNING(CudaVBO::create()): VBO resource not initialized!\n"; }
  return false;
}

inline void CudaVBO::initGL(unsigned long sz)
{
  // delete old buffers
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  if(glVbo > 0) { glDeleteBuffers(1, &glVbo); glVbo = 0; }
  
  // OpenGL VBO
  glGenBuffers(1, &glVbo);
  glBindBuffer(GL_ARRAY_BUFFER, glVbo);
  glBufferData(GL_ARRAY_BUFFER, sz*sizeof(Vertex), NULL, GL_DYNAMIC_COPY); glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Cuda VBO
  cudaGraphicsGLRegisterBuffer(&cuVbo, glVbo, cudaGraphicsMapFlagsWriteDiscard);
}

inline void CudaVBO::destroy()
{
  if(allocated())
    {
      std::cout << "Destroying CudaVBO...\n";
      if(bound)  { release(); } if(mapped) { unmap(); }
      if(glVbo > 0) { glDeleteBuffers(1, &glVbo); glVbo = 0; }
      if(cuVbo)     { cudaGraphicsUnregisterResource(cuVbo); cuVbo = nullptr; }
      getLastCudaError("CudaVBO::destroy()");
      std::cout << "  (DONE)\n";
    }
  size = 0;
}

inline void CudaVBO::bind()
{
  if(allocated() && !bound)
    {
      // enable vertex/color arrays
      glBindBuffer(GL_ARRAY_BUFFER, glVbo);
      glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_COLOR_ARRAY);
      glVertexPointer(4, GL_FLOAT, size*sizeof(Vertex), 0);
      // glColorPointer (4, GL_UNSIGNED_BYTE, size*sizeof(Vertex), (const void*)sizeof(float3));
      bound = true;
    }
}

inline void CudaVBO::release()
{
  if(allocated() && bound)
    {
      glDisableClientState(GL_VERTEX_ARRAY); //glDisableClientState(GL_COLOR_ARRAY);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
      bound = false;
    }
}


inline void CudaVBO::draw()
{
  bind();
  glDrawArrays(GL_LINES, 0, size);
  release();
}


inline Vertex* CudaVBO::map()
{
  if(!glVbo) { std::cout << "====> WARNING(CudaVBO::map()): Buffer not initialized!\n"; return nullptr; }
  if(allocated())
    {
      if(!mapped)
        {
          int err = cudaGraphicsMapResources(1, &cuVbo, 0);
          if(err == CUDA_ERROR_ALREADY_MAPPED)
            {
              std::cout << "====> WARNING: CudaVBO already mapped! (" << err << ") --> cudaGraphicsMapResources\n";
              getLastCudaError(("CudaVBO::map() --> " + std::to_string(err) + "\n").c_str());
              return nullptr;
            }
          else if(err)
            {
              std::cout << "====> WARNING: CudaVBO failed to map! (" << err << ") --> cudaGraphicsMapResources\n";
              getLastCudaError(("CudaVBO::map() --> " + std::to_string(err) + "\n").c_str());
              mapped = false; dData = nullptr; return nullptr;
            }
          else
            {
              size_t nbytes; err = cudaGraphicsResourceGetMappedPointer((void**)&dData, &nbytes, cuVbo);
              if(err)
                {
                  std::cout << "====> WARNING: CudaVBO failed to map! (" << err << ") --> cudaGraphicsResourceGetMappedPointer\n";
                  cudaGraphicsUnmapResources(1, &cuVbo, 0);
                  getLastCudaError(("CudaVBO::map() --> " + std::to_string(err) + "\n").c_str());
                  mapped = false; dData = nullptr; return nullptr;
                }
            }
          mapped = true;
          getLastCudaError(("CudaVBO::map() -->" + std::to_string(err) + "\n").c_str());
        }
      else { std::cout << "====> WARNING: CudaVBO::map() called on mapped buffer!\n"; }
      return dData;
    }
  if(!gCudaInitialized) { std::cout << "====> WARNING(CudaVBO::map()): CUDA device not initialized!\n";  }
  if(size == 0)         { std::cout << "====> WARNING(CudaVBO::map()): zero size! " << size << "\n";     }
  if(!glVbo)            { std::cout << "====> WARNING(CudaVBO::map()): VBO resource not initialized!\n"; }
  return nullptr;
}

inline void CudaVBO::unmap()
{
  if(gCudaInitialized && size > 0 && glVbo)
    {
      if(mapped || dData)
        {
          int err = cudaGraphicsUnmapResources(1, &cuVbo, 0);
          if(err)
            {
              std::cout << "====> WARNING: CudaVBO failed to unmap (" << err << ")! --> cudaGraphicsUnmapResources\n";
              getLastCudaError("CudaVBO::unmap()\n");
              dData = nullptr; mapped = false; return;
            }
          else { getLastCudaError("CudaVBO::unmap()\n"); }
        }
      else { std::cout << "====> WARNING: CudaVBO::unmap() called on unmapped VBO!\n"; }
    }
  if(!gCudaInitialized) { std::cout << "====> WARNING(CudaVBO::unmap()): CUDA device not initialized!\n";  }
  if(size == 0)         { std::cout << "====> WARNING(CudaVBO::unmap()): zero size! " << size << "\n";     }
  if(!glVbo)            { std::cout << "====> WARNING(CudaVBO::unmap()): VBO resource not initialized!\n"; }
  dData = nullptr; mapped = false;
}




#endif // CUDA_VBO_CUH
