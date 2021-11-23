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
#include "glShader.hpp"

// X/Y/Z position and R/G/B/A color for a vertex 
struct Vertex { float3 pos; float4 color; };

//// CUDA VBO ////
// --> contains vertex data for drawing with opengl
struct CudaVBO
{
  unsigned long size = 0; // buffer size (number of vertices)
  bool   mapped = false;
  bool   bound  = false;
  Vertex *dData = nullptr;
  cudaGraphicsResource *cuVbo = nullptr;
  
  GLuint glVbo  = 0;
  GLuint glVao  = 0;
  GlShader *glShader = nullptr;

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
    else { std::cout << "====> WARNING(CudaVBO::copyTo): Buffer not allocated!\n"; }
  }
  
  // GL interop
  void initGL(unsigned long sz);  // initialize CUDA-->opengl interop
  void bind();    // bind for use with opengl
  void release(); // unbind
  void draw();    // draw vertices to screen (NOTE: probably need a shader, etc. attached)
  Vertex* map();  // vertex data mapping to device pointer for rendering via CUDA kernel
  void  unmap();  // vertex data unmapping
  void clear() { if(allocated() && map()) { cudaMemset(dData, 0, size*sizeof(Vertex)); unmap(); } }
};

inline bool CudaVBO::create(unsigned long sz)
{
  initCudaDevice();
  if(gCudaInitialized)
    {
      if(sz == this->size) { return true; } // already created
      else { destroy(); }                   // re-create
      
      std::cout << "==== Creating Cuda VBO ("  << sz << "v)... --> data: " << sz << "(v) * " << sizeof(Vertex) << "(b/v) ==> " << sz*sizeof(Vertex) << "b... ";

      // init device data
      int err = cudaMalloc((void**)&dData, sz*sizeof(Vertex));
      if(err) { std::cout << "====> ERROR: Failed to allocated memory for CudaVBO!\n"; return false; }
      err = cudaMemset(dData, 0, sz*sizeof(Vertex));
      if(err) { std::cout << "====> ERROR: Failed to initialize memory for CudaVBO!\n"; cudaFree(dData); dData = nullptr; return false; }
      // init VBO
      initGL(sz); getLastCudaError(("CudaVBO::create(<"+std::to_string(sz)+">)").c_str());
      size = sz;
      std::cout << " (done)\n";
      return true;
    }
  if(!gCudaInitialized) { std::cout << "====> WARNING(CudaVBO::create): CUDA device not initialized!\n"; }
  if(sz == 0)           { std::cout << "====> WARNING(CudaVBO::create): zero size! " << sz << "\n"; }
  if(!glVbo)            { std::cout << "====> WARNING(CudaVBO::create): VBO resource not initialized!\n"; }
  return false;
}

inline void CudaVBO::initGL(unsigned long sz)
{
  // OpenGL shader
  glShader = new GlShader("vlines.vsh", "vlines.fsh");
  
  // OpenGL VBO
  glGenBuffers(1, &glVbo);
  glBindBuffer(GL_ARRAY_BUFFER, glVbo);
  glBufferData(GL_ARRAY_BUFFER, sz*sizeof(Vertex), NULL, GL_DYNAMIC_DRAW);

  // OpenGL VAO  
  glGenVertexArrays(1, &glVao);
  glBindVertexArray(glVao);
  // glEnableVertexAttribArray(0);
  // glVertexAttribPointer( ... );

  // Cuda VBO
  cudaGraphicsGLRegisterBuffer(&cuVbo, glVbo, cudaGraphicsMapFlagsWriteDiscard);
  
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_VERTEX_ARRAY, 0);
}

inline void CudaVBO::destroy()
{
  if(allocated())
    {
      if(bound) { release(); } if(mapped) { unmap(); }
      
      std::cout << "==== Destroying CudaVBO (" << size << ")... ";
      if(glVbo > 0) { glDeleteBuffers(1, &glVbo);      glVbo = 0; }
      if(glVao > 0) { glDeleteVertexArrays(1, &glVao); glVao = 0; }
      if(cuVbo)     { cudaGraphicsUnregisterResource(cuVbo); cuVbo = nullptr; }
      if(glShader)  { delete glShader; glShader = nullptr; }
      getLastCudaError("CudaVBO::destroy()");
      std::cout << " (done)\n";
    }
  size = 0;
}

inline void CudaVBO::bind()
{
  if(allocated() && !bound)
    {
      glShader->bind();
      glBindVertexArray(glVao);
      glBindBuffer(GL_ARRAY_BUFFER, glVbo);
      bound = true;
    }
}

inline void CudaVBO::release()
{
  if(allocated() && bound)
    {
      glBindBuffer(GL_ARRAY_BUFFER, 0);
      glBindVertexArray(0);
      bound = false;
    }
}

inline void CudaVBO::draw()
{
  if(allocated())
    {
      bind();
      glDrawArrays(GL_LINES, 0, size);
      release();
    }
}

inline Vertex* CudaVBO::map()
{
  if(!glVbo) { std::cout << "====> WARNING(CudaVBO::map): Buffer not initialized!\n"; return nullptr; }
  if(allocated())
    {
      if(!mapped)
        {
          int err = cudaGraphicsMapResources(1, &cuVbo, 0);
          if(err == CUDA_ERROR_ALREADY_MAPPED)
            {
              std::cout << "====> WARNING: CudaVBO already mapped! (" << err << ") --> cudaGraphicsMapResources\n";
              getLastCudaError(("CudaVBO::map --> " + std::to_string(err) + "\n").c_str());
              return nullptr;
            }
          else if(err)
            {
              std::cout << "====> WARNING: CudaVBO failed to map! (" << err << ") --> cudaGraphicsMapResources\n";
              getLastCudaError(("CudaVBO::map --> " + std::to_string(err) + "\n").c_str());
              mapped = false; dData = nullptr; return nullptr;
            }
          else
            {
              size_t nbytes; err = cudaGraphicsResourceGetMappedPointer((void**)&dData, &nbytes, cuVbo);
              if(err)
                {
                  std::cout << "====> WARNING: CudaVBO failed to map! (" << err << ") --> cudaGraphicsResourceGetMappedPointer\n";
                  cudaGraphicsUnmapResources(1, &cuVbo, 0);
                  getLastCudaError(("CudaVBO::map --> " + std::to_string(err) + "\n").c_str());
                  mapped = false; dData = nullptr; return nullptr;
                }
            }
          mapped = true;
          getLastCudaError(("CudaVBO::map() -->" + std::to_string(err) + "\n").c_str());
        }
      else { std::cout << "====> WARNING: CudaVBO::map() called on mapped buffer!\n"; }
      return dData;
    }
  if(!gCudaInitialized) { std::cout << "====> WARNING(CudaVBO::map): CUDA device not initialized!\n";  }
  if(size == 0)         { std::cout << "====> WARNING(CudaVBO::map): zero size! " << size << "\n";     }
  if(!glVbo)            { std::cout << "====> WARNING(CudaVBO::map): VBO resource not initialized!\n"; }
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
  if(!gCudaInitialized) { std::cout << "====> WARNING(CudaVBO::unmap): CUDA device not initialized!\n";  }
  if(size == 0)         { std::cout << "====> WARNING(CudaVBO::unmap): zero size! " << size << "\n";     }
  if(!glVbo)            { std::cout << "====> WARNING(CudaVBO::unmap): VBO resource not initialized!\n"; }
  dData = nullptr; mapped = false;
}



// forward declarations
template<typename T> class FluidParams;
template<typename T> class FluidField;

// (in vlines.cu)
template<typename T> void fillVLines(FluidField<T> &src, CudaVBO &dst, FluidParams<T> &cp);




#endif // CUDA_VBO_CUH
