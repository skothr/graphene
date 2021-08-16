#ifndef RENDER_CUH
#define RENDER_CUH

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include "field.hpp"
#include "vector-operators.h"
#include "cuda-tools.h"
#include "physics.h"
#include "raytrace.h"
#include "material.h"
#include "units.hpp"


// forward declarations
typedef void* ImTextureID;
struct CudaVBO;



extern "C" void vboTest(CudaVBO &vbo, float t);


// field rendering
template<typename T> void renderFieldEM (EMField<T> &src,         CudaTexture &dst, const EmRenderParams &rp);
template<typename T> void renderFieldMat(Field<Material<T>> &src, CudaTexture &dst, const EmRenderParams &rp);
// ray marching for 3D
template<typename T> void raytraceFieldEM (EMField<T> &src, CudaTexture &dst, const Camera<double> &camera,
                                           const EmRenderParams &rp, const FieldParams<T> &cp, const Vec2d &aspect);
template<typename T> void raytraceFieldMat(EMField<T> &src, CudaTexture &dst, const Camera<double> &camera,
                                           const EmRenderParams &rp, const FieldParams<T> &cp, const Vec2d &aspect);

#endif // RENDER_CUH
