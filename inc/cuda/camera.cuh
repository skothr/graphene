#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "vector.hpp"
#include "matrix.hpp"
#include "vector-operators.h"
#include "raytrace.h"

// lightweight camera description
template<typename T>
struct CameraDesc
{
  typedef typename cuda_vec<T,2>::VT VT2;
  typedef typename cuda_vec<T,3>::VT VT3;
  
  VT3 pos   = VT3{0.0,  0.0,  0.0}; // world position
  VT3 dir   = VT3{0.0,  1.0,  0.0}; // orthonormal basis vectors
  VT3 right = VT3{1.0,  0.0,  0.0};
  VT3 up    = VT3{0.0,  0.0, -1.0};
  
  T fov    = 60.0;    // vertical field of view (degrees)
  T aspect = 1.0;     // aspect ratio (width/height)
  T near   = 0.001;   // near plane
  T far    = 10000.0; // far plane

  CameraDesc() = default;
  
  // sp --> screen pos [0.0, 1.0]
  __host__ __device__ Ray<T> castRay(const VT2 &sp) const
  {
    T tf2 = tan((fov/2.0)*(M_PI/180.0));
    Ray<T> ray;
    ray.pos = pos;
    ray.dir = normalize(dir +                               // Z
                        right * 2.0*(sp.x-0.5)*tf2*aspect + // X
                        up    * 2.0*(sp.y-0.5)*tf2);        // Y
    return ray;
  }

  __host__ bool operator==(const CameraDesc &other) const
  {
    return (pos == other.pos && dir  == other.dir  && right == other.right && up     == other.up     &&
            fov == other.fov && near == other.near && far   == other.far   && aspect == other.aspect);
  }
};


#endif // CAMERA_CUH
