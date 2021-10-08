#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "vector.hpp"
#include "matrix.hpp"
#include "glMatrix.hpp"
#include "vector-operators.h"
#include "raytrace.h"

// lightweight camera description
template<typename T>
struct CameraDesc
{
  typedef typename DimType<T,2>::VEC_T VT2;
  typedef typename DimType<T,3>::VEC_T VT3;
  
  VT3 pos   = VT3{0.0,  0.0,  0.0}; // world position
  VT3 dir   = VT3{0.0,  1.0,  0.0}; // orthonormal basis vectors
  VT3 right = VT3{1.0,  0.0,  0.0};
  VT3 up    = VT3{0.0,  0.0, -1.0};
  
  T fov  = 60.0;    // field of view (degrees)
  T near = 0.001;   // near plane
  T far  = 10000.0; // far plane


  CameraDesc() = default;
  
  // sp --> screen pos [0.0, 1.0]
  __host__ __device__ Ray<T> castRay(const VT2 &sp, const VT2 &aspect) const
  {
    T tf2 = tan((fov/2.0)*(M_PI/180.0));
    Ray<T> ray;
    ray.pos = pos;
    ray.dir = normalize(dir +
                        right * 2.0*(sp.x-0.5)*tf2*aspect.x +  // X
                        up    * 2.0*(sp.y-0.5)*tf2*aspect.y ); // Y
    return ray;
  }
};

// full camera description (for use on host)
template<typename T>
struct Camera
{
  typedef typename DimType<T,2>::VEC_T VT2;
  typedef typename DimType<T,3>::VEC_T VT3;
  VT3 upBasis = VT3{(T)0, (T)1, (T)0};
  union
  { // camera description (union for pass-through access)
    struct { VT3 pos; VT3 dir; VT3 right; VT3 up; T fov; T near; T far; };
    CameraDesc<T> desc;
  };
  Matrix<T> view; Matrix<T> proj; Matrix<T> VP; Matrix<T> viewInv; // 4x4 transformation matrices
  Mat4f glView; Mat4f glProj; Mat4f glVP; // 4x4 transformation matrices (TODO: switch over and use OpenGL shader)

  Camera() { }
  // sp --> screen pos [0.0, 1.0]
  Ray<T> castRay(const VT2 &sp, const VT2 &aspect) const;

  Vec2f worldToView(const Vector<T, 3> &wp, const Vector<T, 2> &aspect, Vector<int, 3> *clipped=nullptr);
  Vec2f worldToView(const Vector<T, 3> &wp, const Vector<T, 2> &aspect, bool *clipped=nullptr);  

  void rotate(VT2 angles);
  
  void calculate(T aspect=1.0); // recalculate matrices
};






template<typename T>
Ray<T> Camera<T>::castRay(const VT2 &sp, const VT2 &aspect) const { return desc.castRay(VT2{sp.x, (T)1.0-sp.y}, aspect); }  


template<typename T>
Vec2f Camera<T>::worldToView(const Vector<T, 3> &wp, const Vector<T, 2> &aspect, Vector<int, 3> *clipped)
{
  Matrix<T> vp = Matrix<T>(4, 1); vp.setCol(0, {wp.x-pos.x, wp.y-pos.y, wp.z-pos.z, (T)1.0});
  Matrix<T> result = (proj^(view^vp)); // NOTE: something wrong with Matrix implementation -- result needs to be separate object
  // clip
  if(clipped)
    {
      T a = std::max(max(aspect), max(1.0/aspect));
      clipped->x = (result[0][0]/result[3][0] < -a ? -1 : (result[0][0]/result[3][0] > a ? 1 : 0));
      clipped->y = (result[1][0]/result[3][0] < -a ? -1 : (result[1][0]/result[3][0] > a ? 1 : 0));
      clipped->z = (result[2][0] > -near ? -1 : (result[2][0] < -far ? 1 : 0));
    }
  if(result[2][0] > -near ? -1 : (result[2][0] < -far ? 1 : 0)) { result[0][0] *= -1; result[1][0] *= -1; }
  // normalize
  return (Vec2f(result[0][0], -result[1][0]) / result[3][0] + 1.0) / 2.0;
}

template<typename T>
Vec2f Camera<T>::worldToView(const Vector<T, 3> &wp, const Vector<T, 2> &aspect, bool *clipped)
{
  Vec3i vClipped; Vec2f result = worldToView(wp, aspect, &vClipped);
  // clip (single bool)
  if(clipped) { *clipped = vClipped.x != 0 || vClipped.y != 0 || vClipped.z != 0; }
  return result;
}


template<typename T>
void Camera<T>::calculate(T aspect) // recalculate matrices
{
  // recalculate orthonormal bases
  dir   = normalize(dir);
  right = normalize(cross(dir, upBasis));
  up    = normalize(cross(right, dir));
    
  // create view matrix
  view.resize(4, 4);
  view.setRow(0, {right.x, right.y, right.z, 0}); //-pos.x});
  view.setRow(1, {up.x,    up.y,    up.z,    0}); //-pos.y});
  view.setRow(2, {dir.x,   dir.y,   dir.z,   0}); //-pos.z});
  view.setRow(3, {0.0,     0.0,     0.0,     1.0});
    
  // create perspective projection matrix
  proj.resize(4, 4); proj.zero();
  T S = 1.0/tan((fov/2.0)*M_PI/180.0);
  proj[0][0] = S; // X basis offset
  proj[1][1] = S; // Y basis offset
  proj[2][2] = -(far + near)/(far - near);
  proj[2][3] = -2*far*near/(far - near);
  proj[3][2] = -1.0;
  proj[3][3] = 0.0;

  // alternative matrices (TODO: get rid of other NN class)
  glView.identity();
  glView.translate(pos);
  glView = Mat4f::makeLookAt(pos, pos+dir);
  glProj = Mat4f::makeProjection(fov, 1.0, near, far);
  glVP = glProj ^ glView;
}

template<typename T>
void Camera<T>::rotate(VT2 angles) // angles --> { pitch, yaw, roll } (rotation around x, y, and z bases)
{
  if(abs(up.y) <= 0.05 && ((angles.y < 0 ? -1 : 1) != (pos.y < 0 ? -1 : 1))) { angles.y = 0.0; }
    
  VT3 newDir   = normalize(dir);
  newDir       = (angles.x == 0.0 ? newDir : normalize(to_cuda(::rotate((Vector<T, 3>)newDir, (Vector<T, 3>)upBasis,  angles.x))));
  VT3 newRight = normalize(cross(newDir, upBasis));
  newDir       = (angles.y == 0.0 ? newDir : normalize(to_cuda(::rotate((Vector<T, 3>)newDir, (Vector<T, 3>)newRight, angles.y))));
  VT3 newUp    = normalize(cross(newRight, newDir));

    
  VT3 newPos   = pos;
  newPos       = (angles.x == 0.0 ? newPos : to_cuda(::rotate((Vector<T, 3>)newPos, (Vector<T, 3>)upBasis,  angles.x)));
  VT3 right2   = normalize(cross(newPos, newUp));
  newPos       = (angles.y == 0.0 ? newPos : to_cuda(::rotate((Vector<T, 3>)newPos, (Vector<T, 3>)right2,  -angles.y)));
  dir = newDir; right = newRight; up = newUp;
  pos = newPos;
}



template<typename T>
bool isnan(const CameraDesc<T> &c) { return (isnan(c.pos) || isnan(c.dir) || isnan(c.right) || isnan(c.up) || isnan(c.fov) || isnan(c.near) || isnan(c.far)); }
template<typename T>
bool isinf(const CameraDesc<T> &c) { return (isinf(c.pos) || isinf(c.dir) || isinf(c.right) || isinf(c.up) || isinf(c.fov) || isinf(c.near) || isinf(c.far)); }
template<typename T> bool isnan(const Camera<T> &c) { return isnan(c.desc); }
template<typename T> bool isinf(const Camera<T> &c) { return isinf(c.desc); }


#endif // CAMERA_HPP
