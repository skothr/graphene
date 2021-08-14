#ifndef RAYTRACE_H
#define RAYTRACE_H

#include "vector.hpp"
#include "matrix.hpp"
#include "vector-operators.h"


// ray type
template<typename T> struct Ray
{
  typedef typename DimType<T,3>::VECTOR_T VT3;
  VT3 pos; VT3 dir;
};

// simplified camera description
template<typename T>
struct CameraDesc
{
  typedef typename DimType<T,3>::VECTOR_T VT3;

  VT3 pos; // world position
  
  // orthonormal basis vectors
  VT3 dir;
  VT3 right;
  VT3 up;
  T fov  = 60.0;   // field of view (degrees)
  
  T near = 0.01;   // near plane
  T far  = 1000.0; // far plane
};

// full camera description (for use on host)
template<typename T>
struct Camera
{
  typedef typename DimType<T,2>::VECTOR_T VT2;
  typedef typename DimType<T,3>::VECTOR_T VT3;
  
  union
  { // camera description (union for easy access)
    CameraDesc<T> desc;
    struct { VT3 pos; VT3 dir; VT3 right; VT3 up; T fov; T near; T far; };
  };
  Matrix<T> view; Matrix<T> viewT; Matrix<T> proj; Matrix<T> VP;   // 4x4 transformation matrices
  Matrix<T> viewInv;

  VT3 focus   = VT3{(T)0, (T)0, (T)0};
  VT3 upBasis = VT3{(T)0, (T)1, (T)0};
  
  Camera() : view(4,4), viewT(4,4), proj(4,4), VP(4,4), viewInv(4,4) { view.identity(); viewT.identity(); proj.zero(); VP.identity(); viewInv.identity(); }

  Vec2f worldToView(const Vector<T, 3> &wp, bool *clipped=nullptr)
  {
    // Vec4d wp4 = Vec4d(wp.x, wp.y, wp.z, 1.0f);
    Matrix<T> vp  = proj ^ (view ^ viewT ^ Matrix<T>({wp.x, wp.y, wp.z, (T)1.0}));
    // clip
    if(clipped) { *clipped = (vp[0][3] < 0.0); }
    // normalize
    return (Vec2f(-vp[0][0], vp[1][0]) / vp[3][0] + 1) / 2;
  }
  // Vec3d viewToWorld(const Vec3d &wp)  // TODO -- inverse matrices
  // {
  //   Vec4d wp4 = Vec4d(wp.x, wp.y, wp.z, 1.0f);
  //   Vec4d vp  = viewInv ^ wp;
  //   return Vec3d(vp.x, vp.y, vp.z);
  // }
  
  void calculate() // recalculate matrices
  {
    // recalculate orthonormal bases
    dir   = normalize(dir);
    right = normalize(cross(dir, upBasis));
    up    = normalize(cross(right, dir));
    
    // create view matrix
    view.identity(); viewInv.identity();
    view.setRow(0, {right.x, right.y, right.z,  1}); //0.0}); //viewInv.setRow(0, {right.x, right.y, right.z, 0.0});
    view.setRow(1, {up.x,    up.y,    up.z,     1}); //0.0}); //viewInv.setRow(1, {up.x,    up.y,    up.z,    0.0});
    view.setRow(2, {dir.x,   dir.y,   dir.z,    1}); //0.0}); //viewInv.setRow(2, {dir.x,   dir.y,   dir.z,   0.0});

    // translate to camera position
    viewT.identity();
    viewT[0][3] = pos.x; viewT[1][3] = pos.y; viewT[2][3] = pos.z;

    // view = view ^ viewT;
    
    // create perspective projection matrix
    proj.zero();
    T S = 1.0/tan((fov/2.0)*M_PI/180.0);
    proj[0][0] = S; // X basis offset
    proj[1][1] = S; // Y basis offset
    proj[2][2] = -(far + near)/(far - near);
    proj[2][3] = -2*far*near/(far - near);
    proj[3][2] = -1.0;

    // std::cout << "VIEW:\n"  << view.toString() << "\n"
    //           << "VIEWT:\n" << viewT.toString() << "\n"
    //           << "PROJ:\n"  << proj.toString() << "\n";
    
    // combine matrices
    VP = proj ^ (view);// ^ viewT);
    // std::cout << "VP:\n" << VP.toString() << "\n\n";
  }

  void rotate(VT2 angles) // angles --> { pitch, yaw, roll } (rotation around x, y, and z bases)
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
  
};

#endif // RAYTRACE_H
