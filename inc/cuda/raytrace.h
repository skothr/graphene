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
  typedef typename DimType<T,2>::VECTOR_T VT2;
  typedef typename DimType<T,3>::VECTOR_T VT3;


  VT3 pos = VT3{0.0, 0.0, 0.0}; // world position
  
  // orthonormal basis vectors
  VT3 dir   = VT3{0.0,  1.0,  0.0};
  VT3 right = VT3{1.0,  0.0,  0.0};
  VT3 up    = VT3{0.0,  0.0, -1.0};
  
  T fov  = 60.0;   // field of view (degrees)
  T near = 0.01;   // near plane
  T far  = 1000.0; // far plane
  
  // sp --> screen pos [0.0, 1.0]
  __host__ __device__ Ray<T> castRay(const VT2 &sp, const VT2 &aspect) const
  {
    T tf2 = tan(fov/2.0*(M_PI/180.0));
    Ray<T> ray;
    ray.pos = pos;
    ray.dir = normalize(dir +
                        right * 2.0*(sp.x-0.5)*tf2*aspect.x +
                        up    * 2.0*(sp.y-0.5)*tf2*aspect.y );
    return ray;
  }  
};

// full camera description (for use on host)
template<typename T>
struct Camera
{
  typedef typename DimType<T,2>::VECTOR_T VT2;
  typedef typename DimType<T,3>::VECTOR_T VT3;

  VT3 upBasis = VT3{(T)0, (T)1, (T)0};
  union
  { // camera description (union for easy access)
    struct { VT3 pos; VT3 dir; VT3 right; VT3 up; T fov; T near; T far; };
    CameraDesc<T> desc;
  };
  Matrix<T> view; Matrix<T> viewT; Matrix<T> proj; Matrix<T> VP;   // 4x4 transformation matrices
  Matrix<T> viewInv;

  Camera() { }
  
  // sp --> screen pos [0.0, 1.0]
  Ray<T> castRay(const VT2 &sp, const VT2 &aspect) const { return desc.castRay(VT2{sp.x, (T)1.0-sp.y}, aspect); }  

  Vec2f worldToView(const Vector<T, 3> &wp, const Vector<T, 2> &aspect, Vector<int, 3> *clipped=nullptr)
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
  Vec2f worldToView(const Vector<T, 3> &wp, const Vector<T, 2> &aspect, bool *clipped=nullptr)
  {
    Vec3i vClipped; Vec2f result = worldToView(wp, aspect, &vClipped);
    // clip (single bool)
    if(clipped) { *clipped = vClipped.x != 0 || vClipped.y != 0 || vClipped.z != 0; }
    return result;
  }
  // Vector<T, 3> viewToWorld(const Vector<T, 3> &wp)  // TODO -- inverse matrices (?)
  // {
  //   Vector<T, 4> wp4 = Vector<T, 4>(wp.x, wp.y, wp.z, 1.0f);
  //   Vector<T, 4> result  = viewInv ^ wp;
  //   return Vector<T, 3>(vp.x, vp.y, vp.z);
  // }
  
  void calculate() // recalculate matrices
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





#define TOL 0.0005f // tolerance/epsilon to make sure ray fully intersects

template<typename T>
__host__  __device__ inline T planeIntersect(const typename DimType<T,3>::VECTOR_T &p,
                                             const typename DimType<T,3>::VECTOR_T &n, const Ray<T> &ray) 
{
  T denom = dot(n, ray.dir);
  T t = -1.0;
  if(abs(denom) > TOL) { t = dot((p - ray.pos), n) / denom; }
  return t; // no intersection if t < 0
}

// render 3D --> raytrace field
//  - field assumed to be size (1,1,1) in 3D space
//  - return value < 0 means ray missed, value == 0 means ray started inside cube
// returns {tmin, tmax}
template<typename T> __device__ inline typename DimType<T,2>::VECTOR_T cubeIntersect(const typename DimType<T,3>::VECTOR_T &pos,
                                                                                     const typename DimType<T,3>::VECTOR_T &size, const Ray<T> &ray)
{
  typedef typename DimType<T,2>::VECTOR_T VT2;
  typedef typename DimType<T,3>::VECTOR_T VT3;
  T tnx = (pos.x - ray.pos.x)          / ray.dir.x;
  T tpx = (pos.x - ray.pos.x + size.x) / ray.dir.x;
  T tny = (pos.y - ray.pos.y)          / ray.dir.y;
  T tpy = (pos.y - ray.pos.y + size.y) / ray.dir.y;
  T tnz = (pos.z - ray.pos.z)          / ray.dir.z;
  T tpz = (pos.z - ray.pos.z + size.z) / ray.dir.z;
  T tmin = max(max(min((T)tnx, (T)tpx), min((T)tny, (T)tpy)), min((T)tnz, (T)tpz));
  T tmax = min(min(max((T)tnx, (T)tpx), max((T)tny, (T)tpy)), max((T)tnz, (T)tpz));
  return (tmin < 0 ? VT2{0,0} : (tmin > tmax) ? VT2{-1.0,-1.0} : VT2{tmin, tmax});
}



template<typename T>
__host__ inline Vector<T,2> cubeIntersectHost(const Vector<T,3> &pos, const Vector<T,3> &size, const Ray<T> &ray)
{
  typedef typename DimType<T,2>::VECTOR_T VT2;
  typedef typename DimType<T,3>::VECTOR_T VT3;
  T tnx = (pos.x - ray.pos.x)          / ray.dir.x;
  T tpx = (pos.x - ray.pos.x + size.x) / ray.dir.x;
  T tny = (pos.y - ray.pos.y)          / ray.dir.y;
  T tpy = (pos.y - ray.pos.y + size.y) / ray.dir.y;
  T tnz = (pos.z - ray.pos.z)          / ray.dir.z;
  T tpz = (pos.z - ray.pos.z + size.z) / ray.dir.z;
  T tmin = std::max(std::max(std::min((T)tnx, (T)tpx), std::min((T)tny, (T)tpy)), std::min((T)tnz, (T)tpz));
  T tmax = std::min(std::min(std::max((T)tnx, (T)tpx), std::max((T)tny, (T)tpy)), std::max((T)tnz, (T)tpz));
  return (tmin < 0 ? VT2{0,0} : (tmin > tmax) ? VT2{-1.0,-1.0} : VT2{tmin, tmax});
}




#endif // RAYTRACE_H
