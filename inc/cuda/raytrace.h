#ifndef RAYTRACE_H
#define RAYTRACE_H

#include "vector.hpp"
#include "vector-operators.h"


#define TOL 0.0006f // tolerance/epsilon to make sure ray fully intersects

// ray type
template<typename T> struct Ray
{
  typedef typename DimType<T,3>::VEC_T VT3;
  VT3 pos; VT3 dir;
};


//// CUDA helpers ////

template<typename T>
__host__  __device__ inline T planeIntersect(const typename DimType<T,3>::VEC_T &p,
                                             const typename DimType<T,3>::VEC_T &n, const Ray<T> &ray) 
{
  T denom = dot(n, ray.dir);
  return (abs(denom) > TOL ? dot((p - ray.pos), n) / denom : (T)-1);
  // T t = -1.0;
  // if { t = dot((p - ray.pos), n) / denom; }
  // return t; // no intersection if t < 0
}

// render 3D --> raytrace field
//  - field assumed to be size (1,1,1) in 3D space
//  - return value < 0 means ray missed, value == 0 means ray started inside cube
// returns {tmin, tmax}
template<typename T>
__device__ inline typename DimType<T,2>::VEC_T cubeIntersect(const typename DimType<T,3>::VEC_T &pos,
                                                             const typename DimType<T,3>::VEC_T &size, const Ray<T> &ray)
{
  typedef typename DimType<T,2>::VEC_T VT2;
  typedef typename DimType<T,3>::VEC_T VT3;
  T tnx = (pos.x - ray.pos.x)          / ray.dir.x;
  T tpx = (pos.x - ray.pos.x + size.x) / ray.dir.x;
  T tny = (pos.y - ray.pos.y)          / ray.dir.y;
  T tpy = (pos.y - ray.pos.y + size.y) / ray.dir.y;
  T tnz = (pos.z - ray.pos.z)          / ray.dir.z;
  T tpz = (pos.z - ray.pos.z + size.z) / ray.dir.z;
  T tmin = max(max(min((T)tnx, (T)tpx), min((T)tny, (T)tpy)), min((T)tnz, (T)tpz));
  T tmax = min(min(max((T)tnx, (T)tpx), max((T)tny, (T)tpy)), max((T)tnz, (T)tpz));
  return (tmin < 0 ? VT2{0,tmax} : (tmin > tmax+TOL) ? VT2{-1.0,-1.0} : VT2{tmin, tmax});
}



template<typename T>
__host__ inline Vector<T,2> cubeIntersectHost(const Vector<T,3> &pos, const Vector<T,3> &size, const Ray<T> &ray)
{
  typedef typename DimType<T,2>::VEC_T VT2;
  typedef typename DimType<T,3>::VEC_T VT3;
  T tnx = (pos.x - ray.pos.x)          / ray.dir.x;
  T tpx = (pos.x - ray.pos.x + size.x) / ray.dir.x;
  T tny = (pos.y - ray.pos.y)          / ray.dir.y;
  T tpy = (pos.y - ray.pos.y + size.y) / ray.dir.y;
  T tnz = (pos.z - ray.pos.z)          / ray.dir.z;
  T tpz = (pos.z - ray.pos.z + size.z) / ray.dir.z;
  T tmin = std::max(std::max(std::min((T)tnx, (T)tpx), std::min((T)tny, (T)tpy)), std::min((T)tnz, (T)tpz));
  T tmax = std::min(std::min(std::max((T)tnx, (T)tpx), std::max((T)tny, (T)tpy)), std::max((T)tnz, (T)tpz));
  return (tmin < 0 ? VT2{0,tmax} : (tmin > tmax+TOL) ? VT2{-1.0,-1.0} : VT2{tmin, tmax});
}




#endif // RAYTRACE_H
