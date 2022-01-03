#ifndef RAYTRACE_H
#define RAYTRACE_H

#include "vector.hpp"
#include "vector-operators.h"


#define TOL 0.0006f // tolerance/epsilon to make sure ray fully intersects


// ray type
template<typename T, typename VT3=typename cuda_vec<T,3>::VT>
struct Ray { VT3 pos; VT3 dir; };



//// helpers ////

template<typename T, typename VT3=typename cuda_vec<T,3>::VT>
__host__  __device__ inline T planeIntersect(const VT3 &p, const VT3 &n, const Ray<T> &ray) 
{
  T denom = dot(n, ray.dir);
  return (abs(denom) > TOL ? dot((p - ray.pos), n) / denom : (T)-1);
}

// render 3D --> raytrace field
//  - field assumed to be size (1,1,1) in 3D space
//  - return value < 0 means ray missed, value == 0 means ray started inside cube
// returns {tmin, tmax}
template<typename T, typename VT2=typename cuda_vec<T,2>::VT, typename VT3=typename cuda_vec<T,3>::VT>
__device__ inline typename cuda_vec<T,2>::VT cubeIntersect(const VT3 &pos, const VT3 &size, const Ray<T> &ray)
{
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



template<typename T, typename VT2=typename cuda_vec<T,2>::VT>
__host__ inline Vector<T,2> cubeIntersectHost(const Vector<T,3> &pos, const Vector<T,3> &size, const Ray<T> &ray)
{
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

// returns face vector for first intersection (e.g. <1,0,0> for +X face)
template<typename T, typename VT3=Vector<T,3>>
__host__ inline VT3 cubeIntersectFace(const VT3 &pos, const VT3 &size, const Ray<T> &ray)
{
  T tnx = (pos.x - ray.pos.x)          / ray.dir.x;
  T tpx = (pos.x - ray.pos.x + size.x) / ray.dir.x;
  T tny = (pos.y - ray.pos.y)          / ray.dir.y;
  T tpy = (pos.y - ray.pos.y + size.y) / ray.dir.y;
  T tnz = (pos.z - ray.pos.z)          / ray.dir.z;
  T tpz = (pos.z - ray.pos.z + size.z) / ray.dir.z;
  T tmin = std::max(std::max(std::min(tnx, tpx), std::min(tny, tpy)), std::min(tnz, tpz));
  T tmax = std::min(std::min(std::max(tnx, tpx), std::max(tny, tpy)), std::max(tnz, tpz));
  return (VT3{(tmin == tpx ? (T)1.0 : (tmin == tnx ? (T)-1.0 : (T)0.0)),
              (tmin == tpy ? (T)1.0 : (tmin == tny ? (T)-1.0 : (T)0.0)),
              (tmin == tpz ? (T)1.0 : (tmin == tnz ? (T)-1.0 : (T)0.0))});
}




#endif // RAYTRACE_H
