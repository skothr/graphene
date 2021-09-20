#ifndef CUDA_VECTOR_OPERATORS_H
#define CUDA_VECTOR_OPERATORS_H

#include <cuda_runtime.h>
#include <vector_types.h>
#include <ostream>
#include <cmath>
#include <float.h>

#include "vector.hpp"

//// COMPILE-TIME TEMPLATES FOR GENERALIZED DIMENSIONALITY ////
template<typename T> struct Dim { static constexpr int N = 1; typedef T   BASE_T;    typedef T       LOWER; typedef int  SIZE_T; };
template<> struct Dim<int2>     { static constexpr int N = 2; typedef int BASE_T;    typedef int     LOWER; typedef int2 SIZE_T; };
template<> struct Dim<int3>     { static constexpr int N = 3; typedef int BASE_T;    typedef int2    LOWER; typedef int3 SIZE_T; };
template<> struct Dim<int4>     { static constexpr int N = 4; typedef int BASE_T;    typedef int3    LOWER; typedef int4 SIZE_T; };
                                                                                                                      
template<> struct Dim<float2>   { static constexpr int N = 2; typedef float BASE_T;  typedef float   LOWER; typedef int2 SIZE_T; };
template<> struct Dim<float3>   { static constexpr int N = 3; typedef float BASE_T;  typedef float2  LOWER; typedef int3 SIZE_T; };
template<> struct Dim<float4>   { static constexpr int N = 4; typedef float BASE_T;  typedef float3  LOWER; typedef int4 SIZE_T; };

template<> struct Dim<double2>  { static constexpr int N = 2; typedef double BASE_T; typedef double  LOWER; typedef int2 SIZE_T; };
template<> struct Dim<double3>  { static constexpr int N = 3; typedef double BASE_T; typedef double2 LOWER; typedef int3 SIZE_T; };
template<> struct Dim<double4>  { static constexpr int N = 4; typedef double BASE_T; typedef double3 LOWER; typedef int4 SIZE_T; };

// NOTE: S --> scalar base type (int/float/double)
template<typename S, int D> struct Dim<Vector<S, D>>
{
  static constexpr int N = D;
  typedef typename std::conditional<D==1, S, Vector<S, D-1>>::type LOWER;
  typedef S BASE_T;
  typedef int SIZE_T;
};
template<typename S> struct Dim<Vector<S, 2>>{ static constexpr int N = 2; typedef Vector<S, 1> LOWER;   typedef S BASE_T; typedef int2 SIZE_T; };
template<typename S> struct Dim<Vector<S, 3>>{ static constexpr int N = 3; typedef Vector<S, 2> LOWER;   typedef S BASE_T; typedef int3 SIZE_T; };
template<typename S> struct Dim<Vector<S, 4>>{ static constexpr int N = 4; typedef Vector<S, 3> LOWER;   typedef S BASE_T; typedef int4 SIZE_T; };

// CONVERT SCALAR TYPE AND INT TO CUDA VECTOR STRUCT
template<typename T, int N> struct DimType { typedef T VEC_T; typedef int  SIZE_T; };
template<> struct DimType<int,    1> { typedef int     VEC_T; typedef int  SIZE_T; };
template<> struct DimType<int,    2> { typedef int2    VEC_T; typedef int2 SIZE_T; };
template<> struct DimType<int,    3> { typedef int3    VEC_T; typedef int3 SIZE_T; };
template<> struct DimType<int,    4> { typedef int4    VEC_T; typedef int4 SIZE_T; };
template<> struct DimType<float,  1> { typedef float   VEC_T; typedef int  SIZE_T; };
template<> struct DimType<float,  2> { typedef float2  VEC_T; typedef int2 SIZE_T; };
template<> struct DimType<float,  3> { typedef float3  VEC_T; typedef int3 SIZE_T; };
template<> struct DimType<float,  4> { typedef float4  VEC_T; typedef int4 SIZE_T; };
template<> struct DimType<double, 1> { typedef double  VEC_T; typedef int  SIZE_T; };
template<> struct DimType<double, 2> { typedef double2 VEC_T; typedef int2 SIZE_T; };
template<> struct DimType<double, 3> { typedef double3 VEC_T; typedef int3 SIZE_T; };
template<> struct DimType<double, 4> { typedef double4 VEC_T; typedef int4 SIZE_T; };

// CREATE VECTOR TYPES (T-->output | U-->input)
//   convert scalar to vector
template<typename T, typename U=typename Dim<T>::BASE_T> __host__ __device__ inline T makeV(U s);
template<> __host__ __device__ inline int     makeV<int,     int>   (int    s) { return s; }
template<> __host__ __device__ inline int2    makeV<int2,    int>   (int    s) { return int2   {s, s}; }
template<> __host__ __device__ inline int3    makeV<int3,    int>   (int    s) { return int3   {s, s, s}; }
template<> __host__ __device__ inline int4    makeV<int4,    int>   (int    s) { return int4   {s, s, s, s}; }
template<> __host__ __device__ inline float   makeV<float,   float> (float  s) { return s; }
template<> __host__ __device__ inline float2  makeV<float2,  float> (float  s) { return float2 {s, s}; }
template<> __host__ __device__ inline float3  makeV<float3,  float> (float  s) { return float3 {s, s, s}; }
template<> __host__ __device__ inline float4  makeV<float4,  float> (float  s) { return float4 {s, s, s, s}; }
template<> __host__ __device__ inline double  makeV<double,  double>(double s) { return s; }
template<> __host__ __device__ inline double2 makeV<double2, double>(double s) { return double2{s, s}; }
template<> __host__ __device__ inline double3 makeV<double3, double>(double s) { return double3{s, s, s}; }
template<> __host__ __device__ inline double4 makeV<double4, double>(double s) { return double4{s, s, s, s}; }// convert int vector to float/double vector

template<> __host__ __device__ inline float2  makeV<float2,  int> (int  s) { return float2 {(float) s, (float) s}; }
template<> __host__ __device__ inline float3  makeV<float3,  int> (int  s) { return float3 {(float) s, (float) s, (float) s}; }
template<> __host__ __device__ inline float4  makeV<float4,  int> (int  s) { return float4 {(float) s, (float) s, (float) s, (float) s}; }
template<> __host__ __device__ inline double2 makeV<double2, int> (int  s) { return double2{(double)s, (double)s}; }
template<> __host__ __device__ inline double3 makeV<double3, int> (int  s) { return double3{(double)s, (double)s, (double)s}; }
template<> __host__ __device__ inline double4 makeV<double4, int> (int  s) { return double4{(double)s, (double)s, (double)s, (double)s}; }

template<> __host__ __device__ inline float2  makeV<float2,  int2>(int2 v) { return float2 {(float) v.x, (float) v.y}; }
template<> __host__ __device__ inline float3  makeV<float3,  int3>(int3 v) { return float3 {(float) v.x, (float) v.y, (float) v.z}; }
template<> __host__ __device__ inline float4  makeV<float4,  int4>(int4 v) { return float4 {(float) v.x, (float) v.y, (float) v.z, (float) v.w}; }
template<> __host__ __device__ inline double2 makeV<double2, int2>(int2 v) { return double2{(double)v.x, (double)v.y}; }
template<> __host__ __device__ inline double3 makeV<double3, int3>(int3 v) { return double3{(double)v.x, (double)v.y, (double)v.z}; }
template<> __host__ __device__ inline double4 makeV<double4, int4>(int4 v) { return double4{(double)v.x, (double)v.y, (double)v.z, (double)v.w}; }
// convert float/double vector to int vector
template<> __host__ __device__ inline int2    makeV<int2,    float2> (float2  v) { return int2   {(int)   v.x, (int)   v.y}; }
template<> __host__ __device__ inline int3    makeV<int3,    float3> (float3  v) { return int3   {(int)   v.x, (int)   v.y, (int)   v.z}; }
template<> __host__ __device__ inline int4    makeV<int4,    float4> (float4  v) { return int4   {(int)   v.x, (int)   v.y, (int)   v.z, (int)   v.w}; }
template<> __host__ __device__ inline int2    makeV<int2,    double2>(double2 v) { return int2   {(int)   v.x, (int)   v.y}; }
template<> __host__ __device__ inline int3    makeV<int3,    double3>(double3 v) { return int3   {(int)   v.x, (int)   v.y, (int)   v.z}; }
template<> __host__ __device__ inline int4    makeV<int4,    double4>(double4 v) { return int4   {(int)   v.x, (int)   v.y, (int)   v.z, (int)   v.w}; }
template<> __host__ __device__ inline float2  makeV<float2,  double2>(double2 v) { return float2 {(float) v.x, (float) v.y}; }
template<> __host__ __device__ inline float3  makeV<float3,  double3>(double3 v) { return float3 {(float) v.x, (float) v.y, (float) v.z}; }
template<> __host__ __device__ inline float4  makeV<float4,  double4>(double4 v) { return float4 {(float) v.x, (float) v.y, (float) v.z, (float) v.w}; }
template<> __host__ __device__ inline double2 makeV<double2, float2> (float2  v) { return double2{(double)v.x, (double)v.y}; }
template<> __host__ __device__ inline double3 makeV<double3, float3> (float3  v) { return double3{(double)v.x, (double)v.y, (double)v.z}; }
template<> __host__ __device__ inline double4 makeV<double4, float4> (float4  v) { return double4{(double)v.x, (double)v.y, (double)v.z, (double)v.w}; }

// alternative templating test(s)...
//
// template<typename T, typename U=typename Dim<T>::BASE_T, std::enable_if<Dim<T>::N==1, bool> = true> // N = 1
// __host__ __device__  inline T makeV(U s) { return T((Dim<T>::BASE_T)s); }
// template<typename T, typename U=typename Dim<T>::BASE_T, std::enable_if<Dim<T>::N==1, bool> = true> // N = 1
// __host__ __device__  inline T makeV(U s) { return T((Dim<T>::BASE_T)s); }
// template<typename T, typename U=typename Dim<T>::BASE_T, std::enable_if<Dim<T>::N==2, bool> = true> // N = 2
// __host__ __device__  inline T makeV(U s) { return T{(Dim<T>::BASE_T)s, (Dim<T>::BASE_T)s}; }
// template<typename T, typename U=typename Dim<T>::BASE_T, std::enable_if<Dim<T>::N==3, bool> = true> // N = 3
// __host__ __device__  inline T makeV(U s) { return T{(Dim<T>::BASE_T)s, (Dim<T>::BASE_T)s, (Dim<T>::BASE_T)s}; }
// template<typename T, typename U=typename Dim<T>::BASE_T, std::enable_if<Dim<T>::N==4, bool> = true> // N = 4
// __host__ __device__  inline T makeV(U s) { return T{(Dim<T>::BASE_T)s, (Dim<T>::BASE_T)s, (Dim<T>::BASE_T)s, (Dim<T>::BASE_T)s}; }
//
//// ? should (maybe) work with C++17/C++20...?
// template<typename T, typename U=typename Dim<T>::BASE_T>
// __host__ __device__ inline T makeV(U s)
// {
//   typedef typename Dim<T>::BASE_T TB;
//   typedef typename DimType<TB, 2>::VEC_T TV2;
//   typedef typename DimType<TB, 3>::VEC_T TV3;
//   typedef typename DimType<TB, 4>::VEC_T TV4;
//   typedef typename Dim<U>::BASE_T UB;
//   typedef typename DimType<UB, 2>::VEC_T UV2;
//   typedef typename DimType<UB, 3>::VEC_T UV3;
//   typedef typename DimType<UB, 4>::VEC_T UV4;
//   if constexpr    (std::is_same_v<T, TV2>) // T --> 2D vector
//     { if constexpr(std::is_same_v<U, UB>)  { return T{(TB)s, (TB)s}; }              // U --> scalar
//       else if     (std::is_same_v<U, UV2>) { return T{(TB)s.x, (TB)s.y}; }          // U --> 2D vector
//       else                                 { return T(); } }
//   else if         (std::is_same_v<T, TV3>) // T --> 3D vector
//     { if constexpr(std::is_same_v<U, UB>)  { return T{(TB)s, (TB)s, (TB)s}; }       // U --> scalar   
//       else if     (std::is_same_v<U, UV3>) { return T{(TB)s.x, (TB)s.y, (TB)s.z}; } // U --> 3D vector
//       else                                 { return T(); } }
//   else if         (std::is_same_v<T, TB>)  // T --> scalar
//     { if constexpr(std::is_same_v<U, UB>)  { return (TB)s; }                        // U --> scalar
//       else                                 { return T(); } }
//   // else if(std::is_same_v<T, TV4>)       // T --> 4D vector
//   //   { if constexpr(std::is_same_v<U, UB>)  { return T{(TB)s, (TB)s, (TB)s, (TB)s}; }           // U --> scalar   
//   //     else if     (std::is_same_v<U, UV4>) { return T{(TB)s.x, (TB)s.y, (TB)s.z, (TB)s.w}; } } // U --> 4D vector
//   //else { return T(s); }
//   return T();
// }




// INDEX TYPES
template<typename T> __host__ __device__  inline typename DimType<typename Dim<T>::BASE_T, Dim<T>::N>::SIZE_T makeI(int sx, int sy, int sz);
template<> __host__ __device__ inline int2 makeI<int2>    (int sx, int sy, int sz) { return int2{sx, sy}; }
template<> __host__ __device__ inline int3 makeI<int3>    (int sx, int sy, int sz) { return int3{sx, sy, sz}; }
template<> __host__ __device__ inline int4 makeI<int4>    (int sx, int sy, int sz) { return int4{sx, sy, sz, 0}; }
template<> __host__ __device__ inline int2 makeI<float2>  (int sx, int sy, int sz) { return int2{(int)sx, (int)sy}; }
template<> __host__ __device__ inline int3 makeI<float3>  (int sx, int sy, int sz) { return int3{(int)sx, (int)sy, (int)sz}; }
template<> __host__ __device__ inline int4 makeI<float4>  (int sx, int sy, int sz) { return int4{(int)sx, (int)sy, (int)sz, 0}; }
template<> __host__ __device__ inline int2 makeI<double2> (int sx, int sy, int sz) { return int2{(int)sx, (int)sy}; }
template<> __host__ __device__ inline int3 makeI<double3> (int sx, int sy, int sz) { return int3{(int)sx, (int)sy, (int)sz}; }
template<> __host__ __device__ inline int4 makeI<double4> (int sx, int sy, int sz) { return int4{(int)sx, (int)sy, (int)sz, 0}; }


// ostream
__host__ inline std::ostream& operator<<(std::ostream &os, const int2    &v) { os << "<" << v.x << ", " << v.y << ">"; return os; }
__host__ inline std::ostream& operator<<(std::ostream &os, const int3    &v) { os << "<" << v.x << ", " << v.y << ", " << v.z << ">"; return os; }
__host__ inline std::ostream& operator<<(std::ostream &os, const int4    &v) { os << "<" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ">"; return os; }
__host__ inline std::ostream& operator<<(std::ostream &os, const float2  &v) { os << "<" << v.x << ", " << v.y << ">"; return os; }
__host__ inline std::ostream& operator<<(std::ostream &os, const float3  &v) { os << "<" << v.x << ", " << v.y << ", " << v.z << ">"; return os; }
__host__ inline std::ostream& operator<<(std::ostream &os, const float4  &v) { os << "<" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ">"; return os; }
__host__ inline std::ostream& operator<<(std::ostream &os, const double2 &v) { os << "<" << v.x << ", " << v.y << ">"; return os; }
__host__ inline std::ostream& operator<<(std::ostream &os, const double3 &v) { os << "<" << v.x << ", " << v.y << ", " << v.z << ">"; return os; }
__host__ inline std::ostream& operator<<(std::ostream &os, const double4 &v) { os << "<" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ">"; return os; }
// istream
static std::string junk;
__host__ inline std::istream& operator>>(std::istream &is, int2    &v)
{ char c; is.get(c); if(c=='<') { is >> v.x; is.ignore(1,','); is >> v.y;
    is.ignore('>'); } else { v = int2{0,0}; }                   return is; }
__host__ inline std::istream& operator>>(std::istream &is, int3    &v)
{ char c; is.get(c); if(c=='<') { is >> v.x; is.ignore(1,','); is >> v.y; is.ignore(1,','); is >> v.z;
    is.ignore('>'); } else { v = int3{0,0,0}; }                 return is; }
__host__ inline std::istream& operator>>(std::istream &is, int4    &v)
{ char c; is.get(c); if(c=='<') { is >> v.x; is.ignore(1,','); is >> v.y; is.ignore(1,','); is >> v.z; is.ignore(1,','); is >> v.w;
    is.ignore('>'); } else { v = int4{0,0,0,0}; }               return is; }
__host__ inline std::istream& operator>>(std::istream &is, float2  &v)
{ char c; is.get(c); if(c=='<') { is >> v.x; is.ignore(1,','); is >> v.y;
    is.ignore('>'); } else { v = float2{0.0f,0.0f}; }           return is; }
__host__ inline std::istream& operator>>(std::istream &is, float3  &v)
{ char c; is.get(c); if(c=='<') { is >> v.x; is.ignore(1,','); is >> v.y; is.ignore(1,','); is >> v.z;
    is.ignore('>'); } else { v = float3{0.0f,0.0f,0.0f}; }      return is; }
__host__ inline std::istream& operator>>(std::istream &is, float4  &v)
{ char c; is.get(c); if(c=='<') { is >> v.x; is.ignore(1,','); is >> v.y; is.ignore(1,','); is >> v.z; is.ignore(1,','); is >> v.w;
    is.ignore('>'); } else { v = float4{0.0f,0.0f,0.0f,0.0f}; } return is; }
__host__ inline std::istream& operator>>(std::istream &is, double2 &v)
{ char c; is.get(c); if(c=='<') { is >> v.x; is.ignore(1,','); is >> v.y;
    is.ignore('>'); } else { v = double2{0.0,0.0}; }            return is; }
__host__ inline std::istream& operator>>(std::istream &is, double3 &v)
{ char c; is.get(c); if(c=='<') { is >> v.x; is.ignore(1,','); is >> v.y; is.ignore(1,','); is >> v.z;
    is.ignore('>'); } else { v = double3{0.0,0.0,0.0}; }        return is; }
__host__ inline std::istream& operator>>(std::istream &is, double4 &v)
{ char c; is.get(c); if(c=='<') { is >> v.x; is.ignore(1,','); is >> v.y; is.ignore(1,','); is >> v.z; is.ignore(1,','); is >> v.w;
    is.ignore('>'); } else { v = double4{0.0,0.0,0.0,0.0}; }    return is; }

// NEGATION
__host__ __device__  inline int2    operator-(const int2    &u) { return int2   {-u.x, -u.y}; }
__host__ __device__  inline int3    operator-(const int3    &u) { return int3   {-u.x, -u.y, -u.z}; }
__host__ __device__  inline int4    operator-(const int4    &u) { return int4   {-u.x, -u.y, -u.z, -u.w}; }
__host__ __device__  inline float2  operator-(const float2  &u) { return float2 {-u.x, -u.y}; }
__host__ __device__  inline float3  operator-(const float3  &u) { return float3 {-u.x, -u.y, -u.z}; }
__host__ __device__  inline float4  operator-(const float4  &u) { return float4 {-u.x, -u.y, -u.z, -u.w}; }
__host__ __device__  inline double2 operator-(const double2 &u) { return double2{-u.x, -u.y}; }
__host__ __device__  inline double3 operator-(const double3 &u) { return double3{-u.x, -u.y, -u.z}; }
__host__ __device__  inline double4 operator-(const double4 &u) { return double4{-u.x, -u.y, -u.z, -u.w}; }

// VECTOR + VECTOR
__host__ __device__  inline int2     operator+(const int2    &u, const int2    &v) { return int2   {u.x+v.x, u.y+v.y}; }
__host__ __device__  inline int3     operator+(const int3    &u, const int3    &v) { return int3   {u.x+v.x, u.y+v.y, u.z+v.z}; }
__host__ __device__  inline int4     operator+(const int4    &u, const int4    &v) { return int4   {u.x+v.x, u.y+v.y, u.z+v.z, u.w+v.w}; }
__host__ __device__  inline float2   operator+(const float2  &u, const float2  &v) { return float2 {u.x+v.x, u.y+v.y}; }
__host__ __device__  inline float3   operator+(const float3  &u, const float3  &v) { return float3 {u.x+v.x, u.y+v.y, u.z+v.z}; }
__host__ __device__  inline float4   operator+(const float4  &u, const float4  &v) { return float4 {u.x+v.x, u.y+v.y, u.z+v.z, u.w+v.w}; }
__host__ __device__  inline double2  operator+(const double2 &u, const double2 &v) { return double2{u.x+v.x, u.y+v.y}; }
__host__ __device__  inline double3  operator+(const double3 &u, const double3 &v) { return double3{u.x+v.x, u.y+v.y, u.z+v.z}; }
__host__ __device__  inline double4  operator+(const double4 &u, const double4 &v) { return double4{u.x+v.x, u.y+v.y, u.z+v.z, u.w+v.w}; }
__host__ __device__  inline int2&    operator+=(int2    &u, const int2    &v)  { u = u + v; return u; }
__host__ __device__  inline int3&    operator+=(int3    &u, const int3    &v)  { u = u + v; return u; }
__host__ __device__  inline int4&    operator+=(int4    &u, const int4    &v)  { u = u + v; return u; }
__host__ __device__  inline float2&  operator+=(float2  &u, const float2  &v)  { u = u + v; return u; }
__host__ __device__  inline float3&  operator+=(float3  &u, const float3  &v)  { u = u + v; return u; }
__host__ __device__  inline float4&  operator+=(float4  &u, const float4  &v)  { u = u + v; return u; }
__host__ __device__  inline double2& operator+=(double2 &u, const double2 &v)  { u = u + v; return u; }
__host__ __device__  inline double3& operator+=(double3 &u, const double3 &v)  { u = u + v; return u; }
__host__ __device__  inline double4& operator+=(double4 &u, const double4 &v)  { u = u + v; return u; }
// VECTOR - VECTOR
__host__ __device__  inline int2     operator-(const int2    &u, const int2    &v) { return int2   {u.x-v.x, u.y-v.y}; }
__host__ __device__  inline int3     operator-(const int3    &u, const int3    &v) { return int3   {u.x-v.x, u.y-v.y, u.z-v.z}; }
__host__ __device__  inline int4     operator-(const int4    &u, const int4    &v) { return int4   {u.x-v.x, u.y-v.y, u.z-v.z, u.w-v.w}; }
__host__ __device__  inline float2   operator-(const float2  &u, const float2  &v) { return float2 {u.x-v.x, u.y-v.y}; }
__host__ __device__  inline float3   operator-(const float3  &u, const float3  &v) { return float3 {u.x-v.x, u.y-v.y, u.z-v.z}; }
__host__ __device__  inline float4   operator-(const float4  &u, const float4  &v) { return float4 {u.x-v.x, u.y-v.y, u.z-v.z, u.w-v.w}; }
__host__ __device__  inline double2  operator-(const double2 &u, const double2 &v) { return double2{u.x-v.x, u.y-v.y}; }
__host__ __device__  inline double3  operator-(const double3 &u, const double3 &v) { return double3{u.x-v.x, u.y-v.y, u.z-v.z}; }
__host__ __device__  inline double4  operator-(const double4 &u, const double4 &v) { return double4{u.x-v.x, u.y-v.y, u.z-v.z, u.w-v.w}; }
__host__ __device__  inline int2&    operator-=(int2    &u, const int2    &v)  { u = u - v; return u; }
__host__ __device__  inline int3&    operator-=(int3    &u, const int3    &v)  { u = u - v; return u; }
__host__ __device__  inline int4&    operator-=(int4    &u, const int4    &v)  { u = u - v; return u; }
__host__ __device__  inline float2&  operator-=(float2  &u, const float2  &v)  { u = u - v; return u; }
__host__ __device__  inline float3&  operator-=(float3  &u, const float3  &v)  { u = u - v; return u; }
__host__ __device__  inline float4&  operator-=(float4  &u, const float4  &v)  { u = u - v; return u; }
__host__ __device__  inline double2& operator-=(double2 &u, const double2 &v)  { u = u - v; return u; }
__host__ __device__  inline double3& operator-=(double3 &u, const double3 &v)  { u = u - v; return u; }
__host__ __device__  inline double4& operator-=(double4 &u, const double4 &v)  { u = u - v; return u; }
// VECTOR * VECTOR
__host__ __device__  inline int2     operator*(const int2    &u, const int2    &v) { return int2   {u.x*v.x, u.y*v.y}; }
__host__ __device__  inline int3     operator*(const int3    &u, const int3    &v) { return int3   {u.x*v.x, u.y*v.y, u.z*v.z}; }
__host__ __device__  inline int4     operator*(const int4    &u, const int4    &v) { return int4   {u.x*v.x, u.y*v.y, u.z*v.z, u.w*v.w}; }
__host__ __device__  inline float2   operator*(const float2  &u, const float2  &v) { return float2 {u.x*v.x, u.y*v.y}; }
__host__ __device__  inline float3   operator*(const float3  &u, const float3  &v) { return float3 {u.x*v.x, u.y*v.y, u.z*v.z}; }
__host__ __device__  inline float4   operator*(const float4  &u, const float4  &v) { return float4 {u.x*v.x, u.y*v.y, u.z*v.z, u.w*v.w}; }
__host__ __device__  inline double2  operator*(const double2 &u, const double2 &v) { return double2{u.x*v.x, u.y*v.y}; }
__host__ __device__  inline double3  operator*(const double3 &u, const double3 &v) { return double3{u.x*v.x, u.y*v.y, u.z*v.z}; }
__host__ __device__  inline double4  operator*(const double4 &u, const double4 &v) { return double4{u.x*v.x, u.y*v.y, u.z*v.z, u.w*v.w}; }
__host__ __device__  inline int2&    operator*=(int2    &u, const int2    &v)  { u = u * v; return u; }
__host__ __device__  inline int3&    operator*=(int3    &u, const int3    &v)  { u = u * v; return u; }
__host__ __device__  inline int4&    operator*=(int4    &u, const int4    &v)  { u = u * v; return u; }
__host__ __device__  inline float2&  operator*=(float2  &u, const float2  &v)  { u = u * v; return u; }
__host__ __device__  inline float3&  operator*=(float3  &u, const float3  &v)  { u = u * v; return u; }
__host__ __device__  inline float4&  operator*=(float4  &u, const float4  &v)  { u = u * v; return u; }
__host__ __device__  inline double2& operator*=(double2 &u, const double2 &v)  { u = u * v; return u; }
__host__ __device__  inline double3& operator*=(double3 &u, const double3 &v)  { u = u * v; return u; }
__host__ __device__  inline double4& operator*=(double4 &u, const double4 &v)  { u = u * v; return u; }
// VECTOR / VECTOR
__host__ __device__  inline int2     operator/(const int2    &u, const int2    &v) { return int2   {u.x/v.x, u.y/v.y}; }
__host__ __device__  inline int3     operator/(const int3    &u, const int3    &v) { return int3   {u.x/v.x, u.y/v.y, u.z/v.z}; }
__host__ __device__  inline int4     operator/(const int4    &u, const int4    &v) { return int4   {u.x/v.x, u.y/v.y, u.z/v.z, u.w/v.w}; }
__host__ __device__  inline float2   operator/(const float2  &u, const float2  &v) { return float2 {u.x/v.x, u.y/v.y}; }
__host__ __device__  inline float3   operator/(const float3  &u, const float3  &v) { return float3 {u.x/v.x, u.y/v.y, u.z/v.z}; }
__host__ __device__  inline float4   operator/(const float4  &u, const float4  &v) { return float4 {u.x/v.x, u.y/v.y, u.z/v.z, u.w/v.w}; }
__host__ __device__  inline double2  operator/(const double2 &u, const double2 &v) { return double2{u.x/v.x, u.y/v.y}; }
__host__ __device__  inline double3  operator/(const double3 &u, const double3 &v) { return double3{u.x/v.x, u.y/v.y, u.z/v.z}; }
__host__ __device__  inline double4  operator/(const double4 &u, const double4 &v) { return double4{u.x/v.x, u.y/v.y, u.z/v.z, u.w/v.w}; }
__host__ __device__  inline int2&    operator/=(int2    &u, const int2    &v)  { u = u / v; return u; }
__host__ __device__  inline int3&    operator/=(int3    &u, const int3    &v)  { u = u / v; return u; }
__host__ __device__  inline int4&    operator/=(int4    &u, const int4    &v)  { u = u / v; return u; }
__host__ __device__  inline float2&  operator/=(float2  &u, const float2  &v)  { u = u / v; return u; }
__host__ __device__  inline float3&  operator/=(float3  &u, const float3  &v)  { u = u / v; return u; }
__host__ __device__  inline float4&  operator/=(float4  &u, const float4  &v)  { u = u / v; return u; }
__host__ __device__  inline double2& operator/=(double2 &u, const double2 &v)  { u = u / v; return u; }
__host__ __device__  inline double3& operator/=(double3 &u, const double3 &v)  { u = u / v; return u; }
__host__ __device__  inline double4& operator/=(double4 &u, const double4 &v)  { u = u / v; return u; }

// VECTOR + SCALAR
__host__ __device__  inline int2     operator+(const int2    &u, const int    &s) { return int2   {u.x+s, u.y+s}; }
__host__ __device__  inline int3     operator+(const int3    &u, const int    &s) { return int3   {u.x+s, u.y+s, u.z+s}; }
__host__ __device__  inline int4     operator+(const int4    &u, const int    &s) { return int4   {u.x+s, u.y+s, u.z+s, u.w+s}; }
__host__ __device__  inline float2   operator+(const float2  &u, const float  &s) { return float2 {u.x+s, u.y+s}; }
__host__ __device__  inline float3   operator+(const float3  &u, const float  &s) { return float3 {u.x+s, u.y+s, u.z+s}; }
__host__ __device__  inline float4   operator+(const float4  &u, const float  &s) { return float4 {u.x+s, u.y+s, u.z+s, u.w+s}; }
__host__ __device__  inline double2  operator+(const double2 &u, const double &s) { return double2{u.x+s, u.y+s}; }
__host__ __device__  inline double3  operator+(const double3 &u, const double &s) { return double3{u.x+s, u.y+s, u.z+s}; }
__host__ __device__  inline double4  operator+(const double4 &u, const double &s) { return double4{u.x+s, u.y+s, u.z+s, u.w+s}; }
__host__ __device__  inline int2&    operator+=(int2    &u, int    s)  { u = u + s; return u; }
__host__ __device__  inline int3&    operator+=(int3    &u, int    s)  { u = u + s; return u; }
__host__ __device__  inline int4&    operator+=(int4    &u, int    s)  { u = u + s; return u; }
__host__ __device__  inline float2&  operator+=(float2  &u, float  s)  { u = u + s; return u; }
__host__ __device__  inline float3&  operator+=(float3  &u, float  s)  { u = u + s; return u; }
__host__ __device__  inline float4&  operator+=(float4  &u, float  s)  { u = u + s; return u; }
__host__ __device__  inline double2& operator+=(double2 &u, double s)  { u = u + s; return u; }
__host__ __device__  inline double3& operator+=(double3 &u, double s)  { u = u + s; return u; }
__host__ __device__  inline double4& operator+=(double4 &u, double s)  { u = u + s; return u; }
// VECTOR - SCALAR
__host__ __device__  inline int2     operator-(const int2    &u, const int    &s) { return int2   {u.x-s, u.y-s}; }
__host__ __device__  inline int3     operator-(const int3    &u, const int    &s) { return int3   {u.x-s, u.y-s, u.z-s}; }
__host__ __device__  inline int4     operator-(const int4    &u, const int    &s) { return int4   {u.x-s, u.y-s, u.z-s, u.w-s}; }
__host__ __device__  inline float2   operator-(const float2  &u, const float  &s) { return float2 {u.x-s, u.y-s}; }
__host__ __device__  inline float3   operator-(const float3  &u, const float  &s) { return float3 {u.x-s, u.y-s, u.z-s}; }
__host__ __device__  inline float4   operator-(const float4  &u, const float  &s) { return float4 {u.x-s, u.y-s, u.z-s, u.w-s}; }
__host__ __device__  inline double2  operator-(const double2 &u, const double &s) { return double2{u.x-s, u.y-s}; }
__host__ __device__  inline double3  operator-(const double3 &u, const double &s) { return double3{u.x-s, u.y-s, u.z-s}; }
__host__ __device__  inline double4  operator-(const double4 &u, const double &s) { return double4{u.x-s, u.y-s, u.z-s, u.w-s}; }
__host__ __device__  inline int2&    operator-=(int2    &u, int    s)  { u = u - s; return u; }
__host__ __device__  inline int3&    operator-=(int3    &u, int    s)  { u = u - s; return u; }
__host__ __device__  inline int4&    operator-=(int4    &u, int    s)  { u = u - s; return u; }
__host__ __device__  inline float2&  operator-=(float2  &u, float  s)  { u = u - s; return u; }
__host__ __device__  inline float3&  operator-=(float3  &u, float  s)  { u = u - s; return u; }
__host__ __device__  inline float4&  operator-=(float4  &u, float  s)  { u = u - s; return u; }
__host__ __device__  inline double2& operator-=(double2 &u, double s)  { u = u - s; return u; }
__host__ __device__  inline double3& operator-=(double3 &u, double s)  { u = u - s; return u; }
__host__ __device__  inline double4& operator-=(double4 &u, double s)  { u = u - s; return u; }
// VECTOR * SCALAR
__host__ __device__  inline int2     operator*(const int2    &u, const int    &s) { return int2   {u.x*s, u.y*s}; }
__host__ __device__  inline int3     operator*(const int3    &u, const int    &s) { return int3   {u.x*s, u.y*s, u.z*s}; }
__host__ __device__  inline int4     operator*(const int4    &u, const int    &s) { return int4   {u.x*s, u.y*s, u.z*s, u.w*s}; }
__host__ __device__  inline float2   operator*(const float2  &u, const float  &s) { return float2 {u.x*s, u.y*s}; }
__host__ __device__  inline float3   operator*(const float3  &u, const float  &s) { return float3 {u.x*s, u.y*s, u.z*s}; }
__host__ __device__  inline float4   operator*(const float4  &u, const float  &s) { return float4 {u.x*s, u.y*s, u.z*s, u.w*s}; }
__host__ __device__  inline double2  operator*(const double2 &u, const double &s) { return double2{u.x*s, u.y*s}; }
__host__ __device__  inline double3  operator*(const double3 &u, const double &s) { return double3{u.x*s, u.y*s, u.z*s}; }
__host__ __device__  inline double4  operator*(const double4 &u, const double &s) { return double4{u.x*s, u.y*s, u.z*s, u.w*s}; }
__host__ __device__  inline int2&    operator*=(int2    &u, int    s)  { u = u * s; return u; }
__host__ __device__  inline int3&    operator*=(int3    &u, int    s)  { u = u * s; return u; }
__host__ __device__  inline int4&    operator*=(int4    &u, int    s)  { u = u * s; return u; }
__host__ __device__  inline float2&  operator*=(float2  &u, float  s)  { u = u * s; return u; }
__host__ __device__  inline float3&  operator*=(float3  &u, float  s)  { u = u * s; return u; }
__host__ __device__  inline float4&  operator*=(float4  &u, float  s)  { u = u * s; return u; }
__host__ __device__  inline double2& operator*=(double2 &u, double s)  { u = u * s; return u; }
__host__ __device__  inline double3& operator*=(double3 &u, double s)  { u = u * s; return u; }
__host__ __device__  inline double4& operator*=(double4 &u, double s)  { u = u * s; return u; }
// VECTOR / SCALAR
__host__ __device__  inline int2     operator/(const int2    &u, int    s) { return int2   {u.x/s, u.y/s}; }
__host__ __device__  inline int3     operator/(const int3    &u, int    s) { return int3   {u.x/s, u.y/s, u.z/s}; }
__host__ __device__  inline int4     operator/(const int4    &u, int    s) { return int4   {u.x/s, u.y/s, u.z/s, u.w/s}; }
__host__ __device__  inline float2   operator/(const float2  &u, float  s) { return float2 {u.x/s, u.y/s}; }
__host__ __device__  inline float3   operator/(const float3  &u, float  s) { return float3 {u.x/s, u.y/s, u.z/s}; }
__host__ __device__  inline float4   operator/(const float4  &u, float  s) { return float4 {u.x/s, u.y/s, u.z/s, u.w/s}; }
__host__ __device__  inline double2  operator/(const double2 &u, double s) { return double2{u.x/s, u.y/s}; }
__host__ __device__  inline double3  operator/(const double3 &u, double s) { return double3{u.x/s, u.y/s, u.z/s}; }
__host__ __device__  inline double4  operator/(const double4 &u, double s) { return double4{u.x/s, u.y/s, u.z/s, u.w/s}; }
__host__ __device__  inline int2&    operator/=(int2    &u, int    s)  { u = u / s; return u; }
__host__ __device__  inline int3&    operator/=(int3    &u, int    s)  { u = u / s; return u; }
__host__ __device__  inline int4&    operator/=(int4    &u, int    s)  { u = u / s; return u; }
__host__ __device__  inline float2&  operator/=(float2  &u, float  s)  { u = u / s; return u; }
__host__ __device__  inline float3&  operator/=(float3  &u, float  s)  { u = u / s; return u; }
__host__ __device__  inline float4&  operator/=(float4  &u, float  s)  { u = u / s; return u; }
__host__ __device__  inline double2& operator/=(double2 &u, double s)  { u = u / s; return u; }
__host__ __device__  inline double3& operator/=(double3 &u, double s)  { u = u / s; return u; }
__host__ __device__  inline double4& operator/=(double4 &u, double s)  { u = u / s; return u; }

// SCALAR + VECTOR
__host__ __device__  inline int2     operator+(int    s, const int2    &u)  { return int2   {s+u.x, s+u.y}; }
__host__ __device__  inline int3     operator+(int    s, const int3    &u)  { return int3   {s+u.x, s+u.y, s+u.z}; }
__host__ __device__  inline int4     operator+(int    s, const int4    &u)  { return int4   {s+u.x, s+u.y, s+u.z, s+u.w}; }
__host__ __device__  inline float2   operator+(float  s, const float2  &u)  { return float2 {s+u.x, s+u.y}; }
__host__ __device__  inline float3   operator+(float  s, const float3  &u)  { return float3 {s+u.x, s+u.y, s+u.z}; }
__host__ __device__  inline float4   operator+(float  s, const float4  &u)  { return float4 {s+u.x, s+u.y, s+u.z, s+u.w}; }
__host__ __device__  inline double2  operator+(double s, const double2 &u)  { return double2{s+u.x, s+u.y}; }
__host__ __device__  inline double3  operator+(double s, const double3 &u)  { return double3{s+u.x, s+u.y, s+u.z}; }
__host__ __device__  inline double4  operator+(double s, const double4 &u)  { return double4{s+u.x, s+u.y, s+u.z, s+u.w}; }
// SCALAR - VECTOR
__host__ __device__  inline int2     operator-(int    s, const int2    &u)  { return int2   {s-u.x, s-u.y}; }
__host__ __device__  inline int3     operator-(int    s, const int3    &u)  { return int3   {s-u.x, s-u.y, s-u.z}; }
__host__ __device__  inline int4     operator-(int    s, const int4    &u)  { return int4   {s-u.x, s-u.y, s-u.z, s-u.w}; }
__host__ __device__  inline float2   operator-(float  s, const float2  &u)  { return float2 {s-u.x, s-u.y}; }
__host__ __device__  inline float3   operator-(float  s, const float3  &u)  { return float3 {s-u.x, s-u.y, s-u.z}; }
__host__ __device__  inline float4   operator-(float  s, const float4  &u)  { return float4 {s-u.x, s-u.y, s-u.z, s-u.w}; }
__host__ __device__  inline double2  operator-(double s, const double2 &u)  { return double2{s-u.x, s-u.y}; }
__host__ __device__  inline double3  operator-(double s, const double3 &u)  { return double3{s-u.x, s-u.y, s-u.z}; }
__host__ __device__  inline double4  operator-(double s, const double4 &u)  { return double4{s-u.x, s-u.y, s-u.z, s-u.w}; }
// SCALAR * VECTOR
__host__ __device__  inline int2     operator*(int    s, const int2    &u)  { return int2   {s*u.x, s*u.y}; }
__host__ __device__  inline int3     operator*(int    s, const int3    &u)  { return int3   {s*u.x, s*u.y, s*u.z}; }
__host__ __device__  inline int4     operator*(int    s, const int4    &u)  { return int4   {s*u.x, s*u.y, s*u.z, s*u.w}; }
__host__ __device__  inline float2   operator*(float  s, const float2  &u)  { return float2 {s*u.x, s*u.y}; }
__host__ __device__  inline float3   operator*(float  s, const float3  &u)  { return float3 {s*u.x, s*u.y, s*u.z}; }
__host__ __device__  inline float4   operator*(float  s, const float4  &u)  { return float4 {s*u.x, s*u.y, s*u.z, s*u.w}; }
__host__ __device__  inline double2  operator*(double s, const double2 &u)  { return double2{s*u.x, s*u.y}; }
__host__ __device__  inline double3  operator*(double s, const double3 &u)  { return double3{s*u.x, s*u.y, s*u.z}; }
__host__ __device__  inline double4  operator*(double s, const double4 &u)  { return double4{s*u.x, s*u.y, s*u.z, s*u.w}; }
// SCALAR / VECTOR
__host__ __device__  inline int2     operator/(int    s, const int2    &u)  { return int2   {s/u.x, s/u.y}; }
__host__ __device__  inline int3     operator/(int    s, const int3    &u)  { return int3   {s/u.x, s/u.y, s/u.z}; }
__host__ __device__  inline int4     operator/(int    s, const int4    &u)  { return int4   {s/u.x, s/u.y, s/u.z, s/u.w}; }
__host__ __device__  inline float2   operator/(float  s, const float2  &u)  { return float2 {s/u.x, s/u.y}; }
__host__ __device__  inline float3   operator/(float  s, const float3  &u)  { return float3 {s/u.x, s/u.y, s/u.z}; }
__host__ __device__  inline float4   operator/(float  s, const float4  &u)  { return float4 {s/u.x, s/u.y, s/u.z, s/u.w}; }
__host__ __device__  inline double2  operator/(double s, const double2 &u)  { return double2{s/u.x, s/u.y}; }
__host__ __device__  inline double3  operator/(double s, const double3 &u)  { return double3{s/u.x, s/u.y, s/u.z}; }
__host__ __device__  inline double4  operator/(double s, const double4 &u)  { return double4{s/u.x, s/u.y, s/u.z, s/u.w}; }

// VECTOR LENGTH^2
__host__ __device__  inline float   length2(const int     &v) { return float(v*v); }
__host__ __device__  inline float   length2(const int2    &v) { return float(v.x*v.x + v.y*v.y); }
__host__ __device__  inline float   length2(const int3    &v) { return float(v.x*v.x + v.y*v.y + v.z*v.z); }
__host__ __device__  inline float   length2(const int4    &v) { return float(v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w); }
__host__ __device__  inline float   length2(const float   &v) { return v*v; }
__host__ __device__  inline float   length2(const float2  &v) { return (v.x*v.x + v.y*v.y); }
__host__ __device__  inline float   length2(const float3  &v) { return (v.x*v.x + v.y*v.y + v.z*v.z); }
__host__ __device__  inline float   length2(const float4  &v) { return (v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w); }
__host__ __device__  inline double  length2(const double  &v) { return v*v; }
__host__ __device__  inline double  length2(const double2 &v) { return (v.x*v.x + v.y*v.y); }
__host__ __device__  inline double  length2(const double3 &v) { return (v.x*v.x + v.y*v.y + v.z*v.z); }
__host__ __device__  inline double  length2(const double4 &v) { return (v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w); }
// VECTOR LENGTH
__host__ __device__  inline float   length(const int     &v) { return abs(float(v)); }
__host__ __device__  inline float   length(const int2    &v) { return sqrt(float(v.x*v.x + v.y*v.y)); }
__host__ __device__  inline float   length(const int3    &v) { return sqrt(float(v.x*v.x + v.y*v.y + v.z*v.z)); }
__host__ __device__  inline float   length(const int4    &v) { return sqrt(float(v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w)); }
__host__ __device__  inline float   length(const float   &v) { return abs(v); }
__host__ __device__  inline float   length(const float2  &v) { return sqrt(v.x*v.x + v.y*v.y); }
__host__ __device__  inline float   length(const float3  &v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
__host__ __device__  inline float   length(const float4  &v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w); }
__host__ __device__  inline double  length(const double  &v) { return abs(v); }
__host__ __device__  inline double  length(const double2 &v) { return sqrt(v.x*v.x + v.y*v.y); }
__host__ __device__  inline double  length(const double3 &v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
__host__ __device__  inline double  length(const double4 &v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w); }

// NORMALIZATION
__host__ __device__  inline float2   normalize(const float2  &v) { return v / length(v); }
__host__ __device__  inline float3   normalize(const float3  &v) { return v / length(v); }
__host__ __device__  inline float4   normalize(const float4  &v) { return v / length(v); }
__host__ __device__  inline double2  normalize(const double2 &v) { return v / length(v); }
__host__ __device__  inline double3  normalize(const double3 &v) { return v / length(v); }
__host__ __device__  inline double4  normalize(const double4 &v) { return v / length(v); }

// DOT PRODUCT
__host__ __device__  inline float   dot(const float2  &u, const float2  &v) { return (u.x*v.x + u.y*v.y); }
__host__ __device__  inline float   dot(const float3  &u, const float3  &v) { return (u.x*v.x + u.y*v.y + u.z*v.z); }
__host__ __device__  inline float   dot(const float4  &u, const float4  &v) { return (u.x*v.x + u.y*v.y + u.z*v.z + u.w*v.w); }
__host__ __device__  inline double  dot(const double2 &u, const double2 &v) { return (u.x*v.x + u.y*v.y); }
__host__ __device__  inline double  dot(const double3 &u, const double3 &v) { return (u.x*v.x + u.y*v.y + u.z*v.z); }
__host__ __device__  inline double  dot(const double4 &u, const double4 &v) { return (u.x*v.x + u.y*v.y + u.z*v.z + u.w*v.w); }

// SUM OF ELEMENTS
__host__ __device__  inline int     sum(const int2    &v) { return (v.x+v.y); }
__host__ __device__  inline int     sum(const int3    &v) { return (v.x+v.y+v.z); }
__host__ __device__  inline int     sum(const int4    &v) { return (v.x+v.y+v.z+v.w); }
__host__ __device__  inline float   sum(const float2  &v) { return (v.x+v.y); }
__host__ __device__  inline float   sum(const float3  &v) { return (v.x+v.y+v.z); }
__host__ __device__  inline float   sum(const float4  &v) { return (v.x+v.y+v.z+v.w); }
__host__ __device__  inline double  sum(const double2 &v) { return (v.x+v.y); }
__host__ __device__  inline double  sum(const double3 &v) { return (v.x+v.y+v.z); }
__host__ __device__  inline double  sum(const double4 &v) { return (v.x+v.y+v.z+v.w); }

// PRODUCT OF ELEMENTS
__host__ __device__  inline int     product(const int2    &v) { return (v.x*v.y); }
__host__ __device__  inline int     product(const int3    &v) { return (v.x*v.y*v.z); }
__host__ __device__  inline int     product(const int4    &v) { return (v.x*v.y*v.z*v.w); }
__host__ __device__  inline float   product(const float2  &v) { return (v.x*v.y); }
__host__ __device__  inline float   product(const float3  &v) { return (v.x*v.y*v.z); }
__host__ __device__  inline float   product(const float4  &v) { return (v.x*v.y*v.z*v.w); }
__host__ __device__  inline double  product(const double2 &v) { return (v.x*v.y); }
__host__ __device__  inline double  product(const double3 &v) { return (v.x*v.y*v.z); }
__host__ __device__  inline double  product(const double4 &v) { return (v.x*v.y*v.z*v.w); }






// MODULO
__host__ __device__  inline int2 operator%(const int2    &v, int    m) { return int2{(v.x % m), (v.y % m)}; }
__host__ __device__  inline int3 operator%(const int3    &v, int    m) { return int3{(v.x % m), (v.y % m), (v.z % m)}; }
__host__ __device__  inline int4 operator%(const int4    &v, int    m) { return int4{(v.x % m), (v.y % m), (v.z % m), (v.w % m)}; }
__host__ __device__  inline float2   fmod (const float2  &v, float  m) { return float2 {  fmodf(v.x, m), fmodf(v.y, m) }; }
__host__ __device__  inline float3   fmod (const float3  &v, float  m) { return float3 {  fmodf(v.x, m), fmodf(v.y, m),
                                                                                          fmodf(v.z, m) }; }
__host__ __device__  inline float4   fmod (const float4  &v, float  m) { return float4 {  fmodf(v.x, m), fmodf(v.y, m),
                                                                                          fmodf(v.z, m), fmodf(v.w, m) }; }
__host__ __device__  inline double2  fmod (const double2 &v, double m) { return double2{  fmod (v.x, m), fmod (v.y, m) }; }
__host__ __device__  inline double3  fmod (const double3 &v, double m) { return double3{  fmod (v.x, m), fmod (v.y, m),
                                                                                          fmod (v.z, m) }; }
__host__ __device__  inline double4  fmod (const double4 &v, double m) { return double4{  fmod (v.x, m), fmod (v.y, m),
                                                                                          fmod (v.z, m), fmod (v.w, m) }; }

__host__ __device__  inline int2 operator%(const int2    &v, const int2    &m) { return int2   {(v.x % m.x), (v.y % m.y)}; }
__host__ __device__  inline int3 operator%(const int3    &v, const int3    &m) { return int3   {(v.x % m.x), (v.y % m.y), (v.z % m.z)}; }
__host__ __device__  inline int4 operator%(const int4    &v, const int4    &m) { return int4   {(v.x % m.x), (v.y % m.y), (v.z % m.z), (v.w % m.w)}; }
__host__ __device__  inline float2    fmod(const float2  &v, const float2  &m) { return float2 {  fmodf(v.x, m.x), fmodf(v.y, m.y) }; }
__host__ __device__  inline float3    fmod(const float3  &v, const float3  &m) { return float3 {  fmodf(v.x, m.x), fmodf(v.y, m.y),
                                                                                                  fmodf(v.z, m.z) }; }
__host__ __device__  inline float4    fmod(const float4  &v, const float4  &m) { return float4 {  fmodf(v.x, m.x), fmodf(v.y, m.y),
                                                                                                  fmodf(v.z, m.z), fmodf(v.w, m.w) }; }
__host__ __device__  inline double2   fmod(const double2 &v, const double2 &m) { return double2{  fmod (v.x, m.x), fmod (v.y, m.y) }; }
__host__ __device__  inline double3   fmod(const double3 &v, const double3 &m) { return double3{  fmod (v.x, m.x), fmod (v.y, m.y),
                                                                                                  fmod (v.z, m.z) }; }
__host__ __device__  inline double4   fmod(const double4 &v, const double4 &m) { return double4{  fmod (v.x, m.x), fmod (v.y, m.y),
                                                                                                  fmod (v.z, m.z), fmod (v.w, m.w) }; }

// NaN
__host__ __device__ inline bool isnan(const int2    &v) { return false; } // (int can't be NaN)
__host__ __device__ inline bool isnan(const int3    &v) { return false; }
__host__ __device__ inline bool isnan(const int4    &v) { return false; }
__host__ __device__ inline bool isnan(const float2  &v) { return (v.x != v.x || v.y != v.y); }
__host__ __device__ inline bool isnan(const float3  &v) { return (v.x != v.x || v.y != v.y || v.z != v.z); }
__host__ __device__ inline bool isnan(const float4  &v) { return (v.x != v.x || v.y != v.y || v.z != v.z || v.w != v.w); }
__host__ __device__ inline bool isnan(const double2 &v) { return (v.x != v.x || v.y != v.y); }
__host__ __device__ inline bool isnan(const double3 &v) { return (v.x != v.x || v.y != v.y || v.z != v.z); }
__host__ __device__ inline bool isnan(const double4 &v) { return (v.x != v.x || v.y != v.y || v.z != v.z || v.w != v.w); }

// MIN/MAX (?)
__host__ __device__ inline float fminf(float a, float b) { return a < b ? a : b; }
__host__ __device__ inline float fmaxf(float a, float b) { return a > b ? a : b; }
__host__ __device__ inline int   imax (int   a, int   b) { return a > b ? a : b; }
__host__ __device__ inline int   imin (int   a, int   b) { return a < b ? a : b; }

// MAX
// __host__ __device__  inline int     max(const int     &u, const int     &v) { return (u > v ? u : v); }
__host__ __device__  inline int2    max(int2    u, int2    v) { return (length2(u) > length2(v) ? u : v); }
__host__ __device__  inline int3    max(int3    u, int3    v) { return (length2(u) > length2(v) ? u : v); }
__host__ __device__  inline int4    max(int4    u, int4    v) { return (length2(u) > length2(v) ? u : v); }
__host__ __device__  inline float2  max(float2  u, float2  v) { return (length2(u) > length2(v) ? u : v); }
__host__ __device__  inline float3  max(float3  u, float3  v) { return (length2(u) > length2(v) ? u : v); }
__host__ __device__  inline float4  max(float4  u, float4  v) { return (length2(u) > length2(v) ? u : v); }
// __host__ __device__  inline double  max(const double  &u, const double  &v) { return (u > v ? u : v); }
__host__ __device__  inline double2 max(double2 u, double2 v) { return (length2(u) > length2(v) ? u : v); }
__host__ __device__  inline double3 max(double3 u, double3 v) { return (length2(u) > length2(v) ? u : v); }
__host__ __device__  inline double4 max(double4 u, double4 v) { return (length2(u) > length2(v) ? u : v); } 

__host__ __device__  inline int2    max(int2    u, int    s) { return int2   {(u.x > s ? u.x : s), (u.y > s ? u.y : s)}; }
__host__ __device__  inline int3    max(int3    u, int    s) { return int3   {(u.x > s ? u.x : s), (u.y > s ? u.y : s), (u.z > s ? u.z : s)}; }
__host__ __device__  inline int4    max(int4    u, int    s) { return int4   {(u.x > s ? u.x : s), (u.y > s ? u.y : s), (u.z > s ? u.z : s), (u.w > s ? u.w : s)}; }
__host__ __device__  inline float2  max(float2  u, float  s) { return float2 {(u.x > s ? u.x : s), (u.y > s ? u.y : s)}; }
__host__ __device__  inline float3  max(float3  u, float  s) { return float3 {(u.x > s ? u.x : s), (u.y > s ? u.y : s), (u.z > s ? u.z : s)}; }
__host__ __device__  inline float4  max(float4  u, float  s) { return float4 {(u.x > s ? u.x : s), (u.y > s ? u.y : s), (u.z > s ? u.z : s), (u.w > s ? u.w : s)}; }
__host__ __device__  inline double2 max(double2 u, double s) { return double2{(u.x > s ? u.x : s), (u.y > s ? u.y : s)}; }
__host__ __device__  inline double3 max(double3 u, double s) { return double3{(u.x > s ? u.x : s), (u.y > s ? u.y : s), (u.z > s ? u.z : s)}; }
__host__ __device__  inline double4 max(double4 u, double s) { return double4{(u.x > s ? u.x : s), (u.y > s ? u.y : s), (u.z > s ? u.z : s), (u.w > s ? u.w : s)}; } 

__host__ __device__  inline int    max(int2    v) { return imax (v.x, v.y); }
__host__ __device__  inline int    max(int3    v) { return imax (v.x, imax (v.y, v.z)); }
__host__ __device__  inline int    max(int4    v) { return imax (v.x, imax (v.y, imax (v.z,  v.w))); }
__host__ __device__  inline float  max(float2  v) { return fmaxf(v.x, v.y); }
__host__ __device__  inline float  max(float3  v) { return fmaxf(v.x, fmaxf(v.y, v.z)); }
__host__ __device__  inline float  max(float4  v) { return fmaxf(v.x, fmaxf(v.y, fmaxf(v.z, v.w))); }
__host__ __device__  inline double max(double2 v) { return fmaxf(v.x, v.y); }
__host__ __device__  inline double max(double3 v) { return fmaxf(v.x, fmaxf(v.y, v.z)); }
__host__ __device__  inline double max(double4 v) { return fmaxf(v.x, fmaxf(v.y, fmaxf(v.z, v.w))); }

// MIN
__host__ __device__  inline int2    min(int2    u, int2    v) { return (length2(u) < length2(v) ? u : v); }
__host__ __device__  inline int3    min(int3    u, int3    v) { return (length2(u) < length2(v) ? u : v); }
__host__ __device__  inline int4    min(int4    u, int4    v) { return (length2(u) < length2(v) ? u : v); }
__host__ __device__  inline float2  min(float2  u, float2  v) { return (length2(u) < length2(v) ? u : v); }
__host__ __device__  inline float3  min(float3  u, float3  v) { return (length2(u) < length2(v) ? u : v); }
__host__ __device__  inline float4  min(float4  u, float4  v) { return (length2(u) < length2(v) ? u : v); }
__host__ __device__  inline double2 min(double2 u, double2 v) { return (length2(u) < length2(v) ? u : v); }
__host__ __device__  inline double3 min(double3 u, double3 v) { return (length2(u) < length2(v) ? u : v); }
__host__ __device__  inline double4 min(double4 u, double4 v) { return (length2(u) < length2(v) ? u : v); }

__host__ __device__  inline int2    min(int2    u, int    s) { return int2   {(u.x < s ? u.x : s), (u.y < s ? u.y : s)}; }
__host__ __device__  inline int3    min(int3    u, int    s) { return int3   {(u.x < s ? u.x : s), (u.y < s ? u.y : s), (u.z < s ? u.z : s)}; }
__host__ __device__  inline int4    min(int4    u, int    s) { return int4   {(u.x < s ? u.x : s), (u.y < s ? u.y : s), (u.z < s ? u.z : s), (u.w < s ? u.w : s)}; }
__host__ __device__  inline float2  min(float2  u, float  s) { return float2 {(u.x < s ? u.x : s), (u.y < s ? u.y : s)}; }
__host__ __device__  inline float3  min(float3  u, float  s) { return float3 {(u.x < s ? u.x : s), (u.y < s ? u.y : s), (u.z < s ? u.z : s)}; }
__host__ __device__  inline float4  min(float4  u, float  s) { return float4 {(u.x < s ? u.x : s), (u.y < s ? u.y : s), (u.z < s ? u.z : s), (u.w < s ? u.w : s)}; }
__host__ __device__  inline double2 min(double2 u, double s) { return double2{(u.x < s ? u.x : s), (u.y < s ? u.y : s)}; }
__host__ __device__  inline double3 min(double3 u, double s) { return double3{(u.x < s ? u.x : s), (u.y < s ? u.y : s), (u.z < s ? u.z : s)}; }
__host__ __device__  inline double4 min(double4 u, double s) { return double4{(u.x < s ? u.x : s), (u.y < s ? u.y : s), (u.z < s ? u.z : s), (u.w < s ? u.w : s)}; } 

__host__ __device__  inline int    min(int2    v) { return imin(v.x,  v.y); }
__host__ __device__  inline int    min(int3    v) { return imin(v.x,  imin(v.y,  v.z)); }
__host__ __device__  inline int    min(int4    v) { return imin(v.x,  imin(v.y,  imin(v.z,  v.w))); }
__host__ __device__  inline float  min(float2  v) { return fminf(v.x, v.y); }
__host__ __device__  inline float  min(float3  v) { return fminf(v.x, fminf(v.y, v.z)); }
__host__ __device__  inline float  min(float4  v) { return fminf(v.x, fminf(v.y, fminf(v.z, v.w))); }
__host__ __device__  inline double min(double2 v) { return fminf(v.x, v.y); }
__host__ __device__  inline double min(double3 v) { return fminf(v.x, fminf(v.y, v.z)); }
__host__ __device__  inline double min(double4 v) { return fminf(v.x, fminf(v.y, fminf(v.z, v.w))); }


// FLOOR
__host__ __device__  inline float2  floor(const float2  &v) { return float2 {  floorf(v.x), floorf(v.y) }; }
__host__ __device__  inline float3  floor(const float3  &v) { return float3 {  floorf(v.x), floorf(v.y), floorf(v.z) }; }
__host__ __device__  inline float4  floor(const float4  &v) { return float4 {  floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w) }; }
__host__ __device__  inline double2 floor(const double2 &v) { return double2{  floor(v.x),  floor(v.y) }; }
__host__ __device__  inline double3 floor(const double3 &v) { return double3{  floor(v.x),  floor(v.y),  floor(v.z) }; }
__host__ __device__  inline double4 floor(const double4 &v) { return double4{  floor(v.x),  floor(v.y),  floor(v.z),  floor(v.w) }; } 

// CEIL
__host__ __device__  inline float2  ceil(const float2  &v) { return -floor(v); }
__host__ __device__  inline float3  ceil(const float3  &v) { return -floor(v); }
__host__ __device__  inline float4  ceil(const float4  &v) { return -floor(v); }
__host__ __device__  inline double2 ceil(const double2 &v) { return -floor(v); }
__host__ __device__  inline double3 ceil(const double3 &v) { return -floor(v); }
__host__ __device__  inline double4 ceil(const double4 &v) { return -floor(v); }

// ABS
__host__ __device__  inline int2     abs(const int2    &v)  { return int2   {v.x, v.y}; }
__host__ __device__  inline int3     abs(const int3    &v)  { return int3   {v.x, v.y, v.z}; }
__host__ __device__  inline int4     abs(const int4    &v)  { return int4   {v.x, v.y, v.z, v.w}; }
__host__ __device__  inline float2   abs(const float2  &v)  { return float2 {v.x, v.y}; }
__host__ __device__  inline float3   abs(const float3  &v)  { return float3 {v.x, v.y, v.z}; }
__host__ __device__  inline float4   abs(const float4  &v)  { return float4 {v.x, v.y, v.z, v.w}; }
__host__ __device__  inline double2  abs(const double2 &v)  { return double2{v.x, v.y}; }
__host__ __device__  inline double3  abs(const double3 &v)  { return double3{v.x, v.y, v.z}; }
__host__ __device__  inline double4  abs(const double4 &v)  { return double4{v.x, v.y, v.z, v.w}; }

// LOG
__host__ __device__  inline int      log(const int     &v)  { return (int)logf((float)v);}
__host__ __device__  inline int2     log(const int2    &v)  { return int2{(int)logf((float)v.x),(int)logf((float)v.y)};}
__host__ __device__  inline int3     log(const int3    &v)  { return int3{(int)logf((float)v.x),(int)logf((float)v.y),(int)logf((float)v.z)};}
__host__ __device__  inline int4     log(const int4    &v)  { return int4{(int)logf((float)v.x),(int)logf((float)v.y),(int)logf((float)v.z),(int)logf((float)v.w)};}
__host__ __device__  inline float2   log(const float2  &v)  { return float2 {(float)logf(v.x), (float)logf(v.y)}; }
__host__ __device__  inline float3   log(const float3  &v)  { return float3 {(float)logf(v.x), (float)logf(v.y), (float)logf(v.z)}; }
__host__ __device__  inline float4   log(const float4  &v)  { return float4 {(float)logf(v.x), (float)logf(v.y), (float)logf(v.z), (float)logf(v.w)}; }
__host__ __device__  inline double2  log(const double2 &v)  { return double2{log((float)v.x), log((float)v.y)}; }
__host__ __device__  inline double3  log(const double3 &v)  { return double3{log((float)v.x), log((float)v.y), log((float)v.z)}; }
__host__ __device__  inline double4  log(const double4 &v)  { return double4{log((float)v.x), log((float)v.y), log((float)v.z), log((float)v.w)}; }

// NEGLOG
__host__ __device__  inline int      neglog(const int     &v)  { return (int)log(abs(v))*(v < 0 ? -1 : 1); }
__host__ __device__  inline int2     neglog(const int2    &v)  { return int2{  (int)logf(abs(v.x))*(v.x < 0 ? -1 : 1),
                                                                               (int)logf(abs(v.y))*(v.y < 0 ? -1 : 1) }; }
__host__ __device__  inline int3     neglog(const int3    &v)  { return int3{  (int)logf(abs(v.x))*(v.x < 0 ? -1 : 1),
                                                                               (int)logf(abs(v.y))*(v.y < 0 ? -1 : 1),
                                                                               (int)logf(abs(v.z))*(v.z < 0 ? -1 : 1) };}
__host__ __device__  inline int4     neglog(const int4    &v)  { return int4{  (int)logf(v.x)*(v.x < 0 ? -1 : 1),
                                                                               (int)logf(v.y)*(v.x < 0 ? -1 : 1),
                                                                               (int)logf(v.z)*(v.x < 0 ? -1 : 1),
                                                                               (int)logf(v.w)*(v.x < 0 ? -1 : 1) };}
__host__ __device__  inline float2   neglog(const float2  &v)  { return float2 {  (float)logf(v.x)*(v.x < 0 ? -1 : 1), logf(v.y)*(v.y < 0 ? -1 : 1)}; }
__host__ __device__  inline float3   neglog(const float3  &v)  { return float3 {  (float)logf(v.x)*(v.x < 0 ? -1 : 1), logf(v.y)*(v.y < 0 ? -1 : 1),
                                                                                  (float)logf(v.z)*(v.z < 0 ? -1 : 1)}; }
__host__ __device__  inline float4   neglog(const float4  &v)  { return float4 {  (float)logf(v.x)*(v.x < 0 ? -1 : 1), logf(v.y)*(v.y < 0 ? -1 : 1),
                                                                                  (float)logf(v.z)*(v.z < 0 ? -1 : 1), logf(v.w)*(v.w < 0 ? -1 : 1)}; }
__host__ __device__  inline double2  neglog(const double2 &v)  { return double2{  (double)log((float)v.x)*(v.x < 0 ? -1 : 1),
                                                                                  (double)log((float)v.y)*(v.y < 0 ? -1 : 1) }; }
__host__ __device__  inline double3  neglog(const double3 &v)  { return double3{  (double)log((float)v.x)*(v.x < 0 ? -1 : 1),
                                                                                  (double)log((float)v.y)*(v.y < 0 ? -1 : 1),
                                                                                  (double)log((float)v.z)*(v.z < 0 ? -1 : 1) }; }
__host__ __device__  inline double4  neglog(const double4 &v)  { return double4{  (double)log((float)v.x)*(v.x < 0 ? -1 : 1),
                                                                                  (double)log((float)v.y)*(v.y < 0 ? -1 : 1),
                                                                                  (double)log((float)v.z)*(v.z < 0 ? -1 : 1),
                                                                                  (double)log((float)v.w)*(v.w < 0 ? -1 : 1) }; }
// EXP
__host__ __device__  inline int      exp(const int     &v)  { return (int)exp((float)v); }
__host__ __device__  inline int2     exp(const int2    &v)  { return int2   {exp(v.x), exp(v.y)};}
__host__ __device__  inline int3     exp(const int3    &v)  { return int3   {exp(v.x), exp(v.y), exp(v.z)};}
__host__ __device__  inline int4     exp(const int4    &v)  { return int4   {exp(v.x), exp(v.y), exp(v.z), exp(v.w)};}
__host__ __device__  inline float2   exp(const float2  &v)  { return float2 {  (float) exp(v.x), (float)exp(v.y)}; }
__host__ __device__  inline float3   exp(const float3  &v)  { return float3 {  (float) exp(v.x), (float)exp(v.y), (float)exp(v.z)}; }
__host__ __device__  inline float4   exp(const float4  &v)  { return float4 {  (float) exp(v.x), (float)exp(v.y), (float)exp(v.z), (float)exp(v.w)}; }
__host__ __device__  inline double2  exp(const double2 &v)  { return double2{  (double)exp((float)v.x), (double)exp((float)v.y) }; }
__host__ __device__  inline double3  exp(const double3 &v)  { return double3{  (double)exp((float)v.x), (double)exp((float)v.y),
                                                                               (double)exp((float)v.z)}; }
__host__ __device__  inline double4  exp(const double4 &v)  { return double4{  (double)exp((float)v.x), (double)exp((float)v.y),
                                                                               (double)exp((float)v.z), (double)exp((float)v.w) }; }

// SIN
__host__ __device__  inline float2   sin(const float2  &v)  { return float2 {  (float) sin(v.x), (float)sin(v.y)}; }
__host__ __device__  inline float3   sin(const float3  &v)  { return float3 {  (float) sin(v.x), (float)sin(v.y), (float)sin(v.z)}; }
__host__ __device__  inline float4   sin(const float4  &v)  { return float4 {  (float) sin(v.x), (float)sin(v.y), (float)sin(v.z), (float)sin(v.w)}; }
__host__ __device__  inline double2  sin(const double2 &v)  { return double2{  (double)sin((float)v.x), (double)sin((float)v.y) }; }
__host__ __device__  inline double3  sin(const double3 &v)  { return double3{  (double)sin((float)v.x), (double)sin((float)v.y),
                                                                               (double)sin((float)v.z)}; }
__host__ __device__  inline double4  sin(const double4 &v)  { return double4{  (double)sin((float)v.x), (double)sin((float)v.y),
                                                                               (double)sin((float)v.z), (double)sin((float)v.w) }; }
// COS
__host__ __device__  inline float2   cos(const float2  &v)  { return float2 {  (float) cos(v.x), (float)cos(v.y)}; }
__host__ __device__  inline float3   cos(const float3  &v)  { return float3 {  (float) cos(v.x), (float)cos(v.y), (float)cos(v.z)}; }
__host__ __device__  inline float4   cos(const float4  &v)  { return float4 {  (float) cos(v.x), (float)cos(v.y), (float)cos(v.z), (float)cos(v.w)}; }
__host__ __device__  inline double2  cos(const double2 &v)  { return double2{  (double)cos((float)v.x), (double)cos((float)v.y) }; }
__host__ __device__  inline double3  cos(const double3 &v)  { return double3{  (double)cos((float)v.x), (double)cos((float)v.y),
                                                                               (double)cos((float)v.z)}; }
__host__ __device__  inline double4  cos(const double4 &v)  { return double4{  (double)cos((float)v.x), (double)cos((float)v.y),
                                                                               (double)cos((float)v.z), (double)cos((float)v.w) }; }
// TAN
__host__ __device__  inline float2   tan(const float2  &v)  { return float2 {  (float) tan(v.x), (float)tan(v.y)}; }
__host__ __device__  inline float3   tan(const float3  &v)  { return float3 {  (float) tan(v.x), (float)tan(v.y), (float)tan(v.z)}; }
__host__ __device__  inline float4   tan(const float4  &v)  { return float4 {  (float) tan(v.x), (float)tan(v.y), (float)tan(v.z), (float)tan(v.w)}; }
__host__ __device__  inline double2  tan(const double2 &v)  { return double2{  (double)tan((float)v.x), (double)tan((float)v.y) }; }
__host__ __device__  inline double3  tan(const double3 &v)  { return double3{  (double)tan((float)v.x), (double)tan((float)v.y),
                                                                               (double)tan((float)v.z)}; }
__host__ __device__  inline double4  tan(const double4 &v)  { return double4{  (double)tan((float)v.x), (double)tan((float)v.y),
                                                                               (double)tan((float)v.z), (double)tan((float)v.w) }; }
// SQRT
__host__ __device__  inline float2   sqrt(const float2  &v)  { return float2 {  (float) sqrt(v.x), (float)sqrt(v.y)}; }
__host__ __device__  inline float3   sqrt(const float3  &v)  { return float3 {  (float) sqrt(v.x), (float)sqrt(v.y), (float)sqrt(v.z)}; }
__host__ __device__  inline float4   sqrt(const float4  &v)  { return float4 {  (float) sqrt(v.x), (float)sqrt(v.y), (float)sqrt(v.z), (float)sqrt(v.w)}; }
__host__ __device__  inline double2  sqrt(const double2 &v)  { return double2{  (double)sqrt((float)v.x), (double)sqrt((float)v.y) }; }
__host__ __device__  inline double3  sqrt(const double3 &v)  { return double3{  (double)sqrt((float)v.x), (double)sqrt((float)v.y),
                                                                                (double)sqrt((float)v.z)}; }
__host__ __device__  inline double4  sqrt(const double4 &v)  { return double4{  (double)sqrt((float)v.x), (double)sqrt((float)v.y),
                                                                                (double)sqrt((float)v.z), (double)sqrt((float)v.w) }; }


// POW (vector, scalar)
__host__ __device__  inline float2   pow (const float2  &v, float  m) { return float2 {  powf(v.x, m), powf(v.y, m) }; }
__host__ __device__  inline float3   pow (const float3  &v, float  m) { return float3 {  powf(v.x, m), powf(v.y, m),
                                                                                         powf(v.z, m) }; }
__host__ __device__  inline float4   pow (const float4  &v, float  m) { return float4 {  powf(v.x, m), powf(v.y, m),
                                                                                         powf(v.z, m), powf(v.w, m) }; }
__host__ __device__  inline double2  pow (const double2 &v, double m) { return double2{  pow (v.x, m), pow (v.y, m) }; }
__host__ __device__  inline double3  pow (const double3 &v, double m) { return double3{  pow (v.x, m), pow (v.y, m),
                                                                                         pow (v.z, m) }; }
__host__ __device__  inline double4  pow (const double4 &v, double m) { return double4{  pow (v.x, m), pow (v.y, m),
                                                                                         pow (v.z, m), pow (v.w, m) }; }
// POW (vector, vector)
__host__ __device__  inline float2   pow(const float2  &v, const float2  &m) { return float2 {  powf(v.x, m.x), powf(v.y, m.y) }; }
__host__ __device__  inline float3   pow(const float3  &v, const float3  &m) { return float3 {  powf(v.x, m.x), powf(v.y, m.y),
                                                                                                powf(v.z, m.z) }; }
__host__ __device__  inline float4   pow(const float4  &v, const float4  &m) { return float4 {  powf(v.x, m.x), powf(v.y, m.y),
                                                                                                powf(v.z, m.z), powf(v.w, m.w) }; }
__host__ __device__  inline double2  pow(const double2 &v, const double2 &m) { return double2{  pow (v.x, m.x), pow (v.y, m.y) }; }
__host__ __device__  inline double3  pow(const double3 &v, const double3 &m) { return double3{  pow (v.x, m.x), pow (v.y, m.y),
                                                                                                pow (v.z, m.z) }; }
__host__ __device__  inline double4  pow(const double4 &v, const double4 &m) { return double4{  pow (v.x, m.x), pow (v.y, m.y),
                                                                                                pow (v.z, m.z), pow (v.w, m.w) }; }



////////////////////////////////////////////////////////////////////////////////
// array data
////////////////////////////////////////////////////////////////////////////////

__host__ __device__  inline const int*     arr(const int     &v)  { return (int   *)&v;   }
__host__ __device__  inline const int*     arr(const int2    &v)  { return (int   *)&v.x; }
__host__ __device__  inline const int*     arr(const int3    &v)  { return (int   *)&v.x; }
__host__ __device__  inline const int*     arr(const int4    &v)  { return (int   *)&v.x; }
__host__ __device__  inline const float*   arr(const float   &v)  { return (float *)&v;   }
__host__ __device__  inline const float*   arr(const float2  &v)  { return (float *)&v.x; }
__host__ __device__  inline const float*   arr(const float3  &v)  { return (float *)&v.x; }
__host__ __device__  inline const float*   arr(const float4  &v)  { return (float *)&v.x; }
__host__ __device__  inline const double*  arr(const double  &v)  { return (double*)&v;   }
__host__ __device__  inline const double*  arr(const double2 &v)  { return (double*)&v.x; }
__host__ __device__  inline const double*  arr(const double3 &v)  { return (double*)&v.x; }
__host__ __device__  inline const double*  arr(const double4 &v)  { return (double*)&v.x; }

__host__ __device__  inline int*     arr(int     &v)  { return (int   *)&v;   }
__host__ __device__  inline int*     arr(int2    &v)  { return (int   *)&v.x; }
__host__ __device__  inline int*     arr(int3    &v)  { return (int   *)&v.x; }
__host__ __device__  inline int*     arr(int4    &v)  { return (int   *)&v.x; }
__host__ __device__  inline float*   arr(float   &v)  { return (float *)&v;   }
__host__ __device__  inline float*   arr(float2  &v)  { return (float *)&v.x; }
__host__ __device__  inline float*   arr(float3  &v)  { return (float *)&v.x; }
__host__ __device__  inline float*   arr(float4  &v)  { return (float *)&v.x; }
__host__ __device__  inline double*  arr(double  &v)  { return (double*)&v;   }
__host__ __device__  inline double*  arr(double2 &v)  { return (double*)&v.x; }
__host__ __device__  inline double*  arr(double3 &v)  { return (double*)&v.x; }
__host__ __device__  inline double*  arr(double4 &v)  { return (double*)&v.x; }



////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////


inline __host__ __device__ float2 make_float2(float  s) { return make_float2(s, s); }
inline __host__ __device__ float2 make_float2(float3 a) { return make_float2(a.x, a.y); }
inline __host__ __device__ float2 make_float2(int2   a) { return make_float2(float(a.x), float(a.y)); }
inline __host__ __device__ float2 make_float2(uint2  a) { return make_float2(float(a.x), float(a.y)); }

inline __host__ __device__ double2 make_double2(double  s) { return make_double2(s, s); }
inline __host__ __device__ double2 make_double2(double3 a) { return make_double2(a.x, a.y); }
inline __host__ __device__ double2 make_double2(int2    a) { return make_double2(double(a.x), double(a.y)); }
inline __host__ __device__ double2 make_double2(uint2   a) { return make_double2(double(a.x), double(a.y)); }

inline __host__ __device__ int2 make_int2(int    s) { return make_int2(s, s); }
inline __host__ __device__ int2 make_int2(int3   a) { return make_int2(a.x, a.y); }
inline __host__ __device__ int2 make_int2(uint2  a) { return make_int2(int(a.x), int(a.y)); }
inline __host__ __device__ int2 make_int2(float2 a) { return make_int2(int(a.x), int(a.y)); }

inline __host__ __device__ uint2 make_uint2(uint  s) { return make_uint2(s, s); }
inline __host__ __device__ uint2 make_uint2(uint3 a) { return make_uint2(a.x, a.y); }
inline __host__ __device__ uint2 make_uint2(int2  a) { return make_uint2(uint(a.x), uint(a.y)); }

inline __host__ __device__ float3 make_float3(float  s) { return make_float3(s, s, s); }
inline __host__ __device__ float3 make_float3(float2 a) { return make_float3(a.x, a.y, 0.0f); }
inline __host__ __device__ float3 make_float3(float4 a) { return make_float3(a.x, a.y, a.z); }
inline __host__ __device__ float3 make_float3(int3   a) { return make_float3(float(a.x), float(a.y), float(a.z)); }
inline __host__ __device__ float3 make_float3(uint3  a) { return make_float3(float(a.x), float(a.y), float(a.z)); }
inline __host__ __device__ float3 make_float3(float2 a, float s) { return make_float3(a.x, a.y, s); }

inline __host__ __device__ double3 make_double3(double  s) { return make_double3(s, s, s); }
inline __host__ __device__ double3 make_double3(double2 a) { return make_double3(a.x, a.y, 0.0); }
inline __host__ __device__ double3 make_double3(double4 a) { return make_double3(a.x, a.y, a.z); }
inline __host__ __device__ double3 make_double3(int3    a) { return make_double3(double(a.x), double(a.y), double(a.z)); }
inline __host__ __device__ double3 make_double3(uint3   a) { return make_double3(double(a.x), double(a.y), double(a.z)); }
inline __host__ __device__ double3 make_double3(double2 a, double s) { return make_double3(a.x, a.y, s); }

inline __host__ __device__ int3 make_int3(int    s) { return make_int3(s, s, s); }
inline __host__ __device__ int3 make_int3(int2   a) { return make_int3(a.x, a.y, 0); }
inline __host__ __device__ int3 make_int3(uint3  a) { return make_int3(int(a.x), int(a.y), int(a.z)); }
inline __host__ __device__ int3 make_int3(float3 a) { return make_int3(int(a.x), int(a.y), int(a.z)); }
inline __host__ __device__ int3 make_int3(int2   a, int s) { return make_int3(a.x, a.y, s); }

inline __host__ __device__ uint3 make_uint3(uint  s) { return make_uint3(s, s, s); }
inline __host__ __device__ uint3 make_uint3(uint2 a) { return make_uint3(a.x, a.y, 0); }
inline __host__ __device__ uint3 make_uint3(uint4 a) { return make_uint3(a.x, a.y, a.z); }
inline __host__ __device__ uint3 make_uint3(int3  a) { return make_uint3(uint(a.x), uint(a.y), uint(a.z)); }
inline __host__ __device__ uint3 make_uint3(uint2 a, uint s) { return make_uint3(a.x, a.y, s); }

inline __host__ __device__ float4 make_float4(float  s) { return make_float4(s, s, s, s); }
inline __host__ __device__ float4 make_float4(float3 a) { return make_float4(a.x, a.y, a.z, 0.0f); }
inline __host__ __device__ float4 make_float4(int4   a) { return make_float4(float(a.x), float(a.y), float(a.z), float(a.w)); }
inline __host__ __device__ float4 make_float4(uint4  a) { return make_float4(float(a.x), float(a.y), float(a.z), float(a.w)); }
inline __host__ __device__ float4 make_float4(float3 a, float w) { return make_float4(a.x, a.y, a.z, w); }

inline __host__ __device__ double4 make_double4(double  s) { return make_double4(s, s, s, s); }
inline __host__ __device__ double4 make_double4(double3 a) { return make_double4(a.x, a.y, a.z, 0.0); }
inline __host__ __device__ double4 make_double4(int4    a) { return make_double4(double(a.x), double(a.y), double(a.z), double(a.w)); }
inline __host__ __device__ double4 make_double4(uint4   a) { return make_double4(double(a.x), double(a.y), double(a.z), double(a.w)); }
inline __host__ __device__ double4 make_double4(double3 a, double w) { return make_double4(a.x, a.y, a.z, w); }

inline __host__ __device__ int4 make_int4(int    s) { return make_int4(s, s, s, s); }
inline __host__ __device__ int4 make_int4(int3   a) { return make_int4(a.x, a.y, a.z, 0); }
inline __host__ __device__ int4 make_int4(uint4  a) { return make_int4(int(a.x), int(a.y), int(a.z), int(a.w)); }
inline __host__ __device__ int4 make_int4(float4 a) { return make_int4(int(a.x), int(a.y), int(a.z), int(a.w)); }
inline __host__ __device__ int4 make_int4(int3   a, int w) { return make_int4(a.x, a.y, a.z, w); }

inline __host__ __device__ uint4 make_uint4(uint  s) { return make_uint4(s, s, s, s); }
inline __host__ __device__ uint4 make_uint4(uint3 a) { return make_uint4(a.x, a.y, a.z, 0); }
inline __host__ __device__ uint4 make_uint4(int4  a) { return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w)); }
inline __host__ __device__ uint4 make_uint4(uint3 a, uint w){ return make_uint4(a.x, a.y, a.z, w); }


////////////////////////////////////////////////////////////////////////////////
// comparison
////////////////////////////////////////////////////////////////////////////////

// ==  (AND)
inline __host__ __device__ bool operator==(const int2    &v1, const int2    &v2) { return (v1.x == v2.x && v1.y == v2.y); }
inline __host__ __device__ bool operator==(const int3    &v1, const int3    &v2) { return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z); }
inline __host__ __device__ bool operator==(const int4    &v1, const int4    &v2) { return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z && v1.w == v2.w); }
inline __host__ __device__ bool operator==(const float2  &v1, const float2  &v2) { return (v1.x == v2.x && v1.y == v2.y); }
inline __host__ __device__ bool operator==(const float3  &v1, const float3  &v2) { return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z); }
inline __host__ __device__ bool operator==(const float4  &v1, const float4  &v2) { return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z && v1.w == v2.w); }
inline __host__ __device__ bool operator==(const double2 &v1, const double2 &v2) { return (v1.x == v2.x && v1.y == v2.y); }
inline __host__ __device__ bool operator==(const double3 &v1, const double3 &v2) { return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z); }
inline __host__ __device__ bool operator==(const double4 &v1, const double4 &v2) { return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z && v1.w == v2.w); }
// !=  (OR)
inline __host__ __device__ bool operator!=(const int2    &v1, const int2    &v2) { return (v1.x != v2.x || v1.y != v2.y); }
inline __host__ __device__ bool operator!=(const int3    &v1, const int3    &v2) { return (v1.x != v2.x || v1.y != v2.y || v1.z != v2.z); }
inline __host__ __device__ bool operator!=(const int4    &v1, const int4    &v2) { return (v1.x != v2.x || v1.y != v2.y || v1.z != v2.z || v1.w != v2.w); }
inline __host__ __device__ bool operator!=(const float2  &v1, const float2  &v2) { return (v1.x != v2.x || v1.y != v2.y); }
inline __host__ __device__ bool operator!=(const float3  &v1, const float3  &v2) { return (v1.x != v2.x || v1.y != v2.y || v1.z != v2.z); }
inline __host__ __device__ bool operator!=(const float4  &v1, const float4  &v2) { return (v1.x != v2.x || v1.y != v2.y || v1.z != v2.z || v1.w != v2.w); }
inline __host__ __device__ bool operator!=(const double2 &v1, const double2 &v2) { return (v1.x != v2.x || v1.y != v2.y); }
inline __host__ __device__ bool operator!=(const double3 &v1, const double3 &v2) { return (v1.x != v2.x || v1.y != v2.y || v1.z != v2.z); }
inline __host__ __device__ bool operator!=(const double4 &v1, const double4 &v2) { return (v1.x != v2.x || v1.y != v2.y || v1.z != v2.z || v1.w != v2.w); }

// > VECTOR (AND)
inline __host__ __device__ bool operator>(const int2    &v1, const int2    &v2) { return (v1.x > v2.x && v1.y > v2.y); }
inline __host__ __device__ bool operator>(const int3    &v1, const int3    &v2) { return (v1.x > v2.x && v1.y > v2.y && v1.z > v2.z); }
inline __host__ __device__ bool operator>(const int4    &v1, const int4    &v2) { return (v1.x > v2.x && v1.y > v2.y && v1.z > v2.z && v1.w > v2.w); }
inline __host__ __device__ bool operator>(const float2  &v1, const float2  &v2) { return (v1.x > v2.x && v1.y > v2.y); }
inline __host__ __device__ bool operator>(const float3  &v1, const float3  &v2) { return (v1.x > v2.x && v1.y > v2.y && v1.z > v2.z); }
inline __host__ __device__ bool operator>(const float4  &v1, const float4  &v2) { return (v1.x > v2.x && v1.y > v2.y && v1.z > v2.z && v1.w > v2.w); }
inline __host__ __device__ bool operator>(const double2 &v1, const double2 &v2) { return (v1.x > v2.x && v1.y > v2.y); }
inline __host__ __device__ bool operator>(const double3 &v1, const double3 &v2) { return (v1.x > v2.x && v1.y > v2.y && v1.z > v2.z); }
inline __host__ __device__ bool operator>(const double4 &v1, const double4 &v2) { return (v1.x > v2.x && v1.y > v2.y && v1.z > v2.z && v1.w > v2.w); }
// < VECTOR (AND)
inline __host__ __device__ bool operator<(const int2    &v1, const int2    &v2) { return (v1.x < v2.x && v1.y < v2.y); }
inline __host__ __device__ bool operator<(const int3    &v1, const int3    &v2) { return (v1.x < v2.x && v1.y < v2.y && v1.z < v2.z); }
inline __host__ __device__ bool operator<(const int4    &v1, const int4    &v2) { return (v1.x < v2.x && v1.y < v2.y && v1.z < v2.z && v1.w < v2.w); }
inline __host__ __device__ bool operator<(const float2  &v1, const float2  &v2) { return (v1.x < v2.x && v1.y < v2.y); }
inline __host__ __device__ bool operator<(const float3  &v1, const float3  &v2) { return (v1.x < v2.x && v1.y < v2.y && v1.z < v2.z); }
inline __host__ __device__ bool operator<(const float4  &v1, const float4  &v2) { return (v1.x < v2.x && v1.y < v2.y && v1.z < v2.z && v1.w < v2.w); }
inline __host__ __device__ bool operator<(const double2 &v1, const double2 &v2) { return (v1.x < v2.x && v1.y < v2.y); }
inline __host__ __device__ bool operator<(const double3 &v1, const double3 &v2) { return (v1.x < v2.x && v1.y < v2.y && v1.z < v2.z); }
inline __host__ __device__ bool operator<(const double4 &v1, const double4 &v2) { return (v1.x < v2.x && v1.y < v2.y && v1.z < v2.z && v1.w < v2.w); }
// >= VECTOR (AND)
inline __host__ __device__ bool operator>=(const int2    &v1, const int2    &v2) { return (v1.x >= v2.x && v1.y >= v2.y); }
inline __host__ __device__ bool operator>=(const int3    &v1, const int3    &v2) { return (v1.x >= v2.x && v1.y >= v2.y && v1.z >= v2.z); }
inline __host__ __device__ bool operator>=(const int4    &v1, const int4    &v2) { return (v1.x >= v2.x && v1.y >= v2.y && v1.z >= v2.z && v1.w >= v2.w); }
inline __host__ __device__ bool operator>=(const float2  &v1, const float2  &v2) { return (v1.x >= v2.x && v1.y >= v2.y); }
inline __host__ __device__ bool operator>=(const float3  &v1, const float3  &v2) { return (v1.x >= v2.x && v1.y >= v2.y && v1.z >= v2.z); }
inline __host__ __device__ bool operator>=(const float4  &v1, const float4  &v2) { return (v1.x >= v2.x && v1.y >= v2.y && v1.z >= v2.z && v1.w >= v2.w); }
inline __host__ __device__ bool operator>=(const double2 &v1, const double2 &v2) { return (v1.x >= v2.x && v1.y >= v2.y); }
inline __host__ __device__ bool operator>=(const double3 &v1, const double3 &v2) { return (v1.x >= v2.x && v1.y >= v2.y && v1.z >= v2.z); }
inline __host__ __device__ bool operator>=(const double4 &v1, const double4 &v2) { return (v1.x >= v2.x && v1.y >= v2.y && v1.z >= v2.z && v1.w >= v2.w); }
// <= VECTOR (AND)
inline __host__ __device__ bool operator<=(const int2    &v1, const int2    &v2) { return (v1.x <= v2.x && v1.y <= v2.y); }
inline __host__ __device__ bool operator<=(const int3    &v1, const int3    &v2) { return (v1.x <= v2.x && v1.y <= v2.y && v1.z <= v2.z); }
inline __host__ __device__ bool operator<=(const int4    &v1, const int4    &v2) { return (v1.x <= v2.x && v1.y <= v2.y && v1.z <= v2.z && v1.w <= v2.w); }
inline __host__ __device__ bool operator<=(const float2  &v1, const float2  &v2) { return (v1.x <= v2.x && v1.y <= v2.y); }
inline __host__ __device__ bool operator<=(const float3  &v1, const float3  &v2) { return (v1.x <= v2.x && v1.y <= v2.y && v1.z <= v2.z); }
inline __host__ __device__ bool operator<=(const float4  &v1, const float4  &v2) { return (v1.x <= v2.x && v1.y <= v2.y && v1.z <= v2.z && v1.w <= v2.w); }
inline __host__ __device__ bool operator<=(const double2 &v1, const double2 &v2) { return (v1.x <= v2.x && v1.y <= v2.y); }
inline __host__ __device__ bool operator<=(const double3 &v1, const double3 &v2) { return (v1.x <= v2.x && v1.y <= v2.y && v1.z <= v2.z); }
inline __host__ __device__ bool operator<=(const double4 &v1, const double4 &v2) { return (v1.x <= v2.x && v1.y <= v2.y && v1.z <= v2.z && v1.w <= v2.w); }

// > SCALAR (AND)
inline __host__ __device__ bool operator> (const int2    &v1, int    s) { return (v1.x >  s && v1.y >  s); }
inline __host__ __device__ bool operator> (const int3    &v1, int    s) { return (v1.x >  s && v1.y >  s && v1.z >  s); }
inline __host__ __device__ bool operator> (const int4    &v1, int    s) { return (v1.x >  s && v1.y >  s && v1.z >  s && v1.w >  s); }
inline __host__ __device__ bool operator> (const float2  &v1, float  s) { return (v1.x >  s && v1.y >  s); }
inline __host__ __device__ bool operator> (const float3  &v1, float  s) { return (v1.x >  s && v1.y >  s && v1.z >  s); }
inline __host__ __device__ bool operator> (const float4  &v1, float  s) { return (v1.x >  s && v1.y >  s && v1.z >  s && v1.w >  s); }
inline __host__ __device__ bool operator> (const double2 &v1, double s) { return (v1.x >  s && v1.y >  s); }
inline __host__ __device__ bool operator> (const double3 &v1, double s) { return (v1.x >  s && v1.y >  s && v1.z >  s); }
inline __host__ __device__ bool operator> (const double4 &v1, double s) { return (v1.x >  s && v1.y >  s && v1.z >  s && v1.w >  s); }
// < SCALAR (AND)
inline __host__ __device__ bool operator< (const int2    &v1, int    s) { return (v1.x <  s && v1.y <  s); }
inline __host__ __device__ bool operator< (const int3    &v1, int    s) { return (v1.x <  s && v1.y <  s && v1.z <  s); }
inline __host__ __device__ bool operator< (const int4    &v1, int    s) { return (v1.x <  s && v1.y <  s && v1.z <  s && v1.w <  s); }
inline __host__ __device__ bool operator< (const float2  &v1, float  s) { return (v1.x <  s && v1.y <  s); }
inline __host__ __device__ bool operator< (const float3  &v1, float  s) { return (v1.x <  s && v1.y <  s && v1.z <  s); }
inline __host__ __device__ bool operator< (const float4  &v1, float  s) { return (v1.x <  s && v1.y <  s && v1.z <  s && v1.w <  s); }
inline __host__ __device__ bool operator< (const double2 &v1, double s) { return (v1.x <  s && v1.y <  s); }
inline __host__ __device__ bool operator< (const double3 &v1, double s) { return (v1.x <  s && v1.y <  s && v1.z <  s); }
inline __host__ __device__ bool operator< (const double4 &v1, double s) { return (v1.x <  s && v1.y <  s && v1.z <  s && v1.w <  s); }
// >= SCALAR (AND)
inline __host__ __device__ bool operator>=(const int2    &v1, int    s) { return (v1.x >= s && v1.y >= s); }
inline __host__ __device__ bool operator>=(const int3    &v1, int    s) { return (v1.x >= s && v1.y >= s && v1.z >= s); }
inline __host__ __device__ bool operator>=(const int4    &v1, int    s) { return (v1.x >= s && v1.y >= s && v1.z >= s && v1.w >= s); }
inline __host__ __device__ bool operator>=(const float2  &v1, float  s) { return (v1.x >= s && v1.y >= s); }
inline __host__ __device__ bool operator>=(const float3  &v1, float  s) { return (v1.x >= s && v1.y >= s && v1.z >= s); }
inline __host__ __device__ bool operator>=(const float4  &v1, float  s) { return (v1.x >= s && v1.y >= s && v1.z >= s && v1.w >= s); }
inline __host__ __device__ bool operator>=(const double2 &v1, double s) { return (v1.x >= s && v1.y >= s); }
inline __host__ __device__ bool operator>=(const double3 &v1, double s) { return (v1.x >= s && v1.y >= s && v1.z >= s); }
inline __host__ __device__ bool operator>=(const double4 &v1, double s) { return (v1.x >= s && v1.y >= s && v1.z >= s && v1.w >= s); }
// <= SCALAR (AND)
inline __host__ __device__ bool operator<=(const int2    &v1, int    s) { return (v1.x <= s && v1.y <= s); }
inline __host__ __device__ bool operator<=(const int3    &v1, int    s) { return (v1.x <= s && v1.y <= s && v1.z <= s); }
inline __host__ __device__ bool operator<=(const int4    &v1, int    s) { return (v1.x <= s && v1.y <= s && v1.z <= s && v1.w <= s); }
inline __host__ __device__ bool operator<=(const float2  &v1, float  s) { return (v1.x <= s && v1.y <= s); }
inline __host__ __device__ bool operator<=(const float3  &v1, float  s) { return (v1.x <= s && v1.y <= s && v1.z <= s); }
inline __host__ __device__ bool operator<=(const float4  &v1, float  s) { return (v1.x <= s && v1.y <= s && v1.z <= s && v1.w <= s); }
inline __host__ __device__ bool operator<=(const double2 &v1, double s) { return (v1.x <= s && v1.y <= s); }
inline __host__ __device__ bool operator<=(const double3 &v1, double s) { return (v1.x <= s && v1.y <= s && v1.z <= s); }
inline __host__ __device__ bool operator<=(const double4 &v1, double s) { return (v1.x <= s && v1.y <= s && v1.z <= s && v1.w <= s); }


////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float clamp(float f, float a, float b) { return fmaxf(a, fminf(f, b)); }
inline __device__ __host__ int   clamp(int   f, int   a, int   b) { return  imax(a,  imin(f, b)); }
inline __device__ __host__ uint  clamp(uint  f, uint  a, uint  b) { return  imax(a,  imin(f, b)); }

inline __device__ __host__ float2 clamp(float2 v, float  a, float  b) { return make_float2(clamp(v.x, a,   b),   clamp(v.y, a,   b));   }
inline __device__ __host__ float2 clamp(float2 v, float2 a, float2 b) { return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y)); }
inline __device__ __host__ float3 clamp(float3 v, float  a, float  b) { return make_float3(clamp(v.x, a,   b),   clamp(v.y, a,   b),
                                                                                           clamp(v.z, a,   b));   }
inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b) { return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y),
                                                                                           clamp(v.z, a.z, b.z)); }
inline __device__ __host__ float4 clamp(float4 v, float  a, float  b) { return make_float4(clamp(v.x, a,   b),   clamp(v.y, a,   b),
                                                                                           clamp(v.z, a,   b),   clamp(v.w, a,   b));   }
inline __device__ __host__ float4 clamp(float4 v, float4 a, float4 b) { return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y),
                                                                                           clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w)); }

inline __device__ __host__ int2 clamp(int2 v, int a, int b)   { return make_int2(clamp(v.x, a, b), clamp(v.y, a, b)); }
inline __device__ __host__ int2 clamp(int2 v, int2 a, int2 b) { return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y)); }
inline __device__ __host__ int3 clamp(int3 v, int a, int b)   { return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b)); }
inline __device__ __host__ int3 clamp(int3 v, int3 a, int3 b) { return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z)); }
inline __device__ __host__ int4 clamp(int4 v, int a, int b)   { return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b)); }
inline __device__ __host__ int4 clamp(int4 v, int4 a, int4 b) { return make_int4(clamp(v.x, a.x,b.x), clamp(v.y, a.y, b.y),
                                                                                 clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w)); }

inline __device__ __host__ uint2 clamp(uint2 v, uint a, uint b)   { return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b)); }
inline __device__ __host__ uint2 clamp(uint2 v, uint2 a, uint2 b) { return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y)); }
inline __device__ __host__ uint3 clamp(uint3 v, uint a, uint b)   { return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b)); }
inline __device__ __host__ uint3 clamp(uint3 v, uint3 a, uint3 b) { return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z)); }
inline __device__ __host__ uint4 clamp(uint4 v, uint a, uint b)   { return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b)); }
inline __device__ __host__ uint4 clamp(uint4 v, uint4 a, uint4 b) { return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y),
                                                                                      clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w)); }


// ISNAN
template<typename T> __device__ inline bool isnan(T v)                { return v != v; }
template<>           __device__ inline bool isnan<float2> (float2 v)  { return (isnan(v.x) || isnan(v.y)); }
template<>           __device__ inline bool isnan<float3> (float3 v)  { return (isnan(v.x) || isnan(v.y) || isnan(v.z)); }
template<>           __device__ inline bool isnan<float4> (float4 v)  { return (isnan(v.x) || isnan(v.y) || isnan(v.z) || isnan(v.w)); }
template<>           __device__ inline bool isnan<double2>(double2 v) { return (isnan(v.x) || isnan(v.y)); }
template<>           __device__ inline bool isnan<double3>(double3 v) { return (isnan(v.x) || isnan(v.y) || isnan(v.z)); }
template<>           __device__ inline bool isnan<double4>(double4 v) { return (isnan(v.x) || isnan(v.y) || isnan(v.z) || isnan(v.w)); }

// ISINF
template<typename T> __device__ inline bool isinf(T v);
template<>           __device__ inline bool isinf<float>  (float v)   { return v > FLT_MAX; }
template<>           __device__ inline bool isinf<float2> (float2 v)  { return (isinf(v.x) || isinf(v.y)); }
template<>           __device__ inline bool isinf<float3> (float3 v)  { return (isinf(v.x) || isinf(v.y) || isinf(v.z)); }
template<>           __device__ inline bool isinf<float4> (float4 v)  { return (isinf(v.x) || isinf(v.y) || isinf(v.z) || isinf(v.w)); }
template<>           __device__ inline bool isinf<double> (double v)  { return v > DBL_MAX; }
template<>           __device__ inline bool isinf<double2>(double2 v) { return (isinf(v.x) || isinf(v.y)); }
template<>           __device__ inline bool isinf<double3>(double3 v) { return (isinf(v.x) || isinf(v.y) || isinf(v.z)); }
template<>           __device__ inline bool isinf<double4>(double4 v) { return (isinf(v.x) || isinf(v.y) || isinf(v.z) || isinf(v.w)); }

// LINEAR INTERPOLATION
template<typename T, typename A=float> __device__ T lerp(T x0, T x1, A alpha) { return x1*alpha + x0*(1.0f-alpha); }
// BILINEAR INTERPOLATION
template<typename T, typename A=float> T blerp(const T &p00, const T &p01, const T &p10, const T &p11, const A &alpha2)
{ return lerp(lerp(p00, p01, alpha2.x), lerp(p10, p11, alpha2.x), alpha2.y); }

////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////
inline __device__ __host__ float smoothstep(float a, float b, float x)
{ float y = clamp((x - a) / (b - a), 0.0f, 1.0f); return (y*y*(3.0f - (2.0f*y))); }
inline __device__ __host__ float2 smoothstep(float2 a, float2 b, float2 x)
{ float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f); return (y*y*(make_float2(3.0f) - (make_float2(2.0f)*y))); }
inline __device__ __host__ float3 smoothstep(float3 a, float3 b, float3 x)
{ float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f); return (y*y*(make_float3(3.0f) - (make_float3(2.0f)*y))); }
inline __device__ __host__ float4 smoothstep(float4 a, float4 b, float4 x)
{ float4 y = clamp((x - a) / (b - a), 0.0f, 1.0f); return (y*y*(make_float4(3.0f) - (make_float4(2.0f)*y))); }

// cross product
inline __host__ __device__ float3  cross(const float3  &a, const float3  &b)
{ return float3 {a.y, a.z, a.x} * float3 {b.z, b.x, b.y} - float3 {a.z, a.x, a.y} * float3 {b.y, b.z, b.x}; }
inline __host__ __device__ double3 cross(const double3 &a, const double3 &b)
{ return double3{a.y, a.z, a.x} * double3{b.z, b.x, b.y} - double3{a.z, a.x, a.y} * double3{b.y, b.z, b.x}; }


#endif // CUDA_VECTOR_OPERATORS_H
