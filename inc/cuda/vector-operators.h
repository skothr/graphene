#ifndef CUDA_VECTOR_OPERATORS_H
#define CUDA_VECTOR_OPERATORS_H

#include <cuda_runtime.h>
#include <vector_types.h>
#include <ostream>
#include <cmath>
#include <functional>
#include <float.h>
#include <cstdint>

#include "vector.hpp"




// constexpr test for built-in CUDA vector types
template<typename T> struct is_cuda  : std::false_type { };
template<> struct is_cuda<char2>      : std::true_type { }; // char/uchar
template<> struct is_cuda<uchar2>     : std::true_type { };
template<> struct is_cuda<char3>      : std::true_type { };
template<> struct is_cuda<uchar3>     : std::true_type { };
template<> struct is_cuda<char4>      : std::true_type { };
template<> struct is_cuda<uchar4>     : std::true_type { };
template<> struct is_cuda<short2>     : std::true_type { }; // short/ushort
template<> struct is_cuda<ushort2>    : std::true_type { };
template<> struct is_cuda<short3>     : std::true_type { };
template<> struct is_cuda<ushort3>    : std::true_type { };
template<> struct is_cuda<short4>     : std::true_type { };
template<> struct is_cuda<ushort4>    : std::true_type { };
template<> struct is_cuda<int2>       : std::true_type { }; // int/uint
template<> struct is_cuda<uint2>      : std::true_type { };
template<> struct is_cuda<int3>       : std::true_type { };
template<> struct is_cuda<uint3>      : std::true_type { };
template<> struct is_cuda<int4>       : std::true_type { };
template<> struct is_cuda<uint4>      : std::true_type { };
template<> struct is_cuda<long2>      : std::true_type { }; // long/ulong
template<> struct is_cuda<ulong2>     : std::true_type { };
template<> struct is_cuda<long3>      : std::true_type { };
template<> struct is_cuda<ulong3>     : std::true_type { };
template<> struct is_cuda<long4>      : std::true_type { };
template<> struct is_cuda<ulong4>     : std::true_type { };
template<> struct is_cuda<longlong2>  : std::true_type { }; // longlong/ulonglong
template<> struct is_cuda<ulonglong2> : std::true_type { };
template<> struct is_cuda<longlong3>  : std::true_type { };
template<> struct is_cuda<ulonglong3> : std::true_type { };
template<> struct is_cuda<longlong4>  : std::true_type { };
template<> struct is_cuda<ulonglong4> : std::true_type { };
template<> struct is_cuda<float2>     : std::true_type { }; // float/double
template<> struct is_cuda<double2>    : std::true_type { };
template<> struct is_cuda<float3>     : std::true_type { };
template<> struct is_cuda<double3>    : std::true_type { };
template<> struct is_cuda<float4>     : std::true_type { };
template<> struct is_cuda<double4>    : std::true_type { };
template<typename T> constexpr bool is_cuda_v = is_cuda<T>::value; // (helper)

// convert <SCALAR, N> to cuda vector
template<typename T, int N> struct cuda_vec_t;
template<> struct cuda_vec_t<char,               2> { typedef char2      type; };
template<> struct cuda_vec_t<unsigned char,      2> { typedef uchar2     type; };
template<> struct cuda_vec_t<char,               3> { typedef char3      type; };
template<> struct cuda_vec_t<unsigned char,      3> { typedef uchar3     type; };
template<> struct cuda_vec_t<char,               4> { typedef char4      type; };
template<> struct cuda_vec_t<unsigned char,      4> { typedef uchar4     type; };
template<> struct cuda_vec_t<short,              2> { typedef short2     type; };
template<> struct cuda_vec_t<unsigned short,     2> { typedef ushort2    type; };
template<> struct cuda_vec_t<short,              3> { typedef short3     type; };
template<> struct cuda_vec_t<unsigned short,     3> { typedef ushort3    type; };
template<> struct cuda_vec_t<short,              4> { typedef short4     type; };
template<> struct cuda_vec_t<unsigned short,     4> { typedef ushort4    type; };
template<> struct cuda_vec_t<int,                2> { typedef int2       type; };
template<> struct cuda_vec_t<unsigned int,       2> { typedef uint2      type; };
template<> struct cuda_vec_t<int,                3> { typedef int3       type; };
template<> struct cuda_vec_t<unsigned int,       3> { typedef uint3      type; };
template<> struct cuda_vec_t<int,                4> { typedef int4       type; };
template<> struct cuda_vec_t<unsigned int,       4> { typedef uint4      type; };
template<> struct cuda_vec_t<long,               2> { typedef long2      type; };
template<> struct cuda_vec_t<unsigned long,      2> { typedef ulong2     type; };
template<> struct cuda_vec_t<long,               3> { typedef long3      type; };
template<> struct cuda_vec_t<unsigned long,      3> { typedef ulong3     type; };
template<> struct cuda_vec_t<long,               4> { typedef long4      type; };
template<> struct cuda_vec_t<unsigned long,      4> { typedef ulong4     type; };
template<> struct cuda_vec_t<long long,          2> { typedef longlong2  type; };
template<> struct cuda_vec_t<unsigned long long, 2> { typedef ulonglong2 type; };
template<> struct cuda_vec_t<long long,          3> { typedef longlong3  type; };
template<> struct cuda_vec_t<unsigned long long, 3> { typedef ulonglong3 type; };
template<> struct cuda_vec_t<long long,          4> { typedef longlong4  type; };
template<> struct cuda_vec_t<unsigned long long, 4> { typedef ulonglong4 type; };
template<> struct cuda_vec_t<float,              2> { typedef float2     type; };
template<> struct cuda_vec_t<double,             2> { typedef double2    type; };
template<> struct cuda_vec_t<float,              3> { typedef float3     type; };
template<> struct cuda_vec_t<double,             3> { typedef double3    type; };
template<> struct cuda_vec_t<float,              4> { typedef float4     type; };
template<> struct cuda_vec_t<double,             4> { typedef double4    type; };


// true if type has has an "x" member
template<typename T, typename=int> struct has_x                             : std::false_type { };
template<typename T>               struct has_x<T, decltype((void)T::x, 0)> : std::true_type  { };
// true if type has has an "y" member
template<typename T, typename=int> struct has_y                             : std::false_type { };
template<typename T>               struct has_y<T, decltype((void)T::y, 0)> : std::true_type  { };
// true if type has has an "z" member
template<typename T, typename=int> struct has_z                             : std::false_type { };
template<typename T>               struct has_z<T, decltype((void)T::z, 0)> : std::true_type  { };
// true if type has has an "w" member
template<typename T, typename=int> struct has_w                             : std::false_type { };
template<typename T>               struct has_w<T, decltype((void)T::w, 0)> : std::true_type  { };
// x/y/z/w member helpers
template<typename T> constexpr bool has_x_v = has_x<T>::value;
template<typename T> constexpr bool has_y_v = has_y<T>::value;
template<typename T> constexpr bool has_z_v = has_z<T>::value;
template<typename T> constexpr bool has_w_v = has_w<T>::value;

// returns number of elements in a cuda vector (intN/floatN/doubleN/etc.)
//  e.g.    int N = cudaN<T>();
template<typename T, typename std::enable_if_t<is_cuda_v<T>>* = nullptr>
int constexpr cvN()
{
  if constexpr     (has_w_v<T>) { return 4; }
  else if constexpr(has_z_v<T>) { return 3; }
  else if constexpr(has_y_v<T>) { return 2; }
  else if constexpr(has_x_v<T>) { return 1; }
  return 0;
}

// cuda vector info struct (from cuda vector, or scalar type and dimensionality)   TODO: more concise/flexible specialization
template<typename T, int N_=1> struct cuda_vec { static constexpr int N = N_; typedef T VT; typedef T BASE; typedef T LOWER; typedef int IT; };
template<> struct cuda_vec<int2,    1> { static constexpr int N = cvN<int2>();    typedef int    BASE; typedef int2    VT; typedef int2 IT; typedef int     LOWER; };
template<> struct cuda_vec<int3,    1> { static constexpr int N = cvN<int3>();    typedef int    BASE; typedef int3    VT; typedef int3 IT; typedef int2    LOWER; };
template<> struct cuda_vec<int4,    1> { static constexpr int N = cvN<int4>();    typedef int    BASE; typedef int4    VT; typedef int4 IT; typedef int3    LOWER; };
template<> struct cuda_vec<float2,  1> { static constexpr int N = cvN<float2>();  typedef float  BASE; typedef float2  VT; typedef int2 IT; typedef float   LOWER; };
template<> struct cuda_vec<float3,  1> { static constexpr int N = cvN<float3>();  typedef float  BASE; typedef float3  VT; typedef int3 IT; typedef float2  LOWER; };
template<> struct cuda_vec<float4,  1> { static constexpr int N = cvN<float4>();  typedef float  BASE; typedef float4  VT; typedef int4 IT; typedef float3  LOWER; };
template<> struct cuda_vec<double2, 1> { static constexpr int N = cvN<double2>(); typedef double BASE; typedef double2 VT; typedef int2 IT; typedef double  LOWER; };
template<> struct cuda_vec<double3, 1> { static constexpr int N = cvN<double3>(); typedef double BASE; typedef double3 VT; typedef int3 IT; typedef double2 LOWER; };
template<> struct cuda_vec<double4, 1> { static constexpr int N = cvN<double4>(); typedef double BASE; typedef double4 VT; typedef int4 IT; typedef double3 LOWER; };
// cuda_vec<BASE, N> --> inherit from cuda_vec<VECTOR, 1> 
template<> struct cuda_vec<int,    2> : cuda_vec <int2,    1> { };
template<> struct cuda_vec<int,    3> : cuda_vec <int3,    1> { };
template<> struct cuda_vec<int,    4> : cuda_vec <int4,    1> { };
template<> struct cuda_vec<float,  2> : cuda_vec <float2,  1> { };
template<> struct cuda_vec<float,  3> : cuda_vec <float3,  1> { };
template<> struct cuda_vec<float,  4> : cuda_vec <float4,  1> { };
template<> struct cuda_vec<double, 2> : cuda_vec <double2, 1> { };
template<> struct cuda_vec<double, 3> : cuda_vec <double3, 1> { };
template<> struct cuda_vec<double, 4> : cuda_vec <double4, 1> { };

//// (extensions of vector.hpp)
// specializations of is_any_vec for cuda vectors
template<> struct is_any_vec<char2>      : std::true_type {};
template<> struct is_any_vec<uchar2>     : std::true_type {}; // char
template<> struct is_any_vec<char3>      : std::true_type {};
template<> struct is_any_vec<uchar3>     : std::true_type {};
template<> struct is_any_vec<char4>      : std::true_type {};
template<> struct is_any_vec<uchar4>     : std::true_type {};
template<> struct is_any_vec<short2>     : std::true_type {};
template<> struct is_any_vec<ushort2>    : std::true_type {}; // short
template<> struct is_any_vec<short3>     : std::true_type {};
template<> struct is_any_vec<ushort3>    : std::true_type {};
template<> struct is_any_vec<short4>     : std::true_type {};
template<> struct is_any_vec<ushort4>    : std::true_type {};
template<> struct is_any_vec<int2>       : std::true_type {};
template<> struct is_any_vec<uint2>      : std::true_type {}; // int
template<> struct is_any_vec<int3>       : std::true_type {};
template<> struct is_any_vec<uint3>      : std::true_type {};
template<> struct is_any_vec<int4>       : std::true_type {};
template<> struct is_any_vec<uint4>      : std::true_type {};
template<> struct is_any_vec<long2>      : std::true_type {};
template<> struct is_any_vec<ulong2>     : std::true_type {}; // long
template<> struct is_any_vec<long3>      : std::true_type {};
template<> struct is_any_vec<ulong3>     : std::true_type {};
template<> struct is_any_vec<long4>      : std::true_type {};
template<> struct is_any_vec<ulong4>     : std::true_type {};
template<> struct is_any_vec<longlong2>  : std::true_type {};
template<> struct is_any_vec<ulonglong2> : std::true_type {}; // long long
template<> struct is_any_vec<longlong3>  : std::true_type {};
template<> struct is_any_vec<ulonglong3> : std::true_type {};
template<> struct is_any_vec<longlong4>  : std::true_type {};
template<> struct is_any_vec<ulonglong4> : std::true_type {};
template<> struct is_any_vec<float2>     : std::true_type {};
template<> struct is_any_vec<double2>    : std::true_type {}; // float/double
template<> struct is_any_vec<float3>     : std::true_type {};
template<> struct is_any_vec<double3>    : std::true_type {};
template<> struct is_any_vec<float4>     : std::true_type {};
 template<> struct is_any_vec<double4>   : std::true_type {};
// specialization of any_vec info struct for cuda vectors
template<typename T> struct any_vec<T, typename std::enable_if_t<is_cuda_v<T>>> : cuda_vec<T,1> { };
// convert Vector<T,N> to cuda vector
template<typename T, int N> std::enable_if_t<N==2, typename cuda_vec<T,N>::VT> to_cuda(const Vector<T,N> &v)
{ return typename cuda_vec<T,N>::VT{ v.x, v.y }; }
template<typename T, int N> std::enable_if_t<N==3, typename cuda_vec<T,N>::VT> to_cuda(const Vector<T,N> &v)
{ return typename cuda_vec<T,N>::VT{ v.x, v.y, v.z }; }
template<typename T, int N> std::enable_if_t<N==4, typename cuda_vec<T,N>::VT> to_cuda(const Vector<T,N> &v)
{ return typename cuda_vec<T,N>::VT{ v.x, v.y, v.z, v.w }; }


// get pointer to first element for array-like access
template<typename T> __host__ __device__
std::enable_if_t<is_cuda_v<T>, const typename cuda_vec<T>::BASE*> arr(const T &v) { return &v.x; }
template<typename T> __host__ __device__
std::enable_if_t<is_cuda_v<T>,       typename cuda_vec<T>::BASE*> arr(      T &v) { return &v.x; }

 
// CREATE VECTOR TYPES (T-->output | U-->input)
//   VECTOR(U) --> VECTOR(T) : <X, Y, Z, [...]>
template<typename T, typename U, typename std::enable_if_t<is_cuda_v<U>>* = nullptr> // && std::greater<int>{}(cuda_vec<T>::N, 1)>* = nullptr>
__host__ __device__ inline auto makeV(U v) -> std::enable_if_t<cuda_vec<T>::N==cuda_vec<U>::N, T>
{
  typedef typename cuda_vec<T>::BASE TT;
  if      constexpr(cuda_vec<T>::N == 2) { return T{(TT)v.x, (TT)v.y}; }
  else if constexpr(cuda_vec<T>::N == 3) { return T{(TT)v.x, (TT)v.y, (TT)v.z}; }
  else if constexpr(cuda_vec<T>::N == 4) { return T{(TT)v.x, (TT)v.y, (TT)v.z, (TT)v.w}; }
  return T{(TT)v.x};
}
// single SCALAR(U) --> VECTOR(T) : <S, S, S, [...]>
template<typename T, typename U, typename std::enable_if_t<!is_cuda_v<U>>* = nullptr> // && std::greater<int>{}(cuda_vec<T>::N, 1)>* = nullptr>
__host__ __device__ inline auto makeV(U s) -> T
{
  typedef typename cuda_vec<T>::BASE TT;
  if      constexpr(cuda_vec<T>::N == 2) { return T{(TT)s, (TT)s}; }
  else if constexpr(cuda_vec<T>::N == 3) { return T{(TT)s, (TT)s, (TT)s}; }
  else if constexpr(cuda_vec<T>::N == 4) { return T{(TT)s, (TT)s, (TT)s, (TT)s}; }
  return T{(TT)s};
}

// SCALAR(U)[N] --> VECTOR(T) : <S0, S1, S2, [...]>
template<typename T, typename U>
__host__ __device__ inline auto makeV(U s1, U s2, U s3=0, U s4=0) -> typename std::enable_if_t<!is_cuda_v<U> && std::greater<int>{}(cuda_vec<T>::N, 1), T>
{
  typedef typename cuda_vec<T>::BASE TT;
  if      constexpr(cuda_vec<T>::N == 2) { return T{(TT)s1, (TT)s2}; }
  else if constexpr(cuda_vec<T>::N == 3) { return T{(TT)s1, (TT)s2, (TT)s3}; }
  else if constexpr(cuda_vec<T>::N == 4) { return T{(TT)s1, (TT)s2, (TT)s3, (TT)s4}; }
  return T{(TT)s1};
}



//// STRINGS
// ostream << VECTOR
template<typename T, int N = cuda_vec<T>::N, typename std::enable_if_t<std::greater<int>{}(N, 1)>* = nullptr>
__host__ std::ostream& operator<<(std::ostream &os, const T &v)
{
  os << "<";
  for(int i = 0; i < N; i++)
    { os << arr(v)[i] << (i == N-1 ? "" : ", "); }
  os << ">";
  return os;
}
// VECTOR >> istream
template<typename T, int N = cuda_vec<T>::N, typename std::enable_if_t<std::greater<int>{}(N, 1)>* = nullptr>
__host__ inline std::istream& operator>>(std::istream &is, T &v)
{
  char c; is.get(c);
  if(c == '<')
    {
      is >> v.x;
      if constexpr(N > 1) { is.ignore(1,','); is >> v.y; }
      if constexpr(N > 2) { is.ignore(1,','); is >> v.z; }
      if constexpr(N > 3) { is.ignore(1,','); is >> v.w; }
      is.ignore('>');
    }
  else
    { v = makeV<T>(0); }
  return is;
}
// to_string / from_string
template<typename T> __host__ std::enable_if_t<is_cuda_v<T>, std::string> to_string(const T &v, const int precision=6)
{ std::stringstream ss; ss.precision(precision); ss << v; return ss.str(); }
template<typename T> __host__ std::enable_if_t<is_cuda_v<T>, T> from_string(const std::string &str)
{ std::stringstream ss(str); T v; ss >> v; return v; }



//// ACCUMULATE (combine vector elements via functor)
template<typename T, template<typename> typename Op,
         std::enable_if_t<(!std::is_const_v<T> && std::greater<int>{}(cuda_vec<T>::N, 1))>* = nullptr>
__host__ __device__ auto vaccumulate(const T &v) // -> decltype(Op()(v.x, v.y))
{
  constexpr int N = cuda_vec<T>::N;
  typedef typename cuda_vec<T>::BASE TT;
  if      constexpr(cuda_vec<T>::N==2) { return Op()(v.x, v.y); }
  else if constexpr(cuda_vec<T>::N==3) { return Op()(v.x, Op()(v.y, v.z)); }
  else if constexpr(cuda_vec<T>::N==4) { return Op()(v.x, Op()(v.y, Op()(v.z, v.w))); }
  return v.x;
}

//// VCHECK --> check functors element-wise --> BOOL (integrated accumulate, default &&)
// check a functor (e.g. equal_to/min/etc.) as a unary operator(evaluate each element) + accumulator(combine results) --> (VECTOR) -> BOOL
template<typename T, template<typename> typename F, template<typename> typename Op=std::logical_and,
         std::enable_if_t<(!std::is_const_v<T> && std::greater<int>{}(cuda_vec<T>::N, 1))>* = nullptr>
__host__ __device__ auto vcheck(const T &v) // -> decltype(Op()(v.x, v.x))
{
  typedef typename cuda_vec<T>::BASE TT;
  if      constexpr(cuda_vec<T>::N==2) { return Op()(F<TT>()(v.x), F<TT>()(v.y)); }
  else if constexpr(cuda_vec<T>::N==3) { return Op()(F<TT>()(v.x), Op()(F<TT>()(v.y), F<TT>()(v.z))); }
  else if constexpr(cuda_vec<T>::N==4) { return Op()(F<TT>()(v.x), Op()(F<TT>()(v.y), Op()(F<TT>()(v.z), F<TT>()(v.w)))); }
  return F<TT>()(v.x);
}
// check a functor (e.g. equal_to/min/etc.) as a binary operator(compare respective elements) + accumulator(combine results) --> (VECTOR, VECTOR) -> BOOL
template<typename T, template<typename> typename F, template<typename> typename Op=std::logical_and,
         std::enable_if_t<(!std::is_const_v<T> && std::greater<int>{}(cuda_vec<T>::N, 1))>* = nullptr>
__host__ __device__ auto vcheck(const T &v1, const T &v2) // -> decltype(Op()(v1.x, v2.x))
{
  typedef typename cuda_vec<T>::BASE TT;
  if      constexpr(cuda_vec<T>::N==2) { return Op()(F<TT>()(v1.x, v2.x), F<TT>()(v1.y, v2.y)); }
  else if constexpr(cuda_vec<T>::N==3) { return Op()(F<TT>()(v1.x, v2.x), Op()(F<TT>()(v1.y, v2.y), F<TT>()(v1.z, v2.z))); }
  else if constexpr(cuda_vec<T>::N==4) { return Op()(F<TT>()(v1.x, v2.x), Op()(F<TT>()(v1.y, v2.y), Op()(F<TT>()(v1.z, v2.z), F<TT>()(v1.w, v2.w)))); }
  return F<TT>()(v1.x, v2.x);
}
// check a functor (e.g. equal_to/min/etc.) as a binary operator(compare elements to scalar) + accumulator(combine results) --> (VECTOR, SCALAR) -> BOOL
template<typename T, template<typename> typename F, template<typename> typename Op=std::logical_and,
         std::enable_if_t<(!std::is_const_v<T> && std::greater<int>{}(cuda_vec<T>::N, 1))>* = nullptr>
__host__ __device__ auto vcheck(const T &v, const typename cuda_vec<T>::BASE &s) // -> decltype(Op()(v.x, v.y))
{
  typedef typename cuda_vec<T>::BASE TT;
  if      constexpr(cuda_vec<T>::N==2) { return Op()(F<TT>()(v.x, s), F<TT>()(v.y, s)); }
  else if constexpr(cuda_vec<T>::N==3) { return Op()(F<TT>()(v.x, s), Op()(F<TT>()(v.y, s), F<TT>()(v.z, s))); }
  else if constexpr(cuda_vec<T>::N==4) { return Op()(F<TT>()(v.x, s), Op()(F<TT>()(v.y, s), Op()(F<TT>()(v.z, s), F<TT>()(v.w, s)))); }
  return F<TT>()(v.x, s);
}
// check a functor (e.g. equal_to/min/etc.) as a binary operator(compare elements to scalar) + accumulator(combine results) --> (SCALAR, VECTOR) -> BOOL
template<typename T, template<typename> typename F, template<typename> typename Op=std::logical_and,
         std::enable_if_t<(!std::is_const_v<T> && std::greater<int>{}(cuda_vec<T>::N, 1))>* = nullptr>
__host__ __device__ auto vcheck(const typename cuda_vec<T>::BASE &s, const T &v) // -> decltype(Op()(v.x, v.y))
{
  typedef typename cuda_vec<T>::BASE TT;
  if      constexpr(cuda_vec<T>::N==2) { return Op()(F<TT>()(s, v.x), F<TT>()(s, v.y)); }
  else if constexpr(cuda_vec<T>::N==3) { return Op()(F<TT>()(s, v.x), Op()(F<TT>()(s, v.y), F<TT>()(s, v.z))); }
  else if constexpr(cuda_vec<T>::N==4) { return Op()(F<TT>()(s, v.x), Op()(F<TT>()(s, v.y), Op()(F<TT>()(s, v.z), F<TT>()(s, v.w)))); }
  return F<TT>()(s, v.x);
}

//// VAPPLY (apply functors element-wise --> new VECTOR)
// apply a unary functor (e.g. std::negate) as a operator --> (VECTOR) -> VECTOR
template<typename T, template<typename> typename F>
__host__ __device__ auto vapply(const T &v) -> std::enable_if_t<!std::is_const_v<T> && std::greater<int>{}(cuda_vec<T>::N, 1), T>
{
  typedef typename cuda_vec<T>::BASE TT;
  if      constexpr(cuda_vec<T>::N==2) { return T{F<TT>()(v.x), F<TT>()(v.y)}; }
  else if constexpr(cuda_vec<T>::N==3) { return T{F<TT>()(v.x), F<TT>()(v.y), F<TT>()(v.z)}; }
  else if constexpr(cuda_vec<T>::N==4) { return T{F<TT>()(v.x), F<TT>()(v.y), F<TT>()(v.z), F<TT>()(v.w)}; }
  return T{F<TT>()(v.x)};
}
// apply a binary functor (e.g. std::plus/std::minus, etc.) as an operator --> (VECTOR, VECTOR) -> VECTOR
template<typename T, template<typename> typename F>
__host__ __device__ auto vapply(const T &v1, const T &v2) -> std::enable_if_t<!std::is_const_v<T> && std::greater<int>{}(cuda_vec<T>::N, 1), T>
{
  typedef typename cuda_vec<T>::BASE TT;
  if      constexpr(cuda_vec<T>::N==2) { return T{F<TT>()(v1.x, v2.x), F<TT>()(v1.y, v2.y)}; }
  else if constexpr(cuda_vec<T>::N==3) { return T{F<TT>()(v1.x, v2.x), F<TT>()(v1.y, v2.y), F<TT>()(v1.z, v2.z)}; }
  else if constexpr(cuda_vec<T>::N==4) { return T{F<TT>()(v1.x, v2.x), F<TT>()(v1.y, v2.y), F<TT>()(v1.z, v2.z), F<TT>()(v1.w, v2.w)}; }
  return T{F<TT>()(v1.x, v2.x)};
}
// apply a binary functor (e.g. std::plus/std::minus, etc.) as an operator --> (VECTOR, SCALAR) -> VECTOR
template<typename T, template<typename> typename F, typename TT=typename cuda_vec<T>::BASE>
__host__ __device__ auto vapply(const T &v, const TT &s) -> std::enable_if_t<!std::is_const_v<T> && std::greater<int>{}(cuda_vec<T>::N, 1), T>
{
  if      constexpr(cuda_vec<T>::N==2) { return T{F<TT>()(v.x, s), F<TT>()(v.y, s)}; }
  else if constexpr(cuda_vec<T>::N==3) { return T{F<TT>()(v.x, s), F<TT>()(v.y, s), F<TT>()(v.z, s)}; }
  else if constexpr(cuda_vec<T>::N==4) { return T{F<TT>()(v.x, s), F<TT>()(v.y, s), F<TT>()(v.z, s), F<TT>()(v.w, s)}; }
  return T{F<TT>()(v.x, s)};
}
// apply a binary functor (e.g. std::plus/std::minus, etc.) as an operator --> (SCALAR, VECTOR) -> VECTOR
template<typename T, template<typename> typename F, typename TT=typename cuda_vec<T>::BASE>
__host__ __device__ auto vapply(const TT &s, const T &v) -> std::enable_if_t<!std::is_const_v<T> && std::greater<int>{}(cuda_vec<T>::N, 1), T>
{
  if      constexpr(cuda_vec<T>::N==2) { return T{F<TT>()(s, v.x), F<TT>()(s, v.y)}; }
  else if constexpr(cuda_vec<T>::N==3) { return T{F<TT>()(s, v.x), F<TT>()(s, v.y), F<TT>()(s, v.z)}; }
  else if constexpr(cuda_vec<T>::N==4) { return T{F<TT>()(s, v.x), F<TT>()(s, v.y), F<TT>()(s, v.z), F<TT>()(s, v.w)}; }
  return T{F<TT>()(s, v.x)};
}


//// OPERATOR OVERLOADS
// -(VECTOR)
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T > operator- (const T &v)             { return vapply<T, std::negate>(v);    }
// VECTOR + VECTOR
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T > operator+ (const T &u, const T &v) { return vapply<T, std::plus>(u, v); }
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T&> operator+=(      T &u, const T &v) { u = u + v; return u; }
// VECTOR - VECTOR
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T > operator- (const T &u, const T &v) { return vapply<T, std::minus>(u, v); }
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T&> operator-=(      T &u, const T &v) { u = u - v; return u; }
// VECTOR * VECTOR
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T > operator* (const T &u, const T &v) { return vapply<T, std::multiplies>(u, v); }
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T&> operator*=(      T &u, const T &v) { u = u * v; return u; }
// VECTOR / VECTOR
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T > operator/ (const T &u, const T &v) { return vapply<T, std::divides>(u, v); }
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T&> operator/=(      T &u, const T &v) { u = u / v; return u; }

// VECTOR + SCALAR
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T > operator+ (const T &v, typename cuda_vec<T>::BASE s)
{ return vapply<T, std::plus>(v, s); }
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T&> operator+=(      T &v, typename cuda_vec<T>::BASE s)
{ v = v + s; return v; }
// VECTOR - SCALAR
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T > operator- (const T &v, typename cuda_vec<T>::BASE s)
{ return vapply<T, std::minus>(v, s); }
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T&> operator-=(      T &v, typename cuda_vec<T>::BASE s)
{ v = v - s; return v; }
// VECTOR * SCALAR
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T > operator* (const T &v, typename cuda_vec<T>::BASE s)
{ return vapply<T, std::multiplies>(v,s); }
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T&> operator*=(      T &v, typename cuda_vec<T>::BASE s)
{ v = v * s; return v; }
// VECTOR / SCALAR
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T > operator/ (const T &v, typename cuda_vec<T>::BASE s)
{ return vapply<T, std::divides>(v, s); }
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T&> operator/=(      T &v, typename cuda_vec<T>::BASE s)
{ v = v / s; return v; }

// SCALAR + VECTOR
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T > operator+ (typename cuda_vec<T>::BASE s, const T &v)
{ return vapply<T, std::plus>(s,v); }
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T&> operator+=(typename cuda_vec<T>::BASE s,       T &v)
{ v = s + v; return v; }
// SCALAR - VECTOR
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T > operator- (typename cuda_vec<T>::BASE s, const T &v)
{ return vapply<T, std::minus>(s,v); }
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T&> operator-=(typename cuda_vec<T>::BASE s,       T &v)
{ v = s - v; return v; }
// SCALAR * VECTOR
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T > operator* (typename cuda_vec<T>::BASE s, const T &v)
{ return vapply<T, std::multiplies>(s,v); }
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T&> operator*=(typename cuda_vec<T>::BASE s,       T &v)
{ v = s * v; return v; }
// SCALAR / VECTOR
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T > operator/ (typename cuda_vec<T>::BASE s, const T &v)
{ return vapply<T, std::divides>(s,v); }
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T&> operator/=(typename cuda_vec<T>::BASE s,       T &v)
{ v = s / v; return v; }


// SUM OF VECTOR ELEMENTS
template<typename T> __host__ __device__ typename cuda_vec<T>::BASE sum(const T &v)
{
  if constexpr(cuda_vec<T>::N==2) { return (v.x + v.y); }
  if constexpr(cuda_vec<T>::N==3) { return (v.x + v.y + v.z); }
  if constexpr(cuda_vec<T>::N==4) { return (v.x + v.y + v.z + v.w); }
  return v.x;
}
// PRODUCT OF VECTOR ELEMENTS
template<typename T> __host__ __device__ typename cuda_vec<T>::BASE mul(const T &v)
{
  if constexpr(cuda_vec<T>::N==2) { return (v.x * v.y); }
  if constexpr(cuda_vec<T>::N==3) { return (v.x * v.y * v.z); }
  if constexpr(cuda_vec<T>::N==4) { return (v.x * v.y * v.z * v.w); }
  return v.x;
}
// VECTOR · VECTOR
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, typename cuda_vec<T>::BASE> dot(const T &v1, const T &v2) { return sum(v1*v2); }

// MODULO
template<typename T> struct mod_f; // (type-specialized functor)
template<> struct mod_f<int>    { __host__ __device__ int    operator()(int    x1, int    x2) const   { return x1 % x2;       } };
template<> struct mod_f<float>  { __host__ __device__ float  operator()(float  x1, float  x2) const   { return fmodf(x1, x2); } };
template<> struct mod_f<double> { __host__ __device__ double operator()(double x1, double x2) const   { return fmod (x1, x2); } };
template<typename T> __host__ __device__ inline T operator%(const T &v1, const T &v2)                 { return vapply<T, mod_f>(v1, v2); }
template<typename T> __host__ __device__ inline T operator%(const T &v, typename cuda_vec<T>::BASE s) { return vapply<T, mod_f>(v, s); }
template<typename T> __host__ __device__ inline T operator%(typename cuda_vec<T>::BASE s, const T &v) { return vapply<T, mod_f>(s, v); }
// fmod function for cuda vectors
template<typename T, typename U> __host__ __device__ inline T fmod(const T &v1, const U &v2) { return (v1 % v2); }


// NAN (built-in function __host__ only, so using (x != x))
template<typename T> __host__ __device__ bool isnan(const T &v)
{
  if constexpr(cuda_vec<T>::N==2) { return ((v.x != v.x) || (v.y != v.y)); }
  if constexpr(cuda_vec<T>::N==3) { return ((v.x != v.x) || (v.y != v.y) || (v.z != v.z)); }
  if constexpr(cuda_vec<T>::N==4) { return ((v.x != v.x) || (v.y != v.y) || (v.z != v.z) || (v.w != v.w)); }
  return (v.x != v.x);
}
// INF
template<typename T> __host__ __device__ bool isinf(const T &v)
{
  if constexpr(cuda_vec<T>::N==2) { return (isinf(v.x) || isinf(v.y)); }
  if constexpr(cuda_vec<T>::N==3) { return (isinf(v.x) || isinf(v.y) || isinf(v.y)); }
  if constexpr(cuda_vec<T>::N==4) { return (isinf(v.x) || isinf(v.y) || isinf(v.z) || isinf(v.w)); }
  return isinf(v.x);
}

// FLOOR
template<typename T> struct floor_f; // (type-specialized functor)
template<> struct floor_f<float>  { __host__ __device__ float  operator()(float  x) const { return floorf(x); } };
template<> struct floor_f<double> { __host__ __device__ double operator()(double x) const { return floor (x); } };
template<typename T> __host__ __device__ T floor(const T &v) { return vapply<T, floor_f>(v); }
// CEIL
template<typename T> struct ceil_f; // (type-specialized functor)
template<> struct ceil_f<float>  { __host__ __device__ float  operator()(float  x) const { return ceilf(x); } };
template<> struct ceil_f<double> { __host__ __device__ double operator()(double x) const { return ceil (x); } };
template<typename T> __host__ __device__ T ceil(const T &v) { return vapply<T, ceil_f>(v); }
// ABS
template<typename T> struct abs_f { __host__ __device__ T operator()(const T &x) const { return abs(x); } }; // (scalar functor)
template<typename T> __host__ __device__ T abs(const T &v) { return vapply<T, abs_f>(v); }
// LOG
template<typename T> struct log_f; // (type-specialized functor)
template<> struct log_f<float>  { __host__ __device__ float  operator()(float  x) const { return logf(x); } };
template<> struct log_f<double> { __host__ __device__ double operator()(double x) const { return log (x); } };
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T> log(const T &v) { return vapply<T, log_f>(v); }
// EXP
template<typename T> struct exp_f { __host__ __device__ T operator()(const T &x) const { return exp(x); } }; // (scalar functor)
template<typename T> __host__ __device__ T exp(const T &v) { return vapply<T, exp_f>(v); }
// SIN
template<typename T> struct sin_f { __host__ __device__ T operator()(const T &x) const { return sin(x); } }; // (scalar functor)
template<typename T> __host__ __device__ T sin(const T &v) { return vapply<T, sin_f>(v); }
// COS
template<typename T> struct cos_f { __host__ __device__ T operator()(const T &x) const { return cos(x); } }; // (scalar functor)
template<typename T> __host__ __device__ T cos(const T &v) { return vapply<T, cos_f>(v); }
// TAN
template<typename T> struct tan_f { __host__ __device__ T operator()(const T &x) const { return tan(x); } }; // (scalar functor)
template<typename T> __host__ __device__ T tan(const T &v) { return vapply<T, tan_f>(v); }
// SQRT
template<typename T> struct sqrt_f { __host__ __device__ T operator()(const T &x) const { return sqrt(x); } }; // (scalar functor)
template<typename T> __host__ __device__ std::enable_if_t<is_cuda_v<T>, T> sqrt(const T &v) { return vapply<T, sqrt_f>(v); }


// POW
template<typename T> struct pow_f; // (type-specialized functor)
template<> struct pow_f<float>  { __host__ __device__ float  operator()(float  x, float  e) const { return powf(x, e); } };
template<> struct pow_f<double> { __host__ __device__ double operator()(double x, double e) const { return pow (x, e); } };
template<typename T> // pow(vector, vector)
__host__ __device__ T pow(const T &v, const T &e) { return vapply<T, pow_f>(v, e); }
template<typename T> // pow(vector, scalar)
__host__ __device__ T pow(const T &v, const typename cuda_vec<T>::BASE &e) { return vapply<T, pow_f>(v, e); }

// |VECTOR|^2
template<typename T> // scalar types -- return same value
__host__ __device__ inline std::enable_if_t<cuda_vec<T>::N==1, typename cuda_vec<T>::BASE> length2(const T &v) { return v*v; }
template<typename T> // vector types -- calculate
__host__ __device__ inline std::enable_if_t<cuda_vec<T>::N!=1, typename cuda_vec<T>::BASE> length2(const T &v) { return dot(v, v); }
// |VECTOR|
template<typename T> // scalar types -- return same value
__host__ __device__ inline std::enable_if_t<cuda_vec<T>::N==1, typename cuda_vec<T>::BASE> length (const T &v) { return v; }
template<typename T> // vector types -- calculate
__host__ __device__ inline std::enable_if_t<cuda_vec<T>::N!=1, typename cuda_vec<T>::BASE> length (const T &v) { return sqrt(length2(v)); }
// NORMALIZED VECTOR
template<typename T> __host__ __device__ inline std::enable_if_t<is_cuda_v<T>, T> normalize(const T &v) { return v / length(v); }

// MIN/MAX (?)
__host__ __device__ inline float fminf(float a, float b) { return a < b ? a : b; }
__host__ __device__ inline float fmaxf(float a, float b) { return a > b ? a : b; }
__host__ __device__ inline int   imax (int   a, int   b) { return a > b ? a : b; }
__host__ __device__ inline int   imin (int   a, int   b) { return a < b ? a : b; }

// MAX
template<typename T> struct max_f // (scalar functor)
{ __host__ __device__ T operator()(T x1, T x2) const { return ((x1 > x2) ? x1 : x2); } };

// max(VECTOR, VECTOR) -> VECTOR
template<typename T> __host__ __device__ inline std::enable_if_t<is_cuda_v<T> && cuda_vec<T>::N!=1, T>
max(T v1, T v2) { return vapply<T, max_f>(v1, v2); }
// max(VECTOR, SCALAR) -> VECTOR
template<typename T> __host__ __device__ inline std::enable_if_t<is_cuda_v<T> && cuda_vec<T>::N!=1, T>
max(T v, typename cuda_vec<T>::BASE s) { return vapply<T, max_f>(v, s); }
// max(SCALAR, VECTOR) -> VECTOR
template<typename T> __host__ __device__ inline std::enable_if_t<is_cuda_v<T> && cuda_vec<T>::N!=1, T>
max(typename cuda_vec<T>::BASE s, T v) { return vapply<T, max_f>(s, v); }
// max(VECTOR2>) -> SCALAR
template<typename T> __host__ __device__ inline std::enable_if_t<is_cuda_v<T> && cuda_vec<T>::N==2, typename cuda_vec<T>::BASE>
max(T v) { return std::max(v.x, v.y); }
// max(VECTOR3>) -> SCALAR
template<typename T> __host__ __device__ inline std::enable_if_t<is_cuda_v<T> && cuda_vec<T>::N==3, typename cuda_vec<T>::BASE>
max(T v) { return std::max(v.x, std::max(v.y, v.z)); }
// max(VECTOR4>) -> SCALAR
template<typename T> __host__ __device__ inline std::enable_if_t<is_cuda_v<T> && cuda_vec<T>::N==4, typename cuda_vec<T>::BASE>
max(T v) { return std::max(v.x, std::max(v.y, std::max(v.z, v.w))); }

// MIN
template<typename T> struct min_f // (scalar functor)
{ __host__ __device__ std::enable_if_t<!is_cuda_v<T>, T> operator()(T x1, T x2) const { return ((x1 < x2) ? x1 : x2); } };

// min(VECTOR, VECTOR) -> VECTOR
template<typename T> __host__ __device__ inline std::enable_if_t<is_cuda_v<T> && cuda_vec<T>::N!=1, T>
min(T v1, T v2) { return vapply<T, min_f, typename cuda_vec<T>::BASE>(v1, v2); }
// min(VECTOR, SCALAR) -> VECTOR
template<typename T> __host__ __device__ inline std::enable_if_t<is_cuda_v<T> && cuda_vec<T>::N!=1, T>
min(T s, typename cuda_vec<T>::BASE v) { return vapply<T, min_f>(v, s); }
// min(SCALAR, VECTOR) -> VECTOR
template<typename T> __host__ __device__ inline std::enable_if_t<is_cuda_v<T> && cuda_vec<T>::N!=1, T>
min(typename cuda_vec<T>::BASE v, T s) { return vapply<T, min_f>(s, v); }
// min(VECTOR<2>) -> SCALAR
template<typename T> __host__ __device__ inline auto min(T v) -> std::enable_if_t<is_cuda_v<T> && cuda_vec<T>::N==2, typename cuda_vec<T>::BASE>
{ return std::min(v.x, v.y); }
// min(VECTOR<3>) -> SCALAR
template<typename T> __host__ __device__ inline auto min(T v) -> std::enable_if_t<is_cuda_v<T> && cuda_vec<T>::N==3, typename cuda_vec<T>::BASE>
{ return std::min(v.x, std::min(v.y, v.z)); }
// min(VECTOR<4>) -> SCALAR
template<typename T> __host__ __device__ inline auto min(T v) -> std::enable_if_t<is_cuda_v<T> && cuda_vec<T>::N==4, typename cuda_vec<T>::BASE>
{ return std::min(v.x, std::min(v.y, std::min(v.z, v.w))); }


////////////////////////////////////////////////////////////////////////////////
// comparison
////////////////////////////////////////////////////////////////////////////////

// VECTOR == VECTOR (AND -- true only if all elements equal)
template<typename T>
__host__ __device__ auto operator ==(const T &v1, const T &v2) -> std::enable_if_t<(!std::is_const_v<T> && cuda_vec<T>::N!=1), bool>
{
  return vcheck<T, std::equal_to, std::logical_and>(v1, v2);
}
// VECTOR != VECTOR (OR -- true if any element not equal)
template<typename T>
__host__ __device__ auto operator !=(const T &v1, const T &v2) -> std::enable_if_t<(!std::is_const_v<T> && cuda_vec<T>::N!=1), bool>
{
  return vcheck<T, std::not_equal_to, std::logical_or>(v1, v2);
}
// VECTOR >  VECTOR (AND -- true only if all elements greater)
template<typename T>
__host__ __device__ auto operator > (const T &v1, const T &v2) -> std::enable_if_t<(!std::is_const_v<T> && cuda_vec<T>::N!=1), bool>
{
  return vcheck<T, std::greater, std::logical_and>(v1, v2);
}
// VECTOR <  VECTOR (AND -- true only if all elements lesser)
template<typename T>
__host__ __device__ auto operator < (const T &v1, const T &v2) -> std::enable_if_t<(!std::is_const_v<T> && cuda_vec<T>::N!=1), bool>
{
  return vcheck<T, std::less, std::logical_and>(v1, v2);
}
// VECTOR >= VECTOR (AND -- true only if all elements greater or equal)
template<typename T>
__host__ __device__ auto operator >=(const T &v1, const T &v2) -> std::enable_if_t<(!std::is_const_v<T> && cuda_vec<T>::N!=1), bool>
{
  return vcheck<T, std::greater_equal, std::logical_and>(v1, v2);
}
// VECTOR <= VECTOR (AND -- true only if all elements lesser or equal)
template<typename T>
__host__ __device__ auto operator <=(const T &v1, const T &v2) -> std::enable_if_t<(!std::is_const_v<T> && cuda_vec<T>::N!=1), bool>
{
  return vcheck<T, std::less_equal, std::logical_and>(v1, v2);
}

// VECTOR > SCALAR (AND -- true only if all elements greater)
template<typename T, typename TT=typename cuda_vec<T>::BASE>
__host__ __device__ auto operator > (const T &v, const TT &s) -> std::enable_if_t<(!std::is_const_v<T> && !std::is_const_v<TT> && cuda_vec<T>::N!=1), bool>
{
  return vcheck<T, std::greater, std::logical_and>(v, s);
}
// VECTOR < SCALAR (AND -- true only if all elements lesser)
template<typename T, typename TT=typename cuda_vec<T>::BASE>
__host__ __device__ auto operator < (const T &v, const TT &s) -> std::enable_if_t<(!std::is_const_v<T> && !std::is_const_v<TT> && cuda_vec<T>::N!=1), bool>
{
  return vcheck<T, std::less, std::logical_and>(v, s);
}
// VECTOR >= SCALAR (AND -- true only if all elements greater or equal)
template<typename T, typename TT=typename cuda_vec<T>::BASE>
__host__ __device__ auto operator >=(const T &v, const TT &s) -> std::enable_if_t<(!std::is_const_v<T> && !std::is_const_v<TT> && cuda_vec<T>::N!=1), bool>
{
  return vcheck<T, std::greater_equal, std::logical_and>(v, s);
}
// VECTOR <= SCALAR (AND -- true only if all elements lesser or equal)
template<typename T, typename TT=typename cuda_vec<T>::BASE>
__host__ __device__ auto operator <=(const T &v, const TT &s) -> std::enable_if_t<(!std::is_const_v<T> && !std::is_const_v<TT> && cuda_vec<T>::N!=1), bool>
{
  return vcheck<T, std::less_equal, std::logical_and>(v, s);
}

// SCALAR > VECTOR (AND -- true only if all elements greater)
template<typename T, typename TT=typename cuda_vec<T>::BASE>
__host__ __device__ auto operator > (const TT &s, const T &v) -> std::enable_if_t<(!std::is_const_v<T> && !std::is_const_v<TT> && cuda_vec<T>::N!=1), bool>
{
  return vcheck<T, std::greater, std::logical_and>(s, v);
}
// SCALAR < VECTOR (AND -- true only if all elements lesser)
template<typename T, typename TT=typename cuda_vec<T>::BASE>
__host__ __device__ auto operator < (const TT &s, const T &v) -> std::enable_if_t<(!std::is_const_v<T> && !std::is_const_v<TT> && cuda_vec<T>::N!=1), bool>
{
  return vcheck<T, std::less, std::logical_and>(s, v);
}
// SCALAR >= VECTOR (AND -- true only if all elements greater or equal)
template<typename T, typename TT=typename cuda_vec<T>::BASE>
__host__ __device__ auto operator >=(const TT &s, const T &v) -> std::enable_if_t<(!std::is_const_v<T> && !std::is_const_v<TT> && cuda_vec<T>::N!=1), bool>
{
  return vcheck<T, std::greater_equal, std::logical_and>(s, v);
}
// SCALAR <= VECTOR (AND -- true only if all elements lesser or equal)
template<typename T, typename TT=typename cuda_vec<T>::BASE>
__host__ __device__ auto operator <=(const TT &s, const T &v) -> std::enable_if_t<(!std::is_const_v<T> && !std::is_const_v<TT> && cuda_vec<T>::N!=1), bool>
{
  return vcheck<T, std::less_equal, std::logical_and>(s, v);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
// - 3D vector cross product (only float3/double3)
////////////////////////////////////////////////////////////////////////////////
template<typename T> inline __host__ __device__
auto cross(const T &a, const T &b) -> std::enable_if_t<(!std::is_const_v<T> && (std::is_same_v<T, float3> || std::is_same_v<T, double3>)), T>
{
  return T{a.y, a.z, a.x} * T{b.z, b.x, b.y} - T{a.z, a.x, a.y} * T{b.y, b.z, b.x};
}


#endif // CUDA_VECTOR_OPERATORS_H
