#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <string>
#include <array>
#include <cmath>
#include <istream>
#include <ostream>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <type_traits>
#include <limits>

#include <vector_types.h>
#include <cuda_runtime.h>


// forward declaration
template<typename T, int N> struct Vector;
// shorthand
template<typename T> using Vec1 = Vector<T, 1>;
template<typename T> using Vec2 = Vector<T, 2>;
template<typename T> using Vec3 = Vector<T, 3>;
template<typename T> using Vec4 = Vector<T, 4>;
template<int N> using Vecf = Vector<float, N>;
template<int N> using Vecd = Vector<double, N>;
typedef Vector<int, 1>         Vec1i;
typedef Vector<int, 2>         Vec2i;
typedef Vector<int, 3>         Vec3i;
typedef Vector<int, 4>         Vec4i;
typedef Vector<float, 1>       Vec1f;
typedef Vector<float, 2>       Vec2f;
typedef Vector<float, 3>       Vec3f;
typedef Vector<float, 4>       Vec4f;
typedef Vector<double, 1>      Vec1d;
typedef Vector<double, 2>      Vec2d;
typedef Vector<double, 3>      Vec3d;
typedef Vector<double, 4>      Vec4d;
typedef Vector<long double, 1> Vec1l;
typedef Vector<long double, 2> Vec2l;
typedef Vector<long double, 3> Vec3l;
typedef Vector<long double, 4> Vec4l;

// base template class
template<typename T, int N>
struct Vector
{
  std::array<T, N> data;
  
  Vector()                              : data{{0}}          { }
  Vector(const Vector<T, N> &other)     : Vector(other.data) { }
  Vector(const std::array<T, N> &data_) : data(data_)        { }
  Vector(T val)                         : data(N, val)       { }
  Vector(const std::string &str)        { fromString(str); }
  template<typename U> // convert from other type
  Vector(const Vector<U, N> &other)     { for(int i = 0; i < N; i++) { data[i] = (T)other.data[i]; } }
  // cuda structs
  Vector(const float2  &cv) { for(int i = 0; i < std::min(N, 2); i++) { data[i] = (T)( (const float*)(&cv))[i]; } }
  Vector(const float3  &cv) { for(int i = 0; i < std::min(N, 3); i++) { data[i] = (T)( (const float*)(&cv))[i]; } }
  Vector(const float4  &cv) { for(int i = 0; i < std::min(N, 4); i++) { data[i] = (T)( (const float*)(&cv))[i]; } }
  Vector(const double2 &cv) { for(int i = 0; i < std::min(N, 2); i++) { data[i] = (T)((const double*)(&cv))[i]; } }
  Vector(const double3 &cv) { for(int i = 0; i < std::min(N, 3); i++) { data[i] = (T)((const double*)(&cv))[i]; } }
  Vector(const double4 &cv) { for(int i = 0; i < std::min(N, 4); i++) { data[i] = (T)((const double*)(&cv))[i]; } }
  Vector<T, N>& operator=(const float2  &cv){ for(int i = 0; i < std::min(N, 2); i++) { data[i] = (T)((const float*) (&cv))[i]; } return *this; }
  Vector<T, N>& operator=(const float3  &cv){ for(int i = 0; i < std::min(N, 3); i++) { data[i] = (T)((const float*) (&cv))[i]; } return *this; }
  Vector<T, N>& operator=(const float4  &cv){ for(int i = 0; i < std::min(N, 4); i++) { data[i] = (T)((const float*) (&cv))[i]; } return *this; }
  Vector<T, N>& operator=(const double2 &cv){ for(int i = 0; i < std::min(N, 2); i++) { data[i] = (T)((const double*)(&cv))[i]; } return *this; }
  Vector<T, N>& operator=(const double3 &cv){ for(int i = 0; i < std::min(N, 3); i++) { data[i] = (T)((const double*)(&cv))[i]; } return *this; }
  Vector<T, N>& operator=(const double4 &cv){ for(int i = 0; i < std::min(N, 4); i++) { data[i] = (T)((const double*)(&cv))[i]; } return *this; }

  std::string toString() const            { std::ostringstream ss;      ss << (*this); return ss.str(); }
  void fromString(const std::string &str) { std::istringstream ss(str); ss >> (*this); }

  T& operator[](int dim)             { return data[dim]; }
  const T& operator[](int dim) const { return data[dim]; }
  
  Vector<T, N>& operator=(T scalar)                  { for(int i = 0; i < N; i++) { data[i] = scalar; } return *this; }
  Vector<T, N>& operator=(const Vector<T, N> &other) { for(int i = 0; i < N; i++) { data[i] = other.data[i]; } return *this; }
  bool operator==(const Vector<T, N> &other) const
  {
    for(int i = 0; i < N; i++)
      { if(data[i] != other.data[i]) return false; }
    return true;
  }
  bool operator!=(const Vector<T, N> &other) const { return !(*this == other); }

  // <, > (AND)
  bool operator> (const Vector<T, N> &other) const { for(int i = 0; i < N; i++) { if(data[i] <= other.data[i]) { return false; } } return true; }
  bool operator< (const Vector<T, N> &other) const { for(int i = 0; i < N; i++) { if(data[i] >= other.data[i]) { return false; } } return true; }
  bool operator>=(const Vector<T, N> &other) const { for(int i = 0; i < N; i++) { if(data[i] < other.data[i])  { return false; } } return true; }
  bool operator<=(const Vector<T, N> &other) const { for(int i = 0; i < N; i++) { if(data[i] > other.data[i])  { return false; } } return true; }
  bool operator> (T scalar) const { for(int i = 0; i < N; i++) { if(data[i] <= scalar) { return false; } } return true; }
  bool operator< (T scalar) const { for(int i = 0; i < N; i++) { if(data[i] >= scalar) { return false; } } return true; }
  bool operator>=(T scalar) const { for(int i = 0; i < N; i++) { if(data[i] <  scalar) { return false; } } return true; }
  bool operator<=(T scalar) const { for(int i = 0; i < N; i++) { if(data[i] >  scalar) { return false; } } return true; }

  Vector<T, N>& operator+=(const Vector<T, N> &other) { for(int i = 0; i < N; i++) { data[i] += other.data[i]; } return *this; }
  Vector<T, N>& operator-=(const Vector<T, N> &other) { for(int i = 0; i < N; i++) { data[i] -= other.data[i]; } return *this; }

#ifndef __NVCC__
  Vector<T, N>& operator%=(const T &s)
  {
    for(int i = 0; i < N; i++) { if constexpr(std::is_same<T, int>::value) { data[i] %= s; } else { data[i] = fmod(data[i], s); } }
    return *this;
  }
  Vector<T, N>& operator%=(const Vector<T, N> &other)
  {
    for(int i = 0; i < N; i++) { if constexpr(std::is_same<T, int>::value) { data[i] %= other[i]; } else { data[i] = fmod(data[i], other[i]); } }
    return *this;
  }
  Vector<T, N> operator%(const T &s) const                { Vector<T, N> result(*this); return (result %= s); }
  Vector<T, N> operator%(const Vector<T, N> &other) const { Vector<T, N> result(*this); return (result %= other); }
#endif
  
  void ceil()  { for(auto &d : data) { d = std::ceil(d);  } }
  void floor() { for(auto &d : data) { d = std::floor(d); } }
  Vector<T, N> getCeil() const  { Vector<T, N> v(*this); v.ceil(); return v; }
  Vector<T, N> getFloor() const { Vector<T, N> v(*this); v.floor(); return v; }

  T length2() const { T sqsum = T(); for(auto d : data) { sqsum += d*d; } return sqsum; }
  T length() const  { return sqrt(length2()); }
  
  void normalize()                { (*this) /= length(); }
  Vector<T, N> normalized() const { return Vector<T, N>(*this) / length(); }
  
  template<typename U>
  T dot(const Vector<U, N> &other) const
  {
    T total = 0;
    for(int i = 0; i < N; i++) { total += data[i] * other.data[i]; }
    return total;
  }

  Vector mod(const T &s) const; // { Vector result(*this); for(int i = 0; i < N; i++) { result[i] = fmod(result[i], s); } return result; }
};

template<typename T, int N>
std::ostream& operator<<(std::ostream &os, const Vector<T, N> &v)
{
  os << "<";
  for(int i = 0; i < N; i++)
    { os << v.data[i] << ((i < N-1) ? ", " : ""); }
  os << ">";
  return os;
}
template<typename T, int N>
std::istream& operator>>(std::istream &is, Vector<T, N> &v)
{
  is.ignore(1,'<');
  for(int i = 0; i < N; i++) { is >> v.data[i]; is.ignore(1,','); }
  is.ignore(1,'>');
  return is;
}

template<typename T>
struct Vector<T, 2>
{
  static constexpr int N = 2;
  union { struct { T x, y; }; std::array<T, N> data; };

  Vector()                              : x((T)0), y((T)0)          { }
  Vector(T x_, T y_)                    : x(x_), y(y_)              { }
  Vector(const Vector<T, N> &other)     : x(other.x), y(other.y)    { }
  Vector(const std::array<T, N> &data_) : x(data_[0]), y(data_[1])  { }
  Vector(T val)                         : x(val), y(val)            { }
  Vector(const std::string &str)        { fromString(str); }
  template<typename U> // convert from other type
  Vector(const Vector<U, N> &other)     { for(int i = 0; i < N; i++) { data[i] = (T)other.data[i]; } }
  // cuda structs
  Vector(const int2    &cv) : x((T)cv.x), y((T)cv.y) { }
  Vector(const float2  &cv) : x((T)cv.x), y((T)cv.y) { }
  Vector(const double2 &cv) : x((T)cv.x), y((T)cv.y) { }
  Vector<T, 2>& operator=(const Vector<T, 2> &other) { data = other.data; return *this; }
  Vector<T, N>& operator=(T scalar)                  { for(int i = 0; i < N; i++) { data[i] = scalar; } return *this; }
  Vector<T, N>& operator=(const float2 &cv)  { x = (T)cv.x; y = (T)cv.y; return *this; }
  Vector<T, N>& operator=(const double2 &cv) { x = (T)cv.x; y = (T)cv.y; return *this; }

  T& operator[](int dim)             { return data[dim]; }
  const T& operator[](int dim) const { return data[dim]; }

  // swizzle
  Vector<T, 2> xx() const { return Vector<T, 2>(x, x); }
  Vector<T, 2> yy() const { return Vector<T, 2>(y, y); }
  Vector<T, 2> xy() const { return Vector<T, 2>(x, y); }
  Vector<T, 2> yx() const { return Vector<T, 2>(y, x); }
  
  std::string toString() const            { std::ostringstream ss;      ss << (*this); return ss.str(); }
  void fromString(const std::string &str) { std::istringstream ss(str); ss >> (*this); }
  
  bool operator==(const Vector<T, N> &other) const
  {
    for(int i = 0; i < N; i++)
      { if(data[i] != other.data[i]) return false; }
    return true;
  }
  bool operator!=(const Vector<T, N> &other) const { return !(*this == other); }

  // <, > (AND)
  bool operator> (const Vector<T, N> &other) const { for(int i = 0; i < N; i++) { if(data[i] <= other.data[i]) { return false; } } return true; }
  bool operator< (const Vector<T, N> &other) const { for(int i = 0; i < N; i++) { if(data[i] >= other.data[i]) { return false; } } return true; }
  bool operator>=(const Vector<T, N> &other) const { for(int i = 0; i < N; i++) { if(data[i] < other.data[i])  { return false; } } return true; }
  bool operator<=(const Vector<T, N> &other) const { for(int i = 0; i < N; i++) { if(data[i] > other.data[i])  { return false; } } return true; }
  bool operator> (T scalar) const { for(int i = 0; i < N; i++) { if(data[i] <= scalar) { return false; } } return true; }
  bool operator< (T scalar) const { for(int i = 0; i < N; i++) { if(data[i] >= scalar) { return false; } } return true; }
  bool operator>=(T scalar) const { for(int i = 0; i < N; i++) { if(data[i] <  scalar) { return false; } } return true; }
  bool operator<=(T scalar) const { for(int i = 0; i < N; i++) { if(data[i] >  scalar) { return false; } } return true; }

  Vector<T, N>& operator+=(const Vector<T, N> &other) { for(int i = 0; i < N; i++) { data[i] += other.data[i]; } return *this; }
  Vector<T, N>  operator+ (const Vector<T, N> &other) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] += other.data[i]; }
    return result;
  }
  Vector<T, N>& operator-=(const Vector<T, N> &other) { for(int i = 0; i < N; i++) { data[i] -= other.data[i]; } return *this; }
  Vector<T, N>  operator- (const Vector<T, N> &other) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] -= other.data[i]; }
    return result;
  }
  Vector<T, N>& operator*=(const Vector<T, N> &other) { for(int i = 0; i < N; i++) { data[i] *= other.data[i]; } return *this; }
  Vector<T, N>  operator* (const Vector<T, N> &other) const
  {
    Vector<T, N> result(data);
    for(int i = 0; i < N; i++) { result.data[i] *= other.data[i]; }
    return result;
  }
  Vector<T, N>& operator/=(const Vector<T, N> &other) { for(int i = 0; i < N; i++) { data[i] /= other.data[i]; } return *this; }
  Vector<T, N>  operator/ (const Vector<T, N> &other) const
  {
    Vector<T, N> result(data);
    for(int i = 0; i < N; i++) { result.data[i] /= other.data[i]; }
    return result;
  }
  
  Vector<T, N>& operator*=(T scalar) { for(int i = 0; i < N; i++) { data[i] *= scalar; } return *this; }
  Vector<T, N>  operator* (T scalar) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] *= scalar; }
    return result;
  }
  Vector<T, N>& operator/=(T scalar) { for(int i = 0; i < N; i++) { data[i] /= scalar; } return *this; }
  Vector<T, N>  operator/ (T scalar) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] /= scalar; }
    return result;
  }

#ifndef __NVCC__
  Vector<T, N>& operator%=(const T &s)
  {
    for(int i = 0; i < N; i++) { if constexpr(std::is_same<T, int>::value) { data[i] %= s; } else { data[i] = fmod(data[i], s); } }
    return *this;
  }
  Vector<T, N>& operator%=(const Vector<T, N> &other)
  {
    for(int i = 0; i < N; i++) { if constexpr(std::is_same<T, int>::value) { data[i] %= other[i]; } else { data[i] = fmod(data[i], other[i]); } }
    return *this;
  }
  Vector<T, N> operator%(const T &s) const                { Vector<T, N> result(*this); return (result %= s); }
  Vector<T, N> operator%(const Vector<T, N> &other) const { Vector<T, N> result(*this); return (result %= other); }
#endif
  
  void ceil()  { for(auto &d : data) { d = std::ceil(d);  } }
  void floor() { for(auto &d : data) { d = std::floor(d); } }
  Vector<T, N> getCeil() const  { Vector<T, N> v(*this);  v.ceil();  return v; }
  Vector<T, N> getFloor() const { Vector<T, N> v = *this; v.floor(); return v; }
  
  T length2() const { T sqsum = T(); for(auto d : data) { sqsum += d*d; } return sqsum; }
  T length()  const { return sqrt(length2()); }
  
  void normalize()                { (*this) /= length(); }
  Vector<T, N> normalized() const { return Vector<T, N>(*this) / length(); }
  
  template<typename U>
  T dot(const Vector<U, N> &other) const
  {
    T total = 0;
    for(int i = 0; i < N; i++) { total += data[i] * other.data[i]; }
    return total;
  }
};
  
template<typename T>
struct Vector<T, 3>
{
  static constexpr int N = 3;
  union { struct { T x, y, z; }; std::array<T, N> data; };

  Vector()                              : x((T)0), y((T)0), z((T)0)               { }
  Vector(T x_, T y_, T z_)              : x(x_), y(y_), z(z_)                     { }
  Vector(const Vector<T, N> &other)     : x(other.x), y(other.y), z(other.z)      { }
  Vector(const std::array<T, N> &data_) : x(data_[0]), y(data_[1]), z(data_[2])   { }
  Vector(T val)                         : x(val), y(val), z(val)                  { }
  Vector(const std::string &str)        { fromString(str); }
  template<typename U> // convert from other type
  Vector(const Vector<U, N> &other)     { for(int i = 0; i < N; i++) { data[i] = (T)other.data[i]; } }
  // cuda structs
  Vector(const int3    &cv) : x((T)cv.x), y((T)cv.y), z((T)cv.z) { }
  Vector(const float3  &cv) : x((T)cv.x), y((T)cv.y), z((T)cv.z) { }
  Vector(const double3 &cv) : x((T)cv.x), y((T)cv.y), z((T)cv.z) { }
  Vector<T, 2>& operator=(const Vector<T, 2> &other) { data = other.data; return *this; }
  Vector<T, N>& operator=(T scalar)                  { for(int i = 0; i < N; i++) { data[i] = scalar; }    return *this; }
  Vector<T, N>& operator=(const float3 &cv)  { x = (T)cv.x; y = (T)cv.y; z = (T)cv.z; return *this; }
  Vector<T, N>& operator=(const double3 &cv) { x = (T)cv.x; y = (T)cv.y; z = (T)cv.z; return *this; }
  
  T& operator[](int dim)             { return data[dim]; }
  const T& operator[](int dim) const { return data[dim]; }

  // swizzle
  Vector<T,2> xx() const {return Vector<T, 2>(x, x);} Vector<T,2> yy() const {return Vector<T, 2>(y, y);} Vector<T,2> zz() const {return Vector<T, 2>(z, z);}
  Vector<T,2> xy() const {return Vector<T, 2>(x, y);} Vector<T,2> yx() const {return Vector<T, 2>(y, x);} Vector<T,2> zx() const {return Vector<T, 2>(z, x);}
  Vector<T,2> xz() const {return Vector<T, 2>(x, z);} Vector<T,2> yz() const {return Vector<T, 2>(y, z);} Vector<T,2> zy() const {return Vector<T, 2>(z, y);}
  Vector<T, 3> xxx() const { return Vector<T, 3>(x, x, x); } Vector<T, 3> yyy() const { return Vector<T, 3>(y, y, y); }
  Vector<T, 3> zzz() const { return Vector<T, 3>(z, z, z); }

  Vector<T, 3> xyz() const { return Vector<T, 3>(x, y, z); } Vector<T, 3> xzy() const { return Vector<T, 3>(x, z, y); }
  Vector<T, 3> yxz() const { return Vector<T, 3>(y, x, z); } Vector<T, 3> yzx() const { return Vector<T, 3>(y, z, x); }
  Vector<T, 3> zxy() const { return Vector<T, 3>(z, x, y); } Vector<T, 3> zyx() const { return Vector<T, 3>(z, y, x); }
  
  Vector<T, 3> xxy() const { return Vector<T, 3>(x, x, y); } Vector<T, 3> xyx() const { return Vector<T, 3>(x, y, x); }
  Vector<T, 3> yxx() const { return Vector<T, 3>(y, x, x); } Vector<T, 3> xxz() const { return Vector<T, 3>(x, x, z); }
  Vector<T, 3> xzx() const { return Vector<T, 3>(x, z, x); } Vector<T, 3> zxx() const { return Vector<T, 3>(z, x, x); }

  Vector<T, 3> yyx() const { return Vector<T, 3>(y, y, x); } Vector<T, 3> yxy() const { return Vector<T, 3>(y, x, y); }
  Vector<T, 3> xyy() const { return Vector<T, 3>(x, y, y); } Vector<T, 3> yyz() const { return Vector<T, 3>(y, y, z); }
  Vector<T, 3> yzy() const { return Vector<T, 3>(y, z, y); } Vector<T, 3> zyy() const { return Vector<T, 3>(z, y, y); }
  
  Vector<T, 3> zzx() const { return Vector<T, 3>(z, z, x); } Vector<T, 3> zxz() const { return Vector<T, 3>(z, x, z); }
  Vector<T, 3> xzz() const { return Vector<T, 3>(x, z, z); } Vector<T, 3> zzy() const { return Vector<T, 3>(z, z, y); }
  Vector<T, 3> zyz() const { return Vector<T, 3>(z, y, z); } Vector<T, 3> yzz() const { return Vector<T, 3>(y, z, z); }
    
  std::string toString() const            { std::ostringstream ss; ss << (*this); return ss.str(); }
  void fromString(const std::string &str) { std::istringstream ss(str); ss >> (*this); }

  bool operator==(const Vector<T, N> &other) const
  {
    for(int i = 0; i < N; i++)
      { if(data[i] != other.data[i]) return false; }
    return true;
  }
  bool operator!=(const Vector<T, N> &other) const { return !(*this == other); }
  
  // <, > (AND)
  bool operator> (const Vector<T, N> &other) const { for(int i = 0; i < N; i++) { if(data[i] <= other.data[i]) { return false; } } return true; }
  bool operator< (const Vector<T, N> &other) const { for(int i = 0; i < N; i++) { if(data[i] >= other.data[i]) { return false; } } return true; }
  bool operator>=(const Vector<T, N> &other) const { for(int i = 0; i < N; i++) { if(data[i] < other.data[i])  { return false; } } return true; }
  bool operator<=(const Vector<T, N> &other) const { for(int i = 0; i < N; i++) { if(data[i] > other.data[i])  { return false; } } return true; }
  bool operator> (T scalar) const { for(int i = 0; i < N; i++) { if(data[i] <= scalar) { return false; } } return true; }
  bool operator< (T scalar) const { for(int i = 0; i < N; i++) { if(data[i] >= scalar) { return false; } } return true; }
  bool operator>=(T scalar) const { for(int i = 0; i < N; i++) { if(data[i] <  scalar) { return false; } } return true; }
  bool operator<=(T scalar) const { for(int i = 0; i < N; i++) { if(data[i] >  scalar) { return false; } } return true; }

  Vector<T, N>& operator+=(const Vector<T, N> &other) { for(int i = 0; i < N; i++) { data[i] += other.data[i]; } return *this; }
  Vector<T, N>  operator+ (const Vector<T, N> &other) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++)
      { result.data[i] += other.data[i]; }
    return result;
  }
  Vector<T, N>& operator-=(const Vector<T, N> &other) { for(int i = 0; i < N; i++) { data[i] -= other.data[i]; } return *this; }
  Vector<T, N>  operator- (const Vector<T, N> &other) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] -= other.data[i]; }
    return result;
  }  
  Vector<T, N>& operator*=(const Vector<T, N> &other) { for(int i = 0; i < N; i++) { data[i] *= other.data[i]; } return *this; }
  Vector<T, N>  operator* (const Vector<T, N> &other) const
  {
    Vector<T, N> result(data);
    for(int i = 0; i < N; i++) { result.data[i] *= other.data[i]; }
    return result;
  }
  Vector<T, N>& operator/=(const Vector<T, N> &other) { for(int i = 0; i < N; i++) { data[i] /= other.data[i]; } return *this; }
  Vector<T, N>  operator/ (const Vector<T, N> &other) const
  {
    Vector<T, N> result(data);
    for(int i = 0; i < N; i++) { result.data[i] /= other.data[i]; }
    return result;
  }

  Vector<T, N>& operator*=(T scalar) { for(int i = 0; i < N; i++) { data[i] *= scalar; } return *this; }
  Vector<T, N>  operator* (T scalar) const
  {
    Vector<T, N> result(data);
    for(int i = 0; i < N; i++) { result.data[i] *= scalar; }
    return result;
  }
  Vector<T, N>& operator/=(T scalar) { for(int i = 0; i < N; i++) { data[i] /= scalar; } return *this; }
  Vector<T, N>  operator/ (T scalar) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] /= scalar; }
    return result;
  }
  
#ifndef __NVCC__
  Vector<T, N>& operator%=(const T &s)
  {
    for(int i = 0; i < N; i++) { if constexpr(std::is_same<T, int>::value) { data[i] %= s; } else { data[i] = fmod(data[i], s); } }
    return *this;
  }
  Vector<T, N>& operator%=(const Vector<T, N> &other)
  {
    for(int i = 0; i < N; i++) { if constexpr(std::is_same<T, int>::value) { data[i] %= other[i]; } else { data[i] = fmod(data[i], other[i]); } }
    return *this;
  }
  Vector<T, N> operator%(const T &s) const                { Vector<T, N> result(*this); return (result %= s); }
  Vector<T, N> operator%(const Vector<T, N> &other) const { Vector<T, N> result(*this); return (result %= other); }  
#endif
  
  void ceil()                   { for(auto &d : data) { d = std::ceil(d);  } }
  void floor()                  { for(auto &d : data) { d = std::floor(d); } }
  Vector<T, N> getCeil() const  { Vector<T, N> v(*this); v.ceil();  return v; }
  Vector<T, N> getFloor() const { Vector<T, N> v(*this); v.floor(); return v; }

  T length2() const { T sqsum = T(); for(auto d : data) { sqsum += d*d; } return sqsum; }
  T length() const  { return sqrt(length2()); }

  void normalize()                { (*this) /= length(); }
  Vector<T, N> normalized() const { return Vector<T, N>(*this) / length(); }

  template<typename U>
  T dot(const Vector<U, N> &other) const
  {
    T total = 0;
    for(int i = 0; i < N; i++) { total += data[i] * other.data[i]; }
    return total;
  }
};

template<typename T>
struct Vector<T, 4>
{
  static constexpr int N = 4;
  union { struct { T x, y, z, w; }; std::array<T, N> data; };

  Vector()                              : x((T)0), y((T)0), z((T)0), w((T)1)                 { }
  Vector(T x_, T y_, T z_, T w_)        : x(x_), y(y_), z(z_), w(w_)                         { }
  Vector(const Vector<T, N> &other)     : x(other.x), y(other.y), z(other.z), w(other.w)     { }
  Vector(const std::array<T, N> &data_) : x(data_[0]), y(data_[1]), z(data_[2]), w(data_[3]) { }
  Vector(T val)                         : x(val), y(val), z(val), w(val)                     { }
  Vector(const std::string &str)        { fromString(str); }
  template<typename U> // convert from other type
  Vector(const Vector<U, N> &other)     { for(int i = 0; i < N; i++) { data[i] = (T)other.data[i]; } }

  // cuda structs
  Vector(const int4    &cv) : x((T)cv.x), y((T)cv.y), z((T)cv.z), w((T)cv.w) { }
  Vector(const float4  &cv) : x((T)cv.x), y((T)cv.y), z((T)cv.z), w((T)cv.w) { }
  Vector(const double4 &cv) : x((T)cv.x), y((T)cv.y), z((T)cv.z), w((T)cv.w) { }
  Vector<T, 4>& operator=(const Vector<T, 4> &other) { data = other.data; return *this; }
  Vector<T, N>& operator=(T scalar)                  { for(int i = 0; i < N; i++) { data[i] = scalar; } return *this; }
  Vector<T, N>& operator=(const float4 &cv)  { x = (T)cv.x; y = (T)cv.y; z = (T)cv.z; w = (T)cv.w; return *this; }
  Vector<T, N>& operator=(const double4 &cv) { x = (T)cv.x; y = (T)cv.y; z = (T)cv.z; w = (T)cv.w; return *this; }
  
  T& operator[](int dim)             { return data[dim]; }
  const T& operator[](int dim) const { return data[dim]; }
  
  // swizzle (TODO: etc.)
  Vector<T,2> xx() const {return Vector<T, 2>(x, x);} Vector<T,2> yy() const {return Vector<T, 2>(y, y);} Vector<T,2> zz() const {return Vector<T, 2>(z, z);}
  Vector<T,2> xy() const {return Vector<T, 2>(x, y);} Vector<T,2> yx() const {return Vector<T, 2>(y, x);} Vector<T,2> zx() const {return Vector<T, 2>(z, x);}
  Vector<T,2> xz() const {return Vector<T, 2>(x, z);} Vector<T,2> yz() const {return Vector<T, 2>(y, z);} Vector<T,2> zy() const {return Vector<T, 2>(z, y);}
  Vector<T, 3> xxx() const { return Vector<T, 3>(x, x, x); } Vector<T, 3> yyy() const { return Vector<T, 3>(y, y, y); }
  Vector<T, 3> zzz() const { return Vector<T, 3>(z, z, z); }

  Vector<T, 3> xyz() const { return Vector<T, 3>(x, y, z); } Vector<T, 3> xzy() const { return Vector<T, 3>(x, z, y); }
  Vector<T, 3> yxz() const { return Vector<T, 3>(y, x, z); } Vector<T, 3> yzx() const { return Vector<T, 3>(y, z, x); }
  Vector<T, 3> zxy() const { return Vector<T, 3>(z, x, y); } Vector<T, 3> zyx() const { return Vector<T, 3>(z, y, x); }
  
  Vector<T, 3> xxy() const { return Vector<T, 3>(x, x, y); } Vector<T, 3> xyx() const { return Vector<T, 3>(x, y, x); }
  Vector<T, 3> yxx() const { return Vector<T, 3>(y, x, x); } Vector<T, 3> xxz() const { return Vector<T, 3>(x, x, z); }
  Vector<T, 3> xzx() const { return Vector<T, 3>(x, z, x); } Vector<T, 3> zxx() const { return Vector<T, 3>(z, x, x); }

  Vector<T, 3> yyx() const { return Vector<T, 3>(y, y, x); } Vector<T, 3> yxy() const { return Vector<T, 3>(y, x, y); }
  Vector<T, 3> xyy() const { return Vector<T, 3>(x, y, y); } Vector<T, 3> yyz() const { return Vector<T, 3>(y, y, z); }
  Vector<T, 3> yzy() const { return Vector<T, 3>(y, z, y); } Vector<T, 3> zyy() const { return Vector<T, 3>(z, y, y); }
  
  Vector<T, 3> zzx() const { return Vector<T, 3>(z, z, x); } Vector<T, 3> zxz() const { return Vector<T, 3>(z, x, z); }
  Vector<T, 3> xzz() const { return Vector<T, 3>(x, z, z); } Vector<T, 3> zzy() const { return Vector<T, 3>(z, z, y); }
  Vector<T, 3> zyz() const { return Vector<T, 3>(z, y, z); } Vector<T, 3> yzz() const { return Vector<T, 3>(y, z, z); }
  
  std::string toString() const            { std::ostringstream ss; ss << (*this); return ss.str(); }
  void fromString(const std::string &str) { std::istringstream ss(str); ss >> (*this); }

  bool operator==(const Vector<T, N> &other) const
  {
    for(int i = 0; i < N; i++)
      { if(data[i] != other.data[i]) return false; }
    return true;
  }
  bool operator!=(const Vector<T, N> &other) const { return !(*this == other); }
  
  // <, > (AND)
  bool operator> (const Vector<T, N> &other) const { for(int i = 0; i < N; i++) { if(data[i] <= other.data[i]) { return false; } } return true; }
  bool operator< (const Vector<T, N> &other) const { for(int i = 0; i < N; i++) { if(data[i] >= other.data[i]) { return false; } } return true; }
  bool operator>=(const Vector<T, N> &other) const { for(int i = 0; i < N; i++) { if(data[i] < other.data[i])  { return false; } } return true; }
  bool operator<=(const Vector<T, N> &other) const { for(int i = 0; i < N; i++) { if(data[i] > other.data[i])  { return false; } } return true; }
  bool operator> (T scalar) const { for(int i = 0; i < N; i++) { if(data[i] <= scalar) { return false; } } return true; }
  bool operator< (T scalar) const { for(int i = 0; i < N; i++) { if(data[i] >= scalar) { return false; } } return true; }
  bool operator>=(T scalar) const { for(int i = 0; i < N; i++) { if(data[i] <  scalar) { return false; } } return true; }
  bool operator<=(T scalar) const { for(int i = 0; i < N; i++) { if(data[i] >  scalar) { return false; } } return true; }

  Vector<T, N>& operator+=(const Vector<T, N> &other) { for(int i = 0; i < N; i++) { data[i] += other.data[i]; } return *this; }
  Vector<T, N>  operator+ (const Vector<T, N> &other) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] += other.data[i]; }
    return result;
  }
  Vector<T, N>& operator-=(const Vector<T, N> &other) { for(int i = 0; i < N; i++) { data[i] -= other.data[i]; } return *this; }
  Vector<T, N>  operator- (const Vector<T, N> &other) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] -= other.data[i]; }
    return result;
  }
  Vector<T, N>& operator*=(const Vector<T, N> &other) { for(int i = 0; i < N; i++) { data[i] *= other.data[i]; } return *this; }
  Vector<T, N>  operator* (const Vector<T, N> &other) const
  {
    Vector<T, N> result(data);
    for(int i = 0; i < N; i++) { result.data[i] *= other.data[i]; }
    return result;
  }
  Vector<T, N>& operator/=(const Vector<T, N> &other) { for(int i = 0; i < N; i++) { data[i] /= other.data[i]; } return *this; }
  Vector<T, N>  operator/ (const Vector<T, N> &other) const
  {
    Vector<T, N> result(data);
    for(int i = 0; i < N; i++) { result.data[i] /= other.data[i]; }
    return result;
  }

  Vector<T, N>& operator*=(T scalar) { for(int i = 0; i < N; i++) { data[i] *= scalar; } return *this; }
  Vector<T, N>  operator* (T scalar) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] *= scalar; }
    return result;
  }
  Vector<T, N>& operator/=(T scalar) { for(int i = 0; i < N; i++) { data[i] /= scalar; } return *this; }
  Vector<T, N>  operator/ (T scalar) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] /= scalar; }
    return result;
  }
  
#ifndef __NVCC__
  Vector<T, N>& operator%=(const T &s)
  {
    for(int i = 0; i < N; i++) { if constexpr(std::is_same<T, int>::value) { data[i] %= s; } else { data[i] = fmod(data[i], s); } }
    return *this;
  }
  Vector<T, N>& operator%=(const Vector<T, N> &other)
  {
    for(int i = 0; i < N; i++) { if constexpr(std::is_same<T, int>::value) { data[i] %= other[i]; } else { data[i] = fmod(data[i], other[i]); } }
    return *this;
  }
  Vector<T, N> operator%(const T &s) const                { Vector<T, N> result(*this); return (result %= s); }
  Vector<T, N> operator%(const Vector<T, N> &other) const { Vector<T, N> result(*this); return (result %= other); }
#endif  
  
  void ceil()                   { for(auto &d : data) { d = std::ceil(d);  } }
  void floor()                  { for(auto &d : data) { d = std::floor(d); } }
  Vector<T, N> getCeil() const  { Vector<T, N> v(*this); v.ceil(); return v; }
  Vector<T, N> getFloor() const { Vector<T, N> v(*this); v.floor(); return v; }

  T length2() const { T sqsum = T(); for(auto d : data) { sqsum += d*d; } return sqsum; }
  T length()  const { return sqrt(length2()); }
  
  void normalize()                { (*this) /= length(); }
  Vector<T, N> normalized() const { return Vector<T, N>(*this) / length(); }

  template<typename U>
  T dot(const Vector<U, N> &other) const
  {
    T total = 0;
    for(int i = 0; i < N; i++) { total += data[i] * other.data[i]; }
    return total;
  }
};


template<typename T, int N>
inline Vector<T, N> operator-(const Vector<T, N> &v)
{
  Vector<T, N> result;
  for(int i = 0; i < N; i++) { result.data[i] = -v[i]; }
  return result;
}
template<typename T, typename U, int N>
inline Vector<T, N> operator*(U scalar, const Vector<T, N> &v)
{
  Vector<T, N> result;
  for(int i = 0; i < N; i++) { result.data[i] = scalar * v[i]; }
  return result;
}
template<typename T, typename U, int N>
inline Vector<T, N> operator/(U scalar, const Vector<T, N> &v)
{
  Vector<T, N> result;
  for(int i = 0; i < N; i++) { result.data[i] = scalar / v[i]; }
  return result;
}


// extensions of Dim class from vector-operators.h
template<typename T> struct Dim;
template<> struct Dim<Vector<int,    1>> { static constexpr int N = 1; using BASE_T = int;    using LOWER = int;               using SIZE_T = int;            };
template<> struct Dim<Vector<int,    2>> { static constexpr int N = 2; using BASE_T = int;    using LOWER = Vector<int,    2>; using SIZE_T = Vector<int, 2>; };
template<> struct Dim<Vector<int,    3>> { static constexpr int N = 3; using BASE_T = int;    using LOWER = Vector<int,    3>; using SIZE_T = Vector<int, 3>; };
template<> struct Dim<Vector<int,    4>> { static constexpr int N = 4; using BASE_T = int;    using LOWER = Vector<int,    4>; using SIZE_T = Vector<int, 4>; };
template<> struct Dim<Vector<float,  1>> { static constexpr int N = 1; using BASE_T = float;  using LOWER = float;             using SIZE_T = int;            };
template<> struct Dim<Vector<float,  2>> { static constexpr int N = 2; using BASE_T = float;  using LOWER = Vector<float,  2>; using SIZE_T = Vector<int, 2>; };
template<> struct Dim<Vector<float,  3>> { static constexpr int N = 3; using BASE_T = float;  using LOWER = Vector<float,  3>; using SIZE_T = Vector<int, 3>; };
template<> struct Dim<Vector<float,  4>> { static constexpr int N = 4; using BASE_T = float;  using LOWER = Vector<float,  4>; using SIZE_T = Vector<int, 4>; };
template<> struct Dim<Vector<double, 1>> { static constexpr int N = 1; using BASE_T = double; using LOWER = double;            using SIZE_T = int;            };
template<> struct Dim<Vector<double, 2>> { static constexpr int N = 2; using BASE_T = double; using LOWER = Vector<double, 2>; using SIZE_T = Vector<int, 2>; };
template<> struct Dim<Vector<double, 3>> { static constexpr int N = 3; using BASE_T = double; using LOWER = Vector<double, 3>; using SIZE_T = Vector<int, 3>; };
template<> struct Dim<Vector<double, 4>> { static constexpr int N = 4; using BASE_T = double; using LOWER = Vector<double, 4>; using SIZE_T = Vector<int, 4>; };

// is_vec<T>::value (or) is_vec_v<T> --> true if T is a vector type (e.g. Vector)
template<typename T> struct is_vec : std::false_type { };
template<typename T, int N> struct is_vec<Vector<T, N>> : std::true_type { };
template<typename T> constexpr bool is_vec_v = is_vec<T>::value; // (helper)

// converts a vector to a string with specified precision
template<typename T> std::enable_if_t<is_vec_v<T>, std::string> to_string(const T &val, const int precision=6)
{
  std::ostringstream out;
  out.precision(precision);
  out << std::fixed << val;
  return out.str();
}
// converts given string to a vector (specify explicitly with template parameter)
template<typename T> std::enable_if_t<is_vec_v<T>, T> from_string(const std::string &valStr)
{ std::istringstream out(valStr); T val; out >> val; return val; }


// SCALAR <, > VECTOR (AND)
template<typename T, typename U, int N>
inline bool operator> (U scalar, const Vector<T, N> &v) { for(int i = 0; i < N; i++) { if(scalar <= v.data[i]) { return false; } } return true; }
template<typename T, typename U, int N>
inline bool operator< (U scalar, const Vector<T, N> &v) { for(int i = 0; i < N; i++) { if(scalar >= v.data[i]) { return false; } } return true; }
template<typename T, typename U, int N>
inline bool operator>=(U scalar, const Vector<T, N> &v) { for(int i = 0; i < N; i++) { if(scalar <  v.data[i]) { return false; } } return true; }
template<typename T, typename U, int N>
inline bool operator<=(U scalar, const Vector<T, N> &v) { for(int i = 0; i < N; i++) { if(scalar >  v.data[i]) { return false; } } return true; }

// glsl/cuda-like functions
template<typename T, int N> inline Vector<T, N> normalize(const Vector<T, N> &v) { return v.normalized(); }
template<typename T, int N> inline T length2(const Vector<T, N> &v) { return v.length2(); }
template<typename T, int N> inline T length (const Vector<T, N> &v) { return v.length(); }
template<typename T, int N> inline T dot(const Vector<T, N> &v1, const Vector<T, N> &v2) { return v1.dot(v2); }

template<typename T, int N> inline T isnan(const Vector<T, N> &v) { for(const auto &d : v.data) { if(std::isnan(d)) return true; } return false; }
template<typename T, int N> inline T isinf(const Vector<T, N> &v) { for(const auto &d : v.data) { if(std::isinf(d)) return true; } return false; }

// abs
template<typename T, int N> inline Vector<T, N> abs(const Vector<T, N> &v)
{ Vector<T, N> av; for(int i = 0; i < N; i++) { av.data[i] = std::abs(v.data[i]); } return av; }
// min element
template<typename T, int N> inline T min(const Vector<T, N> &v)
{ T minVal = std::numeric_limits<T>::max(); for(auto &d : v.data) { minVal = std::min(minVal, d); } return minVal; }
// max element
template<typename T, int N> inline T max(const Vector<T, N> &v)
{ T maxVal = std::numeric_limits<T>::min(); for(auto &d : v.data) { maxVal = std::max(maxVal, d); } return maxVal; }
// min per-element
template<typename T, int N> inline Vector<T, N> min(const Vector<T, N> &v1, const Vector<T, N> &v2)
{ Vector<T, N> minV; for(int i = 0; i < N; i++) { minV[i] = std::min(v1[i], v2[i]); } return minV; }
// max per-element
template<typename T, int N> inline Vector<T, N> max(const Vector<T, N> &v1, const Vector<T, N> &v2)
{ Vector<T, N> maxV; for(int i = 0; i < N; i++) { maxV[i] = std::max(v1[i], v2[i]); } return maxV; }
// mod
template<typename T, int N> inline Vector<T, N> mod(const Vector<T, N> &v,  const T &s)             { return v  % s; }
template<typename T, int N> inline Vector<T, N> mod(const Vector<T, N> &v1, const Vector<T, N> &v2) { return v1 % v2; }
// floor
template<typename T, int N> inline Vector<T, N> floor(const Vector<T, N> &v)
{ Vector<T, N> result; for(int i = 0; i < N; i++) { result[i] = std::floor(v[i]); } return result; }
// ceil
template<typename T, int N> inline Vector<T, N> ceil(const Vector<T, N> &v)
{ Vector<T, N> result; for(int i = 0; i < N; i++) { result[i] = std::ceil(v[i]); } return result; }

// complex / quaternion
template<typename T>
inline Vector<T, 2> cMult(const Vector<T, 2> &a, const Vector<T, 2> &b) { return Vector<T, 2>(a.x*a.x - a.y*a.y, a.x*b.y + a.y*b.x); }
template<typename T>
inline Vector<T, 2> cConj(const Vector<T, 2> &a) { return Vector<T, 2>(a.x, -a.y); }
template<typename T>
inline Vector<T, 4> qMult(const Vector<T, 4> &a, const Vector<T, 4> &b)
{
  return Vector<T, 4>(a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w, a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
                      a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y, a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x);
}
template<typename T>
inline Vector<T, 4> qConj(const Vector<T, 4> &a) { return Vector<T, 4>(a.x, -a.y, -a.z, -a.w); }

template<typename T>
inline Vector<T, 3> cross(const Vector<T, 3> &a, const Vector<T, 3> &b)
{ return Vector<T, 3>(a.y, a.z, a.x) * Vector<T, 3>(b.z, b.x, b.y) - Vector<T, 3>(a.z, a.x, a.y) * Vector<T, 3>(b.y, b.z, b.x); }

template<typename T>
inline Vector<T, 3> rotate(const Vector<T, 3> &v, const Vector<T, 3> &ax, T theta)
{ // rotate via quaternions
  T cos_t2 = (T)cos(theta/2.0);
  T sin_t2 = (T)sin(theta/2.0);
  Vector<T, 4> q1(0, v.x, v.y, v.z);
  Vector<T, 4> q2(cos_t2, ax.x*sin_t2, ax.y*sin_t2, ax.z*sin_t2);
  Vector<T, 4> q3 = qMult(qMult(q2, q1), qConj(q2));
  return Vector<T, 3>(q3.y, q3.z, q3.w);
}


inline int2    to_cuda(const Vec2i &v) { return int2   {v.x, v.y}; }
inline int3    to_cuda(const Vec3i &v) { return int3   {v.x, v.y, v.z}; }
inline int4    to_cuda(const Vec4i &v) { return int4   {v.x, v.y, v.z, v.w}; }
inline float2  to_cuda(const Vec2f &v) { return float2 {v.x, v.y}; }
inline float3  to_cuda(const Vec3f &v) { return float3 {v.x, v.y, v.z}; }
inline float4  to_cuda(const Vec4f &v) { return float4 {v.x, v.y, v.z, v.w}; }
inline double2 to_cuda(const Vec2d &v) { return double2{v.x, v.y}; }
inline double3 to_cuda(const Vec3d &v) { return double3{v.x, v.y, v.z}; }
inline double4 to_cuda(const Vec4d &v) { return double4{v.x, v.y, v.z, v.w}; }

// same syntax as CUDA structs, for template flexibility
template<typename T, int N>  T*       arr(      Vector<T, N> &v) { return v.data.data(); }
template<typename T, int N>  const T* arr(const Vector<T, N> &v) { return v.data.data(); }


#endif //VECTOR_HPP
