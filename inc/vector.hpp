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

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <vector_types.h>
#endif // ENABLE_CUDA

// Vector Template Base
template<typename T, int N>
struct Vector
{
  std::array<T, N> data;

  __host__ __device__ Vector()                              : data{{0}}          { }
  __host__ __device__ Vector(const Vector<T, N> &other)     : Vector(other.data) { }
  __host__ __device__ Vector(const std::array<T, N> &data_) : data(data_)        { }
  __host__ __device__ Vector(T val)                         : data(N, val)       { }
  
  __host__ __device__ Vector(const std::string &str)          { fromString(str); }
  template<typename U> // convert from other type
  __host__ __device__ Vector(const Vector<U, N> &other)       { for(int i = 0; i < N; i++) { data[i] = (T)other.data[i]; } }

#ifdef ENABLE_CUDA
  __host__ __device__  Vector(const float2 &cv)  { for(int i = 0; i < std::min(N, 2); i++) { data[i] = (T)( (const float*)(&cv))[i]; } }
  __host__ __device__  Vector(const float3 &cv)  { for(int i = 0; i < std::min(N, 3); i++) { data[i] = (T)( (const float*)(&cv))[i]; } }
  __host__ __device__  Vector(const float4 &cv)  { for(int i = 0; i < std::min(N, 4); i++) { data[i] = (T)( (const float*)(&cv))[i]; } }
  __host__ __device__  Vector(const double2 &cv) { for(int i = 0; i < std::min(N, 2); i++) { data[i] = (T)((const double*)(&cv))[i]; } }
  __host__ __device__  Vector(const double3 &cv) { for(int i = 0; i < std::min(N, 3); i++) { data[i] = (T)((const double*)(&cv))[i]; } }
  __host__ __device__  Vector(const double4 &cv) { for(int i = 0; i < std::min(N, 4); i++) { data[i] = (T)((const double*)(&cv))[i]; } }
  __host__ __device__  Vector<T, N>& operator=(const float2 &cv) { for(int i = 0; i < std::min(N, 2); i++) { data[i] = (T)((const float*)(&cv))[i]; } return *this; }
  __host__ __device__  Vector<T, N>& operator=(const float3 &cv) { for(int i = 0; i < std::min(N, 3); i++) { data[i] = (T)((const float*)(&cv))[i]; } return *this; }
  __host__ __device__  Vector<T, N>& operator=(const float4 &cv) { for(int i = 0; i < std::min(N, 4); i++) { data[i] = (T)((const float*)(&cv))[i]; } return *this; }
  __host__ __device__  Vector<T, N>& operator=(const double2 &cv){ for(int i = 0; i < std::min(N, 2); i++) { data[i] = (T)((const double*)(&cv))[i]; } return *this; }
  __host__ __device__  Vector<T, N>& operator=(const double3 &cv){ for(int i = 0; i < std::min(N, 3); i++) { data[i] = (T)((const double*)(&cv))[i]; } return *this; }
  __host__ __device__  Vector<T, N>& operator=(const double4 &cv){ for(int i = 0; i < std::min(N, 4); i++) { data[i] = (T)((const double*)(&cv))[i]; } return *this; }
#endif // ENABLE_CUDA

  std::string toString() const            { std::ostringstream ss;      ss << (*this); return ss.str(); }
  void fromString(const std::string &str) { std::istringstream ss(str); ss >> (*this); }

  T& operator[](int dim)             { return data[dim]; }
  const T& operator[](int dim) const { return data[dim]; }
  
  __host__ __device__ Vector<T, N>& operator=(T scalar)
  {
    for(int i = 0; i < N; i++) { data[i] = scalar; }
    return *this;
  }
  __host__ __device__ Vector<T, N>& operator=(const Vector<T, N> &other)
  {
    for(int i = 0; i < N; i++) { data[i] = other.data[i]; }
    return *this;
  }
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

  void ceil()  { for(auto &d : data) { d = std::ceil(d);  } }
  void floor() { for(auto &d : data) { d = std::floor(d); } }
  Vector<T, N> getCeil() const  { Vector<T, N> v(*this); v.ceil(); return v; }
  Vector<T, N> getFloor() const { Vector<T, N> v(*this); v.floor(); return v; }

  T length2() const
  {
    T sqsum = T();
    for(auto d : data) { sqsum += d*d; }
    return sqsum;
  }
  T length() const                { return sqrt(length2()); }
  Vector<T, N> normalized() const { return Vector<T, N>(*this) / length(); }
  template<typename U>
  T dot(const Vector<U, N> &other)
  {
    T total = 0;
    for(int i = 0; i < N; i++) { total += data[i] * other.data[i]; }
    return total;
  }
};

//Shorthand typedefs
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
  union
  {
    struct { T x, y; };
    std::array<T, N> data;
  };

  __host__ __device__ Vector()                              : x((T)0), y((T)0)          { }
  __host__ __device__ Vector(T x_, T y_)                    : x(x_), y(y_)              { }
  __host__ __device__ Vector(const Vector<T, N> &other)     : x(other.x), y(other.y)    { }
  __host__ __device__ Vector(const std::array<T, N> &data_) : x(data_[0]), y(data_[1])  { }
  __host__ __device__ Vector(T val)                         : x(val), y(val)            { }
  __host__ __device__ Vector(const std::string &str)        { fromString(str); }
  template<typename U> // convert from other type
  __host__ __device__ Vector(const Vector<U, N> &other)     { for(int i = 0; i < N; i++) { data[i] = (T)other.data[i]; } }
  
  __host__ __device__ Vector<T, 2>& operator=(const Vector<T, 2> &other) { data = other.data; return *this; }
  __host__ __device__ Vector<T, N>& operator=(T scalar)                  { for(int i = 0; i < N; i++) { data[i] = scalar; } return *this; }

#ifdef ENABLE_CUDA
  __host__ __device__  Vector(const int2    &cv) : x((T)cv.x), y((T)cv.y) { }
  __host__ __device__  Vector(const float2  &cv) : x((T)cv.x), y((T)cv.y) { }
  __host__ __device__  Vector(const double2 &cv) : x((T)cv.x), y((T)cv.y) { }
  __host__ __device__  Vector<T, N>& operator=(const float2 &cv)  { x = (T)cv.x; y = (T)cv.y; return *this; }
  __host__ __device__  Vector<T, N>& operator=(const double2 &cv) { x = (T)cv.x; y = (T)cv.y; return *this; }
#endif // ENABLE_CUDA

  T& operator[](int dim)             { return data[dim]; }
  const T& operator[](int dim) const { return data[dim]; }
  
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

  Vector<T, N>& operator+=(const Vector<T, N> &other)
  {
    for(int i = 0; i < N; i++) { data[i] += other.data[i]; }
    return *this;
  }
  Vector<T, N>& operator-=(const Vector<T, N> &other)
  {
    for(int i = 0; i < N; i++) { data[i] -= other.data[i]; }
    return *this;
  }
  Vector<T, N> operator+(const Vector<T, N> &other) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] += other.data[i]; }
    return result;
  }
  Vector<T, N> operator-(const Vector<T, N> &other) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] -= other.data[i]; }
    return result;
  }
  Vector<T, N>& operator*=(T scalar)
  { for(int i = 0; i < N; i++) { data[i] *= scalar; } return *this; }
  Vector<T, N>& operator/=(T scalar)
  { for(int i = 0; i < N; i++) { data[i] /= scalar; } return *this; }
  Vector<T, N> operator*(T scalar) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] *= scalar; }
    return result;
  }
  Vector<T, N> operator/(T scalar) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] /= scalar; }
    return result;
  }

  Vector<T, N>& operator*=(const Vector<T, N> &other)
  { for(int i = 0; i < N; i++) { data[i] *= other.data[i]; } return *this; }
  Vector<T, N>& operator/=(const Vector<T, N> &other)
  { for(int i = 0; i < N; i++) { data[i] /= other.data[i]; } return *this; }
  Vector<T, N> operator*(const Vector<T, N> &other) const
  {
    Vector<T, N> result(data);
    for(int i = 0; i < N; i++) { result.data[i] *= other.data[i]; }
    return result;
  }
  Vector<T, N> operator/(const Vector<T, N> &other) const
  {
    Vector<T, N> result(data);
    for(int i = 0; i < N; i++) { result.data[i] /= other.data[i]; }
    return result;
  }
  
  void ceil()  { for(auto &d : data) { d = std::ceil(d);  } }
  void floor() { for(auto &d : data) { d = std::floor(d); } }
  Vector<T, N> getCeil() const  { Vector<T, N> v(*this);  v.ceil();  return v; }
  Vector<T, N> getFloor() const { Vector<T, N> v = *this; v.floor(); return v; }
  
  T length2() const
  {
    T sqsum = T();
    for(auto d : data) { sqsum += d*d; }
    return sqsum;
  }
  T length() const { return sqrt(length2()); }
  void normalize()                { (*this) /= length(); }
  Vector<T, N> normalized() const { return Vector<T, N>(*this) / length(); }
  template<typename U>
  T dot(const Vector<U, N> &other)
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
  union
  {
    struct { T x, y, z; };
    std::array<T, N> data;
  };

  __host__ __device__ Vector()                              : x((T)0), y((T)0), z((T)0)               { }
  __host__ __device__ Vector(T x_, T y_, T z_)              : x(x_), y(y_), z(z_)                     { }
  __host__ __device__ Vector(const Vector<T, N> &other)     : x(other.x), y(other.y), z(other.z)      { }
  __host__ __device__ Vector(const std::array<T, N> &data_) : x(data_[0]), y(data_[1]), z(data_[2])   { }
  __host__ __device__ Vector(T val)                         : x(val), y(val), z(val)                  { }
  
  __host__ __device__ Vector(const std::string &str)        { fromString(str); }
  template<typename U> // convert from other type
  __host__ __device__ Vector(const Vector<U, N> &other)     { for(int i = 0; i < N; i++) { data[i] = (T)other.data[i]; } }
  
  __host__ __device__ Vector<T, 2>& operator=(const Vector<T, 2> &other) { data = other.data; return *this; }
  __host__ __device__ Vector<T, N>& operator=(T scalar)                  { for(int i = 0; i < N; i++) { data[i] = scalar; }    return *this; }
  
#ifdef ENABLE_CUDA
  __host__ __device__  Vector(const int3    &cv) : x((T)cv.x), y((T)cv.y), z((T)cv.z) { }
  __host__ __device__  Vector(const float3  &cv) : x((T)cv.x), y((T)cv.y), z((T)cv.z) { }
  __host__ __device__  Vector(const double3 &cv) : x((T)cv.x), y((T)cv.y), z((T)cv.z) { }
  __host__ __device__  Vector<T, N>& operator=(const float3 &cv)  { x = (T)cv.x; y = (T)cv.y; z = (T)cv.z; return *this; }
  __host__ __device__  Vector<T, N>& operator=(const double3 &cv) { x = (T)cv.x; y = (T)cv.y; z = (T)cv.z; return *this; }
#endif // ENABLE_CUDA
  
  T& operator[](int dim)             { return data[dim]; }
  const T& operator[](int dim) const { return data[dim]; }
  
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

  Vector<T, N>& operator+=(const Vector<T, N> &other)
  { for(int i = 0; i < N; i++) { data[i] += other.data[i]; } return *this; }
  Vector<T, N>& operator-=(const Vector<T, N> &other)
  { for(int i = 0; i < N; i++) { data[i] -= other.data[i]; } return *this; }
  Vector<T, N> operator+(const Vector<T, N> &other) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++)
      { result.data[i] += other.data[i]; }
    return result;
  }
  Vector<T, N> operator-(const Vector<T, N> &other) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] -= other.data[i]; }
    return result;
  }
  
  Vector<T, N>& operator*=(T scalar)
  { for(int i = 0; i < N; i++) { data[i] *= scalar; } return *this; }
  Vector<T, N>& operator/=(T scalar)
  { for(int i = 0; i < N; i++) { data[i] /= scalar; } return *this; }
  Vector<T, N> operator*(T scalar) const
  {
    Vector<T, N> result(data);
    for(int i = 0; i < N; i++) { result.data[i] *= scalar; }
    return result;
  }
  Vector<T, N> operator/(T scalar) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] /= scalar; }
    return result;
  }
  
  Vector<T, N>& operator*=(const Vector<T, N> &other)
  { for(int i = 0; i < N; i++) { data[i] *= other.data[i]; } return *this; }
  Vector<T, N>& operator/=(const Vector<T, N> &other)
  { for(int i = 0; i < N; i++) { data[i] /= other.data[i]; } return *this; }
  Vector<T, N> operator*(const Vector<T, N> &other) const
  {
    Vector<T, N> result(data);
    for(int i = 0; i < N; i++) { result.data[i] *= other.data[i]; }
    return result;
  }
  Vector<T, N> operator/(const Vector<T, N> &other) const
  {
    Vector<T, N> result(data);
    for(int i = 0; i < N; i++) { result.data[i] /= other.data[i]; }
    return result;
  }
  
  void ceil()  { for(auto &d : data) { d = std::ceil(d); } }
  void floor() { for(auto &d : data) { d = std::floor(d); } }
  Vector<T, N> getCeil() const  { Vector<T, N> v(*this); v.ceil();  return v; }
  Vector<T, N> getFloor() const { Vector<T, N> v(*this); v.floor(); return v; }

  T length2() const
  {
    T sqsum = T();
    for(auto d : data) { sqsum += d*d; }
    return sqsum;
  }
  T length() const                { return sqrt(length2()); }
  void normalize()                { (*this) /= length(); }
  Vector<T, N> normalized() const { return Vector<T, N>(*this) / length(); }
  template<typename U>
  T dot(const Vector<U, N> &other)
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
  union
  {
    struct { T x, y, z, w; };
    std::array<T, N> data;
  };

  __host__ __device__ Vector()                              : x((T)0), y((T)0), z((T)0), w((T)1)                 { }
  __host__ __device__ Vector(T x_, T y_, T z_, T w_)        : x(x_), y(y_), z(z_), w(w_)                         { }
  __host__ __device__ Vector(const Vector<T, N> &other)     : x(other.x), y(other.y), z(other.z), w(other.w)     { }
  __host__ __device__ Vector(const std::array<T, N> &data_) : x(data_[0]), y(data_[1]), z(data_[2]), w(data_[3]) { }
  __host__ __device__ Vector(T val)                         : x(val), y(val), z(val), w(val)                     { }
  __host__ __device__ Vector(const std::string &str)        { fromString(str); }
  template<typename U> // convert from other type
  __host__ __device__ Vector(const Vector<U, N> &other)     { for(int i = 0; i < N; i++) { data[i] = (T)other.data[i]; } }
  
  __host__ __device__ Vector<T, 4>& operator=(const Vector<T, 4> &other) { data = other.data; return *this; }
  __host__ __device__ Vector<T, N>& operator=(T scalar)                  { for(int i = 0; i < N; i++) { data[i] = scalar; } return *this; }

#ifdef ENABLE_CUDA
  __host__ __device__ Vector(const int4    &cv) : x((T)cv.x), y((T)cv.y), z((T)cv.z), w((T)cv.w) { }
  __host__ __device__ Vector(const float4  &cv) : x((T)cv.x), y((T)cv.y), z((T)cv.z), w((T)cv.w) { }
  __host__ __device__ Vector(const double4 &cv) : x((T)cv.x), y((T)cv.y), z((T)cv.z), w((T)cv.w) { }
  __host__ __device__  Vector<T, N>& operator=(const float4 &cv)  { x = (T)cv.x; y = (T)cv.y; z = (T)cv.z; w = (T)cv.w; return *this; }
  __host__ __device__  Vector<T, N>& operator=(const double4 &cv) { x = (T)cv.x; y = (T)cv.y; z = (T)cv.z; w = (T)cv.w; return *this; }
#endif // ENABLE_CUDA
  
  T& operator[](int dim)               { return data[dim]; }
  const T& operator[](int dim) const   { return data[dim]; }
  
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

  Vector<T, N>& operator+=(const Vector<T, N> &other)
  { for(int i = 0; i < N; i++) { data[i] += other.data[i]; } return *this; }
  Vector<T, N>& operator-=(const Vector<T, N> &other)
  { for(int i = 0; i < N; i++) { data[i] -= other.data[i]; } return *this; }
  Vector<T, N> operator+(const Vector<T, N> &other) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] += other.data[i]; }
    return result;
  }
  Vector<T, N> operator-(const Vector<T, N> &other) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] -= other.data[i]; }
    return result;
  }

  Vector<T, N>& operator*=(T scalar)
  { for(int i = 0; i < N; i++) { data[i] *= scalar; } return *this; }
  Vector<T, N>& operator/=(T scalar)
  { for(int i = 0; i < N; i++) { data[i] /= scalar; } return *this; }
  
  Vector<T, N> operator*(T scalar) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] *= scalar; }
    return result;
  }
  Vector<T, N> operator/(T scalar) const
  {
    Vector<T, N> result(*this);
    for(int i = 0; i < N; i++) { result.data[i] /= scalar; }
    return result;
  }
  
  Vector<T, N>& operator*=(const Vector<T, N> &other)
  { for(int i = 0; i < N; i++) { data[i] *= other.data[i]; } return *this; }
  Vector<T, N>& operator/=(const Vector<T, N> &other)
  { for(int i = 0; i < N; i++) { data[i] /= other.data[i]; } return *this; }
  Vector<T, N> operator*(const Vector<T, N> &other) const
  {
    Vector<T, N> result(data);
    for(int i = 0; i < N; i++) { result.data[i] *= other.data[i]; }
    return result;
  }
  Vector<T, N> operator/(const Vector<T, N> &other) const
  {
    Vector<T, N> result(data);
    for(int i = 0; i < N; i++) { result.data[i] /= other.data[i]; }
    return result;
  }
  
  void ceil()  { for(auto &d : data) { d = std::ceil(d);  } }
  void floor() { for(auto &d : data) { d = std::floor(d); } }
  Vector<T, N> getCeil() const  { Vector<T, N> v(*this); v.ceil(); return v; }
  Vector<T, N> getFloor() const { Vector<T, N> v(*this); v.floor(); return v; }

  T length2() const
  {
    T sqsum = T();
    for(auto d : data) { sqsum += d*d; }
    return sqsum;
  }
  T length() const { return sqrt(length2()); }
  void normalize()                { (*this) /= length(); }
  Vector<T, N> normalized() const { return Vector<T, N>(*this) / length(); }
  template<typename U>
  T dot(const Vector<U, N> &other)
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
template<typename T, int N>
inline Vector<T, N> operator*(T scalar, const Vector<T, N> &v)
{
  Vector<T, N> result;
  for(int i = 0; i < N; i++) { result.data[i] = scalar * v[i]; }
  return result;
}
template<typename T, int N>
inline Vector<T, N> operator/(T scalar, const Vector<T, N> &v)
{
  Vector<T, N> result;
  for(int i = 0; i < N; i++) { result.data[i] = scalar / v[i]; }
  return result;
}







// SCALAR <, > VECTOR (AND)
template<typename T, int N>
inline bool operator> (T scalar, const Vector<T, N> &v) { for(int i = 0; i < N; i++) { if(scalar <= v.data[i]) { return false; } } return true; }
template<typename T, int N>
inline bool operator< (T scalar, const Vector<T, N> &v) { for(int i = 0; i < N; i++) { if(scalar >= v.data[i]) { return false; } } return true; }
template<typename T, int N>
inline bool operator>=(T scalar, const Vector<T, N> &v) { for(int i = 0; i < N; i++) { if(scalar <  v.data[i]) { return false; } } return true; }
template<typename T, int N>
inline bool operator<=(T scalar, const Vector<T, N> &v) { for(int i = 0; i < N; i++) { if(scalar >  v.data[i]) { return false; } } return true; }









// glsl/cuda-like functions
template<typename T, int N>
inline Vector<T, N> normalize(const Vector<T, N> &v) { return v.normalized(); }
template<typename T, int N>
inline T length2(const Vector<T, N> &v) { return v.length2(); }
template<typename T, int N>
inline T length(const Vector<T, N> &v) { return v.length(); }
template<typename T, int N>
inline T dot(const Vector<T, N> &v1, const Vector<T, N> &v2) { return v1.dot(v2); }

// abs
template<typename T, int N> inline Vector<T, N> abs(const Vector<T, N> &v)
{ Vector<T, N> av; for(int i = 0; i < N; i++) { av.data[i] = std::abs(v.data[i]); } return av; }
// min
template<typename T, int N> inline T min(const Vector<T, N> &v)
{ T minVal = std::numeric_limits<T>::max(); for(auto &d : v.data) { minVal = std::min(minVal, d); } return minVal; }
// max
template<typename T, int N> inline T max(const Vector<T, N> &v)
{ T maxVal = std::numeric_limits<T>::min(); for(auto &d : v.data) { maxVal = std::max(maxVal, d); } return maxVal; }




template<typename T>
inline Vector<T, 2> cMult(const Vector<T, 2> &a, const Vector<T, 2> &b) { return Vector<T, 2>(a.x*a.x - a.y*a.y, a.x*b.y + a.y*b.x); }
template<typename T>
inline Vector<T, 2> cConj(const Vector<T, 2> &a) { return Vector<T, 2>(a.x, -a.y); }

template<typename T>
inline Vector<T, 4> qMult(const Vector<T, 4> &a, const Vector<T, 4> &b)
{
  return Vector<T, 4>(a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
                      a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
                      a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
                      a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x);
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




#if ENABLE_CUDA
__host__ __device__ inline int2    to_cuda(const Vec2i &v) { return int2   {v.x, v.y}; }
__host__ __device__ inline int3    to_cuda(const Vec3i &v) { return int3   {v.x, v.y, v.z}; }
__host__ __device__ inline int4    to_cuda(const Vec4i &v) { return int4   {v.x, v.y, v.z, v.w}; }
__host__ __device__ inline float2  to_cuda(const Vec2f &v) { return float2 {v.x, v.y}; }
__host__ __device__ inline float3  to_cuda(const Vec3f &v) { return float3 {v.x, v.y, v.z}; }
__host__ __device__ inline float4  to_cuda(const Vec4f &v) { return float4 {v.x, v.y, v.z, v.w}; }
__host__ __device__ inline double2 to_cuda(const Vec2d &v) { return double2{v.x, v.y}; }
__host__ __device__ inline double3 to_cuda(const Vec3d &v) { return double3{v.x, v.y, v.z}; }
__host__ __device__ inline double4 to_cuda(const Vec4d &v) { return double4{v.x, v.y, v.z, v.w}; }
#endif // ENABLE_CUDA

template<typename T, int N> __host__ __device__ T*       arr(      Vector<T, N> &v) { return v.data.data(); }
template<typename T, int N> __host__ __device__ const T* arr(const Vector<T, N> &v) { return v.data.data(); }


#endif //VECTOR_HPP
