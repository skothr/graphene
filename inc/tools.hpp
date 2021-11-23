#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <sys/types.h>
#include <sys/stat.h>

// General tools/helpers

#ifndef __NVCC__
template<typename T>
std::enable_if_t<std::is_arithmetic_v<T>, std::string> to_string(const T &e, int precision=6)
{ std::stringstream ss; ss.precision(precision); ss << e; return ss.str(); }
template<typename T>
std::enable_if_t<std::is_arithmetic_v<T>, T> from_string(const std::string &str)
{ std::stringstream ss(str); T e; ss >> e; return e; }
#endif // __NVCC__

// linear interpolation
template<typename T, typename A=float> T lerp(T x0, T x1, A alpha) { return x1*alpha + x0*(1.0f-alpha); }
// bilinear interpolation
template<typename T, typename A=float> T blerp(const T &p00, const T &p01, const T &p10, const T &p11, const A &alpha2)
{ return lerp(lerp(p00, p01, alpha2.x), lerp(p10, p11, alpha2.x), alpha2.y); }

// simple file management
inline bool directoryExists(const std::string &path)
{
  struct stat info;
  int err = stat(path.c_str(), &info);
  return ((err == 0) && (info.st_mode & S_IFDIR));
}

inline bool fileExists(const std::string &path)
{
  struct stat info;
  int err = stat(path.c_str(), &info);
  return ((err == 0) && !(info.st_mode & S_IFDIR));
}

inline std::string getFileExtension(const std::string &path)
{
  std::string::size_type idx = path.rfind('.');
  return ((idx != std::string::npos) ? path.substr(idx) : "");
}

inline std::string getBasePath(const std::string &path)
{
  std::string::size_type idx = path.rfind('/');
  return ((idx != std::string::npos) ? path.substr(idx+1) : path);
}

inline std::string getBaseName(const std::string &path)
{
  std::string base = getBasePath(path);
  std::string::size_type idx = base.rfind('.');
  return ((idx != std::string::npos) ? base.substr(0, idx) : base);
}

inline bool makeDirectory(const std::string &path)
{
#if defined(_WIN32) || defined(__MINGW32__)
  int err = mkdir(path.c_str());
#elif defined(__linux__)
  mode_t nMode = 0733; // UNIX style permissions
  int err = mkdir(path.c_str(), nMode);
#endif
  return (err == 0);
}

// defines (some) bitwise operators for an enum that works as a flag type
#define ENUM_FLAG_OPERATORS(TYPE)                                       \
  inline constexpr TYPE  operator~ (TYPE t)            { return static_cast<TYPE>(~static_cast<int>(t)); } \
  inline           TYPE& operator|=(TYPE &t0, TYPE t1) { t0   = static_cast<TYPE>( static_cast<int>(t0) | static_cast<int>(t1)); return t0; } \
  inline           TYPE& operator&=(TYPE &t0, TYPE t1) { t0   = static_cast<TYPE>( static_cast<int>(t0) & static_cast<int>(t1)); return t0; } \
  inline constexpr TYPE  operator| (TYPE  t0, TYPE t1) { return static_cast<TYPE>( static_cast<int>(t0) | static_cast<int>(t1)); } \
  inline constexpr TYPE  operator& (TYPE  t0, TYPE t1) { return static_cast<TYPE>( static_cast<int>(t0) & static_cast<int>(t1)); }
#define ENUM_FLAG_OPERATORS_LL(TYPE)                                    \
  inline constexpr TYPE  operator~ (TYPE t)            { return static_cast<TYPE>(~static_cast<long long>(t)); } \
  inline           TYPE& operator|=(TYPE &t0, TYPE t1) { t0   = static_cast<TYPE>( static_cast<long long>(t0) | static_cast<long long>(t1)); return t0; } \
  inline           TYPE& operator&=(TYPE &t0, TYPE t1) { t0   = static_cast<TYPE>( static_cast<long long>(t0) & static_cast<long long>(t1)); return t0; } \
  inline constexpr TYPE  operator| (TYPE  t0, TYPE t1) { return static_cast<TYPE>( static_cast<long long>(t0) | static_cast<long long>(t1)); } \
  inline constexpr TYPE  operator& (TYPE  t0, TYPE t1) { return static_cast<TYPE>( static_cast<long long>(t0) & static_cast<long long>(t1)); }


#endif // TOOLS_HPP
