#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <sys/types.h>
#include <sys/stat.h>

// General tools/helpers

// converts given value to a string with specified precision
// TODO: find a better place for this
template <typename T>
std::string to_string(const T &val, const int precision = 6)
{
  std::ostringstream out;
  out.precision(precision);
  out << std::fixed << val;
  return out.str();
}

// converts given string to a typed value (specify explicitly with template parameter)
template<typename T>
T from_string(const std::string &valStr)
{
  std::istringstream out(valStr);
  T val; out >> val;
  return val;
}

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
#if defined(_WIN32)
  int err = mkdir(path.c_str());
#else 
  mode_t nMode = 0733; // UNIX style permissions
  int err = mkdir(path.c_str(), nMode);
#endif
  return (err == 0);
}


// // provides access to simple info for an std::function object
// template<typename T>  struct FunctionInfo;
// template<typename R, typename ...Args>
// struct FunctionInfo<std::function<R(Args...)>>
// {
//   static const size_t nargs = sizeof...(Args);
//   typedef R returnType;
//   // individual argument types --> print name of type with:
//   //      std::cout << typeid(FunctionInfo<f>::arg<0>::type).name() << "\n";
//   template<size_t i> struct arg
//   { typedef typename std::tuple_element<i, std::tuple<Args...>>::type type; };
// };

// // structure for storing vectors of booleans normally
// struct BoolStruct
// {
//   bool data;
//   BoolStruct(bool val = false) : data(val) { }
//   operator bool() const { return data; }
//   friend std::ostream& operator<<(std::ostream &os, const BoolStruct &b);
//   friend std::istream& operator>>(std::istream &is, BoolStruct &b);
// };
// inline std::ostream& operator<<(std::ostream &os, const BoolStruct &b)
// { os << (b.data ? "1" : "0") << " "; return os; }
// inline std::istream& operator>>(std::istream &is, BoolStruct &b)
// {
//   std::string str;
//   is >> str; b.data = (str != "0");
//   return is;
// }



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
