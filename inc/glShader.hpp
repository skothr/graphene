#ifndef GL_SHADER_HPP
#define GL_SHADER_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "glMatrix.hpp"

class GlShader
{
private:
  static std::string mShaderDir;
  
  GLuint mProgramId = 0;
  GLuint mVShaderId = 0;
  GLuint mFShaderId = 0;
  std::string mVertLog = ""; // vertex shader compile log
  std::string mFragLog = ""; // fragment shader compile log
  std::string mLinkLog = ""; // shader program link log
  std::unordered_map<std::string, int> mAttributes;
  std::unordered_map<std::string, int> mUniforms;

public:
  static bool setShaderDir(const std::string &shaderDir);

  GlShader() { }
  GlShader(const std::string &vPath, const std::string &fPath);
  virtual ~GlShader() { destroy(); }

  bool loadVertex(const std::string &path);
  bool loadFragment(const std::string &path);
  bool link();
  void destroy();

  unsigned int id() const { return mProgramId; }

  int getUniformId(const std::string &name);
  void bind();
  void release();

  const std::unordered_map<std::string, int>& attributes() const { return mAttributes; }
  void enableAttributeArray(int location);
  void setAttributeBuffer(int location, int tupleSize, GLenum type, int offset, int stride);
  void setAttributeBuffer(const std::string &name, int tupleSize, GLenum type, int offset, int stride);
  void getAttribute(int location, int bufSize, GLsizei *length, GLint *tupleSize, GLenum *type, GLchar name[]);

  template<typename T> void setUniform(const std::string &name, const T &value);

  const std::string& vLog() const    { return mVertLog; }
  const std::string& fLog() const    { return mFragLog; }
  const std::string& linkLog() const { return mLinkLog; }
};



//// set by value ////
// values
template<> inline void GlShader::setUniform<int        >(const std::string &name, const int         &val) { glUniform1i(mUniforms[name], val); }
template<> inline void GlShader::setUniform<uint8_t    >(const std::string &name, const uint8_t     &val) { glUniform1i(mUniforms[name], val); }
template<> inline void GlShader::setUniform<uint16_t   >(const std::string &name, const uint16_t    &val) { glUniform1i(mUniforms[name], val); }
template<> inline void GlShader::setUniform<uint32_t   >(const std::string &name, const uint32_t    &val) { glUniform1i(mUniforms[name], val); }
template<> inline void GlShader::setUniform<uint64_t   >(const std::string &name, const uint64_t    &val) { glUniform1i(mUniforms[name], val); }
template<> inline void GlShader::setUniform<float      >(const std::string &name, const float       &val) { glUniform1f(mUniforms[name], val); }
template<> inline void GlShader::setUniform<double     >(const std::string &name, const double      &val) { glUniform1f(mUniforms[name], val); }
template<> inline void GlShader::setUniform<long double>(const std::string &name, const long double &val) { glUniform1f(mUniforms[name], val); }
// vectors
template<> inline void GlShader::setUniform<Vec2i>(const std::string &name, const Vec2i &v) { glUniform2i(mUniforms[name], v[0], v[1]); }
template<> inline void GlShader::setUniform<Vec3i>(const std::string &name, const Vec3i &v) { glUniform3i(mUniforms[name], v[0], v[1], v[2]); }
template<> inline void GlShader::setUniform<Vec4i>(const std::string &name, const Vec4i &v) { glUniform4i(mUniforms[name], v[0], v[1], v[2], v[3]); }
template<> inline void GlShader::setUniform<Vec2f>(const std::string &name, const Vec2f &v) { glUniform2f(mUniforms[name], v[0], v[1]); }
template<> inline void GlShader::setUniform<Vec3f>(const std::string &name, const Vec3f &v) { glUniform3f(mUniforms[name], v[0], v[1], v[2]); }
template<> inline void GlShader::setUniform<Vec4f>(const std::string &name, const Vec4f &v) { glUniform4f(mUniforms[name], v[0], v[1], v[2], v[3]); }
template<> inline void GlShader::setUniform<Vec2d>(const std::string &name, const Vec2d &v) { glUniform2f(mUniforms[name], v[0], v[1]); }
template<> inline void GlShader::setUniform<Vec3d>(const std::string &name, const Vec3d &v) { glUniform3f(mUniforms[name], v[0], v[1], v[2]); }
template<> inline void GlShader::setUniform<Vec4d>(const std::string &name, const Vec4d &v) { glUniform4f(mUniforms[name], v[0], v[1], v[2], v[3]); }
template<> inline void GlShader::setUniform<Vec2l>(const std::string &name, const Vec2l &v) { glUniform2f(mUniforms[name], v[0], v[1]); }
template<> inline void GlShader::setUniform<Vec3l>(const std::string &name, const Vec3l &v) { glUniform3f(mUniforms[name], v[0], v[1], v[2]); }
template<> inline void GlShader::setUniform<Vec4l>(const std::string &name, const Vec4l &v) { glUniform4f(mUniforms[name], v[0], v[1], v[2], v[3]); }
// matrices
template<> inline void GlShader::setUniform<Mat4f>(const std::string &name, const Mat4f &mat)
{ glUniformMatrix4fv(mUniforms[name], 1, GL_TRUE, (const float*)mat.data()); }
template<> inline void GlShader::setUniform<Mat3f>(const std::string &name, const Mat3f &mat)
{ glUniformMatrix3fv(mUniforms[name], 1, GL_TRUE, (const float*)mat.data()); }
template<> inline void GlShader::setUniform<Mat2f>(const std::string &name, const Mat2f &mat)
{ glUniformMatrix2fv(mUniforms[name], 1, GL_TRUE, (const float*)mat.data()); }


#endif // GL_SHADER_HPP
