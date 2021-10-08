#include "glShader.hpp"

#include <sstream>
#include <fstream>
#include <filesystem>


std::string GlShader::mShaderDir = "./shaders";

GlShader::GlShader(const std::string &vPath, const std::string &fPath)
{
  if(!loadVertex(vPath) || !loadFragment(fPath) || !link())
    { std::cout << "====> ERROR: Failed to load shader! (VF)\n"; }
}

bool GlShader::setShaderDir(const std::string &shaderDir)
{
  if(std::filesystem::exists(shaderDir) && std::filesystem::is_directory(shaderDir))
    { mShaderDir = shaderDir; return true; }
  else { return false; }
}

void GlShader::destroy()
{
  glUseProgram(0);
  if(mProgramId > 0) { glDeleteProgram(mProgramId); mProgramId = 0; }
}

void GlShader::bind()    { glUseProgram(mProgramId); }
void GlShader::release() { glUseProgram(0); }

bool GlShader::loadVertex(const std::string &path)
{
  std::cout << "== Opening VERTEX shader -->  '" << path << "'\n";
  std::ifstream file(path, std::ios::in);
  std::stringstream src; src << file.rdbuf();
  std::string shaderStr = src.str();
  const char* shaderCode = shaderStr.c_str();
  
  int success; char infoLog[512];
  mVShaderId = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(mVShaderId, 1, &shaderCode, NULL);
  glCompileShader(mVShaderId);
  glGetShaderiv(mVShaderId, GL_COMPILE_STATUS, &success);
  if(!success)
    {
      glGetShaderInfoLog(mVShaderId, 512, NULL, infoLog);
      mVertLog = infoLog;
      std::cout << "== - Vertex Shader Log:\n"   << "---  | " << mVertLog << "\n";
      mVShaderId = 0;
      return false;
    }
  return true;
}

bool GlShader::loadFragment(const std::string &path)
{
  std::cout << "== Opening FRAGMENT shader -->  '" << path << "'\n";
  std::ifstream file(path, std::ios::in);
  std::stringstream src; src << file.rdbuf();
  std::string shaderStr = src.str();
  const char* shaderCode = shaderStr.c_str();
  
  int success; char infoLog[512];
  mFShaderId = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(mFShaderId, 1, &shaderCode, NULL);
  glCompileShader(mFShaderId);
  glGetShaderiv(mFShaderId, GL_COMPILE_STATUS, &success);
  if(!success)
    {
      glGetShaderInfoLog(mFShaderId, 512, NULL, infoLog);
      mFragLog = infoLog;
      std::cout << "== - Fragment Shader Log:\n"   << "---  | " << mFragLog << "\n";
      mFShaderId = 0;
      return false;
    }
  return true;
}

bool GlShader::link()
{
  mProgramId = glCreateProgram();
  if(mVShaderId > 0) { glAttachShader(mProgramId, mVShaderId); }
  if(mFShaderId > 0) { glAttachShader(mProgramId, mFShaderId); }
  glLinkProgram(mProgramId);
  
  int success; char infoLog[512];
  glGetProgramiv(mProgramId, GL_LINK_STATUS, &success);
  if(!success)
    {
      glGetProgramInfoLog(mProgramId, 512, NULL, infoLog);
      mLinkLog = infoLog;
      std::cout << "== - Link Log:\n" << "---  | " << mLinkLog << "\n";
      glDeleteProgram(mProgramId);
      mProgramId = 0;
      return false;
    }
  // print attributes
  GLint count = -1;
  glGetProgramiv(mProgramId, GL_ACTIVE_ATTRIBUTES, &count);
  std::cout << "== -- Shader Attributes (" << count << "):\n";
  mAttributes.clear();
  for(int i = 0; i < count; i++)
    {
      const GLsizei bufSize = 32; GLchar name[bufSize]; GLint size; GLenum type; GLsizei length;
      glGetActiveAttrib(mProgramId, (GLuint)i, bufSize, &length, &size, &type, name);
      mAttributes.emplace(std::string(name), i);
      std::cout << "---  | " << name << " (length: " << length << ", size: " << size << ", type: " << type << "\n";
    }
  // print uniforms
  glGetProgramiv(mProgramId, GL_ACTIVE_UNIFORMS, &count);
  std::cout << "== -- Shader Uniforms (" << count << "):\n";
  mUniforms.clear();
  for(int i = 0; i < count; i++)
    {
      const GLsizei bufSize = 32; GLchar name[bufSize]; GLint size; GLenum type; GLsizei length;
      glGetActiveUniform(mProgramId, (GLuint)i, bufSize, &length, &size, &type, name);
      mUniforms.emplace(std::string(name), i);
      std::cout << "== ---  | " << std::setw(bufSize) << std::left << name << "\n";
    }
  return true;
}

int GlShader::getUniformId(const std::string &name)
{
  const auto &iter = mUniforms.find(name);
  return ((iter != mUniforms.end()) ? iter->second : -1);
}

void GlShader::enableAttributeArray(int location)
{ glEnableVertexAttribArray(location); }
void GlShader::getAttribute(int location, int bufSize, GLsizei *length, GLint *tupleSize, GLenum *type, GLchar name[])
{ glGetActiveAttrib(mProgramId, (GLuint)location, bufSize, length, tupleSize, type, name); }
void GlShader::setAttributeBuffer(int location, int tupleSize, GLenum type, int stride, int offset)
{ glVertexAttribPointer(location,          tupleSize, type, GL_FALSE, stride, reinterpret_cast<const void*>(offset)); }
void GlShader::setAttributeBuffer(const std::string &name, int tupleSize, GLenum type, int stride, int offset)
{ glVertexAttribPointer(mAttributes[name], tupleSize, type, GL_FALSE, stride, reinterpret_cast<const void*>(offset)); }
