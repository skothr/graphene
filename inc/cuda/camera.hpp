#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "vector.hpp"
#include "matrix.hpp"
#include "vector-operators.h"
#include "raytrace.h"
#include "camera.cuh"


// full camera description (for use on host)
//  --> "turntable" camera (no roll, constant up direction)
template<typename T>
struct Camera
{
  typedef typename DimType<T,2>::VEC_T VT2;
  typedef typename DimType<T,3>::VEC_T VT3;
  
  union
  { // camera description (union for pass-through access)
    struct { VT3 pos; VT3 dir; VT3 right; VT3 up; T fov; T aspect; T near; T far; };
    CameraDesc<T> desc;
  };
  VT3 upBasis = VT3{(T)0, (T)1, (T)0};
  Matrix<T,4,4> view; // view matrix
  Matrix<T,4,4> proj; // projection matrix
  Matrix<T,4,4> VP;   // proj ^ view

  Camera() { }
  Camera(const Camera<T> &other) = default;
  Camera& operator=(const Camera<T> &other) = default;

  // sp --> screen pos [0.0, 1.0]
  Ray<T> castRay(const VT2 &sp) const;

  Vector<T, 4> worldToView(const Vector<T, 3> &wp) const;
  Vector<T, 2> nearClip   (const Vector<T, 4> &v, const Vector<T, 4> &vo) const;
  Vector<T, 2> vNormalize (const Vector<T, 4> &v) const;

  void rotate(VT2 angles);
  void calculate(); // recalculate matrices
};



template<typename T>
Ray<T> Camera<T>::castRay(const VT2 &sp) const
{ return desc.castRay(VT2{sp.x, (T)1.0-sp.y}); }

template<typename T>
Vector<T, 4> Camera<T>::worldToView(const Vector<T, 3> &wp) const
{ return (proj ^ (view ^ Vector<T, 4>(wp.x-pos.x, wp.y-pos.y, wp.z-pos.z, 1.0))); }

template<typename T> // normalizes v to [0,1]
Vector<T, 2> Camera<T>::vNormalize(const Vector<T, 4> &v) const
{ return (Vector<T, 2>(v.x, -v.y)/v.w + 1) / 2.0; }

template<typename T> // clips v for drawing line between v and vo
Vector<T, 2> Camera<T>::nearClip(const Vector<T, 4> &v, const Vector<T, 4> &vo) const
{
  Vector<T, 4> vr;
  if(v.w < near)
    { // clip line to near plane (linear interpolation between orthogonal distances)
      T n = (vo.w - near)/(vo.w - v.w);
      vr = n*v + (1-n)*vo; vr.w = near;
    }
  else { vr = v; }
  return vNormalize(vr);
}


template<typename T>
void Camera<T>::calculate() // recalculate matrices
{
  // recalculate orthonormal bases
  dir   = normalize(dir);
  right = normalize(cross(dir, upBasis));
  up    = normalize(cross(right, dir));
  // create matrices
  view = Mat4f::makeLookAt(pos, pos+dir);
  proj = Mat4f::makeProjection(fov*M_PI/180.0, aspect, near, far);
  VP = proj ^ view;
}


template<typename T>
void Camera<T>::rotate(VT2 angles) // angles --> { pitch, yaw, roll } (rotation around x, y, and z bases)
{
  if(abs(up.y) <= 0.05 && ((angles.y < 0 ? -1 : 1) != (pos.y < 0 ? -1 : 1))) { angles.y = 0.0; }
    
  VT3 newDir   = normalize(dir);
  if(angles.x != 0.0)   { newDir = normalize(to_cuda(::rotate((Vector<T, 3>)newDir, (Vector<T, 3>)upBasis,  angles.x))); }
  VT3 newRight = normalize(cross(newDir, upBasis));
  if(angles.y != 0.0)   { newDir = normalize(to_cuda(::rotate((Vector<T, 3>)newDir, (Vector<T, 3>)newRight, angles.y))); }
  VT3 newUp    = normalize(cross(newRight, newDir));

  VT3 newPos   = pos;
  if(angles.x != 0.0)   { newPos = to_cuda(::rotate((Vector<T, 3>)newPos, (Vector<T, 3>)upBasis,  angles.x)); }
  VT3 right2   = normalize(cross(newPos, newUp));
  if(angles.y != 0.0)   { newPos = to_cuda(::rotate((Vector<T, 3>)newPos, (Vector<T, 3>)right2,  -angles.y)); }
  dir = newDir; right = newRight; up = newUp; pos = newPos;
}



template<typename T> bool isnan(const CameraDesc<T> &c)
{ return (isnan(c.pos) || isnan(c.dir) || isnan(c.right) || isnan(c.up) || isnan(c.fov) || isnan(c.near) || isnan(c.far) || isnan(c.aspect)); }
template<typename T> bool isinf(const CameraDesc<T> &c)
{ return (isinf(c.pos) || isinf(c.dir) || isinf(c.right) || isinf(c.up) || isinf(c.fov) || isinf(c.near) || isinf(c.far) || isinf(c.aspect)); }

template<typename T> bool isnan(const Camera<T> &c) { return isnan(c.desc); }
template<typename T> bool isinf(const Camera<T> &c) { return isinf(c.desc); }



template<typename T>
inline json cameraToJSON(const CameraDesc<T> &cam)
{
  json js = nlohmann::ordered_json();
  js["pos"]   = to_string(cam.pos, 12);
  js["dir"]   = to_string(cam.dir, 12);
  js["right"] = to_string(cam.right, 12);
  js["up"]    = to_string(cam.up, 12);
  js["fov"]   = cam.fov;
  js["near"]  = cam.near;
  js["far"]   = cam.far;
  return js;
}
template<typename T>
inline bool cameraFromJSON(const json &js, CameraDesc<T> &camOut)
{
  using VT3 = float3;
  bool success = true;
  if(js.contains("pos"))   { camOut.pos   = from_string<VT3>(js["pos"]);   }
  if(js.contains("dir"))   { camOut.dir   = from_string<VT3>(js["dir"]);   }
  if(js.contains("right")) { camOut.right = from_string<VT3>(js["right"]); }
  if(js.contains("up"))    { camOut.up    = from_string<VT3>(js["up"]);    }
  if(js.contains("fov"))   { camOut.fov   = js["fov"];  }
  if(js.contains("near"))  { camOut.near  = js["near"]; }
  if(js.contains("far"))   { camOut.far   = js["far"];  }
  return success;
}


#endif // CAMERA_HPP
