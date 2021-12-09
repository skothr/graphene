#ifndef OVERLAY_HPP
#define OVERLAY_HPP

#include <imgui.h>

#include "vector.hpp"
#include "camera.hpp"
#include "screenView.hpp"

// overlay colors
#define X_COLOR Vec4f(0.0f, 1.0f, 0.0f, 0.5f)
#define Y_COLOR Vec4f(1.0f, 0.0f, 0.0f, 0.5f)
#define Z_COLOR Vec4f(0.1f, 0.1f, 1.0f, 0.8f) // (slightly brighter -- hard to see pure blue over dark background)
#define OUTLINE_COLOR Vec4f(1,1,1,0.5f) // field outline
#define GUIDE_COLOR   Vec4f(1,1,1,0.3f) // pen input guides
#define RADIUS_COLOR  Vec4f(1,1,1,0.4f) // intersecting radii ghosts




// shape helper functions


// clips and transforms line to screen
template<typename T>
void drawClippedLine3D(const ScreenView<T> &view, const Camera<T> &camera, const Vec4f &p0, const Vec4f &p1, const Vec4f &color, float width)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  if(p0.w >= 0 || p1.w >= 0)
    {
      drawList->AddLine(view.toScreen(camera.nearClip(p0, p1)),
                        view.toScreen(camera.nearClip(p1, p0)), ImColor(color), width);
    }
}


// ELLIPSE //
template<typename T>
void drawEllipse2D(const ScreenView<T> &view, const Rect<T> &simView, const Vec2f &center, const Vec2f &radius, const Vec4f &color)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();  
  for(int i = 0; i < 32; i++)
    {
      float a0 = 2.0f*M_PI*(i/32.0f);
      float a1 = 2.0f*M_PI*((i+1)/32.0f);
      Vec2f Sp0 = view.simToScreen2D(center + Vec2f(cos(a0), -sin(a0))*radius, simView);
      Vec2f Sp1 = view.simToScreen2D(center + Vec2f(cos(a1), -sin(a1))*radius, simView);
      drawList->AddLine(Sp0, Sp1, ImColor(color), 1);
    }
}
template<typename T>
void drawEllipse3D(const ScreenView<T> &view, const Camera<T> &camera, const Vec3f &center, const Vec3f &radius, const Vec4f &color)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  float S = 2.0*tan(camera.fov/2.0f*M_PI/180.0f);
  for(int i = 0; i < 32; i++)
    {
      float a0 = 2.0f*M_PI*(i/32.0f); float a1 = 2.0f*M_PI*((i+1)/32.0f);
      Vec4f Sp0 = camera.worldToView(center + (camera.right*cos(a0) - camera.up*sin(a0))*S*to_cuda(radius));
      Vec4f Sp1 = camera.worldToView(center + (camera.right*cos(a1) - camera.up*sin(a1))*S*to_cuda(radius));
      drawClippedLine3D(view, camera, Sp0, Sp1, color, 1.0f);
    }
}

// RECTANGLE //
template<typename T>
void drawRect2D(const ScreenView<T> &view, const Rect<T> &simView, const Vec2f &p0, const Vec2f &p1, const Vec4f &color)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  Vec2f Sp00 = view.simToScreen2D(Vec2f(p0.x, p0.y), simView); Vec2f Sp01 = view.simToScreen2D(Vec2f(p0.x, p1.y), simView);
  Vec2f Sp10 = view.simToScreen2D(Vec2f(p1.x, p0.y), simView); Vec2f Sp11 = view.simToScreen2D(Vec2f(p1.x, p1.y), simView);
  drawList->AddLine(Sp00, Sp01, ImColor(color), 1); drawList->AddLine(Sp01, Sp11, ImColor(color), 1);
  drawList->AddLine(Sp11, Sp10, ImColor(color), 1); drawList->AddLine(Sp10, Sp00, ImColor(color), 1);
}
template<typename T>
void drawRect3D(const ScreenView<T> &view, const Camera<T> &camera, const Vec3f &p0, const Vec3f &p1, const Vec4f &color)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  // transform world points to screen
  Vec4f S000 = camera.worldToView(p0);                      Vec4f S001 = camera.worldToView(Vec3f(p0.x, p0.y, p1.z));
  Vec4f S010 = camera.worldToView(Vec3f(p0.x, p1.y, p0.z)); Vec4f S011 = camera.worldToView(Vec3f(p0.x, p1.y, p1.z));
  Vec4f S100 = camera.worldToView(Vec3f(p1.x, p0.y, p0.z)); Vec4f S101 = camera.worldToView(Vec3f(p1.x, p0.y, p1.z));
  Vec4f S110 = camera.worldToView(Vec3f(p1.x, p1.y, p0.z)); Vec4f S111 = camera.worldToView(p1);

  /// draw
  // XY plane (front, S0XX)
  drawClippedLine3D(view, camera, S000, S001, color, 1.0f); drawClippedLine3D(view, camera, S001, S011, color, 1.0f);
  drawClippedLine3D(view, camera, S011, S010, color, 1.0f); drawClippedLine3D(view, camera, S010, S000, color, 1.0f);
  // XY plane (back,  S1XX)
  drawClippedLine3D(view, camera, S100, S101, color, 1.0f); drawClippedLine3D(view, camera, S101, S111, color, 1.0f);
  drawClippedLine3D(view, camera, S111, S110, color, 1.0f); drawClippedLine3D(view, camera, S110, S100, color, 1.0f);
  // Z connections (S0XX --> S1XX)
  drawClippedLine3D(view, camera, S000, S100, color, 1.0f); drawClippedLine3D(view, camera, S001, S101, color, 1.0f);
  drawClippedLine3D(view, camera, S011, S111, color, 1.0f); drawClippedLine3D(view, camera, S010, S110, color, 1.0f);
}


// SIM X/Y/Z AXES //
template<typename T>
void drawAxes2D(const ScreenView<T> &view, const Rect<T> &rSim, const SimParams &params)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  // draw axes at simulation origin
  float scale  = max(params.cp.fs)*params.cp.u.dL*0.25f;
  float zScale = std::max(params.cp.fs.z*params.cp.u.dL*0.15f, scale/3.0f);
  Vec2f tSize = ImGui::CalcTextSize("X");
  float pad = 5.0f;
  float zW0 = 1.0f; float zW1 = 10.0f; // width of visible z layer bar at min and max
  Vec2f WO0 = Vec2f(0,0); // origin
  Vec2f So  = view.simToScreen2D(WO0, rSim);
  
  // X axis
  Vec2f Spx = view.simToScreen2D(WO0 + Vec2f(scale, 0), rSim);
  drawList->AddLine(So, Spx, ImColor(X_COLOR), 2.0f);
  drawList->AddText((Spx+So)/2.0f - Vec2f(tSize.x/2.0f, 0) + Vec2f(0, pad), ImColor(X_COLOR), "X");
  // Y axis
  Vec2f Spy = view.simToScreen2D(WO0 + Vec2f(0, scale), rSim);
  drawList->AddLine(So, Spy, ImColor(Y_COLOR), 2.0f);
  drawList->AddText((Spy+So)/2.0f - Vec2f(tSize.x, tSize.y/2.0f) - Vec2f(2.0f*pad, 0), ImColor(Y_COLOR), "Y");
  // Z axis (angled to imply depth)
  if(params.cp.fs.z > 1)
    {
      float zAngle = M_PI*4.0f/3.0f;
      Vec2f zVec   = Vec2f(cos(zAngle), sin(zAngle));
      Vec2f zVNorm = Vec2f(zVec.y, zVec.x);
      Vec2f Spz    = view.simToScreen2D(WO0 + zScale*zVec, rSim);
      float zMin = params.rp.zRange.x/(float)(params.cp.fs.z-1);
      float zMax = params.rp.zRange.y/(float)(params.cp.fs.z-1);
      Vec2f SpzMin = view.simToScreen2D(WO0 + zScale*zVec*zMin, rSim);
      Vec2f SpzMax = view.simToScreen2D(WO0 + zScale*zVec*zMax, rSim);
      float zMinW = zW0*(1-zMin) + zW1*zMin;
      float zMaxW = zW0*(1-zMax) + zW1*zMax;
      drawList->AddLine(So, Spz, ImColor(Vec4f(1,1,1,0.5)), 1.0f); // grey line bg
      drawList->AddLine(SpzMin, SpzMax, ImColor(Z_COLOR), 2.0f);   // colored over view range
      // markers showing visible layer range
      drawList->AddLine(SpzMin + Vec2f(zMinW,0), SpzMin - Vec2f(zMinW,0), ImColor(Z_COLOR), 2.0f);
      drawList->AddLine(SpzMax + Vec2f(zMaxW,0), SpzMax - Vec2f(zMaxW,0), ImColor(Z_COLOR), 2.0f);
      std::stringstream ss;   ss << params.rp.zRange.x; std::string zMinStr = ss.str();
      ss.str(""); ss.clear(); ss << params.rp.zRange.y; std::string zMaxStr = ss.str();
      if(zMin != zMax) { drawList->AddText(SpzMin + Vec2f(zMinW+pad, 0), ImColor(Z_COLOR), zMinStr.c_str()); }
      drawList->AddText(SpzMax + Vec2f(zMaxW+pad, 0), ImColor(Z_COLOR), zMaxStr.c_str());
      drawList->AddText((Spz + So)/2.0f - tSize - Vec2f(pad,pad), ImColor(Z_COLOR), "Z");
    }
}
template<typename T>
void drawAxes3D(const ScreenView<T> &view, const Camera<T> &camera, const SimParams &params)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  float scale = max(params.cp.fs)*params.cp.u.dL*0.25f;
  Vec3f WO0 = Vec3f(0,0,0); // origin
  Vec3f Wpx = WO0 + Vec3f(scale, 0, 0);
  Vec3f Wpy = WO0 + Vec3f(0, scale, 0);
  Vec3f Wpz = WO0 + Vec3f(0, 0, scale);
  // transform to screen space
  Vec4f So  = camera.worldToView(WO0); Vec4f Spx = camera.worldToView(Wpx);
  Vec4f Spy = camera.worldToView(Wpy); Vec4f Spz = camera.worldToView(Wpz);
  // draw axes
  drawClippedLine3D(view, camera, Spx, So, X_COLOR, 2.0f);
  drawClippedLine3D(view, camera, Spy, So, Y_COLOR, 2.0f);
  drawClippedLine3D(view, camera, Spz, So, Z_COLOR, 2.0f);
}



// FIELD OUTLINE //
template<typename T>
void drawFieldOutline2D(const ScreenView<T> &view, const Rect<T> &rSim, const SimParams &params)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  // draw outline around field
  Vec3f Wfp0 = Vec3f(params.cp.fp.x, params.cp.fp.y, params.cp.fp.z) * params.cp.u.dL;
  Vec3f Wfs  = Vec3f(params.cp.fs.x, params.cp.fs.y, params.cp.fs.z) * params.cp.u.dL;
  drawRect2D(view, rSim, Vec2f(Wfp0.x, Wfp0.y), Vec2f(Wfp0.x+Wfs.x, Wfp0.y+Wfs.y), RADIUS_COLOR);
}
template<typename T>
void drawFieldOutline3D(const ScreenView<T> &view, const Camera<T> &camera, const SimParams &params)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  Vec3f Wp = Vec3f(params.cp.fp.x, params.cp.fp.y, params.cp.fp.z) * params.cp.u.dL;
  Vec3f Ws = Vec3f(params.cp.fs.x, params.cp.fs.y, params.cp.fs.z) * params.cp.u.dL;
  drawRect3D(view, camera, Wp, Wp+Ws, OUTLINE_COLOR);
}




// PEN OUTLINE/GHOST //
template<typename T>
void drawPenOutline2D(const ScreenView<T> &view, const Rect<T> &rSim, const SimParams &params, const Pen<T> *pen, Vec3f pos)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  if(isnan(pos)) { return; } else if(pen->cellAlign) { pos = floor(pos); }
  // draw positional axes of active signal pen
  Vec2f S01n = view.simToScreen2D(Vec3f(params.cp.fp.x, pos.y,                  pos.z)*params.cp.u.dL, rSim);
  Vec2f S01p = view.simToScreen2D(Vec3f(params.cp.fp.x + params.cp.fs.x, pos.y, pos.z)*params.cp.u.dL, rSim);
  Vec2f S10p = view.simToScreen2D(Vec3f(pos.x, params.cp.fp.y + params.cp.fs.y, pos.z)*params.cp.u.dL, rSim);
  Vec2f S10n = view.simToScreen2D(Vec3f(pos.x, params.cp.fp.y,                  pos.z)*params.cp.u.dL, rSim);
  // X guides
  drawList->AddLine(S01n, S01p, ImColor(GUIDE_COLOR), 2.0f);
  drawList->AddCircleFilled(S01n, 3, ImColor(X_COLOR), 6);
  drawList->AddCircleFilled(S01p, 3, ImColor(X_COLOR), 6);
  // Y guides
  drawList->AddLine(S10n, S10p, ImColor(GUIDE_COLOR), 2.0f);
  drawList->AddCircleFilled(S10n, 3, ImColor(Y_COLOR), 6);
  drawList->AddCircleFilled(S10p, 3, ImColor(Y_COLOR), 6);
  
  // draw lens circle centers
  Vec2f SR0 = (pos + pen->rDist*pen->sizeMult*pen->xyzMult/2.0f).xy() * params.cp.u.dL;
  Vec2f SR1 = (pos - pen->rDist*pen->sizeMult*pen->xyzMult/2.0f).xy() * params.cp.u.dL;
  drawList->AddCircleFilled(view.simToScreen2D(SR0, rSim), 3, ImColor(RADIUS_COLOR), 6);
  drawList->AddCircleFilled(view.simToScreen2D(SR1, rSim), 3, ImColor(RADIUS_COLOR), 6);

  // draw intersected lens circles
  Vec3f r0_3 = pen->radius0 * params.cp.u.dL * pen->sizeMult*pen->xyzMult;
  Vec3f r1_3 = pen->radius1 * params.cp.u.dL * pen->sizeMult*pen->xyzMult;
  Vec2f r0 = Vec2f(r0_3.x, r0_3.y); Vec2f r1 = Vec2f(r1_3.x, r1_3.y);
  if(pen->square) { drawRect2D   (view, rSim, SR0-r0, SR0+r0, RADIUS_COLOR); drawRect2D   (view, rSim, SR1-r1, SR1+r1, RADIUS_COLOR); }
  else            { drawEllipse2D(view, rSim, SR0,    r0,     RADIUS_COLOR); drawEllipse2D(view, rSim, SR1,    r1,     RADIUS_COLOR); }
}

template<typename T>
void drawPenOutline3D(const ScreenView<T> &view, const Camera<T> &camera, const SimParams &params, const Pen<T> *pen, Vec3f pos)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  if(isnan(pos)) { return; } else if(pen->cellAlign) { pos = floor(pos); }
  // transform world points to screen space
  Vec4f S001n = camera.worldToView(Vec3f(params.cp.fp.x, pos.y,                  pos.z)*params.cp.u.dL);
  Vec4f S001p = camera.worldToView(Vec3f(params.cp.fp.x + params.cp.fs.x, pos.y, pos.z)*params.cp.u.dL);
  Vec4f S010n = camera.worldToView(Vec3f(pos.x, params.cp.fp.y + params.cp.fs.y, pos.z)*params.cp.u.dL);
  Vec4f S010p = camera.worldToView(Vec3f(pos.x, params.cp.fp.y,                  pos.z)*params.cp.u.dL);
  Vec4f S100n = camera.worldToView(Vec3f(pos.x, pos.y,                  params.cp.fp.z)*params.cp.u.dL);
  Vec4f S100p = camera.worldToView(Vec3f(pos.x, pos.y, params.cp.fp.z + params.cp.fs.z)*params.cp.u.dL);
  // X guides
  drawClippedLine3D(view, camera, S001n, S001p, GUIDE_COLOR, 2.0f);
  drawList->AddCircleFilled(view.toScreen(camera.vNormalize(S001n)), 3, ImColor(X_COLOR), 6);
  drawList->AddCircleFilled(view.toScreen(camera.vNormalize(S001p)), 3, ImColor(X_COLOR), 6);
  // Y guides
  drawClippedLine3D(view, camera, S010n, S010p, GUIDE_COLOR, 1.0f);
  drawList->AddCircleFilled(view.toScreen(camera.vNormalize(S010n)), 3, ImColor(Y_COLOR), 6);
  drawList->AddCircleFilled(view.toScreen(camera.vNormalize(S010p)), 3, ImColor(Y_COLOR), 6);
  // Z guides
  drawClippedLine3D(view, camera, S100n, S100p, GUIDE_COLOR, 1.0f);
  drawList->AddCircleFilled(view.toScreen(camera.vNormalize(S100n)), 3, ImColor(Z_COLOR), 6);
  drawList->AddCircleFilled(view.toScreen(camera.vNormalize(S100p)), 3, ImColor(Z_COLOR), 6);

  // draw lens circle centers
  float S = 2.0*tan(camera.fov/2.0f*M_PI/180.0f);
  Vec3f WR0 = (pos + pen->rDist*pen->sizeMult*pen->xyzMult/2.0f)*params.cp.u.dL;
  Vec3f WR1 = (pos - pen->rDist*pen->sizeMult*pen->xyzMult/2.0f)*params.cp.u.dL;
  Vec4f SR0 = camera.worldToView(WR0); Vec4f SR1 = camera.worldToView(WR1);
  drawList->AddCircleFilled(view.toScreen(camera.vNormalize(SR0)), 3, ImColor(RADIUS_COLOR), 6);
  drawList->AddCircleFilled(view.toScreen(camera.vNormalize(SR1)), 3, ImColor(RADIUS_COLOR), 6);
  // draw intersected lens circles
  Vec3f r0 = Vec3f(S,S,1)*pen->radius0 * params.cp.u.dL * pen->sizeMult*pen->xyzMult;
  Vec3f r1 = Vec3f(S,S,1)*pen->radius1 * params.cp.u.dL * pen->sizeMult*pen->xyzMult;
  if(pen->square)
    { drawRect3D(view, camera, WR0-r0, WR0+r0, RADIUS_COLOR); drawRect3D(view, camera, WR1-r1, WR1+r1, RADIUS_COLOR); }
  else
    { drawEllipse3D(view, camera, WR0, r0, RADIUS_COLOR); drawEllipse3D(view, camera, WR1, r1, RADIUS_COLOR); }
}



// // MATERIAL PEN OUTLINE/GHOST //
// template<typename T>
// void drawPenOutline2D(const ScreenView<T> &view, const Rect<T> &rSim, const SimParams &params, const MaterialPen<T> *matPen, Vec3f pos)
// {
//   ImDrawList *drawList = ImGui::GetWindowDrawList();
//   if(matPen->cellAlign) { pos = floor(pos); }
//   // draw positional axes of active material pen
//   Vec2f S01n  = view.simToScreen2D(Vec3f(params.cp.fp.x,                  pos.y, pos.z)*params.cp.u.dL, rSim);
//   Vec2f S01p  = view.simToScreen2D(Vec3f(params.cp.fp.x + params.cp.fs.x, pos.y, pos.z)*params.cp.u.dL, rSim);
//   Vec2f S10p  = view.simToScreen2D(Vec3f(pos.x, params.cp.fp.y + params.cp.fs.y, pos.z)*params.cp.u.dL, rSim);
//   Vec2f S10n  = view.simToScreen2D(Vec3f(pos.x, params.cp.fp.y,                  pos.z)*params.cp.u.dL, rSim);
//   // X guide
//   drawList->AddLine(S01n, S01p, ImColor(GUIDE_COLOR), 2.0f);
//   drawList->AddCircleFilled(S01n, 3, ImColor(X_COLOR), 6);
//   drawList->AddCircleFilled(S01p, 3, ImColor(X_COLOR), 6);
//   // Y guide
//   drawList->AddLine(S10n, S10p, ImColor(GUIDE_COLOR), 2.0f);
//   drawList->AddCircleFilled(S10n, 3, ImColor(Y_COLOR), 6);
//   drawList->AddCircleFilled(S10p, 3, ImColor(Y_COLOR), 6);

//   // draw lens circle centers
//   Vec2f SR0 = (pos + matPen->rDist*matPen->sizeMult*matPen->xyzMult/2.0f).xy() * params.cp.u.dL;
//   Vec2f SR1 = (pos - matPen->rDist*matPen->sizeMult*matPen->xyzMult/2.0f).xy() * params.cp.u.dL;
//   drawList->AddCircleFilled(view.simToScreen2D(SR0, rSim), 3, ImColor(RADIUS_COLOR), 6);
//   drawList->AddCircleFilled(view.simToScreen2D(SR1, rSim), 3, ImColor(RADIUS_COLOR), 6);
//   // draw intersected lens circles
//   Vec3f r0_3 = matPen->radius0 * params.cp.u.dL * matPen->sizeMult*matPen->xyzMult;
//   Vec3f r1_3 = matPen->radius1 * params.cp.u.dL * matPen->sizeMult*matPen->xyzMult;
//   Vec2f r0 = Vec2f(r0_3.x, r0_3.y); Vec2f r1 = Vec2f(r1_3.x, r1_3.y);
//   if(matPen->square)
//     { drawRect2D(view, rSim, SR0-r0, SR0+r0, RADIUS_COLOR); drawRect2D(view, rSim, SR1-r1, SR1+r1, RADIUS_COLOR); }
//   else
//     { drawEllipse2D(view, rSim, SR0, r0, RADIUS_COLOR); drawEllipse2D(view, rSim, SR1, r1, RADIUS_COLOR); }
// }

// template<typename T>
// void drawPenOutline3D(const ScreenView<T> &view, const Camera<T> &camera, const SimParams &params, const MaterialPen<T> *matPen, Vec3f pos)
// {
//   ImDrawList *drawList = ImGui::GetWindowDrawList();
//   if(matPen->cellAlign) { pos = floor(pos); }
//   // transform to screen space
//   Vec4f S001n = camera.worldToView(Vec3f(params.cp.fp.x, pos.y,                   pos.z)*params.cp.u.dL);
//   Vec4f S010n = camera.worldToView(Vec3f(pos.x, params.cp.fp.y,                   pos.z)*params.cp.u.dL);
//   Vec4f S100n = camera.worldToView(Vec3f(pos.x, pos.y,                   params.cp.fp.z)*params.cp.u.dL);
//   Vec4f S001p = camera.worldToView(Vec3f(params.cp.fp.x + params.cp.fs.x, pos.y, pos.z)*params.cp.u.dL);
//   Vec4f S010p = camera.worldToView(Vec3f(pos.x, params.cp.fp.y + params.cp.fs.y, pos.z)*params.cp.u.dL);
//   Vec4f S100p = camera.worldToView(Vec3f(pos.x, pos.y, params.cp.fp.z + params.cp.fs.z)*params.cp.u.dL);

//   // X guide
//   drawClippedLine3D(view, camera, S001n, S001p, GUIDE_COLOR, 1.0f);
//   drawList->AddCircleFilled(view.toScreen(camera.vNormalize(S001n)), 3, ImColor(X_COLOR), 6);
//   drawList->AddCircleFilled(view.toScreen(camera.vNormalize(S001p)), 3, ImColor(X_COLOR), 6);
//   // Y guide
//   drawClippedLine3D(view, camera, S010n, S010p, GUIDE_COLOR, 1.0f);
//   drawList->AddCircleFilled(view.toScreen(camera.vNormalize(S010n)), 3, ImColor(Y_COLOR), 6);
//   drawList->AddCircleFilled(view.toScreen(camera.vNormalize(S010p)), 3, ImColor(Y_COLOR), 6);
//   // Z guide
//   drawClippedLine3D(view, camera, S100n, S100p, GUIDE_COLOR, 1.0f);
//   drawList->AddCircleFilled(view.toScreen(camera.vNormalize(S100n)), 3, ImColor(Z_COLOR), 6);
//   drawList->AddCircleFilled(view.toScreen(camera.vNormalize(S100p)), 3, ImColor(Z_COLOR), 6);

//   float S = 2.0*tan(camera.fov/2.0f*M_PI/180.0f);
//   // draw lens circle centers
//   Vec4f SR0 = camera.worldToView((pos + matPen->rDist*matPen->sizeMult*matPen->xyzMult/2.0f)*params.cp.u.dL);
//   Vec4f SR1 = camera.worldToView((pos - matPen->rDist*matPen->sizeMult*matPen->xyzMult/2.0f)*params.cp.u.dL);
//   drawList->AddCircleFilled(view.toScreen(camera.vNormalize(SR0)), 3, ImColor(RADIUS_COLOR), 6);
//   drawList->AddCircleFilled(view.toScreen(camera.vNormalize(SR1)), 3, ImColor(RADIUS_COLOR), 6);
//   // draw intersected lens circles
//   Vec3f r0 = Vec3f(S,S,1)*matPen->radius0 * params.cp.u.dL * matPen->sizeMult*matPen->xyzMult;
//   Vec3f r1 = Vec3f(S,S,1)*matPen->radius1 * params.cp.u.dL * matPen->sizeMult*matPen->xyzMult;
//   if(matPen->square)
//     { drawRect3D(view, camera, WR0-r0, WR0+r0, RADIUS_COLOR); drawRect3D(view, camera, WR1-r1, WR1+r1, RADIUS_COLOR); }
//   else
//     { drawEllipse3D(view, camera, WR0, r0, RADIUS_COLOR); drawEllipse3D(view, camera, WR1, r1, RADIUS_COLOR); }
// }





// VECTORS //
inline void drawVector2D(ImDrawList *drawList, const Vec2f &p, const Vec2f &v, const Vec4f &color, float width, float tipW, float tipAspect)
{
  float vL = length(v);
  Vec2f v0 = v / vL;
  Vec2f n0 = Vec2f(v0.y, -v0.x); // CW normal
  float theta = M_PI/3.0;

  drawList->Flags &= ~ImDrawListFlags_AntiAliasedFill; // turn off antialiasing to avoid shape overlap
  
  width      = std::min(width, vL);
  tipW       = std::max(tipW, 3.0f*width);
  
  float tipL = std::min(tipW*tipAspect, vL);
  tipW = tipL/tipAspect;
  
  if(vL > tipL)
    { // line
      Vec2f pl00 = p + n0*width/2.0f;
      Vec2f pl01 = p - n0*width/2.0f;
      Vec2f pl10 = p + n0*width/2.0f + v - v0*tipL;
      Vec2f pl11 = p - n0*width/2.0f + v - v0*tipL;
      drawList->AddTriangleFilled(pl00, pl10, pl11, ImColor(color));
      drawList->AddTriangleFilled(pl11, pl01, pl00, ImColor(color));
    }
  
  // arrow tip
  Vec2f pt0 = p + v;
  Vec2f pt1 = p + v0*(vL-tipL) + n0*tipW/2;
  Vec2f pt2 = p + v0*(vL-tipL) - n0*tipW/2;
  drawList->AddTriangleFilled(pt0, pt1, pt2, ImColor(color));
}

inline void drawVector3D(ImDrawList *drawList, const Vec2f &p, const Vec3f &v, const Vec4f &color, float width, float tipW, float tipAspect)
{
  // float vL = length(v);
  // Vec2f v0 = v / vL;
  // Vec2f n0 = Vec2f(v0.y, -v0.x); // CW normal
  // float theta = M_PI/3.0;

  // drawList->Flags &= ~ImDrawListFlags_AntiAliasedFill; // turn off antialiasing to avoid shape overlap
  
  // width      = std::min(width, vL);
  // tipW       = std::max(tipW, 3.0f*width);
  
  // float tipL = std::min(tipW*tipAspect, vL);
  // tipW = tipL/tipAspect;
  
  // if(vL > tipL)
  //   { // line
  //     Vec2f pl00 = p + n0*width/2.0f;
  //     Vec2f pl01 = p - n0*width/2.0f;
  //     Vec2f pl10 = p + n0*width/2.0f + v - v0*tipL;
  //     Vec2f pl11 = p - n0*width/2.0f + v - v0*tipL;
  //     drawList->AddTriangleFilled(pl00, pl10, pl11, ImColor(color));
  //     drawList->AddTriangleFilled(pl11, pl01, pl00, ImColor(color));
  //   }
  
  // // arrow tip
  // Vec2f pt0 = p + v;
  // Vec2f pt1 = p + v0*(vL-tipL) + n0*tipW/2;
  // Vec2f pt2 = p + v0*(vL-tipL) - n0*tipW/2;
  // drawList->AddTriangleFilled(pt0, pt1, pt2, ImColor(color));
}











#endif // OVERLAY_HPP
