#ifndef OVERLAY_HPP
#define OVERLAY_HPP

#include <imgui.h>

#include "vector.hpp"
#include "camera.hpp"
#include "screenView.hpp"


// shape helper functions
template<typename T>
void drawRect3D(const ScreenView<T> &view, const Camera<T> &camera, const Vec3f &p0, const Vec3f &p1, const Vec4f &color)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  // world points
  Vec3f W000  = p0;
  Vec3f W001  = Vec3f(p0.x, p0.y, p1.z);
  Vec3f W010  = Vec3f(p0.x, p1.y, p0.z);
  Vec3f W011  = Vec3f(p0.x, p1.y, p1.z);
  Vec3f W100  = Vec3f(p1.x, p0.y, p0.z);
  Vec3f W101  = Vec3f(p1.x, p0.y, p1.z);
  Vec3f W110  = Vec3f(p1.x, p1.y, p0.z);
  Vec3f W111  = p1;
  // transform/project
  bool C000 = false; bool C001 = false; bool C010 = false; bool C011 = false;
  bool C100 = false; bool C101 = false; bool C110 = false; bool C111 = false;
  Vec4f S000 = camera.worldToView(W000, &C000); Vec4f S001 = camera.worldToView(W001, &C001);
  Vec4f S010 = camera.worldToView(W010, &C010); Vec4f S011 = camera.worldToView(W011, &C011);
  Vec4f S100 = camera.worldToView(W100, &C100); Vec4f S101 = camera.worldToView(W101, &C101);
  Vec4f S110 = camera.worldToView(W110, &C110); Vec4f S111 = camera.worldToView(W111, &C111);

  /// draw
  // XY plane (front -- 0XX)
  if(!C000 || !C001) { drawList->AddLine(view.toScreen(camera.nearClip(S000, S001)), view.toScreen(camera.nearClip(S001, S000)), ImColor(color), 1.0f); }
  if(!C001 || !C011) { drawList->AddLine(view.toScreen(camera.nearClip(S001, S011)), view.toScreen(camera.nearClip(S011, S001)), ImColor(color), 1.0f); }
  if(!C011 || !C010) { drawList->AddLine(view.toScreen(camera.nearClip(S011, S010)), view.toScreen(camera.nearClip(S010, S011)), ImColor(color), 1.0f); }
  if(!C010 || !C000) { drawList->AddLine(view.toScreen(camera.nearClip(S010, S000)), view.toScreen(camera.nearClip(S000, S010)), ImColor(color), 1.0f); }
  // XY plane (back -- 1XX)
  if(!C100 || !C101) { drawList->AddLine(view.toScreen(camera.nearClip(S100, S101)), view.toScreen(camera.nearClip(S101, S100)), ImColor(color), 1.0f); }
  if(!C101 || !C111) { drawList->AddLine(view.toScreen(camera.nearClip(S101, S111)), view.toScreen(camera.nearClip(S111, S101)), ImColor(color), 1.0f); }
  if(!C111 || !C110) { drawList->AddLine(view.toScreen(camera.nearClip(S111, S110)), view.toScreen(camera.nearClip(S110, S111)), ImColor(color), 1.0f); }
  if(!C110 || !C100) { drawList->AddLine(view.toScreen(camera.nearClip(S110, S100)), view.toScreen(camera.nearClip(S100, S110)), ImColor(color), 1.0f); }
  // Z connections -- (0XX - 1XX)
  if(!C000 || !C100) { drawList->AddLine(view.toScreen(camera.nearClip(S000, S100)), view.toScreen(camera.nearClip(S100, S000)), ImColor(color), 1.0f); }
  if(!C001 || !C101) { drawList->AddLine(view.toScreen(camera.nearClip(S001, S101)), view.toScreen(camera.nearClip(S101, S001)), ImColor(color), 1.0f); }
  if(!C011 || !C111) { drawList->AddLine(view.toScreen(camera.nearClip(S011, S111)), view.toScreen(camera.nearClip(S111, S011)), ImColor(color), 1.0f); }
  if(!C010 || !C110) { drawList->AddLine(view.toScreen(camera.nearClip(S010, S110)), view.toScreen(camera.nearClip(S110, S010)), ImColor(color), 1.0f); }
}


template<typename T>
void drawEllipse3D(const ScreenView<T> &view, const Camera<T> &camera, const Vec3f &center, const Vec3f &radius, const Vec4f &color)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  float S = 2.0*tan(camera.fov/2.0f*M_PI/180.0f);
  for(int i = 0; i < 32; i++)
    {
      float a0 = 2.0f*M_PI*(i/32.0f);
      float a1 = 2.0f*M_PI*((i+1)/32.0f);
      Vec3f Wp0 = center + ((camera.right*cos(a0) - camera.up*sin(a0))*S*to_cuda(radius));
      Vec3f Wp1 = center + ((camera.right*cos(a1) - camera.up*sin(a1))*S*to_cuda(radius));
      bool Cp0 = false; bool Cp1 = false;
      Vec4f Sp0 = camera.worldToView(Wp0, &Cp0);
      Vec4f Sp1 = camera.worldToView(Wp1, &Cp1);
      drawList->AddLine(view.toScreen(camera.nearClip(Sp0, Sp1)), view.toScreen(camera.nearClip(Sp1, Sp0)), ImColor(color), 1);
    }
}


template<typename T>
void drawRect2D(const ScreenView<T> &view, const Rect<T> &simView, const Vec2f &p0, const Vec2f &p1, const Vec4f &color)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();  
  Vec2f Wp00 = Vec2f(p0.x, p0.y);
  Vec2f Wp01 = Vec2f(p0.x, p1.y);
  Vec2f Wp10 = Vec2f(p1.x, p0.y);
  Vec2f Wp11 = Vec2f(p1.x, p1.y);
  Vec2f Sp00 = view.simToScreen2D(Wp00, simView);
  Vec2f Sp01 = view.simToScreen2D(Wp01, simView);
  Vec2f Sp10 = view.simToScreen2D(Wp10, simView);
  Vec2f Sp11 = view.simToScreen2D(Wp11, simView);
  drawList->AddLine(Sp00, Sp01, ImColor(color), 1);
  drawList->AddLine(Sp01, Sp11, ImColor(color), 1);
  drawList->AddLine(Sp11, Sp10, ImColor(color), 1);
  drawList->AddLine(Sp10, Sp00, ImColor(color), 1);
}

template<typename T>
void drawEllipse2D(const ScreenView<T> &view, const Rect<T> &simView, const Vec2f &center, const Vec2f &radius, const Vec4f &color)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();  
  for(int i = 0; i < 32; i++)
    {
      float a0 = 2.0f*M_PI*(i/32.0f);
      float a1 = 2.0f*M_PI*((i+1)/32.0f);
      Vec2f Wp0 = center + Vec2f(cos(a0), -sin(a0))*radius;
      Vec2f Wp1 = center + Vec2f(cos(a1), -sin(a1))*radius;
      Vec2f Sp0 = view.simToScreen2D(Wp0, simView);
      Vec2f Sp1 = view.simToScreen2D(Wp1, simView);
      drawList->AddLine(Sp0, Sp1, ImColor(color), 1);
    }
}





#endif // OVERLAY_HPP
