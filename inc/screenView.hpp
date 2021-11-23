#ifndef SCREEN_VIEW_HPP
#define SCREEN_VIEW_HPP

#include "tools.hpp"
#include "vector.hpp"
#include "rect.hpp"

// simple flag type for mouse buttons 
enum MouseButton
  { // NOTE: Values should match ImGuiButtonFlags_ in imgui.h
   MOUSEBTN_NONE   =  0,         // ImGuiButtonFlags_None
   MOUSEBTN_LEFT   =  1 << 0,    // ImGuiButtonFlags_MouseButtonLeft
   MOUSEBTN_RIGHT  =  1 << 1,    // ImGuiButtonFlags_MouseButtonRight
   MOUSEBTN_MIDDLE =  1 << 2,    // ImGuiButtonFlags_MouseButtonMiddle
   MOUSEBTN_ALL    = (1 << 3)-1, // All flags set (next available flag minus one (e.g. (1 << 4) --> 0x10000 --> 0x10000-1 = 0x01111)
  };
ENUM_FLAG_OPERATORS(MouseButton)

template<typename T>
struct ScreenView
{
  Rect2f r;
  bool         hovered  = false;
  MouseButton  clicked  = MOUSEBTN_NONE; // mouse buttons that were clicked
  int          mods     = 0;             // modifier keys that were held when clicked (GLFW_MOD_XXXX)
  Vec2f        clickPos;                 // screen position of click
  Vector<T, 3> mposSim;                  // sim position of click
  Vector<T, 3> mposFace;                 // face of cube mouse is over (e.g. <1,0,0> for +X face)
  
  MouseButton clickBtns(MouseButton mask=MOUSEBTN_ALL) const { return (clicked & mask); }
  int         clickMods(int mask=0)                    const { return (mods    & mask); }

  bool operator==(const ScreenView &other) const { return  (r == other.r);   }
  bool operator!=(const ScreenView &other) const { return !(*this == other); }

  // scales output from 3D Camera<T> transform (P0 --> camera.nearClip(camera.worldToScreen(P0), camera.worldToScreen(P1)))
  Vec2f toScreen(const Vec2f &p) const { return r.p1 + p*r.size(); }

  Vec2f simToScreen2D(const Vec2f &pSim,    const Rect2f &simView, bool vector=false) const;
  Vec2f screenToSim2D(const Vec2f &pScreen, const Rect2f &simView, bool vector=false) const;
  // (to use X and Y components of 3D vectors)
  Vec2f simToScreen2D(const Vec3f &pSim,    const Rect2f &simView, bool vector=false) const;
  Vec2f screenToSim2D(const Vec3f &pScreen, const Rect2f &simView, bool vector=false) const;

};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Conversion between screen space and sim space
//// p0 --> optional screen-space offset
template<typename T>
Vec2f ScreenView<T>::simToScreen2D(const Vec2f &pSim, const Rect2f &simView, bool vector) const
{
  Vec2f pScreen = (pSim-simView.p1*(vector?0:1)) * (r.size()/simView.size());
  if(!vector) { pScreen = Vec2f(r.p1.x + pScreen.x, r.p2.y - pScreen.y); }
  return pScreen;
}
template<typename T>
Vec2f ScreenView<T>::screenToSim2D(const Vec2f &pScreen, const Rect2f &simView, bool vector) const
{
  Vec2f pSim = (pScreen-r.p1*(vector?0:1)) * (simView.size()/r.size());
  if(!vector) { pSim = Vec2f(simView.p1.x + pSim.x, simView.p2.y - pSim.y); }
  return pSim;
}

// for 3D vectors, uses X/Y components (for convenience)
template<typename T>
Vec2f ScreenView<T>::simToScreen2D(const Vec3f &pSim, const Rect2f &simView, bool vector) const
{ return simToScreen2D(Vec2f(pSim.x,    pSim.y),    simView, vector); }
template<typename T>
Vec2f ScreenView<T>::screenToSim2D(const Vec3f &pScreen, const Rect2f &simView, bool vector) const
{ return screenToSim2D(Vec2f(pScreen.x, pScreen.y), simView, vector); }


#endif // SCREEN_VIEW_HPP
