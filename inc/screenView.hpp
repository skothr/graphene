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
  
  MouseButton clickBtns(MouseButton mask=MOUSEBTN_ALL) const { return (clicked & mask); }
  int         clickMods(int mask=0)                    const { return (mods    & mask); }

  bool operator==(const ScreenView &other) const { return  (r == other.r);   }
  bool operator!=(const ScreenView &other) const { return !(*this == other); }
};


#endif // SCREEN_VIEW_HPP
