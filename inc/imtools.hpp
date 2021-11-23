#ifndef IMTOOLS_HPP
#define IMTOOLS_HPP

#include <imgui.h>
#include <imgui_internal.h>
#include <vector>
#include <string>
#include <cmath>

#include "glfwKeys.hpp"
#include "rect.hpp"


static ImFont *DEFAULT_SUPER_FONT = nullptr;
inline void setDefaultSuperFont(ImFont *superFont) { DEFAULT_SUPER_FONT = superFont; }


// superscript text
inline void TextSuper(const std::string &text, ImFont *superFont=DEFAULT_SUPER_FONT, bool normalSpacing=true)
{
  if(text.empty()) { return; }
  
  Vec2f spacing = Vec2f(ImGui::GetStyle().ItemSpacing.x, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, Vec2f(0,0));
  {
    Vec2f p0     = ImGui::GetCursorPos();
    Vec2f tSize0 = ImGui::CalcTextSize(text.c_str());
    
    if(superFont) { ImGui::PushFont(superFont); }
    Vec2f sSize = ImGui::CalcTextSize(text.c_str());
    Vec2f sDiff = tSize0 - sSize;
    ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos()) + Vec2f(-spacing.x, 0));
    ImGui::TextUnformatted(text.c_str());
    ImGui::SetCursorPos(p0 + Vec2f(sSize.x+sDiff.x, 0.0f));
    if(superFont) { ImGui::PopFont(); }
    
    // remove extra horizontal spacing (?)
    ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos()) - Vec2f(spacing.x, 0.0f));
    if(normalSpacing) { ImGui::TextUnformatted(""); } // next line
  }
  ImGui::PopStyleVar();
}

// subscript text
inline void TextSub(const std::string &text, ImFont *superFont=DEFAULT_SUPER_FONT, bool normalSpacing=true)
{
  if(text.empty()) { return; }
  
  Vec2f spacing = Vec2f(ImGui::GetStyle().ItemSpacing.x, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, Vec2f(0,0));
  {  
    Vec2f p0     = ImGui::GetCursorPos();
    Vec2f tSize0 = ImGui::CalcTextSize(text.c_str());

    if(superFont) { ImGui::PushFont(superFont); }
    Vec2f sSize = ImGui::CalcTextSize(text.c_str());
    Vec2f sDiff = tSize0 - sSize;
    ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos()) + Vec2f(-spacing.x, sDiff.y + sSize.y*0.4f));
    ImGui::TextUnformatted(text.c_str());
    ImGui::SetCursorPos(p0 + Vec2f(sSize.x+sDiff.x, 0.0f));
    if(superFont) { ImGui::PopFont(); }
    // remove extra spacing (?)
    ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos()) - Vec2f(spacing.x, 0.0f));
    if(normalSpacing) { ImGui::TextUnformatted(""); } // next line
  }
  ImGui::PopStyleVar();
}

// keys for denoting equation-related text
static const std::string SUPER_KEY = ">^"; // "x>^(2)" ==> x²
static const std::string SUB_KEY   = "<^"; // "x<^(2)" ==> x₂
static const std::string FRACT_KEY = "/^"; // "x/^(2)" ==> x/2 (vertical) (TODO)

// combined text with plain, superscript, and subscript (denoted using keys)
inline void TextPhysics(const std::string &text, ImFont *superFont=DEFAULT_SUPER_FONT)
{
  if(superFont) { ImGui::PushFont(superFont); }
  Vec2f sSize = ImGui::CalcTextSize(text.c_str());
  if(superFont) { ImGui::PopFont(); }
  
  Vec2f spacing = Vec2f(ImGui::GetStyle().ItemSpacing.x, 0.0f);  
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, Vec2f(0,0));
  Vec2f p0    = ImGui::GetCursorPos();
  Vec2f tSize = ImGui::CalcTextSize(text.c_str());
  
  std::string str = text;
  while(!text.empty())
    {
      bool found = false;

      // check for superscript
      auto openSuper = str.find(SUPER_KEY + "(");
      auto openSub   = str.find(SUB_KEY   + "(");
      //auto openFract = str.find(")" + FRACT_KEY + "("); // TODO: fractions separated by horizontal line

      if(openSuper != std::string::npos && (openSub == std::string::npos || openSuper < openSub))
        { // found superscript
          // skip groups of parentheses within superscript parentheses
          auto offset = openSuper + SUPER_KEY.size()+1;
          auto openParen = str.find("(", offset); auto closeParen = str.find(")", offset);
          bool subParen = false;
          while(openParen != std::string::npos && openParen < closeParen)
            { subParen = true; offset = openParen+1; openParen = str.find("(", offset); closeParen = str.find(")", offset); }
          auto closeSuper = subParen ? str.find(")", closeParen+1) : closeParen;
          
          // extract superscript string
          if(closeSuper != std::string::npos)
            {
              std::string lhs      = str.substr(0, openSuper);
              std::string strSuper = str.substr(openSuper+(SUPER_KEY.size()+1), closeSuper-openSuper-(SUPER_KEY.size()+1));
              str = str.substr(closeSuper+1);
              if(!lhs.empty())      { ImGui::TextUnformatted(lhs.c_str()); ImGui::SameLine(); }
              if(!strSuper.empty()) { TextSuper(strSuper, superFont, false); }
              found = true;
            }
        }
      else if(openSub != std::string::npos && (openSuper == std::string::npos || openSub < openSuper))
        { // found subscript
          // skip groups of parentheses within subscript parentheses
          auto offset = openSub + SUB_KEY.size()+1;
          auto openParen = str.find("(", offset); auto closeParen = str.find(")", offset);
          bool subParen = false;
          while(openParen != std::string::npos && openParen < closeParen)
            { subParen = true; offset = openParen+1; openParen = str.find("(", offset); closeParen = str.find(")", offset); }
          auto closeSub = subParen ? str.find(")", closeParen+1) : closeParen;

          // extract subscript string
          if(closeSub != std::string::npos)
            {
              std::string lhs   = str.substr(0, openSub);
              std::string strSub = str.substr(openSub+(SUB_KEY.size()+1), closeSub-openSub-(SUB_KEY.size()+1));
              str = str.substr(closeSub+1);
              if(!lhs.empty())    { ImGui::TextUnformatted(lhs.c_str()); ImGui::SameLine(); }
              if(!strSub.empty()) { TextSub(strSub, superFont, false); }
              found = true;
            }
        }      
      if(!found)
        { // rest is plain text
          ImGui::TextUnformatted(str.c_str());
          str = ""; break;
        }
    }
  ImGui::PopStyleVar();
}







// small checkbox
inline bool Checkbox(const std::string &label, bool* v, const Vec2f &size=Vec2f(0,0))
{
  ImGuiWindow* window = ImGui::GetCurrentWindow();
  if(window->SkipItems) { return false; }

  // ImGuiContext& g = *GImGui;
  const ImGuiStyle& style = ImGui::GetStyle();
  const ImGuiID     id    = window->GetID(label.c_str());
  const Vec2f label_size  = ImGui::CalcTextSize(label.c_str(), NULL, true);

  float square_sz = (size.x > 0 || size.y > 0 ? std::max(size.x, size.y) : ImGui::GetFrameHeight());
  const float pad_mult  = 0.1f;
  const float frame_pad = square_sz*pad_mult;
  square_sz -= 2.0f*frame_pad;
  
  const Vec2f pos = Vec2f(window->DC.CursorPos) + Vec2f(2.0f*frame_pad);
  Vec2f       sz  = Vec2f(square_sz + (label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f), label_size.y + style.FramePadding.y*2.0f);
  if(size.x > 0.0f) { sz.x = size.x; } if(size.y > 0.0f) { sz.y = size.y; }
    
  const ImRect total_bb(pos, pos + sz);
  ImGui::ItemSize(total_bb, style.FramePadding.y);
  if(!ImGui::ItemAdd(total_bb, id)) { return false; }
  
  bool hovered, held; bool pressed = ImGui::ButtonBehavior(total_bb, id, &hovered, &held);
  if(pressed) { *v = !(*v); ImGui::MarkItemEdited(id); }

  Rect2f check_bb(pos, pos + Vec2f(square_sz, square_sz));
  if(label_size.x > 0.0f)
    {
      ImGui::RenderText(ImVec2(check_bb.p1.x + style.ItemInnerSpacing.x, check_bb.p1.y + style.FramePadding.y), label.c_str());
    }

  check_bb += Vec2f(label_size.x, 0.0f);
  ImGui::RenderNavHighlight(total_bb, id);
  ImGui::RenderFrame(check_bb.p1, check_bb.p2, ImGui::GetColorU32((held && hovered) ? ImGuiCol_FrameBgActive :
                                                                  hovered ? ImGuiCol_FrameBgHovered : ImGuiCol_FrameBg), true, style.FrameRounding);
  
  ImU32 check_col = ImGui::GetColorU32(ImGuiCol_CheckMark);
  if(*v)
    {
      const float pad = ImMax(1.0f, square_sz*pad_mult);
      ImGui::RenderCheckMark(window->DrawList, check_bb.p1+Vec2f(pad, pad), check_col, square_sz-2.0f*pad);
    }
  return pressed;
}


#define SLIDER_BTN_W 10.0f
template<typename T>
inline bool RangeSlider(const std::string &label, T* v0, T* v1, T vMin, T vMax, const Vec2f &size=Vec2f(250,20))
{
  bool changed = false;
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  Vec2f p0 = ImGui::GetCursorScreenPos();

  drawList->AddLine(Vec2f(p0.x, p0.y + size.y/2.0f), Vec2f(p0.x+size.x, p0.y + size.y/2.0f), ImColor(Vec4f(0.5f, 0.5f, 0.5f, 1.0f)), 2.0f);
  ImGui::SetCursorScreenPos(Vec2f(p0.x - SLIDER_BTN_W/2.0f + (size.x*(*v0 - vMin)/(vMax - vMin)), p0.y));
  ImGui::Button((label+"##btn0").c_str(), Vec2f(SLIDER_BTN_W, size.y));
  bool c0 = ImGui::IsItemActive();
  ImGui::SetCursorScreenPos(Vec2f(p0.x - SLIDER_BTN_W/2.0f + (size.x*(*v1 - vMin)/(vMax - vMin)), p0.y));
  ImGui::Button((label+"##btn1").c_str(), Vec2f(SLIDER_BTN_W, size.y));
  bool c1 = ImGui::IsItemActive();

  float mval = (vMax-vMin)*(ImGui::GetMousePos().x - p0.x)/size.x;
  mval = std::max(std::min(mval+0.5f, (float)vMax), (float)vMin);
  
  if     (c0) { *v0 = mval; *v1 = std::max(*v0, *v1); changed = true; }
  else if(c1) { *v1 = mval; *v0 = std::min(*v0, *v1); changed = true; }
  *v0 = std::max(std::min(*v0, vMax), vMin);
  *v1 = std::max(std::min(*v1, vMax), vMin);

  if(!changed)
    {
      ImGui::SetCursorScreenPos(p0);
      ImGui::InvisibleButton((label+"##btnInv").c_str(), size);
      if(ImGui::IsItemActive())
        {
          if     (mval < *v0) { *v0 = mval; }
          else if(mval > *v1) { *v1 = mval;} 
          else if(abs(mval - *v0) < abs(mval - *v1)) { *v0 = mval; }
          else if(abs(mval - *v1) < abs(mval - *v0)) { *v1 = mval; }
        }
    }

  ImGui::SetCursorScreenPos(Vec2f(p0.x, ImGui::GetCursorScreenPos().y)); ImGui::Text("%d", vMin);
  ImGui::SameLine(); ImGui::SetCursorScreenPos(Vec2f(p0.x + size.x, ImGui::GetCursorScreenPos().y)); ImGui::Text("%d", vMax);
  ImGui::SameLine(); ImGui::SetCursorScreenPos(Vec2f(p0.x + size.x*(*v0-vMin)/(vMax-vMin) , ImGui::GetCursorScreenPos().y)); ImGui::Text("%d", *v0);
  ImGui::SameLine(); ImGui::SetCursorScreenPos(Vec2f(p0.x + size.x*(*v1-vMin)/(vMax-vMin) , ImGui::GetCursorScreenPos().y)); ImGui::Text("%d", *v1);

  return changed;
}

/// Draws vertical text (rotated). The position is the bottom left of the text rect.
inline void AddTextVertical(ImDrawList* DrawList, const char *text, Vec2f pos, const Vec4f &textColor, bool upToDown=true)
{
  pos.x = IM_ROUND(pos.x);
  pos.y = IM_ROUND(pos.y);
  ImFont *font = GImGui->Font;
  const ImFontGlyph *glyph;
  char c;
  // ImGuiContext& g = *GImGui;
  Vec2f textSize = ImGui::CalcTextSize(text);
  pos.x += textSize.y;
  
  int dirMult = (upToDown ? 1 : -1); // direction
  
  while((c = *text++))
    {
      glyph = font->FindGlyph(c);
      if(!glyph) { continue; }

      DrawList->PrimReserve(6, 4);
      DrawList->PrimQuadUV(pos + Vec2f(-glyph->Y0*dirMult, glyph->X0*dirMult),
                           pos + Vec2f(-glyph->Y0*dirMult, glyph->X1*dirMult),
                           pos + Vec2f(-glyph->Y1*dirMult, glyph->X1*dirMult),
                           pos + Vec2f(-glyph->Y1*dirMult, glyph->X0*dirMult),
                           Vec2f(glyph->U0, glyph->V0),
                           Vec2f(glyph->U1, glyph->V0),
                           Vec2f(glyph->U1, glyph->V1),
                           Vec2f(glyph->U0, glyph->V1),
                           ImColor(textColor));
      pos.y += glyph->AdvanceX*dirMult;
    }
}
  










// draws a bordered line (simple rectangle)
//   TODO(?): endpoint type (e.g. triangular, rounded, 
inline void drawLine(ImDrawList *drawList, Vec2f p0, Vec2f p1,              // screen points to connect, and center point
                     const Vec4f &color, float lWidth,                      // color/width of border
                     const Vec4f &bColor=Vec4f(0,0,0,0), float bWidth=0.0f, // color/width of border
                     float shear0 = 1.0f, float shear1 = 1.0f)              // shear at each point (scales line width locally)
{
  if(!drawList) { return; }

  // main line  calculations
  Vec2f v  = (p1 - p0).normalized(); // vector between points
  Vec2f n  = Vec2f(v.y, -v.x);       // normal
  Vec2f mp = (p0 + p1) / 2.0f;       // midpoint

  // draw line body (ends at specified points)
  Vec2f p00 = p0 - (shear0*lWidth/2.0f)*n;
  Vec2f p01 = p0 + (shear0*lWidth/2.0f)*n;
  Vec2f p10 = p1 - (shear1*lWidth/2.0f)*n;
  Vec2f p11 = p1 + (shear1*lWidth/2.0f)*n;

  if(bWidth > 0.01 && bColor.w > 0.01)
    { // draw border
      Vec2f bp00 = p00 + (-v - n)*bWidth;
      Vec2f bp01 = p01 + (-v + n)*bWidth;
      Vec2f bp10 = p10 + ( v - n)*bWidth;
      Vec2f bp11 = p11 + ( v + n)*bWidth;    

      std::vector<Vec2f> points;
      std::vector<Vec2f> bPoints;
      // { p00,  p10,  p11,  p01  }; // inner body points   (clockwise)
      // { bp00, bp10, bp11, bp01 }; // outer border points (clockwise)

      points.push_back(p00); bPoints.push_back(p00);
      if(shear0 > 0.1f) { points.push_back(p01); bPoints.push_back(bp01); }
      points.push_back(p11); bPoints.push_back(p11);
      if(shear1 > 0.1f) { points.push_back(p10); bPoints.push_back(bp10); }

      for(int i = 0; i < points.size(); i++)
        {
          int j = (i+1) % points.size();
          drawList->AddTriangleFilled(points[i], bPoints[i], points[j],  ImColor(bColor));
          drawList->AddTriangleFilled(points[j], bPoints[i], bPoints[j], ImColor(bColor));
        }
    }
  // draw inner line  
  if(color.w > 0.01)
    { // draw line body (connecting specified points)
      if(shear0 > 0.1f) { drawList->AddTriangleFilled(p00, p01, p10, ImColor(color)); }
      if(shear1 > 0.1f) { drawList->AddTriangleFilled(p10, p01, p11, ImColor(color)); }
    }
}


inline void drawVector(ImDrawList *drawList, const Vec2f &p, const Vec2f &v, const Vec4f &color, float width, float tipW, float tipAspect)
{
  float vL = length(v);
  Vec2f v0 = v / vL;
  Vec2f n0 = Vec2f(v0.y, -v0.x); // CW normal
  float theta = M_PI/3.0;

  drawList->Flags &= ~ImDrawListFlags_AntiAliasedFill; // turn off antialiasing to avoid shape overlap
  
  width      = std::min(width, vL);
  tipW       = std::max(tipW, 3.0f*width);
  
  // if(tipW > 0.8f*vL)
  //   {
  //     width = vL*0.8f/tipAspect;
  //     tipW  = std::max(tipW, 3.0f*width);
  //   }
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


#endif // IMTOOLS_HPP
