#ifndef SETTINGS_HPP
#define SETTINGS_HPP

#include <string>
#include <vector>
#include <imgui.h>

#include "vector-operators.h"
#include "vector.hpp"

// unified vector data access (Vector<T,N>/intN/floatN/doubleN)
template<typename T> struct VElement
{
  static constexpr typename Dim<T>::BASE_T& get(T &v, unsigned int i)
  {
    if      constexpr(is_vec_v<T>)  { return v[i];      } // Vector<T,N>
    else if constexpr(is_cuda_v<T>) { return arr(v)[i]; } // intN / floatN / doubleN
  }
};


static const std::vector<std::string> g_vecLabels{{"X", "Y", "Z", "W"}}; // e.g. for int/float/double [2-4]

// input for intN / Vector<int, N> types
template<typename T, int N=Dim<T>::N>
inline bool InputIntV(const std::string &id, T *data, T &step=T(), T &step_fast=T(), T *dmin=nullptr, T *dmax=nullptr,
                      bool horizontal=true, bool liveEdit=false, bool extendLabel=false,
                      ImGuiInputTextFlags flags=0)
{
  using Ve = VElement<T>;
  ImGuiStyle &style = ImGui::GetStyle();
  if(extendLabel)
    {
      Vec2f tSize = ImGui::CalcTextSize(g_vecLabels.size() > 0 ? g_vecLabels[0].c_str() : "");
      Vec2f p0    = Vec2f(ImGui::GetCursorPos()) - Vec2f(tSize.x+(tSize.x > 0 ? style.ItemSpacing.x : 0.0f), 0.0f);
      ImGui::SetCursorPos(p0);
    }

  bool edited = liveEdit || (ImGui::IsKeyPressed(GLFW_KEY_ENTER) || ImGui::IsKeyPressed(GLFW_KEY_TAB) ||
                             ImGui::IsMouseDown(ImGuiMouseButton_Left) || ImGui::IsMouseReleased(ImGuiMouseButton_Left));
  bool changed = false;
  if(ImGui::BeginTable(("##"+id).c_str(), (horizontal ? N : 1), (ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoPadOuterX)))
    {
      typename Dim<T>::BASE_T v;
      for(int i = 0; i < N; i++)
        {
          v = Ve::get(*data, i);
          ImGui::TableNextColumn();
          ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted((g_vecLabels.size() > i) ? g_vecLabels[i].c_str() : "");
          ImGui::SameLine(); ImGui::SetNextItemWidth(-1.0f);
          if(ImGui::InputInt(("##"+id+std::to_string(i)).c_str(), &v, Ve::get(step, i), Ve::get(step_fast, i), flags) && edited)
            {
              if(dmax) { v = std::min(Ve::get(*dmax, i), v); } if(dmin) { v = std::max(Ve::get(*dmin, i), v); }
              changed = true; Ve::get(*data, i) = v;
            }
        }
      ImGui::EndTable();
    }
  return changed;
}


// input for floatN / Vector<float, N> types
template<typename T, int N=Dim<T>::N>
inline bool InputFloatV(const std::string &id, T *data, T &step=T(), T &step_fast=T(), T *dmin=nullptr, T *dmax=nullptr,
                        const std::string &format="%.3f", bool horizontal=true, bool liveEdit=false, bool extendLabel=false,
                        ImGuiInputTextFlags flags=0)
{
  using Ve = VElement<T>;
  ImGuiStyle &style = ImGui::GetStyle();

  // 
  bool edited = liveEdit || (ImGui::IsKeyPressed(GLFW_KEY_ENTER) || ImGui::IsKeyPressed(GLFW_KEY_TAB) ||
                             ImGui::IsMouseDown(ImGuiMouseButton_Left) || ImGui::IsMouseReleased(ImGuiMouseButton_Left));

  if(extendLabel)
    { // offset label so start of input lines up to cursor
      Vec2f tSize = ImGui::CalcTextSize(g_vecLabels.size() > 0 ? g_vecLabels[0].c_str() : "");
      ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos()) - Vec2f(tSize.x+(tSize.x > 0 ? style.ItemSpacing.x : 0.0f), 0.0f));
    }
  
  bool changed = false;
  if(ImGui::BeginTable(("##"+id).c_str(), (horizontal ? N : 1), (ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoPadOuterX)))
    {
      typename Dim<T>::BASE_T v;
      for(int i = 0; i < N; i++)
        {
          v = Ve::get(*data, i);
          ImGui::TableNextColumn();
          ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted((g_vecLabels.size() > i) ? g_vecLabels[i].c_str() : "");
          ImGui::SameLine(); ImGui::SetNextItemWidth(-1.0f);
          if(ImGui::InputFloat(("##"+id+std::to_string(i)).c_str(), &v, Ve::get(step, i), Ve::get(step_fast, i),
                               format.c_str(), flags | ImGuiInputTextFlags_CharsScientific) && edited)
            {
              if(dmax) { v = std::min(Ve::get(*dmax, i), v); } if(dmin) { v = std::max(Ve::get(*dmin, i), v); }
              changed = true; Ve::get(*data, i) = v;
            }
        }
      ImGui::EndTable();
    }
  return changed;
}


// input for doubleN / Vector<double, N> types
template<typename T, int N=Dim<T>::N>
inline bool InputDoubleV(const std::string &id, T *data, T &step=T(), T &step_fast=T(), T *dmin=nullptr, T *dmax=nullptr,
                         const std::string &format="%.3f", bool horizontal=true, bool liveEdit=false, bool extendLabel=false,
                         ImGuiInputTextFlags flags=0)
{
  using Ve = VElement<T>;
  ImGuiStyle &style = ImGui::GetStyle();
  if(extendLabel)
    {
      Vec2f tSize = ImGui::CalcTextSize(g_vecLabels.size() > 0 ? g_vecLabels[0].c_str() : "");
      Vec2f p0    = Vec2f(ImGui::GetCursorPos()) - Vec2f(tSize.x+(tSize.x > 0 ? style.ItemSpacing.x : 0.0f), 0.0f);
      ImGui::SetCursorPos(p0);
    }

  bool edited = liveEdit || (ImGui::IsKeyPressed(GLFW_KEY_ENTER) || ImGui::IsKeyPressed(GLFW_KEY_TAB) ||
                             ImGui::IsMouseDown(ImGuiMouseButton_Left) || ImGui::IsMouseReleased(ImGuiMouseButton_Left));
  
  bool changed = false;
  if(ImGui::BeginTable(("##"+id).c_str(), (horizontal ? N : 1), (ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoPadOuterX)))
    {
      typename Dim<T>::BASE_T v;
      for(int i = 0; i < N; i++)
        {
          v = Ve::get(*data, i);
          ImGui::TableNextColumn();
          ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted((g_vecLabels.size() > i) ? g_vecLabels[i].c_str() : "");
          ImGui::SameLine(); ImGui::SetNextItemWidth(-1.0f);
          if(ImGui::InputDouble(("##"+id+std::to_string(i)).c_str(), &v, Ve::get(step, i), Ve::get(step_fast, i),
                                format.c_str(), flags | ImGuiInputTextFlags_CharsScientific) && edited)
            {
              if(dmax) { v = std::min(Ve::get(*dmax, i), v); } if(dmin) { v = std::max(Ve::get(*dmin, i), v); }
              changed = true; Ve::get(*data, i) = v;
            }
        }
      ImGui::EndTable();
    }
  return changed;
}




#endif // SETTINGS_HPP
