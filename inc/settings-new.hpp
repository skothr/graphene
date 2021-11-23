#ifndef SETTINGS_HPP
#define SETTINGS_HPP

#include <string>
#include <vector>
#include <imgui.h>

#include "vector-operators.h"
#include "vector.hpp"

// unified vector data access (Vector<T,N>/intN/floatN/doubleN)
template<typename T> struct VElement { static constexpr typename Dim<T>::BASE_T& get(T &v, unsigned int i) { return v[i]; } };
// (no operator[] overload for cuda structs) 
template<> struct VElement<int2>     { static constexpr int&    get(int2    &v, unsigned int i) { return arr(v)[i]; } };
template<> struct VElement<int3>     { static constexpr int&    get(int3    &v, unsigned int i) { return arr(v)[i]; } };
template<> struct VElement<int4>     { static constexpr int&    get(int4    &v, unsigned int i) { return arr(v)[i]; } };
template<> struct VElement<float2>   { static constexpr float&  get(float2  &v, unsigned int i) { return arr(v)[i]; } };
template<> struct VElement<float3>   { static constexpr float&  get(float3  &v, unsigned int i) { return arr(v)[i]; } };
template<> struct VElement<float4>   { static constexpr float&  get(float4  &v, unsigned int i) { return arr(v)[i]; } };
template<> struct VElement<double2>  { static constexpr double& get(double2 &v, unsigned int i) { return arr(v)[i]; } };
template<> struct VElement<double3>  { static constexpr double& get(double3 &v, unsigned int i) { return arr(v)[i]; } };
template<> struct VElement<double4>  { static constexpr double& get(double4 &v, unsigned int i) { return arr(v)[i]; } };


static const std::vector<std::string> g_vecLabels{{"X", "Y", "Z", "W"}}; // e.g. for int/float/double [2-4]


// input for intN / Vector<int, N> types
template<typename T, int N=Dim<T>::N>
inline bool InputIntV(const std::string &id, T *data, T &step=T(), T &step_fast=T(), T *dmin=nullptr, T *dmax=nullptr,
                      bool horizontal=true, bool liveEdit=false, bool extendLabel=false,
                      ImGuiInputTextFlags flags=0)
{
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
          v = VElement<T>::get(*data, i);
          ImGui::TableNextColumn();
          ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted((g_vecLabels.size() > i) ? g_vecLabels[i].c_str() : "");
          ImGui::SameLine(); ImGui::SetNextItemWidth(-1.0f);
          if(ImGui::InputInt(("##"+id+std::to_string(i)).c_str(), &v, VElement<T>::get(step, i), VElement<T>::get(step_fast, i), flags) && edited)
            {
              if(dmax) { v = std::min(VElement<T>::get(*dmax, i), v); } if(dmin) { v = std::max(VElement<T>::get(*dmin, i), v); }
              changed = true; VElement<T>::get(*data, i) = v;
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
          v = VElement<T>::get(*data, i);
          ImGui::TableNextColumn();
          ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted((g_vecLabels.size() > i) ? g_vecLabels[i].c_str() : "");
          ImGui::SameLine(); ImGui::SetNextItemWidth(-1.0f);
          if(ImGui::InputFloat(("##"+id+std::to_string(i)).c_str(), &v, VElement<T>::get(step, i), VElement<T>::get(step_fast, i),
                               format.c_str(), flags | ImGuiInputTextFlags_CharsScientific) && edited)
            {
              if(dmax) { v = std::min(VElement<T>::get(*dmax, i), v); } if(dmin) { v = std::max(VElement<T>::get(*dmin, i), v); }
              changed = true; VElement<T>::get(*data, i) = v;
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
          v = VElement<T>::get(*data, i);
          ImGui::TableNextColumn();
          ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted((g_vecLabels.size() > i) ? g_vecLabels[i].c_str() : "");
          ImGui::SameLine(); ImGui::SetNextItemWidth(-1.0f);
          if(ImGui::InputDouble(("##"+id+std::to_string(i)).c_str(), &v, VElement<T>::get(step, i), VElement<T>::get(step_fast, i),
                                format.c_str(), flags | ImGuiInputTextFlags_CharsScientific) && edited)
            {
              if(dmax) { v = std::min(VElement<T>::get(*dmax, i), v); } if(dmin) { v = std::max(VElement<T>::get(*dmin, i), v); }
              changed = true; VElement<T>::get(*data, i) = v;
            }
        }
      ImGui::EndTable();
    }
  return changed;
}




#endif // SETTINGS_HPP
