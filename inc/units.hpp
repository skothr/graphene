#ifndef UNITS_HPP
#define UNITS_HPP

#include <imgui.h>
#include "imtools.hpp"
#include "physics.h"
#include "settingForm.hpp"

// ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσΤτΥυΦφΧχΨψΩω

template<typename T>
struct Units
{
  T dt = 0.01; // TIME   (field timestep)
  T dL = 0.1;  // LENGTH (field cell size)
  T e0 = 1.0;  // E -- permittivity of free space
  T m0 = 1.0;  // B -- permeability of free space
  T s0 = 0.0;  // Q -- conductivity of free space (?)
  Material<T> vacuum() const { return Material<T>(e0, m0, s0, true); }  
};

template<typename T>
struct UnitsInterface
{
  Units<T> *units = nullptr;
  ImFont *superFont = nullptr;
  UnitsInterface(Units<T> *u=nullptr, ImFont *sfont=nullptr) : units(u), superFont(sfont) { }
  void draw();
};

template<>
inline void UnitsInterface<float>::draw()
{
  if(units)
    {
      ImGui::TextUnformatted("dt");     ImGui::SameLine(); ImGui::InputFloat("##dt", &units->dt, 0.01, 0.1, "%f");
      ImGui::TextUnformatted("dL");     ImGui::SameLine(); ImGui::InputFloat("##dL", &units->dL, 0.01, 0.1, "%f");
      TextPhysics("ε<^(0)", superFont); ImGui::SameLine(); ImGui::InputFloat("##e0", &units->e0, 0.01, 0.1, "%f");
      TextPhysics("μ<^(0)", superFont); ImGui::SameLine(); ImGui::InputFloat("##m0", &units->m0, 0.01, 0.1, "%f");
      TextPhysics("σ<^(0)", superFont); ImGui::SameLine(); ImGui::InputFloat("##m0", &units->s0, 0.01, 0.1, "%f");
      // ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos())+Vec2f(0.0f, 40.0f));
      // TextPhysics("TEST: σ<^(0) + x<^(0)>^((2+3)*y) / (test)>^(2)*x<^(000).1", superFont); 
    }
}


#endif // UNITS_HPP
