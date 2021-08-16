#ifndef UNITS_HPP
#define UNITS_HPP

#include <imgui.h>
#include "imtools.hpp"
#include "physics.h"
#include "settingForm.hpp"

// ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσΤτΥυΦφΧχΨψΩω

#define UNIT_INPUT_W 137

template<typename T>
struct Units
{

  // discretization
  T dt = 0.1;  // TIME   (field update timestep)
  T dL = 1.0;  // LENGTH (field cell size)
  // EM
  T e  = 1.0;  // CHARGE (elementary charge)
  T a  = 1.0/137.0; // fine structure constant
  T e0 = 1.0;  // electric constant / permittivity of free space    (E)
  T u0 = 1.0;  // magnetic constant / permeability of free space    (B)
  T s0 = 0.0;  // conductivity of free space (may just be abstract) (Q?)

  __host__ __device__ Material<T> vacuum() const { return Material<T>(e0, u0, 0.0, true); }

  // derived
  __host__ __device__ T c() const { return 1/(T)sqrt(e0*u0); }
  __host__ __device__ T h() const { return u0*e*e*c() / (2.0*a); }
  
};

template<typename T>
struct UnitsInterface
{
  Units<T> *units = nullptr;
  ImFont *superFont = nullptr;
  UnitsInterface(Units<T> *u=nullptr, ImFont *sfont=nullptr) : units(u), superFont(sfont) { }
  void draw();
};
template<> inline void UnitsInterface<float>::draw()
{
  if(units)
    {
      ImGui::TextUnformatted("dt");     ImGui::SameLine(); ImGui::SetNextItemWidth(UNIT_INPUT_W); ImGui::InputFloat("##dt", &units->dt, 0.01, 0.1, "%f");
      ImGui::SameLine(); ImGui::TextUnformatted("physics timestep");
      ImGui::TextUnformatted("dL");     ImGui::SameLine(); ImGui::SetNextItemWidth(UNIT_INPUT_W); ImGui::InputFloat("##dL", &units->dL, 0.01, 0.1, "%f");
      ImGui::SameLine(); ImGui::TextUnformatted("field cell size");

      ImGui::TextUnformatted("e ");     ImGui::SameLine(); ImGui::SetNextItemWidth(UNIT_INPUT_W); ImGui::InputFloat("##e",  &units->e, 0.01, 0.1, "%f");
      ImGui::SameLine(); ImGui::TextUnformatted("elementary charge");
      ImGui::TextUnformatted("α ");     ImGui::SameLine(); ImGui::SetNextItemWidth(UNIT_INPUT_W); ImGui::InputFloat("##a",  &units->a, 0.01, 0.1, "%f");
      ImGui::SameLine(); ImGui::TextUnformatted("fine structure constant");
      
      TextPhysics("ε<^(0)", superFont); ImGui::SameLine(); ImGui::SetNextItemWidth(UNIT_INPUT_W); ImGui::InputFloat("##ε0", &units->e0, 0.01, 0.1, "%f");
      ImGui::SameLine(); ImGui::TextUnformatted("electric contant (permittivity)");
      TextPhysics("μ<^(0)", superFont); ImGui::SameLine(); ImGui::SetNextItemWidth(UNIT_INPUT_W); ImGui::InputFloat("##μ0", &units->u0, 0.01, 0.1, "%f");
      ImGui::SameLine(); ImGui::TextUnformatted("magnetic contant (permeability)");
      TextPhysics("σ<^(0)", superFont); ImGui::SameLine(); ImGui::SetNextItemWidth(UNIT_INPUT_W); ImGui::InputFloat("##σ0", &units->s0, 0.01, 0.1, "%f");
      ImGui::SameLine(); ImGui::TextUnformatted("(?) (conductivity)");

      ImGui::Separator();
      ImGui::TextUnformatted("Derived:");
      TextPhysics("    c = 1/sqrt(ε<^(0)μ<^(0)) =  ", superFont); ImGui::SameLine(); ImGui::Text("%f", units->c());
      TextPhysics("    h = e>^(2)μ<^(0)c / (2α) =  ", superFont); ImGui::SameLine(); ImGui::Text("%f", units->h());
      
    }
}


#endif // UNITS_HPP
