#ifndef UNITS_HPP
#define UNITS_HPP

#include <imgui.h>
#include "imtools.hpp"
#include "units.cuh"
#include "setting.hpp"
#include "settingForm.hpp"

// ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσΤτΥυΦφΧχΨψΩω

#define UNIT_INPUT_W 111

template<typename T>
struct UnitsInterface
{
  Units<T> *units  = nullptr;
  ImFont *superFont = nullptr;
  
  SettingForm *mForm = nullptr;
  json toJSON() const           { return (mForm ? mForm->toJSON() : json::object()); }
  bool fromJSON(const json &js) { return (mForm ? mForm->fromJSON(js) : false); }
  
  UnitsInterface(Units<T> *u=nullptr, ImFont *sfont=nullptr)
    : units(u), superFont(sfont)
  {
    mForm = new SettingForm();
    mForm->add(new Setting<T>("dt", "dt", &units->dt));
    mForm->add(new Setting<T>("dL", "dL", &units->dL));
    mForm->add(new Setting<T>("e",  "e",  &units->e ));
    mForm->add(new Setting<T>("α",  "a",  &units->a ));
    mForm->add(new Setting<T>("ε₀", "e0", &units->e0));
    mForm->add(new Setting<T>("μ₀", "u0", &units->u0));
    mForm->add(new Setting<T>("σ₀", "s0", &units->s0));
  }

  ~UnitsInterface() { if(mForm) { delete mForm; mForm = nullptr; } }
  
  void drawUnit(const std::string &name, T *ptr, T step0, T step1, const std::string &format, const std::string &desc);
  void draw();
  void updateAll() { mForm->updateAll(); }
};

template<> inline void UnitsInterface<float>::drawUnit(const std::string &name, float *ptr, float step0, float step1,
                                                       const std::string &format, const std::string &desc)
{
  ImGui::TextUnformatted(name.c_str());
  ImGui::SameLine(); ImGui::SetNextItemWidth(UNIT_INPUT_W); ImGui::InputFloat(("##"+name).c_str(), ptr, step0, step1, format.c_str());
  ImGui::SameLine(); ImGui::TextUnformatted(desc.c_str());
}
template<> inline void UnitsInterface<double>::drawUnit(const std::string &name, double *ptr, double step0, double step1,
                                                        const std::string &format, const std::string &desc)
{
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(name.c_str());
  ImGui::SameLine(); ImGui::SetNextItemWidth(UNIT_INPUT_W); ImGui::InputDouble(("##"+name).c_str(), ptr, step0, step1, format.c_str());
  ImGui::SameLine(); ImGui::TextUnformatted(desc.c_str());
}

template<> inline void UnitsInterface<float>::draw()
{
  if(units)
    {
      drawUnit("dt", &units->dt, 0.01f, 0.1f, "%f", "physics timestep");
      drawUnit("dL", &units->dL, 0.01f, 0.1f, "%f", "field cell size");
      drawUnit("e ", &units->e,  0.01f, 0.1f, "%f", "elementary charge");
      drawUnit("α ", &units->a,  0.01f, 0.1f, "%f", "fine structure constant");
      
      TextPhysics("ε<^(0)", superFont); ImGui::SameLine(); ImGui::SetNextItemWidth(UNIT_INPUT_W); ImGui::InputFloat("##ε0", &units->e0, 0.01, 0.1, "%f");
      ImGui::SameLine(); ImGui::TextUnformatted("electric contant (permittivity)");
      TextPhysics("μ<^(0)", superFont); ImGui::SameLine(); ImGui::SetNextItemWidth(UNIT_INPUT_W); ImGui::InputFloat("##μ0", &units->u0, 0.01, 0.1, "%f");
      ImGui::SameLine(); ImGui::TextUnformatted("magnetic contant (permeability)");
      TextPhysics("σ<^(0)", superFont); ImGui::SameLine(); ImGui::SetNextItemWidth(UNIT_INPUT_W); ImGui::InputFloat("##σ0", &units->s0, 0.01, 0.1, "%f");
      ImGui::SameLine(); ImGui::TextUnformatted("(?)              (conductivity)");

      ImGui::Separator();
      ImGui::TextUnformatted("Derived:");
      TextPhysics("    c = 1 / (ε<^(0)μ<^(0))>^(1/2) ", superFont); ImGui::SameLine(); Vec2f pE = ImGui::GetCursorPos(); ImGui::Text(" = %f", units->c());
      TextPhysics("    h = e>^(2)μ<^(0)c / (2α)      ", superFont); ImGui::SameLine(pE.x); ImGui::Text(" = %f", units->h());
    }
}

template<> inline void UnitsInterface<double>::draw()
{
  if(units)
    {
      drawUnit("dt", &units->dt, 0.01f, 0.1f, "%f", "physics timestep");
      drawUnit("dL", &units->dL, 0.01f, 0.1f, "%f", "field cell size");
      drawUnit("e ", &units->e,  0.01f, 0.1f, "%f", "elementary charge");
      drawUnit("α ", &units->a,  0.01f, 0.1f, "%f", "fine structure constant");

      TextPhysics("ε<^(0)", superFont); ImGui::SameLine(); ImGui::SetNextItemWidth(UNIT_INPUT_W); ImGui::InputDouble("##ε0", &units->e0, 0.01, 0.1, "%f");
      ImGui::SameLine(); ImGui::TextUnformatted("electric contant (permittivity)");
      TextPhysics("μ<^(0)", superFont); ImGui::SameLine(); ImGui::SetNextItemWidth(UNIT_INPUT_W); ImGui::InputDouble("##μ0", &units->u0, 0.01, 0.1, "%f");
      ImGui::SameLine(); ImGui::TextUnformatted("magnetic contant (permeability)");
      TextPhysics("σ<^(0)", superFont); ImGui::SameLine(); ImGui::SetNextItemWidth(UNIT_INPUT_W); ImGui::InputDouble("##σ0", &units->s0, 0.01, 0.1, "%f");
      ImGui::SameLine(); ImGui::TextUnformatted("(?)              (conductivity)");

      ImGui::Separator();
      ImGui::TextUnformatted("Derived:");
      TextPhysics("    c = 1 / (ε<^(0)μ<^(0))>^(1/2) ", superFont); ImGui::SameLine(); Vec2f pE = ImGui::GetCursorPos(); ImGui::Text(" = %f", units->c());
      TextPhysics("    h = e>^(2)μ<^(0)c / (2α)      ", superFont); ImGui::SameLine(pE.x); ImGui::Text(" = %f", units->h());
    }
}


#endif // UNITS_HPP
