#ifndef UNITS_HPP
#define UNITS_HPP

#include <imgui.h>
#include "imtools.hpp"
#include "units.cuh"
#include "setting.hpp"
#include "settingForm.hpp"


#define UNIT_INPUT_W 111

template<typename T>
struct UnitsInterface : public SettingForm
{
  Units<T> *units  = nullptr;
  ImFont *superFont = nullptr;
  
  UnitsInterface(Units<T> *u=nullptr, ImFont *sfont=nullptr)
    : units(u), superFont(sfont)
  {
    auto *g = new SettingGroup("Units", "units", { }); add(g);
    
    auto sDT = new Setting<T>("dt", "dt", &units->dt); g->add(sDT);
    sDT->setHelp("dt --> simulation timestep");
    sDT->setFormat(0.001, 0.01, "%.8f");
    auto sDL = new Setting<T>("dL", "dL", &units->dL); g->add(sDL);
    sDL->setHelp("dL --> simulation length step (i.e. length/width/height of each cell)");
    sDL->setFormat(0.001, 0.01, "%.8f");
    auto sE  = new Setting<T>("e",  "e",  &units->e ); g->add(sE);
    sE->setHelp( "e  --> elementary charge (currently unused)");
    sE->setFormat(0.001, 0.01, "%.8f");
    auto sA  = new Setting<T>("α",  "a",  &units->a ); g->add(sA);
    sA->setHelp( "α  --> fine structure constant (currently unused)");
    sA->setFormat(0.001, 0.01, "%.8f");
    auto sE0 = new Setting<T>("ε<^(0)", "e0", &units->e0); g->add(sE0);
    sE0->setHelp("ε<^(0) --> permittivity of free space / electric constant");
    sE0->setFormat(0.001, 0.01, "%.8f");
    auto sU0 = new Setting<T>("μ<^(0)", "u0", &units->u0); g->add(sU0);
    sU0->setHelp("μ<^(0) --> permeabililty of free space / magnetic constant");
    sU0->setFormat(0.001, 0.01, "%.8f");
    auto sS0 = new Setting<T>("σ<^(0)", "s0", &units->s0); g->add(sS0);
    sS0->setHelp("σ<^(0) --> \"conductivity\" of free space(?) (material properties)");
    sS0->setFormat(0.001, 0.01, "%.8f");
  }
  ~UnitsInterface() = default;
  
  void drawUnit(const std::string &name, T *ptr, T step0, T step1, const std::string &format, const std::string &desc);
  virtual bool draw() override;
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

template<typename T> inline bool UnitsInterface<T>::draw()
{
  bool changed = false;
  if(units)
    {
      changed |= SettingForm::draw();
      ImGui::Separator();
      ImGui::TextUnformatted("Derived:");
      TextPhysics("    c = 1 / (ε<^(0)μ<^(0))>^(1/2) ", superFont); ImGui::SameLine(); Vec2f pE = ImGui::GetCursorPos(); ImGui::Text(" = %f", units->c());
      TextPhysics("    h = e>^(2)μ<^(0)c / (2α) ", superFont); ImGui::SameLine(pE.x); ImGui::Text("= %f", units->h());
    }
  return changed;
}

// ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσΤτΥυΦφΧχΨψΩω


#endif // UNITS_HPP
