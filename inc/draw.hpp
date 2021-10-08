#ifndef DRAW_HPP
#define DRAW_HPP

#include <nlohmann/json.hpp> // json implementation (NOTE: should be in settings.hpp?)
using json = nlohmann::json;

#include "vector-operators.h"
#include "raytrace.h"
#include "material.h"
#include "units.hpp"
#include "draw.cuh"
#include "setting.hpp"
#include "settingForm.hpp"


template<typename T>
struct DrawInterface
{
  // Pen *activePen = nullptr; // TODO: better pen abstraction
  SignalPen<T>    sigPen;
  MaterialPen<T>  matPen;
  Units<T>        *mUnits = nullptr;
  SettingForm     *mForm  = nullptr;
  json toJSON() const           { return (mForm ? mForm->toJSON() : json::object()); }
  bool fromJSON(const json &js) { bool success = (mForm ? mForm->fromJSON(js) : false); return success; }

  DrawInterface(Units<T> *units);
  ~DrawInterface();

  void draw(ImFont *superFont=nullptr);
  void updateAll() { mForm->updateAll(); }
};




template<typename T>
DrawInterface<T>::DrawInterface(Units<T> *units)
  : mUnits(units)
{  
  typedef typename DimType<T, 3>::VEC_T VT3;
  SettingGroup *sigGroup = new SettingGroup("Add Signal (Ctrl+Click)",  "sigPen", { }, false);
  SettingGroup *matGroup = new SettingGroup("Add Material (Alt+Click)", "matPen", { }, false);
  
  // signal pen
  auto *sSPA   = new Setting<bool> ("Active",          "sigPenActive",   &sigPen.active);
  sigGroup->add(sSPA);
  auto *sSPR0  = new Setting<VT3>  ("Radius 0",        "sigPenRad0",     &sigPen.radius0);
  sSPR0->setFormat(VT3{1.0f, 1.0f, 1.0f}, VT3{10.0f, 10.0f, 10.0f}, "%0.4f");
  sigGroup->add(sSPR0);
  auto *sSPR1  = new Setting<VT3>  ("Radius 1",        "sigPenRad1",     &sigPen.radius1);
  sSPR1->setFormat(VT3{0.0f, 0.0f, 0.0f}, VT3{10.0f, 10.0f, 10.0f}, "%1.4f");
  sigGroup->add(sSPR1);
  auto *sSPRD  = new Setting<VT3>  ("R Offset",        "sigPenRDist",    &sigPen.rDist);
  sSPRD->setFormat(VT3{1.0f, 1.0f, 1.0f}, VT3{10.0f, 10.0f, 10.0f}, "%1.4f");
  sigGroup->add(sSPRD);
  auto *sSSM   = new Setting<T>    ("Size Multiplier", "sigPenSizeMult", &sigPen.sizeMult);
  sSSM->setFormat(0.1f, 1.0f, "%0.4f");
  sigGroup->add(sSSM);
  auto *sSXM   = new Setting<VT3>  ("XYZ Multiplier",  "sigPenXYZMult",  &sigPen.xyzMult);
  sSXM->setFormat(VT3{0.1f, 0.1f, 0.1f}, VT3{1.0f, 1.0f, 1.0f}, "%0.4f");
  sigGroup->add(sSXM);
  auto *sSPD   = new Setting<int>  ("Depth",           "sigPenDepth",    &sigPen.depth); sSPD->setFormat(1, 8, ""); 
  sigGroup->add(sSPD);
  auto *sSPS   = new Setting<bool> ("Square",          "sigPenSquare",   &sigPen.square);
  sigGroup->add(sSPS);
  auto *sSPAL  = new Setting<bool> ("Align to Cell",   "sigPenAlign",    &sigPen.cellAlign);
  sigGroup->add(sSPAL);
  auto *sSPR   = new Setting<bool> ("Radial",          "sigPenRadial",   &sigPen.radial);
  sigGroup->add(sSPR);
  auto *sSPMS  = new Setting<bool> ("Mouse Speed",     "mouseSpeed",     &sigPen.speed);
  sigGroup->add(sSPMS);
  auto *sSPMSM = new Setting<T>    ("Speed Multiplier\n","speedMult",    &sigPen.speedMult); sSPMSM->setFormat(0.1f, 1.0f, "%0.4f");
  sigGroup->add(sSPMSM);
  
  auto *sSPM   = new Setting<T>    ("Amplitude",       "sigPenMult",     &sigPen.mult); sSPM->setFormat(0.1f, 1.0f, "%0.4f");
  sigGroup->add(sSPM);
  auto *sSPF   = new Setting<T>    ("Frequency",       "sigPenFreq",     &sigPen.frequency);
  sSPF->updateCallback = [this](){ sigPen.wavelength = mUnits->c() / sigPen.frequency; };
  sSPF->setFormat(0.01f, 0.1f, "%0.4f"); sSPF->setMin(0.0f);
  sigGroup->add(sSPF);
  auto *sSPW   = new Setting<T>    ("Wavelength",      "sigPenWV",     &sigPen.wavelength);
  sSPW->updateCallback = [this]() { sigPen.frequency = mUnits->c() / sigPen.wavelength; };
  sSPW->setFormat(0.1f, 1.0f, "%0.4f"); sSPW->setMin(0.0f);
  sigGroup->add(sSPW);
  
  auto *sSPV  = new Setting<float3>("V",  "sigPenVBase",  &sigPen.pV.base);
  sSPV->setFormat(float3{0.01f, 0.01f, 0.01f}, float3{0.1f, 0.1f, 0.1f}, "%0.4f");
  sigGroup->add(sSPV);
  auto *sSPP  = new Setting<float> ("P",  "sigPenPBase",  &sigPen.pP.base);
  sSPP->setFormat(0.01f, 0.1f, "%0.4f");
  sigGroup->add(sSPP);
  auto *sSPQn = new Setting<float> ("Q-", "sigPenQnBase", &sigPen.pQn.base);
  sSPQn->setFormat(0.01f, 0.1f, "%0.4f");
  sigGroup->add(sSPQn);
  auto *sSPQp = new Setting<float> ("Q+", "sigPenQpBase", &sigPen.pQp.base);
  sSPQp->setFormat(0.01f, 0.1f, "%0.4f");
  sigGroup->add(sSPQp);
  auto *sSPQv = new Setting<float3>("Qv", "sigPenQvBase", &sigPen.pQv.base);
  sSPQv->setFormat(float3{0.01f, 0.01f, 0.01f}, float3{0.1f, 0.1f, 0.1f}, "%0.4f");
  sigGroup->add(sSPQv);
  auto *sSPE  = new Setting<float3>("E", "sigPenEBase", &sigPen.pE.base);
  sSPE->setFormat(float3{0.01f, 0.01f, 0.01f}, float3{0.1f, 0.1f, 0.1f}, "%0.4f");
  sigGroup->add(sSPE);
  auto *sSPB  = new Setting<float3>("B", "sigPenBBase", &sigPen.pB.base);
  sSPB->setFormat(float3{0.01f, 0.01f, 0.01f}, float3{0.1f, 0.1f, 0.1f}, "%0.4f");
  sigGroup->add(sSPB);
  
  auto *sSPVb = new Setting<std::array<bool, 5>>("",  "sigPenVMods",  &sigPen.pV.modArr);
  sSPVb->vColumns = 5;  sSPVb->vRowLabels = {{0, "V  "}};                   
  sSPVb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}}; sSPVb->drawColLabels = true;
  sigGroup->add(sSPVb);                                                     
  auto *sSPPb = new Setting<std::array<bool, 5>>("",  "sigPenPMods",  &sigPen.pP.modArr);
  sSPPb->vColumns = 5;  sSPPb->vRowLabels = {{0, "P  "}};                   
  sSPPb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}};
  sigGroup->add(sSPPb);                                                     
  auto *sSPQNb = new Setting<std::array<bool, 5>>("", "sigPenQnMods", &sigPen.pQn.modArr);
  sSPQNb->vColumns = 5;  sSPQNb->vRowLabels = {{0, "Q- "}};                 
  sSPQNb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}};
  sigGroup->add(sSPQNb);                                                    
  auto *sSPQPb = new Setting<std::array<bool, 5>>("", "sigPenQpMods", &sigPen.pQp.modArr);
  sSPQPb->vColumns = 5;  sSPQPb->vRowLabels = {{0, "Q+ "}};                 
  sSPQPb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}};
  sigGroup->add(sSPQPb);                                                    
  auto *sSPQVb = new Setting<std::array<bool, 5>>("", "sigPenQvMods", &sigPen.pQv.modArr);
  sSPQVb->vColumns = 5;  sSPQVb->vRowLabels = {{0, "QV "}};                 
  sSPQVb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}};
  sigGroup->add(sSPQVb);                                                    
  auto *sSPEb = new Setting<std::array<bool, 5>>("",  "sigPenEMods", &sigPen.pE.modArr);
  sSPEb->vColumns = 5;  sSPEb->vRowLabels = {{0, "E  "}};                   
  sSPEb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}};
  sigGroup->add(sSPEb);                                                     
  auto *sSPBb = new Setting<std::array<bool, 5>>("",  "sigPenBMods",  &sigPen.pB.modArr);
  sSPBb->vColumns = 5;  sSPBb->vRowLabels = {{0, "B  "}};
  sSPBb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}};
  sigGroup->add(sSPBb);

  // material pen
  auto *sMPA  = new Setting<bool> ("Active",          "mPenActive", &matPen.active);
  matGroup->add(sMPA);
  auto *sMPR0 = new Setting<VT3>  ("Radius 0",        "mPenRad0",   &matPen.radius0);
  sMPR0->setFormat(VT3{1.0f, 1.0f, 1.0f}, VT3{10.0f, 10.0f, 10.0f}, "%0.4f");
  matGroup->add(sMPR0);
  auto *sMPR1 = new Setting<VT3>  ("Radius 1",        "mPenRad1",   &matPen.radius1);
  sMPR1->setFormat(VT3{0.0f, 0.0f, 0.0f}, VT3{10.0f, 10.0f, 10.0f}, "%1.4f");
  matGroup->add(sMPR1);
  auto *sMPRD = new Setting<VT3>  ("R Offset",        "mPenRDist",  &matPen.rDist);
  sMPRD->setFormat(VT3{1.0f, 1.0f, 1.0f}, VT3{10.0f, 10.0f, 10.0f}, "%1.4f");
  matGroup->add(sMPRD);
  auto *sMSM  = new Setting<T>    ("Size Multiplier", "mPenSizeMult", &matPen.sizeMult);
  sMSM->setFormat(0.1f, 1.0f, "%0.4f");
  matGroup->add(sMSM);
  auto *sMXM   = new Setting<VT3> ("XYZ Multiplier",  "mPenXYZMult", &matPen.xyzMult);
  sMXM->setFormat(VT3{0.1f, 0.1f, 0.1f}, VT3{1.0f, 1.0f, 1.0f}, "%0.4f");
  matGroup->add(sMXM);
  auto *sMPD  = new Setting<int>  ("Depth",           "mPenDepth",  &matPen.depth);
  sMPD->setFormat(1, 8, "");
  matGroup->add(sMPD);
  auto *sMPS  = new Setting<bool> ("Square",          "mPenSquare", &matPen.square);
  matGroup->add(sMPS);
  auto *sMPR  = new Setting<bool> ("Radial",          "mPenRadial", &matPen.radial);
  matGroup->add(sMPR);
  auto *sMPAL = new Setting<bool> ("Align to Cell",   "mPenAlign",  &matPen.cellAlign);
  matGroup->add(sMPAL);
  auto *sMM   = new Setting<T>    ("Multiplier",      "mPenMult",   &matPen.mult);
  sMM->setFormat(0.1f, 1.0f, "%0.4f");
  matGroup->add(sMM);
  
  auto *sMPV  = new Setting<bool> ("Vacuum (eraser)",    "mPenVacuum",  false, false);
  sMPV->updateCallback = [this, sMPV]() { matPen.mat.setVacuum(sMPV->value()); };
  matGroup->add(sMPV);
  auto *MPERM = new Setting<float> ("Permittivity (ε)",  "mPenEpsilon", &matPen.mat.permittivity); 
  MPERM->setFormat(0.1f, 1.0f, "%0.4f");
  matGroup->add(MPERM);
  auto *sMPMT = new Setting<float> ("Permeability (μ)",  "mPenMu",      &matPen.mat.permeability);
  sMPMT->setFormat(0.1f, 1.0f, "%0.4f");
  matGroup->add(sMPMT);
  auto *sMC   = new Setting<float> ("Conductivity (σ)",  "mPenSigma",   &matPen.mat.conductivity);
  sMC->setFormat(0.1f, 1.0f, "%0.4f");
  matGroup->add(sMC);

  mForm = new SettingForm("Draw Settings", 180, 300);
  mForm->add(sigGroup);
  mForm->add(matGroup);
}

template<typename T>
DrawInterface<T>::~DrawInterface()
{
  if(mForm) { delete mForm; mForm = nullptr; }
}


template<typename T>
void DrawInterface<T>::draw(ImFont *superFont)
{
  mForm->draw();
  ImGui::Spacing();
  ImGui::TextUnformatted("Derived:");
  TextPhysics("   n = (εμ/(ε<^(0)μ<^(0)))>^(1/2)", superFont); ImGui::SameLine();
  ImGui::Text(" = %f  (index of refraction)", matPen.mat.n(*mUnits));
}





#endif // DRAW_HPP
