#ifndef DRAW_HPP
#define DRAW_HPP

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
  // TODO: clean up this mess
  //                          1/R    1/R^2  theta  sin(t)  cos(t)
  std::vector<bool> Qopt   = {false, false, false, false,  false};
  std::vector<bool> QPVopt = {false, false, false, false,  false};
  std::vector<bool> QNVopt = {false, false, false, false,  false};
  std::vector<bool> Eopt   = {false, false, false, true,   false};
  std::vector<bool> Bopt   = {false, false, false, false,  true };

  // Pen *activePen = nullptr; // TODO: better pen abstraction
  SignalPen<T>   sigPen;
  MaterialPen<T> matPen;

  Units<T> *mUnits = nullptr;
  
  std::vector<SettingBase*> mSettings;
  SettingForm *mForm = nullptr;
  DrawInterface(Units<T> *units);
  ~DrawInterface();

  void draw(ImFont *superFont=nullptr);
};


template<typename T>
DrawInterface<T>::DrawInterface(Units<T> *units)
  : mUnits(units)
{
  typedef typename DimType<T, 3>::VECTOR_T VT3;
  SettingGroup *sigGroup = new SettingGroup("Add Signal (Ctrl+Click)",    "sigPen", { }, false);
  SettingGroup *matGroup = new SettingGroup("Add Material (Alt+Click)",   "matPen", { }, false);
  
  // signal pen
  auto *sSPA   = new Setting<bool>  ("Active",        "sPenActive", &sigPen.active);
  mSettings.push_back(sSPA);   sigGroup->add(sSPA);
  auto *sSPR0 = new Setting<VT3>  ("Radius 0",        "sPenRad0",   &sigPen.radius0);
  sSPR0->setFormat(VT3{1.0f, 1.0f, 1.0f}, VT3{10.0f, 10.0f, 10.0f}, "%0.4f");
  mSettings.push_back(sSPR0);  sigGroup->add(sSPR0);
  auto *sSPR1 = new Setting<VT3>  ("Radius 1",        "sPenRad1",   &sigPen.radius1);
  sSPR1->setFormat(VT3{0.0f, 0.0f, 0.0f}, VT3{10.0f, 10.0f, 10.0f}, "%1.4f");
  mSettings.push_back(sSPR1);  sigGroup->add(sSPR1);
  auto *sSPRD = new Setting<VT3>  ("R Offset",        "sPenRDist",  &sigPen.rDist);
  sSPRD->setFormat(VT3{1.0f, 1.0f, 1.0f}, VT3{10.0f, 10.0f, 10.0f}, "%1.4f");
  mSettings.push_back(sSPRD);  sigGroup->add(sSPRD);
  auto *sSPD   = new Setting<int>   ("Depth",         "sPenDepth",  &sigPen.depth);
  sSPD->setFormat(1, 8, "");
  mSettings.push_back(sSPD); sigGroup->add(sSPD);
  auto *sSPS   = new Setting<bool>  ("Square",        "sPenSquare", &sigPen.square);
  mSettings.push_back(sSPS); sigGroup->add(sSPS);
  auto *sSPAL  = new Setting<bool>  ("Align to Cell", "sPenAlign",  &sigPen.cellAlign);
  mSettings.push_back(sSPAL); sigGroup->add(sSPAL);
  auto *sSPM   = new Setting<T>     ("Amplitude",     "sPenMult",   &sigPen.mult);
  sSPM->setFormat(0.1f, 1.0f, "%0.4f");
  mSettings.push_back(sSPM); sigGroup->add(sSPM);
  auto *sSPF   = new Setting<T>     ("Frequency",     "sPenFreq",   &sigPen.frequency, sigPen.frequency,
                                     [this]() { sigPen.wavelength = mUnits->c() / sigPen.frequency; });
  sSPF->setFormat(0.01f, 0.1f, "%0.4f"); sSPF->setMin(0.0f);
  mSettings.push_back(sSPF); sigGroup->add(sSPF);
  auto *sSPW   = new Setting<T>     ("Wavelength",    "sPenWLen",   &sigPen.wavelength, sigPen.wavelength,
                                     [this]() { sigPen.frequency = mUnits->c() / sigPen.wavelength; });
  sSPW->setFormat(0.1f, 1.0f, "%0.4f"); sSPW->setMin(0.0f);
  mSettings.push_back(sSPW); sigGroup->add(sSPW);
  
  auto *sSPQ   = new Setting<float2>("Q",   "sPenQ",   &sigPen.Qmult);
  sSPQ->labels = {"(+)", "(-)"};
  sSPQ->setFormat  (float2{0.01f, 0.01f}, float2{0.1f, 0.1f}, "%0.4f");
  auto *sSPQPV = new Setting<float3>("VQ+", "sPenQpv", &sigPen.QPVmult);
  sSPQPV->setFormat(float3{0.01f, 0.01f, 0.01f},  float3{0.1f, 0.1f, 0.1f}, "%0.4f");
  auto *sSPQNV = new Setting<float3>("VQ-", "sPenQnv", &sigPen.QNVmult);
  sSPQNV->setFormat(float3{0.01f, 0.01f, 0.01f},  float3{0.1f, 0.1f, 0.1f}, "%0.4f");
  auto *sSPE   = new Setting<float3>("E",   "sPenE",   &sigPen.Emult);
  sSPE->setFormat  (float3{0.01f, 0.01f, 0.01f}, float3{0.1f, 0.1f, 0.1f}, "%0.4f");
  auto *sSPB   = new Setting<float3>("B",   "sPenB",   &sigPen.Bmult);
  sSPB->setFormat  (float3{0.01f, 0.01f, 0.01f}, float3{0.1f, 0.1f, 0.1f}, "%0.4f");
  mSettings.push_back(sSPQ);   sigGroup->add(sSPQ);
  mSettings.push_back(sSPQPV); sigGroup->add(sSPQPV);
  mSettings.push_back(sSPQNV); sigGroup->add(sSPQNV);
  mSettings.push_back(sSPE);   sigGroup->add(sSPE);
  mSettings.push_back(sSPB);   sigGroup->add(sSPB);
  
  auto *sSPQb    = new Setting<std::vector<bool>>("Options", "sPenQopt", &Qopt);
  sSPQb->vColumns = 5; sSPQb->vRowLabels = {{0, "Q  "}};
  sSPQb->vColLabels   = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}}; sSPQb->drawColLabels = true;
  sSPQb->vRowLabels   = {{0, "Q  "}};
  sSPQb->updateCallback   =
    [this]() { int idx = IDX_NONE; for(int i=0; i<Qopt.size(); i++)   { idx = (Qopt[i]  ?(idx|(1<<i)) : (idx&~(1<<i))); } sigPen.Qopt   = idx; };
  auto *sSPQPVb  = new Setting<std::vector<bool>>("", "sPenQPVopt", &QPVopt);
  sSPQPVb->vColumns = 5;  sSPQPVb->vRowLabels = {{0, "QPV"}};
  sSPQPVb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}}; 
  sSPQPVb->updateCallback =
    [this]() { int idx = IDX_NONE; for(int i=0; i<QPVopt.size(); i++) { idx = (QPVopt[i]?(idx|(1<<i)) : (idx&~(1<<i))); } sigPen.QPVopt = idx; };
  auto *sSPQNVb  = new Setting<std::vector<bool>>("", "sPenQNVopt", &QNVopt);
  sSPQNVb->vColumns = 5;  sSPQNVb->vRowLabels = {{0, "QNV"}};
  sSPQNVb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}}; 
  sSPQNVb->updateCallback =
    [this]() { int idx = IDX_NONE; for(int i=0; i<QNVopt.size(); i++) { idx = (QNVopt[i]?(idx|(1<<i)) : (idx&~(1<<i))); } sigPen.QNVopt = idx; };
  auto *sSPEb    = new Setting<std::vector<bool>>("", "sPenEopt", &Eopt);
  sSPEb->vColumns = 5;  sSPEb->vRowLabels = {{0, "E  "}};
  sSPEb->vColLabels   = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}}; 
  sSPEb->updateCallback   =
    [this]() { int idx = IDX_NONE; for(int i=0; i<Eopt.size(); i++)   { idx = (Eopt[i]  ?(idx|(1<<i)) : (idx&~(1<<i))); } sigPen.Eopt   = idx; };
  auto *sSPBb    = new Setting<std::vector<bool>>("", "sPenBopt", &Bopt);
  sSPBb->vColumns = 5;  sSPBb->vRowLabels = {{0, "B  "}};
  sSPBb->vColLabels   = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}}; 
  sSPBb->updateCallback   =
    [this]() { int idx = IDX_NONE; for(int i=0; i<Bopt.size(); i++)   { idx = (Bopt[i]  ?(idx|(1<<i)) : (idx&~(1<<i))); } sigPen.Bopt   = idx; };
  mSettings.push_back(sSPQb);   sigGroup->add(sSPQb);
  mSettings.push_back(sSPQPVb); sigGroup->add(sSPQPVb);
  mSettings.push_back(sSPQNVb); sigGroup->add(sSPQNVb);
  mSettings.push_back(sSPEb);   sigGroup->add(sSPEb);
  mSettings.push_back(sSPBb);   sigGroup->add(sSPBb);

  // material pen
  matPen.material.setVacuum(false);
  auto *sMPA  = new Setting<bool> ("Active",        "mPenActive", &matPen.active);
  mSettings.push_back(sMPA);  matGroup->add(sMPA);
  auto *sMPR0 = new Setting<VT3>  ("Radius 0",      "mPenRad0",   &matPen.radius0);
  sMPR0->setFormat(VT3{1.0f, 1.0f, 1.0f}, VT3{10.0f, 10.0f, 10.0f}, "%0.4f");
  mSettings.push_back(sMPR0);  matGroup->add(sMPR0);
  auto *sMPR1 = new Setting<VT3>  ("Radius 1",      "mPenRad1",   &matPen.radius1);
  sMPR1->setFormat(VT3{0.0f, 0.0f, 0.0f}, VT3{10.0f, 10.0f, 10.0f}, "%1.4f");
  mSettings.push_back(sMPR1);  matGroup->add(sMPR1);
  auto *sMPRD = new Setting<VT3>  ("R Offset",      "mPenRDist",  &matPen.rDist);
  sMPRD->setFormat(VT3{1.0f, 1.0f, 1.0f}, VT3{10.0f, 10.0f, 10.0f}, "%1.4f");
  mSettings.push_back(sMPRD);  matGroup->add(sMPRD);
  auto *sMPD  = new Setting<int>  ("Depth",         "mPenDepth",  &matPen.depth);
  sMPD->setFormat(1, 8, "");
  mSettings.push_back(sMPD);  matGroup->add(sMPD);
  auto *sMPS  = new Setting<bool> ("Square",        "mPenSquare", &matPen.square);
  mSettings.push_back(sMPS);  matGroup->add(sMPS);
  auto *sMPAL = new Setting<bool> ("Align to Cell", "mPenAlign",  &matPen.cellAlign);
  mSettings.push_back(sMPAL); matGroup->add(sMPAL);
  auto *sMM   = new Setting<T>    ("Multiplier",    "mPenMult",   &matPen.mult);
  sMM->setFormat(0.1f, 1.0f, "%0.4f");
  mSettings.push_back(sMM);   matGroup->add(sMM);
  
  auto *sMPV  = new Setting<bool> ("Vacuum (eraser)",    "mPenVacuum",  false, false);
  sMPV->updateCallback = [this, sMPV]() { matPen.material.setVacuum(sMPV->value()); };
  mSettings.push_back(sMPV);  matGroup->add(sMPV);
  auto *MPERM = new Setting<float> ("Permittivity (ε)",  "mPenEpsilon", &matPen.material.permittivity); 
  MPERM->setFormat(0.1f, 1.0f, "%0.4f");
  mSettings.push_back(MPERM); matGroup->add(MPERM);
  auto *sMPMT = new Setting<float> ("Permeability (μ)",  "mPenMu",      &matPen.material.permeability);
  sMPMT->setFormat(0.1f, 1.0f, "%0.4f");
  mSettings.push_back(sMPMT); matGroup->add(sMPMT);
  auto *sMC   = new Setting<float> ("Conductivity (σ)",  "mPenSigma",   &matPen.material.conductivity);
  sMC->setFormat(0.1f, 1.0f, "%0.4f");
  mSettings.push_back(sMC);   matGroup->add(sMC);

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
  ImGui::Text(" = %f  (index of refraction)", matPen.material.n(mUnits->e0, mUnits->u0));
}





#endif // DRAW_HPP
