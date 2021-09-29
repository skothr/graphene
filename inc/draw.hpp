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
  std::vector<bool> Vopt  = {false,  false, false, false,  false };
  std::vector<bool> Popt  = {false,  false, false, false,  false };
  std::vector<bool> Qnopt = {false,  false, false, false,  false };
  std::vector<bool> Qpopt = {false,  false, false, false,  false };
  std::vector<bool> Qvopt = {false,  false, false, false,  false };
  std::vector<bool> Eopt  = {false,  false, false, false,  false };
  std::vector<bool> Bopt  = {false,  false, false, false,  false };

  // Pen *activePen = nullptr; // TODO: better pen abstraction
  SignalPen<T>   sigPen;
  MaterialPen<T> matPen;

  Units<T> *mUnits = nullptr;
  
  SettingForm *mForm = nullptr;
  json toJSON() const { return (mForm ? mForm->toJSON() : json::object()); }
  bool fromJSON(const json &js)
  {
    bool success =  (mForm ? mForm->fromJSON(js) : false);
    // update crappy opt interface in case settings are different (TODO: improve)
    int idx = IDX_NONE; for(int i=0; i<Vopt.size();  i++) { idx = (Vopt[i] ?(idx|(1<<i)) : (idx&~(1<<i))); } sigPen.Vopt  = idx;
    idx     = IDX_NONE; for(int i=0; i<Popt.size();  i++) { idx = (Popt[i] ?(idx|(1<<i)) : (idx&~(1<<i))); } sigPen.Popt  = idx;
    idx     = IDX_NONE; for(int i=0; i<Qnopt.size(); i++) { idx = (Qnopt[i]?(idx|(1<<i)) : (idx&~(1<<i))); } sigPen.Qnopt = idx;
    idx     = IDX_NONE; for(int i=0; i<Qpopt.size(); i++) { idx = (Qpopt[i]?(idx|(1<<i)) : (idx&~(1<<i))); } sigPen.Qpopt = idx;
    idx     = IDX_NONE; for(int i=0; i<Qvopt.size(); i++) { idx = (Qvopt[i]?(idx|(1<<i)) : (idx&~(1<<i))); } sigPen.Qvopt = idx;
    idx     = IDX_NONE; for(int i=0; i<Eopt.size();  i++) { idx = (Eopt[i] ?(idx|(1<<i)) : (idx&~(1<<i))); } sigPen.Eopt  = idx;
    idx     = IDX_NONE; for(int i=0; i<Bopt.size();  i++) { idx = (Bopt[i] ?(idx|(1<<i)) : (idx&~(1<<i))); } sigPen.Bopt  = idx;
    return success;
  }
  
  DrawInterface(Units<T> *units);
  ~DrawInterface();

  void draw(ImFont *superFont=nullptr);
  void updateAll() { mForm->updateAll(); }
};


template<typename T>
DrawInterface<T>::DrawInterface(Units<T> *units)
  : mUnits(units)
{
  Vopt  = std::vector<bool>{(bool)(sigPen.Vopt&IDX_R),    (bool)(sigPen.Vopt&IDX_R2),  (bool)(sigPen.Vopt&IDX_T),
                            (bool)(sigPen.Vopt&IDX_SIN),  (bool)(sigPen.Vopt&IDX_COS)};
  Popt  = std::vector<bool>{(bool)(sigPen.Popt&IDX_R),    (bool)(sigPen.Popt&IDX_R2),  (bool)(sigPen.Popt&IDX_T),
                            (bool)(sigPen.Popt&IDX_SIN),  (bool)(sigPen.Popt&IDX_COS)};
  Qnopt = std::vector<bool>{(bool)(sigPen.Qnopt&IDX_R),   (bool)(sigPen.Qnopt&IDX_R2), (bool)(sigPen.Qnopt&IDX_T),
                            (bool)(sigPen.Qnopt&IDX_SIN), (bool)(sigPen.Qnopt&IDX_COS)};
  Qpopt = std::vector<bool>{(bool)(sigPen.Qpopt&IDX_R),   (bool)(sigPen.Qpopt&IDX_R2), (bool)(sigPen.Qpopt&IDX_T),
                            (bool)(sigPen.Qpopt&IDX_SIN), (bool)(sigPen.Qpopt&IDX_COS)};
  Qvopt = std::vector<bool>{(bool)(sigPen.Qvopt&IDX_R),   (bool)(sigPen.Qvopt&IDX_R2), (bool)(sigPen.Qvopt&IDX_T),
                            (bool)(sigPen.Qvopt&IDX_SIN), (bool)(sigPen.Qvopt&IDX_COS)};
  Eopt = std::vector<bool> {(bool)(sigPen.Eopt&IDX_R),    (bool)(sigPen.Eopt&IDX_R2),  (bool)(sigPen.Eopt&IDX_T),
                            (bool)(sigPen.Eopt&IDX_SIN),  (bool)(sigPen.Eopt&IDX_COS)};
  Bopt = std::vector<bool> {(bool)(sigPen.Bopt&IDX_R),    (bool)(sigPen.Bopt&IDX_R2),  (bool)(sigPen.Bopt&IDX_T),
                            (bool)(sigPen.Bopt&IDX_SIN),  (bool)(sigPen.Bopt&IDX_COS)};
  
  typedef typename DimType<T, 3>::VEC_T VT3;
  SettingGroup *sigGroup = new SettingGroup("Add Signal (Ctrl+Click)",    "sigPen", { }, false);
  SettingGroup *matGroup = new SettingGroup("Add Material (Alt+Click)",   "matPen", { }, false);
  
  // signal pen
  auto *sSPA   = new Setting<bool> ("Active",          "sPenActive",   &sigPen.active);
  sigGroup->add(sSPA);
  auto *sSPR0  = new Setting<VT3>  ("Radius 0",        "sPenRad0",     &sigPen.radius0);
  sSPR0->setFormat(VT3{1.0f, 1.0f, 1.0f}, VT3{10.0f, 10.0f, 10.0f}, "%0.4f");
  sigGroup->add(sSPR0);
  auto *sSPR1  = new Setting<VT3>  ("Radius 1",        "sPenRad1",     &sigPen.radius1);
  sSPR1->setFormat(VT3{0.0f, 0.0f, 0.0f}, VT3{10.0f, 10.0f, 10.0f}, "%1.4f");
  sigGroup->add(sSPR1);
  auto *sSPRD  = new Setting<VT3>  ("R Offset",        "sPenRDist",    &sigPen.rDist);
  sSPRD->setFormat(VT3{1.0f, 1.0f, 1.0f}, VT3{10.0f, 10.0f, 10.0f}, "%1.4f");
  sigGroup->add(sSPRD);
  auto *sSSM   = new Setting<T>    ("Size Multiplier", "sPenSizeMult", &sigPen.sizeMult);
  sSSM->setFormat(0.1f, 1.0f, "%0.4f");
  sigGroup->add(sSSM);
  auto *sSXM   = new Setting<VT3>  ("XYZ Multiplier",  "sPenXYZMult",  &sigPen.xyzMult);
  sSXM->setFormat(VT3{0.1f, 0.1f, 0.1f}, VT3{1.0f, 1.0f, 1.0f}, "%0.4f");
  sigGroup->add(sSXM);
  auto *sSPD   = new Setting<int>  ("Depth",           "sPenDepth",    &sigPen.depth);
  sSPD->setFormat(1, 8, "");
  sigGroup->add(sSPD);
  auto *sSPS   = new Setting<bool> ("Square",          "sPenSquare",   &sigPen.square);
  sigGroup->add(sSPS);
  auto *sSPAL  = new Setting<bool> ("Align to Cell",   "sPenAlign",    &sigPen.cellAlign);
  sigGroup->add(sSPAL);
  auto *sSPR   = new Setting<bool> ("Radial",          "sPenRadial",   &sigPen.radial);
  sigGroup->add(sSPR);
  auto *sSPMS  = new Setting<bool> ("Mouse Speed",     "mouseSpeed",   &sigPen.speed);
  sigGroup->add(sSPMS);
  auto *sSPMSM = new Setting<T>    ("Speed Multiplier\n","speedMult",  &sigPen.speedMult);
  sSPMSM->setFormat(0.1f, 1.0f, "%0.4f");
  sigGroup->add(sSPMSM);
  
  auto *sSPM   = new Setting<T>    ("Amplitude",       "sPenMult",     &sigPen.mult);
  sSPM->setFormat(0.1f, 1.0f, "%0.4f");
  sigGroup->add(sSPM);
  auto *sSPF   = new Setting<T>    ("Frequency",       "sPenFreq",     &sigPen.frequency, sigPen.frequency,
                                    [this]() { sigPen.wavelength = mUnits->c() / sigPen.frequency; });
  sSPF->setFormat(0.01f, 0.1f, "%0.4f"); sSPF->setMin(0.0f);
  sigGroup->add(sSPF);
  auto *sSPW   = new Setting<T>    ("Wavelength",      "sPenWLen",     &sigPen.wavelength, sigPen.wavelength,
                                    [this]() { sigPen.frequency = mUnits->c() / sigPen.wavelength; });
  sSPW->setFormat(0.1f, 1.0f, "%0.4f"); sSPW->setMin(0.0f);
  sigGroup->add(sSPW);
  
  auto *sSPV = new Setting<float3>("V",  "sPenV",  &sigPen.Vmult);
  sSPV->setFormat(float3{0.01f, 0.01f, 0.01f}, float3{0.1f, 0.1f, 0.1f}, "%0.4f");
  sigGroup->add(sSPV);
  auto *sSPP  = new Setting<float>("P",  "sPenP",  &sigPen.Pmult);
  sSPP->setFormat(0.01f, 0.1f, "%0.4f");
  sigGroup->add(sSPP);
  auto *sSPQn = new Setting<float>("Q-", "sPenQn", &sigPen.Qnmult);
  sSPQn->setFormat(0.01f, 0.1f, "%0.4f");
  sigGroup->add(sSPQn);
  auto *sSPQp = new Setting<float>("Q+", "sPenQp", &sigPen.Qpmult);
  sSPQp->setFormat(0.01f, 0.1f, "%0.4f");
  sigGroup->add(sSPQp);
  auto *sSPQv = new Setting<float3>("Qv", "sPenQv", &sigPen.Qvmult);
  sSPQv->setFormat(float3{0.01f, 0.01f, 0.01f}, float3{0.1f, 0.1f, 0.1f}, "%0.4f");
  sigGroup->add(sSPQv);
  auto *sSPE = new Setting<float3>("E", "sPenE", &sigPen.Emult);
  sSPE->setFormat(float3{0.01f, 0.01f, 0.01f}, float3{0.1f, 0.1f, 0.1f}, "%0.4f");
  sigGroup->add(sSPE);
  auto *sSPB = new Setting<float3>("B", "sPenB", &sigPen.Bmult);
  sSPB->setFormat(float3{0.01f, 0.01f, 0.01f}, float3{0.1f, 0.1f, 0.1f}, "%0.4f");
  sigGroup->add(sSPB);
  
  auto *sSPVb = new Setting<std::vector<bool>>("", "sPenVopt", &Vopt);
  sSPVb->vColumns = 5;  sSPVb->vRowLabels = {{0, "V  "}};
  sSPVb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}}; sSPVb->drawColLabels = true;
  sSPVb->updateCallback =
    [this]() { int idx = IDX_NONE; for(int i = 0; i < Vopt.size(); i++) { idx = (Vopt[i] ? (idx|(1<<i)) : (idx&~(1<<i))); } sigPen.Vopt = idx; };
  sigGroup->add(sSPVb);
  auto *sSPPb = new Setting<std::vector<bool>>("", "sPenPopt", &Popt);
  sSPPb->vColumns = 5;  sSPPb->vRowLabels = {{0, "P  "}};
  sSPPb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}};
  sSPPb->updateCallback =
    [this]() { int idx = IDX_NONE; for(int i = 0; i < Popt.size(); i++) { idx = (Popt[i] ? (idx|(1<<i)) : (idx&~(1<<i))); } sigPen.Popt = idx; };
  sigGroup->add(sSPPb);
  
  auto *sSPQNb = new Setting<std::vector<bool>>("", "sPenQNopt", &Qnopt);
  sSPQNb->vColumns = 5;  sSPQNb->vRowLabels = {{0, "Q- "}};
  sSPQNb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}};
  sSPQNb->updateCallback =
    [this]() { int idx = IDX_NONE; for(int i = 0; i < Qnopt.size(); i++) { idx = (Qnopt[i] ? (idx|(1<<i)) : (idx&~(1<<i))); } sigPen.Qnopt = idx; };
  sigGroup->add(sSPQNb);
  auto *sSPQPb = new Setting<std::vector<bool>>("", "sPenQPopt", &Qpopt);
  sSPQPb->vColumns = 5;  sSPQPb->vRowLabels = {{0, "Q+ "}};
  sSPQPb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}};
  sSPQPb->updateCallback =
    [this]() { int idx = IDX_NONE; for(int i = 0; i < Qpopt.size(); i++) { idx = (Qpopt[i] ? (idx|(1<<i)) : (idx&~(1<<i))); } sigPen.Qpopt = idx; };
  sigGroup->add(sSPQPb);
  auto *sSPQVb = new Setting<std::vector<bool>>("", "sPenQVopt", &Qvopt);
  sSPQVb->vColumns = 5;  sSPQVb->vRowLabels = {{0, "QV "}};
  sSPQVb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}};
  sSPQVb->updateCallback =
    [this]() { int idx = IDX_NONE; for(int i = 0; i < Qvopt.size(); i++) { idx = (Qvopt[i] ? (idx|(1<<i)) : (idx&~(1<<i))); } sigPen.Qvopt = idx; };
  sigGroup->add(sSPQVb);
  
  auto *sSPEb = new Setting<std::vector<bool>>("", "sPenEopt", &Eopt);
  sSPEb->vColumns = 5;  sSPEb->vRowLabels = {{0, "E  "}};
  sSPEb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}};
  sSPEb->updateCallback =
    [this]() { int idx = IDX_NONE; for(int i = 0; i < Eopt.size(); i++) { idx = (Eopt[i] ? (idx|(1<<i)) : (idx&~(1<<i))); } sigPen.Eopt = idx; };
  sigGroup->add(sSPEb);
  auto *sSPBb = new Setting<std::vector<bool>>("", "sPenBopt", &Bopt);
  sSPBb->vColumns = 5;  sSPBb->vRowLabels = {{0, "B  "}};
  sSPBb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}}; 
  sSPBb->updateCallback =
    [this]() { int idx = IDX_NONE; for(int i = 0; i < Bopt.size(); i++) { idx = (Bopt[i] ? (idx|(1<<i)) : (idx&~(1<<i))); } sigPen.Bopt = idx; };
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
  sMPV->updateCallback = [this, sMPV]() { matPen.material.setVacuum(sMPV->value()); };
  matGroup->add(sMPV);
  auto *MPERM = new Setting<float> ("Permittivity (ε)",  "mPenEpsilon", &matPen.material.permittivity); 
  MPERM->setFormat(0.1f, 1.0f, "%0.4f");
  matGroup->add(MPERM);
  auto *sMPMT = new Setting<float> ("Permeability (μ)",  "mPenMu",      &matPen.material.permeability);
  sMPMT->setFormat(0.1f, 1.0f, "%0.4f");
  matGroup->add(sMPMT);
  auto *sMC   = new Setting<float> ("Conductivity (σ)",  "mPenSigma",   &matPen.material.conductivity);
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
  ImGui::Text(" = %f  (index of refraction)", matPen.material.n(*mUnits));
}





#endif // DRAW_HPP
