#ifndef DRAW_HPP
#define DRAW_HPP

#include "vector-operators.h"
#include "material.h"
#include "units.cuh"
#include "draw.cuh"
#include "setting.hpp"
#include "settingForm.hpp"



template<typename T>
struct DrawInterface : public SettingForm
{
  SignalPen<T>   *sigPen     = nullptr;
  MaterialPen<T> *matPen     = nullptr;
  SignalPen<T>   *newSigPen  = nullptr; // (buffer for UI thread safety)
  MaterialPen<T> *newMatPen  = nullptr;
  
  Units<T>       *mUnits     = nullptr;
  ImFont         *mSuperFont = nullptr;
  DrawInterface(Units<T> *units, ImFont *superFont=nullptr);
  ~DrawInterface() = default;

  void setSigPen(SignalPen<T> *sPen);
  void setMatPen(MaterialPen<T> *mPen);
  void makeSettings();

  virtual bool draw() override;
};

template<typename T>
DrawInterface<T>::DrawInterface(Units<T> *units, ImFont *superFont)
  : mUnits(units), mSuperFont(superFont)
{  }

template<typename T> void DrawInterface<T>::setSigPen(SignalPen<T>   *sPen) { newSigPen = sPen; }
template<typename T> void DrawInterface<T>::setMatPen(MaterialPen<T> *mPen) { newMatPen = mPen; }

template<typename T>
void DrawInterface<T>::makeSettings()
{
  typedef typename DimType<T, 3>::VEC_T VT3;
  
  SettingForm::cleanup(); // destroy old settings
  if(sigPen)
    {
      std::cout << "====== (DrawUI) --> Creating signal pen settings...\n";
      
      // signal pen
      SettingGroup *sigGroup = new SettingGroup("Add Signal (Ctrl+Click)",  "sigPen", { }); add(sigGroup);
  
      auto *sSPA   = new Setting<bool> ("Active",          "sigPenActive",   &sigPen->active);    sigGroup->add(sSPA);
      auto *sSPR0  = new Setting<VT3>  ("Radius 0",        "sigPenRad0",     &sigPen->radius0);   sigGroup->add(sSPR0);
      sSPR0->setFormat(VT3{1.0f, 1.0f, 1.0f}, VT3{10.0f, 10.0f, 10.0f}, "%0.4f");
      auto *sSPR1  = new Setting<VT3>  ("Radius 1",        "sigPenRad1",     &sigPen->radius1);   sigGroup->add(sSPR1);
      sSPR1->setFormat(VT3{1.0f, 1.0f, 1.0f}, VT3{10.0f, 10.0f, 10.0f}, "%1.4f");
      auto *sSPRD  = new Setting<VT3>  ("R Offset",        "sigPenRDist",    &sigPen->rDist);     sigGroup->add(sSPRD);
      sSPRD->setFormat(VT3{1.0f, 1.0f, 1.0f}, VT3{10.0f, 10.0f, 10.0f}, "%1.4f");
      auto *sSSM   = new Setting<T>    ("Size Multiplier", "sigPenSizeMult", &sigPen->sizeMult);  sigGroup->add(sSSM);
      sSSM->setFormat(0.1f, 1.0f, "%0.4f");
      auto *sSXM   = new Setting<VT3>  ("XYZ Multiplier",  "sigPenXYZMult",  &sigPen->xyzMult);   sigGroup->add(sSXM);
      sSXM->setFormat(VT3{0.1f, 0.1f, 0.1f}, VT3{1.0f, 1.0f, 1.0f}, "%0.4f");
      auto *sSPD   = new Setting<int>  ("Depth",           "sigPenDepth",    &sigPen->depth);     sigGroup->add(sSPD);
      sSPD->setFormat(1, 8, "");
      auto *sSPS   = new Setting<bool> ("Square",          "sigPenSquare",   &sigPen->square);    sigGroup->add(sSPS);
      auto *sSPAL  = new Setting<bool> ("Align to Cell",   "sigPenAlign",    &sigPen->cellAlign); sigGroup->add(sSPAL);
      auto *sSPR   = new Setting<bool> ("Radial",          "sigPenRadial",   &sigPen->radial);    sigGroup->add(sSPR);
      auto *sSPMS  = new Setting<bool> ("Mouse Speed",     "mouseSpeed",     &sigPen->speed);     sigGroup->add(sSPMS);
      auto *sSPMSM = new Setting<T>    ("Speed Multiplier\n","speedMult",    &sigPen->speedMult); sigGroup->add(sSPMSM);
      sSPMSM->setFormat(0.1f, 1.0f, "%0.4f");
  
      auto *sSPM   = new Setting<T>    ("Amplitude",       "sigPenMult",     &sigPen->mult);      sigGroup->add(sSPM);
      sSPM->setFormat(0.1f, 1.0f, "%0.4f");
      auto *sSPF   = new Setting<T>    ("Frequency",       "sigPenFreq",     &sigPen->frequency); sigGroup->add(sSPF);
      sSPF->setUpdateCallback([this](){ sigPen->wavelength = mUnits->c() / sigPen->frequency; });
      sSPF->setFormat(0.01f, 0.1f, "%0.4f"); sSPF->setMin(0.0f);
      auto *sSPW   = new Setting<T>    ("Wavelength",      "sigPenWV",     &sigPen->wavelength);  sigGroup->add(sSPW);
      sSPW->setUpdateCallback([this]() { sigPen->frequency = mUnits->c() / sigPen->wavelength; });
      sSPW->setFormat(0.1f, 1.0f, "%0.4f"); sSPW->setMin(0.0f);
  
      auto *sSPV  = new Setting<float3>("V",  "sigPenVBase",  &sigPen->pV.base);  sigGroup->add(sSPV);
      sSPV->setFormat(float3{0.01f, 0.01f, 0.01f}, float3{0.1f, 0.1f, 0.1f}, "%0.4f");
      auto *sSPP  = new Setting<float> ("P",  "sigPenPBase",  &sigPen->pP.base);  sigGroup->add(sSPP);
      sSPP->setFormat(0.01f, 0.1f, "%0.4f");
      auto *sSPQn = new Setting<float> ("Q-", "sigPenQnBase", &sigPen->pQn.base); sigGroup->add(sSPQn);
      sSPQn->setFormat(0.01f, 0.1f, "%0.4f");
      auto *sSPQp = new Setting<float> ("Q+", "sigPenQpBase", &sigPen->pQp.base); sigGroup->add(sSPQp);
      sSPQp->setFormat(0.01f, 0.1f, "%0.4f");
      auto *sSPQnv = new Setting<float3>("Qnv", "sigPenQnvBase", &sigPen->pQnv.base); sigGroup->add(sSPQnv);
      sSPQnv->setFormat(float3{0.01f, 0.01f, 0.01f}, float3{0.1f, 0.1f, 0.1f}, "%0.4f");
      auto *sSPQpv = new Setting<float3>("Qpv", "sigPenQpvBase", &sigPen->pQpv.base); sigGroup->add(sSPQpv);
      sSPQpv->setFormat(float3{0.01f, 0.01f, 0.01f}, float3{0.1f, 0.1f, 0.1f}, "%0.4f");
      auto *sSPE  = new Setting<float3>("E",  "sigPenEBase",  &sigPen->pE.base);    sigGroup->add(sSPE);
      sSPE->setFormat( float3{0.01f, 0.01f, 0.01f}, float3{0.1f, 0.1f, 0.1f}, "%0.4f");
      auto *sSPB  = new Setting<float3>("B",  "sigPenBBase",  &sigPen->pB.base);    sigGroup->add(sSPB);
      sSPB->setFormat( float3{0.01f, 0.01f, 0.01f}, float3{0.1f, 0.1f, 0.1f}, "%0.4f");
  
      auto sSPMods = makeSettingGrid<bool, 5>("Modifiers", "spMods",
                                              { &sigPen->pV.modArr,   &sigPen->pP.modArr,   &sigPen->pQn.modArr, &sigPen->pQp.modArr,
                                                &sigPen->pQnv.modArr, &sigPen->pQpv.modArr, &sigPen->pE.modArr,  &sigPen->pB.modArr },
                                              {"V", "P", "Q-", "Q+", "Qv-", "Qv+", "E", "B"}); sigGroup->add(sSPMods);
      sSPMods->setCenter(true);
      sSPMods->setColumnLabels(std::vector<std::string>{"R", "R^2", "θ", "sin(t)", "cos(t)"});
    }
  if(matPen)
    {
      std::cout << "====== (DrawUI) --> Creating material pen settings...\n";
      // material pen
      SettingGroup *matGroup = new SettingGroup("Add Material (Alt+Click)", "matPen", { }); add(matGroup);
  
      auto *sMPA  = new Setting<bool> ("Active",          "mPenActive",   &matPen->active);    matGroup->add(sMPA);
      auto *sMPR0 = new Setting<VT3>  ("Radius 0",        "mPenRad0",     &matPen->radius0);   matGroup->add(sMPR0);
      sMPR0->setFormat(VT3{1.0f, 1.0f, 1.0f}, VT3{10.0f, 10.0f, 10.0f}, "%0.4f");
      auto *sMPR1 = new Setting<VT3>  ("Radius 1",        "mPenRad1",     &matPen->radius1);   matGroup->add(sMPR1);
      sMPR1->setFormat(VT3{1.0f, 1.0f, 1.0f}, VT3{10.0f, 10.0f, 10.0f}, "%1.4f");
      auto *sMPRD = new Setting<VT3>  ("R Offset",        "mPenRDist",    &matPen->rDist);     matGroup->add(sMPRD);
      sMPRD->setFormat(VT3{1.0f, 1.0f, 1.0f}, VT3{10.0f, 10.0f, 10.0f}, "%1.4f");
      auto *sMSM  = new Setting<T>    ("Size Multiplier", "mPenSizeMult", &matPen->sizeMult);  matGroup->add(sMSM);
      sMSM->setFormat(0.1f, 1.0f, "%0.4f");
      auto *sMXM   = new Setting<VT3> ("XYZ Multiplier",  "mPenXYZMult",  &matPen->xyzMult);   matGroup->add(sMXM);
      sMXM->setFormat(VT3{0.1,0.1,0.1}, VT3{1.0,1.0,1.0}, "%0.4f");
      auto *sMPD  = new Setting<int>  ("Depth",           "mPenDepth",    &matPen->depth);     matGroup->add(sMPD);
      sMPD->setFormat(1, 8, "");
      auto *sMPS  = new Setting<bool> ("Square",          "mPenSquare",   &matPen->square);    matGroup->add(sMPS);
      auto *sMPR  = new Setting<bool> ("Radial",          "mPenRadial",   &matPen->radial);    matGroup->add(sMPR);
      auto *sMPAL = new Setting<bool> ("Align to Cell",   "mPenAlign",    &matPen->cellAlign); matGroup->add(sMPAL);
      auto *sMM   = new Setting<T>    ("Multiplier",      "mPenMult",     &matPen->mult);      matGroup->add(sMM);
      sMM->setFormat(0.1f, 1.0f, "%0.4f");
      auto *sMPV  = new Setting<bool> ("Vacuum (eraser)",  "mPenVacuum",  &matPen->vacuumErase);      matGroup->add(sMPV);
      sMPV->setUpdateCallback([this, sMPV]() { matPen->mat.setVacuum(matPen->vacuumErase); });
      auto *MPERM = new Setting<float>("Permittivity (ε)", "mPenEpsilon", &matPen->mat.permittivity); matGroup->add(MPERM);
      MPERM->setFormat(0.1f, 1.0f, "%0.4f");
      auto *sMPMT = new Setting<float>("Permeability (μ)", "mPenMu",      &matPen->mat.permeability); matGroup->add(sMPMT);
      sMPMT->setFormat(0.1f, 1.0f, "%0.4f");
      auto *sMC   = new Setting<float>("Conductivity (σ)", "mPenSigma",   &matPen->mat.conductivity); matGroup->add(sMC);
      sMC->setFormat(0.1f, 1.0f, "%0.4f");
    }
}

template<typename T>
bool DrawInterface<T>::draw()
{
  bool changed = false;

  // update interface if pen(s) changed
  if(newSigPen) { sigPen = newSigPen; newSigPen = nullptr; changed = true; }
  if(newMatPen) { matPen = newMatPen; newMatPen = nullptr; changed = true; }
  if(changed) { makeSettings(); }
  
  changed |= SettingForm::draw();
  ImGui::Spacing();
  ImGui::TextUnformatted("Derived:");
  TextPhysics("   n = (εμ/(ε<^(0)μ<^(0)))>^(1/2)", mSuperFont);
  ImGui::SameLine();
  ImGui::Text(" = %f  (index of refraction)", matPen->mat.n(*mUnits));
  return changed;
}




#endif // DRAW_HPP
