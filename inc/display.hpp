#ifndef DISPLAY_HPP
#define DISPLAY_HPP

#include "render.cuh"
#include "setting.hpp"
#include "settingForm.hpp"


// parameters for displaying underlying vectors over a field
template<typename T>
struct VectorFieldParams
{
  // vector field
  bool  drawVectors     = false; // draws vector field on screen
  bool  borderedVectors = true;  // uses fancy bordered polygons instead of standard GL_LINES (NOTE: slower)   TODO: optimize with VBO
  bool  smoothVectors   = true;  // uses bilinear interpolation, centering at samples mouse instead of exact cell centers
  int   vecMRadius      = 64;    // draws vectors for cells within radius around mouse
  int   vecSpacing      = 1;     // base spacing 
  int   vecCRadius      = 1024;  // only draws maximum of this radius number of vectors, adding spacing  
  float vecMultE        = 10.0f; // E length multiplier
  float vecMultB        = 10.0f; // B length multiplier
  float vecLineW        = 0.1f;  // line width
  float vecAlpha        = 0.25f; // line opacity
  float vecBorderW      = 0.0f;  // border width
  float vecBAlpha       = 0.0f;  // border opacity
};

template<typename T>
struct DisplayInterface
{
  // flags
  bool showEMField    = true;
  bool showMatField   = false;
  bool show3DField    = true;
  bool drawAxes       = true;  // 3D axes
  bool drawOutline    = true;  // 3D outline of field
  bool vsync          = false; // vertical sync refresh
  
  // main parameters
  RenderParams<T>      *rp = nullptr; bool rpDelete = false;
  VectorFieldParams<T> *vp = nullptr; bool vpDelete = false;
  int *zSize = nullptr;
  
  std::vector<SettingBase*> mSettings;
  SettingForm *mForm = nullptr;
  DisplayInterface(RenderParams<T> *rParams, VectorFieldParams<T> *vParams, int *zs);
  ~DisplayInterface();

  void draw();
};


template<typename T>
DisplayInterface<T>::DisplayInterface(RenderParams<T> *rParams, VectorFieldParams<T> *vParams, int *zs)
  : rp(rParams), vp(vParams), zSize(zs)
{
  if(!rp) { rp = new RenderParams<T>();      rpDelete = true; }
  if(!vp) { vp = new VectorFieldParams<T>(); vpDelete = true; }
  
  mForm = new SettingForm("Display Settings", 180, 300);


  // flags
  auto *sSFEM  = new Setting<bool> ("Show EM Field",       "showEMField",  &showEMField);
  mSettings.push_back(sSFEM); mForm->add(sSFEM);
  auto *sSFMAT  = new Setting<bool>("Show Material Field", "showMatField", &showMatField);
  mSettings.push_back(sSFMAT); mForm->add(sSFMAT);
  auto *sSF3D  = new Setting<bool> ("Show 3D Field",       "show3DField",  &show3DField);
  mSettings.push_back(sSF3D); mForm->add(sSF3D);
  auto *sSFA   = new Setting<bool> ("Show Axes",           "showAxes",     &drawAxes);
  mSettings.push_back(sSFA); mForm->add(sSFA);
  auto *sSFO   = new Setting<bool> ("Show Field Outline",  "showOutline",  &drawOutline);
  mSettings.push_back(sSFO); mForm->add(sSFO);
  auto *sVS    = new Setting<bool> ("vSync",               "vsync",        &vsync, vsync, [&]() { glfwSwapInterval(vsync ? 1 : 0); });
  mSettings.push_back(sVS);  mForm->add(sVS);

  // render params
  auto *sRZL = new Setting<int2> ("Z Range", "zRange", &rp->zRange, rp->zRange,
                                 [&]()
                                 {
                                   if(zSize)
                                     {
                                       if(rp->zRange.y >= *zSize)      { rp->zRange.y = *zSize-1; }
                                       if(rp->zRange.x > rp->zRange.y) { rp->zRange.x = rp->zRange.y; }
                                     }
                                 });
  sRZL->setMin(int2{0,0});
  mSettings.push_back(sRZL); mForm->add(sRZL);
  
  auto *sRCO  = new Setting<float> ("3D Opacity",    "opacity",    &rp->opacity);
  sRCO->setFormat(0.01f, 0.1f, "%0.4f");
  mSettings.push_back(sRCO); mForm->add(sRCO);
  auto *sRCBR = new Setting<float> ("3D Brightness", "brightness", &rp->brightness);
  sRCBR->setFormat(0.01f, 0.1f, "%0.4f");
  mSettings.push_back(sRCBR); mForm->add(sRCBR);
  auto *sRCQ  = new Setting<float4>("Q Color",       "renderColQ", &rp->Qcol);
  sRCQ->drawCustom = [sRCQ, this](bool busy, bool &changed) -> bool
                     {
                       sRCQ->onDraw(1.0f, busy, changed, true); // color picker
                       ImGui::SameLine(); ImGui::SetNextItemWidth(150);
                       changed |= ImGui::InputFloat("##qMult", &rp->Qmult, 0.01f, 0.1f, "%.8f");
                       return busy;
                     };
  sRCQ->setFormat(float4{0.01f, 0.01f, 0.01f, 0.01f}, float4{0.1f, 0.1f, 0.1f, 0.1f}, "%0.8f");  
  mSettings.push_back(sRCQ); mForm->add(sRCQ);
  auto *sRCE = new Setting<float4> ("E Color", "renderColE", &rp->Ecol);
  sRCE->drawCustom = [sRCE, this](bool busy, bool &changed) -> bool
                     {
                       sRCE->onDraw(1.0f, busy, changed, true); // color picker
                       ImGui::SameLine(); ImGui::SetNextItemWidth(150);
                       changed |= ImGui::InputFloat("##eMult", &rp->Emult, 0.01f, 0.1f, "%.8f");
                       return busy;
                     };
  sRCE->setFormat(float4{0.01f, 0.01f, 0.01f, 0.1f}, float4{0.1f, 0.1f, 0.1f, 0.1f}, "%0.8f");
  mSettings.push_back(sRCE); mForm->add(sRCE);
  auto *sRCB = new Setting<float4> ("B Color", "renderColB", &rp->Bcol);
  sRCB->drawCustom = [sRCB, this](bool busy, bool &changed) -> bool
                     {
                       sRCB->onDraw(1.0f, busy, changed, true); // color picker
                       ImGui::SameLine(); ImGui::SetNextItemWidth(150);
                       changed |= ImGui::InputFloat("##bMult", &rp->Bmult, 0.01f, 0.1f, "%.8f");
                       return busy;
                     };
  sRCB->setFormat(float4{0.01f, 0.01f, 0.01f, 0.1f}, float4{0.1f, 0.1f, 0.1f, 0.1f}, "%0.8f");
  mSettings.push_back(sRCB); mForm->add(sRCB);


  SettingGroup *vecGroup = new SettingGroup("Vector Field", "vecField", { }, true);
  
  // vector draw params
  auto *sVF = new Setting<bool>("Draw Vectors", "drawVec",     &vp->drawVectors);
  mSettings.push_back(sVF); vecGroup->add(sVF);
  auto *sFV = new Setting<bool>("Bordered",     "vecBordered", &vp->borderedVectors);
  mSettings.push_back(sFV); vecGroup->add(sFV);
  auto *sVI = new Setting<bool>("Smooth",       "vecSmooth",   &vp->smoothVectors);
  mSettings.push_back(sVI); vecGroup->add(sVI);
  auto *sVMR = new Setting<int>("Radius",       "vecMRad",     &vp->vecMRadius);
  sVMR->setMin(0);
  mSettings.push_back(sVMR); vecGroup->add(sVMR);
  auto *sVSP = new Setting<int>("Spacing",      "vecSpacing",  &vp->vecSpacing);
  sVSP->setMin(1);
  mSettings.push_back(sVSP); vecGroup->add(sVSP);
  auto *sVCR = new Setting<int>("Max Count",    "vecCRad",     &vp->vecCRadius);
  mSettings.push_back(sVCR); vecGroup->add(sVCR);
  
  auto *sVLME = new Setting<float>("E Length",  "vecMultE",    &vp->vecMultE);
  sVLME->setFormat(0.1f, 1.0f, "%0.4f");
  mSettings.push_back(sVLME); vecGroup->add(sVLME);
  auto *sVLMB = new Setting<float>("B Length",  "vecMultB",    &vp->vecMultB);
  sVLMB->setFormat(0.1f, 1.0f, "%0.4f");
  mSettings.push_back(sVLMB); vecGroup->add(sVLMB);
  auto *sVLW = new Setting<float>("Line Width", "vWidth",      &vp->vecLineW);
  sVLW->setFormat(0.1f, 1.0f, "%0.4f");
  mSettings.push_back(sVLW); vecGroup->add(sVLW);
  auto *sVLA = new Setting<float>("Line Alpha", "vecAlpha",    &vp->vecAlpha);
  sVLA->setFormat(0.01f, 0.1f, "%0.4f"); sVLA->setMin(0.0f); sVLA->setMax(1.0f);
  mSettings.push_back(sVLA); vecGroup->add(sVLA);

  if(vp->borderedVectors)
    {
      auto *sVBW = new Setting<float>("Border Width", "bWidth",  &vp->vecBorderW);
      sVBW->setFormat(0.1f, 1.0f, "%0.4f");
      mSettings.push_back(sVBW); vecGroup->add(sVBW);  
      auto *sVBA = new Setting<float>("Border Alpha", "vecBAlpha",  &vp->vecBAlpha);
      sVBA->setFormat(0.01f, 0.1f, "%0.4f"); sVBA->setMin(0.0f); sVBA->setMax(1.0f);
      mSettings.push_back(sVBA); vecGroup->add(sVBA);
    }
  mForm->add(vecGroup);
}

template<typename T>
inline DisplayInterface<T>::~DisplayInterface()
{
  if(mForm)          { delete mForm; mForm = nullptr; }
  if(rp && rpDelete) { delete rp;    rp    = nullptr; }
  if(vp && vpDelete) { delete vp;    vp    = nullptr; }
}


template<typename T>
inline void DisplayInterface<T>::draw()
{
  mForm->draw();
}


#endif // DISPLAY_HPP
