#ifndef DISPLAY_HPP
#define DISPLAY_HPP

#include "render.cuh"
#include "settingForm.hpp"


// parameters for displaying underlying vectors over a field
template<typename T>
struct VectorFieldParams
{
  // vector field
  bool  drawVectors     = false; // draws vector field on screen
  bool  borderedVectors = false; // uses fancy bordered polygons instead of standard GL_LINES (NOTE: slower)   TODO: optimize with VBO
  bool  smoothVectors   = true;  // uses bilinear interpolation, centering at samples mouse instead of exact cell centers
  int   vecMRadius      = 64;    // draws vectors for cells within radius around mouse
  int   vecSpacing      = 1;     // base spacing 
  int   vecCRadius      = 1024;  // only draws maximum of this radius number of vectors, adding spacing  
  float vecMultE        = 1.0f;  // E length multiplier
  float vecMultB        = 1.0f;  // B length multiplier
  float vecLineW        = 0.2f;  // line width
  float vecAlpha        = 0.2f;  // line opacity
  float vecBorderW      = 1.0f;  // border width
  float vecBAlpha       = 1.0f;  // border opacity
};

template<typename T>
struct DisplayInterface
{
  // flags
  bool vsync       = true; // vertical sync refresh
  bool showEMView  = true;
  bool showMatView = true;
  bool show3DView  = true;
  bool drawAxes    = true; // axes at origin in each view
  bool drawOutline = true; // outline of field in each view
  
  // main parameters
  RenderParams<T>      *rp = nullptr; bool rpDelete = false;
  VectorFieldParams<T> *vp = nullptr; bool vpDelete = false;
  int *zSize = nullptr;
  
  SettingForm *mForm = nullptr;
  json toJSON() const           { return (mForm ? mForm->toJSON() : json::object()); }
  bool fromJSON(const json &js) { return (mForm ? mForm->fromJSON(js) : false); }
  
  DisplayInterface(RenderParams<T> *rParams, VectorFieldParams<T> *vParams, int *zs);
  ~DisplayInterface();

  void draw();
  void updateAll() { mForm->updateAll(); }
};


template<typename T>
DisplayInterface<T>::DisplayInterface(RenderParams<T> *rParams, VectorFieldParams<T> *vParams, int *zs)
  : rp(rParams), vp(vParams), zSize(zs)
{
  if(!rp) { rp = new RenderParams<T>();      rpDelete = true; }
  if(!vp) { vp = new VectorFieldParams<T>(); vpDelete = true; }
  
  mForm = new SettingForm("Display Settings", 180, 300);

  // flags
  auto *sVS    = new Setting<bool> ("vSync",               "vsync",        &vsync, vsync, [&]() { glfwSwapInterval(vsync ? 1 : 0); });
  sVS->setHelp("");
  mForm->add(sVS);
  auto *sSFEM  = new Setting<bool> ("Show EM View",        "showEMView",   &showEMView);
  mForm->add(sSFEM);
  auto *sSFMAT  = new Setting<bool>("Show Material View",  "showMatView",  &showMatView);
  mForm->add(sSFMAT);
  auto *sSF3D  = new Setting<bool> ("Show 3D View",        "show3DView",   &show3DView);
  mForm->add(sSF3D);
  auto *sSFA   = new Setting<bool> ("Show Axes",           "showAxes",     &drawAxes);
  mForm->add(sSFA);
  auto *sSFO   = new Setting<bool> ("Show Field Outline",  "showOutline",  &drawOutline);
  mForm->add(sSFO);

  // render params
  auto *sRZL = new Setting<int2> ("Z Range", "zRange", &rp->zRange, rp->zRange);
  sRZL->drawCustom = [this](bool busy, bool &changed) -> bool
                     {
                       changed |= RangeSlider("##zSlider", &rp->zRange.x, &rp->zRange.y, 0, (zSize ? *zSize-1 : 0), Vec2f(250, 20));
                       return busy;
                     };
  sRZL->setMin(int2{0,0});
  mForm->add(sRZL);
  
  auto *sRCO  = new Setting<float> ("Opacity",    "opacity",    &rp->opacity);
  sRCO->setFormat(0.01f, 0.1f, "%0.4f");
  mForm->add(sRCO);
  auto *sRCBR = new Setting<float> ("Brightness", "brightness", &rp->brightness);
  sRCBR->setFormat(0.01f, 0.1f, "%0.4f");
  mForm->add(sRCBR);
  auto *sRCS = new Setting<bool> ("Surfaces",     "surfaces",   &rp->surfaces);
  mForm->add(sRCS);
  auto *sRCE = new Setting<float4> ("E Color", "renderColE", &rp->Ecol);
  sRCE->drawCustom = [sRCE, this](bool busy, bool &changed) -> bool
                     {
                       sRCE->onDraw(1.0f, busy, changed, true); // color picker
                       ImGui::SameLine(); ImGui::SetNextItemWidth(150);
                       changed |= ImGui::InputFloat("##eMult", &rp->Emult, 0.01f, 0.1f, "%.8f");
                       return busy;
                     };
  sRCE->setFormat(float4{0.01f, 0.01f, 0.01f, 0.1f}, float4{0.1f, 0.1f, 0.1f, 0.1f}, "%0.8f");
  mForm->add(sRCE);
  auto *sRCB = new Setting<float4> ("B Color", "renderColB", &rp->Bcol);
  sRCB->drawCustom = [sRCB, this](bool busy, bool &changed) -> bool
                     {
                       sRCB->onDraw(1.0f, busy, changed, true); // color picker
                       ImGui::SameLine(); ImGui::SetNextItemWidth(150);
                       changed |= ImGui::InputFloat("##bMult", &rp->Bmult, 0.01f, 0.1f, "%.8f");
                       return busy;
                     };
  sRCB->setFormat(float4{0.01f, 0.01f, 0.01f, 0.1f}, float4{0.1f, 0.1f, 0.1f, 0.1f}, "%0.8f");
  mForm->add(sRCB);


  SettingGroup *vecGroup = new SettingGroup("Vector Field", "vecField", { }, true);
  
  // vector draw params
  auto *sVF = new Setting<bool>("Draw Vectors", "drawVec",     &vp->drawVectors);
  vecGroup->add(sVF);
  auto *sFV = new Setting<bool>("Bordered",     "vecBordered", &vp->borderedVectors);
  vecGroup->add(sFV);
  auto *sVI = new Setting<bool>("Smooth",       "vecSmooth",   &vp->smoothVectors);
  vecGroup->add(sVI);
  auto *sVMR = new Setting<int>("Radius",       "vecMRad",     &vp->vecMRadius);
  sVMR->setMin(0);
  vecGroup->add(sVMR);
  auto *sVSP = new Setting<int>("Spacing",      "vecSpacing",  &vp->vecSpacing);
  sVSP->setMin(1);
  vecGroup->add(sVSP);
  auto *sVCR = new Setting<int>("Max Count",    "vecCRad",     &vp->vecCRadius);
  vecGroup->add(sVCR);
  
  auto *sVLME = new Setting<float>("E Length",  "vecMultE",    &vp->vecMultE);
  sVLME->setFormat(0.1f, 1.0f, "%0.4f");
  vecGroup->add(sVLME);
  auto *sVLMB = new Setting<float>("B Length",  "vecMultB",    &vp->vecMultB);
  sVLMB->setFormat(0.1f, 1.0f, "%0.4f");
  vecGroup->add(sVLMB);
  auto *sVLW = new Setting<float>("Line Width", "vWidth",      &vp->vecLineW);
  sVLW->setFormat(0.1f, 1.0f, "%0.4f");
  vecGroup->add(sVLW);
  auto *sVLA = new Setting<float>("Line Alpha", "vecAlpha",    &vp->vecAlpha);
  sVLA->setFormat(0.01f, 0.1f, "%0.4f"); sVLA->setMin(0.0f); sVLA->setMax(1.0f);
  vecGroup->add(sVLA);

  auto *sVBW = new Setting<float>("Border Width", "bWidth",  &vp->vecBorderW);
  sVBW->setFormat(0.1f, 1.0f, "%0.4f");
  vecGroup->add(sVBW);  
  auto *sVBA = new Setting<float>("Border Alpha", "vecBAlpha",  &vp->vecBAlpha);
  sVBA->setFormat(0.01f, 0.1f, "%0.4f"); sVBA->setMin(0.0f); sVBA->setMax(1.0f);
  vecGroup->add(sVBA);

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
