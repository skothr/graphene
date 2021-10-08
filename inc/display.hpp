#ifndef DISPLAY_HPP
#define DISPLAY_HPP

#include <nlohmann/json.hpp> // json implementation (NOTE: should be in settings.hpp?)
using json = nlohmann::json;

#include "render.cuh"
#include "settingForm.hpp"


// parameters for displaying underlying vectors over a field
template<typename T>
struct VectorFieldParams
{
  // vector field
  bool  drawVectors     = true;  // draws vector field on screen
  bool  smoothVectors   = false; // uses bilinear interpolation, centering at samples mouse instead of exact cell centers
  //bool  borderedVectors = false; // uses fancy bordered polygons instead of standard GL_LINES (slower -- TODO: optimize with cudaVBO(?))
  bool  mouseRadius     = true;  // only draw vectors within some radius of the mouse (for isolated inspection)
  int   vecMRadius      = 64;    //  --> radius around mouse
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
  bool showEMView  = true;
  bool showMatView = true;
  bool show3DView  = true;
  bool drawAxes    = true; // axes at origin in each view
  bool drawOutline = true; // outline of field in each view
  
  bool vsync       = true;  // vertical sync

  bool  limitFps  = false; // FPS limiter (inactive if <= 0)
  float maxFps    = 30.0f;
  
  // main parameters
  RenderParams<T>      *rp = nullptr; bool rpDelete = false;
  VectorFieldParams<T> *vp = nullptr; bool vpDelete = false;
  int *zSize = nullptr;
  
  SettingForm *mForm = nullptr;
  std::vector<SettingBase*> mExtraSettings;
  json toJSON() const
  {
    json js = (mForm ? mForm->toJSON() : json::object());
    for(auto s : mExtraSettings) { js[s->getId()] = s->toJSON(); }
    return js;
  }
  bool fromJSON(const json &js)
  {
    bool success = (mForm ? mForm->fromJSON(js) : false);
    for(auto s : mExtraSettings)
      {
        if(js.contains(s->getId())) { s->fromJSON(js[s->getId()]); }
        else                        { success = false; }
      }
    return success;
  }
  
  DisplayInterface(RenderParams<T> *rParams, VectorFieldParams<T> *vParams, int *zs);
  ~DisplayInterface();

  void draw();
  void updateAll() { mForm->updateAll(); }
};

#define COLOR_SMALLSTEP float4{0.01f, 0.01f, 0.01f, 0.1f}
#define COLOR_BIGSTEP   float4{0.1f, 0.1f, 0.1f, 0.1f}
#define COLOR_FORMAT    "%0.8f"
#define MULT_SMALLSTEP  0.01f
#define MULT_BIGSTEP    0.1f
#define MULT_FORMAT     "%0.8f"


template<typename T>
DisplayInterface<T>::DisplayInterface(RenderParams<T> *rParams, VectorFieldParams<T> *vParams, int *zs)
  : rp(rParams), vp(vParams), zSize(zs)
{
  typedef typename DimType<T, 4>::VEC_T VT4;
  
  if(!rp) { rp = new RenderParams<T>();      rpDelete = true; }
  if(!vp) { vp = new VectorFieldParams<T>(); vpDelete = true; }
  
  mForm = new SettingForm("Display Settings", 180, 300);

  // flags
  auto *sSFEM  = new Setting<bool> ("Show EM View",        "showEMView",  &showEMView);  mForm->add(sSFEM);
  auto *sSFMAT = new Setting<bool> ("Show Material View",  "showMatView", &showMatView); mForm->add(sSFMAT);
  auto *sSF3D  = new Setting<bool> ("Show 3D View",        "show3DView",  &show3DView);  mForm->add(sSF3D);
  auto *sSFA   = new Setting<bool> ("Show Axes",           "showAxes",    &drawAxes);    mForm->add(sSFA);
  auto *sSFO   = new Setting<bool> ("Show Field Outline",  "showOutline", &drawOutline); mForm->add(sSFO);


  auto *sVS    = new Setting<bool> ("VSync",               "vsync",       &vsync);       mForm->add(sVS);
  sVS->updateCallback = [&](){ glfwSwapInterval(vsync ? 1 : 0); };
  sVS->setHelp("");
  auto *sLFPS  = new Setting<float>("Limit Physics FPS",   "maxFps",      &maxFps);      mForm->add(sLFPS);
  sLFPS->setToggle(&limitFps); sLFPS->setFormat(1.0f, 10.0f, "%0.2f");
  sLFPS->setHelp("");

  
  // render params
  auto *sRZL   = new Setting<int2> ("Z Range", "zRange", &rp->zRange, rp->zRange); mForm->add(sRZL);
  sRZL->setMin(int2{0,0}); sRZL->setMax(int2{1,1}*(zSize ? *zSize-1 : 0));
  sRZL->drawCustom = [this](bool busy, bool &changed) -> bool
                     {
                       changed |= RangeSlider("##zSlider", &rp->zRange.x, &rp->zRange.y, 0, (zSize ? *zSize-1 : 0), Vec2f(250, 20));
                       return busy;
                     };
  auto *sRCS  = new Setting<bool> ("Surfaces", "surfaces", &rp->surfaces); mForm->add(sRCS);
  auto *sRSMP = new Setting<bool> ("Simple",   "simple",   &rp->simple);   mForm->add(sRSMP);
  
  SettingGroup *fDGroup   = new SettingGroup("Fluid Components",    "fData",   { }, true);
  SettingGroup *emDGroup  = new SettingGroup("EM Components",       "emData",  { }, true);
  SettingGroup *matDGroup = new SettingGroup("Material Components", "matData", { }, true);

  auto *sRCOF  = new Setting<float> ("Opacity",    "fOpacity",       &rp->fOpacity);      fDGroup->add(sRCOF);
  sRCOF->setFormat (0.01f, 0.1f, "%0.4f"); 
  auto *sRCBRF = new Setting<float> ("Brightness", "fBrightness",    &rp->fBrightness);   fDGroup->add(sRCBRF);
  sRCBRF->setFormat(0.01f, 0.1f, "%0.4f"); 
  auto *sRCOS  = new Setting<float> ("Opacity",     "emOpacity",     &rp->emOpacity);     emDGroup->add(sRCOS);
  sRCOS->setFormat (0.01f, 0.1f, "%0.4f"); 
  auto *sRCBRS = new Setting<float> ("Brightness",  "emBrightness",  &rp->emBrightness);  emDGroup->add(sRCBRS);
  sRCBRS->setFormat(0.01f, 0.1f, "%0.4f"); 
  auto *sRCOM  = new Setting<float> ("Opacity",     "matOpacity",    &rp->matOpacity);    matDGroup->add(sRCOM);
  sRCOM->setFormat (0.01f, 0.1f, "%0.4f"); 
  auto *sRCBRM = new Setting<float> ("Brightness",  "matBrightness", &rp->matBrightness); matDGroup->add(sRCBRM);
  sRCBRM->setFormat(0.01f, 0.1f, "%0.4f"); 

  SettingGroup *activeGroup = fDGroup;
  for(long long i = 0LL; i < RENDER_FLAG_COUNT; i++)
    {
      RenderFlags f = (RenderFlags)(1LL << i);
      VT4        *c = rp->getColor(f);
      T          *m = rp->getMult(f);
      
      if     (f == FLUID_RENDER_EMOFFSET)  { activeGroup = emDGroup;  }
      else if(f == FLUID_RENDER_MATOFFSET) { activeGroup = matDGroup; }
      if(!c || !m) { std::cout << "====> WARNING: DisplayInterface skipping RenderFlag " << (unsigned long long)f << " (" << i << ")\n"; continue; }
      
      std::string name  = renderFlagName(f);
      std::string nameC = name + " Color";
      std::string nameM = name + " Multiplier";
      std::string idC   = std::string("##") + name + "Color";
      std::string idM   = std::string("##") + name + "Mult";
      auto sRFC = new Setting<VT4>(nameC, idC, c);
      sRFC->setToggle(rp->getToggle(f));
      sRFC->visibleCallback = [f, this]() -> bool { return ((RenderParams<T>::MAIN & f) || !rp->simple); };
      sRFC->drawCustom = [sRFC, idM, m, this](bool busy, bool &changed) -> bool
                         {
                           sRFC->onDraw(1.0f, busy, changed, true); // color picker
                           ImGui::SameLine(); ImGui::SetNextItemWidth(150);
                           changed |= ImGui::InputFloat(idM.c_str(), m, 0.01f, 0.1f, "%.8f");
                           return busy;
                         };
      sRFC->setFormat(COLOR_SMALLSTEP, COLOR_BIGSTEP, COLOR_FORMAT);
      activeGroup->add(sRFC);
      auto sRFM = new Setting<T>(nameM, idM, m);
      sRFM->setFormat(MULT_SMALLSTEP, MULT_BIGSTEP, MULT_FORMAT);
      mExtraSettings.emplace_back(sRFM);
    }

  SettingGroup *vecGroup = new SettingGroup("Vector Field", "vecField", { }, true);
  
  // vector draw params
  auto *sVF    = new Setting<bool>("Draw Vectors", "drawVec",     &vp->drawVectors);  vecGroup->add(sVF);
  auto *sVI    = new Setting<bool>("Smooth",       "vecSmooth",   &vp->smoothVectors);vecGroup->add(sVI);
  //auto *sFV  = new Setting<bool>("Bordered",     "vecBordered", &vp->borderedVectors); vecGroup->add(sFV);
  auto *sVR    = new Setting<bool>("Mouse Radius", "mouseRadius", &vp->mouseRadius);  vecGroup->add(sVR);
  
  auto *sVMR   = new Setting<int> ("Radius",       "vecMRad",     &vp->vecMRadius);   vecGroup->add(sVMR);  sVMR->setMin(0);
  auto *sVSP   = new Setting<int> ("Spacing",      "vecSpacing",  &vp->vecSpacing);   vecGroup->add(sVSP);  sVSP->setMin(1); 
  auto *sVCR   = new Setting<int> ("Max Count",    "vecCRad",     &vp->vecCRadius);   vecGroup->add(sVCR);
  
  auto *sVLME = new Setting<float>("E Length",     "vecMultE",    &vp->vecMultE);     vecGroup->add(sVLME); sVLME->setFormat(0.1f,  1.0f, "%0.4f"); 
  auto *sVLMB = new Setting<float>("B Length",     "vecMultB",    &vp->vecMultB);     vecGroup->add(sVLMB); sVLMB->setFormat(0.1f,  1.0f, "%0.4f"); 
  auto *sVLW  = new Setting<float>("Line Width",   "vWidth",      &vp->vecLineW);     vecGroup->add(sVLW);  sVLW->setFormat (0.1f,  1.0f, "%0.4f"); 
  auto *sVLA  = new Setting<float>("Line Alpha",   "vecAlpha",    &vp->vecAlpha);     vecGroup->add(sVLA);  sVLA->setFormat (0.01f, 0.1f, "%0.4f");
  sVLA->setMin(0.0f); sVLA->setMax(1.0f);
  auto *sVBW  = new Setting<float>("Border Width", "bWidth",      &vp->vecBorderW);   vecGroup->add(sVBW);  sVBW->setFormat(0.1f,  1.0f, "%0.4f"); 
  auto *sVBA  = new Setting<float>("Border Alpha", "vecBAlpha",   &vp->vecBAlpha);    vecGroup->add(sVBA);  sVBA->setFormat(0.01f, 0.1f, "%0.4f");
  sVBA->setMin(0.0f); sVBA->setMax(1.0f);

  mForm->add(fDGroup);
  mForm->add(emDGroup);
  mForm->add(matDGroup);
  mForm->add(vecGroup);
}

template<typename T>
inline DisplayInterface<T>::~DisplayInterface()
{
  if(mForm)          { delete mForm; mForm = nullptr; }
  for(auto s : mExtraSettings) { if(s) { delete s; }  } mExtraSettings.clear();
  if(rp && rpDelete) { delete rp;    rp    = nullptr; }
  if(vp && vpDelete) { delete vp;    vp    = nullptr; }
}


template<typename T>
inline void DisplayInterface<T>::draw()
{
  mForm->draw();
}


#endif // DISPLAY_HPP
