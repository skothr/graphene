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
  bool  drawVectors   = true;  // draws vector field on screen
  bool  mouseRadius   = true;  // only draw vectors within some radius of the mouse (for isolated inspection)
  bool  smoothVectors = false; // uses bilinear interpolation, centering at samples mouse instead of exact cell centers
  bool  scaleToView   = false; // scale line width based on view scale
  int   spacing       = 1;     // base spacing
  int   mRadius       = 32;    //  --> radius around mouse
  int   cRadius       = 256;   //  -- maximum radius (in number of vectors drawn) -- adds spacing if needed
  
  float aBase         = 0.5f;  // base alpha mult
  float lBase         = 1.0f;  // base length mult
  float wBase         = 3.0f;  // base line width mult
  
  // toggles
  bool   drawV   = false; // V  (fluid velocity)
  bool   drawQpv = false; // (Q+)v (charge velocity)
  bool   drawQnv = false; // (Q-)v (charge velocity)
  bool   drawE   = true;  // E  (electric field strength)
  bool   drawB   = true;  // B  (magnetic field strength)
  // line colors
  float4 colV   = float4{1.0f, 0.0f, 0.0f, 1.0f};
  float4 colQnv = float4{1.0f, 0.0f, 1.0f, 1.0f};
  float4 colQpv = float4{1.0f, 0.0f, 1.0f, 1.0f};
  float4 colE   = float4{0.0f, 1.0f, 0.0f, 1.0f};
  float4 colB   = float4{0.0f, 0.0f, 1.0f, 1.0f};
  // length multipliers
  float multV   = 1.0f;
  float multQnv = 1.0f;
  float multQpv = 1.0f;
  float multE   = 1.0f;
  float multB   = 1.0f;
  // width multipliers
  float lwV   = 1.0f;
  float lwQnv = 1.0f;
  float lwQpv = 1.0f;
  float lwE   = 1.0f;
  float lwB   = 1.0f;
};

template<typename T>
struct DisplayInterface : public SettingForm
{
  // flags
  bool showEMView  = true;
  bool showMatView = false;
  bool show3DView  = true;
  bool drawAxes    = true; // X/Y/Z axes from <0,0,0> in each view
  bool drawOutline = true; // outline of field in each view
  
  // main parameters
  RenderParams<T>      *rp = nullptr; bool rpDelete = false;
  VectorFieldParams<T> *vp = nullptr; bool vpDelete = false;
  int *zSize = nullptr;
  
  DisplayInterface(RenderParams<T> *rParams, VectorFieldParams<T> *vParams, int *zs);
  ~DisplayInterface();
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
  if(!rp) { rp = new RenderParams<T>();      rpDelete = true; }
  if(!vp) { vp = new VectorFieldParams<T>(); vpDelete = true; }
  
  // render params
  SettingGroup *rGroup = new SettingGroup("Render Data",  "renderData", { }); add(rGroup);
  auto *sRZL   = new SliderSetting<int>("Z Range", "zRange", &rp->zRange, nullptr, zSize); rGroup->add(sRZL);
  auto *sRCS  = new Setting<bool> ("Surfaces", "surfaces", &rp->surfaces); rGroup->add(sRCS);
  sRCS->setHelp ("Interesting ray marching option that solidifies wavefront surfaces \n"
                 " off --> Ray marching breaks if A >= 1\n"
                 "  on --> Ray marching breaks if R, G, B, or A is >= 1");

  // vector draw params
  SettingGroup *vGroup = new SettingGroup("Vector Field", "vecField", { }); add(vGroup);
  auto *sVE    = new Setting<bool> ("Enable",        "vEnable",     &vp->drawVectors);   vGroup->add(sVE);
  sVE->setHelp ("Enable vector field overlay over 2D view (NOTE: may be slow with a lot of vectors)");
  auto *sVR    = new Setting<bool> ("Mouse Radius",  "mouseRadius", &vp->mouseRadius);   vGroup->add(sVR);
  sVR->setHelp (" off --> draws vectors across entire field\n"
                "  on --> draws vectors within radius around mouse position");
  auto *sVI    = new Setting<bool> ("Smooth",        "vecSmooth",   &vp->smoothVectors); vGroup->add(sVI);
  sVI->setHelp (" off --> shows exact cell contents at cell centers\n"
                "  on --> interpolates between cells (based on mouse position)");
  auto *sVWV   = new Setting<bool> ("Scale to View", "scaleToView", &vp->scaleToView);   vGroup->add(sVWV);
  sVWV->setHelp(" off --> line widths relative to screen (unaffected by zoom) \n"
                "  on --> line widths relative to cell size");
  auto *sVMR   = new Setting<int>  ("Radius",        "mRad",        &vp->mRadius);       vGroup->add(sVMR);   sVMR->setMin(0);
  sVMR->setHelp("Radius around mouse within which vectors are drawn");
  auto *sVCR   = new Setting<int>  ("Max Count",     "cRad",        &vp->cRadius);       vGroup->add(sVCR);   sVCR->setMin(0);
  sVCR->setHelp("Spacing is adjusted if Mouse Radius is greater than this");
  auto *sVSP   = new Setting<int>  ("Spacing",       "spacing",     &vp->spacing);       vGroup->add(sVSP);   sVSP->setMin(1);
  sVSP->setHelp("Vector spacing/stride --> decreases density by only showing 1/N cells");
  
  auto *sVLCV   = new ColorSetting  ("", "colV",     &vp->colV);
  auto *sVLCQnv = new ColorSetting  ("", "colQnv",   &vp->colQnv);
  auto *sVLCQpv = new ColorSetting  ("", "colQpv",   &vp->colQpv);
  auto *sVLCE   = new ColorSetting  ("", "colE",     &vp->colE);
  auto *sVLCB   = new ColorSetting  ("", "colB",     &vp->colB);
  auto *sVLAV   = new Setting<float>("", "alphaV",   &vp->colV.w);   sVLAV->setMin(0.0f);   sVLAV->setFormat  (0.01f, 0.1f, "%0.4f");
  auto *sVLAQnv = new Setting<float>("", "alphaQnv", &vp->colQnv.w); sVLAQnv->setMin(0.0f); sVLAQnv->setFormat(0.01f, 0.1f, "%0.4f");
  auto *sVLAQpv = new Setting<float>("", "alphaQpv", &vp->colQpv.w); sVLAQpv->setMin(0.0f); sVLAQpv->setFormat(0.01f, 0.1f, "%0.4f");
  auto *sVLAE   = new Setting<float>("", "alphaE",   &vp->colE.w);   sVLAE->setMin(0.0f);   sVLAE->setFormat  (0.01f, 0.1f, "%0.4f");
  auto *sVLAB   = new Setting<float>("", "alphaB",   &vp->colB.w);   sVLAB->setMin(0.0f);   sVLAB->setFormat  (0.01f, 0.1f, "%0.4f");
  
  auto *sVLMV   = new Setting<float>("", "multV",   &vp->multV);   sVLMV->setFormat  (0.01f, 0.1f, "%0.4f");
  auto *sVLMQnv = new Setting<float>("", "multQnv", &vp->multQnv); sVLMQnv->setFormat(0.01f, 0.1f, "%0.4f");
  auto *sVLMQpv = new Setting<float>("", "multQpv", &vp->multQpv); sVLMQpv->setFormat(0.01f, 0.1f, "%0.4f");
  auto *sVLME   = new Setting<float>("", "multE",   &vp->multE);   sVLME->setFormat  (0.01f, 0.1f, "%0.4f");
  auto *sVLMB   = new Setting<float>("", "multB",   &vp->multB);   sVLMB->setFormat  (0.01f, 0.1f, "%0.4f");
  auto *sVLWV   = new Setting<float>("", "lwV",     &vp->lwV);     sVLWV->setFormat  (0.01f, 0.1f, "%0.4f");
  auto *sVLWQnv = new Setting<float>("", "lwQnv",   &vp->lwQnv);   sVLWQnv->setFormat(0.01f, 0.1f, "%0.4f");
  auto *sVLWQpv = new Setting<float>("", "lwQpv",   &vp->lwQpv);   sVLWQpv->setFormat(0.01f, 0.1f, "%0.4f");
  auto *sVLWE   = new Setting<float>("", "lwE",     &vp->lwE);     sVLWE->setFormat  (0.01f, 0.1f, "%0.4f");
  auto *sVLWB   = new Setting<float>("", "lwB",     &vp->lwB);     sVLWB->setFormat  (0.01f, 0.1f, "%0.4f");

  auto *sVA = new Setting<float>("", "vAlpha",  &vp->aBase); sVA->setFormat(0.01f, 0.1f, "%0.4f"); sVA->setMin(0.0f); sVA->setMax(1.0f);
  auto *sVL = new Setting<float>("", "vLength", &vp->lBase); sVL->setFormat(0.01f, 0.1f, "%0.4f");
  auto *sVW = new Setting<float>("", "vWidth",  &vp->wBase); sVW->setFormat( 0.1f, 1.0f, "%0.4f");
  std::vector<SettingBase*> sGlobal{{sVA, sVL, sVW}};
  auto *g = new SettingGroup("Global", "globalMask", sGlobal); vGroup->add(g); g->setColumns(3); g->setHorizontal(true);
  g->setColumnLabels(std::vector<std::string>{"Alpha", "Length", "Width"});

  auto *sVVG   = new SettingGroup("V",  "vV",  {sVLCV,  sVLAV,  sVLMV,  sVLWV});  sVVG->setToggle(&vp->drawV); sVVG->setColumns(4);  sVVG->setHorizontal(true);
  sVVG->setColumnLabels(std::vector<std::string>{"", "Alpha", "Length", "Width"});
  sVVG->setHelp("Fluid velocity vector parameters");
  auto *sVQnvG = new SettingGroup("Qnv", "vQnv", {sVLCQnv, sVLAQnv, sVLMQnv, sVLWQnv}); sVQnvG->setToggle(&vp->drawQnv);
  sVQnvG->setColumns(4); sVQnvG->setHorizontal(true);
  sVQnvG->setHelp("Charge velocity vector parameters");
  auto *sVQpvG = new SettingGroup("Qpv", "vQpv", {sVLCQpv, sVLAQpv, sVLMQpv, sVLWQpv}); sVQpvG->setToggle(&vp->drawQpv);
  sVQpvG->setColumns(4); sVQpvG->setHorizontal(true);
  sVQpvG->setHelp("Charge velocity vector parameters");
  auto *sVEG   = new SettingGroup("E",  "vE",  {sVLCE,  sVLAE,  sVLME,  sVLWE});  sVEG->setToggle(&vp->drawE); sVEG->setColumns(4);  sVEG->setHorizontal(true);
  sVEG->setHelp("E (electric) field vector parameters");
  auto *sVBG   = new SettingGroup("B",  "vB",  {sVLCB,  sVLAB,  sVLMB,  sVLWB});  sVBG->setToggle(&vp->drawB); sVBG->setColumns(4);  sVBG->setHorizontal(true);
  sVBG->setHelp("B (magnetic) field vector parameters");
  
  auto *vvGroup = new SettingGroup("Fields", "vV",  {sVVG, sVQnvG, sVQpvG, sVEG, sVBG}); vGroup->add(vvGroup); // vvGroup->setToggle(&vp->drawVectors);

  // visualized render data
  auto *sRSMP = new Setting<bool> ("Simple", "simple", &rp->simple);
  sRSMP->setHelp("Hide extra options for a more compact interface and slightly faster performance");
  
  auto *sRCOF  = new Setting<float> ("Opacity",    "fOpacity",       &rp->fOpacity);      sRCOF->setFormat (MULT_SMALLSTEP, MULT_BIGSTEP, MULT_FORMAT);
  auto *sRCBRF = new Setting<float> ("Brightness", "fBrightness",    &rp->fBrightness);   sRCBRF->setFormat(MULT_SMALLSTEP, MULT_BIGSTEP, MULT_FORMAT);
  auto *sRCOS  = new Setting<float> ("Opacity",     "emOpacity",     &rp->emOpacity);     sRCOS->setFormat (MULT_SMALLSTEP, MULT_BIGSTEP, MULT_FORMAT);
  auto *sRCBRS = new Setting<float> ("Brightness",  "emBrightness",  &rp->emBrightness);  sRCBRS->setFormat(MULT_SMALLSTEP, MULT_BIGSTEP, MULT_FORMAT);
  auto *sRCOM  = new Setting<float> ("Opacity",     "matOpacity",    &rp->matOpacity);    sRCOM->setFormat (MULT_SMALLSTEP, MULT_BIGSTEP, MULT_FORMAT);
  auto *sRCBRM = new Setting<float> ("Brightness",  "matBrightness", &rp->matBrightness); sRCBRM->setFormat(MULT_SMALLSTEP, MULT_BIGSTEP, MULT_FORMAT);
  SettingGroup *fDGroup   = new SettingGroup("Fluid",             "fRender",    { sRCBRF, sRCOF }); fDGroup->setColumns(1);   fDGroup->setCollapsible(true);
  SettingGroup *emDGroup  = new SettingGroup("EM",                "emRender",   { sRCBRS, sRCOS }); emDGroup->setColumns(1);  emDGroup->setCollapsible(true);
  SettingGroup *matDGroup = new SettingGroup("Material",          "matRender",  { sRCBRM, sRCOM }); matDGroup->setColumns(1); matDGroup->setCollapsible(true);
  SettingGroup *compGroup = new SettingGroup("Render Components", "components", { sRSMP, fDGroup, emDGroup, matDGroup }); add(compGroup);
  
  g = fDGroup;
  SettingGroup *subgroup = nullptr;
  for(long long i = 0LL; i < RENDER_FLAG_COUNT; i++)
    {
      RenderFlags f = (RenderFlags)(1LL << i);
      float4     *c = rp->getColor(f);
      T          *m = rp->getMult(f);
      
      if     (f == FLUID_RENDER_EMOFFSET)  { g = emDGroup;  subgroup = nullptr; }
      else if(f == FLUID_RENDER_MATOFFSET) { g = matDGroup; subgroup = nullptr; }
      if(!c || !m) { std::cout << "====> WARNING: DisplayInterface skipping RenderFlag " << renderFlagName(f) << " (2^" << i << ")\n"; continue; }
      
      std::string name  = renderFlagName(f);
      std::string id    = name + "(2^"+std::to_string(i)+")";
      std::string idC   = id + "-color";
      std::string idM   = id + "-mult";
      
      std::string gName = renderFlagGroupName(f);
      if(gName.find("INVALID") == std::string::npos)
        { // create new tree group (TODO --> tree)
          subgroup = new SettingGroup(gName, gName, { }); g->add(subgroup);
        }
      
      auto sRFC = new ColorSetting("", idC, c); sRFC->setFormat(COLOR_SMALLSTEP, COLOR_BIGSTEP, COLOR_FORMAT);
      auto sRFM = new Setting<T>  ("", idM, m); sRFM->setFormat(MULT_SMALLSTEP,  MULT_BIGSTEP,  MULT_FORMAT);
      
      SettingGroup *rfg = new SettingGroup(name, id, { sRFC, sRFM });
      rfg->setHorizontal(true); rfg->setColumns(2); rfg->setToggle(rp->getToggle(f));
      rfg->setVisibleCallback([f, this]() -> bool { return ((RenderParams<T>::MAIN & f) || !rp->simple); });
      (subgroup ? subgroup : g)->add(rfg);
    }
  
  // views
  auto *sSFEM  = new Setting<bool>("EM View",       "EMView",   &showEMView);
  auto *sSFMAT = new Setting<bool>("Material View", "MatView",  &showMatView);
  auto *sSF3D  = new Setting<bool>("3D View",       "3DView",   &show3DView);
  auto *viewGroup = new SettingGroup("Views",       "viewGroup", { sSFEM, sSFMAT, sSF3D }); add(viewGroup, true);
  // features (axes, field outline)
  auto *sSFA   = new Setting<bool>("Axes",          "axes",     &drawAxes);
  auto *sSFO   = new Setting<bool>("Field Outline", "fOutline", &drawOutline);
  auto *featGroup = new SettingGroup("Features",    "featureFlags", { sSFA, sSFO }); add(featGroup, true);

}

template<typename T>
inline DisplayInterface<T>::~DisplayInterface()
{
  if(rp && rpDelete) { delete rp; rp = nullptr; }
  if(vp && vpDelete) { delete vp; vp = nullptr; }
}


#endif // DISPLAY_HPP
