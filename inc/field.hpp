#ifndef FIELD_HPP
#define FIELD_HPP

#include "field.cuh"
#include "settingForm.hpp"
#include "mathParser.hpp"

#define FIELD_RES_SMALL_STEP int3{  8,   8,   8}
#define FIELD_RES_LARGE_STEP int3{128, 128,  64}
#define FIELD_RES_MIN        int3{  1,   1,   1}
#define FIELD_RES_MAX        int3{2048, 2048, 512} // probably higher than it could reasonably be set, unless only in one dimension
#define TEX_RES_SMALL_STEP   int2{  8,   8, }
#define TEX_RES_LARGE_STEP   int2{128, 128, }
#define TEX_RES_MIN          int2{  1,   1, }
#define TEX_RES_MAX          int2{4096, 4096}

template<typename T>
struct FieldInterface
{
  typedef typename DimType<T, 3>::VEC_T VT3;
  int3 fieldRes = int3{256, 256, 8}; // desired resolution (number of cells) of charge field
  int2 texRes2D = int2{1024, 1024};  // desired resolution (number of cells/pixels) of rendered texture
  int2 texRes3D = int2{1024, 1024};  // desired resolution (number of cells/pixels) of rendered texture
  bool texRes3DMatch = true;         // if true, overrides to match current view resolution (depends on size of window)
  
  FieldParams<T> *cp;     // cuda field params
  bool running = true;    // play/pause
  bool updateQ = false;   // toggle charge update step
  bool updateE = true;    // toggle electric update step
  bool updateB = true;    // toggle magnetic update step
  
  bool inputDecay = true; // toggle input signal decay
  
  // (start empty) TODO
  bool initEActive   = false; // electric field   (Maxwell's equations)
  bool initBActive   = false; // magnetic field   (Maxwell's equations)
  bool initEpActive  = false; // material epsilon (permittivity)
  bool initMuActive  = false; // material mu      (permeability)
  bool initSigActive = false; // material sigma   (conductivity)
  
  std::string initEStr   = "0"; // electric field   (Maxwell's equations)
  std::string initBStr   = "0"; // magnetic field   (Maxwell's equations)
  std::string initEpStr  = "1"; // material epsilon (permittivity)
  std::string initMuStr  = "1"; // material mu      (permeability)
  std::string initSigStr = "0"; // material sigma   (conductivity)
  Expression<VT3> *initEExpr   = nullptr; // electric field (Maxwell's equations)
  Expression<VT3> *initBExpr   = nullptr; // magnetic field (Maxwell's equations)
  Expression<T>   *initEpExpr  = nullptr; // material epsilon (permittivity)
  Expression<T>   *initMuExpr  = nullptr; // material mu      (permeability)
  Expression<T>   *initSigExpr = nullptr; // material sigma   (conductivity)

  //// CUDA fill expressions
  CudaExpression<VT3>   *mFillE   = nullptr; // electric field (Maxwell's equations)
  CudaExpression<VT3>   *mFillB   = nullptr; // magnetic field (Maxwell's equations)
  CudaExpression<float> *mFillEp  = nullptr; // material epsilon (permittivity)
  CudaExpression<float> *mFillMu  = nullptr; // material mu      (permeability)
  CudaExpression<float> *mFillSig = nullptr; // material sigma   (conductivity)

  std::function<void()> fieldResCallback; // callback when field resolution is changed
  std::function<void()> texRes2DCallback; // callback when 2D texture resolution is changed
  std::function<void()> texRes3DCallback; // callback when 3D texture resolution is changed
  
  SettingForm *mForm = nullptr;
  
  json toJSON() const           { return (mForm ? mForm->toJSON() : json::object()); }
  bool fromJSON(const json &js) { return (mForm ? mForm->fromJSON(js) : false); }
  
  FieldInterface(FieldParams<T> *cp_, const std::function<void()> &fieldResCb,
                 const std::function<void()> &texRes2DCb, const std::function<void()> &texRes3DCb);
  ~FieldInterface();
  
  void draw();
  void updateAll() { mForm->updateAll(); }

  // update callback for when an init expression string is changed
  template<typename U>
  void initStrUpdateCb(std::string *str, CudaExpression<U> **fillExpr)
  {
    if(str && str->empty())   // empty string defaults to 0 (cleared field)
      { *str = "0"; }
    if(fillExpr && *fillExpr) // destroy cuda expression (reinitialized with new string by SimWindow)
      { cudaFree(*fillExpr); *fillExpr = nullptr; }
  }
};

template<typename T>
FieldInterface<T>::FieldInterface(FieldParams<T> *cp_, const std::function<void()> &frCb, const std::function<void()> &tr2DCb, const std::function<void()> &tr3DCb)
  : cp(cp_), fieldResCallback(frCb), texRes2DCallback(tr2DCb), texRes3DCallback(tr3DCb)
{
  SettingGroup *paramGroup = new SettingGroup("Field Parameters",   "fieldParams",       { }, false);
  paramGroup->setHelp("TEST");
  SettingGroup *stepGroup  = new SettingGroup("Physics Steps",      "physicsSteps",      { }, false);
  SettingGroup *initGroup  = new SettingGroup("Initial Conditions", "initialConditions", { }, false);

  // flags
  auto *sRUN   = new Setting<bool>("Running",                "running",    &running);
  paramGroup->add(sRUN);
  
  // field/texture sizes
  auto *sFRES = new Setting<int3> ("Field Resolution",       "fieldRes",   &fieldRes, fieldRes, [this]() { if(fieldResCallback) { fieldResCallback(); } });
  sFRES->setHelp("Size of field (number of cells) in each dimension");
  sFRES->setFormat(FIELD_RES_SMALL_STEP, FIELD_RES_LARGE_STEP);
  sFRES->setMin(FIELD_RES_MIN); sFRES->setMax(FIELD_RES_MAX);
  paramGroup->add(sFRES);
  auto *sTRES2 = new Setting<int2> ("Tex Resolution (2D)",   "texRes2D",   &texRes2D, texRes2D, [this]() { if(texRes2DCallback) { texRes2DCallback(); } });
  sTRES2->setHelp("Size of 2D rendered texture (EM and Material views)");
  sTRES2->setFormat(TEX_RES_SMALL_STEP, TEX_RES_LARGE_STEP);
  sTRES2->setMin(TEX_RES_MIN); sTRES2->setMax(TEX_RES_MAX);
  paramGroup->add(sTRES2);
  auto *sTRES3 = new Setting<int2> ("Tex Resolution (3D)",   "texRes3D",   &texRes3D, texRes3D, [this]() { if(texRes3DCallback) { texRes3DCallback(); } });
  sTRES3->setHelp("Size of 3D rendered texture (3D view)");
  sTRES3->setFormat(TEX_RES_SMALL_STEP, TEX_RES_LARGE_STEP);
  sTRES3->setMin(TEX_RES_MIN); sTRES3->setMax(TEX_RES_MAX);
  paramGroup->add(sTRES3);
  auto *sTRES3M = new Setting<bool>("   (Match Viewport)",  "texRes3DMatch", &texRes3DMatch);
  sTRES3M->setHelp("Dynamically fit 3D texture size to actual resolution (changes if window is resized)");
  paramGroup->add(sTRES3M);
  
  auto *sFPP = new Setting<VT3>    ("Position",          "fPos",    &cp->fp);
  sFPP->setHelp("3D position of field within sim");
  paramGroup->add(sFPP);
  auto *sREF = new Setting<bool>   ("Reflective Bounds", "reflect", &cp->reflect);
  sREF->setHelp("Reflect signals at boundary (some reflections are still apparent even if disabled)");
  paramGroup->add(sREF);
  auto *sSD = new Setting<T>       ("Input Decay",       "decay",   &cp->decay);
  sSD->setHelp("Input signal field is multiplied by decay^dt each frame to prevent stuck vectors");
  sSD->setFormat(0.01f, 0.1f);
  sSD->setMin(-2.0f); sSD->setMax(2.0f);
  paramGroup->add(sSD);

  auto *sUE  = new Setting<bool> ("Update E",           "updateE",    &updateE);    stepGroup->add(sUE);
  sUE->setHelp("Toggle electric field physics update");
  auto *sUB  = new Setting<bool> ("Update B",           "updateB",    &updateB);    stepGroup->add(sUB);
  sUB->setHelp("Toggle magnetic field physics update");
  auto *sUQ  = new Setting<bool> ("Update Q",           "updateQ",    &updateQ);    stepGroup->add(sUQ);
  sUQ->setHelp("Toggle charge field physics update (NOTE: unimplemented)");
  auto *sISD = new Setting<bool> ("Input Signal Decay", "inputDecay", &inputDecay); stepGroup->add(sISD);
  sISD->setHelp("Toggle decay for input");

  // init
  auto *sEINIT   = new Setting<std::string>("E init",   "EInit",            &initEStr,   initEStr,   [&]() { initStrUpdateCb(&initEStr,   &mFillE);   });
  initGroup->add(sEINIT);
  auto *sBINIT   = new Setting<std::string>("B init",   "BInit",            &initBStr,   initBStr,   [&]() { initStrUpdateCb(&initBStr,   &mFillB);   });
  initGroup->add(sBINIT);
  auto *sEPINIT  = new Setting<std::string>("Material init (ε)", "epInit",  &initEpStr,  initEpStr,  [&]() { initStrUpdateCb(&initEpStr,  &mFillEp);  });
  initGroup->add(sEPINIT);
  auto *sMUINIT  = new Setting<std::string>("Material init (μ)", "muInit",  &initMuStr,  initMuStr,  [&]() { initStrUpdateCb(&initMuStr,  &mFillMu);  });
  initGroup->add(sMUINIT);
  auto *sSIGINIT = new Setting<std::string>("Material init (σ)", "sigInit", &initSigStr, initSigStr, [&]() { initStrUpdateCb(&initSigStr, &mFillSig); });
  initGroup->add(sSIGINIT);

  mForm = new SettingForm("Simulation Settings", 180, 300);
  mForm->add(paramGroup);
  mForm->add(stepGroup);
  mForm->add(initGroup);
}

template<typename T>
FieldInterface<T>::~FieldInterface()
{ if(mForm) { delete mForm; mForm = nullptr; } }

template<typename T>
void FieldInterface<T>::draw()
{ mForm->draw(); }

#endif // FIELD_HPP
