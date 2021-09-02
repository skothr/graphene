#ifndef FIELD_HPP
#define FIELD_HPP


#include "field.cuh"
#include "setting.hpp"
#include "settingForm.hpp"
#include "mathParser.hpp"

template<typename T>
struct FieldInterface
{
  typedef typename DimType<T, 3>::VECTOR_T VT3;
  int3 fieldRes = int3{256, 256, 16}; // desired resolution (number of cells) of charge field
  int2 texRes2D = int2{1024, 1024};   // desired resolution (number of cells) of rendered texture
  int2 texRes3D = int2{1024, 1024};   // desired resolution (number of cells) of rendered texture
  bool texRes3DMatch = true;          // matches resolution of viewport
  
  FieldParams<T> *cp; // cuda field params

  bool running = false; // play/pause  
  bool updateQ = true;  // toggle charge   field update step
  bool updateE = true;  // toggle electric field update step
  bool updateB = true;  // toggle magnetic field update step

  std::string initQPStr  = "0";
  std::string initQNStr  = "0";
  std::string initQPVStr = "0";
  std::string initQNVStr = "0";
  std::string initEStr   = "cos((len(r)^2)/512)";
  std::string initBStr   = "sin(t*137)";

  Expression<T>   *initQPExpr  = nullptr;
  Expression<T>   *initQNExpr  = nullptr;
  Expression<VT3> *initQPVExpr = nullptr;
  Expression<VT3> *initQNVExpr = nullptr;
  Expression<VT3> *initEExpr   = nullptr;
  Expression<VT3> *initBExpr   = nullptr;

  // CUDA fill expressions
  CudaExpression<T>   *mFillQP  = nullptr;
  CudaExpression<T>   *mFillQN  = nullptr;
  CudaExpression<VT3> *mFillQPV = nullptr;
  CudaExpression<VT3> *mFillQNV = nullptr;
  CudaExpression<VT3> *mFillE   = nullptr;
  CudaExpression<VT3> *mFillB   = nullptr;


  std::function<void()> fieldResCallback; // callback when field resolution is changed
  std::function<void()> texRes2DCallback; // callback when 2D texture resolution is changed
  std::function<void()> texRes3DCallback; // callback when 3D texture resolution is changed
  
  std::vector<SettingBase*> mSettings;
  SettingForm *mForm = nullptr;
  
  FieldInterface(FieldParams<T> *cp_, const std::function<void()> &fieldResCb,
                 const std::function<void()> &texRes2DCb, const std::function<void()> &texRes3DCb);
  ~FieldInterface();

  void draw();
};

template<typename T>
FieldInterface<T>::FieldInterface(FieldParams<T> *cp_, const std::function<void()> &frCb, const std::function<void()> &tr2DCb, const std::function<void()> &tr3DCb)
  : cp(cp_), fieldResCallback(frCb), texRes2DCallback(tr2DCb), texRes3DCallback(tr3DCb)
{
  SettingGroup *simGroup  = new SettingGroup("Field Parameters",   "fieldParams",       { }, false);
  SettingGroup *stepGroup = new SettingGroup("Physics Steps",      "physicsSteps",      { }, false);
  SettingGroup *initGroup = new SettingGroup("Initial Conditions", "initialConditions", { }, false);

  // flags
  auto *sRUN   = new Setting<bool> ("running",             "running",      &running);
  mSettings.push_back(sRUN); simGroup->add(sRUN);

  
  // field/texture sizes
  auto *sFRES = new Setting<int3> ("Field Resolution", "fieldRes", &fieldRes, fieldRes,
                                   [this]() 
                                   {
                                     std::cout << "RESIZING FIELD --> " << fieldRes << "\n";
                                     if(fieldResCallback) { fieldResCallback(); }
                                   });
  sFRES->setFormat(int3{64, 64, 32}, int3{128, 128, 64});
  sFRES->setMin(int3{1, 1, 1}); sFRES->setMax(int3{2048, 2048, 1024});
  mSettings.push_back(sFRES); simGroup->add(sFRES);
  auto *sTRES2 = new Setting<int2> ("Tex Resolution (2D)",   "texRes2D",   &texRes2D,   texRes2D,
                                    [this]()
                                    {
                                      std::cout << "RESIZING 2D TEXTURES --> " << texRes2D << "\n";
                                      if(texRes2DCallback) { texRes2DCallback(); }
                                    });
  sTRES2->setFormat(int2{64, 64}, int2{128, 128});
  sTRES2->setMin(int2{1,1}); sTRES2->setMax(int2{2048,2048});
  mSettings.push_back(sTRES2); simGroup->add(sTRES2);
  auto *sTRES3 = new Setting<int2> ("Tex Resolution (3D)",   "texRes3D",   &texRes3D,   texRes3D,
                                    [this]()
                                    {
                                      std::cout << "RESIZING 3D TEXTURE --> " << texRes3D << "\n";
                                      if(texRes3DCallback) { texRes3DCallback(); }
                                    });
  sTRES3->setFormat(int2{64, 64}, int2{128, 128});
  sTRES3->setMin(int2{1,1}); sTRES3->setMax(int2{2048,2048});
  mSettings.push_back(sTRES3); simGroup->add(sTRES3);
  auto *sTRES3M = new Setting<bool> ("   (Match Viewport)", "texRes3DMatch",   &texRes3DMatch);
  mSettings.push_back(sTRES3M); simGroup->add(sTRES3M);
  
  auto *sFPP = new Setting<float3> ("Position", "fPos", &cp->fp);
  mSettings.push_back(sFPP); simGroup->add(sFPP);
  auto *sREF = new Setting<bool> ("Reflective Bounds", "reflect", &cp->reflect);
  mSettings.push_back(sREF); simGroup->add(sREF);

  
  auto *sUE = new Setting<bool> ("Update E", "updateE", &updateE); mSettings.push_back(sUE); stepGroup->add(sUE);
  auto *sUB = new Setting<bool> ("Update B", "updateB", &updateB); mSettings.push_back(sUB); stepGroup->add(sUB);
  auto *sUQ = new Setting<bool> ("Update Q", "updateQ", &updateQ); mSettings.push_back(sUQ); stepGroup->add(sUQ);

  // init
  auto *sQPINIT  = new Setting<std::string>("q+ init",  "qpInit",  &initQPStr, initQPStr,
                                            [&]() {  if(initQPStr.empty())  { initQPStr  = "0"; } if(mFillQP)  { cudaFree(mFillQP);  mFillQP  = nullptr; } });
  mSettings.push_back(sQPINIT); initGroup->add(sQPINIT);
  auto *sQNINIT  = new Setting<std::string>("q- init",  "qnInit",  &initQNStr, initQNStr,
                                            [&]() {  if(initQNStr.empty())  { initQNStr  = "0"; } if(mFillQN)  { cudaFree(mFillQN);  mFillQN  = nullptr; } });
  mSettings.push_back(sQNINIT); initGroup->add(sQNINIT);
  auto *sQPVINIT = new Setting<std::string>("Vq+ init", "qpvInit", &initQPVStr, initQPVStr,
                                            [&]() {  if(initQPVStr.empty()) { initQPVStr = "0"; } if(mFillQPV) { cudaFree(mFillQPV); mFillQPV = nullptr; } });
  mSettings.push_back(sQPVINIT); initGroup->add(sQPVINIT);
  auto *sQNVINIT = new Setting<std::string>("Vq- init", "qnvInit", &initQNVStr, initQNVStr,
                                            [&]() {  if(initQNVStr.empty()) { initQNVStr = "0"; } if(mFillQNV) { cudaFree(mFillQNV); mFillQNV = nullptr; } });
  mSettings.push_back(sQNVINIT); initGroup->add(sQNVINIT);
  auto *sEINIT   = new Setting<std::string>("E init",   "EInit",   &initEStr, initEStr,
                                            [&]() {  if(initEStr.empty())   { initEStr   = "0"; } if(mFillE)   { cudaFree(mFillE);   mFillE   = nullptr; } });
  mSettings.push_back(sEINIT); initGroup->add(sEINIT);
  auto *sBINIT   = new Setting<std::string>("B init",   "BInit",   &initBStr, initBStr,
                                            [&]() {  if(initBStr.empty())   { initBStr   = "0"; } if(mFillB)   { cudaFree(mFillB);   mFillB   = nullptr; } });
  mSettings.push_back(sBINIT); initGroup->add(sBINIT);


  mForm = new SettingForm("Simulation Settings", 180, 300);
  mForm->add(simGroup);
  mForm->add(stepGroup);
  mForm->add(initGroup);
  
}

template<typename T>
FieldInterface<T>::~FieldInterface()
{
  if(mForm) { delete mForm; mForm = nullptr; }
}

template<typename T>
void FieldInterface<T>::draw()
{
  mForm->draw();
}


#endif // FIELD_HPP
