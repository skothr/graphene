#include "simWindow.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <functional>
#include <thread>
#include <mutex>
#include <unistd.h>

#include "glfwKeys.hpp"
#include "image.hpp"
#include "imtools.hpp"
#include "tools.hpp"
#include "settingForm.hpp"
#include "setting.hpp"
#include "tabMenu.hpp"


inline std::string fAlign(float f, int maxDigits)
{
  std::stringstream ss;
  ss << std::setprecision(4);
  if(log10(f) >= maxDigits) { ss << std::scientific << f; }
  else                      { ss << std::fixed      << f; }
  return ss.str();
}


static SimWindow *simWin = nullptr;
void SimWindow::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) { simWin->keyPress(mods, key, action); }


// handle key events manually to make sure none are missed
void SimWindow::keyPress(int mods, int key, int action)
{
  bool press    = (action == GLFW_PRESS);
  bool repeat   = (action == GLFW_REPEAT);
  bool release  = (action == GLFW_RELEASE);
  bool anyPress = (press || repeat);

  if(anyPress) { mKeysDown[key] = true;  }
  else         { mKeysDown[key] = false; }

  int mkl, mkr;
  mkl = GLFW_KEY_LEFT_SHIFT;   mkr = GLFW_KEY_RIGHT_SHIFT;
  bool shift = ((mKeysDown.find(mkl) != mKeysDown.end() && mKeysDown[mkl]) ||
                (mKeysDown.find(mkr) != mKeysDown.end() && mKeysDown[mkr]));
  mkl = GLFW_KEY_LEFT_CONTROL; mkr = GLFW_KEY_RIGHT_CONTROL;
  bool ctrl  = ((mKeysDown.find(mkl) != mKeysDown.end() && mKeysDown[mkl]) ||
                (mKeysDown.find(mkr) != mKeysDown.end() && mKeysDown[mkr]));
  mkl = GLFW_KEY_LEFT_ALT;     mkr = GLFW_KEY_RIGHT_ALT;
  bool alt   = ((mKeysDown.find(mkl) != mKeysDown.end() && mKeysDown[mkl]) ||
                (mKeysDown.find(mkr) != mKeysDown.end() && mKeysDown[mkr]));
  float shiftMult = (shift ? 0.1 : 1);
  float ctrlMult  = (ctrl  ? 10  : 1);
  float altMult   = (alt   ? 100 : 1);
  float keyMult   = shiftMult*altMult;//*ctrlMult;

  if(!ImGui::GetIO().WantCaptureKeyboard)
    {
      if(!mParams.running)
        {
          if(key == GLFW_KEY_DOWN  && anyPress) { mSingleStepMult = -shiftMult*ctrlMult*altMult; }
          if(key == GLFW_KEY_UP    && anyPress) { mSingleStepMult =  shiftMult*ctrlMult*altMult; }
        }
  
      // LEFT/RIGHT --> adjust active parameters
      int signMult = 0;
      if(key == GLFW_KEY_LEFT  && anyPress) { signMult = -1; }
      if(key == GLFW_KEY_RIGHT && anyPress) { signMult =  1; }
      if(signMult != 0)
        {
          Vec2i  fSizeStep1  = 64;  Vec2i  fSizeStep2  = 256;
          Vec2i  tSizeStep1  = 64;  Vec2i  tSizeStep2  = 256;
          double dtStep1     = 0.1; double dtStep2     = 1.0;
          
          for(auto iter : mKeysDown)
            {
              if(iter.second)
                {
                  switch(iter.first)
                    {
                    case GLFW_KEY_T: // T -- adjust timestep
                      mParams.dt    += signMult*keyMult*(ctrl ? dtStep2 : dtStep1);
                      std::cout << "TIMESTEP:            " << mParams.dt << "\n";
                      break;
                    }
                }
            }
        }

      if(press)
        {
          if(key == GLFW_KEY_SPACE)       // SPACE -- play/pause physics
            { togglePause(); }
          else if(key == GLFW_KEY_ESCAPE) // CTRL+ESCAPE -- quit
            {
              if(ctrl) { quit(); }
              else     { resetViews(); }
            } 
          else if(shift && alt && key == GLFW_KEY_D) // Alt+Shift+D -- toggle ImGui demo
            {
              mImGuiDemo = !mImGuiDemo;
              std::cout << "IMGUI DEMO " << (mImGuiDemo ? "ENABLED" : "DISABLED") << "\n";
            }
          else if(alt && key == GLFW_KEY_D)          // Alt+D -- toggle debug mode (overlay)
            {
              mParams.debug = !mParams.debug;
              std::cout << "DEBUG MODE " << (mParams.debug ? "ENABLED" : "DISABLED") << "\n";
            }
        }
    }
  
  // always process function keys (no mixups)
  if(anyPress && key == GLFW_KEY_F1)     // F1   -- apply sim adjustments
    {
      // mParams.adjApplyCount *= keyMult;
      //fieldFillAdjust(*mFluid1, &mParams);
      // mParams.adjApplyCount = std::max(1.0f, std::ceil(mParams.adjApplyCount/keyMult));
    }
  else if(press && key == GLFW_KEY_F5)             // F5   -- restart full simulation from initial conditions
    { resetSim(); }
  else if(press && key == GLFW_KEY_F9)             // F9   -- clear EM signals
    { resetSignals(); }
  else if(press && key == GLFW_KEY_F10)            // F10  -- clear EM materials
    { resetMaterials(); }

}

template<typename T >
static bool drawSignalPenUI(SettingBase *setting, SignalPen<typename Dims<T>::BASE> &p, int idx, bool busy, bool &changed)
{
  Vec2f p0 = ImGui::GetCursorPos();
  ImGui::PushID(("##pen"+std::to_string(idx)).c_str());
  ImGui::SetNextItemWidth(111);
  setting->onDraw(1.0f, busy, changed, true); // color picker
  
  ImGui::SetCursorPos(Vec2f(p0.x, ImGui::GetCursorPos().y));
  ImGui::TextUnformatted(" R ");  ImGui::SameLine(); ImGui::TextUnformatted("R^2"); ImGui::SameLine();
  ImGui::TextUnformatted("sin");  ImGui::SameLine(); ImGui::TextUnformatted("cos"); ImGui::SameLine(); ImGui::TextUnformatted("th ");
  ImGui::SetCursorPos(Vec2f(p0.x, ImGui::GetCursorPos().y));
  bool flag = (p.rMult & idx);
  if(ImGui::Checkbox("##RMult",  &flag)) { if(flag) { p.rMult   |= idx; } else { p.rMult   &= ~idx; } changed=true; }
  ImGui::SameLine(); flag = (p.r2Mult & idx);
  if(ImGui::Checkbox("##R2Mult", &flag)) { if(flag) { p.r2Mult  |= idx; } else { p.r2Mult  &= ~idx; } changed=true; }
  ImGui::SameLine(); flag = (p.sinMult & idx);
  if(ImGui::Checkbox("##SimMult",&flag)) { if(flag) { p.sinMult |= idx; } else { p.sinMult &= ~idx; } changed=true; }
  ImGui::SameLine(); flag = (p.cosMult & idx);
  if(ImGui::Checkbox("##CosMult",&flag)) { if(flag) { p.cosMult |= idx; } else { p.cosMult &= ~idx; } changed=true; }
  ImGui::SameLine(); flag = (p.tMult & idx);
  if(ImGui::Checkbox("##TMult",  &flag)) { if(flag) { p.tMult   |= idx; } else { p.tMult   &= ~idx; } changed=true; }
  ImGui::PopID();
  return busy;
}



SimWindow::SimWindow(GLFWwindow *window)
  : mWindow(window)
{
  simWin = this; // NOTE/TODO: only one window total allowed for now
  glfwSetKeyCallback(mWindow, &keyCallback); // key event callback

  SettingGroup *simGroup    = new SettingGroup("Simulation",    "sim",    { }, true);
  SettingGroup *matGroup    = new SettingGroup("Material",      "mat",    { }, true);
  SettingGroup *initGroup   = new SettingGroup("Initial State", "init",   { }, true);
  SettingGroup *renderGroup = new SettingGroup("Rendering",     "render", { }, true);
  SettingGroup *vecGroup      = new SettingGroup("Vector Field", "vecField", { }, true);
  SettingGroup *interactGroup = new SettingGroup("Interaction",  "interact", { }, true);
  SettingGroup *flagGroup     = new SettingGroup("Flags",         "flags",   { }, true);

  
  // simulation

  auto *sFRES = new Setting<int3> ("Field Resolution", "fieldRes", &mParams.fieldRes);
  sFRES->setFormat(int3{32, 32, 32}, int3{64, 64, 64}, "%4f");
  sFRES->setMin(int3{1, 1, 1});
  sFRES->updateCallback = [sFRES, this]() 
                          {
                            cudaDeviceSynchronize();
                            std::cout << "NEW FIELD SIZE --> " << mParams.fieldRes << "\n";
                            resizeFields(mParams.fieldRes);
                            resetViews();
                          };
  mSettings.push_back(sFRES); simGroup->add(sFRES);
  auto *sTRES = new Setting<int2> ("Tex Resolution",   "texRes",   &mParams.texRes,   mParams.texRes,
                                   [&]()
                                   {
                                     cudaDeviceSynchronize();
                                     int3 ts = int3{mParams.texRes.x, mParams.texRes.y, 1};
                                     std::cout << "RESIZING TEXTURE --> " << ts << "\n";
                                     mEMTex.create(ts); mMatTex.create(ts); m3DTex.create(ts);
                                     cudaDeviceSynchronize();
                                   });
  sTRES->setMin(int2{1,1});
  sTRES->setMax(int2{2048,2048});
  mSettings.push_back(sTRES); simGroup->add(sTRES);
  
  auto *sDT = new Setting<float> ("dt", "dt", &mParams.dt);
  sDT->setFormat(0.01f, 0.1f, "%0.12f");
  mSettings.push_back(sDT); simGroup->add(sDT);
  
  auto *sFPP = new Setting<float3> ("Field Position", "fPos", &mParams.fieldPos);
  mSettings.push_back(sFPP); simGroup->add(sFPP);
  auto *sCS = new Setting<float3> ("Cell Size", "cs", &mParams.cp.cs);
  sCS->setFormat(float3{0.01f, 0.01f, 0.01f}, float3{0.1f, 0.1f, 0.1f}, "%.4f");
  sCS->setMin(float3{1.0f, 1.0f, 1.0f}*1e-35f);
  mSettings.push_back(sCS); simGroup->add(sCS);
  
  auto *sREF = new Setting<bool> ("Reflective Edges", "reflect", &mParams.cp.boundReflect);
  mSettings.push_back(sREF); simGroup->add(sREF);
  auto *sBS = new Setting<float3> ("Border Size",      "bs", &mParams.cp.bs);
  mSettings.push_back(sBS); simGroup->add(sBS);
  sBS->setMin(float3{0.0f, 0.0f, 0.0f});

  auto *sPE = new Setting<float> ("Permittivity", "permittivity", &mParams.cp.material.permittivity);
  sPE->setFormat(0.01f, 0.1f, "%0.6f");
  mSettings.push_back(sPE); matGroup->add(sPE);
  auto *sPB = new Setting<float> ("Permeability", "permeability", &mParams.cp.material.permeability);
  sPB->setFormat(0.01f, 0.1f, "%0.6f");
  mSettings.push_back(sPB); matGroup->add(sPB);
  auto *sPC = new Setting<float> ("Conductivity", "conductivity", &mParams.cp.material.conductivity);
  sPC->setFormat(0.01f, 0.1f, "%0.6f");
  mSettings.push_back(sPC); matGroup->add(sPC);

  // init
  auto *sQPINIT  = new Setting<std::string>("q+ init", "qpInit",   &mParams.initQPStr, mParams.initQPStr,
                                            [&]() {  if(mFillQP)  { if(mFillQP)  { cudaFree(mFillQP);  mFillQP  = nullptr; } } });
  mSettings.push_back(sQPINIT); initGroup->add(sQPINIT);
  auto *sQNINIT  = new Setting<std::string>("q- init", "qnInit",   &mParams.initQNStr, mParams.initQPStr,
                                            [&]() {  if(mFillQN)  { if(mFillQN)  { cudaFree(mFillQN);  mFillQN  = nullptr; } } });
  mSettings.push_back(sQNINIT); initGroup->add(sQNINIT);
  auto *sQPVINIT = new Setting<std::string>("Vq+ init", "qpvInit", &mParams.initQPVStr, mParams.initQPStr,
                                            [&]() {  if(mFillQPV) { if(mFillQPV) { cudaFree(mFillQNV); mFillQPV = nullptr; } } });
  mSettings.push_back(sQPVINIT); initGroup->add(sQPVINIT);
  auto *sQNVINIT = new Setting<std::string>("Vq- init", "qnvInit", &mParams.initQNVStr, mParams.initQPStr,
                                            [&]() {  if(mFillQNV) { if(mFillQNV) { cudaFree(mFillQNV); mFillQNV = nullptr; } } });
  mSettings.push_back(sQNVINIT); initGroup->add(sQNVINIT);
  auto *sEINIT   = new Setting<std::string>("E init",   "EInit",   &mParams.initEStr, mParams.initEStr,
                                            [&]() {  if(mFillE)   { if(mFillE)   { cudaFree(mFillE);   mFillE   = nullptr; } } });
  mSettings.push_back(sEINIT); initGroup->add(sEINIT);
  auto *sBINIT   = new Setting<std::string>("B init",   "BInit",   &mParams.initBStr, mParams.initBStr,
                                            [&]() {  if(mFillB)   { if(mFillB)   { cudaFree(mFillB);   mFillB   = nullptr; } } });
  mSettings.push_back(sBINIT); initGroup->add(sBINIT);

  // interaction
  auto *sSPA   = new Setting<bool> ("Active",  "sPenActive",      &mParams.signalPen.active);
  mSettings.push_back(sSPA);   interactGroup->add(sSPA);
  auto *sSPR   = new Setting<float> ("Radius", "sPenRad",         &mParams.signalPen.radius);
  sSPR->setFormat(1.0f, 10.0f, "%0.4f");
  mSettings.push_back(sSPR); interactGroup->add(sSPR);
  auto *sSPS   = new Setting<bool> ("Square",  "sPenSquare",      &mParams.signalPen.square);
  mSettings.push_back(sSPS); interactGroup->add(sSPS);
  auto *sSPAL  = new Setting<bool> ("Align to Cell", "sPenAlign", &mParams.signalPen.cellAlign);
  mSettings.push_back(sSPAL); interactGroup->add(sSPAL);
  
  auto *sSPDLZ = new Setting<int> ("Z Layer", "zLayer",           &mParams.zLayer2D, mParams.zLayer2D,
                                   [&]()
                                   {
                                     if(mParams.zLayer2D >= mParams.fieldRes.z) { mParams.zLayer2D = mParams.fieldRes.z - 1; }
                                     else if(mParams.zLayer2D < 0)              { mParams.zLayer2D = 0; }
                                   });
  mSettings.push_back(sSPDLZ); interactGroup->add(sSPDLZ);
  
  auto *sSPF   = new Setting<float> ("Frequency", "sPenFreq",     &mParams.signalPen.frequency);
  sSPF->setFormat(0.1f, 1.0f, "%0.4f"); sSPF->setMin(0.0f);
  mSettings.push_back(sSPF); interactGroup->add(sSPF);
  auto *sSPM   = new Setting<float> ("Mult",      "sPenMult",     &mParams.signalPen.mult);
  sSPM->setFormat(0.1f, 1.0f, "%0.4f");
  mSettings.push_back(sSPM); interactGroup->add(sSPM);
  
  auto *sSPQ   = new Setting<float2>("Q",   "sPenQ",   &mParams.signalPen.Q);
  sSPQ->labels = {"(+)", "(-)"};
  sSPQ->setFormat  (float2{0.01f, 0.01f}, float2{0.1f, 0.1f}, "%0.4f");
  auto *sSPQPV = new Setting<float3>("VQ+", "sPenQpv", &mParams.signalPen.QPV);
  sSPQPV->setFormat(float3{0.01f, 0.01f, 0.01f},  float3{0.1f, 0.1f, 0.1f}, "%0.4f");
  auto *sSPQNV = new Setting<float3>("VQ-", "sPenQnv", &mParams.signalPen.QNV);
  sSPQNV->setFormat(float3{0.01f, 0.01f, 0.01f},  float3{0.1f, 0.1f, 0.1f}, "%0.4f");
  auto *sSPE   = new Setting<float3>("E",   "sPenE",   &mParams.signalPen.E);
  sSPE->setFormat  (float3{0.01f, 0.01f, 0.01f}, float3{0.1f, 0.1f, 0.1f}, "%0.4f");
  auto *sSPB   = new Setting<float3>("B",   "sPenB",   &mParams.signalPen.B);
  sSPB->setFormat  (float3{0.01f, 0.01f, 0.01f}, float3{0.1f, 0.1f, 0.1f}, "%0.4f");
  mSettings.push_back(sSPQ);   interactGroup->add(sSPQ);
  mSettings.push_back(sSPQPV); interactGroup->add(sSPQPV);
  mSettings.push_back(sSPQNV); interactGroup->add(sSPQNV);
  mSettings.push_back(sSPE);   interactGroup->add(sSPE);
  mSettings.push_back(sSPB);   interactGroup->add(sSPB);
  
  auto *sSPQb    = new Setting<std::vector<bool>>("Options", "sPenQopt", &Qopt);
  sSPQb->vColumns = 5; sSPQb->vRowLabels = {{0, "Q  "}};
  sSPQb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}}; sSPQb->drawColLabels = true;
  sSPQb->vRowLabels = {{0, "Q  "}};
  sSPQb->updateCallback   = [this]() {
                              int idx = IDX_NONE; for(int i = 0; i < Qopt.size(); i++)   { idx = (Qopt[i]   ? (idx | (1<<i)) : (idx & ~(1<<i))); }
                              mParams.signalPen.Qopt = idx; };
  auto *sSPQPVb  = new Setting<std::vector<bool>>("", "sPenQPVopt", &QPVopt);
  sSPQPVb->vColumns = 5;  sSPQPVb->vRowLabels = {{0, "QPV"}};
  sSPQPVb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}}; 
  sSPQPVb->updateCallback = [this]() {
                              int idx = IDX_NONE; for(int i = 0; i < QPVopt.size(); i++) { idx = (QPVopt[i] ? (idx | (1<<i)) : (idx & ~(1<<i))); }
                              mParams.signalPen.QPVopt = idx; };
  auto *sSPQNVb  = new Setting<std::vector<bool>>("", "sPenQNVopt", &QNVopt);
  sSPQNVb->vColumns = 5;  sSPQNVb->vRowLabels = {{0, "QNV"}};
  sSPQNVb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}}; 
  sSPQNVb->updateCallback = [this]() {
                              int idx = IDX_NONE; for(int i = 0; i < QNVopt.size(); i++) { idx = (QNVopt[i] ? (idx | (1<<i)) : (idx & ~(1<<i))); }
                              mParams.signalPen.QNVopt = idx; };
  auto *sSPEb    = new Setting<std::vector<bool>>("", "sPenEopt", &Eopt);
  sSPEb->vColumns = 5;  sSPEb->vRowLabels = {{0, "E  "}};
  sSPEb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}}; 
  sSPEb->updateCallback   = [this]() {
                              int idx = IDX_NONE; for(int i = 0; i < Eopt.size(); i++)   { idx = (Eopt[i]   ? (idx | (1<<i)) : (idx & ~(1<<i))); }
                              mParams.signalPen.Eopt = idx; };
  auto *sSPBb    = new Setting<std::vector<bool>>("", "sPenBopt", &Bopt);
  sSPBb->vColumns = 5;  sSPBb->vRowLabels = {{0, "B  "}};
  sSPBb->vColLabels = {{0, "   R "}, {1, "  R^2"}, {2, "   θ "}, {3, "sin(t)"}, {4, "cos(t)"}}; 
  sSPBb->updateCallback   = [this]() {
                              int idx = IDX_NONE; for(int i = 0; i < Bopt.size(); i++)   { idx = (Bopt[i]   ? (idx | (1<<i)) : (idx & ~(1<<i))); }
                              mParams.signalPen.Bopt = idx; };
  mSettings.push_back(sSPQb);   interactGroup->add(sSPQb);
  mSettings.push_back(sSPQPVb); interactGroup->add(sSPQPVb);
  mSettings.push_back(sSPQNVb); interactGroup->add(sSPQNVb);
  mSettings.push_back(sSPEb);   interactGroup->add(sSPEb);
  mSettings.push_back(sSPBb);   interactGroup->add(sSPBb);

  
  // material
  auto *sMPA  = new Setting<bool> ("Active",                 "mPenActive",  &mParams.materialPen.active);
  mSettings.push_back(sMPA);  interactGroup->add(sMPA);
  auto *sMPV  = new Setting<bool> ("Vacuum (eraser)",        "mPenVacuum",  &mParams.materialPen.vacuum);
  mSettings.push_back(sMPV);  interactGroup->add(sMPV);
  auto *sMPR  = new Setting<float> ("Radius",                "mPenRad",     &mParams.materialPen.radius);
  sMPR->setFormat(0.1f, 1.0f, "%0.4f");
  mSettings.push_back(sMPR);  interactGroup->add(sMPR);
  auto *sMM   = new Setting<float> ("Mult",                  "mPenMult",    &mParams.materialPen.mult);
  sMPR->setFormat(0.1f, 1.0f, "%0.4f");
  mSettings.push_back(sMM);   interactGroup->add(sMM);
  auto *MPERM = new Setting<float> ("Permittivity (e/e0)",   "mPenEpsilon", &mParams.materialPen.permittivity); 
  MPERM->setFormat(0.1f, 1.0f, "%0.4f");
  mSettings.push_back(MPERM); interactGroup->add(MPERM);
  auto *sMPMT = new Setting<float> ("Permittivity (u/u0)",   "mPenMu",      &mParams.materialPen.permeability);
  sMPMT->setFormat(0.1f, 1.0f, "%0.4f");
  mSettings.push_back(sMPMT); interactGroup->add(sMPMT);
  auto *sMC   = new Setting<float> ("Conductivity (sigma)",  "mPenSigma",   &mParams.materialPen.conductivity);
  sMC->setFormat(0.1f, 1.0f, "%0.4f");
  mSettings.push_back(sMC);   interactGroup->add(sMC);
  
  // vector draw
  auto *sVF = new Setting<bool>("Draw Vectors", "drawVec",     &mParams.drawVectors);
  mSettings.push_back(sVF); vecGroup->add(sVF);
  auto *sFV = new Setting<bool>("Bordered",     "vecBordered", &mParams.borderedVectors);
  mSettings.push_back(sFV); vecGroup->add(sFV);
  auto *sVI = new Setting<bool>("Smooth",       "vecSmooth",   &mParams.smoothVectors);
  mSettings.push_back(sVI); vecGroup->add(sVI);
  auto *sVMR = new Setting<int>("Radius",       "vecMRad",     &mParams.vecMRadius);
  sVMR->setMin(0);
  mSettings.push_back(sVMR); vecGroup->add(sVMR);
  auto *sVSP = new Setting<int>("Spacing",      "vecSpacing",  &mParams.vecSpacing);
  sVSP->setMin(1);
  mSettings.push_back(sVSP); vecGroup->add(sVSP);
  auto *sVCR = new Setting<int>("Max Count",    "vecCRad",     &mParams.vecCRadius);
  mSettings.push_back(sVCR); vecGroup->add(sVCR);
  
  auto *sVLME = new Setting<float>("E Length", "vecMultE",  &mParams.vecMultE);
  sVLME->setFormat(0.1f, 1.0f, "%0.4f");
  mSettings.push_back(sVLME); vecGroup->add(sVLME);
  auto *sVLMB = new Setting<float>("B Length", "vecMultB",  &mParams.vecMultB);
  sVLMB->setFormat(0.1f, 1.0f, "%0.4f");
  mSettings.push_back(sVLMB); vecGroup->add(sVLMB);
  auto *sVLW = new Setting<float>("Line Width", "vWidth",   &mParams.vecLineW);
  sVLW->setFormat(0.1f, 1.0f, "%0.4f");
  mSettings.push_back(sVLW); vecGroup->add(sVLW);
  auto *sVLA = new Setting<float>("Line Alpha", "vecAlpha", &mParams.vecAlpha);
  sVLA->setFormat(0.01f, 0.1f, "%0.4f"); sVLA->setMin(0.0f); sVLA->setMax(1.0f);
  mSettings.push_back(sVLA); vecGroup->add(sVLA);

  if(mParams.borderedVectors)
    {
      auto *sVBW = new Setting<float>("Border Width", "bWidth",  &mParams.vecBorderW);
      sVBW->setFormat(0.1f, 1.0f, "%0.4f");
      mSettings.push_back(sVBW); vecGroup->add(sVBW);  
      auto *sVBA = new Setting<float>("Border Alpha", "vecBAlpha",  &mParams.vecBAlpha);
      sVBA->setFormat(0.01f, 0.1f, "%0.4f"); sVBA->setMin(0.0f); sVBA->setMax(1.0f);
      mSettings.push_back(sVBA); vecGroup->add(sVBA);
    }
  
  // render
  auto *sRCO = new Setting<float> ("3D Opacity", "opacity", &mParams.rp.opacity);
  sRCO->setFormat(0.01f, 0.1f, "%0.4f");
  mSettings.push_back(sRCO); renderGroup->add(sRCO);
  auto *sRCBR = new Setting<float> ("3D Brightness", "brightness", &mParams.rp.brightness);
  sRCBR->setFormat(0.01f, 0.1f, "%0.4f");
  mSettings.push_back(sRCBR); renderGroup->add(sRCBR);
  auto *sRCQ = new Setting<float4> ("Q Color", "renderColQ", &mParams.rp.Qcol);
  sRCQ->drawCustom = [sRCQ, this](bool busy, bool &changed) -> bool
                     {
                       sRCQ->onDraw(1.0f, busy, changed, true); // color picker
                       ImGui::SameLine(); ImGui::SetNextItemWidth(150);
                       changed |= ImGui::InputFloat("##qMult", &mParams.rp.Qmult, 0.01f, 0.1f, "%.8f");
                       return busy;
                     };
  sRCQ->setFormat(float4{0.01f, 0.01f, 0.01f, 0.01f}, float4{0.1f, 0.1f, 0.1f, 0.1f}, "%0.8f");  
  mSettings.push_back(sRCQ); renderGroup->add(sRCQ);
  auto *sRCE = new Setting<float4> ("E Color", "renderColE", &mParams.rp.Ecol);
  sRCE->drawCustom = [sRCE, this](bool busy, bool &changed) -> bool
                     {
                       sRCE->onDraw(1.0f, busy, changed, true); // color picker
                       ImGui::SameLine(); ImGui::SetNextItemWidth(150);
                       changed |= ImGui::InputFloat("##eMult", &mParams.rp.Emult, 0.01f, 0.1f, "%.8f");
                       return busy;
                     };
  sRCE->setFormat(float4{0.01f, 0.01f, 0.01f, 0.1f}, float4{0.1f, 0.1f, 0.1f, 0.1f}, "%0.8f");
  mSettings.push_back(sRCE); renderGroup->add(sRCE);
  auto *sRCB = new Setting<float4> ("B Color", "renderColB", &mParams.rp.Bcol);
  sRCB->drawCustom = [sRCB, this](bool busy, bool &changed) -> bool
                     {
                       sRCB->onDraw(1.0f, busy, changed, true); // color picker
                       ImGui::SameLine(); ImGui::SetNextItemWidth(150);
                       changed |= ImGui::InputFloat("##bMult", &mParams.rp.Bmult, 0.01f, 0.1f, "%.8f");
                       return busy;
                     };
  sRCB->setFormat(float4{0.01f, 0.01f, 0.01f, 0.1f}, float4{0.1f, 0.1f, 0.1f, 0.1f}, "%0.8f");
  mSettings.push_back(sRCB); renderGroup->add(sRCB);
  auto *sRL = new Setting<int> ("2D Layers", "layers2D", &mParams.rp.numLayers2D, mParams.rp.numLayers2D,
                                [this](){ if(mParams.rp.numLayers2D > mParams.fieldRes.z) { mParams.rp.numLayers2D = mParams.fieldRes.z; } });
  sRL->setFormat(1, 8, ""); sRL->setMin(0); 
  mSettings.push_back(sRL); renderGroup->add(sRL);
  
  
  // flags
  auto *sSFEM  = new Setting<bool> ("show EM field",       "showEMField",  &mParams.showEMField);
  mSettings.push_back(sSFEM); flagGroup->add(sSFEM);
  auto *sSFMAT  = new Setting<bool>("show Material field", "showMatField", &mParams.showMatField);
  mSettings.push_back(sSFMAT); flagGroup->add(sSFMAT);
  auto *sSF3D  = new Setting<bool> ("show 3D field",       "show3DField",  &mParams.show3DField);
  mSettings.push_back(sSF3D); flagGroup->add(sSF3D);
  auto *sSFA   = new Setting<bool> ("show axes",           "showAxes",     &mParams.drawAxes);
  mSettings.push_back(sSFA); flagGroup->add(sSFA);
  auto *sRUN   = new Setting<bool> ("running",      "running", &mParams.running);
  mSettings.push_back(sRUN); flagGroup->add(sRUN);
  auto *sDBG   = new Setting<bool> ("debug",        "debug",   &mParams.debug);
  mSettings.push_back(sDBG); flagGroup->add(sDBG);
  auto *sVERB  = new Setting<bool> ("verbose",      "verbose", &mParams.verbose);
  mSettings.push_back(sVERB); flagGroup->add(sVERB);
  auto *sVSYNC = new Setting<bool> ("vSync",        "vsync",   &mParams.vsync, mParams.vsync, [&]() { glfwSwapInterval(mParams.vsync ? 1 : 0); });
  mSettings.push_back(sVSYNC); flagGroup->add(sVSYNC);



  // old settings (ad hoc)
  mSettingFormOld = new SettingForm("Settings", SETTINGS_LABEL_COL_W, SETTINGS_INPUT_COL_W);
  mSettingsSize.x = mSettingFormOld->labelColWidth() + mSettingFormOld->inputColWidth();
  mSettingFormOld->add(simGroup);
  mSettingFormOld->add(matGroup);
  mSettingFormOld->add(initGroup);
  mSettingFormOld->add(renderGroup);
  mSettingFormOld->add(vecGroup);
  mSettingFormOld->add(interactGroup);
  mSettingFormOld->add(flagGroup);
}

SimWindow::~SimWindow()
{
  cleanup();
  if(mSettingFormOld) { delete mSettingFormOld; mSettingFormOld = nullptr; }
}

bool SimWindow::init()    //const SimParams<float2> &p)
{
  if(!mInitialized)
    {
      std::cout << "Creating SimWindow...\n";
      
      //// set up fonts
      ImGuiIO &io = ImGui::GetIO();
      fontConfig = new ImFontConfig();
      fontConfig->OversampleH = FONT_OVERSAMPLE;
      fontConfig->OversampleV = FONT_OVERSAMPLE;
      
      ImVector<ImWchar> ranges;
      ImFontGlyphRangesBuilder builder;
      builder.AddText("ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσΤτΥυΦφΧχΨψΩω");
      builder.AddRanges(io.Fonts->GetGlyphRangesDefault());  // Add one of the default ranges
      builder.AddChar(0x2207);                               // Add a specific character
      builder.BuildRanges(&ranges);                          // Build the final result (ordered ranges with all the unique characters submitted)
      fontConfig->GlyphRanges = ranges.Data;
      
      mainFont   = io.Fonts->AddFontFromFileTTF(FONT_PATH_REGULAR,      MAIN_FONT_HEIGHT,  fontConfig);
      mainFontB  = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD,         MAIN_FONT_HEIGHT,  fontConfig);
      mainFontI  = io.Fonts->AddFontFromFileTTF(FONT_PATH_ITALIC,       MAIN_FONT_HEIGHT,  fontConfig);
      mainFontBI = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD_ITALIC,  MAIN_FONT_HEIGHT,  fontConfig);
      fontConfig->SizePixels  = SMALL_FONT_HEIGHT+1.0f; // small
      smallFont   = io.Fonts->AddFontFromFileTTF(FONT_PATH_REGULAR,     SMALL_FONT_HEIGHT, fontConfig);
      smallFontB  = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD,        SMALL_FONT_HEIGHT, fontConfig);
      smallFontI  = io.Fonts->AddFontFromFileTTF(FONT_PATH_ITALIC,      SMALL_FONT_HEIGHT, fontConfig);
      smallFontBI = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD_ITALIC, SMALL_FONT_HEIGHT, fontConfig);
      fontConfig->SizePixels  = TITLE_FONT_HEIGHT+1.0f; // title
      titleFont   = io.Fonts->AddFontFromFileTTF(FONT_PATH_REGULAR,     TITLE_FONT_HEIGHT, fontConfig);
      titleFontB  = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD,        TITLE_FONT_HEIGHT, fontConfig);
      titleFontI  = io.Fonts->AddFontFromFileTTF(FONT_PATH_ITALIC,      TITLE_FONT_HEIGHT, fontConfig);
      titleFontBI = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD_ITALIC, TITLE_FONT_HEIGHT, fontConfig);
      fontConfig->SizePixels  = SUPER_FONT_HEIGHT+1.0f; // superscript
      superFont   = io.Fonts->AddFontFromFileTTF(FONT_PATH_REGULAR,     SUPER_FONT_HEIGHT, fontConfig);
      superFontB  = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD,        SUPER_FONT_HEIGHT, fontConfig);
      superFontI  = io.Fonts->AddFontFromFileTTF(FONT_PATH_ITALIC,      SUPER_FONT_HEIGHT, fontConfig);
      superFontBI = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD_ITALIC, SUPER_FONT_HEIGHT, fontConfig);
      io.Fonts->Build();

      mUnitsForm = new UnitsInterface<CFT>(&mUnits, superFont);
  
      mTabs = new TabMenu(20, 1080, true);
      mTabs->setCollapsible(true);
      mTabs->add(TabDesc{"Units",   "Unit Manager", [this](){ mUnitsForm->draw(); },
                         (int)SETTINGS_TOTAL_W, (int)SETTINGS_TOTAL_W+40, (int)SETTINGS_TOTAL_W+40, titleFont});
      mTabs->add(TabDesc{"Forces",  "Forces",       [this](){  },
                         (int)SETTINGS_TOTAL_W, (int)SETTINGS_TOTAL_W+40, (int)SETTINGS_TOTAL_W+40, titleFont});
      mTabs->add(TabDesc{"Display", "Display",      [this](){  },
                         (int)SETTINGS_TOTAL_W, (int)SETTINGS_TOTAL_W+40, (int)SETTINGS_TOTAL_W+40, titleFont});
      mTabs->add(TabDesc{"(Old)",   "Old Settings", [this](){ mSettingFormOld->draw(); },
                         (int)SETTINGS_TOTAL_W, (int)SETTINGS_TOTAL_W+40, (int)SETTINGS_TOTAL_W+40, titleFont});
      mTabs->select(3);
      
      std::cout << "Creating CUDA objects...\n";
      // initialize CUDA and check for a compatible device
      if(!initCudaDevice())
        {
          std::cout << "====> ERROR: failed to initialize CUDA device!\n";
          delete fontConfig; fontConfig = nullptr;
          return false;
        }

      mParams.initQPExpr  = toExpression<float >(mParams.initQPStr);
      mParams.initQNExpr  = toExpression<float >(mParams.initQNStr);
      mParams.initQPVExpr = toExpression<float3>(mParams.initQPVStr);
      mParams.initQNVExpr = toExpression<float3>(mParams.initQNVStr);
      mParams.initEExpr   = toExpression<float3>(mParams.initEStr);
      mParams.initBExpr   = toExpression<float3>(mParams.initBStr);

      //// set up CUDA fields
      // create state queue
      if(mParams.fieldRes.x > 0 && mParams.fieldRes.y > 0 && mParams.fieldRes.z > 0)
        {
          for(int i = 0; i < STATE_BUFFER_SIZE; i++)
            { mStates.push_back(new ChargeField<CFT>()); }
          resizeFields(mParams.fieldRes);
        }
      
      int3 ts = int3{mParams.texRes.x, mParams.texRes.y, 1};
      std::cout << "Creating textures (" << ts << ")\n";
      if(!mEMTex.create(ts))  { std::cout << "====> ERROR: Texture creation for EM view failed!\n";  }
      if(!mMatTex.create(ts)) { std::cout << "====> ERROR: Texture creation for Mat view failed!\n"; }
      if(!m3DTex.create(ts))  { std::cout << "====> ERROR: Texture creation for 3D view failed!\n";  }
      resetViews();
      
      //// set up base path for rendering
      checkBaseRenderPath();
      mInitialized = true;
    }
  return true;
}

void SimWindow::cleanup()
{
  if(mInitialized)
    {
      cudaDeviceSynchronize();
      std::cout << "Destroying CUDA field states...\n";
      std::vector<FieldBase*> deleted;
      for(int i = 0; i < mStates.size(); i++)
        {
          auto f = mStates[i];
          if(f)
            {
              std::cout << "    --> " << i << "\n";
              auto iter = std::find(deleted.begin(), deleted.end(), f);
              if(iter != deleted.end()) { std::cout << "====> WARNING: State already deleted! (" << i << ")\n"; continue; }
              
              f->destroy();
              delete f;
              deleted.push_back(f);
            }
        }
      mStates.clear();
      std::cout << "Destroying CUDA textures...\n";
      mEMTex.destroy(); mMatTex.destroy(); m3DTex.destroy();
      
      std::cout << "Destroying fonts...\n";
      if(fontConfig) { delete fontConfig; }
      
      if(mTabs) { delete mTabs; mTabs = nullptr; }
      
      std::cout << "Cleaning GL...\n";
      cleanupGL();

      mInitialized = false;
    }
}

bool SimWindow::checkBaseRenderPath()
{
  // check base render directory
  if(!directoryExists(mBaseDir))
    {
      std::cout << "Creating base directory for rendered simulations (" << mBaseDir << ")...\n";
      if(makeDirectory(mBaseDir)) { std::cout << "Successfully created directory.\n"; }
      else                        { std::cout << "====> ERROR: Could not create base render directory.\n";  return false; }
    }
  return true;
}

bool SimWindow::checkSimRenderPath()
{
  if(!checkBaseRenderPath()) { return false; } // make sure base directory exists
  else
    {
      // check simulation image directory
      mImageDir = mBaseDir + "/" + mParams.simName;
      if(!directoryExists(mImageDir))
        {
          std::cout << "Creating directory for simulation '" << mParams.simName << "' --> (" << mImageDir << ")...\n";
          if(makeDirectory(mImageDir)) { std::cout << "Successfully created directory.\n"; }
          else                         { std::cout << "====> ERROR: Could not make sim image directory.\n"; return false; }
        }
    }
  return true;
}











bool SimWindow::resizeFields(const Vec3i &sz)
{
  if(min(sz) <= 0) { std::cout << "====> ERROR: Field with zero size not allowed.\n"; return false; }
  bool success = true;
  for(int i = 0; i < STATE_BUFFER_SIZE; i++)
    {
      ChargeField<CFT> *f = reinterpret_cast<ChargeField<CFT>*>(mStates[i]);
      if(!f->create(mParams.fieldRes)) { std::cout << "Field creation failed! Invalid state.\n"; success = false; break; }
      fieldFillValue(reinterpret_cast<ChargeField<CFT>*>(f)->mat, mParams.cp.material);
    }
  mParams.zLayer2D = mParams.fieldRes.z/2;
  mParams.rp.numLayers2D = mParams.fieldRes.z;
  if(success) { resetSim(); }
  return success;
}



// TODO: improve expression variable framework
template<typename T> std::vector<std::string> getVarNames() { return {"px", "py", "pz", "sx", "sy", "sz", "r", "t"}; }
template<> std::vector<std::string> getVarNames<float>()    { return {"px", "py", "pz", "sx", "sy", "sz", "r", "t"}; }
template<> std::vector<std::string> getVarNames<float3>()   { return {"p", "s", "r", "n", "t"}; }

void SimWindow::resetSignals()
{
  std::cout << "SIGNAL RESET\n";
  if(mParams.verbose)
    {
      std::cout << "QP:  "  << mParams.initQPExpr->toString(true)  << "\n";
      std::cout << "QN:  "  << mParams.initQNExpr->toString(true)  << "\n";
      std::cout << "QPV:  " << mParams.initQPVExpr->toString(true) << "\n";
      std::cout << "QNV:  " << mParams.initQNVExpr->toString(true) << "\n";
      std::cout << "E:  "   << mParams.initEExpr->toString(true)   << "\n";
      std::cout << "B:  "   << mParams.initBExpr->toString(true)   << "\n";
    }

  mParams.initQPExpr  = toExpression<float >(mParams.initQPStr,  false);
  mParams.initQNExpr  = toExpression<float >(mParams.initQNStr,  false);
  mParams.initQPVExpr = toExpression<float3>(mParams.initQPVStr, false);
  mParams.initQNVExpr = toExpression<float3>(mParams.initQNVStr, false);
  mParams.initEExpr   = toExpression<float3>(mParams.initEStr,   false);
  mParams.initBExpr   = toExpression<float3>(mParams.initBStr,   false);
  
  // create/update expressions 
  std::cout << "Filling field states on device...\n";
  if(!mFillQP)  { mFillQP  = toCudaExpression<float> (mParams.initQPExpr,  getVarNames<float>());  std::cout << "  --> QP  EXPRESSION UPDATED\n"; }
  if(!mFillQN)  { mFillQN  = toCudaExpression<float> (mParams.initQNExpr,  getVarNames<float>());  std::cout << "  --> QN  EXPRESSION UPDATED\n"; }
  if(!mFillQPV) { mFillQPV = toCudaExpression<float3>(mParams.initQPVExpr, getVarNames<float3>()); std::cout << "  --> QPV EXPRESSION UPDATED\n"; }
  if(!mFillQNV) { mFillQNV = toCudaExpression<float3>(mParams.initQNVExpr, getVarNames<float3>()); std::cout << "  --> QNV EXPRESSION UPDATED\n"; }
  if(!mFillE)   { mFillE   = toCudaExpression<float3>(mParams.initEExpr,   getVarNames<float3>()); std::cout << "  --> E   EXPRESSION UPDATED\n"; }
  if(!mFillB)   { mFillB   = toCudaExpression<float3>(mParams.initBExpr,   getVarNames<float3>()); std::cout << "  --> B   EXPRESSION UPDATED\n"; }
  // fill all states
  for(int i = 0; i < mStates.size(); i++)
    {
      ChargeField<CFT> *f = reinterpret_cast<ChargeField<CFT>*>(mStates[mStates.size()-1-i]);
      fieldFillChannel<float2>(f->Q,   mFillQP, 0); // q+ --> Q[i].x
      fieldFillChannel<float2>(f->Q,   mFillQN, 1); // q- --> Q[i].y
      fieldFill       <float3>(f->QPV, mFillQPV);
      fieldFill       <float3>(f->QNV, mFillQNV);
      fieldFill       <float3>(f->E,   mFillE);
      fieldFill       <float3>(f->B,   mFillB);
    }
  cudaDeviceSynchronize();
}


void SimWindow::resetMaterials()
{
  std::cout << "MATERIAL RESET\n";
  for(int i = 0; i < mStates.size(); i++)
    { // reset materials in each state
      ChargeField<CFT> *f = reinterpret_cast<ChargeField<CFT>*>(mStates[mStates.size()-1-i]);
      fieldFillValue<Material<CFT>>(*reinterpret_cast<Field<Material<CFT>>*>(&f->mat), mParams.cp.material);
    } 
  cudaDeviceSynchronize();
}

void SimWindow::resetSim()
{
  std::cout << "SIMULATION RESET\n";
  mInfo.t = 0.0f;
  mInfo.frame = 0;
  mInfo.uStep = 0;
  std::cout << " --> "; resetSignals();
  std::cout << " --> "; resetMaterials();
  cudaDeviceSynchronize();
}

void SimWindow::togglePause()
{
  mParams.running = !mParams.running;
  std::cout << (mParams.running ? "STARTED" : "STOPPED") << " SIMULATION.\n";
}

static ChargeField<CFT> *g_temp = nullptr;
// static ChargeField<CFT> *g_temp2 = nullptr;
void SimWindow::update()
{
  if(mParams.fieldRes.x > 0 && mParams.fieldRes.y > 0)
    {
      bool singleStep = false;
      mParams.cp.dt = mParams.dt;
      mParams.cp.t  = mInfo.t;
      ChargeParams cp = mParams.cp;
      if(mSingleStepMult != 0.0f)
        {
          cp.dt *= mSingleStepMult;
          mSingleStepMult = 0.0f;
          singleStep = true;
        }
      
      float3 mposSim = float3{NAN, NAN, NAN};
      if     (mEMView.hovered)  { mposSim = to_cuda(mEMView.mposSim);  }
      else if(mMatView.hovered) { mposSim = to_cuda(mMatView.mposSim); }
      else if(m3DView.hovered)  { mposSim = to_cuda(m3DView.mposSim);  }
      float3 cs = mParams.cp.cs;
      float3 fs = float3{(float)mParams.fieldRes.x, (float)mParams.fieldRes.y, (float)mParams.fieldRes.z};
      float3 mpfi = (mposSim/cs);
      
      ChargeField<CFT> *src = reinterpret_cast<ChargeField<CFT>*>(mStates.back());  // previous field state
      if(!g_temp)                          { g_temp = new ChargeField<CFT>();  }
      if(g_temp->size != mParams.fieldRes) { g_temp->create(mParams.fieldRes); }
      ChargeField<CFT> *dst  = reinterpret_cast<ChargeField<CFT>*>(mStates.front()); // oldest state (recycle)
      ChargeField<CFT> *temp = reinterpret_cast<ChargeField<CFT>*>(g_temp); // temp intermediate state
      
      // draw signal
      if(((mEMView.clicked && mEMView.hovered) || (mMatView.clicked && mMatView.hovered)) && mParams.signalPen.active &&
         (ImGui::IsKeyDown(GLFW_KEY_LEFT_CONTROL) || ImGui::IsKeyDown(GLFW_KEY_RIGHT_CONTROL)) && ImGui::IsMouseDown(ImGuiMouseButton_Left))
        {
          cudaDeviceSynchronize();
          addSignal(float3{mpfi.x, mpfi.y, mpfi.z}, *src, mParams.signalPen, cp);
        }
      
      // add material
      if(((mEMView.clicked && mEMView.hovered) || (mMatView.clicked && mMatView.hovered)) && mParams.materialPen.active &&
         (ImGui::IsKeyDown(GLFW_KEY_LEFT_ALT) || ImGui::IsKeyDown(GLFW_KEY_RIGHT_ALT)))
        {
          cudaDeviceSynchronize();
          addMaterial(float3{mpfi.x, mpfi.y, mpfi.z}, *src, mParams.materialPen, cp);
        }
      
      if(mParams.running || singleStep)
        {
          src->copyTo(*temp);// don't overwrite previous state
          
          if(!mParams.running) { std::cout << "SIM STEP --> dt = " << cp.dt << "\n"; }
          updateElectric(*temp, *dst, cp); std::swap(temp, dst); if(mParams.debug) { cudaDeviceSynchronize(); getLastCudaError("UPDATE ELECTRIC FAILED!"); }
          updateMagnetic(*temp, *dst, cp); std::swap(temp, dst); if(mParams.debug) { cudaDeviceSynchronize(); getLastCudaError("UPDATE MAGNETIC FAILED!"); }
          updateCharge  (*temp, *dst, cp); std::swap(temp, dst); if(mParams.debug) { cudaDeviceSynchronize(); getLastCudaError("UPDATE CHARGE FAILED!"); }
          //cudaDeviceSynchronize(); getLastCudaError("EM update failed!");
          //if(dst == g_temp) { std::swap(dst, g_temp); }

          //dst->copyTo(*temp);
          mStates.pop_front(); mStates.push_back(dst);
          g_temp = temp;
          // increment time/frame info
          mInfo.t += cp.dt;
          mInfo.uStep++;
          if(mInfo.uStep >= mParams.uSteps)
            {
              mInfo.frame++;
              mInfo.uStep = 0;
              mNewFrame = true;
            }
        }
      
      //// render field
      src = reinterpret_cast<ChargeField<CFT>*>(mStates.back());
      // clear texture
      cudaDeviceSynchronize(); mEMTex.clear(); mMatTex.clear(); m3DTex.clear(); cudaDeviceSynchronize();
      // render charge field
      if(mParams.showEMField)  { renderFieldEM  (*src,     mEMTex,  mParams.rp); }
      if(mParams.showMatField) { renderFieldMat (src->mat, mMatTex, mParams.rp); }
      if(mParams.show3DField)  { raytraceFieldEM(*src,     m3DTex,  mCamera, mParams.rp, mParams.cp, m3DView.r.aspect()); }
      cudaDeviceSynchronize(); getLastCudaError("Field render failed! (update)");
    }
}



void SimWindow::resetViews()
{
  float pad = SIM_VIEW_RESET_INTERNAL_PADDING;
  Vec2f cs  = Vec2f(mParams.cp.cs.x, mParams.cp.cs.y);
  Vec2f fs  = Vec2f(mParams.fieldRes.x, mParams.fieldRes.y);
  mSimView2D = Rect2f(fs*cs*Vec2f(-pad, -pad), fs*cs*Vec2f(1.0f+pad,1.0f+pad));
  
  mCamera.fov   = 55.0;
  mCamera.near  = 0.01;
  mCamera.far   = 10000.0;

  double aspect = m3DView.r.size().x/m3DView.r.size().y;
  double zOffset = (max(Vec2d(mParams.fieldRes.x*mParams.cp.cs.x, mParams.fieldRes.y*mParams.cp.cs.y)*Vec2d(aspect, 1.0)) /
                    (2.0*tan((mCamera.fov/2.0)*M_PI/180.0)));
  
  mCamera.pos   = double3{0.0,  0.0, zOffset};
  mCamera.dir   = double3{0.0,  0.0,  -1.0};
  mCamera.right = double3{1.0,  0.0,  0.0};
  mCamera.up    = double3{0.0,  1.0,  0.0};
  std::cout << "RESETTING 3D VIEW CAMERA --> pos=" << mCamera.pos << "; dir=" <<  mCamera.dir << "\n";
}


void SimWindow::handleInput(ScreenView &view)
{
  ImGuiStyle &style = ImGui::GetStyle();
  ImGuiIO    &io    = ImGui::GetIO();

  Vec2f mpos        = ImGui::GetMousePos();
  Vec2f mposSim     = screenToSim2D(mpos, mSimView2D, view.r);
  bool  viewHovered = mSimView2D.contains(mposSim);
  view.hovered = viewHovered;
  if(viewHovered)
    {
      view.hovered = true;
      view.clickPos = mpos;
      view.mposSim = float3{mposSim.x, mposSim.y, (float)mParams.zLayer2D*mParams.cp.cs.z};
    }
  else
    { view.mposSim = float3{NAN, NAN, NAN}; }
  
  if(view.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) { view.clicked = true;  }
  else if(ImGui::IsMouseReleased(ImGuiMouseButton_Left))           { view.clicked = false; }

  if(view.clicked && ImGui::IsMouseDragging(ImGuiMouseButton_Left) &&
     !(ImGui::IsKeyDown(GLFW_KEY_LEFT_CONTROL) || ImGui::IsKeyDown(GLFW_KEY_RIGHT_CONTROL)) && // reserved for signal drawing
     !(ImGui::IsKeyDown(GLFW_KEY_LEFT_ALT)     || ImGui::IsKeyDown(GLFW_KEY_RIGHT_ALT)))       // reserved for material drawing
    {
      Vec2f dmp = Vec2f(ImGui::GetMouseDragDelta(ImGuiMouseButton_Left));
      dmp.x *= -1.0f;
      ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
      mSimView2D.move(screenToSim2D(dmp, mSimView2D, view.r, true)); // recenter mouse at same sim position
    }

  if(view.hovered && std::abs(io.MouseWheel) > 0.0f)
    {
      float vel = (io.KeyAlt ? 1.36 : (io.KeyShift ? 1.011 : 1.055)); // scroll velocity
      float scale = (io.MouseWheel > 0.0f ? 1.0/vel : vel);
      mSimView2D.scale(scale);
      Vec2f mposSim2 = screenToSim2D(mpos, mSimView2D, view.r, false);
      mSimView2D.move(mposSim-mposSim2); // center so mouse doesn't change position
    }
  
  // tooltip 
  if(view.hovered && io.KeyShift)
    {
      float2 fs = float2{(float)mParams.fieldRes.x, (float)mParams.fieldRes.y};
      float2 cs = float2{mParams.cp.cs.x, mParams.cp.cs.y};
      Vec2i fi    = makeV<int2>(floor((float2{mposSim.x, mposSim.y} / cs)));
      Vec2i fiAdj = Vec2i(std::max(0, std::min(mParams.fieldRes.x-1, fi.x)), std::max(0, std::min(mParams.fieldRes.y-1, fi.y)));

      // pull device data
      std::vector<float2> Q  (mStates.size(), float2{NAN, NAN});
      std::vector<float3> QPV(mStates.size(), float3{NAN, NAN, NAN});
      std::vector<float3> QNV(mStates.size(), float3{NAN, NAN, NAN});
      std::vector<float3> E  (mStates.size(), float3{NAN, NAN, NAN});
      std::vector<float3> B  (mStates.size(), float3{NAN, NAN, NAN});
      std::vector<Material<float>> mat(mStates.size(), Material<float>());
      if(fi.x >= 0 && fi.x < mParams.fieldRes.x && fi.y >= 0 && fi.y < mParams.fieldRes.y)
        {
          for(int i = 0; i < mStates.size(); i++)
            {
              ChargeField<CFT> *src = reinterpret_cast<ChargeField<CFT>*>(mStates[mStates.size()-1-i]);
              if(src)
                {
                  cudaMemcpy(&Q[i],   src->Q.dData   + src->Q.idx  (fi.x, fi.y, mParams.zLayer2D), sizeof(float2), cudaMemcpyDeviceToHost);
                  cudaMemcpy(&QPV[i], src->QPV.dData + src->QPV.idx(fi.x, fi.y, mParams.zLayer2D), sizeof(float3), cudaMemcpyDeviceToHost);
                  cudaMemcpy(&QNV[i], src->QNV.dData + src->QNV.idx(fi.x, fi.y, mParams.zLayer2D), sizeof(float3), cudaMemcpyDeviceToHost);
                  cudaMemcpy(&E[i],   src->E.dData   + src->E.idx  (fi.x, fi.y, mParams.zLayer2D), sizeof(float3), cudaMemcpyDeviceToHost);
                  cudaMemcpy(&B[i],   src->B.dData   + src->B.idx  (fi.x, fi.y, mParams.zLayer2D), sizeof(float3), cudaMemcpyDeviceToHost);
                  cudaMemcpy(&mat[i], src->mat.dData + src->mat.idx(fi.x, fi.y, mParams.zLayer2D), sizeof(Material<float>), cudaMemcpyDeviceToHost);
                }
            }
        }
      Vec2f dataPadding = Vec2f(6.0f, 6.0f);
      
      ImGui::BeginTooltip();
      {
        ImDrawList *ttDrawList = ImGui::GetWindowDrawList(); // maximum text size per column
        ImGui::Text(" mousePos: <%.3f, %.3f> (index: <%d, %d>)",  mposSim.x, mposSim.y, fi.x, fi.y);
        
        Vec2f tSize = ImGui::CalcTextSize(("T"+std::to_string(mStates.size()-1)).c_str()); // max state label width
        Vec2f p0 = ImGui::GetCursorScreenPos();
        for(int i = 0; i < mStates.size(); i++)
          {
            // int i = mStates.size() - 1 - i0;
            ImGui::BeginGroup();
            {
              Vec2f sp0 = Vec2f(ImGui::GetCursorScreenPos()) + Vec2f(tSize.x + dataPadding.x, 0);
                
              // draw state label
              std::string labelStr = "T" + std::to_string(i);
              ImGui::TextUnformatted(labelStr.c_str());
              // draw state data
              ImGui::SameLine();
              ImGui::SetCursorScreenPos(sp0 + dataPadding);
              ImGui::BeginGroup();
              {
                float xpos = ImGui::GetCursorPos().x;
                ImGui::Text("(State Pointer: %ld", (long)(void*)(mStates[mStates.size()-1-i]));
                ImGui::Text(" Q   = (+)%s | (-)%s ==> %s", fAlign(Q[i].x,   4).c_str(), fAlign(Q[i].y,   4).c_str(), fAlign((Q[i].x-Q[i].y), 4).c_str());
                ImGui::Spacing();
                ImGui::Text(" VQ+ = < %12s, %12s, %12s >", fAlign(QPV[i].x, 4).c_str(), fAlign(QPV[i].y, 4).c_str(), fAlign(QPV[i].z, 4).c_str());
                ImGui::Text(" VQ- = < %12s, %12s, %12s >", fAlign(QNV[i].x, 4).c_str(), fAlign(QNV[i].y, 4).c_str(), fAlign(QNV[i].z, 4).c_str());
                ImGui::Text(" E   = < %12s, %12s, %12s >", fAlign(E[i].x,   4).c_str(), fAlign(E[i].y,   4).c_str(), fAlign(E[i].z,   4).c_str());
                ImGui::Text(" B   = < %12s, %12s, %12s >", fAlign(B[i].x,   4).c_str(), fAlign(B[i].y,   4).c_str(), fAlign(B[i].z,   4).c_str());
                ImGui::Spacing(); ImGui::Spacing();
                ImGui::Text(" Material %-13s:  ep = %10.4f", (mat[i].vacuum() ? "(vacuum)" : ""), mat[i].permittivity);
                ImGui::Text("          %13s   mu = %10.4f", "", mat[i].permeability);
                ImGui::Text("          %13s  sig = %10.4f", "", mat[i].conductivity);
              }
              ImGui::EndGroup();
              // draw border
              Vec2f sSize = Vec2f(ImGui::GetItemRectMax()) - ImGui::GetItemRectMin() + 2.0f*dataPadding;
              ImGui::PushClipRect(sp0, sp0+sSize, false);
                
              float alpha = mStates.size() == 1 ? 1.0f : (mStates.size()-1-i) / (float)(mStates.size()-1);
              Vec4f color = Vec4f(0.4f+(0.6f*alpha), 0.3f, 0.3f, 1.0f);  // fading color for older states
              ttDrawList->AddRect(sp0, sp0+sSize, ImColor(color), 0, 0, 2.0f);
              ImGui::PopClipRect();
              // underline label
              ttDrawList->AddLine(sp0+Vec2f(-tSize.x-dataPadding.x, tSize.y), sp0+Vec2f(0, tSize.y), ImColor(color), 1.0f);
            }
            ImGui::EndGroup();
            if(i != 0.0f) { ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos()) + Vec2f(0.0f, dataPadding.y)); }
          }
      }
      ImGui::EndTooltip();
    }
}

void SimWindow::handleInput3D(ScreenView &view)
{
  ImGuiStyle &style = ImGui::GetStyle();
  ImGuiIO    &io    = ImGui::GetIO();

  Vec2f mpos = ImGui::GetMousePos();
  view.hovered = view.r.contains(mpos);
  
  if(view.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left))  { view.clicked = true;  }
  else if(ImGui::IsMouseReleased(ImGuiMouseButton_Left))            { view.clicked = false; }
  if(view.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) { view.rightClicked = true;  }
  else if(ImGui::IsMouseReleased(ImGuiMouseButton_Right))           { view.rightClicked = false; }

  double3 upBasis = double3{0.0, 1.0, 0.0};

  double shiftMult = (((ImGui::IsKeyDown(GLFW_KEY_LEFT_SHIFT)   || ImGui::IsKeyDown(GLFW_KEY_RIGHT_SHIFT))   ? 0.1f  : 1.0f));
  double ctrlMult  = (((ImGui::IsKeyDown(GLFW_KEY_LEFT_CONTROL) || ImGui::IsKeyDown(GLFW_KEY_RIGHT_CONTROL)) ? 10.0f : 1.0f));
  double altMult   = (((ImGui::IsKeyDown(GLFW_KEY_LEFT_ALT)     || ImGui::IsKeyDown(GLFW_KEY_RIGHT_ALT))     ? 20.0f : 1.0f));
  double keyMult   = shiftMult * ctrlMult * altMult;
      
  Vec2d  viewSize = m3DView.r.size();
  if(ImGui::IsMouseDragging(ImGuiMouseButton_Right) && view.rightClicked)
    { // rotate camera
      Vec2f dmp = Vec2f(ImGui::GetMouseDragDelta(ImGuiMouseButton_Right));
      ImGui::ResetMouseDragDelta(ImGuiMouseButton_Right);
      dmp = -dmp;
      double2 rAngles  = double2{dmp.x, dmp.y} / double2{viewSize.y, viewSize.y} * 6.0 * tan(mCamera.fov/2*M_PI/180.0) * shiftMult*ctrlMult;
      mCamera.rotate(rAngles);
    }
  if(ImGui::IsMouseDragging(ImGuiMouseButton_Left) && view.clicked)
    { // pan camera
      Vec2d dmp = Vec2d(ImGui::GetMouseDragDelta(ImGuiMouseButton_Left));
      ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
      dmp.x *= -1.0;
      dmp /= viewSize.y;
      //dmp *= Vec2d(mParams.cp.cs.x, mParams.cp.cs.y);
      double mult = length(mCamera.pos)*1.9*tan(mCamera.fov/2.0*M_PI/180.0)*keyMult;
      mCamera.pos += (mCamera.right*dmp.x + mCamera.up*dmp.y)*mult;
    }

  if(view.hovered && std::abs(io.MouseWheel) > 0.0f)
    {
      mCamera.pos += mCamera.dir*keyMult*25.0f*(io.MouseWheel/20.0);
    }
}



void SimWindow::drawVectorField(const Rect2f &sr)
{
  ImGuiStyle &style = ImGui::GetStyle();
  ImGuiIO    &io    = ImGui::GetIO();
  Vec2f mpos = ImGui::GetMousePos();
  Vec2f fp   = screenToSim2D(mpos, mSimView2D, sr);// / float2{(float)mParams.cp.cs.x, (float)mParams.cp.cs.y};
  
  // draw vector field data
  ChargeField<CFT> *src = reinterpret_cast<ChargeField<CFT>*>(mStates.back());
  if(src && (mEMView.hovered || mMatView.hovered) && mParams.drawVectors && mFieldDrawList)
    {
      Vec2i fi    = makeV<int2>(float2{floor(fp.x), floor(fp.y)});
      Vec2f fo    = fp - fi;
      Vec2i fiAdj = Vec2i(std::max(0, std::min(mParams.fieldRes.x, fi.x)), std::max(0, std::min(mParams.fieldRes.y, fi.y)));

      int vRad     = mParams.vecMRadius;
      int cRad     = mParams.vecCRadius;
      int vSpacing = mParams.vecSpacing;
      if(vRad > cRad) { vSpacing = std::max(vSpacing, (int)ceil(vRad/(float)cRad)); }
      float viewScale = max(mSimView2D.size());
      
      int2 iMin = int2{std::max(fi.x-vRad*vSpacing, 0), std::max(fi.y-vRad*vSpacing, 0)};
      int2 iMax = int2{std::min(fi.x+vRad*vSpacing, mParams.fieldRes.x-1)+1,
                       std::min(fi.y+vRad*vSpacing, mParams.fieldRes.y-1)+1};
      
      int2 iStart = int2{0, 0};
      int2 iEnd   = int2{(iMax.x - iMin.x)/vSpacing, (iMax.y - iMin.y)/vSpacing};

      src->E.pullData(); src->B.pullData();
      float avgE = 0.0f; float avgB = 0.0f;
      
      for(int ix = iStart.x; ix <= iEnd.x; ix++)
        for(int iy = iStart.y; iy <= iEnd.y; iy++)
          {
            int xi = iMin.x + ix*vSpacing; int yi = iMin.y + iy*vSpacing;
            float2 dp = float2{(float)(xi-fi.x), (float)(yi-fi.y)};
            if(dot(dp, dp) <= (float)(vRad*vRad))
              {
                int i = src->idx(xi, yi);
                Vec2f sp = simToScreen2D(Vec2f(xi+0.5f, yi+0.5f), mSimView2D, sr);
                
                Vec3f vE; Vec3f vB;
                if(mParams.smoothVectors)
                  {
                    Vec2f sampleP = Vec2f(xi+fo.x, yi+fo.y);
                    if(sampleP.x >= 0 && sampleP.x < src->size.x && sampleP.y >= 0 && sampleP.y < src->size.y)
                      {
                        bool x1p  = sampleP.x+1 >= src->size.x; bool y1p = sampleP.y+1 >= src->size.y;
                        bool x1y1p = sampleP.x+1 >= src->size.x || sampleP.y+1 >= src->size.y;
                    
                        Vec3f E00 = Vec3f(src->E.hData[src->E.idx((int)sampleP.x,(int)sampleP.y, 0)]);
                        Vec3f E01 = (x1p   ? E00 : Vec3f(src->E.hData[src->E.idx((int)sampleP.x+1, (int)sampleP.y,   0)]));
                        Vec3f E10 = (y1p   ? E00 : Vec3f(src->E.hData[src->E.idx((int)sampleP.x,   (int)sampleP.y+1, 0)]));
                        Vec3f E11 = (x1y1p ? E00 : Vec3f(src->E.hData[src->E.idx((int)sampleP.x+1, (int)sampleP.y+1, 0)]));
                        Vec3f B00 = Vec3f(src->B.hData[src->B.idx((int)sampleP.x,(int)sampleP.y, 0)]);
                        Vec3f B01 = (x1p   ? B00 : Vec3f(src->B.hData[src->B.idx((int)sampleP.x+1, (int)sampleP.y,   0)]));
                        Vec3f B10 = (y1p   ? B00 : Vec3f(src->B.hData[src->B.idx((int)sampleP.x,   (int)sampleP.y+1, 0)]));
                        Vec3f B11 = (x1y1p ? B00 : Vec3f(src->B.hData[src->B.idx((int)sampleP.x+1, (int)sampleP.y+1, 0)]));

                        sp = simToScreen2D(sampleP, mSimView2D, sr);
                        vE = blerp(E00, E01, E10, E11, fo) / mParams.cp.cs; // scale by cell size
                        vB = blerp(B00, B01, B10, B11, fo) / mParams.cp.cs; // scale by cell size
                      }
                  }
                else
                  {
                    Vec2f sampleP = Vec2f(xi, yi);
                    if(sampleP.x >= 0 && sampleP.x < src->size.x && sampleP.y >= 0 && sampleP.y < src->size.y)
                      {
                        sp = simToScreen2D((sampleP+0.5f), mSimView2D, sr);
                        i = src->idx(sampleP.x, sampleP.y);
                        vE = src->E.hData[i] / mParams.cp.cs;
                        vB = src->B.hData[i] / mParams.cp.cs;
                      }
                  }
                
                Vec2f dpE = simToScreen2D(Vec2f(vE.x, vE.y), mSimView2D, sr, true)*mParams.vecMultE;
                Vec2f dpB = simToScreen2D(Vec2f(vB.x, vB.y), mSimView2D, sr, true)*mParams.vecMultB;
                Vec3f dpE3 = Vec3f(dpE.x, dpE.y, 0.0f);
                Vec3f dpB3 = Vec3f(dpB.x, dpB.y, 0.0f);
                float lw     = mParams.vecLineW   / viewScale;
                float bw     = mParams.vecBorderW / viewScale;
                float lAlpha = mParams.vecAlpha;
                float bAlpha = (lAlpha + mParams.vecBAlpha)/2.0f;
                float vLenE = length(dpE3);
                float vLenB = length(dpB3);
                float shear0 = 0.1f;//0.5f;
                float shear1 = 1.0f;

                float magE = length(screenToSim2D(Vec2f(vLenE, 0), mSimView2D, sr, true) / float2{(float)mParams.cp.cs.x, (float)mParams.cp.cs.y});
                float magB = length(screenToSim2D(Vec2f(vLenB, 0), mSimView2D, sr, true) / float2{(float)mParams.cp.cs.x, (float)mParams.cp.cs.y});
                Vec4f Ecol1 = mParams.rp.Ecol;
                Vec4f Bcol1 = mParams.rp.Bcol;
                Vec4f Ecol = Vec4f(Ecol1.x, Ecol1.y, Ecol1.z, lAlpha);
                Vec4f Bcol = Vec4f(Bcol1.x, Bcol1.y, Bcol1.z, lAlpha);
                
                if(!mParams.borderedVectors)
                  {
                    mFieldDrawList->AddLine(Vec2f(sp.x, sp.y), Vec2f(sp.x+dpB.x, sp.y+dpB.y), ImColor(Bcol), lw);
                    mFieldDrawList->AddLine(Vec2f(sp.x, sp.y), Vec2f(sp.x+dpE.x, sp.y+dpE.y), ImColor(Ecol), lw);
                  }
                else
                  {
                    drawLine(mFieldDrawList, Vec2f(sp.x, sp.y), Vec2f(sp.x+dpB.x, sp.y+dpB.y), Bcol, lw, Vec4f(0, 0, 0, bAlpha), bw, shear0, shear1);
                    drawLine(mFieldDrawList, Vec2f(sp.x, sp.y), Vec2f(sp.x+dpE.x, sp.y+dpE.y), Ecol, lw, Vec4f(0, 0, 0, bAlpha), bw, shear0, shear1);
                  }
                avgE += magE; avgB += magB;
              }
          }
      std::cout << "VEC AVG E: " << avgE/((iMax.x - iMin.x)*(iMax.y - iMin.y)) << "\n";
    }
}

void SimWindow::draw(const Vec2f &frameSize)
{
  if(frameSize.x <= 0 || frameSize.y <= 0) { return; }
  mFrameSize = frameSize;
  mInfo.fps  = calcFps();
  
  ImGuiStyle &style = ImGui::GetStyle();
  
  //// draw (imgui)
  ImGuiWindowFlags wFlags = (ImGuiWindowFlags_NoTitleBar        | ImGuiWindowFlags_NoCollapse        |
                             ImGuiWindowFlags_NoMove            | ImGuiWindowFlags_NoResize          |
                             ImGuiWindowFlags_NoScrollbar       | ImGuiWindowFlags_NoScrollWithMouse |
                             ImGuiWindowFlags_NoSavedSettings  | ImGuiWindowFlags_NoBringToFrontOnFocus );
  ImGui::SetNextWindowPos(Vec2f(0,0));
  ImGui::SetNextWindowSize(Vec2f(mFrameSize.x, mFrameSize.y));
  ImGui::PushStyleColor(ImGuiCol_WindowBg, Vec4f(0.05f, 0.05f, 0.05f, 1.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,   0); // square frames by default
  ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,    0);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,    Vec2f(10, 10));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, Vec2f(1, 1));
  ImGui::Begin("##mainView", nullptr, wFlags); // ImGui window covering full application window
  {
    ImGui::PushFont(mainFont);
    Vec2f p0 = ImGui::GetCursorScreenPos();
    
    Vec2f settingsSize = mTabs->getSize();
    mDisplaySize = (mFrameSize - Vec2f(settingsSize.x, 0.0f) - 2.0f*Vec2f(style.WindowPadding) - Vec2f(style.ItemSpacing.x, 0.0f));

    int numViews = ((int)mParams.showEMField + (int)mParams.showMatField + (int)mParams.show3DField);

    if(numViews == 1)
      {
        Rect2f r0 =  Rect2f(p0, p0+mDisplaySize); // whole display
        if(mParams.showEMField)  { mEMView.r  = r0; }
        if(mParams.show3DField)  { m3DView.r  = r0; }
        if(mParams.showMatField) { mMatView.r = r0; }
      }
    else if(numViews == 2)
      {
        Rect2f r0 =  Rect2f(p0, p0+Vec2f(mDisplaySize.x/2.0f, mDisplaySize.y));    // left side
        Rect2f r1 =  Rect2f(p0+Vec2f(mDisplaySize.x/2.0f, 0.0f), p0+mDisplaySize); // right side
        int n = 0;
        if(mParams.showEMField)  { mEMView.r  = (n==0 ? r0 : r1); n++; }
        if(mParams.show3DField)  { m3DView.r  = (n==0 ? r0 : r1); n++; }
        if(mParams.showMatField) { mMatView.r = (n==0 ? r0 : r1); n++; }
      }
    else
      {
        Rect2f r0 =  Rect2f(p0, p0+mDisplaySize/2.0f);              // top-left quarter
        Rect2f r1 =  Rect2f(p0+Vec2f(mDisplaySize.x/2.0f, 0.0f),    // top-right quarter
                            p0+Vec2f(mDisplaySize.x, mDisplaySize.y/2.0f));
        Rect2f r2 =  Rect2f(p0+mDisplaySize/2.0f, p0+mDisplaySize); // bottom-right quarter
        int n = 0;
        if(mParams.showEMField)  { mEMView.r  = (n==0 ? r0 : (n==1 ? r1 : r2)); n++; }
        if(mParams.show3DField)  { m3DView.r  = (n==0 ? r0 : (n==1 ? r1 : r2)); n++; }
        if(mParams.showMatField) { mMatView.r = (n==0 ? r0 : (n==1 ? r1 : r2)); n++; }
      }

    // adjust sim view aspect ratio if window size has changed
    float simAspect  = mSimView2D.aspect();
    float dispAspect = mEMView.r.aspect();
    if(dispAspect > 1.0f) { mSimView2D.scale(Vec2f(dispAspect/simAspect, 1.0f)); }
    else                  { mSimView2D.scale(Vec2f(1.0f, simAspect/dispAspect)); }
    
    // mSimHovered = false;
  
    // draw rendered views of simulation on screen
    ImGui::BeginChild("##simView", mDisplaySize, false, wFlags);
    {
      // EM view
      if(mParams.showEMField)
        {
          handleInput(mEMView);
          ImGui::SetCursorScreenPos(mEMView.r.p1);
          ImGui::BeginChild("##emView", mEMView.r.size(), false, wFlags);
          {
            mFieldDrawList = ImGui::GetWindowDrawList();
            Vec2f fp = Vec2f(mParams.fieldPos.x, mParams.fieldPos.y);
            Vec2f fScreenPos  = simToScreen2D(fp, mSimView2D, mEMView.r);
            Vec2f fCursorPos  = simToScreen2D(fp + Vec2f(0.0f, mParams.fieldRes.y*mParams.cp.cs.y), mSimView2D, mEMView.r);
            Vec2f fScreenSize = simToScreen2D(makeV<float3>(mParams.fieldRes)*mParams.cp.cs, mSimView2D, mEMView.r, true);
            Vec2f t0(0.0f, 1.0f); Vec2f t1(1.0f, 0.0f);
            mEMTex.bind();
            ImGui::SetCursorScreenPos(fCursorPos);
            ImGui::Image(reinterpret_cast<ImTextureID>(mEMTex.texId()), fScreenSize, t0, t1, ImColor(Vec4f(1,1,1,1)));
            mEMTex.release();
            
            drawVectorField(mEMView.r);
          }
          ImGui::EndChild();
        }
      
      // Material view
      if(mParams.showMatField)
        {
          handleInput(mMatView);
          ImGui::SetCursorScreenPos(mMatView.r.p1);
          ImGui::BeginChild("##matView", mMatView.r.size(), false, wFlags);
          {
            mFieldDrawList = ImGui::GetWindowDrawList();
            Vec2f fp = Vec2f(mParams.fieldPos.x, mParams.fieldPos.y);
            Vec2f fScreenPos  = simToScreen2D(fp, mSimView2D, mMatView.r);
            Vec2f fCursorPos  = simToScreen2D(fp + Vec2f(0.0f, mParams.fieldRes.y*mParams.cp.cs.y), mSimView2D, mMatView.r);
            Vec2f fScreenSize = simToScreen2D(makeV<float3>(mParams.fieldRes)*mParams.cp.cs, mSimView2D, mMatView.r, true);
            Vec2f t0(0.0f, 1.0f); Vec2f t1(1.0f, 0.0f);
            mMatTex.bind();
            ImGui::SetCursorScreenPos(fCursorPos);
            ImGui::Image(reinterpret_cast<ImTextureID>(mMatTex.texId()), fScreenSize, t0, t1, ImColor(Vec4f(1,1,1,1)));
            mMatTex.release();
            drawVectorField(mMatView.r);
          }
          ImGui::EndChild();
        }
      
      // Raytraced 3D view
      if(mParams.show3DField)
        {
          handleInput3D(m3DView);
          ImGui::SetCursorScreenPos(m3DView.r.p1);
          ImGui::BeginChild("##3DView", m3DView.r.size(), false, wFlags);
          {
            mFieldDrawList = ImGui::GetWindowDrawList();
            Vec2f t0(0.0f, 1.0f); Vec2f t1(1.0f, 0.0f);
            m3DTex.bind();
            ImGui::SetCursorScreenPos(m3DView.r.p1);
            ImGui::Image(reinterpret_cast<ImTextureID>(m3DTex.texId()), m3DView.r.size(), t0, t1, ImColor(Vec4f(1,1,1,1)));
            m3DTex.release();
            // drawVectorField(m3DView.r);

            if(mParams.drawAxes)
              {
                // draw X/Y/Z axes at origin (R/G/B)
                ImDrawList *drawList = ImGui::GetWindowDrawList();
                Vec2f mpos = ImGui::GetMousePos();
        
                mCamera.calculate();
                // axis points
                Vec3d WO0 = Vec3d(0,0,0);
                Vec3d Wpx = WO0 + Vec3f(mParams.cp.cs.x*mParams.fieldRes.x, 0, 0);
                Vec3d Wpy = WO0 + Vec3f(0, mParams.cp.cs.y*mParams.fieldRes.y, 0);
                Vec3d Wpz = WO0 + Vec3f(0, 0, mParams.cp.cs.z*mParams.fieldRes.z);
                // transform
                bool oClipped = false; bool xClipped = false; bool yClipped = false; bool zClipped = false;
                Vec2f Sorigin = m3DView.r.p1 + (mCamera.worldToView(WO0, &oClipped)) * m3DView.r.size().y;
                Vec2f Spx     = m3DView.r.p1 + (mCamera.worldToView(Wpx, &xClipped)) * m3DView.r.size().y;
                Vec2f Spy     = m3DView.r.p1 + (mCamera.worldToView(Wpy, &yClipped)) * m3DView.r.size().y;
                Vec2f Spz     = m3DView.r.p1 + (mCamera.worldToView(Wpz, &zClipped)) * m3DView.r.size().y;
                // draw axes
                if(!oClipped || !xClipped) { drawList->AddLine(Sorigin, Spx, ImColor(Vec4f(1, 0, 0, 1)), 2.0f); }
                if(!oClipped || !yClipped) { drawList->AddLine(Sorigin, Spy, ImColor(Vec4f(0, 1, 0, 1)), 2.0f); }
                if(!oClipped || !zClipped) { drawList->AddLine(Sorigin, Spz, ImColor(Vec4f(0, 0, 1, 1)), 2.0f); }
              }
          }
          ImGui::EndChild();
        }
    }
    ImGui::EndChild();
    // ImGui::PopStyleVar();

    if(mEMView.hovered)  { mMouseSimPos = Vec2f(mEMView.mposSim.x,  mEMView.mposSim.y);  }
    if(mMatView.hovered) { mMouseSimPos = Vec2f(mMatView.mposSim.x, mMatView.mposSim.y); }
    if(m3DView.hovered)  { mMouseSimPos = Vec2f(m3DView.mposSim.x,  m3DView.mposSim.y);  }

    // Settings
    ImGui::SameLine();
    ImGui::BeginGroup();
    {
      Vec2f sp0 = Vec2f(mFrameSize.x - settingsSize.x - style.WindowPadding.x, ImGui::GetCursorPos().y);
      ImGui::SetCursorPos(sp0);
      
      mTabs->setLength(mDisplaySize.y);
      mTabs->draw();
        
      settingsSize = mTabs->getSize() + Vec2f(2.0f*style.WindowPadding.x + style.ScrollbarSize, 0.0f) + Vec2f(mTabs->getBarWidth(), 0.0f);
      //mSettingsSize = Vec2f(settingsSize.x, settingsSize.y);
      //mSettingsSize.x += 2.0f*style.WindowPadding.x + 2.0f*style.ScrollbarSize;
      //mSettingsSize.y = mFrameSize.y - 2.0f*style.WindowPadding.y;// - tSize.y - 2.0f*style.ItemSpacing.y;
          
      Vec2f minSize = Vec2f(512+settingsSize.x+style.ItemSpacing.x, 512) + Vec2f(style.WindowPadding)*2.0f + Vec2f(style.FramePadding)*2.0f;
      glfwSetWindowSizeLimits(mWindow, (int)minSize.x, (int)minSize.y, GLFW_DONT_CARE, GLFW_DONT_CARE);
      ImGui::SetWindowSize(settingsSize + Vec2f(style.ScrollbarSize, 0.0f));
    }
    ImGui::EndGroup();


    // debug overlay
    if(mParams.debug)
      {
        ImDrawList *drawList = ImGui::GetForegroundDrawList();
        ImGui::PushClipRect(Vec2f(0,0), mFrameSize, false);
        ImGui::SetCursorPos(Vec2f(10.0f, 10.0f));
        ImGui::PushStyleColor(ImGuiCol_ChildBg, Vec4f(0.0f, 0.0f, 0.0f, 0.0f));
        ImGui::BeginChild("##debugOverlay", mDisplaySize, false, wFlags);
        {
          // Vec2f cursor = Vec2f(10,10);
          // drawList->AddText(cursor, ImColor(Vec4f(1,1,1,1)), Text.Text.c_str());
          ImGui::PushFont(titleFontB);
          ImGui::Text("%.2f FPS", mInfo.fps);
          ImGui::PopFont();
          ImGui::Spacing();
          ImGui::Text("t =  %f  (sim time %.3fx real time)", mInfo.t, mParams.running ? (fpsLast*mParams.dt) : 0.0f);
          ImGui::Text("Mouse Sim Pos: < %f, %f>",              mMouseSimPos.x,        mMouseSimPos.y);
          ImGui::Text("SimView:       < %f, %f> : < %f, %f >", mSimView2D.p1.x,       mSimView2D.p1.y, mSimView2D.p2.x, mSimView2D.p2.y);
          ImGui::Text("SimView Size:  < %f, %f>",              mSimView2D.size().x,   mSimView2D.size().y);
          ImGui::Text("EM  2D View:   < %f, %f> : < %f, %f>",  mEMView.r.p1.x,  mEMView.r.p1.y,  mEMView.r.p2.x,  mEMView.r.p2.y);
          ImGui::Text("Mat 2D View:   < %f, %f> : < %f, %f>",  mMatView.r.p1.x, mMatView.r.p1.y, mMatView.r.p2.x, mMatView.r.p2.y);
          ImGui::Text("3D  EM View:   < %f, %f> : < %f, %f>",  m3DView.r.p1.x,  m3DView.r.p1.y,  m3DView.r.p2.x,  m3DView.r.p2.y);

          ImGui::SetCursorScreenPos(Vec2f(ImGui::GetCursorScreenPos()) + Vec2f(0.0f, 45.0f));
          ImGui::Text("Camera Pos:   < %f, %f, %f>", mCamera.pos.x,   mCamera.pos.y,   mCamera.pos.z  );
          ImGui::Text("Camera Focus: < %f, %f, %f>", mCamera.focus.x, mCamera.focus.y, mCamera.focus.z);
          ImGui::Text("Camera Dir:   < %f, %f, %f>", mCamera.dir.x,   mCamera.dir.y,   mCamera.dir.z  );
          ImGui::Text("Camera Up:    < %f, %f, %f>", mCamera.up.x,    mCamera.up.y,    mCamera.up.z   );
          ImGui::Text("Camera Right: < %f, %f, %f>", mCamera.right.x, mCamera.right.y, mCamera.right.z);
          ImGui::Text("Camera Fov:   %f", mCamera.fov);
          ImGui::Text("Camera Near:  %f", mCamera.near);
          ImGui::Text("Camera Far:   %f", mCamera.far);
          mCamera.calculate();
          ImGui::Text("Camera Proj:  \n%s", mCamera.proj.toString().c_str());
          ImGui::Text("Camera View:  \n%s", mCamera.view.toString().c_str());
          ImGui::Text("Camera ViewInv:  \n%s", mCamera.viewInv.toString().c_str());
          ImGui::Text("Camera VP:  \n%s", mCamera.VP.toString().c_str());

          ImGui::TextUnformatted("Greek Test: ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσΤτΥυΦφΧχΨψΩω");
        }
        ImGui::EndChild();
        ImGui::PopClipRect();
        ImGui::PopStyleColor();
      }
    
    
    ImGui::PopFont();
    if(mImGuiDemo) { ImGui::ShowDemoWindow(&mImGuiDemo); } // show imgui demo window (Alt+Shift+D)
  }
  ImGui::End();
  ImGui::PopStyleVar(4);
  ImGui::PopStyleColor(1);
  
  if(mClosing) { glfwSetWindowShouldClose(mWindow, GLFW_TRUE); }
}

double SimWindow::calcFps()
{
  tNow  = CLOCK_T::now();
  dt    = std::chrono::duration_cast<std::chrono::nanoseconds>(tNow - tLast).count()/1000000000.0;
  tLast = tNow;
  
  tDiff += dt; // convert to nanoseconds
  nFrames++;
  double fps = fpsLast;
  if(tDiff > FPS_UPDATE_INTERVAL)
    {
      fps     = nFrames / tDiff;
      fpsLast = fps;
      tDiff   = 0.0;
      nFrames = 0;
    }
  return fps;
}


















//// TODO: fix up image sequence rendering
void SimWindow::initGL(const Vec2i &texSize)
{
  if(texSize != mGlTexSize || mParams.outAlpha != mGlAlpha)
    {
      std::cout << "GL TEX SIZE: " << mGlTexSize << " --> " << texSize <<  ")...\n";
      cleanupGL();
      
      mGlTexSize = texSize;
      mGlAlpha   = mParams.outAlpha;
      
      std::cout << "INITIALIZING GL RESOURCES...\n";
      // create framebuffer  
      glGenFramebuffers(1, &mRenderFB);
      glBindFramebuffer(GL_FRAMEBUFFER, mRenderFB);
      // create texture
      glGenTextures(1, &mRenderTex);
      glBindTexture(GL_TEXTURE_2D, mRenderTex);
      //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texSize.x, texSize.y, 0, GL_RGB, GL_FLOAT, 0);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mGlTexSize.x, mGlTexSize.y, 0, GL_RGB, GL_FLOAT, 0);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      // set as color attachment
      glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, mRenderTex, 0);
      glDrawBuffers(1, mDrawBuffers);
      if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        { std::cout << "====> ERROR: Failed to create framebuffer for rendering ImGui text/etc. onto texture.\n"; return; }
      else
        { std::cout << "Framebuffer success!\n"; }

      // initialize
      glViewport(0, 0, mGlTexSize.x, mGlTexSize.y);
      glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      glBindTexture(GL_TEXTURE_2D, 0);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);

      int channels = (mGlAlpha ? 4 : 3);
      // allocate host memory
      mTexData  = new unsigned char[channels*mGlTexSize.x*mGlTexSize.y];
      mTexData2 = new unsigned char[channels*mGlTexSize.x*mGlTexSize.y];
    }
}
void SimWindow::cleanupGL()
{
  if(mTexData)
    {
      std::cout << "CLEANING UP GL RESOURCES...\n";
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glBindTexture(GL_TEXTURE_2D, 0);
      glDeleteFramebuffers(1, &mRenderFB); mRenderFB  = 0;
      glDeleteTextures(1, &mRenderTex); mRenderTex = 0;
      delete[] mTexData;  mTexData  = nullptr;
      delete[] mTexData2; mTexData2 = nullptr;
      mGlTexSize = Vec2i(0,0);
      // mGlAlpha   = true;
    }
}

void SimWindow::renderToFile()
{
  ImGuiIO    &io    = ImGui::GetIO();
  ImGuiStyle &style = ImGui::GetStyle();
  if(mFileRendering && (mNewFrame || (mInfo.frame == 0 && mInfo.uStep == 0)))
    {
      // Vec2i outSize = (Vec2i(mTex.size.x, mTex.size.y) + Vec2i(mLegend->size().x, mInfoDisp->size().y) +
      //                  Vec2i(style.ItemInnerSpacing.x, style.ItemInnerSpacing.y) + 2*Vec2i(style.WindowPadding.x, style.WindowPadding.y));
      Vec2i outSize = Vec2i(mParams.outSize);
      initGL(outSize);
      
      ImGui_ImplOpenGL3_NewFrame();
      io.DisplaySize             = Vec2f(mGlTexSize.x, mGlTexSize.y);
      io.DisplayFramebufferScale = Vec2f(1.0f, 1.0f);
      ImGui::NewFrame();
      {
        ImGuiStyle &style = ImGui::GetStyle();
        
        //// draw (imgui)
        ImGuiWindowFlags wFlags = (ImGuiWindowFlags_NoTitleBar        | ImGuiWindowFlags_NoCollapse        |
                                   ImGuiWindowFlags_NoMove            | ImGuiWindowFlags_NoResize          |
                                   ImGuiWindowFlags_NoScrollbar       | ImGuiWindowFlags_NoScrollWithMouse |
                                   ImGuiWindowFlags_NoSavedSettings   | ImGuiWindowFlags_NoBringToFrontOnFocus );
        ImGui::SetNextWindowPos(Vec2f(0,0));
        ImGui::SetNextWindowSize(Vec2f(mGlTexSize.x, mGlTexSize.y));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0); // square frames by default
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,  0);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,  Vec2f(0, 0));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, Vec4f(0.0f, 0.0f, 0.0f, 1.0f));
        ImGui::Begin("##renderView", nullptr, wFlags); // ImGui window covering full application window
        {
          ImGui::PushFont(mainFont);
          ImGui::SetCursorScreenPos(Vec2f(1.0f, 1.0f));
          
          // float scale = mFluidTex.size.y >= mGlTexSize.y ? (0.8f*mGlTexSize.y / mFluidTex.size.y) : 1.0f;
          // // draw simulation textures and params/legend with ImGui onto a separate texture
          // ImGui::BeginGroup();
          // {
          //   // mFluidTex.bind();
          //   // ImGui::Image(reinterpret_cast<ImTextureID>(mFluidTex.texId), scale*Vec2f(mFluidTex.size.x, mFluidTex.size.y),
          //   //              Vec2f(0), Vec2f(1), Vec4f(1), Vec4f(0.5f, 0.1f, 0.1f, 1.0f));
          //   // mFluidTex.release();
          
          //   // mGrapheneTex.bind();
          //   // scale = mGrapheneTex.size.y >= mGlTexSize.y ? (0.8f*mGlTexSize.y / mGrapheneTex.size.y) : 1.0f;
          //   // ImGui::Image(reinterpret_cast<ImTextureID>(mGrapheneTex.texId), scale*Vec2f(mGrapheneTex.size.x, mGrapheneTex.size.y),
          //   //              Vec2f(0), Vec2f(1), Vec4f(1), Vec4f(0.5f, 0.1f, 0.1f, 1.0f));
          //   // mGrapheneTex.release();

            
          //   // calculate offset/scaling from simulation space

          //   Vec2f oldDispSize = mDisplaySize;
          //   Rect2f oldSimView = mSimView2D;
          //   mDisplaySize = scale*Vec2f(mFluidTex.size.x, mFluidTex.size.y);
          //   mSimView2D = Rect2f(Vec2f(0,0), Vec2f(1,1));
            
          //   Rect2f viewRect = mSimView2D;
          //   Vec2f  gSimSize = mParams.gp.simSize; Vec2f gSimPos  = mParams.gp.simPos;
          //   Vec2f  fSimSize = mParams.fp.simSize; Vec2f fSimPos  = mParams.fp.simPos;
        
          //   // render fluid
          //   Vec2f p1 = ImGui::GetCursorScreenPos();
          //   Vec2f drawSize = simToScreen2D(fSimPos + fSimSize, &p1) - simToScreen2D(fSimPos, &p1);
          //   drawSize.x = std::abs(drawSize.x); drawSize.y = std::abs(drawSize.y);
          //   Vec2f t0 = Vec2f(0,1); Vec2f t1 = Vec2f(1,0);
          //   Vec2f cursor = simToScreen2D(fSimPos + Vec2f(0.0f, fSimSize.y), &p1);
        
          //   // if(mParams.tileX)
          //   //   { // view rect used as texture coordinates for wrapped sampling
          //   //     drawSize.x = std::abs(simToScreen2D(viewRect.size()).x);
          //   //     t0.x = viewRect.p1.x; t1.x = viewRect.p2.x;
          //   //     cursor.x = p1.x;
          //   //   }
          //   // if(mParams.tileY)
          //   //   {
          //   //     drawSize.y = std::abs(simToScreen2D(viewRect.size()).y);
          //   //     t0.y = viewRect.p2.y; t1.y = viewRect.p1.y;
          //   //     cursor.y = p1.y;
          //   //   }
          //   ImGui::SetCursorScreenPos(cursor);
          //   mFluidTex.bind();
          //   ImGui::Image(reinterpret_cast<ImTextureID>(mFluidTex.texId), drawSize, t0, t1, ImColor(Vec4f(1,1,1,1)));
          //   mFluidTex.release();

          //   if(mParams.runGraphene)
          //     { // render graphene
          //       //mDisplaySize = scale*Vec2f(mGrapheneTex.size.x, mGrapheneTex.size.y);
          //       drawSize = simToScreen2D(gSimPos + gSimSize, &p1) - simToScreen2D(gSimPos, &p1);
          //       drawSize.x = std::abs(drawSize.x); drawSize.y = std::abs(drawSize.y);
          //       t0 = Vec2f(0,1); t1 = Vec2f(1,0);
          //       cursor = simToScreen2D(gSimPos+Vec2f(0, gSimSize.y), &p1);            
          //       // if(mParams.tileX)
          //       //   { // view rect used as texture coordinates for wrapped sampling
          //       //     drawSize.x = mDisplaySize.x;
          //       //     t0.x = viewRect.p1.x; t1.x = viewRect.p2.x;
          //       //     cursor.x = p1.x;
          //       //   }
          //       // if(mParams.tileY)
          //       //   { // view rect used as texture coordinates for wrapped sampling
          //       //     drawSize.y = std::abs(simToScreen2D(gSimPos + gSimSize, &p1).y - simToScreen2D(gSimPos, &p1).y);
          //       //     t0.y = viewRect.p2.y; t1.y = viewRect.p1.y;
          //       //   }
          //       ImGui::SetCursorScreenPos(cursor);
          //       mGrapheneTex.bind();
          //       ImGui::Image(reinterpret_cast<ImTextureID>(mGrapheneTex.texId), drawSize, t0, t1, ImColor(Vec4f(1,1,1,1)));
          //       mGrapheneTex.release();

          //       // draw graphene velocity arrows
          //       int gcount = mGraphene1->size.x*mGraphene1->size.y;
          //       float  *gqn  = new float[gcount];
          //       float2 *gqnv = new float2[gcount];
          //       cudaMemcpy(gqn,  mGraphene1->qn,  gcount*sizeof(float), cudaMemcpyDeviceToHost);
          //       cudaMemcpy(gqnv, mGraphene1->qnv, gcount*sizeof(float2), cudaMemcpyDeviceToHost);

          //       ImDrawList *drawList = ImGui::GetWindowDrawList();
          //       for(int i = 0; i < gcount; i++)
          //         {
          //           Vec2f lp = cursor + Vec2f((i+0.5f)/(float)gcount, 0.5f)*drawSize;
          //           drawList->AddLine(lp, lp + Vec2f(normalize(gqnv[i]).x*0.01f, -10.0f*std::min(1.0f, gqn[i])*(gqnv[i].x < 0 ? -1 : 1))*drawSize,
          //                             ImColor(gqnv[i].x < 0.0f ? Vec4f(1,0,1,1) : Vec4f(1,1,1,1)), 1);
          //         }
                      
          //       delete[] gqn;
          //       delete[] gqnv;
          //     }
          
          //   mDisplaySize = oldDispSize;
          //   mSimView2D     = oldSimView;
          
          // }
          // ImGui::EndGroup();          
          // //ImGui::SetCursorScreenPos(Vec2f(mTex.size.x - mLegend->size().x - 20.0f, 20.0f));
          // ImGui::SameLine();
          // mLegend->draw(&mParams, true);
          // //ImGui::SetCursorScreenPos(Vec2f(20.0f, mTex.size.y - mInfoDisp->size().y - 20.0f));
          // ImGui::SetCursorScreenPos(Vec2f(21.0f, ImGui::GetCursorScreenPos().y));
          // mInfoDisp->draw(&mParams, &mInfo, Vec2f(scale*mFluidTex.size.x - 2.0f*style.WindowPadding.x, 0.0f), true);
          ImGui::PopFont();
        }
        ImGui::End();
        ImGui::PopStyleVar(3); ImGui::PopStyleColor();
      }
      ImGui::EndFrame();

      // render to ImGui frame to separate framebuffer
      glUseProgram(0); ImGui::Render();
      glBindFramebuffer(GL_FRAMEBUFFER, mRenderFB);
      glViewport(0, 0, mGlTexSize.x, mGlTexSize.y);
      glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

      // copy fluid texture data to host
      glBindTexture(GL_TEXTURE_2D, mRenderTex);
      glReadPixels(0, 0, mGlTexSize.x, mGlTexSize.y, (mGlAlpha ? GL_RGBA : GL_RGB), GL_UNSIGNED_BYTE, (GLvoid*)mTexData);
      // unbind texture and framebuffer
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glBindTexture(GL_TEXTURE_2D, 0);
      
      // flip y --> copy each row of fluid into opposite row of mTexData2
      int channels = (mGlAlpha ? 4 : 3);
              
      for(int y = 0; y < mGlTexSize.y; y++)
        {
          std::copy(&mTexData [y*mGlTexSize.x*channels],                   // row y
                    &mTexData [(y+1)*mGlTexSize.x*channels],               // row y+1
                    &mTexData2[(mGlTexSize.y-1-y)*mGlTexSize.x*channels]); // flipped row
        }
      // write data to file
      std::stringstream ss; ss << mBaseDir << "/" << mParams.simName << "/"
                               << mParams.simName << "-" << std::setfill('0') << std::setw(5) << mInfo.frame << ".png";
      std::string imagePath = ss.str();
      std::cout << "WRITING FRAME " << mInfo.frame << " TO " << imagePath << "...\n";
      setPngCompression(mParams.pngCompression);
      writeTexture(imagePath, (const void*)mTexData2, mGlTexSize, mGlAlpha);
      mNewFrame = false;
    }
}

