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

#include "draw.hpp"
#include "display.hpp"
#include "render.cuh"


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
      if(!mFieldUI->running)
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
          CFT    dtStep1     = 0.1; CFT    dtStep2     = 1.0;
          
          for(auto iter : mKeysDown)
            {
              if(iter.second)
                {
                  switch(iter.first)
                    {
                    case GLFW_KEY_T: // T -- adjust timestep
                      mUnits.dt    += signMult*keyMult*(ctrl ? dtStep2 : dtStep1);
                      std::cout << "TIMESTEP:            " << mUnits.dt << "\n";
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


SimWindow::SimWindow(GLFWwindow *window)
  : mWindow(window)
{
  simWin = this; // NOTE/TODO: only one window total allowed for now
  glfwSetKeyCallback(mWindow, &keyCallback); // key event callback


  // setting flags
  SettingGroup *flagGroup = new SettingGroup("Flags", "flags", { }, false);
  auto *sDBG   = new Setting<bool> ("Debug",   "debug",   &mParams.debug);   mSettings.push_back(sDBG);  flagGroup->add(sDBG);
  auto *sVERB  = new Setting<bool> ("Verbose", "verbose", &mParams.verbose); mSettings.push_back(sVERB); flagGroup->add(sVERB);
  
  // old settings (ad hoc)
  mFlagUI = new SettingForm("Other Settings", SETTINGS_LABEL_COL_W, SETTINGS_INPUT_COL_W);
  mSettingsSize.x = mFlagUI->labelColWidth() + mFlagUI->inputColWidth();
  mFlagUI->add(flagGroup);
}

SimWindow::~SimWindow()
{
  cleanup();
  if(mFlagUI) { delete mFlagUI; mFlagUI = nullptr; }
}

bool SimWindow::init()
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

      mUnitsUI   = new UnitsInterface<CFT>(&mUnits, superFont);
      mDrawUI    = new DrawInterface<CFT>(&mUnits);
      mFieldUI   = new FieldInterface<CFT>(&mParams.cp,
                                           [this]() // fieldRes update
                                           { resizeFields(mFieldUI->fieldRes); resetSim(); },
                                           [this]() // texRes2D update
                                           {
                                             mEMTex.create(int3{mFieldUI->texRes2D.x, mFieldUI->texRes2D.y, 1});
                                             mMatTex.create(int3{mFieldUI->texRes2D.x, mFieldUI->texRes2D.y, 1});
                                           },
                                           [this]() // texRes3D update
                                           { m3DTex.create(int3{mFieldUI->texRes3D.x, mFieldUI->texRes3D.y, 1}); });
      mDisplayUI = new DisplayInterface<CFT>(&mParams.rp, &mParams.vp, &mFieldUI->fieldRes.z);
      
  
      mTabs = new TabMenu(20, 1080, true);
      mTabs->setCollapsible(true);
      mTabs->add(TabDesc{"Field",   "Field Settings", [this](){ mFieldUI->draw(); },
                         (int)SETTINGS_TOTAL_W, (int)SETTINGS_TOTAL_W+40, (int)SETTINGS_TOTAL_W+40, titleFont});
      mTabs->add(TabDesc{"Display", "Display Settings",  [this](){ mDisplayUI->draw(); },
                         (int)SETTINGS_TOTAL_W, (int)SETTINGS_TOTAL_W+40, (int)SETTINGS_TOTAL_W+40, titleFont});
      mTabs->add(TabDesc{"Draw",    "Draw Settings",  [this](){ mDrawUI->draw(superFont); },
                         (int)SETTINGS_TOTAL_W, (int)SETTINGS_TOTAL_W+40, (int)SETTINGS_TOTAL_W+40, titleFont});
      mTabs->add(TabDesc{"Units",   "Base Units",          [this](){ mUnitsUI->draw(); },
                         (int)SETTINGS_TOTAL_W, (int)SETTINGS_TOTAL_W+40, (int)SETTINGS_TOTAL_W+40, titleFont});
      mTabs->add(TabDesc{"Other",   "Other Settings", [this](){ mFlagUI->draw(); },
                         (int)SETTINGS_TOTAL_W, (int)SETTINGS_TOTAL_W+40, (int)SETTINGS_TOTAL_W+40, titleFont});
      
      std::cout << "Creating CUDA objects...\n";
      // initialize CUDA and check for a compatible device
      if(!initCudaDevice())
        {
          std::cout << "====> ERROR: failed to initialize CUDA device!\n";
          delete fontConfig; fontConfig = nullptr;
          return false;
        }

      mFieldUI->initQPExpr  = toExpression<float >(mFieldUI->initQPStr);
      mFieldUI->initQNExpr  = toExpression<float >(mFieldUI->initQNStr);
      mFieldUI->initQPVExpr = toExpression<float3>(mFieldUI->initQPVStr);
      mFieldUI->initQNVExpr = toExpression<float3>(mFieldUI->initQNVStr);
      mFieldUI->initEExpr   = toExpression<float3>(mFieldUI->initEStr);
      mFieldUI->initBExpr   = toExpression<float3>(mFieldUI->initBStr);

      //// set up CUDA fields
      // create state queue
      if(mFieldUI->fieldRes.x > 0 && mFieldUI->fieldRes.y > 0 && mFieldUI->fieldRes.z > 0)
        {
          for(int i = 0; i < STATE_BUFFER_SIZE; i++)
            { mStates.push_back(new EMField<CFT>()); }
          resizeFields(mFieldUI->fieldRes);
        }
      
      int3 ts2 = int3{mFieldUI->texRes2D.x, mFieldUI->texRes2D.y, 1};
      int3 ts3 = int3{mFieldUI->texRes3D.x, mFieldUI->texRes3D.y, 1};
      std::cout << "Creating 2D textures (" << ts2 << ")\n";
      if(!mEMTex.create(ts2))  { std::cout << "====> ERROR: Texture creation for EM view failed!\n";  }
      if(!mMatTex.create(ts2)) { std::cout << "====> ERROR: Texture creation for Mat view failed!\n"; }
      std::cout << "Creating 3D texture (" << ts3 << ")\n";
      if(!m3DTex.create(ts3))  { std::cout << "====> ERROR: Texture creation for 3D view failed!\n";  }
      
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
      
      if(mTabs)      { delete mTabs;       mTabs      = nullptr; }
      if(mUnitsUI)   { delete mUnitsUI;    mUnitsUI   = nullptr; }
      if(mDrawUI)    { delete mDrawUI;     mDrawUI    = nullptr; }
      if(mDisplayUI) { delete mDisplayUI;  mDisplayUI = nullptr; }
      if(mFieldUI)   { delete mFieldUI;    mFieldUI   = nullptr; }
      
      std::cout << "Cleaning GL...\n";
      cleanupGL();

      mInitialized = false;
    }
}



bool SimWindow::resizeFields(const Vec3i &sz)
{
  if(min(sz) <= 0) { std::cout << "====> ERROR: Field with zero size not allowed.\n"; return false; }
  bool success = true;
  for(int i = 0; i < STATE_BUFFER_SIZE; i++)
    {
      EMField<CFT> *f = reinterpret_cast<EMField<CFT>*>(mStates[i]);
      if(!f->create(mFieldUI->fieldRes)) { std::cout << "Field creation failed! Invalid state.\n"; success = false; break; }
      fieldFillValue(reinterpret_cast<EMField<CFT>*>(f)->mat, mUnits.vacuum());
    }
  mDrawUI->sigPen.depth = mFieldUI->fieldRes.z/2;
  mDrawUI->matPen.depth = mFieldUI->fieldRes.z/2;
  mDisplayUI->rp->zRange = int2{0, mFieldUI->fieldRes.z-1};
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
      std::cout << "QP:  "  << mFieldUI->initQPExpr->toString(true)  << "\n";
      std::cout << "QN:  "  << mFieldUI->initQNExpr->toString(true)  << "\n";
      std::cout << "QPV:  " << mFieldUI->initQPVExpr->toString(true) << "\n";
      std::cout << "QNV:  " << mFieldUI->initQNVExpr->toString(true) << "\n";
      std::cout << "E:  "   << mFieldUI->initEExpr->toString(true)   << "\n";
      std::cout << "B:  "   << mFieldUI->initBExpr->toString(true)   << "\n";
    }

  mFieldUI->initQPExpr  = toExpression<float >(mFieldUI->initQPStr,  false);
  mFieldUI->initQNExpr  = toExpression<float >(mFieldUI->initQNStr,  false);
  mFieldUI->initQPVExpr = toExpression<float3>(mFieldUI->initQPVStr, false);
  mFieldUI->initQNVExpr = toExpression<float3>(mFieldUI->initQNVStr, false);
  mFieldUI->initEExpr   = toExpression<float3>(mFieldUI->initEStr,   false);
  mFieldUI->initBExpr   = toExpression<float3>(mFieldUI->initBStr,   false);
  
  // create/update expressions 
  std::cout << "Filling field states on device...\n";
  if(!mFieldUI->mFillQP)
    { mFieldUI->mFillQP  = toCudaExpression<float> (mFieldUI->initQPExpr,  getVarNames<float>());  std::cout << "  --> QP  EXPRESSION UPDATED\n"; }
  if(!mFieldUI->mFillQN)
    { mFieldUI->mFillQN  = toCudaExpression<float> (mFieldUI->initQNExpr,  getVarNames<float>());  std::cout << "  --> QN  EXPRESSION UPDATED\n"; }
  if(!mFieldUI->mFillQPV)
    { mFieldUI->mFillQPV = toCudaExpression<float3>(mFieldUI->initQPVExpr, getVarNames<float3>()); std::cout << "  --> QPV EXPRESSION UPDATED\n"; }
  if(!mFieldUI->mFillQNV)
    { mFieldUI->mFillQNV = toCudaExpression<float3>(mFieldUI->initQNVExpr, getVarNames<float3>()); std::cout << "  --> QNV EXPRESSION UPDATED\n"; }
  if(!mFieldUI->mFillE)
    { mFieldUI->mFillE   = toCudaExpression<float3>(mFieldUI->initEExpr,   getVarNames<float3>()); std::cout << "  --> E   EXPRESSION UPDATED\n"; }
  if(!mFieldUI->mFillB)
    { mFieldUI->mFillB   = toCudaExpression<float3>(mFieldUI->initBExpr,   getVarNames<float3>()); std::cout << "  --> B   EXPRESSION UPDATED\n"; }
  // fill all states
  for(int i = 0; i < mStates.size(); i++)
    {
      EMField<CFT> *f = reinterpret_cast<EMField<CFT>*>(mStates[mStates.size()-1-i]);
      fieldFillChannel<float2>(f->Q,   mFieldUI->mFillQP, 0); // q+ --> Q[i].x
      fieldFillChannel<float2>(f->Q,   mFieldUI->mFillQN, 1); // q- --> Q[i].y
      fieldFill       <float3>(f->QPV, mFieldUI->mFillQPV);
      fieldFill       <float3>(f->QNV, mFieldUI->mFillQNV);
      fieldFill       <float3>(f->E,   mFieldUI->mFillE);
      fieldFill       <float3>(f->B,   mFieldUI->mFillB);
    }
  cudaDeviceSynchronize();
}


void SimWindow::resetMaterials()
{
  std::cout << "MATERIAL RESET\n";
  for(int i = 0; i < mStates.size(); i++)
    { // reset materials in each state
      EMField<CFT> *f = reinterpret_cast<EMField<CFT>*>(mStates[mStates.size()-1-i]);
      fieldFillValue<Material<CFT>>(*reinterpret_cast<Field<Material<CFT>>*>(&f->mat), mUnits.vacuum());
    }
  // cudaDeviceSynchronize();
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


void SimWindow::resetViews()
{  
  CFT    pad = SIM_VIEW_RESET_INTERNAL_PADDING;
  Vec2f  fs  = Vec2f(mFieldUI->fieldRes.x, mFieldUI->fieldRes.y);
  Vec2f  fp  = Vec2f(mFieldUI->cp->fp.x,    mFieldUI->cp->fp.y);
  Vec2f  fsPadded = fs * (1.0 + 2.0*pad);
  
  Vec2f aspect2D = Vec2f(mEMView.r.aspect(), 1.0f);
  if(aspect2D.x < 1.0) { aspect2D.y = 1.0f/aspect2D.x; aspect2D.x = 1.0f; } // make sure the whole field stays inside view
  Vec2f aspect3D = Vec2f(m3DView.r.aspect(), 1.0f);
  if(aspect3D.x < 1.0) { aspect3D.y = 1.0f/aspect3D.x; aspect3D.x = 1.0f; } // make sure the whole field stays inside view
  
  // reset 2D sim view
  Vec2f fp2D = -(fsPadded * pad/aspect2D);
  mSimView2D = Rect2f(fp2D, fp2D + fsPadded*aspect2D);
  Vec2f offset2D = fs/2.0f - mSimView2D.center();
  mSimView2D.move(offset2D);

  float S = 2.0*tan((mCamera.fov/2.0f)*M_PI/180.0f);
  float zOffset = min(aspect3D*fsPadded*mUnits.dL) / S;
  
  // reset 3D camera
  mCamera.fov   = 55.0f; mCamera.near = 0.001f; mCamera.far = 100000.0f;
  mCamera.pos   = float3{ fs.x*mUnits.dL/2,
                          fs.y*mUnits.dL/2,
                          zOffset+mFieldUI->fieldRes.z*mUnits.dL}; // camera position (centered over field)
  mCamera.right = float3{1.0f,  0.0f,  0.0f};     // camera x direction
  mCamera.up    = float3{0.0f,  1.0f,  0.0f};     // camera y direction
  mCamera.dir   = float3{0.0f,  0.0f, -1.0f};     // camera z direction
}


void SimWindow::togglePause()
{
  mFieldUI->running = !mFieldUI->running;
  std::cout << (mFieldUI->running ? "STARTED" : "STOPPED") << " SIMULATION.\n";
}

static EMField<CFT> *g_temp = nullptr;
// static EMField<CFT> *g_temp2 = nullptr;
void SimWindow::update()
{
  if(mFieldUI->fieldRes.x > 0 && mFieldUI->fieldRes.y > 0)
    {
      bool singleStep = false;
      mParams.cp.t    = mInfo.t;
      FieldParams cp = mParams.cp;
      cp.u = mUnits;
      if(mSingleStepMult != 0.0f)
        {
          cp.u.dt *= mSingleStepMult;
          mSingleStepMult = 0.0f;
          singleStep = true;
        }
      
      if(!g_temp)                          { g_temp = new EMField<CFT>();  }
      if(g_temp->size != mFieldUI->fieldRes) { g_temp->create(mFieldUI->fieldRes); }
      
      EMField<CFT> *src  = reinterpret_cast<EMField<CFT>*>(mStates.back());  // previous field state
      EMField<CFT> *dst  = reinterpret_cast<EMField<CFT>*>(mStates.front()); // oldest state (recycle)
      EMField<CFT> *temp = reinterpret_cast<EMField<CFT>*>(g_temp); // temp intermediate state

      // apply external forces from user
      float3 mposSim = float3{NAN, NAN, NAN};
      if     (mEMView.hovered)  { mposSim = to_cuda(mEMView.mposSim);  }
      else if(mMatView.hovered) { mposSim = to_cuda(mMatView.mposSim); }
      else if(m3DView.hovered)  { mposSim = to_cuda(m3DView.mposSim);  }
      float  cs = mUnits.dL;
      float3 fs = float3{(float)mFieldUI->fieldRes.x, (float)mFieldUI->fieldRes.y, (float)mFieldUI->fieldRes.z};
      float3 mpfi = (mposSim);
      // draw signal
      mParams.rp.sigPenHighlight = false;
      mSigMPos = float3{NAN, NAN, NAN}; 
      if((ImGui::IsKeyDown(GLFW_KEY_LEFT_CONTROL) || ImGui::IsKeyDown(GLFW_KEY_RIGHT_CONTROL)))
        {
          bool hover = m3DView.hovered || mEMView.hovered || mMatView.hovered;
          bool apply = false;
          float3 p = float3{NAN, NAN, NAN};
          if(m3DView.hovered)
            {
              float3 fp = to_cuda(m3DView.mposSim);
              float3 vDepth = float3{(fp.x <= 1 ? 1.0f : (fp.x >= mFieldUI->fieldRes.x-1 ? -1.0f : 0.0f)),
                                     (fp.y <= 1 ? 1.0f : (fp.y >= mFieldUI->fieldRes.y-1 ? -1.0f : 0.0f)),
                                     (fp.z <= 1 ? 1.0f : (fp.z >= mFieldUI->fieldRes.z-1 ? -1.0f : 0.0f)) };
              p = fp + vDepth*mDrawUI->sigPen.depth;
              apply = (m3DView.leftClicked && m3DView.ctrlClick);
            }
          if(mEMView.hovered || mMatView.hovered)
            {
              mpfi.z = mFieldUI->fieldRes.z - 1 - mDrawUI->sigPen.depth; // Z depth relative to top visible layer
              p = float3{mpfi.x, mpfi.y, mpfi.z};
              apply = ((mEMView.leftClicked && mEMView.ctrlClick) ||
                       (mMatView.leftClicked && mMatView.ctrlClick));
            }
          
          if(hover)
            {
              mSigMPos = p; 
              mParams.rp.penPos = p;
              mParams.rp.sigPenHighlight = true;
              mParams.rp.sigPen = mDrawUI->sigPen;
              if(apply) { addSignal(p, *src, mDrawUI->sigPen, cp); }
            }
        }
      // add material
      mParams.rp.matPenHighlight = false;
      mMatMPos = float3{NAN, NAN, NAN}; 
      if((ImGui::IsKeyDown(GLFW_KEY_LEFT_ALT) || ImGui::IsKeyDown(GLFW_KEY_RIGHT_ALT)))
        {
          bool hover = m3DView.hovered || mEMView.hovered || mMatView.hovered;
          float3 p = float3{NAN, NAN, NAN};
          bool apply = false;
          if(m3DView.hovered)
            {
              float3 fp = to_cuda(m3DView.mposSim);
              float3 vDepth = float3{(fp.x <= 1 ? 1.0f : (fp.x >= mFieldUI->fieldRes.x-1 ? -1.0f : 0.0f)),
                                     (fp.y <= 1 ? 1.0f : (fp.y >= mFieldUI->fieldRes.y-1 ? -1.0f : 0.0f)),
                                     (fp.z <= 1 ? 1.0f : (fp.z >= mFieldUI->fieldRes.z-1 ? -1.0f : 0.0f)) };
              p = fp + vDepth*mDrawUI->matPen.depth;
              apply = (m3DView.leftClicked && m3DView.altClick);
            }
          if(mEMView.hovered || mMatView.hovered)
            {
              mpfi.z = mFieldUI->fieldRes.z-1-mDrawUI->matPen.depth; // Z depth relative to top visible layer
              p = float3{mpfi.x, mpfi.y, mpfi.z};
              apply = ((mEMView.leftClicked  && mEMView.altClick) ||
                       (mMatView.leftClicked && mMatView.altClick));
            }

          if(hover)
            {
              mMatMPos = p; 
              mParams.rp.penPos = p;
              mParams.rp.matPenHighlight = true;
              mParams.rp.matPen = mDrawUI->matPen;
              if(apply) { addMaterial(p, *src, mDrawUI->matPen, cp); }
            }
        }
      
      if(mFieldUI->running || singleStep)
        {
          cudaDeviceSynchronize();
          src->copyTo(*temp);// don't overwrite previous state
          dst->clear();
          if(!mFieldUI->running) { std::cout << "SIM STEP --> dt = " << mUnits.dt << "\n"; }
          if(mFieldUI->updateQ)
            { updateCharge  (*temp, *dst, cp); std::swap(temp, dst); if(mParams.debug) { cudaDeviceSynchronize(); getLastCudaError("UPDATE CHARGE FAILED!"); } }
          if(mFieldUI->updateE)
            { updateElectric(*temp, *dst, cp); std::swap(temp, dst); if(mParams.debug) { cudaDeviceSynchronize(); getLastCudaError("UPDATE ELECTRIC FAILED!"); } }
          if(mFieldUI->updateB)
            { updateMagnetic(*temp, *dst, cp); std::swap(temp, dst); if(mParams.debug) { cudaDeviceSynchronize(); getLastCudaError("UPDATE MAGNETIC FAILED!"); } }
          if(mParams.debug) { cudaDeviceSynchronize(); getLastCudaError("EM update failed!"); }

          if(temp != g_temp) { std::swap(temp, dst); } // odd number of steps -- fix pointers
          mStates.pop_front(); mStates.push_back(dst);

          // increment time/frame info
          mInfo.t += mUnits.dt;
          mInfo.uStep++;
          if(mInfo.uStep >= mParams.uSteps) { mInfo.frame++; mInfo.uStep = 0; mNewFrame = true; }
        }
      
      //// render field
      EMField<CFT> *renderSrc = reinterpret_cast<EMField<CFT>*>(mStates.back());
      // render 2D EM views
      if(mDisplayUI->showEMField)  { mEMTex.clear();  renderFieldEM  (*renderSrc,     mEMTex,  mParams.rp, mParams.cp); }
      if(mDisplayUI->showMatField) { mMatTex.clear(); renderFieldMat (renderSrc->mat, mMatTex, mParams.rp, mParams.cp); }
      // render 3D EM view
      Vec2f aspect = Vec2f(m3DView.r.aspect(), 1.0);
      if(aspect.x < 1.0) { aspect.y = 1.0/aspect.x; aspect.x = 1.0; }
      if(mDisplayUI->show3DField) { m3DTex.clear();  raytraceFieldEM(*renderSrc, m3DTex, mCamera, mParams.rp, cp, aspect); }
      
      cudaDeviceSynchronize(); getLastCudaError("Field render failed! (update)");
    }
}

void SimWindow::handleInput(ScreenView &view)
{
  ImGuiStyle &style = ImGui::GetStyle();
  ImGuiIO    &io    = ImGui::GetIO();

  Vec2f mpos    = ImGui::GetMousePos();
  Vec2f mposSim = screenToSim2D(mpos, mSimView2D, view.r);
  view.hovered  = mSimView2D.contains(mposSim);
  if(view.hovered)
    {
      view.clickPos = mpos;
      view.mposSim = float3{(float)mposSim.x-mFieldUI->cp->fp.x, (float)mposSim.y-mFieldUI->cp->fp.y, (float)mDisplayUI->rp->zRange.y};
    }
  else
    { view.mposSim = float3{NAN, NAN, NAN}; }
  
  if(view.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
    {
      view.leftClicked  = true;
      view.shiftClick = (ImGui::IsKeyDown(GLFW_KEY_LEFT_SHIFT)   || ImGui::IsKeyDown(GLFW_KEY_RIGHT_SHIFT));
      view.ctrlClick  = (ImGui::IsKeyDown(GLFW_KEY_LEFT_CONTROL) || ImGui::IsKeyDown(GLFW_KEY_RIGHT_CONTROL));
      view.altClick   = (ImGui::IsKeyDown(GLFW_KEY_LEFT_ALT)     || ImGui::IsKeyDown(GLFW_KEY_RIGHT_ALT));
    }
  else if(ImGui::IsMouseReleased(ImGuiMouseButton_Left)) { view.leftClicked  = false; view.shiftClick = false; view.ctrlClick = false; view.altClick = false; }
  if(view.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
    {
      view.rightClicked = true;
      view.shiftClick = (ImGui::IsKeyDown(GLFW_KEY_LEFT_SHIFT)   || ImGui::IsKeyDown(GLFW_KEY_RIGHT_SHIFT));
      view.ctrlClick  = (ImGui::IsKeyDown(GLFW_KEY_LEFT_CONTROL) || ImGui::IsKeyDown(GLFW_KEY_RIGHT_CONTROL));
      view.altClick   = (ImGui::IsKeyDown(GLFW_KEY_LEFT_ALT)     || ImGui::IsKeyDown(GLFW_KEY_RIGHT_ALT));
    }
  else if(ImGui::IsMouseReleased(ImGuiMouseButton_Right)) { view.rightClicked = false; view.shiftClick = false; view.ctrlClick = false; view.altClick = false; }
  if(view.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Middle))
    {
      view.middleClicked = true;
      view.shiftClick = (ImGui::IsKeyDown(GLFW_KEY_LEFT_SHIFT)   || ImGui::IsKeyDown(GLFW_KEY_RIGHT_SHIFT));
      view.ctrlClick  = (ImGui::IsKeyDown(GLFW_KEY_LEFT_CONTROL) || ImGui::IsKeyDown(GLFW_KEY_RIGHT_CONTROL));
      view.altClick   = (ImGui::IsKeyDown(GLFW_KEY_LEFT_ALT)     || ImGui::IsKeyDown(GLFW_KEY_RIGHT_ALT));
    }
  else if(ImGui::IsMouseReleased(ImGuiMouseButton_Middle)) { view.middleClicked = false; view.shiftClick = false; view.ctrlClick = false; view.altClick = false; }

  if(((view.leftClicked && ImGui::IsMouseDragging(ImGuiMouseButton_Left) && !view.ctrlClick && !view.altClick) ||
      (view.middleClicked && ImGui::IsMouseDragging(ImGuiMouseButton_Middle))))
    {
      ImGuiMouseButton btn = ImGui::IsMouseDragging(ImGuiMouseButton_Left) ? ImGuiMouseButton_Left : ImGuiMouseButton_Middle;
      Vec2f dmp = ImGui::GetMouseDragDelta(btn);
      dmp.x *= -1.0f;
      ImGui::ResetMouseDragDelta(btn);
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
      float2 fs = float2{(float)mFieldUI->fieldRes.x, (float)mFieldUI->fieldRes.y};
      float2 cs = float2{mUnits.dL, mUnits.dL};
      Vec2i fi    = makeV<int2>(floor((float2{mposSim.x, mposSim.y} / cs)));
      Vec2i fiAdj = Vec2i(std::max(0, std::min(mFieldUI->fieldRes.x-1, fi.x)), std::max(0, std::min(mFieldUI->fieldRes.y-1, fi.y)));

      // pull device data
      std::vector<float2> Q  (mStates.size(), float2{NAN, NAN});
      std::vector<float3> QPV(mStates.size(), float3{NAN, NAN, NAN});
      std::vector<float3> QNV(mStates.size(), float3{NAN, NAN, NAN});
      std::vector<float3> E  (mStates.size(), float3{NAN, NAN, NAN});
      std::vector<float3> B  (mStates.size(), float3{NAN, NAN, NAN});
      std::vector<Material<float>> mat(mStates.size(), Material<float>());
      if(fi.x >= 0 && fi.x < mFieldUI->fieldRes.x && fi.y >= 0 && fi.y < mFieldUI->fieldRes.y)
        {
          int zi = mDisplayUI->rp->zRange.y;
          for(int i = 0; i < mStates.size(); i++)
            {
              EMField<CFT> *src = reinterpret_cast<EMField<CFT>*>(mStates[mStates.size()-1-i]);
              if(src)
                {
                  // get data from top displayed layer (TODO: average of layers as well?)
                  cudaMemcpy(&Q[i],   src->Q.dData   + src->Q.idx  (fi.x, fi.y, zi), sizeof(float2), cudaMemcpyDeviceToHost);
                  cudaMemcpy(&QPV[i], src->QPV.dData + src->QPV.idx(fi.x, fi.y, zi), sizeof(float3), cudaMemcpyDeviceToHost);
                  cudaMemcpy(&QNV[i], src->QNV.dData + src->QNV.idx(fi.x, fi.y, zi), sizeof(float3), cudaMemcpyDeviceToHost);
                  cudaMemcpy(&E[i],   src->E.dData   + src->E.idx  (fi.x, fi.y, zi), sizeof(float3), cudaMemcpyDeviceToHost);
                  cudaMemcpy(&B[i],   src->B.dData   + src->B.idx  (fi.x, fi.y, zi), sizeof(float3), cudaMemcpyDeviceToHost);
                  cudaMemcpy(&mat[i], src->mat.dData + src->mat.idx(fi.x, fi.y, zi), sizeof(Material<float>), cudaMemcpyDeviceToHost);
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
                if(mat[i].vacuum())
                  {
                    ImGui::Text(" Material (vacuum): ep = %10.4f", mUnits.vacuum().permittivity);
                    ImGui::Text("                    mu = %10.4f", mUnits.vacuum().permeability);
                    ImGui::Text("                   sig = %10.4f", mUnits.vacuum().conductivity);
                  }
                else
                  {
                    ImGui::Text(" Material:          ep = %10.4f", mat[i].permittivity);
                    ImGui::Text("                    mu = %10.4f", mat[i].permeability);
                    ImGui::Text("                   sig = %10.4f", mat[i].conductivity);
                  }
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
  
  Vec2f  aspect = Vec2f(view.r.aspect(), 1.0); if(aspect.x < 1.0) { aspect.y = 1.0/aspect.x; aspect.x = 1.0; }
  float3 fs     = float3{(CFT)mFieldUI->fieldRes.x, (CFT)mFieldUI->fieldRes.y, (CFT)mFieldUI->fieldRes.z};
  float3 fSize  = fs*mUnits.dL;
  Ray<CFT> ray; // ray projected from mouse position within view
  Vec2f mpos = ImGui::GetMousePos();
  view.hovered = view.r.contains(mpos);
  Vec3f fpLast = view.mposSim;
  if(view.hovered)
    {
      view.clickPos = mpos;
      ray = mCamera.castRay(to_cuda(Vec2f((mpos - view.r.p1)/view.r.size())), float2{aspect.x, aspect.y});

      Vec2f tp = cubeIntersectHost(Vec3f(mFieldUI->cp->fp*mUnits.dL), Vec3f(fSize), ray);

      if(tp.x >= 0.0) // tmin
        {
          Vec3f wp = ray.pos + ray.dir*(tp.x+0.0001); // world-space pos of field outer intersection
          Vec3f fp = (wp - mFieldUI->cp->fp*mUnits.dL) / fSize * fs;
          view.mposSim = fp;
        }
      else { view.mposSim = float3{NAN, NAN, NAN}; }
    }
  else { view.mposSim = float3{NAN, NAN, NAN}; }

  if(view.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
    {
      view.leftClicked = true;
      view.shiftClick  = (ImGui::IsKeyDown(GLFW_KEY_LEFT_SHIFT)   || ImGui::IsKeyDown(GLFW_KEY_RIGHT_SHIFT));
      view.ctrlClick   = (ImGui::IsKeyDown(GLFW_KEY_LEFT_CONTROL) || ImGui::IsKeyDown(GLFW_KEY_RIGHT_CONTROL));
      view.altClick    = (ImGui::IsKeyDown(GLFW_KEY_LEFT_ALT)     || ImGui::IsKeyDown(GLFW_KEY_RIGHT_ALT));
    }
  else if(ImGui::IsMouseReleased(ImGuiMouseButton_Left)) { view.leftClicked   = false; view.shiftClick = false; view.ctrlClick = false; view.altClick = false;  }
  if(view.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
    {
      view.rightClicked = true; 
      view.shiftClick   = (ImGui::IsKeyDown(GLFW_KEY_LEFT_SHIFT)   || ImGui::IsKeyDown(GLFW_KEY_RIGHT_SHIFT));
      view.ctrlClick    = (ImGui::IsKeyDown(GLFW_KEY_LEFT_CONTROL) || ImGui::IsKeyDown(GLFW_KEY_RIGHT_CONTROL));
      view.altClick     = (ImGui::IsKeyDown(GLFW_KEY_LEFT_ALT)     || ImGui::IsKeyDown(GLFW_KEY_RIGHT_ALT));
    }
  else if(ImGui::IsMouseReleased(ImGuiMouseButton_Right)) { view.rightClicked  = false; view.shiftClick = false; view.ctrlClick = false; view.altClick = false; }
  if(view.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Middle))
    {
      view.middleClicked = true; 
      view.shiftClick    = (ImGui::IsKeyDown(GLFW_KEY_LEFT_SHIFT)   || ImGui::IsKeyDown(GLFW_KEY_RIGHT_SHIFT));
      view.ctrlClick     = (ImGui::IsKeyDown(GLFW_KEY_LEFT_CONTROL) || ImGui::IsKeyDown(GLFW_KEY_RIGHT_CONTROL));
      view.altClick      = (ImGui::IsKeyDown(GLFW_KEY_LEFT_ALT)     || ImGui::IsKeyDown(GLFW_KEY_RIGHT_ALT));
    }
  else if(ImGui::IsMouseReleased(ImGuiMouseButton_Middle)) { view.middleClicked = false; view.shiftClick = false; view.ctrlClick = false; view.altClick = false; }

  float3 upBasis = float3{0.0, 1.0, 0.0};

  CFT shiftMult = (io.KeyShift   ? 0.1f  : 1.0f);
  CFT ctrlMult  = (((ImGui::IsKeyDown(GLFW_KEY_LEFT_CONTROL) || ImGui::IsKeyDown(GLFW_KEY_RIGHT_CONTROL)) ? 4.0f  : 1.0f));
  CFT altMult   = (((ImGui::IsKeyDown(GLFW_KEY_LEFT_ALT)     || ImGui::IsKeyDown(GLFW_KEY_RIGHT_ALT))     ? 16.0f : 1.0f));
  CFT keyMult   = shiftMult * ctrlMult * altMult;
  
  Vec2f viewSize = m3DView.r.size();
  if(((ImGui::IsMouseDragging(ImGuiMouseButton_Left) && view.leftClicked && !view.ctrlClick && !view.altClick) ||
      (ImGui::IsMouseDragging(ImGuiMouseButton_Middle) && view.middleClicked)))
    { // pan camera
      ImGuiMouseButton btn = ImGui::IsMouseDragging(ImGuiMouseButton_Left) ? ImGuiMouseButton_Left : ImGuiMouseButton_Middle;
      Vec2f dmp = ImGui::GetMouseDragDelta(btn);
      ImGui::ResetMouseDragDelta(btn);
      dmp.x *= -1.0f;
      // CFT mult = length(mCamera.pos)*1.9f*tan(mCamera.fov/2.0f*M_PI/180.0f);
      CFT mult = (2.0*tan(mCamera.fov/2.0f*M_PI/180.0f));
      float3 fpos = float3{mFieldUI->cp->fp.x, mFieldUI->cp->fp.y, mFieldUI->cp->fp.z};
      float3 cpos = float3{mCamera.pos.x, mCamera.pos.y, mCamera.pos.z};
      float3 fsize = float3{(float)mFieldUI->fieldRes.x, (float)mFieldUI->fieldRes.y, (float)mFieldUI->fieldRes.z};
      CFT lengthMult = (length(cpos-fpos) +
                        length(cpos - (fpos + fsize)) +
                        length(cpos-(fpos + fsize/2.0f)) +
                        length(cpos - (fpos + fsize/2.0f)) +
                        length(cpos-float3{fpos.x, fpos.y + fsize.y/2.0f, fpos.z}) +
                        length(cpos-float3{fpos.x + fsize.y/2.0f, fpos.y, fpos.z} ))/6.0f;

      mCamera.pos += (mCamera.right*dmp.x/viewSize.x*aspect.x + mCamera.up*dmp.y/viewSize.y*aspect.y)*lengthMult*mult*0.8;
    }
  if(ImGui::IsMouseDragging(ImGuiMouseButton_Right) && view.rightClicked)
    { // rotate camera
      Vec2f dmp = Vec2f(ImGui::GetMouseDragDelta(ImGuiMouseButton_Right));
      ImGui::ResetMouseDragDelta(ImGuiMouseButton_Right);
      dmp = -dmp;
      float2 rAngles = float2{dmp.x, dmp.y} / float2{viewSize.x, viewSize.y} * 6.0 * tan(mCamera.fov/2*M_PI/180.0) * shiftMult;
      float3 rOffset = float3{(CFT)mFieldUI->fieldRes.x, (CFT)mFieldUI->fieldRes.y, (CFT)mFieldUI->fieldRes.z}*mUnits.dL / 2.0;
      mCamera.pos -= rOffset; // offset to center rotation on field center (if cp->fp is 0)
      mCamera.rotate(rAngles);
      mCamera.pos += rOffset;
    }
  if(view.hovered && std::abs(io.MouseWheel) > 0.0f)
    {
      // mCamera.pos += mCamera.dir*keyMult*(io.MouseWheel/20.0)*mUnits.dL*250.0;
      mCamera.pos += ray.dir*shiftMult*(io.MouseWheel/20.0)*length(mCamera.pos);
    }
}



void SimWindow::drawVectorField(const Rect2f &sr)
{
  ImGuiStyle &style = ImGui::GetStyle();
  ImGuiIO    &io    = ImGui::GetIO();
  Vec2f mpos = ImGui::GetMousePos();
  Vec2f fp   = screenToSim2D(mpos, mSimView2D, sr);
  
  // draw vector field data
  EMField<CFT> *src = reinterpret_cast<EMField<CFT>*>(mStates.back());
  if(src && (mEMView.hovered || mMatView.hovered) && mParams.vp.drawVectors && mFieldDrawList)
    {
      Vec2i fi    = makeV<int2>(float2{floor(fp.x), floor(fp.y)});
      Vec2f fo    = fp - fi;
      Vec2i fiAdj = Vec2i(std::max(0, std::min(mFieldUI->fieldRes.x, fi.x)), std::max(0, std::min(mFieldUI->fieldRes.y, fi.y)));

      int vRad     = mParams.vp.vecMRadius;
      int cRad     = mParams.vp.vecCRadius;
      int vSpacing = mParams.vp.vecSpacing;
      if(vRad > cRad) { vSpacing = std::max(vSpacing, (int)ceil(vRad/(float)cRad)); }
      float viewScale = max(mSimView2D.size());
      
      int2 iMin = int2{std::max(fi.x-vRad*vSpacing, 0), std::max(fi.y-vRad*vSpacing, 0)};
      int2 iMax = int2{std::min(fi.x+vRad*vSpacing, mFieldUI->fieldRes.x-1)+1,
                       std::min(fi.y+vRad*vSpacing, mFieldUI->fieldRes.y-1)+1};
      
      int2 iStart = int2{0, 0};
      int2 iEnd   = int2{(iMax.x - iMin.x)/vSpacing, (iMax.y - iMin.y)/vSpacing};

      src->E.pullData(); src->B.pullData();
      float avgE = 0.0f; float avgB = 0.0f;
      
      for(int ix = iStart.x; ix <= iEnd.x; ix++)
        for(int iy = iStart.y; iy <= iEnd.y; iy++)
          {
            int xi = iMin.x + ix*vSpacing; int yi = iMin.y + iy*vSpacing;
            float2 dp = float2{(float)(xi-fi.x), (float)(yi-fi.y)};// * float2{mUnits.dL, mUnits.dL};
            if(dot(dp, dp) <= (float)(vRad*vRad))
              {
                int i = src->idx(xi, yi);
                Vec2f sp = simToScreen2D(Vec2f(xi+0.5f, yi+0.5f)*Vec2f(mUnits.dL, mUnits.dL), mSimView2D, sr);
                
                Vec3f vE; Vec3f vB;
                if(mParams.vp.smoothVectors)
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
                        vE = blerp(E00, E01, E10, E11, fo); // / mUnits.dL; // scale by cell size
                        vB = blerp(B00, B01, B10, B11, fo); // / mUnits.dL; // scale by cell size
                      }
                  }
                else
                  {
                    Vec2f sampleP = Vec2f(xi, yi);
                    if(sampleP.x >= 0 && sampleP.x < src->size.x && sampleP.y >= 0 && sampleP.y < src->size.y)
                      {
                        sp = simToScreen2D((sampleP+0.5f), mSimView2D, sr);
                        i = src->idx(sampleP.x, sampleP.y);
                        vE = src->E.hData[i]; // / mParams.cp.u.dL;
                        vB = src->B.hData[i]; // / mParams.cp.u.dL;
                      }
                  }
                
                Vec2f dpE = simToScreen2D(Vec2f(vE.x, vE.y), mSimView2D, sr, true)*mParams.vp.vecMultE;
                Vec2f dpB = simToScreen2D(Vec2f(vB.x, vB.y), mSimView2D, sr, true)*mParams.vp.vecMultB;
                Vec3f dpE3 = Vec3f(dpE.x, dpE.y, 0.0f);
                Vec3f dpB3 = Vec3f(dpB.x, dpB.y, 0.0f);
                float lw     = (mParams.vp.vecLineW   / viewScale);
                float bw     = (mParams.vp.vecBorderW / viewScale);
                float lAlpha = mParams.vp.vecAlpha;
                float bAlpha = (lAlpha + mParams.vp.vecBAlpha)/2.0f;
                float vLenE = length(dpE3);
                float vLenB = length(dpB3);
                float shear0 = 0.1f;//0.5f; 
                float shear1 = 1.0f;

                float magE = length(screenToSim2D(Vec2f(vLenE, 0), mSimView2D, sr, true) ); // / float2{(float)mParams.cp.u.dL, (float)mParams.cp.u.dL});
                float magB = length(screenToSim2D(Vec2f(vLenB, 0), mSimView2D, sr, true) ); // / float2{(float)mParams.cp.u.dL, (float)mParams.cp.u.dL});
                Vec4f Ecol1 = mParams.rp.Ecol;
                Vec4f Bcol1 = mParams.rp.Bcol;
                Vec4f Ecol = Vec4f(Ecol1.x, Ecol1.y, Ecol1.z, lAlpha);
                Vec4f Bcol = Vec4f(Bcol1.x, Bcol1.y, Bcol1.z, lAlpha);
                
                if(!mParams.vp.borderedVectors)
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
      // std::cout << "VEC AVG E: " << avgE/((iMax.x - iMin.x)*(iMax.y - iMin.y)) << "\n";
      // std::cout << "VEC AVG B: " << avgB/((iMax.x - iMin.x)*(iMax.y - iMin.y)) << "\n";
    }
  
  // TODO: CudaVBO
  //loadVectorField(mVecBuffer);
}

void SimWindow::draw(const Vec2f &frameSize)
{
  if(frameSize.x <= 0 || frameSize.y <= 0) { return; }
  mFrameSize = frameSize;
  mInfo.fps  = calcFps();
  
  ImGuiStyle &style = ImGui::GetStyle();
  
  //// draw (imgui)
  ImGuiWindowFlags wFlags = (ImGuiWindowFlags_NoTitleBar      | ImGuiWindowFlags_NoCollapse        |
                             ImGuiWindowFlags_NoMove          | ImGuiWindowFlags_NoResize          |
                             ImGuiWindowFlags_NoScrollbar     | ImGuiWindowFlags_NoScrollWithMouse |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus );
  ImGui::SetNextWindowPos(Vec2f(0,0));
  ImGui::SetNextWindowSize(Vec2f(mFrameSize.x, mFrameSize.y));
  ImGui::PushStyleColor(ImGuiCol_WindowBg, Vec4f(0.05f, 0.05f, 0.05f, 1.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,   0); // square frames by default
  ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,    0);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,    Vec2f(10, 10));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, Vec2f(3, 3));
  ImGui::Begin("##mainView", nullptr, wFlags); // ImGui window covering full application window
  {
    ImGui::PushFont(mainFont);
    Vec2f p0 = ImGui::GetCursorScreenPos();
    
    Vec2f settingsSize = mTabs->getSize();
    mDisplaySize = (mFrameSize - Vec2f(settingsSize.x, 0.0f) - 2.0f*Vec2f(style.WindowPadding) - Vec2f(style.ItemSpacing.x, 0.0f));

    int numViews = ((int)mDisplayUI->showEMField + (int)mDisplayUI->showMatField + (int)mDisplayUI->show3DField);

    if(numViews == 1)
      {
        Rect2f r0 =  Rect2f(p0, p0+mDisplaySize); // whole display
        if(mDisplayUI->showEMField)  { mEMView.r  = r0; }
        if(mDisplayUI->show3DField)  { m3DView.r  = r0; }
        if(mDisplayUI->showMatField) { mMatView.r = r0; }
      }
    else if(numViews == 2)
      {
        Rect2f r0 =  Rect2f(p0, p0+Vec2f(mDisplaySize.x/2.0f, mDisplaySize.y));    // left side
        Rect2f r1 =  Rect2f(p0+Vec2f(mDisplaySize.x/2.0f, 0.0f), p0+mDisplaySize); // right side
        int n = 0;
        if(mDisplayUI->showEMField)  { mEMView.r  = (n==0 ? r0 : r1); n++; }
        if(mDisplayUI->show3DField)  { m3DView.r  = (n==0 ? r0 : r1); n++; }
        if(mDisplayUI->showMatField) { mMatView.r = (n==0 ? r0 : r1); n++; }
      }
    else
      {
        Rect2f r0 =  Rect2f(p0, p0+mDisplaySize/2.0f);              // top-left quarter
        Rect2f r1 =  Rect2f(p0+Vec2f(mDisplaySize.x/2.0f, 0.0f),    // top-right quarter
                            p0+mDisplaySize);
        Rect2f r2 =  Rect2f(p0+Vec2f(0.0f, mDisplaySize.y/2.0f),
                            p0+Vec2f(mDisplaySize.x/2.0f, mDisplaySize.y)); // bottom-left quarter
        int n = 0;
        if(mDisplayUI->showEMField)  { mEMView.r  = (n==0 ? r0 : (n==1 ? r1 : r2)); n++; }
        if(mDisplayUI->show3DField)  { m3DView.r  = (n==0 ? r0 : (n==1 ? r1 : r2)); n++; }
        if(mDisplayUI->showMatField) { mMatView.r = (n==0 ? r0 : (n==1 ? r1 : r2)); n++; }
      }

    // adjust sim view aspect ratio if window size has changed
    float simAspect  = mSimView2D.aspect();
    float dispAspect = mEMView.r.aspect();
    if(dispAspect > 1.0f) { mSimView2D.scale(Vec2f(dispAspect/simAspect, 1.0f)); }
    else                  { mSimView2D.scale(Vec2f(1.0f, simAspect/dispAspect)); }
    
    // draw rendered views of simulation on screen
    ImGui::BeginChild("##simView", mDisplaySize, false, wFlags);
    {
      // EM view
      if(mDisplayUI->showEMField)
        {
          handleInput(mEMView);
          ImGui::SetCursorScreenPos(mEMView.r.p1);
          ImGui::PushStyleColor(ImGuiCol_ChildBg, Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
          ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, Vec2f(0, 0));
          ImGui::BeginChild("##emView", mEMView.r.size(), true, wFlags);
          {
            mFieldDrawList = ImGui::GetWindowDrawList();
            Vec2f fp = Vec2f(mFieldUI->cp->fp.x, mFieldUI->cp->fp.y);
            Vec2f fScreenPos  = simToScreen2D(fp, mSimView2D, mEMView.r);
            Vec2f fCursorPos  = simToScreen2D(fp + Vec2f(0.0f, mFieldUI->fieldRes.y*mUnits.dL), mSimView2D, mEMView.r);
            Vec2f fScreenSize = simToScreen2D(makeV<float3>(mFieldUI->fieldRes)*mUnits.dL, mSimView2D, mEMView.r, true);
            Vec2f t0(0.0f, 1.0f); Vec2f t1(1.0f, 0.0f);
            mEMTex.bind();
            ImGui::SetCursorScreenPos(fCursorPos);
            ImGui::Image(reinterpret_cast<ImTextureID>(mEMTex.texId()), fScreenSize, t0, t1, ImColor(Vec4f(1,1,1,1)));
            mEMTex.release();
            
            drawVectorField(mEMView.r);
          }
          ImGui::EndChild();
          ImGui::PopStyleVar();
          ImGui::PopStyleColor();
        }
      
      // Material view
      if(mDisplayUI->showMatField)
        {
          handleInput(mMatView);
          ImGui::SetCursorScreenPos(mMatView.r.p1);
          ImGui::PushStyleColor(ImGuiCol_ChildBg, Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
          ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, Vec2f(0, 0));
          ImGui::BeginChild("##matView", mMatView.r.size(), true, wFlags);
          {
            mFieldDrawList = ImGui::GetWindowDrawList();
            Vec2f fp = Vec2f(mFieldUI->cp->fp.x, mFieldUI->cp->fp.y);
            Vec2f fScreenPos  = simToScreen2D(fp, mSimView2D, mMatView.r);
            Vec2f fCursorPos  = simToScreen2D(fp + Vec2f(0.0f, mFieldUI->fieldRes.y*mUnits.dL), mSimView2D, mMatView.r);
            Vec2f fScreenSize = simToScreen2D(makeV<float3>(mFieldUI->fieldRes)*mUnits.dL, mSimView2D, mMatView.r, true);
            Vec2f t0(0.0f, 1.0f); Vec2f t1(1.0f, 0.0f);
            mMatTex.bind();
            ImGui::SetCursorScreenPos(fCursorPos);
            ImGui::Image(reinterpret_cast<ImTextureID>(mMatTex.texId()), fScreenSize, t0, t1, ImColor(Vec4f(1,1,1,1)));
            mMatTex.release();
            drawVectorField(mMatView.r);
          }
          ImGui::EndChild();
          ImGui::PopStyleVar();
          ImGui::PopStyleColor();
        }
      
      // Raytraced 3D view
      if(mDisplayUI->show3DField)
        {
          cudaDeviceSynchronize();
          handleInput3D(m3DView);
          ImGui::SetCursorScreenPos(m3DView.r.p1);
          ImGui::PushStyleColor(ImGuiCol_ChildBg, Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
          ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, Vec2f(0, 0));
          ImGui::BeginChild("##3DView", m3DView.r.size(), true, wFlags);
          {
            mFieldDrawList = ImGui::GetWindowDrawList();
            Vec2f t0(0.0f, 1.0f); Vec2f t1(1.0f, 0.0f);
            m3DTex.bind();
            ImGui::SetCursorScreenPos(m3DView.r.p1);
            ImGui::Image(reinterpret_cast<ImTextureID>(m3DTex.texId()), m3DView.r.size(), t0, t1, ImColor(Vec4f(1,1,1,1)));
            m3DTex.release();
            // drawVectorField(m3DView.r);

            ImDrawList *drawList = ImGui::GetWindowDrawList();
            Vec2f mpos = ImGui::GetMousePos();
            Vec2f aspect = Vec2f(m3DView.r.aspect(), 1.0); if(aspect.x < 1.0) { aspect.y = 1.0/aspect.x; aspect.x = 1.0; }
            Vec2f vSize = m3DView.r.size();
            Vec2f aOffset = -(aspect.x > 1 ? Vec2f(vSize.x/aspect.x - vSize.x, 0.0f) : Vec2f(0.0f, vSize.y/aspect.y - vSize.y))/2.0f;

            mCamera.calculate();
            
            // draw X/Y/Z axes at origin (R/G/B)
            if(mDisplayUI->drawAxes)
              {
                // axis points
                float scale = max(mFieldUI->fieldRes)*mUnits.dL*0.25f;
                Vec3f WO0 = Vec3f(0,0,0); // origin
                Vec3f Wpx = WO0 + Vec3f(scale, 0, 0);
                Vec3f Wpy = WO0 + Vec3f(0, scale, 0);
                Vec3f Wpz = WO0 + Vec3f(0, 0, scale);
                // transform
                bool oClipped = false; bool xClipped = false; bool yClipped = false; bool zClipped = false;
                
                Vec2f Sorigin = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(WO0, aspect, &oClipped)) * m3DView.r.size()/aspect + aOffset;
                Vec2f Spx     = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(Wpx, aspect, &xClipped)) * m3DView.r.size()/aspect + aOffset;
                Vec2f Spy     = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(Wpy, aspect, &yClipped)) * m3DView.r.size()/aspect + aOffset;
                Vec2f Spz     = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(Wpz, aspect, &zClipped)) * m3DView.r.size()/aspect + aOffset;
                // draw axes
                if(!oClipped || !xClipped) { drawList->AddLine(Sorigin, Spx, ImColor(Vec4f(1, 0, 0, 0.5f)), 2.0f); }
                if(!oClipped || !yClipped) { drawList->AddLine(Sorigin, Spy, ImColor(Vec4f(0, 1, 0, 0.5f)), 2.0f); }
                if(!oClipped || !zClipped) { drawList->AddLine(Sorigin, Spz, ImColor(Vec4f(0, 0, 1, 0.5f)), 2.0f); }
              }

            // draw outline around field
            if(mDisplayUI->drawOutline)
              {
                // axis points
                Vec3f Wfp0 = Vec3f(mFieldUI->cp->fp.x,    mFieldUI->cp->fp.y,    mFieldUI->cp->fp.z)*mUnits.dL;
                Vec3f Wfs  = Vec3f(mFieldUI->fieldRes.x, mFieldUI->fieldRes.y, mFieldUI->fieldRes.z)*mUnits.dL;

                Vec3f W000  = Vec3f(Wfp0.x,       Wfp0.y,       Wfp0.z);
                Vec3f W001  = Vec3f(Wfp0.x,       Wfp0.y,       Wfp0.z+Wfs.z);
                Vec3f W010  = Vec3f(Wfp0.x,       Wfp0.y+Wfs.y, Wfp0.z);
                Vec3f W011  = Vec3f(Wfp0.x,       Wfp0.y+Wfs.y, Wfp0.z+Wfs.z);
                Vec3f W100  = Vec3f(Wfp0.x+Wfs.x, Wfp0.y,       Wfp0.z);
                Vec3f W101  = Vec3f(Wfp0.x+Wfs.x, Wfp0.y,       Wfp0.z+Wfs.z);
                Vec3f W110  = Vec3f(Wfp0.x+Wfs.x, Wfp0.y+Wfs.y, Wfp0.z);
                Vec3f W111  = Vec3f(Wfp0.x+Wfs.x, Wfp0.y+Wfs.y, Wfp0.z+Wfs.z);
                
                // transform
                bool C000 = false; bool C001 = false; bool C010 = false; bool C011 = false;
                bool C100 = false; bool C101 = false; bool C110 = false; bool C111 = false;
                // NOTE: hacky transformations...
                Vec2f S000 = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W000, aspect, &C000)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S001 = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W001, aspect, &C001)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S010 = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W010, aspect, &C010)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S011 = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W011, aspect, &C011)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S100 = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W100, aspect, &C100)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S101 = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W101, aspect, &C101)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S110 = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W110, aspect, &C110)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S111 = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W111, aspect, &C111)) * m3DView.r.size()/aspect + aOffset;
                
                // XY plane (front)
                if(!C000 || !C001) { drawList->AddLine(S000, S001, ImColor(Vec4f(0.5f, 0.5f, 0.5f, 0.3f)), 1.0f); }
                if(!C001 || !C011) { drawList->AddLine(S001, S011, ImColor(Vec4f(0.5f, 0.5f, 0.5f, 0.3f)), 1.0f); }
                if(!C011 || !C010) { drawList->AddLine(S011, S010, ImColor(Vec4f(0.5f, 0.5f, 0.5f, 0.3f)), 1.0f); }
                if(!C010 || !C000) { drawList->AddLine(S010, S000, ImColor(Vec4f(0.5f, 0.5f, 0.5f, 0.3f)), 1.0f); }
                // XY plane (back)                
                if(!C100 || !C101) { drawList->AddLine(S100, S101, ImColor(Vec4f(0.5f, 0.5f, 0.5f, 0.3f)), 1.0f); }
                if(!C101 || !C111) { drawList->AddLine(S101, S111, ImColor(Vec4f(0.5f, 0.5f, 0.5f, 0.3f)), 1.0f); }
                if(!C111 || !C110) { drawList->AddLine(S111, S110, ImColor(Vec4f(0.5f, 0.5f, 0.5f, 0.3f)), 1.0f); }
                if(!C110 || !C100) { drawList->AddLine(S110, S100, ImColor(Vec4f(0.5f, 0.5f, 0.5f, 0.3f)), 1.0f); }
                // Z connections
                if(!C000 || !C100) { drawList->AddLine(S000, S100, ImColor(Vec4f(0.5f, 0.5f, 0.5f, 0.3f)), 1.0f); }
                if(!C001 || !C101) { drawList->AddLine(S001, S101, ImColor(Vec4f(0.5f, 0.5f, 0.5f, 0.3f)), 1.0f); }
                if(!C011 || !C111) { drawList->AddLine(S011, S111, ImColor(Vec4f(0.5f, 0.5f, 0.5f, 0.3f)), 1.0f); }
                if(!C010 || !C110) { drawList->AddLine(S010, S110, ImColor(Vec4f(0.5f, 0.5f, 0.5f, 0.3f)), 1.0f); }
              }

            // draw positional axes of active signal pen 
            if(!isnan(mSigMPos))
              {
                //Vec3f W000  = Vec3f(mSigMPos);
                Vec3f W001n  = Vec3f(mFieldUI->cp->fp.x, mSigMPos.y, mSigMPos.z);
                Vec3f W010n  = Vec3f(mSigMPos.x, mFieldUI->cp->fp.y, mSigMPos.z);
                Vec3f W100n  = Vec3f(mSigMPos.x, mSigMPos.y, mFieldUI->cp->fp.x);
                Vec3f W001p  = Vec3f(mFieldUI->cp->fp.x + mFieldUI->fieldRes.x, mSigMPos.y, mSigMPos.z);
                Vec3f W010p  = Vec3f(mSigMPos.x, mFieldUI->cp->fp.y + mFieldUI->fieldRes.y, mSigMPos.z);
                Vec3f W100p  = Vec3f(mSigMPos.x, mSigMPos.y, mFieldUI->cp->fp.z + mFieldUI->fieldRes.z);
                bool C001n = false; bool C010n = false; bool C100n = false;
                bool C001p = false; bool C010p = false; bool C100p = false;
                // Vec2f S000 = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W000, aspect, &C000)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S001n = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W001n, aspect, &C001n)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S010n = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W010n, aspect, &C010n)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S100n = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W100n, aspect, &C100n)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S001p = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W001p, aspect, &C001p)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S010p = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W010p, aspect, &C010p)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S100p = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W100p, aspect, &C100p)) * m3DView.r.size()/aspect + aOffset;
                if(!C001n || !C001p)
                  {
                    drawList->AddLine(S001n, S001p, ImColor(Vec4f(0.5f, 0.5f, 0.5f, 0.6f)), 2.0f);
                    drawList->AddCircleFilled(S001n, 3, ImColor(Vec4f(1, 0, 0, 0.6f)), 6);
                    drawList->AddCircleFilled(S001p, 3, ImColor(Vec4f(1, 0, 0, 0.6f)), 6);
                  }
                if(!C010n || !C010p)
                  {
                    drawList->AddLine(S010n, S010p, ImColor(Vec4f(0.5f, 0.5f, 0.5f, 0.6f)), 2.0f);
                    drawList->AddCircleFilled(S010n, 3, ImColor(Vec4f(0, 1, 0, 0.6)), 6);
                    drawList->AddCircleFilled(S010p, 3, ImColor(Vec4f(0, 1, 0, 0.6)), 6);
                  }
                if(!C100n || !C100p)
                  {
                    drawList->AddLine(S100n, S100p,     ImColor(Vec4f(0.5f, 0.5f, 0.5f, 0.6f)), 2.0f);
                    drawList->AddCircleFilled(S100n, 3, ImColor(Vec4f(0, 0, 1, 0.6f)), 6);
                    drawList->AddCircleFilled(S100p, 3, ImColor(Vec4f(0, 1.0f, 1.0f, 0.6f)), 6);
                  }
              }
            
            // draw positional axes of active material pen 
            if(!isnan(mMatMPos))
              {
                //Vec3f W000  = Vec3f(mMatMPos);
                Vec3f W001n  = Vec3f(mFieldUI->cp->fp.x, mMatMPos.y, mMatMPos.z);
                Vec3f W010n  = Vec3f(mMatMPos.x, mFieldUI->cp->fp.y, mMatMPos.z);
                Vec3f W100n  = Vec3f(mMatMPos.x, mMatMPos.y, mFieldUI->cp->fp.x);
                Vec3f W001p  = Vec3f(mFieldUI->cp->fp.x + mFieldUI->fieldRes.x, mMatMPos.y, mMatMPos.z);
                Vec3f W010p  = Vec3f(mMatMPos.x, mFieldUI->cp->fp.y + mFieldUI->fieldRes.y, mMatMPos.z);
                Vec3f W100p  = Vec3f(mMatMPos.x, mMatMPos.y, mFieldUI->cp->fp.z + mFieldUI->fieldRes.z);
                bool C001n = false; bool C010n = false; bool C100n = false;
                bool C001p = false; bool C010p = false; bool C100p = false;
                // Vec2f S000 = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W000, aspect, &C000)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S001n = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W001n, aspect, &C001n)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S010n = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W010n, aspect, &C010n)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S100n = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W100n, aspect, &C100n)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S001p = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W001p, aspect, &C001p)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S010p = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W010p, aspect, &C010p)) * m3DView.r.size()/aspect + aOffset;
                Vec2f S100p = m3DView.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W100p, aspect, &C100p)) * m3DView.r.size()/aspect + aOffset;
                if(!C001n || !C001p)
                  {
                    drawList->AddLine(S001n, S001p, ImColor(Vec4f(0.5f, 0.5f, 0.5f, 0.6f)), 2.0f);
                    drawList->AddCircleFilled(S001n, 3, ImColor(Vec4f(1, 0, 0, 0.6f)), 6);
                    drawList->AddCircleFilled(S001p, 3, ImColor(Vec4f(1, 0, 0, 0.6f)), 6);
                  }
                if(!C010n || !C010p)
                  {
                    drawList->AddLine(S010n, S010p, ImColor(Vec4f(0.5f, 0.5f, 0.5f, 0.6f)), 2.0f);
                    drawList->AddCircleFilled(S010n, 3, ImColor(Vec4f(0, 1, 0, 0.6)), 6);
                    drawList->AddCircleFilled(S010p, 3, ImColor(Vec4f(0, 1, 0, 0.6)), 6);
                  }
                if(!C100n || !C100p)
                  {
                    drawList->AddLine(S100n, S100p,     ImColor(Vec4f(0.5f, 0.5f, 0.5f, 0.6f)), 2.0f);
                    drawList->AddCircleFilled(S100n, 3, ImColor(Vec4f(0, 0, 1, 0.6f)), 6);
                    drawList->AddCircleFilled(S100p, 3, ImColor(Vec4f(0, 1.0f, 1.0f, 0.6f)), 6);
                  }
              }
            
          }
          ImGui::EndChild();
          ImGui::PopStyleVar();
          ImGui::PopStyleColor();
        }
    }
    ImGui::EndChild();

    if(mEMView.hovered)  { mMouseSimPos = Vec2f(mEMView.mposSim.x,  mEMView.mposSim.y);  }
    if(mMatView.hovered) { mMouseSimPos = Vec2f(mMatView.mposSim.x, mMatView.mposSim.y); }
    if(m3DView.hovered)  { mMouseSimPos = Vec2f(m3DView.mposSim.x,  m3DView.mposSim.y);  }

    // setting tabs
    ImGui::SameLine();
    ImGui::BeginGroup();
    {
      Vec2f sp0 = Vec2f(mFrameSize.x - settingsSize.x - style.WindowPadding.x, ImGui::GetCursorPos().y);
      ImGui::SetCursorPos(sp0);
      
      mTabs->setLength(mDisplaySize.y);
      mTabs->draw();
      
      settingsSize = mTabs->getSize() + Vec2f(2.0f*style.WindowPadding.x + style.ScrollbarSize, 0.0f) + Vec2f(mTabs->getBarWidth(), 0.0f);
          
      Vec2f minSize = Vec2f(512+settingsSize.x+style.ItemSpacing.x, 512) + Vec2f(style.WindowPadding)*2.0f + Vec2f(style.FramePadding)*2.0f;
      glfwSetWindowSizeLimits(mWindow, (int)minSize.x, (int)minSize.y, GLFW_DONT_CARE, GLFW_DONT_CARE);
      ImGui::SetWindowSize(settingsSize + Vec2f(style.ScrollbarSize, 0.0f));
    }
    ImGui::EndGroup();


    // debug overlay
    if(mParams.debug)
      {
        Vec2f aspect3D = Vec2f(m3DView.r.aspect(), 1.0); if(aspect3D.x < 1.0) { aspect3D.y = 1.0/aspect3D.x; aspect3D.x = 1.0; }
            
        ImDrawList *drawList = ImGui::GetForegroundDrawList();
        ImGui::PushClipRect(Vec2f(0,0), mFrameSize, false);
        ImGui::SetCursorPos(Vec2f(10.0f, 10.0f));
        ImGui::PushStyleColor(ImGuiCol_ChildBg, Vec4f(0.0f, 0.0f, 0.0f, 0.0f));
        ImGui::BeginChild("##debugOverlay", mDisplaySize, false, wFlags);
        {
          ImGui::PushFont(titleFontB);
          ImGui::Text("%.2f FPS", mInfo.fps);
          ImGui::PopFont();
          ImGui::Spacing();
          ImGui::Text("t =  %f  (sim time %.3fx real time)", mInfo.t, mFieldUI->running ? (fpsLast*mUnits.dt) : 0.0f);
          ImGui::Text("Mouse Sim Pos: < %f, %f>",              mMouseSimPos.x,        mMouseSimPos.y);
          ImGui::Text("SimView:       < %f, %f> : < %f, %f >", mSimView2D.p1.x,       mSimView2D.p1.y, mSimView2D.p2.x, mSimView2D.p2.y);
          ImGui::Text("SimView Size:  < %f, %f>",              mSimView2D.size().x,   mSimView2D.size().y);
          ImGui::Text("EM  2D View:   < %f, %f> : < %f, %f>",  mEMView.r.p1.x,  mEMView.r.p1.y,  mEMView.r.p2.x,  mEMView.r.p2.y);
          ImGui::Text("Mat 2D View:   < %f, %f> : < %f, %f>",  mMatView.r.p1.x, mMatView.r.p1.y, mMatView.r.p2.x, mMatView.r.p2.y);
          ImGui::Text("3D  EM View:   < %f, %f> : < %f, %f>",  m3DView.r.p1.x,  m3DView.r.p1.y,  m3DView.r.p2.x,  m3DView.r.p2.y);
          ImGui::Text("3D  Aspect:    < %f,  %f>",  aspect3D.x, aspect3D.y);

          ImGui::SetCursorScreenPos(Vec2f(ImGui::GetCursorScreenPos()) + Vec2f(0.0f, 45.0f));
          ImGui::Text("Camera Pos:   < %f, %f, %f>", mCamera.pos.x,   mCamera.pos.y,   mCamera.pos.z  );
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
    
    if(mFirstFrame || isinf(mCamera.pos.z)) { resetViews(); mFirstFrame = false; }
    
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

  mVecBuffer.bind();
  glDrawArrays(GL_LINES, 0, mVecBuffer.size);
  mVecBuffer.release();

  
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

