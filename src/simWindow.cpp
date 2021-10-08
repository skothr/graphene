#include "simWindow.hpp"

#include <imgui.h>
#include <imgui_freetype.h>
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

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "glfwKeys.hpp"
#include "keyManager.hpp"
#include "keyFrameWidget.hpp"
#include "image.hpp"
#include "imtools.hpp"
#include "tools.hpp"
#include "settingForm.hpp"
#include "tabMenu.hpp"

#include "draw.hpp"
#include "display.hpp"
#include "render.cuh"

#include "fluid.cuh"
#include "maxwell.cuh"

// prints aligned numbers for tooltip
inline std::string fAlign(float f, int maxDigits)
{
  std::stringstream ss;
  ss << std::setprecision(4);
  if(log10(f) >= maxDigits) { ss << std::scientific << f; }
  else                      { ss << std::fixed      << f; }
  return ss.str();
}

static SimWindow *simWin = nullptr; // set in SimWindow::SimWindow()
void SimWindow::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) { simWin->keyPress(mods, key, action); }

// handle key events manually to make sure none are missed
void SimWindow::keyPress(int mods, int key, int action) { if(mKeyManager) { mKeyManager->keyPress(mods, key, action); } }

// helpers for key binding callbacks
//  NOTE: unused -- std::function ambiguous constructor overloads (?)
#define KBIND(func)       std::bind(&SimWindow::func, this)
#define KBINDV(func, ...) std::bind(&SimWindow::func, this, __VA_ARGS__)
#define KBIND1(func)      std::bind(&SimWindow::func, this, std::placeholders::_1)
#define KBIND2(func)      std::bind(&SimWindow::func, this, std::placeholders::_1, std::placeholders::_2)


SimWindow::SimWindow(GLFWwindow *window)
  : mWindow(window)
{
  simWin = this; // NOTE/TODO: only one window total allowed for now
  glfwSetKeyCallback(mWindow, &keyCallback); // set key event callback (before ImGui GLFW initialization)

  std::vector<KeyBinding> keyBindings =
    {
     KeyBinding("Quit",            "Ctrl+Esc", "Quit program",                                  KEYBINDING_GLOBAL,[this](){ quit();                }),
     KeyBinding("Reset Views",     "F1",       "Reset viewports (align field in each view)",    KEYBINDING_GLOBAL,[this](){ resetViews(false);     }),
     KeyBinding("Reset Sim",       "F5",       "Reset full simulation (signal/material/fluid)", KEYBINDING_GLOBAL,[this](){ resetSim(false);       }),
     KeyBinding("Reset Signals",   "F7",       "Reset signals, leaving materials intact",       KEYBINDING_GLOBAL,[this](){ resetSignals(false);   }),
     KeyBinding("Reset Fluid",     "F8",       "Reset fluid variables",                         KEYBINDING_GLOBAL,[this](){ resetFluid(false);     }),
     KeyBinding("Reset Materials", "F9",       "Reset materials, leaving signals intact",       KEYBINDING_GLOBAL,[this](){ resetMaterials(false); }),
     KeyBinding("Prev Setting Tab", "Ctrl+PgUp", "Switch to previous setting tab",            KEYBINDING_GLOBAL,[this](){ mSideTabs->prev(); }),
     KeyBinding("Next Setting Tab", "Ctrl+PgDn", "Switch to previous setting tab",            KEYBINDING_GLOBAL,[this](){ mSideTabs->next(); }),
     
     KeyBinding("Toggle Physics",  "Space",    "Start/stop simulation physics",              KEYBINDING_EXTRA_MODS, [this](){ togglePause(); }),
     KeyBinding("Step Forward",    "Up",       "Single physics step (+dt)", (KEYBINDING_REPEAT|KEYBINDING_MOD_MULT),[this](CFT mult){singleStepField(mult);}),
     KeyBinding("Step Backward",   "Down",     "Single physics step (-dt)", (KEYBINDING_REPEAT|KEYBINDING_MOD_MULT),[this](CFT mult){singleStepField(-mult);}),

     KeyBinding("Toggle Debug",    "Alt+D",    "Toggle debug mode",                             KEYBINDING_GLOBAL, [this](){ toggleDebug(); }),
     KeyBinding("Toggle Verbose",  "Alt+V",    "Toggle verbose mode",                           KEYBINDING_GLOBAL, [this](){ toggleVerbose(); }),
     KeyBinding("Key Bindings",    "Alt+K",    "Open Key Bindings window (view/edit bindings)", KEYBINDING_GLOBAL, [this](){ mKeyManager->togglePopup(); }),
     KeyBinding("ImGui Demo",   "Alt+Shift+D", "Toggle ImGui demo window (examples/tools)",       KEYBINDING_NONE, [this](){ toggleImGuiDemo(); }),
     KeyBinding("Font Demo",    "Alt+Shift+F", "Toggle Font demo window",                         KEYBINDING_NONE, [this](){ toggleFontDemo(); }),
    };

  
  // NOTE: Any bindings not added to a group will be added to a "misc" group
  std::vector<KeyBindingGroup> keyGroups =
    {
     { "System",       { "Quit", "Prev Setting Tab", "Next Setting Tab" },
       
       {} },
     { "Sim Control",  { "Reset Sim", "Reset Signals", "Reset Materials","Reset Fluid", "Reset Views",
                         "Toggle Physics", "Step Forward", "Step Backward" },
       {} },
     { "Modes/Popups", { "Toggle Debug", "Toggle Verbose", "Key Bindings", "ImGui Demo", "Font Demo" },
       
       {} },
    };

  mKeyManager = new KeyManager(this, keyBindings, keyGroups);
}

SimWindow::~SimWindow()
{
  cleanup();
  if(mKeyManager) { delete mKeyManager; mKeyManager = nullptr; }
}


bool SimWindow::preFrame() { return (ftDemo ? ftDemo->PreNewFrame() : false); }

bool SimWindow::init()
{
  if(!mInitialized)
    {
      std::cout << "Creating SimWindow...\n";

      //// set up fonts=
      ImGuiIO &io = ImGui::GetIO();
      fontConfig = new ImFontConfig();
      fontConfig->OversampleH = FONT_OVERSAMPLE;
      fontConfig->OversampleV = FONT_OVERSAMPLE;
      std::cout << "===> FONTS(imgui/freetype) --> default flags: " << fontConfig->FontBuilderFlags << "\n";
      io.Fonts->FontBuilderFlags |= (ImGuiFreeTypeBuilderFlags_LightHinting |
                                     ImGuiFreeTypeBuilderFlags_LoadColor    |
                                     ImGuiFreeTypeBuilderFlags_Bitmap);
      
      fontBuilder.AddText("ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσΤτΥυΦφΧχΨψΩω");
      fontBuilder.AddRanges(io.Fonts->GetGlyphRangesDefault());  // Add one of the default ranges
      fontBuilder.AddChar(0x2207);                               // Add a specific character
      fontBuilder.BuildRanges(&fontRanges);                      // Build the final result (ordered ranges with all the unique characters submitted)
      fontConfig->GlyphRanges = fontRanges.Data;

      fontConfig->SizePixels  = MAIN_FONT_HEIGHT; // main
      mainFont   = io.Fonts->AddFontFromFileTTF(FONT_PATH_REGULAR,      MAIN_FONT_HEIGHT,  fontConfig);
      mainFontB  = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD,         MAIN_FONT_HEIGHT,  fontConfig);
      mainFontI  = io.Fonts->AddFontFromFileTTF(FONT_PATH_ITALIC,       MAIN_FONT_HEIGHT,  fontConfig);
      mainFontBI = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD_ITALIC,  MAIN_FONT_HEIGHT,  fontConfig);
      fontConfig->SizePixels  = SMALL_FONT_HEIGHT; // small
      smallFont   = io.Fonts->AddFontFromFileTTF(FONT_PATH_REGULAR,     SMALL_FONT_HEIGHT, fontConfig);
      smallFontB  = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD,        SMALL_FONT_HEIGHT, fontConfig);
      smallFontI  = io.Fonts->AddFontFromFileTTF(FONT_PATH_ITALIC,      SMALL_FONT_HEIGHT, fontConfig);
      smallFontBI = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD_ITALIC, SMALL_FONT_HEIGHT, fontConfig);
      fontConfig->SizePixels  = TITLE_FONT_HEIGHT; // title
      titleFont   = io.Fonts->AddFontFromFileTTF(FONT_PATH_REGULAR,     TITLE_FONT_HEIGHT, fontConfig);
      titleFontB  = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD,        TITLE_FONT_HEIGHT, fontConfig);
      titleFontI  = io.Fonts->AddFontFromFileTTF(FONT_PATH_ITALIC,      TITLE_FONT_HEIGHT, fontConfig);
      titleFontBI = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD_ITALIC, TITLE_FONT_HEIGHT, fontConfig);
      fontConfig->SizePixels  = SUPER_FONT_HEIGHT; // superscript
      superFont   = io.Fonts->AddFontFromFileTTF(FONT_PATH_REGULAR,     SUPER_FONT_HEIGHT, fontConfig);
      superFontB  = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD,        SUPER_FONT_HEIGHT, fontConfig);
      superFontI  = io.Fonts->AddFontFromFileTTF(FONT_PATH_ITALIC,      SUPER_FONT_HEIGHT, fontConfig);
      superFontBI = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD_ITALIC, SUPER_FONT_HEIGHT, fontConfig);
      fontConfig->SizePixels  = TINY_FONT_HEIGHT; // tiny
      tinyFont    = io.Fonts->AddFontFromFileTTF(FONT_PATH_REGULAR,     TINY_FONT_HEIGHT,  fontConfig);
      tinyFontB   = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD,        TINY_FONT_HEIGHT,  fontConfig);
      tinyFontI   = io.Fonts->AddFontFromFileTTF(FONT_PATH_ITALIC,      TINY_FONT_HEIGHT,  fontConfig);
      tinyFontBI  = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD_ITALIC, TINY_FONT_HEIGHT,  fontConfig);      
      io.Fonts->Build();
      
      ftDemo = new FreeTypeTest(); // font demo window
      
      mFieldUI = new FieldInterface<CFT>(&mParams.cp,
                                         [this]() // field size update
                                         { resizeFields(mParams.cp.fs); resetSim(); },
                                         [this]() // texRes2D update
                                         {
                                           std::cout << "Resizing 2D textures --> " << mEMTex.size << "/" << mMatTex.size << " --> " << mFieldUI->texRes2D << "\n";
                                           mEMTex.create (int3{mFieldUI->texRes2D.x, mFieldUI->texRes2D.y, 1});
                                           mMatTex.create(int3{mFieldUI->texRes2D.x, mFieldUI->texRes2D.y, 1});
                                         },
                                         [this]() // texRes3D update
                                         {
                                           std::cout << "Resizing 3D textures --> " << m3DTex.size << " --> " << mFieldUI->texRes3D << "\n";
                                           m3DTex.create (int3{mFieldUI->texRes3D.x, mFieldUI->texRes3D.y, 1});
                                         });
      
      mUnitsUI   = new UnitsInterface<CFT>(&mUnits, superFont);
      mDrawUI    = new DrawInterface<CFT>(&mUnits);
      mDisplayUI = new DisplayInterface<CFT>(&mParams.rp, &mParams.vp, &mParams.cp.fs.z);

      // file output (TODO: separate object (?))
      mFileOutUI = new SettingForm("File Output", SETTINGS_LABEL_COL_W, SETTINGS_INPUT_COL_W);
      auto *sA  = new Setting<bool>       ("Write to File",  "outActive",  &mFileRendering);
      mFileOutUI->add(sA);
      auto *sLV = new Setting<bool>       ("Lock Views",     "lockViews",  &mLockViews);
      mFileOutUI->add(sLV);
      auto *sOR = new Setting<int2>       ("Resolution",     "outRes",     &mParams.outSize);
      mFileOutUI->add(sOR);
      auto *sN  = new Setting<std::string>("Base Name",      "baseName",   &mParams.simName);
      mFileOutUI->add(sN);
      auto *sE  = new Setting<std::string>("File Extension", "outExt",     &mParams.outExt);
      auto *sPA = new Setting<bool>       ("Alpha Channel",  "outAlpha",   &mParams.outAlpha);
      sE->drawCustom = [this, sE, sPA](bool busy, bool changed) -> bool
                       {
                         ImGui::SetNextItemWidth(111); sE->onDraw(1.0f, busy, changed, true);
                         if(changed)
                           {
                             if(mParams.outExt.empty())        { mParams.outExt = ".png"; }
                             else if(mParams.outExt[0] != '.') { mParams.outExt = "." + mParams.outExt; }
                           }
                         ImGui::SameLine(); ImGui::TextUnformatted("Alpha:"); ImGui::SameLine(); sPA->onDraw(1.0f, busy, changed, true);
                         return busy;
                       };
      mFileOutUI->add(sE);
      auto *sPC  = new Setting<int> ("PNG Compression", "pngComp",   &mParams.pngCompression);
      sPC->visibleCallback = [this]() -> bool { return (mParams.outExt.find(".png") != std::string::npos); };
      sPC->drawCustom = [this, sPC](bool busy, bool changed) -> bool
                        { changed |= ImGui::SliderInt("##pngComp", &mParams.pngCompression, 1, 10); return busy; };
      mFileOutUI->add(sPC);
      // miscellaneous flags
      mOtherUI = new SettingForm("Other Settings", SETTINGS_LABEL_COL_W, SETTINGS_INPUT_COL_W);
      SettingGroup *infoGroup = new SettingGroup("Info", "infoGroup", { }, false);
      auto *sDBG   = new Setting<bool> ("Debug",   "debug",   &mParams.debug);   infoGroup->add(sDBG);
      auto *sVERB  = new Setting<bool> ("Verbose", "verbose", &mParams.verbose); infoGroup->add(sVERB);
      mOtherUI->add(infoGroup);
      
      // create side tabs
      mSideTabs = new TabMenu(20, 1080);
      mSideTabs->setCollapsible(true);
      mSideTabs->add(TabDesc{"Field",       "Field Settings",    [this](){ mFieldUI->draw(); },
                             (int)SETTINGS_TOTAL_W, (int)SETTINGS_TOTAL_W+40, (int)SETTINGS_TOTAL_W+40, titleFont});
      mSideTabs->add(TabDesc{"Units",       "Base Units",        [this](){ mUnitsUI->draw(); },
                             (int)SETTINGS_TOTAL_W, (int)SETTINGS_TOTAL_W+40, (int)SETTINGS_TOTAL_W+40, titleFont});
      mSideTabs->add(TabDesc{"Draw",        "Draw Settings",     [this](){ mDrawUI->draw(superFont); },
                             (int)SETTINGS_TOTAL_W, (int)SETTINGS_TOTAL_W+40, (int)SETTINGS_TOTAL_W+40, titleFont});
      mSideTabs->add(TabDesc{"Display",     "Display Settings",  [this](){ mDisplayUI->draw(); },
                             (int)SETTINGS_TOTAL_W, (int)SETTINGS_TOTAL_W+40, (int)SETTINGS_TOTAL_W+40, titleFont});
      mSideTabs->add(TabDesc{"File Output", "Render to File",    [this](){ mFileOutUI->draw(); },
                             (int)SETTINGS_TOTAL_W, (int)SETTINGS_TOTAL_W+40, (int)SETTINGS_TOTAL_W+40, titleFont});
      mSideTabs->add(TabDesc{"Other",       "Other Settings",    [this](){ mOtherUI->draw(); },
                             (int)SETTINGS_TOTAL_W, (int)SETTINGS_TOTAL_W+40, (int)SETTINGS_TOTAL_W+40, titleFont});

      mBottomTabs = new TabMenu(20, 1080);
      mBottomTabs->setCollapsible(true);
      mBottomTabs->setHorizontal(true);
      mBottomTabs->add(TabDesc{"Key Framing", "", [this](){ mKeyFrameUI->draw(); }, //Vec2f(mDisplaySize.x, KEYFRAME_WIDGET_H)); },
                               (int)KEYFRAME_WIDGET_H, (int)KEYFRAME_WIDGET_H+40, (int)KEYFRAME_WIDGET_H+40, nullptr});
      
      mKeyFrameUI = new KeyFrameWidget(&mDrawUI->sigPen, &mDrawUI->matPen);
      mKeyFrameUI->view2DCallback = [this](const Rect2f &r)            { mSimView2D   = r;   };
      mKeyFrameUI->view3DCallback = [this](const CameraDesc<CFT> &cam) { mCamera.desc = cam; };

      // load settings config file (.settings.conf)
      loadSettings();
            
      // initialize CUDA and check for a compatible device
      std::cout << "Creating CUDA objects...\n";
      if(!initCudaDevice())
        {
          std::cout << "====> ERROR: failed to initialize CUDA device!\n";
          delete fontConfig; fontConfig = nullptr;
          return false;
        }
      // create field state queue
      if(mParams.cp.fs.x > 0 && mParams.cp.fs.y > 0 && mParams.cp.fs.z > 0)
        {
          std::cout << "Creating field state queue (" << STATE_BUFFER_SIZE << "x " << mParams.cp.fs << ")...\n";
          for(int i = 0; i < STATE_BUFFER_SIZE; i++) { mStates.push_back(new FluidField<CFT>()); }
          mInputV  = new Field<CFV3>(); mInputP  = new Field<CFT>();  
          mInputQn = new Field<CFT>();  mInputQp = new Field<CFT>();  mInputQv = new Field<CFV3>();
          mInputE  = new Field<CFV3>(); mInputB  = new Field<CFV3>();
          resizeFields(mParams.cp.fs);
        }
      // create textures
      int3 ts2 = int3{mFieldUI->texRes2D.x, mFieldUI->texRes2D.y, 1};
      int3 ts3 = int3{mFieldUI->texRes3D.x, mFieldUI->texRes3D.y, 1};
      std::cout << "Creating 2D textures (" << ts2 << ")...\n";
      if(!mEMTex.create(ts2))   { std::cout << "====> ERROR: Texture creation for EM view failed!\n";  }
      if(!mMatTex.create(ts2))  { std::cout << "====> ERROR: Texture creation for Mat view failed!\n"; }
      std::cout << "Creating 3D texture (" << ts3 << ")...\n";
      if(!m3DTex.create(ts3))   { std::cout << "====> ERROR: Texture creation for 3D view failed!\n";  }
      if(!m3DGlTex.create(ts3)) { std::cout << "====> ERROR: Texture creation for 3D gl view failed!\n";  }
      
      // create initial state expressions
      std::cout << "Creating initial condition expressions...\n";
      mFieldUI->initV.hExpr   = toExpression<CFV3>(mFieldUI->initV.str);
      mFieldUI->initP.hExpr   = toExpression<CFT> (mFieldUI->initP.str);
      mFieldUI->initQn.hExpr  = toExpression<CFT> (mFieldUI->initQn.str);
      mFieldUI->initQp.hExpr  = toExpression<CFT> (mFieldUI->initQp.str);
      mFieldUI->initQv.hExpr  = toExpression<CFV3>(mFieldUI->initQv.str);
      mFieldUI->initE.hExpr   = toExpression<CFV3>(mFieldUI->initE.str);
      mFieldUI->initB.hExpr   = toExpression<CFV3>(mFieldUI->initB.str);
      mFieldUI->initEp.hExpr  = toExpression<CFT>(mFieldUI->initEp.str);
      mFieldUI->initMu.hExpr  = toExpression<CFT>(mFieldUI->initMu.str);
      mFieldUI->initSig.hExpr = toExpression<CFT>(mFieldUI->initSig.str);
      
      // finalize loaded settings
      mFieldUI->updateAll();
      mUnitsUI->updateAll();
      mDrawUI->updateAll();
      mDisplayUI->updateAll();
      mFileOutUI->updateAll();
      mOtherUI->updateAll();

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
      saveSettings();
      
      // cudaDeviceSynchronize(); // TODO: settle on specific synchronization point(s)
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
              f->destroy(); delete f; deleted.push_back(f);
            }
        }
      mStates.clear();      
      if(mInputV)  { mInputV->destroy();  delete mInputV;  mInputV  = nullptr; }
      if(mInputP)  { mInputP->destroy();  delete mInputP;  mInputP  = nullptr; }
      if(mInputQn) { mInputQn->destroy(); delete mInputQn; mInputQn = nullptr; }
      if(mInputQp) { mInputQp->destroy(); delete mInputQp; mInputQp = nullptr; }
      if(mInputQv) { mInputQv->destroy(); delete mInputQv; mInputQv = nullptr; }
      if(mInputE)  { mInputE->destroy();  delete mInputE;  mInputE  = nullptr; }
      if(mInputB)  { mInputB->destroy();  delete mInputB;  mInputB  = nullptr; }
      
      std::cout << "Destroying CUDA textures...\n";
      mEMTex.destroy(); mMatTex.destroy(); m3DTex.destroy(); m3DGlTex.destroy();
      std::cout << "Destroying CUDA VBOs...\n";
      // mVBuffer2D.destroy();
      // mVBuffer3D.destroy();
      
      std::cout << "Destroying fonts...\n";
      if(ftDemo)     { delete ftDemo; ftDemo = nullptr; }
      if(fontConfig) { delete fontConfig; }

      std::cout << "Destroying UI components...\n";      
      if(mSideTabs)   { delete mSideTabs;   mSideTabs   = nullptr; }
      if(mBottomTabs) { delete mBottomTabs; mBottomTabs = nullptr; }
      if(mFieldUI)    { delete mFieldUI;    mFieldUI    = nullptr; }
      if(mUnitsUI)    { delete mUnitsUI;    mUnitsUI    = nullptr; }
      if(mDrawUI)     { delete mDrawUI;     mDrawUI     = nullptr; }
      if(mDisplayUI)  { delete mDisplayUI;  mDisplayUI  = nullptr; }
      if(mFileOutUI)  { delete mFileOutUI;  mFileOutUI  = nullptr; }
      if(mOtherUI)    { delete mOtherUI;    mOtherUI    = nullptr; }
      if(mKeyFrameUI) { delete mKeyFrameUI; mKeyFrameUI = nullptr; }
      
      std::cout << "Cleaning GL...\n";
      cleanupGL();

      mInitialized = false;
    }
}

void SimWindow::loadSettings(const std::string &path)
{
  std::cout << "Loading settings (" << path << ")...\n";
  if(fileExists(path))
    {
      std::ifstream f(path, std::ios::in);
      json js; f >> js; // load JSON from file

      // load settings
      if(js.contains("SimSettings"))
        {
          json sim = js["SimSettings"];
          if(sim.contains("Field"))
            {
              if(!mFieldUI->fromJSON(sim["Field"]))
                { std::cout << "====> WARNING: Failed to load Field setting group\n"; }
            }
          else  { std::cout << "====> WARNING: Could not find Field setting group\n"; }
          
          if(sim.contains("Units"))
            {
              if(!mUnitsUI->fromJSON(sim["Units"]))
                { std::cout << "====> WARNING: Failed to load Units setting group\n"; }
            }
          else  { std::cout << "====> WARNING: Could not find Units setting group\n"; }
          
          if(sim.contains("Draw"))
            {
              if(!mDrawUI->fromJSON(sim["Draw"]))
                { std::cout << "====> WARNING: Failed to load Draw setting group\n"; }
            }
          else  { std::cout << "====> WARNING: Could not find Draw setting group\n"; }
          
          if(sim.contains("Display"))
            {
              if(!mDisplayUI->fromJSON(sim["Display"]))
                { std::cout << "====> WARNING: Failed to load Display setting group\n"; }
            }
          else  { std::cout << "====> WARNING: Could not find Display setting group\n"; }
          
          if(sim.contains("FileOutput"))
            {
              if(!mFileOutUI->fromJSON(sim["FileOutput"]))
                { std::cout << "====> WARNING: Failed to load FileOutput setting group\n"; }
            }
          else  { std::cout << "====> WARNING: Could not find FileOutput setting group\n"; }
          // override --> don't lock views or begin rendering to files on startup
          mFileRendering = false;
          mLockViews     = false;
          
          if(sim.contains("Other"))
            {
              if(!mOtherUI->fromJSON(sim["Other"]))
                { std::cout << "====> WARNING: Failed to load Other setting group\n"; }
            }
          else  { std::cout << "====> WARNING: Could not find Other setting group\n"; }
          
        }
      else { std::cout << "====> WARNING: No SimSettings group in settings file\n"; }

      // load key bindings
      if(js.contains("KeyBindings"))
        {
          if(!mKeyManager->fromJSON(js["KeyBindings"]))
            { std::cout << "====> WARNING: Failed to load Key Binding settings\n"; }
        }
      else  { std::cout << "====> WARNING: No KeyBindings in settings file\n"; }
    }
  else
    {
      std::cout << "====> Could not find settings file (" << path << ")\n";
      saveSettings();
    }
}

void SimWindow::saveSettings(const std::string &path)
{
  std::cout << "SAVING SETTINGS...\n";
  
  json sim = json::object();
  sim["Field"]      = mFieldUI->toJSON();
  sim["Units"]      = mUnitsUI->toJSON();
  sim["Draw"]       = mDrawUI->toJSON();
  sim["Display"]    = mDisplayUI->toJSON();
  sim["FileOutput"] = mFileOutUI->toJSON();
  sim["Other"]      = mOtherUI->toJSON();

  json js  = json::object();
  js["SimSettings"] = sim;
  js["KeyBindings"] = mKeyManager->toJSON();
  
  std::ofstream f(path, std::ios::out);
  f << std::setw(JSON_SPACES) << js;
}


bool SimWindow::resizeFields(const Vec3i &sz)
{
  if(min(sz) <= 0) { std::cout << "====> ERROR: Field with zero size not allowed.\n"; return false; }
  std::cout << "RESIZING FIELD: " << sz << "\n";
  bool success = true;
  for(int i = 0; i < STATE_BUFFER_SIZE; i++)
    {
      FluidField<CFT> *f = reinterpret_cast<FluidField<CFT>*>(mStates[i]);
      if(!f->create(mParams.cp.fs)) { std::cout << "Field creation failed! Invalid state.\n"; success = false; break; }
      fillFieldValue(reinterpret_cast<FluidField<CFT>*>(f)->mat, mUnits.vacuum());
    }
  if(!mInputV->create (mParams.cp.fs)) { std::cout << "Field creation failed! Invalid signal input field (v).\n";  success = false; }
  if(!mInputP->create (mParams.cp.fs)) { std::cout << "Field creation failed! Invalid signal input field (p).\n";  success = false; }
  if(!mInputQn->create(mParams.cp.fs)) { std::cout << "Field creation failed! Invalid signal input field (Qn).\n"; success = false; }
  if(!mInputQp->create(mParams.cp.fs)) { std::cout << "Field creation failed! Invalid signal input field (Qp).\n"; success = false; }
  if(!mInputQv->create(mParams.cp.fs)) { std::cout << "Field creation failed! Invalid signal input field (Qv).\n"; success = false; }
  if(!mInputE->create (mParams.cp.fs)) { std::cout << "Field creation failed! Invalid signal input field (E).\n";  success = false; }
  if(!mInputB->create (mParams.cp.fs)) { std::cout << "Field creation failed! Invalid signal input field (B).\n";  success = false; }

  mDrawUI->sigPen.depth = mParams.cp.fs.z/2;
  mDrawUI->matPen.depth = mParams.cp.fs.z/2;
  mDisplayUI->rp->zRange = int2{0, mParams.cp.fs.z-1};
  if(success) { resetSim(); }
  return success;
}

// TODO: improve expression variable framework
template<typename T> std::vector<std::string> getVarNames() { return {"px", "py", "pz", "sx", "sy", "sz", "r", "t"}; }
template<> std::vector<std::string> getVarNames<CFT> ()     { return {"px", "py", "pz", "sx", "sy", "sz", "r", "t"}; }
template<> std::vector<std::string> getVarNames<CFV3>()     { return {"p", "s", "r", "n", "t"}; }

void SimWindow::resetSignals(bool cudaThread)
{
  if(cudaThread)
    {
      std::cout << "SIGNAL RESET\n";
      // recreate expressions
      if(mFieldUI->initQn.hExpr) { delete mFieldUI->initQn.hExpr; }
      mFieldUI->initQn.hExpr = toExpression<CFT>((mFieldUI->initQn.active?mFieldUI->initQn.str  : "0"), false);
      if(mFieldUI->initQp.hExpr) { delete mFieldUI->initQp.hExpr; }
      mFieldUI->initQp.hExpr = toExpression<CFT>((mFieldUI->initQp.active?mFieldUI->initQp.str  : "0"), false);
      if(mFieldUI->initQv.hExpr) { delete mFieldUI->initQv.hExpr; }
      mFieldUI->initQv.hExpr = toExpression<CFV3>((mFieldUI->initQv.active?mFieldUI->initQv.str : "0"), false);
      if(mFieldUI->initE.hExpr)  { delete mFieldUI->initE.hExpr; }
      mFieldUI->initE.hExpr  = toExpression<CFV3>((mFieldUI->initE.active ?mFieldUI->initE.str  : "0"), false);
      if(mFieldUI->initB.hExpr)  { delete mFieldUI->initB.hExpr; }
      mFieldUI->initB.hExpr  = toExpression<CFV3>((mFieldUI->initB.active ?mFieldUI->initB.str  : "0"), false);
      if(mParams.verbose)
        {
          std::cout << "Q-: " << mFieldUI->initQn.hExpr->toString(true) << "\n";
          std::cout << "Q+: " << mFieldUI->initQp.hExpr->toString(true) << "\n";
          std::cout << "Qv: " << mFieldUI->initQv.hExpr->toString(true) << "\n";
          std::cout << "E:  " << mFieldUI->initE.hExpr->toString(true)  << "\n";
          std::cout << "B:  " << mFieldUI->initB.hExpr->toString(true)  << "\n";
        }
  
      // create or update expressions
      if(!mFieldUI->initQn.dExpr)
        { mFieldUI->initQn.dExpr = toCudaExpression<CFT> (mFieldUI->initQn.hExpr, getVarNames<CFT>());  std::cout << " --> Q- INIT EXPRESSION UPDATED\n"; }
      if(!mFieldUI->initQp.dExpr)
        { mFieldUI->initQp.dExpr = toCudaExpression<CFT> (mFieldUI->initQp.hExpr, getVarNames<CFT>());  std::cout << " --> Q+ INIT EXPRESSION UPDATED\n"; }
      if(!mFieldUI->initQv.dExpr)
        { mFieldUI->initQv.dExpr = toCudaExpression<CFV3>(mFieldUI->initQv.hExpr, getVarNames<CFV3>()); std::cout << " --> Qv INIT EXPRESSION UPDATED\n"; }
      if(!mFieldUI->initE.dExpr)
        { mFieldUI->initE.dExpr  = toCudaExpression<CFV3>(mFieldUI->initE.hExpr,  getVarNames<CFV3>()); std::cout << " --> E  INIT EXPRESSION UPDATED\n"; }
      if(!mFieldUI->initB.dExpr)
        { mFieldUI->initB.dExpr  = toCudaExpression<CFV3>(mFieldUI->initB.hExpr,  getVarNames<CFV3>()); std::cout << " --> B  INIT EXPRESSION UPDATED\n"; }
      // fill all states
      for(int i = 0; i < mStates.size(); i++)
        {
          FluidField<CFT> *f = reinterpret_cast<FluidField<CFT>*>(mStates[mStates.size()-1-i]);
          fillField<CFT> (f->Qn, mFieldUI->initQn.dExpr);
          fillField<CFT> (f->Qp, mFieldUI->initQp.dExpr);
          fillField<CFV3>(f->Qv, mFieldUI->initQv.dExpr);
          fillField<CFV3>(f->E,  mFieldUI->initE.dExpr);
          fillField<CFV3>(f->B,  mFieldUI->initB.dExpr);
        }
      mInputQn->clear(); mInputQp->clear(); mInputQv->clear(); mInputE->clear(); mInputB->clear(); // clear remaining inputs
      mNeedResetSignals = false;
    }
  else
    { mNeedResetSignals = true; }
}

void SimWindow::resetMaterials(bool cudaThread)
{
  if(cudaThread)
    {
      std::cout << "MATERIAL RESET\n";
      // recreate expressions
      if(mFieldUI->initEp.hExpr) { delete mFieldUI->initEp.hExpr; }
      mFieldUI->initEp.hExpr  = toExpression<CFT>((mFieldUI->initEp.active ?mFieldUI->initEp.str : "1"), false);
      if(mFieldUI->initMu.hExpr) { delete mFieldUI->initMu.hExpr; }
      mFieldUI->initMu.hExpr  = toExpression<CFT>((mFieldUI->initMu.active ?mFieldUI->initMu.str : "1"), false);
      if(mFieldUI->initSig.hExpr){ delete mFieldUI->initSig.hExpr; }
      mFieldUI->initSig.hExpr = toExpression<CFT>((mFieldUI->initSig.active?mFieldUI->initSig.str : "0"), false);
  
      if(mParams.verbose)
        {
          std::cout << "ε:  " << mFieldUI->initEp.hExpr->toString(true)  << "\n";
          std::cout << "μ:  " << mFieldUI->initMu.hExpr->toString(true)  << "\n";
          std::cout << "σ:  " << mFieldUI->initSig.hExpr->toString(true) << "\n";
        }
  
      // create or update expressions
      if(!mFieldUI->initEp.dExpr)
        { mFieldUI->initEp.dExpr  = toCudaExpression<CFT>(mFieldUI->initEp.hExpr,  getVarNames<CFT>()); std::cout << " --> ε  INIT EXPRESSION UPDATED\n"; }
      if(!mFieldUI->initMu.dExpr)
        { mFieldUI->initMu.dExpr  = toCudaExpression<CFT>(mFieldUI->initMu.hExpr,  getVarNames<CFT>()); std::cout << " --> μ  INIT EXPRESSION UPDATED\n"; }
      if(!mFieldUI->initSig.dExpr)
        { mFieldUI->initSig.dExpr = toCudaExpression<CFT>(mFieldUI->initSig.hExpr, getVarNames<CFT>()); std::cout << " --> σ  INIT EXPRESSION UPDATED\n"; }

      // fill all states
      if(mFieldUI->initEp.str == "1" && mFieldUI->initMu.str == "1" && mFieldUI->initSig.str == "0")
        {
          for(int i = 0; i < mStates.size(); i++)
            { // reset materials in each state to VACUUM
              FluidField<CFT> *f = reinterpret_cast<FluidField<CFT>*>(mStates[mStates.size()-1-i]);
              f->mat.clear();
            }
        }
      else
        {
          for(int i = 0; i < mStates.size(); i++)
            { // fill materials in each state with parametetric expressions
              FluidField<CFT> *f = reinterpret_cast<FluidField<CFT>*>(mStates[mStates.size()-1-i]);
              fillFieldMaterial<CFT>(f->mat, mFieldUI->initEp.dExpr, mFieldUI->initMu.dExpr, mFieldUI->initSig.dExpr);
            }
        }
      mNeedResetMaterials = false;
    }
  else
    { mNeedResetMaterials = true; }
}


void SimWindow::resetFluid(bool cudaThread)
{
  if(cudaThread)
    {
      std::cout << "FLUID RESET\n";

      // recreate expressions  
      if(mFieldUI->initV.hExpr){delete mFieldUI->initV.hExpr;}  mFieldUI->initV.hExpr = toExpression<CFV3>((mFieldUI->initV.active?mFieldUI->initV.str:"0"), false);
      if(mFieldUI->initP.hExpr){delete mFieldUI->initP.hExpr;}  mFieldUI->initP.hExpr = toExpression<CFT> ((mFieldUI->initP.active?mFieldUI->initP.str:"0"), false);
      if(mParams.verbose)
        {
          std::cout << "V:  " << mFieldUI->initV.hExpr->toString(true)  << "\n";
          std::cout << "P:  " << mFieldUI->initP.hExpr->toString(true)  << "\n";
        }
  
      // create or update expressions
      if(!mFieldUI->initV.dExpr)
        { mFieldUI->initV.dExpr  = toCudaExpression<CFV3>(mFieldUI->initV.hExpr,  getVarNames<CFV3>()); std::cout << " --> V  INIT EXPRESSION UPDATED\n"; }
      if(!mFieldUI->initP.dExpr)
        { mFieldUI->initP.dExpr  = toCudaExpression<CFT> (mFieldUI->initP.hExpr,  getVarNames<CFT>());  std::cout << " --> P  INIT EXPRESSION UPDATED\n"; }
      // fill all states
      for(int i = 0; i < mStates.size(); i++)
        {
          FluidField<CFT> *f = reinterpret_cast<FluidField<CFT>*>(mStates[mStates.size()-1-i]);
          fillField<CFV3>(f->v,  mFieldUI->initV.dExpr);
          fillField<CFT> (f->p,  mFieldUI->initP.dExpr);
        }
      mInputV->clear(); mInputP->clear(); // clear remaining inputs
      mNeedResetFluid = false;
    }
  else { mNeedResetFluid = true; }
}


void SimWindow::resetSim(bool cudaThread)
{
  if(cudaThread)
    {
      std::cout << "SIMULATION RESET\n";
      mInfo.t = 0.0f;
      mInfo.frame = 0;
      mInfo.uStep = 0;
      mKeyFrameUI->reset();
      
      std::cout << " --> "; resetSignals();
      std::cout << " --> "; resetMaterials();
      std::cout << " --> "; resetFluid();
  
      cudaRender(mParams.cp);
      cudaDeviceSynchronize();
      mNewFrameVec = true;  mNewFrameOut = true; // frame 0
      mNeedResetSignals = false; mNeedResetFluid = false; mNeedResetMaterials = false;  mNeedResetSim = false;
    }
  else { mNeedResetSim = true; }
}

void SimWindow::resetViews(bool cudaThread)
{
  if(!mLockViews && !mForcePause)
    {
      if(cudaThread)
        {
          CFT    pad = SIM_VIEW_RESET_INTERNAL_PADDING;
          Vec2f  fs  = Vec2f(mParams.cp.fs.x, mParams.cp.fs.y);
          Vec2f  fp  = Vec2f(mParams.cp.fp.x, mParams.cp.fp.y);
          Vec2f  fsPadded = fs * (1.0 + 2.0*pad);

          float fAspect = mParams.cp.fs.x/(float)mParams.cp.fs.y;
          Vec2f aspect2D = Vec2f(mEMView.r.aspect()/fAspect, 1.0);
          if(aspect2D.x < 1.0) { aspect2D.y = 1.0/aspect2D.x; aspect2D.x = 1.0; }
          Vec2f aspect3D = Vec2f(m3DView.r.aspect()/fAspect, 1.0);
          if(aspect3D.x < 1.0) { aspect3D.y = 1.0/aspect3D.x; aspect3D.x = 1.0; }
      
          // reset 2D sim view
          Vec2f fp2D = -(fsPadded * pad);
          Vec2f fs2D = fsPadded*aspect2D * mUnits.dL;
          mSimView2D = Rect2f(fp2D, fp2D + fs2D);
          Vec2f offset2D = fs/2.0*mUnits.dL - mSimView2D.center();
          mSimView2D.move(offset2D);

          // calculate 3D camera Z offset
          float fov = 55.0f;
          float S = 2.0*tan((fov/2.0)*M_PI/180.0);
          Vec2f fs3D = fsPadded*aspect3D * mUnits.dL;
          float zOffset = ((max(to_cuda(aspect3D)) < fAspect) ? fs3D.x : fs3D.y) / S;
          // reset 3D camera
          mCamera.fov   = fov; mCamera.near = 0.001f; mCamera.far = 100000.0f;
          mCamera.pos   = CFV3{ fs.x*mUnits.dL/2, fs.y*mUnits.dL/2, // camera position (centered over field)
                                zOffset + mParams.cp.fs.z*mUnits.dL};
          mCamera.right = CFV3{1.0f,  0.0f,  0.0f}; // camera x direction
          mCamera.up    = CFV3{0.0f,  1.0f,  0.0f}; // camera y direction
          mCamera.dir   = CFV3{0.0f,  0.0f, -1.0f}; // camera z direction
          
          mKeyFrameUI->view2D() = mSimView2D;
          mKeyFrameUI->view3D() = mCamera.desc;
          
          cudaRender(mParams.cp);
          mNeedResetViews = false;
        }
      else { mNeedResetViews = true; }
    }
}




void SimWindow::togglePause()
{
  mFieldUI->running = !mFieldUI->running;
  std::cout << (mFieldUI->running ? "STARTED" : "STOPPED") << " SIMULATION.\n";
}
void SimWindow::toggleDebug()             { mParams.debug =   !mParams.debug;   }
void SimWindow::toggleVerbose()           { mParams.verbose = !mParams.verbose; }
void SimWindow::toggleKeyBindings()       { mKeyManager->togglePopup(); }
void SimWindow::singleStepField(CFT mult) { if(!mFieldUI->running) { mSingleStepMult = mult; } }
void SimWindow::toggleImGuiDemo()         { mImGuiDemo = !mImGuiDemo; mLockViews = (mImGuiDemo || mFontDemo);  }
void SimWindow::toggleFontDemo()          { mFontDemo  = !mFontDemo;  mLockViews = (mFontDemo  || mImGuiDemo); }




static FluidField<CFT> *g_temp = nullptr; // temp state (avoids destroying input source state)
void SimWindow::update()
{
  auto t = CLOCK_T::now();
  double dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t - tLastUpdate).count()/1000000000.;
  dtAcc += dt;
  
  int    numFrames = 0; // limited to 1 for now (?)
  double frameTime = 1.0/mDisplayUI->maxFps;
  if(mDisplayUI->limitFps)// && mDisplayUI->maxFps > 0.1)
  {
    if(dtAcc >= frameTime)
      {
        numFrames = 1;
        mFpsUpdate = 1.0 / dtAcc;
        dtAcc = fmod(dtAcc, frameTime);
      }
  }
  else
    {
      numFrames = 1;
      mFpsUpdate = 1.0 / dt;
      dtAcc = 0.0;
    }
  tLastUpdate = t;

  bool newFrame = (numFrames > 0);  
  if(!newFrame)
    { // sleep 
      double sleepOffset = 0.1; // to tune if sleep time is affecting target framerate (?)
      double timeLeft = frameTime - fmod(dtAcc, frameTime) - sleepOffset;
      if(timeLeft < dt) { std::this_thread::sleep_for(std::chrono::nanoseconds((long int)((timeLeft-sleepOffset)*1000000000.0))); }
    }  

  ImGuiIO &io = ImGui::GetIO();
  if(mParams.cp.fs.x > 0 && mParams.cp.fs.y > 0)
    {
      if(newFrame)
        {
          bool singleStep = false;
          mParams.cp.t    = mInfo.t; cpCopy = mParams.cp; cpCopy.u = mUnits;
          if(mSingleStepMult != 0.0f)
            {
              cpCopy.u.dt *= mSingleStepMult;
              mSingleStepMult = 0.0f;
              singleStep = true;
            }
      
          if(!g_temp)                       { g_temp = new FluidField<CFT>();  }
          if(g_temp->size != mParams.cp.fs) { g_temp->create(mParams.cp.fs); }

          if(mNeedResetSim)           { resetSim(true); }
          else
            {
              if(mNeedResetSignals)   { resetSignals(true); }
              if(mNeedResetMaterials) { resetMaterials(true); }
              if(mNeedResetFluid)     { resetFluid(true); }
            }
          if(mNeedResetViews)         { resetViews(true); }      
      
          FluidField<CFT> *src  = reinterpret_cast<FluidField<CFT>*>(mStates.back());  // previous field state
          FluidField<CFT> *dst  = reinterpret_cast<FluidField<CFT>*>(mStates.front()); // oldest state (recycle)
          FluidField<CFT> *temp = reinterpret_cast<FluidField<CFT>*>(g_temp);          // temp intermediate state

          // apply external forces from user (TODO: handle input separately) 
          CFV3 mposSim = CFV3{NAN, NAN, NAN};
          if     (mEMView.hovered)  { mposSim = to_cuda(mEMView.mposSim);  }
          else if(mMatView.hovered) { mposSim = to_cuda(mMatView.mposSim); }
          else if(m3DView.hovered)  { mposSim = to_cuda(m3DView.mposSim);  }
          float  cs = mUnits.dL;
          CFV3 fs = CFV3{(float)mParams.cp.fs.x, (float)mParams.cp.fs.y, (float)mParams.cp.fs.z};
          CFV3 mpfi = (mposSim) / cs;
      
          // draw signal
          mParams.rp.sigPenHighlight = false;
          CFV3 mposLast = mSigMPos;
          mSigMPos = CFV3{NAN, NAN, NAN};
          bool active = false;
          if(!mLockViews && !mForcePause && io.KeyCtrl)
            {
              bool hover = m3DView.hovered || mEMView.hovered || mMatView.hovered;
              bool apply = false;
              CFV3 p = CFV3{NAN, NAN, NAN};
              if(m3DView.hovered)
                {
                  CFV3 fp = to_cuda(m3DView.mposSim);
                  CFV3 vDepth = CFV3{(fp.x <= 1 ? 1.0f : (fp.x >= mParams.cp.fs.x-1 ? -1.0f : 0.0f)),
                                     (fp.y <= 1 ? 1.0f : (fp.y >= mParams.cp.fs.y-1 ? -1.0f : 0.0f)),
                                     (fp.z <= 1 ? 1.0f : (fp.z >= mParams.cp.fs.z-1 ? -1.0f : 0.0f)) };
                  p = fp + vDepth*mDrawUI->sigPen.depth;
                  apply = (m3DView.clickBtns(MOUSEBTN_LEFT) && m3DView.clickMods(GLFW_MOD_CONTROL));
                }
              if(mEMView.hovered || mMatView.hovered)
                {
                  mpfi.z = mParams.cp.fs.z - 1 - mDrawUI->sigPen.depth; // Z depth relative to top visible layer
                  p = CFV3{mpfi.x, mpfi.y, mpfi.z};
                  apply = (( mEMView.clickBtns(MOUSEBTN_LEFT) && (mDrawUI->sigPen.active || mEMView.clickMods(GLFW_MOD_CONTROL))) ||
                           (mMatView.clickBtns(MOUSEBTN_LEFT) && (mDrawUI->sigPen.active || mMatView.clickMods(GLFW_MOD_CONTROL))));
                }
          
              if(hover)
                {
                  mSigMPos = p;
                  mDrawUI->sigPen.mouseSpeed = length(mSigMPos - mposLast);
                  mParams.rp.penPos = p;
                  mParams.rp.sigPenHighlight = true;
                  mParams.rp.sigPen = mDrawUI->sigPen;
              
                  if(apply)
                    { // draw signal to intermediate E/B fields (needs to be blended to avoid peristent blobs of
                      active = true;
                      if(mDrawUI->sigPen.startTime <= 0.0)
                        { // set signal start time
                          mDrawUI->sigPen.startTime = cpCopy.t;
                        }
                  
                      if((mFieldUI->running || singleStep) && !mForcePause)
                        { addSignal(p, *mInputV, *mInputP, *mInputQn, *mInputQp, *mInputQv, *mInputE, *mInputB, mDrawUI->sigPen, cpCopy, cpCopy.u.dt); }
                      else
                        {
                          addSignal(p, src->v, src->p, src->Qn, src->Qp, src->Qv, src->E, src->B, mDrawUI->sigPen, cpCopy, cpCopy.u.dt*(CFT)0.1); //cpCopy.u.dt);
                          // addSignal(*mInputE, src->E, cpCopy, cpCopy.u.dt); decaySignal(*mInputE, cpCopy); addSignal(*mInputE, src->E, cpCopy, -cpCopy.u.dt); 
                          // addSignal(*mInputB, src->B, cpCopy, cpCopy.u.dt); decaySignal(*mInputB, cpCopy); addSignal(*mInputB, src->B, cpCopy, -cpCopy.u.dt); 
                        }
                      //     addSignal(*mInputV,  src->v,  cpCopy, (CFT)1.0); mInputV->clear();  // decaySignal(*mInputV,  cpCopy);
                      //     addSignal(*mInputP,  src->p,  cpCopy, (CFT)1.0); mInputP->clear();  // decaySignal(*mInputP,  cpCopy);
                      //     addSignal(*mInputQn, src->Qn, cpCopy, (CFT)1.0); mInputQn->clear(); // decaySignal(*mInputQn, cpCopy);
                      //     addSignal(*mInputQp, src->Qp, cpCopy, (CFT)1.0); mInputQp->clear(); // decaySignal(*mInputQp, cpCopy);
                      //     addSignal(*mInputQv, src->Qv, cpCopy, (CFT)1.0); mInputQv->clear(); // decaySignal(*mInputQv, cpCopy);
                      // }
                    }
                }
            }
          if(!active) { mDrawUI->sigPen.startTime = -1.0; } // reset start time
      
          // // add material
          // mParams.rp.matPenHighlight = false;
          // mMatMPos = CFV3{NAN, NAN, NAN}; 
          // if(!mLockViews && !mForcePause && io.KeyAlt)
          //   {
          //     bool hover = m3DView.hovered || mEMView.hovered || mMatView.hovered;
          //     CFV3 p = CFV3{NAN, NAN, NAN};
          //     bool apply = false;
          //     if(m3DView.hovered)
          //       {
          //         CFV3 fp = to_cuda(m3DView.mposSim);
          //         CFV3 vDepth = CFV3{(fp.x <= 1 ? 1.0f : (fp.x >= mParams.cp.fs.x-1 ? -1.0f : 0.0f)),
          //                            (fp.y <= 1 ? 1.0f : (fp.y >= mParams.cp.fs.y-1 ? -1.0f : 0.0f)),
          //                            (fp.z <= 1 ? 1.0f : (fp.z >= mParams.cp.fs.z-1 ? -1.0f : 0.0f)) };
          //         p = fp + vDepth*mDrawUI->matPen.depth;
          //         apply = (m3DView.clickBtns(MOUSEBTN_LEFT) && m3DView.clickMods(GLFW_MOD_ALT));
          //       }
          //     if(mEMView.hovered || mMatView.hovered)
          //       {
          //         mpfi.z = mParams.cp.fs.z-1-mDrawUI->matPen.depth; // Z depth relative to top visible layer
          //         p = CFV3{mpfi.x, mpfi.y, mpfi.z};
          //         apply = ((mEMView.clickBtns(MOUSEBTN_LEFT)  && mEMView.clickMods(GLFW_MOD_ALT)) ||
          //                  (mMatView.clickBtns(MOUSEBTN_LEFT) && mMatView.clickMods(GLFW_MOD_ALT)));
          //       }
          
          //     if(hover)
          //       {
          //         mMatMPos = p; 
          //         mParams.rp.penPos = p;
          //         mParams.rp.matPenHighlight = true;
          //         mParams.rp.matPen = mDrawUI->matPen;
          //         if(apply)
          //           {
          //             mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_MATERIAL>(mKeyFrameUI->cursor(), mFieldUI->matPen));
          //             addMaterial(p, *src, mDrawUI->matPen, cpCopy);
          //           }
          //       }
          //   }


          // add keyframe signals to field
          std::map<std::string, SignalSource> &sources = mKeyFrameUI->sources();
          for(const auto &iter : sources)
            {
              if((mFieldUI->running || singleStep) && !mForcePause)
                { addSignal(iter.second.pos, *mInputV, *mInputP, *mInputQn, *mInputQp, *mInputQv, *mInputE, *mInputB, iter.second.pen, cpCopy, cpCopy.u.dt); }
              else
                { addSignal(iter.second.pos, src->v, src->p, src->Qn, src->Qp, src->Qv, src->E, src->B, iter.second.pen, cpCopy, cpCopy.u.dt*(CFT)0.1); }
            }
      
          // add keyframe materials to field
          std::vector<MaterialPlaced> &placed = mKeyFrameUI->placed();
          for(const auto &m : placed) { addMaterial(m.pos, *src, m.pen, cpCopy); }
          placed.clear();

          
          if((mFieldUI->running || singleStep) && !mForcePause)
            { // step simulation
              if(!mFieldUI->running) { std::cout << "SIM STEP --> dt = " << cpCopy.u.dt << "\n"; }
          
              if(DESTROY_LAST_STATE) { temp = src; }         // overwrite source state
              else                   { src->copyTo(*temp); } // don't overwrite previous state (use temp)
          
              // (NOTE: remove added sources to avoid persistent lumps building up)
              if(mFieldUI->updateFluid)
                { // fluid input
                  addSignal(*mInputV,  temp->v,  cpCopy, cpCopy.u.dt); //// add V input signal
                  addSignal(*mInputP,  temp->p,  cpCopy, cpCopy.u.dt); //// add P input signal
                }
              if(mFieldUI->updateQ)
                { // charge input
                  addSignal(*mInputQn, temp->Qn, cpCopy, cpCopy.u.dt); //// add Qn input signal
                  addSignal(*mInputQp, temp->Qp, cpCopy, cpCopy.u.dt); //// add Qp input signal
                  addSignal(*mInputQv, temp->Qv, cpCopy, cpCopy.u.dt); //// add Qv input signal
                }
              if(mFieldUI->updateE || mFieldUI->updateB)
                { // EM input
                  addSignal(*mInputE,  temp->E,  cpCopy, cpCopy.u.dt); //// add E input signal
                  addSignal(*mInputB,  temp->B,  cpCopy, cpCopy.u.dt); //// add B input signal
                }
          
              if(mFieldUI->updateE) { updateElectric(*temp, *dst, cpCopy); std::swap(temp, dst); }  // E
              cpCopy.t += cpCopy.u.dt/2.0f; // increment first half time step
              if(mFieldUI->updateB) { updateMagnetic(*temp, *dst, cpCopy); std::swap(temp, dst); }  // B
              cpCopy.t += cpCopy.u.dt/2.0f; // increment second half time step
              if(mFieldUI->updateQ) { updateCharge  (*temp, *dst, cpCopy); std::swap(temp, dst); }  // Q
          
              if(mFieldUI->updateE || mFieldUI->updateB)
                {
                  addSignal(*mInputE,  temp->E,  cpCopy, -cpCopy.u.dt);      //// remove added E  input signal
                  addSignal(*mInputB,  temp->B,  cpCopy, -cpCopy.u.dt);      //// remove added B  input signal
                  if(mFieldUI->inputDecay) { decaySignal(*mInputE, cpCopy); } //// decay E  input signal (blend over time)
                  if(mFieldUI->inputDecay) { decaySignal(*mInputB, cpCopy); } //// decay B  input signal (blend over time)
                }

              //// V, P
              if(mFieldUI->updateFluid) { fluidStep (*temp, *dst, cpCopy); std::swap(temp, dst); }  // V, P
              if(mFieldUI->inputDecay)
                {
                  if(mFieldUI->updateQ)
                    {
                      // addSignal(*mInputQp, temp->Qp, cpCopy, -1.0f); //// remove added Qp input signal
                      // addSignal(*mInputQn, temp->Qn, cpCopy, -1.0f); //// remove added Qn input signal
                      // addSignal(*mInputQv, temp->Qv, cpCopy, -1.0f); //// remove added Qv input signal
                      decaySignal(*mInputQn, cpCopy); //// decay Qn input signal (blend over time)
                      decaySignal(*mInputQp, cpCopy); //// decay Qp input signal (blend over time)
                      decaySignal(*mInputQv, cpCopy); //// decay Qv input signal (blend over time)
                    }
                  // mInputQn->clear(); mInputQp->clear(); mInputQv->clear(); mInputV->clear(); mInputP->clear();
                  if(mFieldUI->updateFluid)
                    {
                      // addSignal(*mInputV, temp->v, cpCopy, -1.0f); //// remove added V input signal
                      // addSignal(*mInputP, temp->p, cpCopy, -1.0f); //// remove added P input signal
                      decaySignal(*mInputV, cpCopy); //// decay Qp input signal (blend over time)
                      decaySignal(*mInputP, cpCopy); //// decay Qv input signal (blend over time)
                    }
                }

              std::swap(temp,   dst);  // swap final result back into dst
              std::swap(g_temp, temp); // use other state as new temp (pointer changes if number of steps is odd)
              mStates.pop_front(); mStates.push_back(dst);

              // increment time/frame info
              mInfo.t += mUnits.dt;
              mInfo.uStep++;
              if(mInfo.uStep >= mParams.uSteps) { mInfo.frame++; mInfo.uStep = 0; mNewFrameVec = true; mNewFrameOut = true; }
              mKeyFrameUI->nextFrame(cpCopy.u.dt);
            }
        }
      cudaRender(cpCopy);
    }
  
}

void SimWindow::cudaRender(FluidParams<CFT> &cp)
{
  //// render field
  FluidField<CFT> *renderSrc = reinterpret_cast<FluidField<CFT>*>(mStates.back());
  // render 2D EM views
  if(mDisplayUI->showEMView)  { mEMTex.clear();  renderFieldEM  (*renderSrc,     mEMTex,  mParams.rp, mParams.cp); }
  if(mDisplayUI->showMatView) { mMatTex.clear(); renderFieldMat (renderSrc->mat, mMatTex, mParams.rp, mParams.cp); }
  // render 3D EM view
  if(mFileRendering)
    { // render separately file output (different aspect ratio)
      Vec2f aspect = Vec2f(m3DGlView.r.aspect(), 1.0);
      if(mDisplayUI->show3DView) { m3DGlTex.clear(); raytraceFieldEM(*renderSrc, m3DGlTex, mCamera, mParams.rp, cp, aspect); }
    }
  else
    {
      Vec2f aspect = Vec2f(m3DView.r.aspect(), 1.0f);
      if(mDisplayUI->show3DView) { m3DTex.clear(); raytraceFieldEM(*renderSrc, m3DTex, mCamera, mParams.rp, cp, aspect); }
    }
}




////////////////////////////////////////////////
//// INPUT HANDLING ////////////////////////////
////////////////////////////////////////////////


// handle input for 2D views
void SimWindow::handleInput2D(ScreenView<CFT> &view, const std::string &id)
{
  ImGuiStyle &style = ImGui::GetStyle();
  ImGuiIO    &io    = ImGui::GetIO();
  
  Vec2f mpos    = ImGui::GetMousePos();
  Vec2f mposSim = screenToSim2D(mpos, mSimView2D, view.r);
  view.hovered  = mSimView2D.contains(mposSim);
  if(view.hovered)
    {
      // view.clickPos = mpos;
      view.mposSim = CFV3{(float)mposSim.x-mFieldUI->cp->fp.x, (float)mposSim.y-mFieldUI->cp->fp.y, (float)mDisplayUI->rp->zRange.y};
      mNewFrameVec = true;
    }
  else
    { view.mposSim = CFV3{NAN, NAN, NAN}; }

  bool newClick = false;
  
  if(view.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left))   { view.clicked |=  MOUSEBTN_LEFT;   newClick = true; }
  else if(ImGui::IsMouseReleased(ImGuiMouseButton_Left))             { view.clicked &= ~MOUSEBTN_LEFT;   view.mods = 0;   }
  if(view.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right))  { view.clicked |=  MOUSEBTN_RIGHT;  newClick = true; }
  else if(ImGui::IsMouseReleased(ImGuiMouseButton_Right))            { view.clicked &= ~MOUSEBTN_RIGHT;  view.mods = 0;   }
  if(view.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) { view.clicked |=  MOUSEBTN_MIDDLE; newClick = true; }
  else if(ImGui::IsMouseReleased(ImGuiMouseButton_Middle))           { view.clicked &= ~MOUSEBTN_MIDDLE; view.mods = 0;   }

  if(mLockViews || mForcePause) { return; } // ignore user input for rendering
  if(newClick) // new mouse click
    {
      // if(view.clicked == MOUSEBTN_LEFT || view.clicked == MOUSEBTN_RIGHT || view.clicked == MOUSEBTN_MIDDLE)
      //   { view.clickPos = mpos; } // set new position only on first button click
      view.mods = ((io.KeyShift ? GLFW_MOD_SHIFT   : 0) |
                   (io.KeyCtrl  ? GLFW_MOD_CONTROL : 0) |
                   (io.KeyAlt   ? GLFW_MOD_ALT     : 0));
    }
  
  //// view manipulation ////
  bool viewChanged = false;

  // left/middle drag --> pan
  ImGuiMouseButton btn = (ImGui::IsMouseDragging(ImGuiMouseButton_Left) ? ImGuiMouseButton_Left :
                          (ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ? ImGuiMouseButton_Middle : -1));
  if(((view.clickBtns(MOUSEBTN_LEFT)   && btn == ImGuiMouseButton_Left && !view.clickMods(GLFW_MOD_CONTROL | GLFW_MOD_ALT)) ||
      (view.clickBtns(MOUSEBTN_MIDDLE) && btn == ImGuiMouseButton_Middle)))
    {
      Vec2f dmp = ImGui::GetMouseDragDelta(btn); ImGui::ResetMouseDragDelta(btn);
      dmp.x *= -1.0f;
      mSimView2D.move(screenToSim2D(dmp, mSimView2D, view.r, true)); // recenter mouse at same sim position
      viewChanged = true;
    }
  
  // mouse scroll --> zoom
  if(view.hovered && std::abs(io.MouseWheel) > 0.0f)
    {
      if(io.KeyCtrl && io.KeyShift)     // signal pen depth
        { mDrawUI->sigPen.depth += (io.MouseWheel > 0 ? 1 : -1); }
      else if(io.KeyAlt && io.KeyShift) // material pen depth
        { mDrawUI->matPen.depth += (io.MouseWheel > 0 ? 1 : -1); }
      else // zoom camera
        {
          float vel = (io.KeyAlt ? 1.36 : (io.KeyShift ? 1.011 : 1.055)); // scroll velocity
          float scale = (io.MouseWheel > 0.0f ? 1.0/vel : vel);
          mSimView2D.scale(scale);
          Vec2f mposSim2 = screenToSim2D(mpos, mSimView2D, view.r, false);
          mSimView2D.move(mposSim-mposSim2); // center so mouse doesn't change position
          viewChanged = true;
        }
    }
  
  // tooltip 
  if(view.hovered && io.KeyShift)
    {
      float2 fs = float2{(float)mParams.cp.fs.x, (float)mParams.cp.fs.y};
      float2 cs = float2{mUnits.dL, mUnits.dL};
      Vec2i fi    = makeV<int2>(floor((float2{mposSim.x, mposSim.y} / cs)));
      Vec2i fiAdj = Vec2i(std::max(0, std::min(mParams.cp.fs.x-1, fi.x)), std::max(0, std::min(mParams.cp.fs.y-1, fi.y)));

      int zi = mDisplayUI->rp->zRange.y;
      
      // pull device data
      std::vector<CFV3> v  (mStates.size(), CFV3{NAN, NAN, NAN});
      std::vector<CFT>  p  (mStates.size(), (CFT)NAN);
      std::vector<CFT>  Qn (mStates.size(), (CFT)NAN);
      std::vector<CFT>  Qp (mStates.size(), (CFT)NAN);
      std::vector<CFV3> Qv (mStates.size(), CFV3{NAN, NAN, NAN});
      std::vector<CFV3> E  (mStates.size(), CFV3{NAN, NAN, NAN});
      std::vector<CFV3> B  (mStates.size(), CFV3{NAN, NAN, NAN});
      std::vector<Material<float>> mat(mStates.size(), Material<float>());
      if(fi.x >= 0 && fi.x < mParams.cp.fs.x && fi.y >= 0 && fi.y < mParams.cp.fs.y)
        {
          for(int i = 0; i < mStates.size(); i++)
            {
              FluidField<CFT> *src = reinterpret_cast<FluidField<CFT>*>(mStates[mStates.size()-1-i]);
              if(src)
                {
                  // get data from top displayed layer (TODO: average or graph layers?)
                  cudaMemcpy(&v[i],   src->v.dData   + src->v.idx  (fi.x, fi.y, zi), sizeof(CFV3), cudaMemcpyDeviceToHost);
                  cudaMemcpy(&p[i],   src->p.dData   + src->p.idx  (fi.x, fi.y, zi), sizeof(CFT),  cudaMemcpyDeviceToHost);
                  cudaMemcpy(&Qn[i],  src->Qn.dData  + src->Qn.idx (fi.x, fi.y, zi), sizeof(CFT),  cudaMemcpyDeviceToHost);
                  cudaMemcpy(&Qp[i],  src->Qp.dData  + src->Qp.idx (fi.x, fi.y, zi), sizeof(CFT),  cudaMemcpyDeviceToHost);
                  cudaMemcpy(&Qv[i],  src->Qv.dData  + src->Qv.idx (fi.x, fi.y, zi), sizeof(CFV3), cudaMemcpyDeviceToHost);

                  cudaMemcpy(&E[i],   src->E.dData   + src->E.idx  (fi.x, fi.y, zi), sizeof(CFV3), cudaMemcpyDeviceToHost);
                  cudaMemcpy(&B[i],   src->B.dData   + src->B.idx  (fi.x, fi.y, zi), sizeof(CFV3), cudaMemcpyDeviceToHost);
                  cudaMemcpy(&mat[i], src->mat.dData + src->mat.idx(fi.x, fi.y, zi), sizeof(Material<CFT>), cudaMemcpyDeviceToHost);
                }
            }
        }
      Vec2f dataPadding = Vec2f(6.0f, 6.0f);
      
      ImGui::BeginTooltip();
      {
        ImDrawList *ttDrawList = ImGui::GetWindowDrawList(); // maximum text size per column
        ImGui::Text(" mousePos: <%.3f, %.3f> (index: <%d, %d, %d>)",  mposSim.x, mposSim.y, fi.x, fi.y, zi);
        
        Vec2f tSize = ImGui::CalcTextSize(("T"+std::to_string(mStates.size()-1)).c_str()); // max state label width
        Vec2f p0 = ImGui::GetCursorScreenPos();
        for(int i = 0; i < mStates.size(); i++)
          {
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
                ImGui::Text(" v   = < %12s, %12s, %12s >", fAlign(v[i].x,   4).c_str(), fAlign(v[i].y,   4).c_str(), fAlign(v[i].z,   4).c_str());
                ImGui::Text(" p   =   %12s",               fAlign(p[i],     4).c_str());
                ImGui::Text(" Q   =   (-) %8s, (+) %8s = %8s", fAlign(Qn[i],   4).c_str(), fAlign(Qp[i],   4).c_str(), fAlign(Qp[i]-Qn[i],   4).c_str());
                ImGui::Text(" Qv  = < %12s, %12s, %12s >", fAlign(Qv[i].x,  4).c_str(), fAlign(Qv[i].y,  4).c_str(), fAlign(Qv[i].z,  4).c_str());
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
    
  // add keyframe
  if(viewChanged) { mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_VIEW2D>(mKeyFrameUI->cursor(), mSimView2D)); }
}

// handle input for 3D views
void SimWindow::handleInput3D(ScreenView<CFT> &view, const std::string &id)
{
  ImGuiStyle &style = ImGui::GetStyle();
  ImGuiIO    &io    = ImGui::GetIO();
  
  Vec2f aspect = Vec2f(view.r.aspect(), 1.0);
  CFV3  fs     = CFV3{(CFT)mParams.cp.fs.x, (CFT)mParams.cp.fs.y, (CFT)mParams.cp.fs.z};
  CFV3  fSize  = fs*mUnits.dL;
  Ray<CFT> ray; // ray projected from mouse position within view
  Vec2f mpos = ImGui::GetMousePos();
  view.hovered = view.r.contains(mpos);
  Vec3f fpLast = view.mposSim;
  if(view.hovered)
    {
      ray = mCamera.castRay(to_cuda(Vec2f((mpos - view.r.p1)/view.r.size())), float2{aspect.x, aspect.y});
      Vec2f tp = cubeIntersectHost(Vec3f(mParams.cp.fp*mUnits.dL), Vec3f(fSize), ray);

      if(tp.x > 0.0) // tmin
        {
          Vec3f wp = ray.pos + ray.dir*(tp.x+0.00001); // world-space pos of field outer intersection
          Vec3f fp = (wp - mParams.cp.fp*mUnits.dL) / fSize * fs;
          view.mposSim = fp;
        }
      else { view.mposSim = CFV3{NAN, NAN, NAN}; }
    }
  else { view.mposSim = CFV3{NAN, NAN, NAN}; }

  bool newClick = false;
  if(view.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left))   { view.clicked |=  MOUSEBTN_LEFT;   newClick = true;  }
  else if(ImGui::IsMouseReleased(ImGuiMouseButton_Left))             { view.clicked &= ~MOUSEBTN_LEFT;   view.mods = 0;    }
  if(view.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right))  { view.clicked |=  MOUSEBTN_RIGHT;  newClick = true;  }
  else if(ImGui::IsMouseReleased(ImGuiMouseButton_Right))            { view.clicked &= ~MOUSEBTN_RIGHT;  view.mods = 0;    }
  if(view.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) { view.clicked |=  MOUSEBTN_MIDDLE; newClick = true;  }
  else if(ImGui::IsMouseReleased(ImGuiMouseButton_Middle))           { view.clicked &= ~MOUSEBTN_MIDDLE; view.mods = 0;    }

  if(mLockViews || mForcePause) { return; } // ignore user input for rendering
  if(newClick) // new mouse click
    {
      // if(view.clicked == MOUSEBTN_LEFT || view.clicked == MOUSEBTN_RIGHT || view.clicked == MOUSEBTN_MIDDLE)
      //   { view.clickPos = mpos; } // set new position only on first button click
      view.mods = ((io.KeyShift ? GLFW_MOD_SHIFT   : 0) |
                   (io.KeyCtrl  ? GLFW_MOD_CONTROL : 0) |
                   (io.KeyAlt   ? GLFW_MOD_ALT     : 0));
    }
  
  //// view manipulation
  bool viewChanged = false;
  CFT shiftMult = (io.KeyShift ? 0.1f  : 1.0f);
  CFT ctrlMult  = (io.KeyCtrl  ? 4.0f  : 1.0f);
  CFT altMult   = (io.KeyAlt   ? 16.0f : 1.0f);
  CFT keyMult   = shiftMult * ctrlMult * altMult;

  Vec2f viewSize = view.r.size();
  CFT S = (2.0*tan(mCamera.fov/2.0*M_PI/180.0));
  
  ImGuiMouseButton btn = (ImGui::IsMouseDragging(ImGuiMouseButton_Left) ? ImGuiMouseButton_Left :
                          (ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ? ImGuiMouseButton_Middle : -1));
  if(((btn == ImGuiMouseButton_Left && view.clickBtns(MOUSEBTN_LEFT) && !view.clickMods(GLFW_MOD_CONTROL | GLFW_MOD_ALT)) ||
      (btn == ImGuiMouseButton_Middle && view.clickBtns(MOUSEBTN_MIDDLE))))
    { // left/middle drag --> pan camera
      Vec2f dmp = ImGui::GetMouseDragDelta(btn); ImGui::ResetMouseDragDelta(btn);
      dmp.x *= -1.0f;
      CFV3 fpos = CFV3{mParams.cp.fp.x, mParams.cp.fp.y, mParams.cp.fp.z};
      CFV3 cpos = CFV3{mCamera.pos.x, mCamera.pos.y, mCamera.pos.z};
      CFV3 fsize = CFV3{(float)mParams.cp.fs.x, (float)mParams.cp.fs.y, (float)mParams.cp.fs.z};
      CFT lengthMult = (length(cpos-fpos) +
                        length(cpos - (fpos + fsize)) +
                        length(cpos-(fpos + fsize/2.0)) +
                        length(cpos - (fpos + fsize/2.0)) +
                        length(cpos-CFV3{fpos.x, fpos.y + fsize.y/2.0f, fpos.z}) +
                        length(cpos-CFV3{fpos.x + fsize.x/2.0f, fpos.y, fpos.z} ))/6.0f;

      mCamera.pos += (mCamera.right*dmp.x/viewSize.x*aspect.x + mCamera.up*dmp.y/viewSize.y*aspect.y)*lengthMult*S*shiftMult*0.8;
      viewChanged = true;
    }

  btn = (ImGui::IsMouseDragging(ImGuiMouseButton_Right) ? ImGuiMouseButton_Right : -1);
  if(btn == ImGuiMouseButton_Right && view.clickBtns(MOUSEBTN_RIGHT))
    { // right drag --> rotate camera
      Vec2f dmp = Vec2f(ImGui::GetMouseDragDelta(btn)); ImGui::ResetMouseDragDelta(btn);
      dmp = -dmp;
      float2 rAngles = float2{dmp.x, dmp.y} / float2{viewSize.x, viewSize.y} * 6.0 * tan(mCamera.fov/2*M_PI/180.0) * shiftMult;
      CFV3 rOffset = CFV3{(CFT)mParams.cp.fs.x, (CFT)mParams.cp.fs.y, (CFT)mParams.cp.fs.z}*mUnits.dL / 2.0;
      
      mCamera.pos -= rOffset; // offset to center rotation
      mCamera.rotate(rAngles);
      mCamera.pos += rOffset;
      viewChanged = true;
    }
  
  if(view.hovered && std::abs(io.MouseWheel) > 0.0f)
    { // mouse scroll --> zoom
      if(io.KeyCtrl && io.KeyShift)     // signal pen depth
        { mDrawUI->sigPen.depth += (io.MouseWheel > 0 ? 1 : -1); }
      else if(io.KeyAlt && io.KeyShift) // material pen depth
        { mDrawUI->matPen.depth += (io.MouseWheel > 0 ? 1 : -1); }
      else // zoom camera
        { mCamera.pos += ray.dir*shiftMult*(io.MouseWheel/20.0)*length(mCamera.pos); }
      viewChanged = true;
    }
  
  // add keyframe (TODO: only while recording)
  if(viewChanged) { mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_VIEW3D>(mKeyFrameUI->cursor(), mCamera.desc)); }
}


// global input handling/routing
void SimWindow::handleInput(const Vec2f &frameSize, const std::string &id)
{
  ImGuiStyle &style = ImGui::GetStyle();
  ImGuiIO    &io    = ImGui::GetIO();

  if(ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) && std::abs(io.MouseWheel) > 0.0f)
    {
      int delta = (int)io.MouseWheel;
      if(io.MouseWheel != 0 && delta == 0) { delta = io.MouseWheel < 0 ? -1 : 1; }
      if(ImGui::IsKeyDown(GLFW_KEY_MINUS))      // (-)+SCROLL --> adjust lower z index (absorbs mouse wheel input)
        {
          mParams.rp.zRange.x = std::max(0, std::min(mParams.cp.fs.z-1, mParams.rp.zRange.x+delta)); io.MouseWheel = 0.0f;
          mParams.rp.zRange.y = std::max(mParams.rp.zRange.x, mParams.rp.zRange.y);
        }
      else if(ImGui::IsKeyDown(GLFW_KEY_EQUAL)) // (+)+SCROLL --> adjust upper z index (absorbs mouse wheel input)
        {
          mParams.rp.zRange.y = std::max(0, std::min(mParams.cp.fs.z-1, mParams.rp.zRange.y+delta)); io.MouseWheel = 0.0f;
          mParams.rp.zRange.x = std::min(mParams.rp.zRange.x, mParams.rp.zRange.y);
        }      
    }
  
  // draw rendered views of simulation (live on screen)
  ImGuiWindowFlags wFlags = (ImGuiWindowFlags_NoTitleBar      | ImGuiWindowFlags_NoCollapse        |
                             ImGuiWindowFlags_NoMove          | ImGuiWindowFlags_NoResize          |
                             ImGuiWindowFlags_NoScrollbar     | ImGuiWindowFlags_NoScrollWithMouse |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus );
  if(ImGui::BeginChild(("##simView"+id).c_str(), frameSize, false, wFlags))
  {
    // cudaDeviceSynchronize(); // TODO: settle on specific synchronization point(s)
    if(mDisplayUI->showEMView)  { handleInput2D(mEMView,  "EM");  }      // EM view
    if(mDisplayUI->showMatView) { handleInput2D(mMatView, "Material"); } // Material view
    if(mDisplayUI->show3DView)  { handleInput3D(m3DView,  "3D");  }      // 3D view (ray marching)
  }
  ImGui::EndChild();

  // record external forces from user (keyframing)
  CFV3 mposSim = CFV3{NAN, NAN, NAN};
  if     (mEMView.hovered)  { mposSim = to_cuda(mEMView.mposSim);  }
  else if(mMatView.hovered) { mposSim = to_cuda(mMatView.mposSim); }
  else if(m3DView.hovered)  { mposSim = to_cuda(m3DView.mposSim);  }
  float  cs = mUnits.dL;
  CFV3 fs = CFV3{(float)mParams.cp.fs.x, (float)mParams.cp.fs.y, (float)mParams.cp.fs.z};
  CFV3 mpfi = (mposSim) / cs;

  FluidField<CFT> *src  = reinterpret_cast<FluidField<CFT>*>(mStates.back());  // previous field state
  
  // draw signal
  CFV3 mposLast = mSigMPos;
  mSigMPos = CFV3{NAN, NAN, NAN};
  mParams.rp.sigPenHighlight = false;
  
  bool hover = m3DView.hovered || mEMView.hovered || mMatView.hovered;
  bool active = false;
  bool apply = false;
  CFV3 p = CFV3{NAN, NAN, NAN};
  bool posChanged = false;
  bool penChanged = false; // (mDrawUI->sigPen);// TODO: find good way to keep trazck of pen changes (settings?)

  if(!mLockViews && !mForcePause && io.KeyCtrl)
    {
      if(m3DView.hovered)
        {
          CFV3 fp = to_cuda(m3DView.mposSim);
          CFV3 vDepth = CFV3{(fp.x <= 1 ? 1.0f : (fp.x >= cpCopy.fs.x-1 ? -1.0f : 0.0f)),
                             (fp.y <= 1 ? 1.0f : (fp.y >= cpCopy.fs.y-1 ? -1.0f : 0.0f)),
                             (fp.z <= 1 ? 1.0f : (fp.z >= cpCopy.fs.z-1 ? -1.0f : 0.0f)) };
          p = fp + vDepth*mDrawUI->sigPen.depth;
          apply = (m3DView.clickBtns(MOUSEBTN_LEFT) && (m3DView.clickMods(GLFW_MOD_CONTROL) || mDrawUI->sigPen.active));
        }
      if(mEMView.hovered || mMatView.hovered)
        {
          mpfi.z = cpCopy.fs.z - 1 - mDrawUI->sigPen.depth; // Z depth relative to top visible layer
          p = CFV3{mpfi.x, mpfi.y, mpfi.z};
          apply = ((mEMView.clickBtns(MOUSEBTN_LEFT) &&  (mEMView.clickMods(GLFW_MOD_CONTROL)  || mDrawUI->sigPen.active)) ||
                   (mMatView.clickBtns(MOUSEBTN_LEFT) && (mMatView.clickMods(GLFW_MOD_CONTROL) || mDrawUI->sigPen.active)));
        }
      if(hover)
        {
          posChanged = (mposLast != p) && (mDrawUI->sigPen.startTime >= 0.0);
          //penChanged = false; // (mDrawUI->sigPen);// TODO: find good way to keep track of pen changes (settings?)

          mSigMPos = p;
          mDrawUI->sigPen.mouseSpeed = length(mSigMPos - mposLast);
          mParams.rp.penPos = p;
          mParams.rp.sigPenHighlight = true;
          mParams.rp.sigPen = mDrawUI->sigPen;
          if(apply)
            { // draw signal to intermediate E/B fields (needs to be blended to avoid peristent blobs of
              active = true;
              if(mDrawUI->sigPen.startTime < 0.0)
                { // signal started
                  mDrawUI->sigPen.startTime = mInfo.t;
                  std::cout << "NEW SIG ADD EVENT\n";
                  mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_SIGNAL_ADD>(mKeyFrameUI->cursor(), "MANUAL", mSigMPos, mDrawUI->sigPen));
                }
              if((mFieldUI->running || cpCopy.u.dt != 0) && !mForcePause)
                { addSignal(p, *mInputV, *mInputP, *mInputQn, *mInputQp, *mInputQv, *mInputE, *mInputB, mDrawUI->sigPen, cpCopy, mUnits.dt); }
              else
                { addSignal(p, src->v, src->p, src->Qn, src->Qp, src->Qv, src->E, src->B, mDrawUI->sigPen, cpCopy, mUnits.dt*(CFT)0.1); }
            }
        }
    }

  if(active)
    {
      if(posChanged)
        {
          std::cout << "NEW SIG MOVE EVENT\n";
          mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_SIGNAL_MOVE>(mKeyFrameUI->cursor(), "MANUAL", mSigMPos));
       }
      else if(penChanged)
        {
          std::cout << "NEW SIG ADD EVENT\n";
          mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_SIGNAL_PEN >(mKeyFrameUI->cursor(), "MANUAL", mDrawUI->sigPen));
        }
    }
  else 
    {
      if(mParams.verbose) { std::cout << "SIG NOT ACTIVE\n"; }
      if(mDrawUI->sigPen.startTime >= 0.0) // signal stopped
        {
          std::cout << "NEW SIG REMOVE EVENT\n";
          mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_SIGNAL_REMOVE>(mKeyFrameUI->cursor(), "MANUAL"));
        }
      mDrawUI->sigPen.startTime = -1.0; // reset start time
      mSigMPos = CFV3{NAN, NAN, NAN};
    }
      
  // record  material placement
  mposLast = mMatMPos;
  mMatMPos = CFV3{NAN, NAN, NAN};
  mParams.rp.matPenHighlight = false;
  
  hover = m3DView.hovered | mEMView.hovered || mMatView.hovered;
  active = false;
  apply = false;
  p = CFV3{NAN, NAN, NAN};
  posChanged = false;
  penChanged = false; // (mDrawUI->matPen);// TODO: find good way to keep trazck of pen changes (settings?)

  if(!mLockViews && !mForcePause && io.KeyAlt)
    {
      if(m3DView.hovered)
        {
          CFV3 fp = to_cuda(m3DView.mposSim);
          CFV3 vDepth = CFV3{(fp.x <= 1 ? 1.0f : (fp.x >= cpCopy.fs.x-1 ? -1.0f : 0.0f)),
                             (fp.y <= 1 ? 1.0f : (fp.y >= cpCopy.fs.y-1 ? -1.0f : 0.0f)),
                             (fp.z <= 1 ? 1.0f : (fp.z >= cpCopy.fs.z-1 ? -1.0f : 0.0f)) };
          p = fp + vDepth*mDrawUI->matPen.depth;
          apply = (m3DView.clickBtns(MOUSEBTN_LEFT) && (m3DView.clickMods(GLFW_MOD_ALT) || mDrawUI->matPen.active));
        }
      if(mEMView.hovered || mMatView.hovered)
        {
          mpfi.z = cpCopy.fs.z-1-mDrawUI->matPen.depth; // Z depth relative to top visible layer
          p = CFV3{mpfi.x, mpfi.y, mpfi.z};
          apply = ((mEMView.clickBtns(MOUSEBTN_LEFT)  && (mEMView.clickMods(GLFW_MOD_ALT)  || mDrawUI->matPen.active)) ||
                   (mMatView.clickBtns(MOUSEBTN_LEFT) && (mMatView.clickMods(GLFW_MOD_ALT) || mDrawUI->matPen.active)));
        }

      apply &= !isnan(p);
      if(hover)
        {
          posChanged = (mposLast != p) && (mDrawUI->matPen.startTime >= 0.0);
          //penChanged = false; // (mDrawUI->matPen);// TODO: find good way to keep trazck of pen changes (settings?)

          mMatMPos = p; 
          mParams.rp.penPos = p;
          mParams.rp.matPenHighlight = true;
          mParams.rp.matPen = mDrawUI->matPen;
          if(apply)
            {
              active = true;
              if(mDrawUI->sigPen.startTime < 0.0)
                { // signal started
                  mDrawUI->sigPen.startTime = mInfo.t;
                }
              std::cout << "NEW MATERIAL EVENT\n";
              mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_MATERIAL>(mKeyFrameUI->cursor(), mMatMPos, mDrawUI->matPen));
              //mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_MATERIAL>(mInfo.t, p, mDrawUI->matPen));
            }
        }
    }
  
  if(active && (posChanged || penChanged))
    {
      std::cout << "MAT ADD/MOVE/PEN EVENT\n";
      mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_MATERIAL>(mKeyFrameUI->cursor(), mMatMPos, mDrawUI->matPen));
    }
  else 
    {
      mDrawUI->matPen.startTime = -1.0; // reset start time
      mMatMPos = CFV3{NAN, NAN, NAN};
    }
}



//////////////////////////////////////////////////////
//// OVERLAYS ////////////////////////////////////////
//////////////////////////////////////////////////////

void SimWindow::makeVectorField3D()
{
  if(mNewFrameVec)
    {
      FluidField<CFT> *src = reinterpret_cast<FluidField<CFT>*>(mStates.back());
      if(!mVBuffer3D.allocated()) { mVBuffer3D.create(src->size.x*src->size.y*2); } // CudaVBO test
      fillVLines(*src, mVBuffer3D, mParams.cp);
    }
}

// draws 2D vector field overlay
void SimWindow::drawVectorField2D(ScreenView<CFT> &view)
{
  ImGuiStyle &style = ImGui::GetStyle();
  ImGuiIO    &io    = ImGui::GetIO();
  Vec2f mpos = ImGui::GetMousePos();
  Vec2f fp   = screenToSim2D(mpos, mSimView2D, view.r)/mUnits.dL;
  
  // draw vector field data
  FluidField<CFT> *src = reinterpret_cast<FluidField<CFT>*>(mStates.back());
  if(src && mParams.vp.drawVectors && mFieldDrawList)
    {
      Vec2i fi    = makeV<int2>(float2{floor(fp.x), floor(fp.y)});
      Vec2f fo    = fp - fi;
      Vec2i fiAdj = Vec2i(std::max(0, std::min(mParams.cp.fs.x-1, fi.x)), std::max(0, std::min(mParams.cp.fs.y-1, fi.y)));

      float viewScale = max(mSimView2D.size() / view.r.size());
      
      int vRad     = mParams.vp.vecMRadius;
      int cRad     = mParams.vp.vecCRadius;
      int vSpacing = mParams.vp.vecSpacing;

      int2 iMin; int2 iMax;
      if(mParams.vp.mouseRadius)
        {
          if(vRad > cRad) { vSpacing = std::max(vSpacing, (int)ceil(vRad/(float)cRad)); }
          iMin = int2{std::max(fi.x-vRad*vSpacing, 0), std::max(fi.y-vRad*vSpacing, 0)};
          iMax = int2{std::min(fi.x+vRad*vSpacing, mParams.cp.fs.x-1)+1,
                      std::min(fi.y+vRad*vSpacing, mParams.cp.fs.y-1)+1};
        }
      else
        {
          iMin = int2{0,0};
          iMax = int2{mParams.cp.fs.x, mParams.cp.fs.y};
        }
      
      int2 iStart = int2{0, 0};
      int2 iEnd   = int2{(iMax.x - iMin.x)/vSpacing, (iMax.y - iMin.y)/vSpacing};

      // if(mNewFrameVec)
      {
        //if(!mVBuffer2D.allocated()) { mVBuffer2D.create(src->size.x*src->size.y*2); } // CudaVBO test
        
        float avgE = 0.0f; float avgB = 0.0f;
        src->E.pullData(); src->B.pullData();
        mVectorField2D.clear();
        int zTop = mParams.rp.zRange.y;
        for(int ix = iStart.x; ix <= iEnd.x; ix++)
          for(int iy = iStart.y; iy <= iEnd.y; iy++)
            {
              int xi = iMin.x + ix*vSpacing; int yi = iMin.y + iy*vSpacing; int zi = zTop;
              float2 dp = float2{(float)(xi-fi.x), (float)(yi-fi.y)};
              if(!mParams.vp.mouseRadius || dot(dp, dp) <= (float)(vRad*vRad))
                {
                  int i = src->idx(xi, yi, zi);
                  Vec3f p = Vec3f(xi+0.5f, yi+0.5f, zi+0.5f);
                  Vec2f sp = simToScreen2D(p, mSimView2D, view.r);
                  Vec3f sampleP = Vec3f(xi, yi, zi);
                  Vec3f vE; Vec3f vB;
                  if(mParams.vp.smoothVectors)
                    {
                      sampleP = Vec3f(xi+fo.x, yi+fo.y, zi);
                      if(sampleP.x >= 0 && sampleP.x < src->size.x && sampleP.y >= 0 && sampleP.y < src->size.y)
                        {
                          bool x1p   = sampleP.x+1 >= src->size.x; bool y1p = sampleP.y+1 >= src->size.y;
                          bool x1y1p = x1p || y1p;
                    
                          Vec3f E00 = Vec3f(src->E.hData[src->E.idx((int)sampleP.x,(int)sampleP.y, zi)]);
                          Vec3f E01 = (x1p   ? E00 : Vec3f(src->E.hData[src->E.idx((int)sampleP.x+1, (int)sampleP.y,   zi)]));
                          Vec3f E10 = (y1p   ? E00 : Vec3f(src->E.hData[src->E.idx((int)sampleP.x,   (int)sampleP.y+1, zi)]));
                          Vec3f E11 = (x1y1p ? E00 : Vec3f(src->E.hData[src->E.idx((int)sampleP.x+1, (int)sampleP.y+1, zi)]));
                          Vec3f B00 = Vec3f(src->B.hData[src->B.idx((int)sampleP.x,(int)sampleP.y, zi)]);
                          Vec3f B01 = (x1p   ? B00 : Vec3f(src->B.hData[src->B.idx((int)sampleP.x+1, (int)sampleP.y,   zi)]));
                          Vec3f B10 = (y1p   ? B00 : Vec3f(src->B.hData[src->B.idx((int)sampleP.x,   (int)sampleP.y+1, zi)]));
                          Vec3f B11 = (x1y1p ? B00 : Vec3f(src->B.hData[src->B.idx((int)sampleP.x+1, (int)sampleP.y+1, zi)]));

                          sp = simToScreen2D(sampleP, mSimView2D, view.r);
                          vE = blerp(E00, E01, E10, E11, fo);
                          vB = blerp(B00, B01, B10, B11, fo);
                        }
                    }
                  else
                    {
                      if(sampleP.x >= 0 && sampleP.x < src->size.x && sampleP.y >= 0 && sampleP.y < src->size.y && sampleP.z >= 0 && sampleP.z < src->size.z)
                        {
                          sp = simToScreen2D((sampleP+0.5f), mSimView2D, view.r);
                          i  = src->idx(sampleP.x, sampleP.y, sampleP.z);
                          vE = src->E.hData[i];
                          vB = src->B.hData[i];
                        }
                    }
                  float magE = length(vE);
                  float magB = length(vB);

                  mVectorField2D.push_back(FVector{p, sampleP, vE, vB});
                  avgE += magE; avgB += magB;
                }
            }
        // std::cout << "VEC AVG E: " << avgE/((iMax.x - iMin.x)*(iMax.y - iMin.y)) << "\n";
        // std::cout << "VEC AVG B: " << avgB/((iMax.x - iMin.x)*(iMax.y - iMin.y)) << "\n";
        mNewFrameVec = false;
      }
      for(auto &v : mVectorField2D)
        {
          Vec2f sp  = simToScreen2D(v.sp*mUnits.dL, mSimView2D, view.r);
          Vec2f dpE = simToScreen2D(v.vE, mSimView2D, view.r, true)*mParams.vp.vecMultE;
          Vec2f dpB = simToScreen2D(v.vB, mSimView2D, view.r, true)*mParams.vp.vecMultB;
          
          float lAlpha = mParams.vp.vecAlpha;
          Vec4f Ecol1  = *mParams.rp.getColor(FLUID_RENDER_E);
          Vec4f Bcol1  = *mParams.rp.getColor(FLUID_RENDER_B);
          Vec4f Ecol   = Vec4f(Ecol1.x, Ecol1.y, Ecol1.z, lAlpha);
          Vec4f Bcol   = Vec4f(Bcol1.x, Bcol1.y, Bcol1.z, lAlpha);
          float lw = (mParams.vp.vecLineW   / viewScale);
          // if(!mParams.vp.borderedVectors)
            // {
              mFieldDrawList->AddLine(sp, sp+dpE, ImColor(Ecol), lw);
              mFieldDrawList->AddLine(sp+dpE, sp+dpE*0.9f-Vec2f(dpE.y, -dpE.x)*0.1f, ImColor(Ecol), lw/2);
              mFieldDrawList->AddLine(sp+dpE, sp+dpE*0.9f+Vec2f(dpE.y, -dpE.x)*0.1f, ImColor(Ecol), lw/2);
              mFieldDrawList->AddLine(sp, sp+dpB, ImColor(Bcol), lw);
              mFieldDrawList->AddLine(sp+dpB, sp+dpB*0.9f-Vec2f(dpB.y, -dpB.x)*0.1f, ImColor(Bcol), lw/2);
              mFieldDrawList->AddLine(sp+dpB, sp+dpB*0.9f+Vec2f(dpB.y, -dpB.x)*0.1f, ImColor(Bcol), lw/2);
          //   }
          // else
          //   {
          //     float bAlpha = (lAlpha + mParams.vp.vecBAlpha)/2.0f;
          //     float shear0 = 0.4f;
          //     float shear1 = 1.0f;
          //     float bw     = (mParams.vp.vecBorderW / viewScale);
          //     drawLine(mFieldDrawList, sp, sp+dpE, Ecol, lw, Vec4f(0, 0, 0, bAlpha), bw, shear0, shear1);
          //     drawLine(mFieldDrawList, sp, sp+dpB, Bcol, lw, Vec4f(0, 0, 0, bAlpha), bw, shear0, shear1);
          //   }
        }
    }
}

// overlay for 2D sim views
void SimWindow::draw2DOverlay(ScreenView<CFT> &view)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  // draw axes at origin
  if(mDisplayUI->drawAxes)
    {
      float scale  = max(mParams.cp.fs)*mUnits.dL*0.25f;
      float zScale = std::max(mParams.cp.fs.z*mUnits.dL*0.15f, scale/3.0f);
      Vec2f tSize = ImGui::CalcTextSize("X");
      float pad = 5.0f;
      float zW0 = 1.0f; float zW1 = 10.0f; // width of 
      
      // X/Y
      Vec2f WO0 = Vec2f(0,0); // origin
      Vec2f Sorigin = simToScreen2D(WO0, mSimView2D, view.r);
      Vec2f Spx = simToScreen2D(WO0 + Vec2f(scale, 0), mSimView2D, view.r);
      Vec2f Spy = simToScreen2D(WO0 + Vec2f(0, scale), mSimView2D, view.r);
      drawList->AddLine(Sorigin, Spx, ImColor(X_COLOR), 2.0f);
      drawList->AddLine(Sorigin, Spy, ImColor(Y_COLOR), 2.0f);
      ImGui::PushFont(titleFontB);
      drawList->AddText((Spx+Sorigin)/2.0f - Vec2f(tSize.x/2.0f, 0) + Vec2f(0, pad), ImColor(X_COLOR), "X");
      drawList->AddText((Spy+Sorigin)/2.0f - Vec2f(tSize.x, tSize.y/2.0f) - Vec2f(2.0f*pad, 0), ImColor(Y_COLOR), "Y");

      // Z
      if(mParams.cp.fs.z > 1)
        {
          float zAngle = M_PI*4.0f/3.0f;
          Vec2f zVec   = Vec2f(cos(zAngle), sin(zAngle));
          Vec2f zVNorm = Vec2f(zVec.y, zVec.x);
          Vec2f Spz    = simToScreen2D(WO0 + zScale*zVec, mSimView2D, view.r);
          float zMin = mParams.rp.zRange.x/(float)(mParams.cp.fs.z-1);
          float zMax = mParams.rp.zRange.y/(float)(mParams.cp.fs.z-1);

          Vec2f SpzMin = simToScreen2D(WO0 + zScale*zVec*zMin, mSimView2D, view.r);
          Vec2f SpzMax = simToScreen2D(WO0 + zScale*zVec*zMax, mSimView2D, view.r);
          float zMinW = zW0*(1-zMin) + zW1*zMin;
          float zMaxW = zW0*(1-zMax) + zW1*zMax;

          drawList->AddLine(Sorigin, Spz, ImColor(Vec4f(1,1,1,0.5)), 1.0f); // grey line bg
          drawList->AddLine(SpzMin, SpzMax, ImColor(Z_COLOR), 2.0f);        // colored over view range

          // visible z range markers
          drawList->AddLine(SpzMin + Vec2f(zMinW,0), SpzMin - Vec2f(zMinW,0), ImColor(Z_COLOR), 2.0f);
          drawList->AddLine(SpzMax + Vec2f(zMaxW,0), SpzMax - Vec2f(zMaxW,0), ImColor(Z_COLOR), 2.0f);
          std::stringstream ss;   ss << mParams.rp.zRange.x; std::string zMinStr = ss.str();
          ss.str(""); ss.clear(); ss << mParams.rp.zRange.y; std::string zMaxStr = ss.str();
          if(zMin != zMax) { drawList->AddText(SpzMin + Vec2f(zMinW+pad, 0), ImColor(Z_COLOR), zMinStr.c_str()); }
          drawList->AddText(SpzMax + Vec2f(zMaxW+pad, 0), ImColor(Z_COLOR), zMaxStr.c_str());
          drawList->AddText((Spz + Sorigin)/2.0f - tSize - Vec2f(pad,pad), ImColor(Z_COLOR), "Z");
      
        }
      ImGui::PopFont();
    }
  
  // draw outline around field
  if(mDisplayUI->drawOutline)
    {
      Vec3f Wfp0 = Vec3f(mParams.cp.fp.x, mParams.cp.fp.y, mParams.cp.fp.z) * mUnits.dL;
      Vec3f Wfs  = Vec3f(mParams.cp.fs.x, mParams.cp.fs.y, mParams.cp.fs.z) * mUnits.dL;
      drawRect2D(view, Vec2f(Wfp0.x, Wfp0.y), Vec2f(Wfp0.x+Wfs.x, Wfp0.y+Wfs.y), RADIUS_COLOR);
    }

  // draw positional axes of active signal pen 
  if(!isnan(mSigMPos) && !mLockViews && !mForcePause)
    {
      Vec3f W01n  = Vec3f(mParams.cp.fp.x, mSigMPos.y, mSigMPos.z)*mUnits.dL;
      Vec3f W01p  = Vec3f(mParams.cp.fp.x + mParams.cp.fs.x, mSigMPos.y, mSigMPos.z)*mUnits.dL;
      Vec3f W10p  = Vec3f(mSigMPos.x, mParams.cp.fp.y + mParams.cp.fs.y, mSigMPos.z)*mUnits.dL;
      Vec3f W10n  = Vec3f(mSigMPos.x, mParams.cp.fp.y, mSigMPos.z)*mUnits.dL;
      // transform (NOTE: hacky)
      Vec2f S01n = simToScreen2D(W01n, mSimView2D, view.r);
      Vec2f S01p = simToScreen2D(W01p, mSimView2D, view.r);
      Vec2f S10n = simToScreen2D(W10n, mSimView2D, view.r);
      Vec2f S10p = simToScreen2D(W10p, mSimView2D, view.r);
      // X guides
      drawList->AddLine(S01n, S01p, ImColor(GUIDE_COLOR), 2.0f);
      drawList->AddCircleFilled(S01n, 3, ImColor(X_COLOR), 6);
      drawList->AddCircleFilled(S01p, 3, ImColor(X_COLOR), 6);
      // Y guides
      drawList->AddLine(S10n, S10p, ImColor(GUIDE_COLOR), 2.0f);
      drawList->AddCircleFilled(S10n, 3, ImColor(Y_COLOR), 6);
      drawList->AddCircleFilled(S10p, 3, ImColor(Y_COLOR), 6);
 
      // draw intersected radii (lenses)
      Vec3f WR0 = ((mDrawUI->sigPen.cellAlign ? floor(mSigMPos) : mSigMPos)
                   + mDrawUI->sigPen.rDist*mDrawUI->sigPen.sizeMult*mDrawUI->sigPen.xyzMult/2.0f)*mUnits.dL;
      Vec3f WR1 = ((mDrawUI->sigPen.cellAlign ? floor(mSigMPos) : mSigMPos)
                   - mDrawUI->sigPen.rDist*mDrawUI->sigPen.sizeMult*mDrawUI->sigPen.xyzMult/2.0f)*mUnits.dL;
      
      Vec2f SR0 = Vec2f(WR0.x, WR0.y);
      Vec2f SR1 = Vec2f(WR1.x, WR1.y);

      // centers
      drawList->AddCircleFilled(simToScreen2D(SR0, mSimView2D, view.r), 3, ImColor(RADIUS_COLOR), 6);
      drawList->AddCircleFilled(simToScreen2D(SR1, mSimView2D, view.r), 3, ImColor(RADIUS_COLOR), 6);

      // outlines
      Vec3f r0_3 = mDrawUI->sigPen.radius0 * mUnits.dL * mDrawUI->sigPen.sizeMult*mDrawUI->sigPen.xyzMult;
      Vec3f r1_3 = mDrawUI->sigPen.radius1 * mUnits.dL * mDrawUI->sigPen.sizeMult*mDrawUI->sigPen.xyzMult;
      Vec2f r0 = Vec2f(r0_3.x, r0_3.y); Vec2f r1 = Vec2f(r1_3.x, r1_3.y);
      if(mDrawUI->sigPen.square)
        {
          drawRect2D(view, SR0-r0, SR0+r0, RADIUS_COLOR);
          drawRect2D(view, SR1-r1, SR1+r1, RADIUS_COLOR);
        }
      else
        {
          drawEllipse2D(view, SR0, r0, RADIUS_COLOR);
          drawEllipse2D(view, SR1, r1, RADIUS_COLOR);
        }
    }
  
  // draw positional axes of active material pen 
  if(!isnan(mMatMPos) && !mLockViews && !mForcePause)
    {
      Vec3f W01n  = Vec3f(mParams.cp.fp.x, mMatMPos.y, mMatMPos.z)*mUnits.dL;
      Vec3f W01p  = Vec3f(mParams.cp.fp.x + mParams.cp.fs.x, mMatMPos.y, mMatMPos.z)*mUnits.dL;
      Vec3f W10p  = Vec3f(mMatMPos.x, mParams.cp.fp.y + mParams.cp.fs.y, mMatMPos.z)*mUnits.dL;
      Vec3f W10n  = Vec3f(mMatMPos.x, mParams.cp.fp.y, mMatMPos.z)*mUnits.dL;
      // transform (NOTE: hacky)
      Vec2f S01n = simToScreen2D(W01n, mSimView2D, view.r);
      Vec2f S01p = simToScreen2D(W01p, mSimView2D, view.r);
      Vec2f S10n = simToScreen2D(W10n, mSimView2D, view.r);
      Vec2f S10p = simToScreen2D(W10p, mSimView2D, view.r);
      // X guides
      drawList->AddLine(S01n, S01p, ImColor(GUIDE_COLOR), 2.0f);
      drawList->AddCircleFilled(S01n, 3, ImColor(X_COLOR), 6);
      drawList->AddCircleFilled(S01p, 3, ImColor(X_COLOR), 6);
      // Y guides
      drawList->AddLine(S10n, S10p, ImColor(GUIDE_COLOR), 2.0f);
      drawList->AddCircleFilled(S10n, 3, ImColor(Y_COLOR), 6);
      drawList->AddCircleFilled(S10p, 3, ImColor(Y_COLOR), 6);

      // draw intersected radii (lenses)
      Vec3f WR0 = ((mDrawUI->matPen.cellAlign ? floor(mMatMPos) : mMatMPos)
                   + mDrawUI->matPen.rDist*mDrawUI->matPen.sizeMult*mDrawUI->matPen.xyzMult/2.0f)*mUnits.dL;
      Vec3f WR1 = ((mDrawUI->matPen.cellAlign ? floor(mMatMPos) : mMatMPos)
                   - mDrawUI->matPen.rDist*mDrawUI->matPen.sizeMult*mDrawUI->matPen.xyzMult/2.0f)*mUnits.dL;
      Vec2f SR0 = Vec2f(WR0.x, WR0.y);
      Vec2f SR1 = Vec2f(WR1.x, WR1.y);

      // centers
      drawList->AddCircleFilled(simToScreen2D(SR0, mSimView2D, view.r), 3, ImColor(RADIUS_COLOR), 6);
      drawList->AddCircleFilled(simToScreen2D(SR1, mSimView2D, view.r), 3, ImColor(RADIUS_COLOR), 6);
      
      // outlines
      Vec3f r0_3 = mDrawUI->matPen.radius0 * mUnits.dL * mDrawUI->matPen.sizeMult*mDrawUI->matPen.xyzMult;
      Vec3f r1_3 = mDrawUI->matPen.radius1 * mUnits.dL * mDrawUI->matPen.sizeMult*mDrawUI->matPen.xyzMult;
      Vec2f r0 = Vec2f(r0_3.x, r0_3.y); Vec2f r1 = Vec2f(r1_3.x, r1_3.y);
      if(mDrawUI->matPen.square)
        {
          drawRect2D(view, SR0-r0, SR0+r0, RADIUS_COLOR);
          drawRect2D(view, SR1-r1, SR1+r1, RADIUS_COLOR);
        }
      else
        {
          drawEllipse2D(view, SR0, r0, RADIUS_COLOR);
          drawEllipse2D(view, SR1, r1, RADIUS_COLOR);
        }
    }
}

// overlay for 3D sim views
void SimWindow::draw3DOverlay(ScreenView<CFT> &view)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  Vec2f mpos = ImGui::GetMousePos();
  Vec2f aspect = Vec2f(view.r.aspect(), 1.0);
  Vec2f vSize = view.r.size();
  Vec2f aOffset = Vec2f(vSize.x/aspect.x - vSize.x, vSize.y/aspect.y - vSize.y)/2.0f;

  mCamera.calculate();
  
  // draw X/Y/Z axes at origin (R/G/B)
  if(mDisplayUI->drawAxes)
    {
      float scale = max(mParams.cp.fs)*mUnits.dL*0.25f;
      Vec3f WO0 = Vec3f(0,0,0); // origin
      Vec3f Wpx = WO0 + Vec3f(scale, 0, 0);
      Vec3f Wpy = WO0 + Vec3f(0, scale, 0);
      Vec3f Wpz = WO0 + Vec3f(0, 0, scale);
      // transform (NOTE: hacky)
      bool oClipped = false; bool xClipped = false; bool yClipped = false; bool zClipped = false;
      Vec2f Sorigin = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(WO0, aspect, &oClipped)) * view.r.size()/aspect - aOffset;
      Vec2f Spx     = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(Wpx, aspect, &xClipped)) * view.r.size()/aspect - aOffset;
      Vec2f Spy     = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(Wpy, aspect, &yClipped)) * view.r.size()/aspect - aOffset;
      Vec2f Spz     = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(Wpz, aspect, &zClipped)) * view.r.size()/aspect - aOffset;
      // draw axes
      if(!oClipped || !xClipped) { drawList->AddLine(Sorigin, Spx, ImColor(X_COLOR), 2.0f); }
      if(!oClipped || !yClipped) { drawList->AddLine(Sorigin, Spy, ImColor(Y_COLOR), 2.0f); }
      if(!oClipped || !zClipped) { drawList->AddLine(Sorigin, Spz, ImColor(Z_COLOR), 2.0f); }
    }

  // draw outline around field
  if(mDisplayUI->drawOutline)
    {
      Vec3f Wp = Vec3f(mParams.cp.fp.x, mParams.cp.fp.y, mParams.cp.fp.z) * mUnits.dL;
      Vec3f Ws = Vec3f(mParams.cp.fs.x, mParams.cp.fs.y, mParams.cp.fs.z) * mUnits.dL;
      drawRect3D(view, Wp, Wp+Ws, OUTLINE_COLOR);
    }

  // draw positional axes of active signal pen 
  if(!isnan(mSigMPos) && !mLockViews && !mForcePause)
    {
      Vec3f W001n  = Vec3f(mParams.cp.fp.x, mSigMPos.y, mSigMPos.z)*mUnits.dL;
      Vec3f W001p  = Vec3f(mParams.cp.fp.x + mParams.cp.fs.x, mSigMPos.y, mSigMPos.z)*mUnits.dL;
      Vec3f W010p  = Vec3f(mSigMPos.x, mParams.cp.fp.y + mParams.cp.fs.y, mSigMPos.z)*mUnits.dL;
      Vec3f W010n  = Vec3f(mSigMPos.x, mParams.cp.fp.y, mSigMPos.z)*mUnits.dL;
      Vec3f W100n  = Vec3f(mSigMPos.x, mSigMPos.y, mParams.cp.fp.x)*mUnits.dL;
      Vec3f W100p  = Vec3f(mSigMPos.x, mSigMPos.y, mParams.cp.fp.z + mParams.cp.fs.z)*mUnits.dL;
      // transform (NOTE: hacky)
      bool C001n = false; bool C001p = false; bool C010n = false; bool C010p = false; bool C100n = false; bool C100p = false;
      Vec2f S001n = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W001n, aspect, &C001n)) * view.r.size()/aspect - aOffset;
      Vec2f S001p = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W001p, aspect, &C001p)) * view.r.size()/aspect - aOffset;
      Vec2f S010n = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W010n, aspect, &C010n)) * view.r.size()/aspect - aOffset;
      Vec2f S010p = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W010p, aspect, &C010p)) * view.r.size()/aspect - aOffset;
      Vec2f S100n = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W100n, aspect, &C100n)) * view.r.size()/aspect - aOffset;
      Vec2f S100p = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W100p, aspect, &C100p)) * view.r.size()/aspect - aOffset;              
      if(!C001n || !C001p)
        { // X guides
          drawList->AddLine(S001n, S001p, ImColor(GUIDE_COLOR), 2.0f);
          drawList->AddCircleFilled(S001n, 3, ImColor(X_COLOR), 6);
          drawList->AddCircleFilled(S001p, 3, ImColor(X_COLOR), 6);
        }
      if(!C010n || !C010p)
        { // Y guides
          drawList->AddLine(S010n, S010p, ImColor(GUIDE_COLOR), 2.0f);
          drawList->AddCircleFilled(S010n, 3, ImColor(Y_COLOR), 6);
          drawList->AddCircleFilled(S010p, 3, ImColor(Y_COLOR), 6);
        }
      if(!C100n || !C100p)
        { // Z guides
          drawList->AddLine(S100n, S100p,     ImColor(GUIDE_COLOR), 2.0f);
          drawList->AddCircleFilled(S100n, 3, ImColor(Z_COLOR), 6);
          drawList->AddCircleFilled(S100p, 3, ImColor(Z_COLOR), 6);
        }

      // draw intersected radii (lenses)
      float S = 2.0*tan(mCamera.fov/2.0f*M_PI/180.0f);
      Vec3f WR0 = ((mDrawUI->sigPen.cellAlign ? floor(mSigMPos) : mSigMPos)
                   + mDrawUI->sigPen.rDist*mDrawUI->sigPen.sizeMult*mDrawUI->sigPen.xyzMult/2.0f)*mUnits.dL;
      Vec3f WR1 = ((mDrawUI->sigPen.cellAlign ? floor(mSigMPos) : mSigMPos)
                   - mDrawUI->sigPen.rDist*mDrawUI->sigPen.sizeMult*mDrawUI->sigPen.xyzMult/2.0f)*mUnits.dL;
      bool CR0 = false; bool CR1 = false;
      Vec2f SR0 = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(WR0, aspect, &CR0)) * view.r.size()/aspect - aOffset;
      Vec2f SR1 = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(WR1, aspect, &CR1)) * view.r.size()/aspect - aOffset;
      
      // centers
      drawList->AddCircleFilled(SR0, 3, ImColor(RADIUS_COLOR), 6);
      drawList->AddCircleFilled(SR1, 3, ImColor(RADIUS_COLOR), 6);

      Vec3f r0 = Vec3f(S,S,1)*mDrawUI->sigPen.radius0 * mUnits.dL * mDrawUI->sigPen.sizeMult*mDrawUI->sigPen.xyzMult;
      Vec3f r1 = Vec3f(S,S,1)*mDrawUI->sigPen.radius1 * mUnits.dL * mDrawUI->sigPen.sizeMult*mDrawUI->sigPen.xyzMult;
      if(mDrawUI->sigPen.square)
        {
          drawRect3D(view, WR0-r0, WR0+r0, RADIUS_COLOR);
          drawRect3D(view, WR1-r1, WR1+r1, RADIUS_COLOR);
        }
      else
        {
          drawEllipse3D(view, WR0, r0, RADIUS_COLOR);
          drawEllipse3D(view, WR1, r1, RADIUS_COLOR);
        }
    }
  
  // draw positional axes of active material pen 
  if(!isnan(mMatMPos) && !mLockViews && !mForcePause)
    {
      Vec3f W001n  = Vec3f(mParams.cp.fp.x, mMatMPos.y, mMatMPos.z)*mUnits.dL;
      Vec3f W010n  = Vec3f(mMatMPos.x, mParams.cp.fp.y, mMatMPos.z)*mUnits.dL;
      Vec3f W100n  = Vec3f(mMatMPos.x, mMatMPos.y, mParams.cp.fp.z)*mUnits.dL;
      Vec3f W001p  = Vec3f(mParams.cp.fp.x + mParams.cp.fs.x, mMatMPos.y, mMatMPos.z)*mUnits.dL;
      Vec3f W010p  = Vec3f(mMatMPos.x, mParams.cp.fp.y + mParams.cp.fs.y, mMatMPos.z)*mUnits.dL;
      Vec3f W100p  = Vec3f(mMatMPos.x, mMatMPos.y, mParams.cp.fp.z + mParams.cp.fs.z)*mUnits.dL;
      // transform (NOTE: hacky)
      bool C001n = false; bool C010n = false; bool C100n = false; bool C001p = false; bool C010p = false; bool C100p = false;
      Vec2f S001n = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W001n, aspect, &C001n)) * view.r.size()/aspect - aOffset;
      Vec2f S010n = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W010n, aspect, &C010n)) * view.r.size()/aspect - aOffset;
      Vec2f S100n = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W100n, aspect, &C100n)) * view.r.size()/aspect - aOffset;
      Vec2f S001p = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W001p, aspect, &C001p)) * view.r.size()/aspect - aOffset;
      Vec2f S010p = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W010p, aspect, &C010p)) * view.r.size()/aspect - aOffset;
      Vec2f S100p = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(W100p, aspect, &C100p)) * view.r.size()/aspect - aOffset;
      if(!C001n || !C001p)
        {
          drawList->AddLine(S001n, S001p, ImColor(GUIDE_COLOR), 2.0f);
          drawList->AddCircleFilled(S001n, 3, ImColor(X_COLOR), 6);
          drawList->AddCircleFilled(S001p, 3, ImColor(X_COLOR), 6);
        }
      if(!C010n || !C010p)
        {
          drawList->AddLine(S010n, S010p, ImColor(GUIDE_COLOR), 2.0f);
          drawList->AddCircleFilled(S010n, 3, ImColor(Y_COLOR), 6);
          drawList->AddCircleFilled(S010p, 3, ImColor(Y_COLOR), 6);
        }
      if(!C100n || !C100p)
        {
          drawList->AddLine(S100n, S100p,     ImColor(GUIDE_COLOR), 2.0f);
          drawList->AddCircleFilled(S100n, 3, ImColor(Z_COLOR), 6);
          drawList->AddCircleFilled(S100p, 3, ImColor(Z_COLOR), 6);
        }
      
      // draw intersected radii (lenses)
      float S = 2.0*tan(mCamera.fov/2.0f*M_PI/180.0f);
      Vec3f WR0 = ((mDrawUI->matPen.cellAlign ? floor(mMatMPos) : mMatMPos)
                   + mDrawUI->matPen.rDist*mDrawUI->matPen.sizeMult*mDrawUI->matPen.xyzMult/2.0f)*mUnits.dL;
      Vec3f WR1 = ((mDrawUI->matPen.cellAlign ? floor(mMatMPos) : mMatMPos)
                   - mDrawUI->matPen.rDist*mDrawUI->matPen.sizeMult*mDrawUI->matPen.xyzMult/2.0f)*mUnits.dL;
      bool CR0 = false; bool CR1 = false;
      Vec2f SR0 = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(WR0, aspect, &CR0)) * view.r.size()/aspect - aOffset;
      Vec2f SR1 = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(WR1, aspect, &CR1)) * view.r.size()/aspect - aOffset;

      // centers
      drawList->AddCircleFilled(SR0, 3, ImColor(RADIUS_COLOR), 6);
      drawList->AddCircleFilled(SR1, 3, ImColor(RADIUS_COLOR), 6);

      Vec3f r0 = Vec3f(S,S,1)*mDrawUI->matPen.radius0 * mUnits.dL * mDrawUI->matPen.sizeMult*mDrawUI->matPen.xyzMult;
      Vec3f r1 = Vec3f(S,S,1)*mDrawUI->matPen.radius1 * mUnits.dL * mDrawUI->matPen.sizeMult*mDrawUI->matPen.xyzMult;
      if(mDrawUI->matPen.square)
        {
          drawRect3D(view, WR0-r0, WR0+r0, RADIUS_COLOR);
          drawRect3D(view, WR1-r1, WR1+r1, RADIUS_COLOR);
        }
      else
        {
          drawEllipse3D(view, WR0, r0, RADIUS_COLOR);
          drawEllipse3D(view, WR1, r1, RADIUS_COLOR);
        }
    }
}





//////////////////////////////////////////////////////
//// DRAW/RENDER /////////////////////////////////////
//////////////////////////////////////////////////////


// render simulation views based on frame size and id
//   (id="offline" --> signifies offline rendering, uses separate texture/view)
void SimWindow::render(const Vec2f &frameSize, const std::string &id)
{
  ImGuiStyle &style = ImGui::GetStyle();
  Vec2f p0 = ImGui::GetCursorScreenPos();

  // use separate texture for file output
  CudaTexture     *tex3D  = (id == "offline" ? &m3DGlTex  : &m3DTex);
  ScreenView<CFT> *view3D = (id == "offline" ? &m3DGlView : &m3DView);

  // adjust view layout
  int numViews = ((int)mDisplayUI->showEMView + (int)mDisplayUI->showMatView + (int)mDisplayUI->show3DView);
  if(numViews == 1)
    {
      Rect2f r0 =  Rect2f(p0, p0+frameSize); // whole display
      if(mDisplayUI->showEMView)  { mEMView.r  = r0; }
      if(mDisplayUI->showMatView) { mMatView.r = r0; }
      if(mDisplayUI->show3DView)  { view3D->r  = r0; }
    }
  else if(numViews == 2)
    {
      Rect2f r0 =  Rect2f(p0, p0+Vec2f(frameSize.x/2.0f, frameSize.y));    // left side
      Rect2f r1 =  Rect2f(p0+Vec2f(frameSize.x/2.0f, 0.0f), p0+frameSize); // right side
      int n = 0;
      if(mDisplayUI->showEMView)  { mEMView.r  = (n==0 ? r0 : r1); n++; }
      if(mDisplayUI->showMatView) { mMatView.r = (n==0 ? r0 : r1); n++; }
      if(mDisplayUI->show3DView)  { view3D->r  = (n==0 ? r0 : r1); n++; }
    }
  else
    {
      Rect2f r0 =  Rect2f(p0, p0+frameSize/2.0f);              // top-left quarter
      Rect2f r1 =  Rect2f(p0+Vec2f(0.0f, frameSize.y/2.0f),    // bottom-left quarter
                          p0+Vec2f(frameSize.x/2.0f, frameSize.y));
      Rect2f r2 =  Rect2f(p0+Vec2f(frameSize.x/2.0f, 0.0f),    // right side
                          p0+frameSize);
      int n = 0;
      if(mDisplayUI->showEMView)  { mEMView.r  = (n==0 ? r0 : (n==1 ? r1 : r2)); n++; }
      if(mDisplayUI->showMatView) { mMatView.r = (n==0 ? r0 : (n==1 ? r1 : r2)); n++; }
      if(mDisplayUI->show3DView)  { view3D->r  = (n==0 ? r0 : (n==1 ? r1 : r2)); n++; }
    }

  // adjust aspect ratios if window size has changed (double precision used due to noticeable floating point error while resizing (?))
  //  TODO: improve
  double  fAspect      = mParams.cp.fs.x/(double)mParams.cp.fs.y;
  double  simAspect    = mSimView2D.aspect()/fAspect;
  double  dispAspect2D = mEMView.r.aspect()/fAspect;
  double  aRatio       = dispAspect2D/simAspect;
  if(dispAspect2D < 1.0)      { mSimView2D.scale(Vec2d(1.0, 1.0/aRatio)); }
  else if(dispAspect2D > 1.0) { mSimView2D.scale(Vec2d(aRatio, 1.0)); }

  // update 3D texture sizes
  int2 vSize3D = int2{(int)view3D->r.size().x, (int)view3D->r.size().y};
  if(mFieldUI->texRes3DMatch && mFieldUI->texRes3D != vSize3D)
    {
      mFieldUI->texRes3D = vSize3D;
      tex3D->create(int3{vSize3D.x, vSize3D.y, 1});
      cudaRender(mParams.cp);
    }
  
  // draw rendered views of simulation on screen    
  ImGuiWindowFlags wFlags = (ImGuiWindowFlags_NoTitleBar      | ImGuiWindowFlags_NoCollapse        |
                             ImGuiWindowFlags_NoMove          | ImGuiWindowFlags_NoResize          |
                             ImGuiWindowFlags_NoScrollbar     | ImGuiWindowFlags_NoScrollWithMouse |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus );
  if(ImGui::BeginChild(("##simView"+id).c_str(), frameSize, false, wFlags))
  {
    // EM view
    if(mDisplayUI->showEMView)
      {
        ImGui::SetCursorScreenPos(mEMView.r.p1);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, SIM_BG_COLOR);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, Vec2f(0, 0));
        if(ImGui::BeginChild("##emView", mEMView.r.size(), true, wFlags))
        {
          ImGui::SetCursorPos(Vec2f(10,10));
          ImGui::PushFont(titleFontB);
          ImGui::TextUnformatted("E/M");
          ImGui::PopFont();
          
          mFieldDrawList = ImGui::GetWindowDrawList();
          Vec2f fp = Vec2f(mParams.cp.fp.x, mParams.cp.fp.y);
          Vec2f fScreenPos  = simToScreen2D(fp, mSimView2D, mEMView.r);
          Vec2f fCursorPos  = simToScreen2D(fp + Vec2f(0.0f, mParams.cp.fs.y*mUnits.dL), mSimView2D, mEMView.r);
          Vec2f fScreenSize = simToScreen2D(makeV<CFV3>(mParams.cp.fs)*mUnits.dL, mSimView2D, mEMView.r, true);
          Vec2f t0(0.0f, 1.0f); Vec2f t1(1.0f, 0.0f);
          mEMTex.bind();
          ImGui::SetCursorScreenPos(fCursorPos);
          ImGui::Image(reinterpret_cast<ImTextureID>(mEMTex.texId()), fScreenSize, t0, t1, ImColor(Vec4f(1,1,1,1)));
          mEMTex.release();

          if(!mLockViews && !mForcePause) { drawVectorField2D(mEMView); }
          draw2DOverlay(mEMView);
        }
        ImGui::EndChild();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
      }
      
    // Material view
    if(mDisplayUI->showMatView)
      {
        ImGui::SetCursorScreenPos(mMatView.r.p1);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, SIM_BG_COLOR);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, Vec2f(0, 0));
        if(ImGui::BeginChild("##matView", mMatView.r.size(), true, wFlags))
        {
          ImGui::SetCursorPos(Vec2f(10,10));
          ImGui::PushFont(titleFontB);
          ImGui::TextUnformatted("Material");
          ImGui::PopFont();
          
          mFieldDrawList = ImGui::GetWindowDrawList();
          Vec2f fp = Vec2f(mParams.cp.fp.x, mParams.cp.fp.y);
          Vec2f fScreenPos  = simToScreen2D(fp, mSimView2D, mMatView.r);
          Vec2f fCursorPos  = simToScreen2D(fp + Vec2f(0.0f, mParams.cp.fs.y*mUnits.dL), mSimView2D, mMatView.r);
          Vec2f fScreenSize = simToScreen2D(makeV<CFV3>(mParams.cp.fs)*mUnits.dL, mSimView2D, mMatView.r, true);
          Vec2f t0(0.0f, 1.0f); Vec2f t1(1.0f, 0.0f);
          mMatTex.bind();
          ImGui::SetCursorScreenPos(fCursorPos);
          ImGui::Image(reinterpret_cast<ImTextureID>(mMatTex.texId()), fScreenSize, t0, t1, ImColor(Vec4f(1,1,1,1)));
          mMatTex.release();
          
          if(!mLockViews && !mForcePause) { drawVectorField2D(mMatView); }
          draw2DOverlay(mMatView);
        }
        ImGui::EndChild();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
      }
      
    // Raytraced 3D view
    if(mDisplayUI->show3DView)
      {
        ImGui::SetCursorScreenPos(view3D->r.p1);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, SIM_BG_COLOR);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, Vec2f(0, 0));
        if(ImGui::BeginChild("##3DView", view3D->r.size(), true, wFlags))
        {
          Vec2f aspect = Vec2f(1.0f, 1.0f);
          if(mFileRendering && id != "offline")
            {
              aspect.x = 1.0f/(tex3D->size.x/(float)tex3D->size.y);
              tex3D = &m3DGlTex;
              aspect.x *= tex3D->size.x/(float)tex3D->size.y;
            }
          mFieldDrawList = ImGui::GetWindowDrawList();
          Vec2f vSize = view3D->r.size()*aspect;
          Vec2f diff = (view3D->r.size() - vSize);
          
          Vec2f t0(0.0f, 1.0f); Vec2f t1(1.0f, 0.0f);
          tex3D->bind();
          ImGui::SetCursorScreenPos(view3D->r.p1 + diff/2.0f);
          ImGui::Image(reinterpret_cast<ImTextureID>(tex3D->texId()), vSize, t0, t1, ImColor(Vec4f(1,1,1,1)));
          tex3D->release();
          
          // makeVectorField3D(); // rendered separately (after ImGui frame)
          draw3DOverlay(*view3D);
          
          ImGui::SetCursorScreenPos(view3D->r.p1+Vec2f(10,10));
          ImGui::PushFont(titleFontB);
          ImGui::TextUnformatted("3D");
          ImGui::PopFont();
          
        }
        ImGui::EndChild();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
      }
  }
  ImGui::EndChild();
}



// draws live/interactive window
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
  if(ImGui::Begin("##mainView", nullptr, wFlags)) // covers full application window
  {
    ImGui::PushFont(mainFont);
    Vec2f p0 = ImGui::GetCursorScreenPos();

    // key bindings/shortcuts
    if(mKeyManager)
      {
        mCaptured = (ImGui::GetIO().WantCaptureKeyboard && // NOTE: WantCaptureKeyboard set to true when mouse clicked
                     !mEMView.clickBtns() && !mMatView.clickBtns() && !m3DView.clickBtns()); // --> enable shortcuts anyway if clicked
        //mKeyManager->update(mCaptured, mParams.verbose);
        mKeyManager->draw(mFrameSize);
        mForcePause = mKeyManager->popupOpen();
      }
    
    Vec2f sideBarSize   = mSideTabs->getSize();
    Vec2f bottomBarSize = mBottomTabs->getSize();
    mDisplaySize = (mFrameSize - Vec2f(sideBarSize.x, bottomBarSize.y) -
                    2.0f*Vec2f(style.WindowPadding) - Vec2f(1,1)*style.ItemSpacing);

    //ImGui::PushID("");
    ImGui::BeginGroup();
    { // draw views
      render(mDisplaySize);
      handleInput(mDisplaySize);
      mMouseSimPosLast = mMouseSimPos;
      if(mEMView.hovered)  { mMouseSimPos = Vec2f(mEMView.mposSim.x,  mEMView.mposSim.y);  }
      if(mMatView.hovered) { mMouseSimPos = Vec2f(mMatView.mposSim.x, mMatView.mposSim.y); }
      if(m3DView.hovered)  { mMouseSimPos = Vec2f(m3DView.mposSim.x,  m3DView.mposSim.y);  }
      // key frame UI
      // mKeyFrameUI->draw(Vec2f(mDisplaySize.x, KEYFRAME_WIDGET_H));
      
      // bottom tabs
      // ImGui::BeginGroup();
      {
        // Vec2f sp0 = Vec2f(ImGui::GetCursorPos().x, mFrameSize.y - bottomBarSize.y - style.WindowPadding.y);
        // ImGui::SetCursorPos(sp0);

        float l = frameSize.x - sideBarSize.x - 2.0f*style.WindowPadding.x - style.ItemSpacing.x;
        mBottomTabs->setLength(l);
        // mKeyFrameUI->draw(Vec2f(l, KEYFRAME_WIDGET_H));
        mBottomTabs->draw("bottomTabs");
      
        bottomBarSize = mBottomTabs->getSize();// + Vec2f(0.0f, 2.0f*style.WindowPadding.y + style.ScrollbarSize) + Vec2f(0.0f, mBottomTabs->getBarWidth());
      }
      // ImGui::EndGroup();
    }
    ImGui::EndGroup();

    // side tabs
    ImGui::SameLine();
    //ImGui::BeginGroup();
    {
      Vec2f sp0 = Vec2f(mFrameSize.x - sideBarSize.x - style.WindowPadding.x, ImGui::GetCursorPos().y);
      ImGui::SetCursorPos(sp0);
      
      mSideTabs->setLength(mFrameSize.y - 2.0f*style.WindowPadding.y - style.ItemSpacing.y);
      mSideTabs->draw("sideTabs");
      
      sideBarSize = mSideTabs->getSize();// + Vec2f(2.0f*style.WindowPadding.x + style.ScrollbarSize, 0.0f) + Vec2f(mSideTabs->getBarWidth(), 0.0f);
          
      // Vec2f minSize = Vec2f(512+sideBarSize.x+style.ItemSpacing.x, 512) + Vec2f(style.WindowPadding)*2.0f + Vec2f(style.FramePadding)*2.0f;
      // glfwSetWindowSizeLimits(mWindow, (int)minSize.x, (int)minSize.y, GLFW_DONT_CARE, GLFW_DONT_CARE);
      // ImGui::SetWindowSize(sideBarSize + Vec2f(style.ScrollbarSize, 0.0f));
    }
    //ImGui::EndGroup();

    // enforce a minimum size
    Vec2f minSize = (Vec2f(512, 512) + Vec2f(sideBarSize.x, bottomBarSize.y) +
                     style.ItemSpacing + Vec2f(style.WindowPadding)*2.0f + Vec2f(style.FramePadding)*2.0f);
    glfwSetWindowSizeLimits(mWindow, (int)minSize.x, (int)minSize.y, GLFW_DONT_CARE, GLFW_DONT_CARE);
    ImGui::SetWindowSize(sideBarSize + Vec2f(style.ScrollbarSize, 0.0f));
    
    // debug overlay
    if(mParams.debug)
      {
        Vec2f aspect3D = Vec2f(m3DView.r.aspect(), 1.0); //if(aspect3D.x < 1.0) { aspect3D.y = 1.0/aspect3D.x; aspect3D.x = 1.0; }
            
        ImDrawList *drawList = ImGui::GetForegroundDrawList();
        ImGui::PushClipRect(Vec2f(0,0), mFrameSize, false);
        ImGui::SetCursorPos(Vec2f(10.0f, 10.0f));
        ImGui::PushStyleColor(ImGuiCol_ChildBg, Vec4f(0.0f, 0.0f, 0.0f, 0.0f));
        if(ImGui::BeginChild("##debugOverlay", mDisplaySize, false, wFlags))
        {
          ImGui::Indent();
          ImGui::PushFont(titleFontB);
          ImGui::Text("Render: %.2f FPS", mInfo.fps);
          ImGui::Text("Update: %.2f FPS", mFpsUpdate);
          ImGui::PopFont();
          ImGui::Spacing();
          ImGui::Text("t =  %f  (sim time %.3fx real time)", mInfo.t, mFieldUI->running ? (fpsLast*mUnits.dt) : 0.0f);
          ImGui::Text("Mouse Sim Pos: <%f, %f>",              mMouseSimPos.x,        mMouseSimPos.y);
          ImGui::Text("3D Click Pos:  <%f, %f>",              m3DView.clickPos.x,    m3DView.clickPos.y);
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
          ImGui::Text("Camera Proj:   \n%s", mCamera.proj.toString().c_str());   ImGui::Text("Camera glProj:   \n%s", mCamera.glProj.toString().c_str());
          ImGui::Text("Camera View:   \n%s", mCamera.view.toString().c_str());   ImGui::Text("Camera glView:   \n%s", mCamera.glView.toString().c_str());
          ImGui::Text("Camera VP:     \n%s", mCamera.VP.toString().c_str());     ImGui::Text("Camera glVP:     \n%s", mCamera.glVP.toString().c_str());

          ImGui::TextUnformatted("Greek Test: ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσΤτΥυΦφΧχΨψΩω");
          
          ImGui::Unindent();
        }
        ImGui::EndChild();
        ImGui::PopClipRect();
        ImGui::PopStyleColor();
      }
    ImGui::PopFont();

    if(mFirstFrame || isinf(mCamera.pos.z) || isnan(mKeyFrameUI->view2DInit()) || isnan(mKeyFrameUI->view2DInit()))
      {
        resetViews(); mKeyFrameUI->setInitView3D(mCamera.desc); mKeyFrameUI->setInitView2D(mSimView2D);
        mFirstFrame = false;
      }


    
    if(mImGuiDemo)          // show imgui demo window    (Alt+Shift+D)
      { ImGui::ShowDemoWindow(&mImGuiDemo); }
    if(ftDemo && mFontDemo) // show FreeType test window (Alt+Shift+F)
      { ftDemo->ShowFontsOptionsWindow(&mFontDemo); mLockViews = (mImGuiDemo || mFontDemo); }
  }
  ImGui::End();
  ImGui::PopStyleVar(4);
  ImGui::PopStyleColor(1);
  
  if(mClosing) { glfwSetWindowShouldClose(mWindow, GLFW_TRUE); }
}


void SimWindow::postRender()
{
  // if(mVBuffer3D.allocated())
  //   {
  //     glDisable(GL_DEPTH_TEST);
  //     mVBuffer3D.glShader->setUniform("VP", mCamera.glVP);
  //     mVBuffer3D.draw();
  //   }
}




//
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







////////////////////////////////////////////////
//// RENDER TO FILE ////////////////////////////
////////////////////////////////////////////////


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
  bool success = true;
  if(!checkBaseRenderPath()) { return false; } // make sure base directory exists
  else
    { // check named image sub-directory for this simulation render
      mImageDir = mBaseDir + "/" + mParams.simName;
      if(!directoryExists(mImageDir))
        {
          std::cout << "Creating directory for simulation '" << mParams.simName << "' --> (" << mImageDir << ")...\n";
          if(makeDirectory(mImageDir)) { std::cout << "Successfully created directory.\n";                  success = true;  }
          else                         { std::cout << "====> ERROR: Could not make sim image directory.\n"; success = false; }
        }
    }
  return success;
}

void SimWindow::initGL(const Vec2i &texSize)
{
  if(texSize != mGlTexSize || mParams.outAlpha != mGlAlpha)
    {
      std::cout << "GL tex size: " << mGlTexSize << " --> " << texSize <<  ")...\n";
      cleanupGL();
      
      mGlTexSize = texSize;
      mGlAlpha   = mParams.outAlpha;
      
      std::cout << "Initializing GL resources...\n";
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
      std::cout << "Cleaning up GL resources...\n";
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glBindTexture(GL_TEXTURE_2D, 0);
      glDeleteFramebuffers(1, &mRenderFB); mRenderFB  = 0;
      glDeleteTextures(1, &mRenderTex); mRenderTex = 0;
      delete[] mTexData;  mTexData  = nullptr;
      delete[] mTexData2; mTexData2 = nullptr;
      mGlTexSize = Vec2i(0,0);
    }
}

void SimWindow::renderToFile()
{
  if(mFileRendering && mNewFrameOut) // || (mInfo.frame == 0 && mInfo.uStep == 0)))
    {
      Vec2i outSize = Vec2i(mParams.outSize);
      initGL(outSize);
      ImGui_ImplOpenGL3_NewFrame();
      // ImGui_ImplGlfw_NewFrame();
      
      ImGuiIO &io = ImGui::GetIO();
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
        if(ImGui::Begin("##renderView", nullptr, wFlags)); // ImGui window covering full application window
        {
          ImGui::PushFont(mainFont);
          Vec2f p0 = Vec2f(0.0f, 0.0f);
          ImGui::SetCursorScreenPos(p0);
          ImGui::BeginGroup();
          
          render(mGlTexSize, "offline");

          ImGui::EndGroup();
          ImGui::PopFont();
        }
        ImGui::End();
        ImGui::PopStyleVar(3); ImGui::PopStyleColor();
      }
      // ImGui::EndFrame();

      // render to ImGui frame to separate framebuffer
      glUseProgram(0);
      ImGui::Render();
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
      checkSimRenderPath();
      std::stringstream ss; ss << mBaseDir << "/" << mParams.simName << "/"
                               << mParams.simName << "-" << std::setfill('0') << std::setw(5) << mInfo.frame << ".png";
      std::string imagePath = ss.str();
      std::cout << "Writing Frame " << mInfo.frame << " to " << imagePath << "...\n";
      setPngCompression(mParams.pngCompression);
      writeTexture(imagePath, (const void*)mTexData2, mGlTexSize, mGlAlpha);
      mNewFrameOut = false;
    }
}

