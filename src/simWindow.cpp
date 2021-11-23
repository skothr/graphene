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
#include "settingForm.hpp"
#include "image.hpp"
#include "imtools.hpp"
#include "tools.hpp"
#include "tabMenu.hpp"
#include "toolbar.hpp"
#include "frameWriter.hpp"

#include "draw.hpp"
#include "display.hpp"
#include "overlay.hpp"
#include "render.cuh"
#include "fluid.cuh"
#include "em.cuh"

// prints aligned numbers (for field tooltip)
inline std::string fAlign(float f, int maxDigits)
{
  std::stringstream ss;
  ss << std::setprecision(8);
  if(log10(f) >= maxDigits) { ss << std::scientific << f; }
  else                      { ss << std::fixed      << f; }
  return ss.str();
}

static SimWindow *simWin = nullptr; // set in SimWindow::SimWindow()
void SimWindow::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) { simWin->keyPress(mods, key, action); }

// handle key events manually to make sure none are missed
void SimWindow::keyPress(int mods, int key, int action) { if(mKeyManager) { mKeyManager->keyPress(mods, key, action); } }

SimWindow::SimWindow(GLFWwindow *window)
  : mWindow(window)
{
  simWin = this; // NOTE/TODO: only one window total allowed for now
  glfwSetKeyCallback(mWindow, &keyCallback); // set key event callback (before ImGui GLFW initialization)

  std::vector<KeyBinding> keyBindings =
    {
     KeyBinding("Quit",            "Ctrl+Esc", "Quit program",                                    KEYBINDING_GLOBAL,[this](){ quit();                }),
     KeyBinding("Reset Views",     "F1",       "Reset viewports (align field in each view)",      KEYBINDING_GLOBAL,[this](){ resetViews(false);     }),
     KeyBinding("Reset Sim",       "F5",       "Reset full simulation (signal/material/fluid)",   KEYBINDING_GLOBAL,[this](){ resetSim(false);       }),
     KeyBinding("Reset Signals",   "F6",       "Reset signals, leaving materials intact",         KEYBINDING_GLOBAL,[this](){ resetSignals(false);   }),
     KeyBinding("Reset Materials", "F7",       "Reset materials, leaving signals intact",         KEYBINDING_GLOBAL,[this](){ resetMaterials(false); }),
     KeyBinding("Reset Fluid",     "F8",       "Reset fluid variables",                           KEYBINDING_GLOBAL,[this](){ resetFluid(false);     }),
     KeyBinding("Prev Setting Tab", "Ctrl+PgUp", "Switch to previous setting tab",                KEYBINDING_GLOBAL,[this](){ mSideTabs->prev(); }),
     KeyBinding("Next Setting Tab", "Ctrl+PgDn", "Switch to previous setting tab",                KEYBINDING_GLOBAL,[this](){ mSideTabs->next(); }),

     KeyBinding("Toggle Physics",  "Space",    "Start/stop simulation physics",              KEYBINDING_EXTRA_MODS, [this](){ togglePause(); }),
     KeyBinding("Step Forward",    "Up",       "Single physics step (+dt)", (KEYBINDING_REPEAT|KEYBINDING_MOD_MULT),[this](CFT mult){singleStepField(mult);}),
     KeyBinding("Step Backward",   "Down",     "Single physics step (-dt)", (KEYBINDING_REPEAT|KEYBINDING_MOD_MULT),[this](CFT mult){singleStepField(-mult);}),

     KeyBinding("Toggle Debug",    "Alt+D",    "Toggle debug mode",                              KEYBINDING_GLOBAL, [this](){ toggleDebug(); }),
     KeyBinding("Toggle Verbose",  "Alt+V",    "Toggle verbose mode",                            KEYBINDING_GLOBAL, [this](){ toggleVerbose(); }),
     KeyBinding("Key Bindings",    "Alt+K",    "Open Key Bindings window (view/edit bindings)",  KEYBINDING_GLOBAL, [this](){ mKeyManager->togglePopup(); }),
     KeyBinding("ImGui Demo",   "Alt+Shift+D", "Toggle ImGui demo window (examples/tools)",        KEYBINDING_NONE, [this](){ toggleImGuiDemo(); }),
     KeyBinding("Font Demo",    "Alt+Shift+F", "Toggle Font demo window",                          KEYBINDING_NONE, [this](){ toggleFontDemo(); }),
     
     // signal pen hotkeys
     KeyBinding("Signal Pen 1",    "Ctrl+1",  "Select signal pen 1",   KEYBINDING_NONE, [this](){ if(mSigPenBar) { mSigPenBar->select(0); } }),
     KeyBinding("Signal Pen 2",    "Ctrl+2",  "Select signal pen 2",   KEYBINDING_NONE, [this](){ if(mSigPenBar) { mSigPenBar->select(1); } }),
     KeyBinding("Signal Pen 3",    "Ctrl+3",  "Select signal pen 3",   KEYBINDING_NONE, [this](){ if(mSigPenBar) { mSigPenBar->select(2); } }),
     KeyBinding("Signal Pen 4",    "Ctrl+4",  "Select signal pen 4",   KEYBINDING_NONE, [this](){ if(mSigPenBar) { mSigPenBar->select(3); } }),
     KeyBinding("Signal Pen 5",    "Ctrl+5",  "Select signal pen 5",   KEYBINDING_NONE, [this](){ if(mSigPenBar) { mSigPenBar->select(4); } }),
     KeyBinding("Signal Pen 6",    "Ctrl+6",  "Select signal pen 6",   KEYBINDING_NONE, [this](){ if(mSigPenBar) { mSigPenBar->select(5); } }),
     KeyBinding("Signal Pen 7",    "Ctrl+7",  "Select signal pen 7",   KEYBINDING_NONE, [this](){ if(mSigPenBar) { mSigPenBar->select(6); } }),
     KeyBinding("Signal Pen 8",    "Ctrl+8",  "Select signal pen 8",   KEYBINDING_NONE, [this](){ if(mSigPenBar) { mSigPenBar->select(7); } }),
     KeyBinding("Signal Pen 9",    "Ctrl+9",  "Select signal pen 9",   KEYBINDING_NONE, [this](){ if(mSigPenBar) { mSigPenBar->select(8); } }),
     KeyBinding("Signal Pen 10",   "Ctrl+0",  "Select signal pen 10",  KEYBINDING_NONE, [this](){ if(mSigPenBar) { mSigPenBar->select(9); } }),
     // material pen hotkeys
     KeyBinding("Material Pen 1",   "Alt+1", "Select material pen 1",  KEYBINDING_NONE, [this](){ if(mMatPenBar) { mMatPenBar->select(0); } }),
     KeyBinding("Material Pen 2",   "Alt+2", "Select material pen 2",  KEYBINDING_NONE, [this](){ if(mMatPenBar) { mMatPenBar->select(1); } }),
     KeyBinding("Material Pen 3",   "Alt+3", "Select material pen 3",  KEYBINDING_NONE, [this](){ if(mMatPenBar) { mMatPenBar->select(2); } }),
     KeyBinding("Material Pen 4",   "Alt+4", "Select material pen 4",  KEYBINDING_NONE, [this](){ if(mMatPenBar) { mMatPenBar->select(3); } }),
     KeyBinding("Material Pen 5",   "Alt+5", "Select material pen 5",  KEYBINDING_NONE, [this](){ if(mMatPenBar) { mMatPenBar->select(4); } }),
     KeyBinding("Material Pen 6",   "Alt+6", "Select material pen 6",  KEYBINDING_NONE, [this](){ if(mMatPenBar) { mMatPenBar->select(5); } }),
     KeyBinding("Material Pen 7",   "Alt+7", "Select material pen 7",  KEYBINDING_NONE, [this](){ if(mMatPenBar) { mMatPenBar->select(6); } }),
     KeyBinding("Material Pen 8",   "Alt+8", "Select material pen 8",  KEYBINDING_NONE, [this](){ if(mMatPenBar) { mMatPenBar->select(7); } }),
     KeyBinding("Material Pen 9",   "Alt+9", "Select material pen 9",  KEYBINDING_NONE, [this](){ if(mMatPenBar) { mMatPenBar->select(8); } }),
     KeyBinding("Material Pen 10",  "Alt+0", "Select material pen 10", KEYBINDING_NONE, [this](){ if(mMatPenBar) { mMatPenBar->select(9); } }),
    };

 

  // NOTE: Any bindings not added to a group will be added to a "misc" group
  std::vector<KeyBindingGroup> keyGroups =
    {
     {"System",         {"Quit", "Prev Setting Tab", "Next Setting Tab"}},
     {"Sim Control",    {"Reset Sim", "Reset Signals", "Reset Materials","Reset Fluid", "Reset Views", "Toggle Physics", "Step Forward", "Step Backward"}},
     {"Modes/Popups",   {"Toggle Debug", "Toggle Verbose", "Key Bindings", "ImGui Demo", "Font Demo"}},
     {"Pen Selection",  {"Signal Pen 1",   "Signal Pen 2",   "Signal Pen 3",   "Signal Pen 4",   "Signal Pen 5",
                         "Signal Pen 6",   "Signal Pen 7",   "Signal Pen 8",   "Signal Pen 9",   "Signal Pen 10",
                         "Material Pen 1", "Material Pen 2", "Material Pen 3", "Material Pen 4", "Material Pen 5",
                         "Material Pen 6", "Material Pen 7", "Material Pen 8", "Material Pen 9", "Material Pen 10"}},
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
      std::cout << "= Creating SimWindow...\n";

      // set up fonts
      std::cout << "========================================================================================\n"
                << "== LOADING FONTS\n"
                << "========================================================================================\n";
      ImGuiIO &io = ImGui::GetIO();
      fontConfig = new ImFontConfig();
      fontConfig->OversampleH = FONT_OVERSAMPLE; fontConfig->OversampleV = FONT_OVERSAMPLE;

      std::cout << "====   Default flags: " << fontConfig->FontBuilderFlags << "\n";
      io.Fonts->FontBuilderFlags |= (ImGuiFreeTypeBuilderFlags_LightHinting |
                                     ImGuiFreeTypeBuilderFlags_LoadColor    |
                                     ImGuiFreeTypeBuilderFlags_Bitmap);
      fontBuilder.AddText("ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσΤτΥυΦφΧχΨψΩω");
      fontBuilder.AddText("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹");
      fontBuilder.AddText("°∥"); // NOTE: '∥' not working?
      fontBuilder.AddRanges(io.Fonts->GetGlyphRangesDefault());  // Add one of the default ranges
      fontBuilder.AddChar(0x2207);                               // Add a specific character
      fontBuilder.AddChar(0x2225);                               // Add a specific character
      fontBuilder.BuildRanges(&fontRanges);                      // Build the final result (ordered ranges with all the unique characters submitted)
      fontConfig->GlyphRanges = fontRanges.Data;

      fontConfig->SizePixels  = MAIN_FONT_HEIGHT;
      mainFont   = io.Fonts->AddFontFromFileTTF(FONT_PATH_REGULAR.string().c_str(),      MAIN_FONT_HEIGHT,  fontConfig); // main font
      mainFontB  = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD.string().c_str(),         MAIN_FONT_HEIGHT,  fontConfig);
      mainFontI  = io.Fonts->AddFontFromFileTTF(FONT_PATH_ITALIC.string().c_str(),       MAIN_FONT_HEIGHT,  fontConfig);
      mainFontBI = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD_ITALIC.string().c_str(),  MAIN_FONT_HEIGHT,  fontConfig);
      fontConfig->SizePixels  = SMALL_FONT_HEIGHT;
      smallFont   = io.Fonts->AddFontFromFileTTF(FONT_PATH_REGULAR.string().c_str(),     SMALL_FONT_HEIGHT, fontConfig); // slightly smaller font (needed?)
      smallFontB  = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD.string().c_str(),        SMALL_FONT_HEIGHT, fontConfig);
      smallFontI  = io.Fonts->AddFontFromFileTTF(FONT_PATH_ITALIC.string().c_str(),      SMALL_FONT_HEIGHT, fontConfig);
      smallFontBI = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD_ITALIC.string().c_str(), SMALL_FONT_HEIGHT, fontConfig);
      fontConfig->SizePixels  = TITLE_FONT_HEIGHT;
      titleFont   = io.Fonts->AddFontFromFileTTF(FONT_PATH_REGULAR.string().c_str(),     TITLE_FONT_HEIGHT, fontConfig); // larger font for titles/headers
      titleFontB  = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD.string().c_str(),        TITLE_FONT_HEIGHT, fontConfig);
      titleFontI  = io.Fonts->AddFontFromFileTTF(FONT_PATH_ITALIC.string().c_str(),      TITLE_FONT_HEIGHT, fontConfig);
      titleFontBI = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD_ITALIC.string().c_str(), TITLE_FONT_HEIGHT, fontConfig);
      fontConfig->SizePixels  = SUPER_FONT_HEIGHT;
      superFont   = io.Fonts->AddFontFromFileTTF(FONT_PATH_REGULAR.string().c_str(),     SUPER_FONT_HEIGHT, fontConfig); // smaller font for superscript/subscript
      superFontB  = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD.string().c_str(),        SUPER_FONT_HEIGHT, fontConfig);
      superFontI  = io.Fonts->AddFontFromFileTTF(FONT_PATH_ITALIC.string().c_str(),      SUPER_FONT_HEIGHT, fontConfig);
      superFontBI = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD_ITALIC.string().c_str(), SUPER_FONT_HEIGHT, fontConfig);
      fontConfig->SizePixels  = TINY_FONT_HEIGHT;
      tinyFont    = io.Fonts->AddFontFromFileTTF(FONT_PATH_REGULAR.string().c_str(),     TINY_FONT_HEIGHT,  fontConfig); // even smaller font (needed?)
      tinyFontB   = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD.string().c_str(),        TINY_FONT_HEIGHT,  fontConfig);
      tinyFontI   = io.Fonts->AddFontFromFileTTF(FONT_PATH_ITALIC.string().c_str(),      TINY_FONT_HEIGHT,  fontConfig);
      tinyFontBI  = io.Fonts->AddFontFromFileTTF(FONT_PATH_BOLD_ITALIC.string().c_str(), TINY_FONT_HEIGHT,  fontConfig);
      io.Fonts->Build();
      ftDemo = new FreeTypeTest(); // font demo window
      std::cout << "== (done)\n"
                << "========================================================================================\n\n";

      setDefaultSuperFont(superFont);
      
      // create setting UIs
      mFieldUI = new FieldInterface<CFT>(&mParams.cp,
                                         [this]() { resizeFields(mParams.cp.fs, false); }, // field size update
                                         [this]() // texRes2D update
                                         {
                                           std::cout << "==== Resizing 2D textures: " << mEMTex.size << "/" << mMatTex.size
                                                     << " --> " << mFieldUI->texRes2D << "\n";
                                           mEMTex.create (int3{mFieldUI->texRes2D.x, mFieldUI->texRes2D.y, 1});
                                           mMatTex.create(int3{mFieldUI->texRes2D.x, mFieldUI->texRes2D.y, 1});
                                         },
                                         [this]() // texRes3D update
                                         {
                                           std::cout << "==== Resizing 3D textures: " << m3DTex.size << " --> " << mFieldUI->texRes3D << "\n";
                                           m3DTex.create (int3{mFieldUI->texRes3D.x, mFieldUI->texRes3D.y, 1});
                                         });
      mUnitsUI   = new UnitsInterface  <CFT>(&mUnits, superFont);
      mDrawUI    = new DrawInterface   <CFT>(&mUnits, superFont);
      mDisplayUI = new DisplayInterface<CFT>(&mParams.rp, &mParams.vp, &mParams.cp.fs.z);
      mFileOutUI = new OutputInterface(&mParams.op, nullptr);
      mOtherUI   = new SettingForm("Other Settings");
      // info flags
      auto *sDBG  = new Setting<bool> ("Debug",   "debug",   &mParams.debug);   sDBG->setHelp("Show debug overlay");
      auto *sVRB  = new Setting<bool> ("Verbose", "verbose", &mParams.verbose); sVRB->setHelp("Print out more info");
      SettingGroup *infoGroup = new SettingGroup("Info", "infoGroup", { sDBG, sVRB }); mOtherUI->add(infoGroup);
      //infoGroup->setColumns(2);
      // camera flags
      auto *sCFOV = new Setting<CFT>("FOV (°)",  "fov",  &mCamera.fov);  sCFOV->setFormat(0.1f, 1.0f, "%5f"); sCFOV->setHelp("Show debug overlay");
      auto *sCN   = new Setting<CFT>("Near",     "near", &mCamera.near); sCN->setFormat(0.01f, 0.1f, "%4f");  sCN->setHelp("Print out more info");
      auto *sCF   = new Setting<CFT>("Far",      "far",  &mCamera.far);  sCF->setFormat(0.1f, 1.0f, "%4f");   sCF->setHelp("Print out more info");
      auto *sCV = new MatrixSetting<CFT,4>("View Matrix", "viewMat", &mCamera.view);
      auto *sCP = new MatrixSetting<CFT,4>("Projection Matrix", "projMat", &mCamera.proj);
      SettingGroup *camGroup = new SettingGroup("3D Camera", "cam3D", { sCFOV, sCN, sCF, sCV, sCP }); mOtherUI->add(camGroup);
      //camGroup->setColumns(3);
      // misc flags
      SettingGroup *etcGroup = new SettingGroup("Etc.", "etcGroup", { }); mOtherUI->add(etcGroup);
      auto *sVS   = new Setting<bool> ("VSync",   "vsync",   &mParams.vsync); etcGroup->add(sVS);
      sVS->setUpdateCallback([&](){ glfwSwapInterval(mParams.vsync ? 1 : 0); });
      sVS->setHelp("Toggle Vertical Sync -- syncs FPS to ratio of monitor framerate (stabilizes FPS)");
      auto *sLPF  = new Setting<float>("Physics FPS Limit", "maxPhysicsFps", &mParams.maxPhysicsFps); etcGroup->add(sLPF);
      sLPF->setToggle(&mParams.limitPhysicsFps); sLPF->setFormat(1.0f, 10.0f, "%0.2f");
      sLPF->setHelp("(experimental) Cap CUDA physics framerate");
      auto *sLRF  = new Setting<float>("Render FPS Limit",  "maxRenderFps",  &mParams.maxRenderFps);  etcGroup->add(sLRF);
      sLRF->setToggle(&mParams.limitRenderFps);  sLRF->setFormat(1.0f, 10.0f, "%0.2f");
      sLRF->setHelp("(experimental) Cap CUDA render framerate");

      
      mKeyFrameUI = new KeyFrameWidget();

      // create pen toolbar
      mSigPenBar = new Toolbar();
      mSigPenBar->setDefault(ToolButton("sp", "",
                                        [](int i, ImDrawList *drawList, const Vec2f &p0, const Vec2f &p1)
                                        { drawList->AddCircle((p0+p1)/2.0f, 0.6f*(p1.x-p0.x)/2.0f, ImColor(Vec4f(0,0,1,1)), 6, 3.0f); }));
      mSigPenBar->setHotkeyFont(smallFont);
      mSigPenBar->setAddCallback([this](int i)
      {
        if(mSigPens.size() <= i)
          {
            mSigPens.resize(i+1, nullptr);
            mSigPens[i] = (ImGui::GetIO().KeyShift && activeSigPen() ? new SignalPen<CFT>(*activeSigPen()) : new SignalPen<CFT>());
            mKeyFrameUI->addPen("SIG"+std::to_string(i), mSigPens[i]);
          }
      });
      mSigPenBar->setRemoveCallback([this](int i)
      {
        if(mSigPens.size() > i)
          {
            delete mSigPens[i];
            mSigPens.erase(mSigPens.begin()+i);
            mKeyFrameUI->removePen("SIG"+std::to_string(i));
          }
      });
      mSigPenBar->setSelectCallback([this](int i)
      {
        mDrawUI->setSigPen(mSigPens[i]);
        mKeyFrameUI->setSignalPen("SIG"+std::to_string(i));
      });
      mSigPenBar->setHorizontal(true);
      mSigPenBar->setTitle("Signals  ", titleFont);
      mSigPenBar->setWidth(69);
      addSigPen(new SignalPen<CFT>());
      mSigPenBar->select(0);

      mMatPenBar = new Toolbar();
      mMatPenBar->setDefault(ToolButton("sp", "",
                                        [](int i, ImDrawList *drawList, const Vec2f &p0, const Vec2f &p1)
                                        { drawList->AddCircle((p0+p1)/2.0f, 0.6f*(p1.x-p0.x)/2.0f, ImColor(Vec4f(0,0,1,1)), 6, 3.0f); }));
      mMatPenBar->setHotkeyFont(smallFont);
      mMatPenBar->setAddCallback([this](int i)
      {
        if(mMatPens.size() <= i)
          {
            mMatPens.resize(i+1, nullptr);
            mMatPens[i] = (ImGui::GetIO().KeyShift && activeMatPen() ? new MaterialPen<CFT>(*activeMatPen()) : new MaterialPen<CFT>());
            mKeyFrameUI->addPen("MAT"+std::to_string(i), mMatPens[i]);
          }
      });
      mMatPenBar->setRemoveCallback([this](int i)
      {
        if(mMatPens.size() > i)
          {
            delete mMatPens[i];
            mMatPens.erase(mMatPens.begin()+i);
            mKeyFrameUI->removePen("MAT"+std::to_string(i));
          }
      });
      mMatPenBar->setSelectCallback([this](int i)
      {
        mDrawUI->setMatPen(mMatPens[i]);
        mKeyFrameUI->setMaterialPen("MAT"+std::to_string(i));
      });
      mMatPenBar->setHorizontal(true);
      mMatPenBar->setTitle("Materials", titleFont);
      mMatPenBar->setWidth(69);
      addMatPen(new MaterialPen<CFT>());
      mMatPenBar->select(0);

      // create side tabs
      mSideTabs = new TabMenu(20, 1080); mSideTabs->setCollapsible(true);
      mSideTabs->add(TabDesc{[this](){ mFieldUI->draw(); },   "Field",       "Field Settings",   SETTINGS_W, 0.0f, 0.0f, titleFont});
      mSideTabs->add(TabDesc{[this](){ mUnitsUI->draw(); },   "Units",       "Base Units",       SETTINGS_W, 0.0f, 0.0f, titleFont});
      mSideTabs->add(TabDesc{[this](){ mDrawUI->draw(); },    "Draw",        "Draw Settings",    SETTINGS_W, 0.0f, 0.0f, titleFont});
      mSideTabs->add(TabDesc{[this](){ mDisplayUI->draw(); }, "Display",     "Display Settings", SETTINGS_W, 0.0f, 0.0f, titleFont});
      mSideTabs->add(TabDesc{[this](){ mFileOutUI->draw(); }, "File Output", "Render to File",   SETTINGS_W, 0.0f, 0.0f, titleFont});
      mSideTabs->add(TabDesc{[this](){ mOtherUI->draw(); },   "Other",       "Other Settings",   SETTINGS_W, 0.0f, 0.0f, titleFont});

      mBottomTabs = new TabMenu(20, 1920); mBottomTabs->setCollapsible(true); mBottomTabs->setHorizontal(true);
      mBottomTabs->add(TabDesc{[this](){ mKeyFrameUI->drawTimeline();  }, "Timeline", "Key Frame Timeline", TIMELINE_H, 0.0f, 0.0f, titleFont});
      mBottomTabs->add(TabDesc{[this](){ mKeyFrameUI->drawEventList(); }, "Events",   "Key Frame Events",   TIMELINE_H, 0.0f, 0.0f, titleFont});
      mBottomTabs->add(TabDesc{[this](){ mKeyFrameUI->drawSources();   }, "Sources",  "Key Frame Sources",  TIMELINE_H, 0.0f, 0.0f, titleFont});

      mKeyFrameUI->setSignalPen("SIG1");
      mKeyFrameUI->setMaterialPen("MAT1");
      mKeyFrameUI->view2DCallback = [this](const Rect2f &r)            { mSimView2D   = r;   };
      mKeyFrameUI->view3DCallback = [this](const CameraDesc<CFT> &cam) { mCamera.desc = cam; };

      // load settings config file (.settings.conf)
      loadSettings();

      // initialize CUDA and check for a compatible device
      if(!initCudaDevice())
        {
          std::cout << "====> ERROR: failed to initialize CUDA device!\n";
          delete fontConfig; fontConfig = nullptr;
          return false;
        }

      // create field state queue
      std::cout << "== Creating field state queue (" << STATE_BUFFER_SIZE << "x " << mParams.cp.fs << ")...\n";
      for(int i = 0; i < STATE_BUFFER_SIZE; i++) { mStates.push_back(new FluidField<CFT>()); }
      mTempState = new FluidField<CFT>();

      // create input buffers
      std::cout << "\n== Creating input buffers...\n";
      mInputV  = new Field<CFV3>(); mInputP  = new Field<CFT>();
      mInputQn = new Field<CFT>();  mInputQp = new Field<CFT>();  mInputQnv = new Field<CFV3>(); mInputQpv = new Field<CFV3>();
      mInputE  = new Field<CFV3>(); mInputB  = new Field<CFV3>();

      // create textures
      int3 ts2 = int3{mFieldUI->texRes2D.x, mFieldUI->texRes2D.y, 1};
      int3 ts3 = int3{mFieldUI->texRes3D.x, mFieldUI->texRes3D.y, 1};
      std::cout << "\n== Creating 2D textures " << std::setw(18) << ("("+to_string(ts2)+")") << "...\n";
      if(!mEMTex.create(ts2))   { std::cout << "====> ERROR: Texture creation for EM view failed!\n"; }
      if(!mMatTex.create(ts2))  { std::cout << "====> ERROR: Texture creation for Mat view failed!\n"; }
      std::cout << "\n== Creating 3D textures " << std::setw(18) << ("("+to_string(ts3)+")") << "...\n";
      if(!m3DTex.create(ts3))   { std::cout << "====> ERROR: Texture creation for 3D view failed!\n"; }
      if(!m3DGlTex.create(ts3)) { std::cout << "====> ERROR: Texture creation for 3D gl view failed!\n"; }

      // create initial state expressions
      std::cout << "\n== Creating initial condition expressions...\n";
      mFieldUI->initV.hExpr   = toExpression<CFV3>(mFieldUI->initV.str  ); mFieldUI->initP.hExpr  = toExpression<CFT> (mFieldUI->initP.str);
      mFieldUI->initQn.hExpr  = toExpression<CFT> (mFieldUI->initQn.str ); mFieldUI->initQp.hExpr = toExpression<CFT> (mFieldUI->initQp.str);
      mFieldUI->initQnv.hExpr = toExpression<CFV3>(mFieldUI->initQnv.str );
      mFieldUI->initQpv.hExpr = toExpression<CFV3>(mFieldUI->initQpv.str );
      mFieldUI->initE.hExpr   = toExpression<CFV3>(mFieldUI->initE.str  ); mFieldUI->initB.hExpr  = toExpression<CFV3>(mFieldUI->initB.str);
      mFieldUI->initEp.hExpr  = toExpression<CFT> (mFieldUI->initEp.str ); mFieldUI->initMu.hExpr = toExpression<CFT> (mFieldUI->initMu.str);
      mFieldUI->initSig.hExpr = toExpression<CFT> (mFieldUI->initSig.str);

      mCamera.fov = 60.0f; mCamera.near = 0.1f; mCamera.far = 10000.0f; // camera default config
      
      // finalize loaded settings
      mFieldUI->updateAll();   mUnitsUI->updateAll();   mDrawUI->updateAll();
      mDisplayUI->updateAll(); mFileOutUI->updateAll(); mOtherUI->updateAll();
      
      mFieldUI->running = false; // always start paused
      
      // checkBaseRenderPath();
      mInitialized = true;
    }
  return true;
}

void SimWindow::cleanup()
{
  if(mInitialized)
    {
      std::cout << "== SimWindow cleanup...\n";
      saveSettings();
      cudaDeviceSynchronize();
      
      std::cout << "==== Destroying CUDA field states...\n";
      std::vector<FieldBase*> deleted;
      for(int i = 0; i < mStates.size(); i++)
        {
          auto f = mStates[i];
          if(f)
            {
              std::cout << "==== --> " << i << "\n";
              auto iter = std::find(deleted.begin(), deleted.end(), f);
              if(iter != deleted.end()) { std::cout << "====> WARNING: State already deleted! (" << i << ")\n"; continue; }
              f->destroy(); delete f; deleted.push_back(f);
            }
        }
      mStates.clear();
      if(mTempState) { mTempState->destroy(); delete mTempState; mTempState = nullptr; }
      
      std::cout << "==== Destroying input buffers...\n";
      if(mInputV)   { mInputV->destroy();  delete mInputV;  mInputV  = nullptr; }
      if(mInputP)   { mInputP->destroy();  delete mInputP;  mInputP  = nullptr; }
      if(mInputQn)  { mInputQn->destroy(); delete mInputQn; mInputQn = nullptr; }
      if(mInputQp)  { mInputQp->destroy(); delete mInputQp; mInputQp = nullptr; }
      if(mInputQnv) { mInputQnv->destroy(); delete mInputQnv; mInputQnv = nullptr; }
      if(mInputQpv) { mInputQpv->destroy(); delete mInputQpv; mInputQpv = nullptr; }
      if(mInputE)   { mInputE->destroy();  delete mInputE;  mInputE  = nullptr; }
      if(mInputB)   { mInputB->destroy();  delete mInputB;  mInputB  = nullptr; }

      std::cout << "==== Destroying CUDA textures...\n";
      mEMTex.destroy(); mMatTex.destroy(); m3DTex.destroy(); m3DGlTex.destroy();

      // std::cout << "==== Destroying CUDA VBOs...\n";
      // mVBuffer2D.destroy(); mVBuffer3D.destroy();
      
      std::cout << "==== Destroying fonts...\n";
      if(ftDemo)      { delete ftDemo;      ftDemo      = nullptr; }
      if(fontConfig)  { delete fontConfig;  fontConfig  = nullptr; }
      std::cout << "==== Destroying UI components...\n";
      if(mSideTabs)   { delete mSideTabs;   mSideTabs   = nullptr; }
      if(mBottomTabs) { delete mBottomTabs; mBottomTabs = nullptr; }
      if(mFieldUI)    { delete mFieldUI;    mFieldUI    = nullptr; }
      if(mUnitsUI)    { delete mUnitsUI;    mUnitsUI    = nullptr; }
      if(mDrawUI)     { delete mDrawUI;     mDrawUI     = nullptr; }
      if(mDisplayUI)  { delete mDisplayUI;  mDisplayUI  = nullptr; }
      if(mFileOutUI)  { delete mFileOutUI;  mFileOutUI  = nullptr; }
      if(mOtherUI)    { delete mOtherUI;    mOtherUI    = nullptr; }
      if(mKeyFrameUI) { delete mKeyFrameUI; mKeyFrameUI = nullptr; }

      if(mFrameWriter) { delete mFrameWriter; mFrameWriter = nullptr; }
      std::cout << "==== Cleaning GL...\n";
      cleanupGL();
      std::cout << "== (done)\n";
      mInitialized = false;
    }
}

void SimWindow::saveSettings(const fs::path &path)
{
  std::cout << "========================================================================================\n"
            << "SAVING SETTINGS (" << path << ")...\n"
            << "========================================================================================\n";
  // settings
  json sim = json::object();
  sim["Field"]      = mFieldUI->toJSON();
  sim["Units"]      = mUnitsUI->toJSON();
  // sim["Draw"]       = mDrawUI->toJSON();
  sim["Display"]    = mDisplayUI->toJSON();
  sim["FileOutput"] = mFileOutUI->toJSON();
  sim["Other"]      = mOtherUI->toJSON();
  // top-level
  json js  = json::object();
  js["SimSettings"] = sim;
  js["KeyBindings"] = mKeyManager->toJSON();
  
  json spens = json::array();
  for(int i = 0; i < mSigPens.size(); i++) { spens.push_back(sigPenToJSON(*mSigPens[i])); }
  js["Signal Pens"] = spens;
  json mpens = json::array();
  for(int i = 0; i < mMatPens.size(); i++) { mpens.push_back(matPenToJSON(*mMatPens[i])); }
  js["Material Pens"] = mpens;
  
  std::ofstream f(path, std::ios::out); f << std::setw(JSON_SPACES) << js;
}

void SimWindow::loadSettings(const fs::path &path)
{
  std::cout << "========================================================================================\n"
            << "== LOADING SETTINGS (" << path << ")...\n"
            << "========================================================================================\n";
  if(fs::exists(path))
    {
      std::ifstream f(path, std::ios::in);
      json js; f >> js; // load JSON from file
      
      // key bindings
      std::string sid = "KeyBindings";
      if(js.contains(sid))
        { if(!mKeyManager->fromJSON(js[sid])) { std::cout << "====> WARNING: Failed to load '" << sid << "' group\n"; } }
      else                                    { std::cout << "====> WARNING: Could not find '" << sid << "' group\n"; }

      // signal pens
      mSigPenBar->clear();
      sid = "Signal Pens";
      if(js.contains(sid))
        {
          json jsp = js[sid];
          for(int i = 0; i < jsp.size(); i++)
            {
              SignalPen<CFT> *sPen = new SignalPen<CFT>();
              if(sigPenFromJSON(jsp[i], *sPen)) { addSigPen(sPen); }
              else { std::cout << "====> WARNING: Failed to load Signal Pen " << i << "\n"; std::cout << "\n" << jsp[i] << "\n\n"; delete sPen; }
            }
        } else     { std::cout << "====> WARNING: Could not find '" << sid << "' group\n"; }
      if(mSigPens.size() == 0) { addSigPen(new SignalPen<CFT>()); }
      mSigPenBar->select(0);
      
      // material pens
      mMatPenBar->clear();
      sid = "Material Pens";
      if(js.contains(sid))
        {
          json jmp = js[sid];
          for(int i = 0; i < jmp.size(); i++)
            {
              MaterialPen<CFT> *mPen = new MaterialPen<CFT>();
              if(matPenFromJSON(jmp[i], *mPen)) { addMatPen(mPen); }
              else { std::cout << "====> WARNING: Failed to load Material Pen " << i << "\n"; std::cout << "\n" << jmp[i] << "\n\n"; delete mPen; }
            }
        } else     { std::cout << "====> WARNING: Could not find '" << sid << "' group\n"; }
      if(mMatPens.size() == 0) { addMatPen(new MaterialPen<CFT>()); }
      mMatPenBar->select(0);
      
      // main settings
      sid = "SimSettings";
      if(js.contains(sid))
        {
          json jsSim = js[sid];
          std::map<std::string, SettingForm*> groups =
            {{ "Field",   (SettingForm*)mFieldUI   }, { "Units",      (SettingForm*)mUnitsUI   }, // { "Draw",  (SettingForm*)mDrawUI  },
             { "Display", (SettingForm*)mDisplayUI }, { "FileOutput", (SettingForm*)mFileOutUI }, { "Other", (SettingForm*)mOtherUI }};
          for(const auto &iter : groups)
            {
              sid = iter.first;
              if(jsSim.contains(sid))
                { if(!iter.second->fromJSON(jsSim[sid])) { std::cout << "====> WARNING: Failed to load '" << sid << "' setting group\n"; } }
              else                                       { std::cout << "====> WARNING: Could not find '" << sid << "' seatting group\n"; }
            }
        }
      else { std::cout << "====> WARNING: No '" << sid << "' group in settings file\n"; }
    }
  else
    {
      std::cout << "====> WARNING: Could not find settings file (" << path << ")\n";
      saveSettings();
    }
  std::cout << "== (done)\n"
            << "========================================================================================\n\n";
  // override --> don't lock views or begin rendering to file on startup
  mParams.op.active = false; mParams.op.lockViews = false;
}

bool SimWindow::resizeFields(const Vec3i &sz, bool cudaThread)
{
  if(min(sz) <= 0) { std::cout << "====> ERROR: Field with zero size not allowed.\n"; return false; }
  bool success = true;
  //if(cudaThread)
    {
      std::cout << "========================================================================================\n"
                << "== RESIZING FIELD: " << sz << "\n"
                << "========================================================================================\n";
      //cudaDeviceSynchronize();
      for(int i = 0; i < STATE_BUFFER_SIZE; i++)
        {
          FluidField<CFT> *f = reinterpret_cast<FluidField<CFT>*>(mStates[i]);
          if(!f->create(mParams.cp.fs))
            { std::cout << "Field creation failed! Invalid state(" << i << " / " << STATE_BUFFER_SIZE << ").\n"; success = false; break; }
          f->mat.clear(); // NOTE: clearing material field sets it to vacuum
        }
      cudaDeviceSynchronize();
      if(!mTempState->create(mParams.cp.fs)) { std::cout << "Field creation failed! Invalid temp state.\n"; success = false; }
      if(!mInputV->create   (mParams.cp.fs)) { std::cout << "Field creation failed! Invalid signal input field (v).\n";   success = false; }
      if(!mInputP->create   (mParams.cp.fs)) { std::cout << "Field creation failed! Invalid signal input field (p).\n";   success = false; }
      if(!mInputQn->create  (mParams.cp.fs)) { std::cout << "Field creation failed! Invalid signal input field (Qn).\n";  success = false; }
      if(!mInputQp->create  (mParams.cp.fs)) { std::cout << "Field creation failed! Invalid signal input field (Qp).\n";  success = false; }
      if(!mInputQnv->create (mParams.cp.fs)) { std::cout << "Field creation failed! Invalid signal input field (Qnv).\n"; success = false; }
      if(!mInputQpv->create (mParams.cp.fs)) { std::cout << "Field creation failed! Invalid signal input field (Qpv).\n"; success = false; }
      if(!mInputE->create   (mParams.cp.fs)) { std::cout << "Field creation failed! Invalid signal input field (E).\n";   success = false; }
      if(!mInputB->create   (mParams.cp.fs)) { std::cout << "Field creation failed! Invalid signal input field (B).\n";   success = false; }
      cudaDeviceSynchronize();

      // for(auto p : mSigPens) { p->depth = mParams.cp.fs.z/2; }
      // for(auto p : mMatPens) { p->depth = mParams.cp.fs.z/2; }
      activeSigPen()->depth = mParams.cp.fs.z/2;
      activeMatPen()->depth = mParams.cp.fs.z/2;
      mDisplayUI->rp->zRange = int2{0, mParams.cp.fs.z-1};
      if(success) { resetSim(true); }
      mNewResize = Vec3i(0,0,0);
    }
  // else { mNewResize = sz; }
  return success;
}

// TODO: improve expression variable framework
template<typename T> std::vector<std::string> getVarNames() { return {"px", "py", "pz", "sx", "sy", "sz", "r", "t"}; }
template<> std::vector<std::string> getVarNames<CFT> ()     { return {"px", "py", "pz", "sx", "sy", "sz", "r", "t"}; }
template<> std::vector<std::string> getVarNames<CFV3>()     { return {"p", "s", "r", "n", "t"}; }

void SimWindow::resetSignals(bool cudaThread)
{
  if(cudaThread) // only call from main CUDA thread
    {
      std::cout << "==== SIGNAL RESET\n";
      // recreate expressions (host)
      if(mFieldUI->initQn.hExpr) { delete mFieldUI->initQn.hExpr; }
      mFieldUI->initQn.hExpr  = toExpression<CFT> ((mFieldUI->initQn.active  ? mFieldUI->initQn.str  : "0"), false);
      if(mFieldUI->initQp.hExpr) { delete mFieldUI->initQp.hExpr; }
      mFieldUI->initQp.hExpr  = toExpression<CFT> ((mFieldUI->initQp.active  ? mFieldUI->initQp.str  : "0"), false);
      if(mFieldUI->initQnv.hExpr) { delete mFieldUI->initQnv.hExpr; }
      mFieldUI->initQnv.hExpr = toExpression<CFV3>((mFieldUI->initQnv.active ? mFieldUI->initQnv.str : "0"), false);
      if(mFieldUI->initQpv.hExpr) { delete mFieldUI->initQpv.hExpr; }
      mFieldUI->initQpv.hExpr = toExpression<CFV3>((mFieldUI->initQpv.active ? mFieldUI->initQpv.str : "0"), false);
      if(mFieldUI->initE.hExpr)  { delete mFieldUI->initE.hExpr; }
      mFieldUI->initE.hExpr   = toExpression<CFV3>((mFieldUI->initE.active   ? mFieldUI->initE.str   : "0"), false);
      if(mFieldUI->initB.hExpr)  { delete mFieldUI->initB.hExpr; }
      mFieldUI->initB.hExpr   = toExpression<CFV3>((mFieldUI->initB.active   ? mFieldUI->initB.str   : "0"), false);

      if(mParams.verbose)
        {
          std::cout << "====== Q-:  " << mFieldUI->initQn.hExpr->toString(true)  << "\n";
          std::cout << "====== Q+:  " << mFieldUI->initQp.hExpr->toString(true)  << "\n";
          std::cout << "====== Qv-: " << mFieldUI->initQnv.hExpr->toString(true) << "\n";
          std::cout << "====== Qv+: " << mFieldUI->initQpv.hExpr->toString(true) << "\n";
          std::cout << "====== E:   " << mFieldUI->initE.hExpr->toString(true)   << "\n";
          std::cout << "====== B:   " << mFieldUI->initB.hExpr->toString(true)   << "\n";
        }

      // create or update expressions (device)
      if(!mFieldUI->initQn.dExpr)  { mFieldUI->initQn.dExpr  = toCudaExpression<CFT >(mFieldUI->initQn.hExpr,  getVarNames<CFT >()); }
      if(!mFieldUI->initQp.dExpr)  { mFieldUI->initQp.dExpr  = toCudaExpression<CFT >(mFieldUI->initQp.hExpr,  getVarNames<CFT >()); }
      if(!mFieldUI->initQnv.dExpr) { mFieldUI->initQnv.dExpr = toCudaExpression<CFV3>(mFieldUI->initQnv.hExpr, getVarNames<CFV3>()); }
      if(!mFieldUI->initQpv.dExpr) { mFieldUI->initQpv.dExpr = toCudaExpression<CFV3>(mFieldUI->initQpv.hExpr, getVarNames<CFV3>()); }
      if(!mFieldUI->initE.dExpr)   { mFieldUI->initE.dExpr   = toCudaExpression<CFV3>(mFieldUI->initE.hExpr,   getVarNames<CFV3>()); }
      if(!mFieldUI->initB.dExpr)   { mFieldUI->initB.dExpr   = toCudaExpression<CFV3>(mFieldUI->initB.hExpr,   getVarNames<CFV3>()); }

      // fill all states
      for(int i = 0; i < mStates.size(); i++)
        {
          FluidField<CFT> *f = reinterpret_cast<FluidField<CFT>*>(mStates[mStates.size()-1-i]);
          fillField<CFT> (f->Qn, mFieldUI->initQn.dExpr); fillField<CFT >(f->Qp, mFieldUI->initQp.dExpr);
          fillField<CFV3>(f->Qnv, mFieldUI->initQnv.dExpr);
          fillField<CFV3>(f->Qpv, mFieldUI->initQpv.dExpr);
          fillField<CFV3>(f->E,  mFieldUI->initE.dExpr);  fillField<CFV3>(f->B,  mFieldUI->initB.dExpr);
          f->divE.clear(); f->divB.clear();
        }
      mTempState->Qn.clear(); mTempState->Qp.clear(); mTempState->Qnv.clear(); mTempState->Qpv.clear(); mTempState->E.clear();  mTempState->B.clear();
      mInputQn->clear(); mInputQp->clear(); mInputQnv->clear(); mInputQpv->clear(); mInputE->clear(); mInputB->clear(); // clear remaining inputs
      mNeedResetSignals = false;
    }
  else { mNeedResetSignals = true; }
}

void SimWindow::resetMaterials(bool cudaThread)
{
  if(cudaThread) // only call from main CUDA thread
    {
      std::cout << "==== MATERIAL RESET\n";
      // recreate expressions (host)
      if(mFieldUI->initEp.hExpr) { delete mFieldUI->initEp.hExpr; }
      mFieldUI->initEp.hExpr  = toExpression<CFT>((mFieldUI->initEp.active ?mFieldUI->initEp.str  : "1"), false);
      if(mFieldUI->initMu.hExpr) { delete mFieldUI->initMu.hExpr; }
      mFieldUI->initMu.hExpr  = toExpression<CFT>((mFieldUI->initMu.active ?mFieldUI->initMu.str  : "1"), false);
      if(mFieldUI->initSig.hExpr){ delete mFieldUI->initSig.hExpr; }
      mFieldUI->initSig.hExpr = toExpression<CFT>((mFieldUI->initSig.active?mFieldUI->initSig.str : "0"), false);
      if(mParams.verbose)
        {
          std::cout << "====== ε:  " << mFieldUI->initEp.hExpr->toString(true)  << "\n";
          std::cout << "====== μ:  " << mFieldUI->initMu.hExpr->toString(true)  << "\n";
          std::cout << "====== σ:  " << mFieldUI->initSig.hExpr->toString(true) << "\n";
        }

      // create or update expressions (device)
      if(!mFieldUI->initEp.dExpr)  { mFieldUI->initEp.dExpr  = toCudaExpression<CFT>(mFieldUI->initEp.hExpr,  getVarNames<CFT>()); }
      if(!mFieldUI->initMu.dExpr)  { mFieldUI->initMu.dExpr  = toCudaExpression<CFT>(mFieldUI->initMu.hExpr,  getVarNames<CFT>()); }
      if(!mFieldUI->initSig.dExpr) { mFieldUI->initSig.dExpr = toCudaExpression<CFT>(mFieldUI->initSig.hExpr, getVarNames<CFT>()); }

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
      mTempState->mat.clear();
      mNeedResetMaterials = false;
    }
  else { mNeedResetMaterials = true; }
}


void SimWindow::resetFluid(bool cudaThread)
{
  if(cudaThread) // only call from main CUDA thread
    {
      std::cout << "==== FLUID RESET\n";
      // recreate expressions (host)
      if(mFieldUI->initV.hExpr) { delete mFieldUI->initV.hExpr; }
      mFieldUI->initV.hExpr = toExpression<CFV3>((mFieldUI->initV.active?mFieldUI->initV.str:"0"), false);
      if(mFieldUI->initP.hExpr) { delete mFieldUI->initP.hExpr; }
      mFieldUI->initP.hExpr = toExpression<CFT >((mFieldUI->initP.active?mFieldUI->initP.str:"0"), false);
      if(mParams.verbose)
        {
          std::cout << "====== V:  " << mFieldUI->initV.hExpr->toString(true) << "\n";
          std::cout << "====== P:  " << mFieldUI->initP.hExpr->toString(true) << "\n";
        }

      // create or update expressions (device)
      if(!mFieldUI->initV.dExpr) { mFieldUI->initV.dExpr = toCudaExpression<CFV3>(mFieldUI->initV.hExpr, getVarNames<CFV3>()); }
      if(!mFieldUI->initP.dExpr) { mFieldUI->initP.dExpr = toCudaExpression<CFT >(mFieldUI->initP.hExpr, getVarNames<CFT >()); }
      
      // fill all states
      for(int i = 0; i < mStates.size(); i++)
        {
          FluidField<CFT> *f = reinterpret_cast<FluidField<CFT>*>(mStates[mStates.size()-1-i]);
          fillField<CFV3>(f->v,  mFieldUI->initV.dExpr); fillField<CFT> (f->p,  mFieldUI->initP.dExpr);
        }
      mTempState->v.clear(); mTempState->p.clear();
      mInputV->clear(); mInputP->clear(); // clear remaining inputs
      mNeedResetFluid = false;
    }
  else { mNeedResetFluid = true; }
}


void SimWindow::resetSim(bool cudaThread)
{
  if(cudaThread) // only call from main CUDA thread
    {
      std::cout << "== SIMULATION RESET\n";
      mKeyFrameUI->reset();
      
      if(mFrameWriter) { mFrameWriter->flush(); } // wait for all frames to be written
      
      mInfo.t = 0.0f; mInfo.frame = 0; mInfo.uStep = 0;
      resetSignals(cudaThread);
      resetMaterials(cudaThread);
      resetFluid(cudaThread);

      cudaRender(mParams.cp);
      cudaDeviceSynchronize();
      mNeedResetSignals = false; mNeedResetFluid = false; mNeedResetMaterials = false;  mNeedResetSim = false;
    }
  else { mNeedResetSim = true; }
}

void SimWindow::resetViews(bool cudaThread)
{
  if(!mParams.op.lockViews && !mForcePause)
    {
      if(cudaThread) // only call from main CUDA thread
        {
          CFT    pad = SIM_VIEW_RESET_INTERNAL_PADDING;
          Vec2f  fs  = Vec2f(mParams.cp.fs.x, mParams.cp.fs.y);
          Vec2f  fp  = Vec2f(mParams.cp.fp.x, mParams.cp.fp.y);
          Vec2f  fsPadded = fs * (1.0 + 2.0*pad);

          float fAspect = mParams.cp.fs.x/(float)mParams.cp.fs.y;
          Vec2f aspect2D = Vec2f(mEMView.r.aspect()/fAspect, 1.0); if(aspect2D.x < 1.0) { aspect2D.y = 1.0/aspect2D.x; aspect2D.x = 1.0; }
          Vec2f aspect3D = Vec2f(m3DView.r.aspect()/fAspect, 1.0); if(aspect3D.x < 1.0) { aspect3D.y = 1.0/aspect3D.x; aspect3D.x = 1.0; }

          // reset 2D sim view
          Vec2f fp2D = -(fsPadded * pad);
          Vec2f fs2D = fsPadded*aspect2D * mUnits.dL;
          mSimView2D = Rect2f(fp2D, fp2D + fs2D);
          Vec2f offset2D = fs/2.0*mUnits.dL - mSimView2D.center();
          mSimView2D.move(offset2D);

          // calculate 3D camera Z offset
          float S = 2.0*tan((mCamera.fov/2.0)*M_PI/180.0);
          Vec2f fs3D = fsPadded*aspect3D * mUnits.dL;
          float zOffset = ((max(to_cuda(aspect3D)) < fAspect) ? fs3D.x : fs3D.y) / S;
          // reset 3D camera
          mCamera.aspect = m3DView.r.aspect();     // aspect
          mCamera.pos    = (CFV3{fs.x*mUnits.dL/2, // camera position (centered over field in +Z direction)
                                 fs.y*mUnits.dL/2,
                                 zOffset + mParams.cp.fs.z*mUnits.dL });
          mCamera.right  = CFV3{1.0f,  0.0f,  0.0f}; // camera x direction
          mCamera.up     = CFV3{0.0f,  1.0f,  0.0f}; // camera y direction
          mCamera.dir    = CFV3{0.0f,  0.0f, -1.0f}; // camera z direction

          mKeyFrameUI->view2D() = mSimView2D;
          mKeyFrameUI->view3D() = mCamera.desc;
          
          mNeedResetViews = false;
        }
      else { mNeedResetViews = true; }
    }
}




void SimWindow::togglePause()
{
  mFieldUI->running = !mFieldUI->running;
  std::cout << "== " << (mFieldUI->running ? "STARTED" : "STOPPED") << " SIMULATION.\n";
}
void SimWindow::toggleDebug()             { mParams.debug =   !mParams.debug;   }
void SimWindow::toggleVerbose()           { mParams.verbose = !mParams.verbose; }
void SimWindow::toggleKeyBindings()       { mKeyManager->togglePopup(); }
void SimWindow::singleStepField(CFT mult) { if(!mFieldUI->running) { mSingleStepMult = mult; } }
void SimWindow::toggleImGuiDemo()         { mImGuiDemo = !mImGuiDemo; mParams.op.lockViews = (mImGuiDemo || mFontDemo);  }
void SimWindow::toggleFontDemo()          { mFontDemo  = !mFontDemo;  mParams.op.lockViews = (mFontDemo  || mImGuiDemo); }

SignalPen<CFT>*   SimWindow::activeSigPen() { return mSigPenBar->selected() >= 0 ? mSigPens[mSigPenBar->selected()] : nullptr; }
MaterialPen<CFT>* SimWindow::activeMatPen() { return mMatPenBar->selected() >= 0 ? mMatPens[mMatPenBar->selected()] : nullptr; }

void SimWindow::addSigPen(SignalPen<CFT> *pen)
{
  int i = mSigPens.size();
  std::string id = "sp" + std::to_string(i);
  mSigPenBar->add(ToolButton(id, "",
                             [](int i, ImDrawList *drawList, const Vec2f &p0, const Vec2f &p1)
                             {
                               drawList->AddCircle((p0+p1)/2.0f, 0.6f*(p1.x-p0.x)/2.0f, ImColor(Vec4f(0,0,1,1)), 6, 3.0f);
                             }));
  *mSigPens[i] = *pen;
}

void SimWindow::addMatPen(MaterialPen<CFT> *pen)
{
  int i = mMatPens.size();
  std::string id = "mp" + std::to_string(i);
  mMatPenBar->add(ToolButton(id, "",
                             [](int i, ImDrawList *drawList, const Vec2f &p0, const Vec2f &p1)
                             {
                               drawList->AddCircle((p0+p1)/2.0f, 0.6f*(p1.x-p0.x)/2.0f, ImColor(Vec4f(0,0,1,1)), 6, 3.0f);
                             }));
  *mMatPens[i] = *pen;
}


void SimWindow::update()
{
  ImGuiIO &io = ImGui::GetIO();
  
  mUpdateFps.update();
  auto t = CLOCK_T::now();
  bool newFramePhysics = mPhysicsFps.update(mParams.limitPhysicsFps ? mParams.maxPhysicsFps : -1.0f, t);
  bool newFrameRender  = mRenderFps.update(mParams.limitRenderFps ? mParams.maxRenderFps : -1.0f, t);

  mParams.cp.t = mInfo.t; cpCopy = mParams.cp; cpCopy.u = mUnits;
  if(mParams.cp.fs.x > 0 && mParams.cp.fs.y > 0 && mParams.cp.fs.z > 0)
    {
      if(newFramePhysics)
        {
          bool singleStep = false;
          if(mSingleStepMult != 0.0f)
            {
              cpCopy.u.dt *= mSingleStepMult;
              mSingleStepMult = 0.0f;
              singleStep = true;
            }

          if(mNeedResetSim)           { resetSim(true); }
          else
            {
              if(mNewResize > 0) { resizeFields(mNewResize, true); }
              if(mNeedResetSignals)   { resetSignals(true); }
              if(mNeedResetMaterials) { resetMaterials(true); }
              if(mNeedResetFluid)     { resetFluid(true); }
            }
          if(mNeedResetViews)         { resetViews(true); }

          FluidField<CFT> *src  = reinterpret_cast<FluidField<CFT>*>(mStates.back());  // previous field state
          FluidField<CFT> *dst  = reinterpret_cast<FluidField<CFT>*>(mStates.front()); // oldest state (recycle)
          FluidField<CFT> *temp = reinterpret_cast<FluidField<CFT>*>(mTempState);      // temp intermediate state

          // apply external forces from user (TODO: handle input separately)
          CFV3 mposSim = CFV3{NAN, NAN, NAN};
          if     (mEMView.hovered)  { mposSim = to_cuda(mEMView.mposSim);  }
          else if(mMatView.hovered) { mposSim = to_cuda(mMatView.mposSim); }
          else if(m3DView.hovered)  { mposSim = to_cuda(m3DView.mposSim);  }
          float  cs = mUnits.dL;
          CFV3 fs = CFV3{(float)mParams.cp.fs.x, (float)mParams.cp.fs.y, (float)mParams.cp.fs.z};
          CFV3 mpfi = (mposSim) / cs;

          SignalPen<CFT>   *sigPen = activeSigPen();
          MaterialPen<CFT> *matPen = activeMatPen();
          
          // draw signal
          mParams.rp.sigPenHighlight = false;
          CFV3 mposLast = mSigMPos;
          mSigMPos = CFV3{NAN, NAN, NAN};
          bool active = false;
          if(!mParams.op.lockViews && !mForcePause && io.KeyCtrl)
            {
              bool hover = m3DView.hovered || mEMView.hovered || mMatView.hovered;
              bool apply = false;
              CFV3 p = CFV3{NAN, NAN, NAN};
              if(m3DView.hovered)
                {
                  p = to_cuda(m3DView.mposSim - m3DView.mposFace*sigPen->depth);
                  apply = (m3DView.clickBtns(MOUSEBTN_LEFT) && m3DView.clickMods(GLFW_MOD_CONTROL));
                }
              if(mEMView.hovered || mMatView.hovered)
                {
                  mpfi.z = mParams.cp.fs.z - 1 - sigPen->depth; // Z depth relative to top visible layer
                  p = CFV3{mpfi.x, mpfi.y, mpfi.z};
                  apply = (( mEMView.clickBtns(MOUSEBTN_LEFT) && (sigPen->active || mEMView.clickMods(GLFW_MOD_CONTROL))) ||
                           (mMatView.clickBtns(MOUSEBTN_LEFT) && (sigPen->active || mMatView.clickMods(GLFW_MOD_CONTROL))));
                }

              if(hover)
                {
                  mSigMPos = p;
                  sigPen->mouseSpeed = length(mSigMPos - mposLast);
                  mParams.rp.penPos = p;
                  mParams.rp.sigPenHighlight = true;
                  mParams.rp.sigPen = *sigPen;

                  if(apply && !isnan(mSigMPos))
                    { // draw signal to intermediate E/B fields (needs to be blended to avoid peristent blobs)
                      active = true;
                      if(sigPen->startTime <= 0.0) // set signal start time
                        { sigPen->startTime = cpCopy.t; }

                      if((mFieldUI->running || singleStep) && !mForcePause && mFieldUI->inputDecay)
                        { // add to intermediary source fields
                          addSignal(p, *mInputV, *mInputP, *mInputQn, *mInputQp, *mInputQnv, *mInputQpv, *mInputE, *mInputB,
                                    *sigPen, cpCopy, cpCopy.u.dt);
                        }
                      else
                        {
                          addSignal(p, *mInputV, *mInputP, *mInputQn, *mInputQp, *mInputQnv, *mInputQpv, *mInputE, *mInputB,
                                    *sigPen, cpCopy, cpCopy.u.dt);
                          addSignal(*mInputV,   src->v,   cpCopy, cpCopy.u.dt);
                          addSignal(*mInputP,   src->p,   cpCopy, cpCopy.u.dt);
                          addSignal(*mInputQn,  src->Qn,  cpCopy, cpCopy.u.dt);
                          addSignal(*mInputQp,  src->Qp,  cpCopy, cpCopy.u.dt);
                          addSignal(*mInputQnv, src->Qnv, cpCopy, cpCopy.u.dt);
                          addSignal(*mInputQpv, src->Qpv, cpCopy, cpCopy.u.dt);
                          addSignal(*mInputE,   src->E,   cpCopy, cpCopy.u.dt);
                          addSignal(*mInputB,   src->B,   cpCopy, cpCopy.u.dt);
                          if(mFieldUI->inputDecay)
                            {
                              decaySignal(*mInputE,   cpCopy); decaySignal(*mInputB,   cpCopy);
                              decaySignal(*mInputQn,  cpCopy); decaySignal(*mInputQp,  cpCopy);
                              decaySignal(*mInputQnv, cpCopy); decaySignal(*mInputQpv, cpCopy);
                              decaySignal(*mInputV,   cpCopy); decaySignal(*mInputP,   cpCopy);
                            }
                          else
                            {
                              mInputV->clear();  mInputP->clear();
                              mInputQn->clear(); mInputQp->clear(); mInputQnv->clear(); mInputQpv->clear();
                              mInputE->clear();  mInputB->clear();
                            }
                        }
                      mNewSimFrame = true; // new frame even if paused
                    }
                }
            }
          if(!active) { sigPen->startTime = -1.0; } // reset start time

          // add keyframe signals to field
          std::vector<SignalSource> sources = mKeyFrameUI->sources(mKeyFrameUI->cursor());//, mKeyFrameUI->cursor()+cpCopy.u.dt);
          for(const auto &s : sources)
            {
              if((mFieldUI->running || singleStep) && !mForcePause && mFieldUI->inputDecay)
                { // add to intermediary source fields
                  addSignal(s.pos, *mInputV, *mInputP, *mInputQn, *mInputQp, *mInputQnv, *mInputQpv, *mInputE, *mInputB,
                            s.pen, cpCopy, cpCopy.u.dt);
                }
              else
                {
                  addSignal(s.pos, *mInputV, *mInputP, *mInputQn, *mInputQp, *mInputQnv, *mInputQpv, *mInputE, *mInputB,
                            s.pen, cpCopy, cpCopy.u.dt);
                  addSignal(*mInputV,   src->v,   cpCopy, cpCopy.u.dt);
                  addSignal(*mInputP,   src->p,   cpCopy, cpCopy.u.dt);
                  addSignal(*mInputQn,  src->Qn,  cpCopy, cpCopy.u.dt);
                  addSignal(*mInputQp,  src->Qp,  cpCopy, cpCopy.u.dt);
                  addSignal(*mInputQnv, src->Qnv, cpCopy, cpCopy.u.dt);
                  addSignal(*mInputQpv, src->Qpv, cpCopy, cpCopy.u.dt);
                  addSignal(*mInputE,   src->E,   cpCopy, cpCopy.u.dt);
                  addSignal(*mInputB,   src->B,   cpCopy, cpCopy.u.dt);
                  if(mFieldUI->inputDecay)
                    {
                      decaySignal(*mInputE,   cpCopy); decaySignal(*mInputB,   cpCopy);
                      decaySignal(*mInputQn,  cpCopy); decaySignal(*mInputQp,  cpCopy);
                      decaySignal(*mInputQnv, cpCopy); decaySignal(*mInputQpv, cpCopy);
                      decaySignal(*mInputV,   cpCopy); decaySignal(*mInputP,   cpCopy);
                    }
                  else
                    {
                      mInputV->clear();  mInputP->clear();
                      mInputQn->clear(); mInputQp->clear(); mInputQnv->clear(); mInputQpv->clear();
                      mInputE->clear();  mInputB->clear();
                    }
                }
            }

          // add material
          mParams.rp.matPenHighlight = false;
          mMatMPos = CFV3{NAN, NAN, NAN};
          if(!mParams.op.lockViews && !mForcePause && io.KeyAlt)
            {
              bool hover = m3DView.hovered || mEMView.hovered || mMatView.hovered;
              CFV3 p = CFV3{NAN, NAN, NAN};
              bool apply = false;
              if(m3DView.hovered)
                {
                  CFV3 fp = to_cuda(m3DView.mposSim);
                  p = to_cuda(m3DView.mposSim - m3DView.mposFace*matPen->depth);
                  apply = (m3DView.clickBtns(MOUSEBTN_LEFT) && m3DView.clickMods(GLFW_MOD_ALT));
                }
              if(mEMView.hovered || mMatView.hovered)
                {
                  mpfi.z = mParams.cp.fs.z-1-matPen->depth; // Z depth relative to top visible layer
                  p = CFV3{mpfi.x, mpfi.y, mpfi.z};
                  apply = ((mEMView.clickBtns(MOUSEBTN_LEFT)  && mEMView.clickMods(GLFW_MOD_ALT)) ||
                           (mMatView.clickBtns(MOUSEBTN_LEFT) && mMatView.clickMods(GLFW_MOD_ALT)));
                }
              if(hover)
                {
                  mMatMPos = p;
                  mParams.rp.penPos = p;
                  mParams.rp.matPenHighlight = true;
                  mParams.rp.matPen = *matPen;
                  if(apply && !isnan(mMatMPos)) { addMaterial(p, *src, *matPen, cpCopy); mNewSimFrame = true;  } // new frame even if paused
                }
            }

          // add keyframe materials to field
          std::vector<MaterialPlaced> &placed = mKeyFrameUI->placed();
          for(const auto &m : placed) { addMaterial(m.pos, *src, m.pen, cpCopy); }
          placed.clear();


          if((mFieldUI->running || singleStep) && !mForcePause)
            { // step simulation
              if(!mFieldUI->running) { std::cout << "==== SIM STEP --> dt = " << cpCopy.u.dt << "\n"; }
              if(DESTROY_LAST_STATE) { std::swap(temp, src); } // overwrite source state
              else                   { src->copyTo(*temp); }   // don't overwrite previous state (use temp)

              //// INPUTS
              // (NOTE: remove added sources to avoid persistent lumps building up)
              // fluid input
              addSignal(*mInputV,   temp->v,   cpCopy, cpCopy.u.dt); // add V input signal
              addSignal(*mInputP,   temp->p,   cpCopy, cpCopy.u.dt); // add P input signal
              // EM input
              addSignal(*mInputQn,  temp->Qn,  cpCopy, cpCopy.u.dt); // add Qn input signal
              addSignal(*mInputQp,  temp->Qp,  cpCopy, cpCopy.u.dt); // add Qp input signal
              addSignal(*mInputQnv, temp->Qnv, cpCopy, cpCopy.u.dt); // add Qnv input signal
              addSignal(*mInputQpv, temp->Qpv, cpCopy, cpCopy.u.dt); // add Qpv input signal
              addSignal(*mInputE,   temp->E,   cpCopy, cpCopy.u.dt); // add E input signal
              addSignal(*mInputB,   temp->B,   cpCopy, cpCopy.u.dt); // add B input signal

              //// FLUID STEP (V/P)
              if(mFieldUI->updateFluid) // V, P
                {
                  if(mFieldUI->updateP1)     { fluidPressure (*temp, *dst, cpCopy, cpCopy.pIter1); std::swap(temp, dst); } // PRESSURE SOLVE (1)
                  if(mFieldUI->updateAdvect) { fluidAdvect   (*temp, *dst, cpCopy);                std::swap(temp, dst); } // ADVECT
                  if(mFieldUI->updateVisc)   { fluidViscosity(*temp, *dst, cpCopy, cpCopy.vIter);  std::swap(temp, dst); } // VISCOSITY SOLVE
                  if(mFieldUI->applyGravity) { fluidExternalForces(*temp, cpCopy); }                                       // EXTERNAL FORCES (in-place)
                  if(mFieldUI->updateP2)     { fluidPressure (*temp, *dst, cpCopy, cpCopy.pIter2); std::swap(temp, dst); } // PRESSURE SOLVE (2)
                }
              //// EM STEP (Q/Qv/E/B)
              if(mFieldUI->updateEM)
                {
                  if(mFieldUI->updateCoulomb) { updateCoulomb(*temp, *dst, cpCopy);  std::swap(temp, dst); } // Coulomb forces
                  
                  if(mFieldUI->updateQ)       { updateCharge(*temp, *dst, cpCopy);   std::swap(temp, dst); } // Q
                  if(mFieldUI->updateE)       { updateElectric(*temp, *dst, cpCopy); std::swap(temp, dst); } // E
                  //cpCopy.t += cpCopy.u.dt/2.0f; // increment first half time step
                  if(mFieldUI->updateB)       { updateMagnetic(*temp, *dst, cpCopy); std::swap(temp, dst); } // B
                  //cpCopy.t += cpCopy.u.dt/2.0f; // increment second half time step
                }
              cpCopy.t += cpCopy.u.dt;
  
              // decay input signals (blend over time)
              if(mFieldUI->inputDecay) // ?
              {
                addSignal(*mInputE,  temp->E,  cpCopy, -cpCopy.u.dt); // remove added E  input signal
                addSignal(*mInputB,  temp->B,  cpCopy, -cpCopy.u.dt); // remove added B  input signal
              }
              
              if(mFieldUI->inputDecay)
                {
                  decaySignal(*mInputE,   cpCopy); decaySignal(*mInputB,   cpCopy);
                  decaySignal(*mInputQn,  cpCopy); decaySignal(*mInputQp,  cpCopy);
                  decaySignal(*mInputQnv, cpCopy); decaySignal(*mInputQpv, cpCopy);
                  decaySignal(*mInputV,   cpCopy); decaySignal(*mInputP,   cpCopy);
                }
              else
               {
                 mInputV->clear();  mInputP->clear();
                 mInputQn->clear(); mInputQp->clear(); mInputQnv->clear(); mInputQpv->clear();
                 mInputE->clear();  mInputB->clear();
               }
              
              std::swap(temp, dst); // (un-)swap final result back into dst
              if(!DESTROY_LAST_STATE) { std::swap(mTempState, temp); } // use other state as new temp (pointer changes if number of steps is odd)}
              else                    { std::swap((FluidField<CFT>*&)mStates.back(), temp); }
              mStates.pop_front(); mStates.push_back(dst);

              // increment time/frame info
              mInfo.t += mUnits.dt;
              mInfo.uStep++;
              if(mInfo.uStep >= mParams.uSteps) { mInfo.frame++; mInfo.uStep = 0; mNewSimFrame = true; }
              mKeyFrameUI->nextFrame(cpCopy.u.dt);
            }
          else
            {
              mInputV->clear();  mInputP->clear(); mInputQn->clear(); mInputQp->clear();
              mInputQnv->clear(); mInputQpv->clear(); mInputE->clear(); mInputB->clear();
            }
        }
      if(newFrameRender) { cudaRender(cpCopy); }
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
  if(mDisplayUI->show3DView)
    {
      if(mParams.op.active && mNewSimFrame) // output file rendering
        { // render file output separately (likely different aspect ratio)
          mGlCamera = mCamera;
          mGlCamera.aspect = m3DGlView.r.aspect();
          m3DGlTex.clear(); raytraceFieldEM(*renderSrc, m3DGlTex, mGlCamera, mParams.rp, cp);
        }
      m3DTex.clear(); raytraceFieldEM(*renderSrc, m3DTex, mCamera, mParams.rp, cp);
    }
  if(mNewSimFrame) { mNewFrameOut = true; }
  mNewSimFrame = false;
}




////////////////////////////////////////////////
//// INPUT HANDLING ////////////////////////////
////////////////////////////////////////////////


// handle input for 2D views
void SimWindow::handleInput2D(ScreenView<CFT> &view, Rect<CFT> &simView)
{
  ImGuiStyle &style = ImGui::GetStyle();
  ImGuiIO    &io    = ImGui::GetIO();

  Vec2f mpos    = ImGui::GetMousePos();
  Vec2f mposSim = view.screenToSim2D(mpos, simView);
  view.hovered  = simView.contains(mposSim);
  if(view.hovered) { view.mposSim = CFV3{(float)mposSim.x-mFieldUI->cp->fp.x, (float)mposSim.y-mFieldUI->cp->fp.y, (float)mDisplayUI->rp->zRange.y}; }
  else             { view.mposSim = CFV3{NAN, NAN, NAN}; }

  bool newClick = false;
  if(view.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left))   { view.clicked |=  MOUSEBTN_LEFT;   newClick = true; }
  else if(ImGui::IsMouseReleased(ImGuiMouseButton_Left))             { view.clicked &= ~MOUSEBTN_LEFT;   view.mods = 0;   }
  if(view.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right))  { view.clicked |=  MOUSEBTN_RIGHT;  newClick = true; }
  else if(ImGui::IsMouseReleased(ImGuiMouseButton_Right))            { view.clicked &= ~MOUSEBTN_RIGHT;  view.mods = 0;   }
  if(view.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) { view.clicked |=  MOUSEBTN_MIDDLE; newClick = true; }
  else if(ImGui::IsMouseReleased(ImGuiMouseButton_Middle))           { view.clicked &= ~MOUSEBTN_MIDDLE; view.mods = 0;   }

  if(mParams.op.lockViews || mForcePause) { return; } // ignore user input for rendering
  if(newClick) // new mouse click
    {
      if(view.clicked == MOUSEBTN_LEFT) { view.clickPos = mpos; } // set new position only on first button click
      view.mods = ((io.KeyShift ? GLFW_MOD_SHIFT   : 0) |
                   (io.KeyCtrl  ? GLFW_MOD_CONTROL : 0) |
                   (io.KeyAlt   ? GLFW_MOD_ALT     : 0));
    }

  //// view manipulation ////
  bool viewChanged = false;

  // left/middle drag --> pan
  ImGuiMouseButton btn = (//ImGui::IsMouseDragging(ImGuiMouseButton_Left) ? ImGuiMouseButton_Left :
                          (ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ? ImGuiMouseButton_Middle : -1));
  if((//(view.clickBtns(MOUSEBTN_LEFT)   && btn == ImGuiMouseButton_Left && !view.clickMods(GLFW_MOD_CONTROL | GLFW_MOD_ALT)) ||
      (view.clickBtns(MOUSEBTN_MIDDLE) && btn == ImGuiMouseButton_Middle)))
    {
      Vec2f dmp = ImGui::GetMouseDragDelta(btn); ImGui::ResetMouseDragDelta(btn);
      dmp.x *= -1.0f;
      simView.move(view.screenToSim2D(dmp, simView, true)); // recenter mouse at same sim position
      viewChanged = true;
    }

  // mouse scroll --> zoom
  if(view.hovered && std::abs(io.MouseWheel) > 0.0f)
    {
      if(io.KeyCtrl && io.KeyShift)     // signal pen depth
        { activeSigPen()->depth += (io.MouseWheel > 0 ? 1 : -1); }
      else if(io.KeyAlt && io.KeyShift) // material pen depth
        { activeMatPen()->depth += (io.MouseWheel > 0 ? 1 : -1); }
      else // zoom camera
        {
          float vel = (io.KeyAlt ? 1.36 : (io.KeyShift ? 1.011 : 1.055)); // scroll velocity
          float scale = (io.MouseWheel > 0.0f ? 1.0/vel : vel);
          simView.scale(scale);
          Vec2f mposSim2 = view.screenToSim2D(mpos, simView, false);
          simView.move(mposSim-mposSim2); // center so mouse doesn't change position
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
      std::vector<FluidField<CFT>*> states;
      unsigned long i0 = 0; unsigned long i1 = 0;
      if(fi.x >= 0 && fi.x < mParams.cp.fs.x && fi.y >= 0 && fi.y < mParams.cp.fs.y)
        {
          i0 = mStates.front()->idx(fi.x, fi.y, zi);
          i1 = i0 + 1UL;
          for(int i = 0; i < mStates.size(); i++)
            {
              FluidField<CFT> *src = reinterpret_cast<FluidField<CFT>*>(mStates[mStates.size()-1-i]);
              if(src) { src->pullData(i0, i1); states.push_back(src); }
            }
          mTempState->pullData(i0, i1);
          states.push_back(mTempState);
        }
      
      Vec2f dataPadding = Vec2f(6.0f, 6.0f);
      ImGui::BeginTooltip();
      {
        ImDrawList *ttDrawList = ImGui::GetWindowDrawList(); // maximum text size per column
        ImGui::Text(" mousePos: <%.3f, %.3f> (index: <%d, %d, %d>)", mposSim.x, mposSim.y, fi.x, fi.y, zi);

        Vec2f tSize = ImGui::CalcTextSize(("T"+std::to_string(mStates.size()-1)).c_str()); // max state label width
        tSize = max(tSize, Vec2f(ImGui::CalcTextSize("(buffer)")));
        Vec2f p0 = ImGui::GetCursorScreenPos();
        if(states.size() > 0) { ImGui::BeginGroup(); }
        for(int i = 0; i < states.size(); i++)
          {
            FluidField<CFT> *d = states[i];
            bool last = (i >= states.size()-1);
            
            if(last) { ImGui::EndGroup(); ImGui::SameLine(); ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos())+Vec2f(10, 0)); }
            ImGui::BeginGroup();
            {
              std::string labelStr = (last) ? "(buffer)" : ("T" + std::to_string(i));
              Vec2f sp0 = Vec2f(ImGui::GetCursorScreenPos()) + Vec2f(tSize.x + dataPadding.x, 0);

              // draw state label
              ImGui::TextUnformatted(labelStr.c_str());
              // draw state data
              ImGui::SameLine();
              ImGui::SetCursorScreenPos(sp0 + dataPadding);
              ImGui::BeginGroup();
              {
                float xpos = ImGui::GetCursorPos().x;
                CFT Q = d->Qp[i0] - d->Qn[i0];
                ImGui::Text("(State Pointer: %ld", (long)(void*)(d));
                ImGui::Text(" v   = < %14s, %14s, %14s >",
                            fAlign(d->v[i0].x,   6).c_str(), fAlign(d->v[i0].y,   9).c_str(), fAlign(d->v[i0].z,   9).c_str());
                ImGui::Text(" p   =   %14s",                 fAlign(d->p[i0],     9).c_str());
                ImGui::Text(" Q   =   %10s %10s | %14s",
                            fAlign(d->Qp[i0],    9).c_str(), fAlign(-d->Qn[i0],   9).c_str(), fAlign(Q,            9).c_str());
                ImGui::Text(" Qv- = < %14s, %14s, %14s >",
                            fAlign(d->Qnv[i0].x, 9).c_str(), fAlign(d->Qnv[i0].y, 9).c_str(), fAlign(d->Qnv[i0].z, 9).c_str());
                ImGui::Text(" Qv+ = < %14s, %14s, %14s >",
                            fAlign(d->Qpv[i0].x, 9).c_str(), fAlign(d->Qpv[i0].y, 9).c_str(), fAlign(d->Qpv[i0].z, 9).c_str());
                ImGui::Text(" E   = < %14s, %14s, %14s >",
                            fAlign(d->E[i0].x,   9).c_str(), fAlign(d->E[i0].y,   9).c_str(), fAlign(d->E[i0].z,   9).c_str());
                ImGui::Text(" B   = < %14s, %14s, %14s >",
                            fAlign(d->B[i0].x,   9).c_str(), fAlign(d->B[i0].y,   9).c_str(), fAlign(d->B[i0].z,   9).c_str());
                ImGui::Spacing(); ImGui::Spacing();
                if(d->mat[i0].vacuum())
                  {
                    ImGui::Text(" Material (vacuum): ep = %10.4f", mUnits.vacuum().permittivity);
                    ImGui::Text("                    mu = %10.4f", mUnits.vacuum().permeability);
                    ImGui::Text("                   sig = %10.4f", mUnits.vacuum().conductivity);
                  }
                else
                  {
                    ImGui::Text(" Material:          ep = %10.4f", d->mat[i0].permittivity);
                    ImGui::Text("                    mu = %10.4f", d->mat[i0].permeability);
                    ImGui::Text("                   sig = %10.4f", d->mat[i0].conductivity);
                  }
              }
              ImGui::EndGroup();
              // draw border
              Vec2f sSize = Vec2f(ImGui::GetItemRectMax()) - ImGui::GetItemRectMin() + 2.0f*dataPadding;
              ImGui::PushClipRect(sp0, sp0+sSize, false);
              Vec4f color;
              if(last)
                {  color = Vec4f(0.5f, 0.5f, 0.8f, 0.5f); }
              else
                {
                  float alpha = last ? 0.0f : (mStates.size() == 1 ? 1.0f : (mStates.size()-1-i) / (float)(mStates.size()-1));
                  color = Vec4f(0.4f+(0.6f*alpha), 0.3f, 0.3f, 1.0f);  // fading color for older states
                }
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
  if(viewChanged)
    {
      mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_VIEW2D>(mKeyFrameUI->cursor(), simView)); 
      mNewSimFrame = true; // new frame on view change
    }
}

// handle input for 3D views
void SimWindow::handleInput3D(ScreenView<CFT> &view, Camera<CFT> &camera)
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
      ray = camera.castRay(to_cuda(Vec2f((mpos - view.r.p1)/view.r.size()))); //, float2{aspect.x, aspect.y});
      Vec2f tp = cubeIntersectHost(Vec3f(mParams.cp.fp*mUnits.dL), Vec3f(fSize), ray);

      if(tp.x > 0.0) // tmin
        {
          Vec3f wp = ray.pos + ray.dir*(tp.x+0.00001); // world-space pos of field outer intersection
          Vec3f fp = (wp - mParams.cp.fp*mUnits.dL) / fSize * fs;
          view.mposSim  = fp;
          view.mposFace = cubeIntersectFace(Vec3f(mParams.cp.fp*mUnits.dL), Vec3f(fSize), ray);
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

  if(mParams.op.lockViews || mForcePause) { return; } // ignore user input for rendering
  if(newClick) // new mouse click
    {
      if(view.clicked == MOUSEBTN_LEFT) { view.clickPos = mpos; } // set new position only on first button click
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
  CFT S = (2.0*tan(camera.fov/2.0*M_PI/180.0));

  ImGuiMouseButton btn = (ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ? ImGuiMouseButton_Middle : -1);
  if((btn == ImGuiMouseButton_Middle && view.clickBtns(MOUSEBTN_MIDDLE)))
    { // left/middle drag --> pan camera
      Vec2f dmp = ImGui::GetMouseDragDelta(btn); ImGui::ResetMouseDragDelta(btn);
      dmp.x *= -1.0f;
      CFV3 fpos = CFV3{mParams.cp.fp.x, mParams.cp.fp.y, mParams.cp.fp.z};
      CFV3 cpos = CFV3{camera.pos.x, camera.pos.y, camera.pos.z};
      CFV3 fsize = CFV3{(float)mParams.cp.fs.x, (float)mParams.cp.fs.y, (float)mParams.cp.fs.z};
      CFT lengthMult = (length(cpos-fpos) +
                        length(cpos - (fpos + fsize)) +
                        length(cpos-(fpos + fsize/2.0)) +
                        length(cpos - (fpos + fsize/2.0)) +
                        length(cpos-CFV3{fpos.x, fpos.y + fsize.y/2.0f, fpos.z}) +
                        length(cpos-CFV3{fpos.x + fsize.x/2.0f, fpos.y, fpos.z} ))/6.0f;

      camera.pos += (camera.right*dmp.x/viewSize.x*aspect.x + camera.up*dmp.y/viewSize.y*aspect.y)*lengthMult*S*shiftMult*0.8;
      viewChanged = true;
    }

  btn = (ImGui::IsMouseDragging(ImGuiMouseButton_Right) ? ImGuiMouseButton_Right : -1);
  if(btn == ImGuiMouseButton_Right && view.clickBtns(MOUSEBTN_RIGHT))
    { // right drag --> rotate camera
      Vec2f dmp = Vec2f(ImGui::GetMouseDragDelta(btn)); ImGui::ResetMouseDragDelta(btn);
      dmp = -dmp;
      float2 rAngles = float2{dmp.x, dmp.y} / float2{viewSize.x, viewSize.y} * 6.0 * tan(camera.fov/2*M_PI/180.0) * shiftMult;
      CFV3 rOffset = CFV3{(CFT)mParams.cp.fs.x, (CFT)mParams.cp.fs.y, (CFT)mParams.cp.fs.z}*mUnits.dL / 2.0;

      camera.pos -= rOffset; // offset to center of field
      camera.rotate(rAngles);
      camera.pos += rOffset;
      viewChanged = true;
    }

  if(view.hovered && std::abs(io.MouseWheel) > 0.0f)
    { // mouse scroll --> zoom
      if(io.KeyCtrl && io.KeyShift)     // signal pen depth
        { activeSigPen()->depth += (io.MouseWheel > 0 ? 1 : -1); }
      else if(io.KeyAlt && io.KeyShift) // material pen depth
        { activeMatPen()->depth += (io.MouseWheel > 0 ? 1 : -1); }
      else // zoom camera
        { camera.pos += ray.dir*shiftMult*(io.MouseWheel/20.0)*length(camera.pos); }
      viewChanged = true;
    }

  // add keyframe (TODO: only while recording)
  if(viewChanged)
    {
      mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_VIEW3D>(mKeyFrameUI->cursor(), camera.desc));
      mNewSimFrame = true; // new frame on view change
    }
}


// global input handling/routing
void SimWindow::handleInput(const Vec2f &frameSize)
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
  if(ImGui::BeginChild("##simView-main", frameSize, false, wFlags)) // begin main (interactive) view
  {
    if(mDisplayUI->showEMView)  { handleInput2D(mEMView,  mSimView2D);  } // EM view
    if(mDisplayUI->showMatView) { handleInput2D(mMatView, mSimView2D);  } // Material view
    if(mDisplayUI->show3DView)  { handleInput3D(m3DView,  mCamera);     } // 3D view (ray marching)
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

  SignalPen<CFT>   *sigPen = activeSigPen();
  MaterialPen<CFT> *matPen = activeMatPen();

  bool hover = m3DView.hovered || mEMView.hovered || mMatView.hovered;
  bool active = false;
  bool apply = false;
  CFV3 p = CFV3{NAN, NAN, NAN};
  bool posChanged = false;
  bool penChanged = false; // (mDrawUI->sigPen);// TODO: find good way to keep track of pen changes (settings?)
      
  if(!mParams.op.lockViews && !mForcePause && io.KeyCtrl)
    {
      if(m3DView.hovered)
        {
          p = to_cuda(m3DView.mposSim - m3DView.mposFace*sigPen->depth);
          apply = (m3DView.clickBtns(MOUSEBTN_LEFT) && (m3DView.clickMods(GLFW_MOD_CONTROL) || sigPen->active));
        }
      if(mEMView.hovered || mMatView.hovered)
        {
          mpfi.z = cpCopy.fs.z - 1 - sigPen->depth; // Z depth relative to top visible layer
          p = CFV3{mpfi.x, mpfi.y, mpfi.z};
          apply = ((mEMView.clickBtns(MOUSEBTN_LEFT) &&  (mEMView.clickMods(GLFW_MOD_CONTROL)  || sigPen->active)) ||
                   (mMatView.clickBtns(MOUSEBTN_LEFT) && (mMatView.clickMods(GLFW_MOD_CONTROL) || sigPen->active)));
        }
      if(hover)
        {
          posChanged = (mposLast != p) && (sigPen->startTime >= 0.0);
          //penChanged = false; // (mDrawUI->sigPen);// TODO: find good way to keep track of pen changes (settings?)

          mSigMPos = p;
          sigPen->mouseSpeed = length(mSigMPos - mposLast);
          mParams.rp.penPos = p;
          mParams.rp.sigPenHighlight = true;
          mParams.rp.sigPen = *sigPen;
          if(apply)
            { // draw signal to intermediate E/B fields (needs to be blended to avoid peristent blobs of
              active = true;
              if(sigPen->startTime < 0.0)
                { // signal started
                  sigPen->startTime = mInfo.t;
                  std::cout << "====> (SimWindow::handleInput) NEW SIG ADD EVENT\n";
                  mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_SIGNAL_ADD>(mKeyFrameUI->cursor(), "MANUAL", mSigMPos, *sigPen));
                }
              if((mFieldUI->running || cpCopy.u.dt != 0) && !mForcePause)
                { addSignal(p, *mInputV, *mInputP, *mInputQn, *mInputQp, *mInputQnv, *mInputQpv, *mInputE, *mInputB, *sigPen, cpCopy, mUnits.dt); }
              else
                { addSignal(p, src->v, src->p, src->Qn, src->Qp, src->Qnv, src->Qpv, src->E, src->B, *sigPen, cpCopy, mUnits.dt); }
            }
        }
    }

  if(active)
    {
      if(posChanged)
        {
          std::cout << "====> (SimWindow::handleInput) NEW SIG MOVE EVENT\n";
          mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_SIGNAL_MOVE>(mKeyFrameUI->cursor(), "MANUAL", mSigMPos));
       }
      else if(penChanged)
        {
          std::cout << "====> (SimWindow::handleInput) NEW SIG ADD EVENT\n";
          mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_SIGNAL_PEN >(mKeyFrameUI->cursor(), "MANUAL", *sigPen));
        }
    }
  else
    {
      if(mParams.verbose) { std::cout << "====> (SimWindow::handleInput) SIG NOT ACTIVE\n"; }
      if(sigPen->startTime >= 0.0) // signal stopped
        {
          std::cout << "====> (SimWindow::handleInput) NEW SIG REMOVE EVENT\n";
          mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_SIGNAL_REMOVE>(mKeyFrameUI->cursor(), "MANUAL"));
        }
      sigPen->startTime = -1.0; // reset start time
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
  penChanged = false; // (mDrawUI->matPen);// TODO: find good way to keep trazck of pen changes (settings?)
  if(!mParams.op.lockViews && !mForcePause && io.KeyAlt)
    {
      if(m3DView.hovered)
        {
          p = to_cuda(m3DView.mposSim - m3DView.mposFace*matPen->depth);
          apply = (m3DView.clickBtns(MOUSEBTN_LEFT) && (m3DView.clickMods(GLFW_MOD_ALT) || matPen->active));
        }
      if(mEMView.hovered || mMatView.hovered)
        {
          mpfi.z = cpCopy.fs.z-1-matPen->depth; // Z depth relative to top visible layer
          p = CFV3{mpfi.x, mpfi.y, mpfi.z};
          apply = ((mEMView.clickBtns(MOUSEBTN_LEFT)  && (mEMView.clickMods(GLFW_MOD_ALT)  || matPen->active)) ||
                   (mMatView.clickBtns(MOUSEBTN_LEFT) && (mMatView.clickMods(GLFW_MOD_ALT) || matPen->active)));
        }

      apply &= !isnan(p);
      if(hover)
        {
          posChanged = (mposLast != p) && (matPen->startTime >= 0.0);
          //penChanged = false; // (mDrawUI->matPen);// TODO: find good way to keep track of pen changes (settings?)

          mMatMPos = p;
          mParams.rp.penPos = p;
          mParams.rp.matPenHighlight = true;
          mParams.rp.matPen = *matPen;
          if(apply)
            {
              active = true;
              if(sigPen->startTime < 0.0)
                { // signal started
                  sigPen->startTime = mInfo.t;
                }
              std::cout << "====> (SimWindow::handleInput) NEW MATERIAL EVENT\n";
              mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_MATERIAL>(mKeyFrameUI->cursor(), mMatMPos, *matPen));
              //mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_MATERIAL>(mInfo.t, p, *matPen));
            }
        }
    }

  if(active && (posChanged || penChanged))
    {
      std::cout << "====> (SimWindow::handleInput) MAT ADD/MOVE/PEN EVENT\n";
      mKeyFrameUI->addEvent(new KeyEvent<KEYEVENT_MATERIAL>(mKeyFrameUI->cursor(), mMatMPos, *matPen));
    }
  else
    {
      matPen->startTime = -1.0; // reset start time
      mMatMPos = CFV3{NAN, NAN, NAN};
    }
}



//////////////////////////////////////////////////////
//// OVERLAYS ////////////////////////////////////////
//////////////////////////////////////////////////////

void SimWindow::makeVectorField3D()
{
  // if(mNewFrameVec)
    {
      FluidField<CFT> *src = reinterpret_cast<FluidField<CFT>*>(mStates.back());
      if(!mVBuffer3D.allocated()) { mVBuffer3D.create(src->size.x*src->size.y*2); } // CudaVBO test
      fillVLines(*src, mVBuffer3D, mParams.cp);
    }
}

// draws 2D vector field overlay
void SimWindow::drawVectorField2D(ScreenView<CFT> &view, const Rect<CFT> &simView)
{
  ImGuiStyle &style = ImGui::GetStyle();
  ImGuiIO    &io    = ImGui::GetIO();
  Vec2f mpos = ImGui::GetMousePos();
  Vec2f fp   = view.screenToSim2D(mpos, simView)/mUnits.dL;

  // draw vector field data
  FluidField<CFT> *src = reinterpret_cast<FluidField<CFT>*>(mStates.back());
  if(src && mParams.vp.drawVectors && mFieldDrawList)
    {
      Vec2i fi    = makeV<int2>(float2{floor(fp.x), floor(fp.y)});
      Vec2f fo    = fp - fi;
      Vec2i fiAdj = Vec2i(std::min(std::max(fi.x, 0), mParams.cp.fs.x-1), std::min(std::max(fi.y, 0), mParams.cp.fs.y-1));

      float viewScale = max(simView.size()/(view.r.size()/4.0f));

      int vRad     = mParams.vp.mRadius;
      int cRad     = mParams.vp.cRadius;
      int vSpacing = mParams.vp.spacing;

      int2 iMin; int2 iMax;
      if(mParams.vp.mouseRadius)
        {
          if(vRad > cRad) { vSpacing = std::max(vSpacing, (int)ceil(vRad/(float)cRad)); }
          iMin = int2{std::max(fi.x-vRad*vSpacing, 0),               std::max(fi.y-vRad*vSpacing, 0)};
          iMax = int2{std::min(fi.x+vRad*vSpacing, mParams.cp.fs.x), std::min(fi.y+vRad*vSpacing, mParams.cp.fs.y)};
        }
      else
        {
          iMin = int2{0,0};
          iMax = int2{mParams.cp.fs.x, mParams.cp.fs.y};
        }

      Vec2f vp1 = view.screenToSim2D(view.r.p1, simView)/mUnits.dL;
      Vec2f vp2 = view.screenToSim2D(view.r.p2, simView)/mUnits.dL;
      if(!isnan(vp1) && !isinf(vp1) && !isnan(vp2) && !isinf(vp2))
        {
          Rect2f r0 = Rect2f(iMin, iMax).fixed();
          Rect2f r1 = Rect2f(vp1, vp2).fixed();  r1.p2 += Vec2f(1,1);
          Rect2f ir = r0.intersection(r1);
          iMin = to_cuda(Vec2i(floor(ir.p1))); iMax = to_cuda(Vec2i(ceil(ir.p2))+Vec2i(1,1));
        }      
      iMin.x = std::min(std::max(iMin.x, 0), mParams.cp.fs.x); iMax.x = std::min(std::max(iMax.x, 0), mParams.cp.fs.x);
      iMin.y = std::min(std::max(iMin.y, 0), mParams.cp.fs.y); iMax.y = std::min(std::max(iMax.y, 0), mParams.cp.fs.y);
      
      int2 iStart = int2{0, 0};
      int2 iEnd   = int2{(iMax.x - iMin.x)/vSpacing, (iMax.y - iMin.y)/vSpacing};
      int  zi     = mParams.rp.zRange.y;
      
      Vec2i offset = ((mParams.vp.mouseRadius ? (fiAdj % vSpacing) : Vec2i(0,0)) -
                      (max(Vec2i(std::floor(vp1.x), std::floor(vp1.y)), Vec2i(0,0)) % vSpacing)); // correct view shift
      
      unsigned long i0 = std::min(src->numCells-1, src->idx(std::max(0, std::min(mParams.cp.fs.x-1, iMin.x)),
                                                            std::max(0, std::min(mParams.cp.fs.y-1, iMin.y)), zi));
      unsigned long i1 = std::min(src->numCells-1, src->idx(std::max(0, std::min(mParams.cp.fs.x-1, iMax.x)),
                                                            std::max(0, std::min(mParams.cp.fs.y-1, iMax.y)), zi)) + 1UL;
      if(i0 > i1) { std::swap(i0, i1); }
      if(mParams.vp.drawV)   { src->v.pullData(i0, i1); }
      if(mParams.vp.drawQnv) { src->Qnv.pullData(i0, i1); } if(mParams.vp.drawQpv) { src->Qpv.pullData(i0, i1); }
      if(mParams.vp.drawE)   { src->E.pullData(i0, i1); } if(mParams.vp.drawB)  { src->B.pullData (i0, i1); }
      
      //if(!mVBuffer2D.allocated()) { mVBuffer2D.create(src->size.x*src->size.y*2); } // CudaVBO test

      mVectorField2D.clear();
      for(int ix = iStart.x; ix <= iEnd.x; ix++)
        for(int iy = iStart.y; iy <= iEnd.y; iy++)
          {
            int xi = iMin.x + ix*vSpacing + offset.x; int yi = iMin.y + iy*vSpacing + offset.y;
            float2 dp = float2{(float)(xi-fi.x), (float)(yi-fi.y)};
            if(!mParams.vp.mouseRadius || dot(dp, dp) <= (float)(vRad*vRad))
              {
                Vec3f p = Vec3f(xi, yi, zi);
                Vec3f sampleP = Vec3f(xi+0.5f, yi+0.5f, zi+0.5f);
                Vec2f sp = view.simToScreen2D(sampleP, simView);
                Vec3f vV; Vec3f vQnv; Vec3f vQpv; Vec3f vE; Vec3f vB;
                if(mParams.vp.smoothVectors)
                  {
                    sampleP = Vec3f(xi+fo.x, yi+fo.y, zi);
                    if(sampleP.x >= 0 && sampleP.x < src->size.x && sampleP.y >= 0 && sampleP.y < src->size.y)
                      {
                        bool x1p   = (sampleP.x+1 >= src->size.x); bool y1p = (sampleP.y+1 >= src->size.y);
                        bool x1y1p = (x1p || y1p);

                        if(mParams.vp.drawV)
                          {
                            Vec3f V00 = Vec3f(src->v.hData[src->v.idx((int)sampleP.x,(int)sampleP.y, zi)]);
                            Vec3f V01 = (x1p   ? V00 : Vec3f(src->v.hData[src->v.idx((int)sampleP.x+1, (int)sampleP.y,   zi)]));
                            Vec3f V10 = (y1p   ? V00 : Vec3f(src->v.hData[src->v.idx((int)sampleP.x,   (int)sampleP.y+1, zi)]));
                            Vec3f V11 = (x1y1p ? V00 : Vec3f(src->v.hData[src->v.idx((int)sampleP.x+1, (int)sampleP.y+1, zi)]));
                            vV = blerp(V00, V01, V10, V11, fo);
                          }
                        if(mParams.vp.drawQnv)
                          {
                            Vec3f Qnv00 = Vec3f(src->Qnv.hData[src->Qnv.idx((int)sampleP.x,(int)sampleP.y, zi)]);
                            Vec3f Qnv01 = (x1p   ? Qnv00 : Vec3f(src->Qnv.hData[src->Qnv.idx((int)sampleP.x+1, (int)sampleP.y,   zi)]));
                            Vec3f Qnv10 = (y1p   ? Qnv00 : Vec3f(src->Qnv.hData[src->Qnv.idx((int)sampleP.x,   (int)sampleP.y+1, zi)]));
                            Vec3f Qnv11 = (x1y1p ? Qnv00 : Vec3f(src->Qnv.hData[src->Qnv.idx((int)sampleP.x+1, (int)sampleP.y+1, zi)]));
                            vQnv = blerp(Qnv00, Qnv01, Qnv10, Qnv11, fo);
                          }
                        if(mParams.vp.drawQpv)
                          {
                            Vec3f Qpv00 = Vec3f(src->Qpv.hData[src->Qpv.idx((int)sampleP.x,(int)sampleP.y, zi)]);
                            Vec3f Qpv01 = (x1p   ? Qpv00 : Vec3f(src->Qpv.hData[src->Qpv.idx((int)sampleP.x+1, (int)sampleP.y,   zi)]));
                            Vec3f Qpv10 = (y1p   ? Qpv00 : Vec3f(src->Qpv.hData[src->Qpv.idx((int)sampleP.x,   (int)sampleP.y+1, zi)]));
                            Vec3f Qpv11 = (x1y1p ? Qpv00 : Vec3f(src->Qpv.hData[src->Qpv.idx((int)sampleP.x+1, (int)sampleP.y+1, zi)]));
                            vQpv = blerp(Qpv00, Qpv01, Qpv10, Qpv11, fo);
                          }
                        if(mParams.vp.drawE)
                          {
                            Vec3f pE  = sampleP;// - 0.5f;
                            Vec3f E00 = Vec3f(src->E.hData[src->E.idx((int)pE.x,(int)pE.y, zi)]);
                            Vec3f E01 = (x1p   ? E00 : Vec3f(src->E.hData[src->E.idx((int)pE.x+1, (int)pE.y,   zi)]));
                            Vec3f E10 = (y1p   ? E00 : Vec3f(src->E.hData[src->E.idx((int)pE.x,   (int)pE.y+1, zi)]));
                            Vec3f E11 = (x1y1p ? E00 : Vec3f(src->E.hData[src->E.idx((int)pE.x+1, (int)pE.y+1, zi)]));
                            vE = blerp(E00, E01, E10, E11, fo);
                          }
                        if(mParams.vp.drawB)
                          {
                            Vec3f pB  = sampleP + 0.5f;
                            Vec3f B00 = Vec3f(src->B.hData[src->B.idx((int)pB.x,(int)pB.y, zi)]);
                            Vec3f B01 = (x1p   ? B00 : Vec3f(src->B.hData[src->B.idx((int)pB.x+1, (int)pB.y,   zi)]));
                            Vec3f B10 = (y1p   ? B00 : Vec3f(src->B.hData[src->B.idx((int)pB.x,   (int)pB.y+1, zi)]));
                            Vec3f B11 = (x1y1p ? B00 : Vec3f(src->B.hData[src->B.idx((int)pB.x+1, (int)pB.y+1, zi)]));
                            vB = blerp(B00, B01, B10, B11, fo);
                          }
                      }
                  }
                else
                  {
                    if(sampleP.x >= 0 && sampleP.x < src->size.x && sampleP.y >= 0 && sampleP.y < src->size.y && sampleP.z >= 0 && sampleP.z < src->size.z)
                      {
                        int i = src->idx(xi, yi, zi);
                        i  = src->idx(sampleP.x, sampleP.y, sampleP.z);
                        if(mParams.vp.drawV)   { vV   = src->v.hData[i];   }
                        if(mParams.vp.drawQnv) { vQnv = src->Qnv.hData[i]; }
                        if(mParams.vp.drawQpv) { vQpv = src->Qpv.hData[i]; }
                        if(mParams.vp.drawE)   { vE   = src->E.hData[i];   }
                        if(mParams.vp.drawB)   { vB   = src->B.hData[i];   }
                      }
                  }
                vV   *= mParams.vp.lBase; vV.y   *= -1;
                vQnv *= mParams.vp.lBase; vQnv.y *= -1;
                vQpv *= mParams.vp.lBase; vQpv.y *= -1;
                vE   *= mParams.vp.lBase; vE.y   *= -1;
                vB   *= mParams.vp.lBase; vB.y   *= -1;
                mVectorField2D.push_back(FVector{p, sampleP, vV, vQnv, vQpv, vE, vB});
              }
          }

      for(auto &v : mVectorField2D)
        {
          Vec2f sp  = view.simToScreen2D(v.sp*mUnits.dL, simView);

          float lAlpha = mParams.vp.aBase;
          float lw0    = mParams.vp.wBase / (mParams.vp.scaleToView ? viewScale : 1.0f);

          if(mParams.vp.drawV)
            {
              Vec4f Vcol = Vec4f(mParams.vp.colV.x,  mParams.vp.colV.y,  mParams.vp.colV.z,  lAlpha*mParams.vp.colV.w);
              Vec2f dpV  = view.simToScreen2D(v.vV, simView, true)*mParams.vp.multV;
              float lw   = std::min(lw0*mParams.vp.lwV, length(dpV));
              if(lw > 0.001f && Vcol.w >= 0.001f)
                {
                  float tipW = std::max(3.0f*lw, 10.0f);
                  drawVector(mFieldDrawList, sp, dpV, Vcol, lw, tipW, tan(M_PI/3.0));
                }
            }
          if(mParams.vp.drawQnv)
            {
              Vec4f Qnvcol = Vec4f(mParams.vp.colQnv.x, mParams.vp.colQnv.y, mParams.vp.colQnv.z, lAlpha*mParams.vp.colQnv.w);
              Vec2f dpQnv  = view.simToScreen2D(v.vQnv, simView, true)*mParams.vp.multQnv;
              float lw    = std::min(lw0*mParams.vp.lwQnv, length(dpQnv));
              if(lw > 0.001f && Qnvcol.w >= 0.001f)
                {
                  float tipW = std::max(3.0f*lw, 10.0f);
                  drawVector(mFieldDrawList, sp, dpQnv, Qnvcol, lw, tipW, tan(M_PI/3.0));
                }
            }
          if(mParams.vp.drawQpv)
            {
              Vec4f Qpvcol = Vec4f(mParams.vp.colQpv.x, mParams.vp.colQpv.y, mParams.vp.colQpv.z, lAlpha*mParams.vp.colQpv.w);
              Vec2f dpQpv  = view.simToScreen2D(v.vQpv, simView, true)*mParams.vp.multQpv;
              float lw    = std::min(lw0*mParams.vp.lwQpv, length(dpQpv));
              if(lw > 0.001f && Qpvcol.w >= 0.001f)
                {
                  float tipW = std::max(3.0f*lw, 10.0f);
                  drawVector(mFieldDrawList, sp, dpQpv, Qpvcol, lw, tipW, tan(M_PI/3.0));
                }
            }
          if(mParams.vp.drawE)
            {
              Vec4f Ecol = Vec4f(mParams.vp.colE.x,  mParams.vp.colE.y,  mParams.vp.colE.z,  lAlpha*mParams.vp.colE.w);
              Vec2f dpE  = view.simToScreen2D(v.vE, simView, true)*mParams.vp.multE;
              float lw   = std::min(lw0*mParams.vp.lwE, length(dpE));
              if(lw > 0.001f && Ecol.w >= 0.001f)
                {
                  Vec2f spE  = view.simToScreen2D((v.sp // - 0.5f
                                                   )*mUnits.dL, simView);
                  float tipW = std::max(3.0f*lw, 10.0f);
                  drawVector(mFieldDrawList, spE, dpE, Ecol, lw, tipW, tan(M_PI/3.0));
                }
            }
          if(mParams.vp.drawB)
            {
              Vec4f Bcol = Vec4f(mParams.vp.colB.x,  mParams.vp.colB.y,  mParams.vp.colB.z,  lAlpha*mParams.vp.colB.w);
              Vec2f dpB  = view.simToScreen2D(v.vB, simView, true)*mParams.vp.multB;
              float lw   = std::min(lw0*mParams.vp.lwB, length(dpB));
              if(lw > 0.001f && Bcol.w >= 0.001f)
                {
                  Vec2f spB  = view.simToScreen2D((v.sp + 0.5f)*mUnits.dL, simView);
                  float tipW = std::max(3.0f*lw, 10.0f);
                  drawVector(mFieldDrawList, spB, dpB, Bcol, lw, tipW, tan(M_PI/3.0));
                }
            }
        }
    }
}

// overlay for 2D sim views
void SimWindow::draw2DOverlay(const ScreenView<CFT> &view, const Rect<CFT> &simView)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  // draw axes at origin
  if(mDisplayUI->drawAxes)
    {
      float scale  = max(mParams.cp.fs)*mUnits.dL*0.25f;
      float zScale = std::max(mParams.cp.fs.z*mUnits.dL*0.15f, scale/3.0f);
      Vec2f tSize = ImGui::CalcTextSize("X");
      float pad = 5.0f;
      float zW0 = 1.0f; float zW1 = 10.0f; // width of visible z layer bar at min and max 
      // X/Y
      Vec2f WO0 = Vec2f(0,0); // origin
      Vec2f So  = view.simToScreen2D(WO0, simView);
      Vec2f Spx = view.simToScreen2D(WO0 + Vec2f(scale, 0), simView);
      Vec2f Spy = view.simToScreen2D(WO0 + Vec2f(0, scale), simView);
      drawList->AddLine(So, Spx, ImColor(X_COLOR), 2.0f);
      drawList->AddLine(So, Spy, ImColor(Y_COLOR), 2.0f);
      ImGui::PushFont(titleFontB);
      drawList->AddText((Spx+So)/2.0f - Vec2f(tSize.x/2.0f, 0) + Vec2f(0, pad), ImColor(X_COLOR), "X");
      drawList->AddText((Spy+So)/2.0f - Vec2f(tSize.x, tSize.y/2.0f) - Vec2f(2.0f*pad, 0), ImColor(Y_COLOR), "Y");

      // Z
      if(mParams.cp.fs.z > 1)
        {
          float zAngle = M_PI*4.0f/3.0f;
          Vec2f zVec   = Vec2f(cos(zAngle), sin(zAngle));
          Vec2f zVNorm = Vec2f(zVec.y, zVec.x);
          Vec2f Spz    = view.simToScreen2D(WO0 + zScale*zVec, simView);
          float zMin = mParams.rp.zRange.x/(float)(mParams.cp.fs.z-1);
          float zMax = mParams.rp.zRange.y/(float)(mParams.cp.fs.z-1);
          Vec2f SpzMin = view.simToScreen2D(WO0 + zScale*zVec*zMin, simView);
          Vec2f SpzMax = view.simToScreen2D(WO0 + zScale*zVec*zMax, simView);
          float zMinW = zW0*(1-zMin) + zW1*zMin;
          float zMaxW = zW0*(1-zMax) + zW1*zMax;

          drawList->AddLine(So, Spz, ImColor(Vec4f(1,1,1,0.5)), 1.0f); // grey line bg
          drawList->AddLine(SpzMin, SpzMax, ImColor(Z_COLOR), 2.0f);        // colored over view range

          // visible z range markers
          drawList->AddLine(SpzMin + Vec2f(zMinW,0), SpzMin - Vec2f(zMinW,0), ImColor(Z_COLOR), 2.0f);
          drawList->AddLine(SpzMax + Vec2f(zMaxW,0), SpzMax - Vec2f(zMaxW,0), ImColor(Z_COLOR), 2.0f);
          std::stringstream ss;   ss << mParams.rp.zRange.x; std::string zMinStr = ss.str();
          ss.str(""); ss.clear(); ss << mParams.rp.zRange.y; std::string zMaxStr = ss.str();
          if(zMin != zMax) { drawList->AddText(SpzMin + Vec2f(zMinW+pad, 0), ImColor(Z_COLOR), zMinStr.c_str()); }
          drawList->AddText(SpzMax + Vec2f(zMaxW+pad, 0), ImColor(Z_COLOR), zMaxStr.c_str());
          drawList->AddText((Spz + So)/2.0f - tSize - Vec2f(pad,pad), ImColor(Z_COLOR), "Z");

        }
      ImGui::PopFont();
    }

  // draw outline around field
  if(mDisplayUI->drawOutline)
    {
      Vec3f Wfp0 = Vec3f(mParams.cp.fp.x, mParams.cp.fp.y, mParams.cp.fp.z) * mUnits.dL;
      Vec3f Wfs  = Vec3f(mParams.cp.fs.x, mParams.cp.fs.y, mParams.cp.fs.z) * mUnits.dL;
      drawRect2D(view, mSimView2D, Vec2f(Wfp0.x, Wfp0.y), Vec2f(Wfp0.x+Wfs.x, Wfp0.y+Wfs.y), RADIUS_COLOR);
    }

  // draw positional axes of active signal pen
  if(!isnan(mSigMPos) && !mParams.op.lockViews && !mForcePause)
    {
      Vec3f W01n  = Vec3f(mParams.cp.fp.x, mSigMPos.y, mSigMPos.z)*mUnits.dL;
      Vec3f W01p  = Vec3f(mParams.cp.fp.x + mParams.cp.fs.x, mSigMPos.y, mSigMPos.z)*mUnits.dL;
      Vec3f W10p  = Vec3f(mSigMPos.x, mParams.cp.fp.y + mParams.cp.fs.y, mSigMPos.z)*mUnits.dL;
      Vec3f W10n  = Vec3f(mSigMPos.x, mParams.cp.fp.y, mSigMPos.z)*mUnits.dL;
      // transform to screen space
      Vec2f S01n = view.simToScreen2D(W01n, simView);
      Vec2f S01p = view.simToScreen2D(W01p, simView);
      Vec2f S10n = view.simToScreen2D(W10n, simView);
      Vec2f S10p = view.simToScreen2D(W10p, simView);
      // X guides
      drawList->AddLine(S01n, S01p, ImColor(GUIDE_COLOR), 2.0f);
      drawList->AddCircleFilled(S01n, 3, ImColor(X_COLOR), 6);
      drawList->AddCircleFilled(S01p, 3, ImColor(X_COLOR), 6);
      // Y guides
      drawList->AddLine(S10n, S10p, ImColor(GUIDE_COLOR), 2.0f);
      drawList->AddCircleFilled(S10n, 3, ImColor(Y_COLOR), 6);
      drawList->AddCircleFilled(S10p, 3, ImColor(Y_COLOR), 6);

      // draw intersected radii (lenses)
      SignalPen<CFT> *sigPen = activeSigPen();
      Vec3f WR0 = ((sigPen->cellAlign ? floor(mSigMPos) : mSigMPos)
                   + sigPen->rDist*sigPen->sizeMult*sigPen->xyzMult/2.0f)*mUnits.dL;
      Vec3f WR1 = ((sigPen->cellAlign ? floor(mSigMPos) : mSigMPos)
                   - sigPen->rDist*sigPen->sizeMult*sigPen->xyzMult/2.0f)*mUnits.dL;
      Vec2f SR0 = Vec2f(WR0.x, WR0.y);
      Vec2f SR1 = Vec2f(WR1.x, WR1.y);

      // centers
      drawList->AddCircleFilled(view.simToScreen2D(SR0, simView), 3, ImColor(RADIUS_COLOR), 6);
      drawList->AddCircleFilled(view.simToScreen2D(SR1, simView), 3, ImColor(RADIUS_COLOR), 6);

      // outlines
      Vec3f r0_3 = sigPen->radius0 * mUnits.dL * sigPen->sizeMult*sigPen->xyzMult;
      Vec3f r1_3 = sigPen->radius1 * mUnits.dL * sigPen->sizeMult*sigPen->xyzMult;
      Vec2f r0 = Vec2f(r0_3.x, r0_3.y); Vec2f r1 = Vec2f(r1_3.x, r1_3.y);
      if(sigPen->square)
        {
          drawRect2D(view, mSimView2D, SR0-r0, SR0+r0, RADIUS_COLOR);
          drawRect2D(view, mSimView2D, SR1-r1, SR1+r1, RADIUS_COLOR);
        }
      else
        {
          drawEllipse2D(view, mSimView2D, SR0, r0, RADIUS_COLOR);
          drawEllipse2D(view, mSimView2D, SR1, r1, RADIUS_COLOR);
        }
    }

  // draw positional axes of active material pen
  if(!isnan(mMatMPos) && !mParams.op.lockViews && !mForcePause)
    { // world points
      Vec3f W01n  = Vec3f(mParams.cp.fp.x, mMatMPos.y, mMatMPos.z)*mUnits.dL;
      Vec3f W01p  = Vec3f(mParams.cp.fp.x + mParams.cp.fs.x, mMatMPos.y, mMatMPos.z)*mUnits.dL;
      Vec3f W10p  = Vec3f(mMatMPos.x, mParams.cp.fp.y + mParams.cp.fs.y, mMatMPos.z)*mUnits.dL;
      Vec3f W10n  = Vec3f(mMatMPos.x, mParams.cp.fp.y, mMatMPos.z)*mUnits.dL;
      // transform to screen space
      Vec2f S01n = view.simToScreen2D(W01n, simView);
      Vec2f S01p = view.simToScreen2D(W01p, simView);
      Vec2f S10n = view.simToScreen2D(W10n, simView);
      Vec2f S10p = view.simToScreen2D(W10p, simView);
      // X guides
      drawList->AddLine(S01n, S01p, ImColor(GUIDE_COLOR), 2.0f);
      drawList->AddCircleFilled(S01n, 3, ImColor(X_COLOR), 6);
      drawList->AddCircleFilled(S01p, 3, ImColor(X_COLOR), 6);
      // Y guides
      drawList->AddLine(S10n, S10p, ImColor(GUIDE_COLOR), 2.0f);
      drawList->AddCircleFilled(S10n, 3, ImColor(Y_COLOR), 6);
      drawList->AddCircleFilled(S10p, 3, ImColor(Y_COLOR), 6);

      // draw intersected radii (lenses)
      MaterialPen<CFT> *matPen = activeMatPen();
      Vec3f WR0 = ((matPen->cellAlign ? floor(mMatMPos) : mMatMPos)
                   + matPen->rDist*matPen->sizeMult*matPen->xyzMult/2.0f)*mUnits.dL;
      Vec3f WR1 = ((matPen->cellAlign ? floor(mMatMPos) : mMatMPos)
                   - matPen->rDist*matPen->sizeMult*matPen->xyzMult/2.0f)*mUnits.dL;
      Vec2f SR0 = Vec2f(WR0.x, WR0.y);
      Vec2f SR1 = Vec2f(WR1.x, WR1.y);

      // centers
      drawList->AddCircleFilled(view.simToScreen2D(SR0, simView), 3, ImColor(RADIUS_COLOR), 6);
      drawList->AddCircleFilled(view.simToScreen2D(SR1, simView), 3, ImColor(RADIUS_COLOR), 6);

      // outlines
      Vec3f r0_3 = matPen->radius0 * mUnits.dL * matPen->sizeMult*matPen->xyzMult;
      Vec3f r1_3 = matPen->radius1 * mUnits.dL * matPen->sizeMult*matPen->xyzMult;
      Vec2f r0 = Vec2f(r0_3.x, r0_3.y); Vec2f r1 = Vec2f(r1_3.x, r1_3.y);
      if(matPen->square)
        {
          drawRect2D(view, mSimView2D, SR0-r0, SR0+r0, RADIUS_COLOR);
          drawRect2D(view, mSimView2D, SR1-r1, SR1+r1, RADIUS_COLOR);
        }
      else
        {
          drawEllipse2D(view, mSimView2D, SR0, r0, RADIUS_COLOR);
          drawEllipse2D(view, mSimView2D, SR1, r1, RADIUS_COLOR);
        }
    }
}

// overlay for 3D sim views
void SimWindow::draw3DOverlay(const ScreenView<CFT> &view, const Camera<CFT> &simView)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  Vec2f mpos = ImGui::GetMousePos();
  // Vec2f aspect = Vec2f(view.r.aspect(), 1.0);
  // Vec2f vSize = view.r.size();
  // Vec2f aOffset = Vec2f(vSize.x/aspect.x - vSize.x, vSize.y/aspect.y - vSize.y)/2.0f;

  mCamera.aspect = view.r.aspect();
  mCamera.calculate();

  // draw X/Y/Z axes at origin (R/G/B)
  if(mDisplayUI->drawAxes)
    {
      float scale = max(mParams.cp.fs)*mUnits.dL*0.25f;
      Vec3f WO0 = Vec3f(0,0,0); // origin
      Vec3f Wpx = WO0 + Vec3f(scale, 0, 0);
      Vec3f Wpy = WO0 + Vec3f(0, scale, 0);
      Vec3f Wpz = WO0 + Vec3f(0, 0, scale);
      // transform to screen space
      bool oClipped = false; bool xClipped = false; bool yClipped = false; bool zClipped = false;
      Vec4f So  = mCamera.worldToView(WO0, &oClipped); Vec4f Spx = mCamera.worldToView(Wpx, &xClipped);
      Vec4f Spy = mCamera.worldToView(Wpy, &yClipped); Vec4f Spz = mCamera.worldToView(Wpz, &zClipped);
      // draw axes
      if(!oClipped || !xClipped) { drawList->AddLine(view.toScreen(mCamera.nearClip(So, Spx)), view.toScreen(mCamera.nearClip(Spx, So)), ImColor(X_COLOR), 2.0f); }
      if(!oClipped || !yClipped) { drawList->AddLine(view.toScreen(mCamera.nearClip(So, Spy)), view.toScreen(mCamera.nearClip(Spy, So)), ImColor(Y_COLOR), 2.0f); }
      if(!oClipped || !zClipped) { drawList->AddLine(view.toScreen(mCamera.nearClip(So, Spz)), view.toScreen(mCamera.nearClip(Spz, So)), ImColor(Z_COLOR), 2.0f); }
    }

  // draw outline around field
  if(mDisplayUI->drawOutline)
    {
      Vec3f Wp = Vec3f(mParams.cp.fp.x, mParams.cp.fp.y, mParams.cp.fp.z) * mUnits.dL;
      Vec3f Ws = Vec3f(mParams.cp.fs.x, mParams.cp.fs.y, mParams.cp.fs.z) * mUnits.dL;
      drawRect3D(view, mCamera, Wp, Wp+Ws, OUTLINE_COLOR);
    }

  // draw positional axes of active signal pen
  if(!isnan(mSigMPos) && !mParams.op.lockViews && !mForcePause)
    {
      Vec3f W001n  = Vec3f(mParams.cp.fp.x, mSigMPos.y, mSigMPos.z)*mUnits.dL;
      Vec3f W001p  = Vec3f(mParams.cp.fp.x + mParams.cp.fs.x, mSigMPos.y, mSigMPos.z)*mUnits.dL;
      Vec3f W010p  = Vec3f(mSigMPos.x, mParams.cp.fp.y + mParams.cp.fs.y, mSigMPos.z)*mUnits.dL;
      Vec3f W010n  = Vec3f(mSigMPos.x, mParams.cp.fp.y, mSigMPos.z)*mUnits.dL;
      Vec3f W100n  = Vec3f(mSigMPos.x, mSigMPos.y, mParams.cp.fp.x)*mUnits.dL;
      Vec3f W100p  = Vec3f(mSigMPos.x, mSigMPos.y, mParams.cp.fp.z + mParams.cp.fs.z)*mUnits.dL;
      // transform to screen space
      bool C001n = false; bool C001p = false; bool C010n = false; bool C010p = false; bool C100n = false; bool C100p = false;
      Vec4f S001n = mCamera.worldToView(W001n, &C001n); Vec4f S001p = mCamera.worldToView(W001p, &C001p);
      Vec4f S010n = mCamera.worldToView(W010n, &C010n); Vec4f S010p = mCamera.worldToView(W010p, &C010p);
      Vec4f S100n = mCamera.worldToView(W100n, &C100n); Vec4f S100p = mCamera.worldToView(W100p, &C100p);
      if(!C001n || !C001p)
        { // X guides
          drawList->AddLine(view.toScreen(mCamera.nearClip(S001n, S001p)), view.toScreen(mCamera.nearClip(S001p, S001n)), ImColor(GUIDE_COLOR), 2.0f);
          drawList->AddCircleFilled(view.toScreen(mCamera.vNormalize(S001n)), 3, ImColor(X_COLOR), 6);
          drawList->AddCircleFilled(view.toScreen(mCamera.vNormalize(S001p)), 3, ImColor(X_COLOR), 6);
        }
      if(!C010n || !C010p)
        { // Y guides
          drawList->AddLine(view.toScreen(mCamera.nearClip(S010n, S010p)), view.toScreen(mCamera.nearClip(S010p, S010n)), ImColor(GUIDE_COLOR), 2.0f);
          drawList->AddCircleFilled(view.toScreen(mCamera.vNormalize(S010n)), 3, ImColor(Y_COLOR), 6);
          drawList->AddCircleFilled(view.toScreen(mCamera.vNormalize(S010p)), 3, ImColor(Y_COLOR), 6);
        }
      if(!C100n || !C100p)
        { // Z guides
          drawList->AddLine(view.toScreen(mCamera.nearClip(S100n, S100p)), view.toScreen(mCamera.nearClip(S100p, S100n)), ImColor(GUIDE_COLOR), 2.0f);
          drawList->AddCircleFilled(view.toScreen(mCamera.vNormalize(S100n)), 3, ImColor(Z_COLOR), 6);
          drawList->AddCircleFilled(view.toScreen(mCamera.vNormalize(S100p)), 3, ImColor(Z_COLOR), 6);
        }

      // draw intersected radii (lenses)
      SignalPen<CFT> *sigPen = activeSigPen();
      float S = 2.0*tan(mCamera.fov/2.0f*M_PI/180.0f);
      Vec3f WR0 = ((sigPen->cellAlign ? floor(mSigMPos) : mSigMPos)
                   + sigPen->rDist*sigPen->sizeMult*sigPen->xyzMult/2.0f)*mUnits.dL;
      Vec3f WR1 = ((sigPen->cellAlign ? floor(mSigMPos) : mSigMPos)
                   - sigPen->rDist*sigPen->sizeMult*sigPen->xyzMult/2.0f)*mUnits.dL;
      bool CR0 = false; bool CR1 = false;
      Vec4f SR0 = mCamera.worldToView(WR0, &CR0); Vec4f SR1 = mCamera.worldToView(WR1, &CR1);
      // centers
      drawList->AddCircleFilled(view.toScreen(mCamera.vNormalize(SR0)), 3, ImColor(RADIUS_COLOR), 6);
      drawList->AddCircleFilled(view.toScreen(mCamera.vNormalize(SR1)), 3, ImColor(RADIUS_COLOR), 6);

      Vec3f r0 = Vec3f(S,S,1)*sigPen->radius0 * mUnits.dL * sigPen->sizeMult*sigPen->xyzMult;
      Vec3f r1 = Vec3f(S,S,1)*sigPen->radius1 * mUnits.dL * sigPen->sizeMult*sigPen->xyzMult;
      if(sigPen->square)
        {
          drawRect3D(view, mCamera, WR0-r0, WR0+r0, RADIUS_COLOR);
          drawRect3D(view, mCamera, WR1-r1, WR1+r1, RADIUS_COLOR);
        }
      else
        {
          drawEllipse3D(view, mCamera, WR0, r0, RADIUS_COLOR);
          drawEllipse3D(view, mCamera, WR1, r1, RADIUS_COLOR);
        }
    }

  // draw positional axes of active material pen
  if(!isnan(mMatMPos) && !mParams.op.lockViews && !mForcePause)
    {
      Vec3f W001n  = Vec3f(mParams.cp.fp.x, mMatMPos.y, mMatMPos.z)*mUnits.dL;
      Vec3f W010n  = Vec3f(mMatMPos.x, mParams.cp.fp.y, mMatMPos.z)*mUnits.dL;
      Vec3f W100n  = Vec3f(mMatMPos.x, mMatMPos.y, mParams.cp.fp.z)*mUnits.dL;
      Vec3f W001p  = Vec3f(mParams.cp.fp.x + mParams.cp.fs.x, mMatMPos.y, mMatMPos.z)*mUnits.dL;
      Vec3f W010p  = Vec3f(mMatMPos.x, mParams.cp.fp.y + mParams.cp.fs.y, mMatMPos.z)*mUnits.dL;
      Vec3f W100p  = Vec3f(mMatMPos.x, mMatMPos.y, mParams.cp.fp.z + mParams.cp.fs.z)*mUnits.dL;
      // transform to screen space
      bool C001n = false; bool C010n = false; bool C100n = false; bool C001p = false; bool C010p = false; bool C100p = false;
      Vec4f S001n = mCamera.worldToView(W001n, &C001n); Vec4f S010n = mCamera.worldToView(W010n, &C010n);
      Vec4f S100n = mCamera.worldToView(W100n, &C100n); Vec4f S001p = mCamera.worldToView(W001p, &C001p);
      Vec4f S010p = mCamera.worldToView(W010p, &C010p); Vec4f S100p = mCamera.worldToView(W100p, &C100p);
      if(!C001n || !C001p)
        {
          drawList->AddLine(view.toScreen(mCamera.nearClip(S001n, S001p)), view.toScreen(mCamera.nearClip(S001p, S001n)), ImColor(GUIDE_COLOR), 2.0f);
          drawList->AddCircleFilled(view.toScreen(mCamera.vNormalize(S001n)), 3, ImColor(X_COLOR), 6);
          drawList->AddCircleFilled(view.toScreen(mCamera.vNormalize(S001p)), 3, ImColor(X_COLOR), 6);
        }
      if(!C010n || !C010p)
        {
          drawList->AddLine(view.toScreen(mCamera.nearClip(S010n, S010p)), view.toScreen(mCamera.nearClip(S010p, S010n)), ImColor(GUIDE_COLOR), 2.0f);
          drawList->AddCircleFilled(view.toScreen(mCamera.vNormalize(S010n)), 3, ImColor(Y_COLOR), 6);
          drawList->AddCircleFilled(view.toScreen(mCamera.vNormalize(S010p)), 3, ImColor(Y_COLOR), 6);
        }
      if(!C100n || !C100p)
        {
          drawList->AddLine(view.toScreen(mCamera.nearClip(S100n, S100p)), view.toScreen(mCamera.nearClip(S100p, S100n)), ImColor(GUIDE_COLOR), 2.0f);
          drawList->AddCircleFilled(view.toScreen(mCamera.vNormalize(S100n)), 3, ImColor(Z_COLOR), 6);
          drawList->AddCircleFilled(view.toScreen(mCamera.vNormalize(S100p)), 3, ImColor(Z_COLOR), 6);
        }

      // draw intersected radii (lenses)
      MaterialPen<CFT> *matPen = activeMatPen();
      float S = 2.0*tan(mCamera.fov/2.0f*M_PI/180.0f);
      Vec3f WR0 = ((matPen->cellAlign ? floor(mMatMPos) : mMatMPos)
                   + matPen->rDist*matPen->sizeMult*matPen->xyzMult/2.0f)*mUnits.dL;
      Vec3f WR1 = ((matPen->cellAlign ? floor(mMatMPos) : mMatMPos)
                   - matPen->rDist*matPen->sizeMult*matPen->xyzMult/2.0f)*mUnits.dL;
      bool CR0 = false; bool CR1 = false;
      Vec4f SR0 = mCamera.worldToView(WR0, &CR0); Vec4f SR1 = mCamera.worldToView(WR1, &CR1);
      // centers
      drawList->AddCircleFilled(view.toScreen(mCamera.vNormalize(SR0)), 3, ImColor(RADIUS_COLOR), 6);
      drawList->AddCircleFilled(view.toScreen(mCamera.vNormalize(SR1)), 3, ImColor(RADIUS_COLOR), 6);

      Vec3f r0 = Vec3f(S,S,1)*matPen->radius0 * mUnits.dL * matPen->sizeMult*matPen->xyzMult;
      Vec3f r1 = Vec3f(S,S,1)*matPen->radius1 * mUnits.dL * matPen->sizeMult*matPen->xyzMult;
      if(matPen->square)
        {
          drawRect3D(view, mCamera, WR0-r0, WR0+r0, RADIUS_COLOR);
          drawRect3D(view, mCamera, WR1-r1, WR1+r1, RADIUS_COLOR);
        }
      else
        {
          drawEllipse3D(view, mCamera, WR0, r0, RADIUS_COLOR);
          drawEllipse3D(view, mCamera, WR1, r1, RADIUS_COLOR);
        }
    }
}


//////////////////////////////////////////////////////
//// DRAW/RENDER /////////////////////////////////////
//////////////////////////////////////////////////////

// render simulation views based on frame size and id
//   (id="main"    --> signifies main interactive view
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

  // adjust aspect ratios if window size has changed (double precision used due to noticeable floating point error while resizing continuously (?))
  //   TODO: improve
  double aRatio  = (double)mEMView.r.aspect()/(double)mSimView2D.aspect();
  Rect2d newSimView = mSimView2D; // Rect2d(Vec2d(mSimView2D.p1), Vec2d(mSimView2D.p2));
  double fAspect =  (double)mParams.cp.fs.x/(double)mParams.cp.fs.y;
  if(mEMView.r.aspect() > fAspect) { newSimView.scaleX(aRatio); }
  else                         { newSimView.scaleY(1.0/aRatio); }
  mSimView2D = newSimView; //Rect<CFT>(Vector<CFT,2>(newSimView.p1), Vector<CFT,2>(newSimView.p2));
  
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
  if(ImGui::BeginChild(("##simView-"+id).c_str(), frameSize, false, wFlags))
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
          Vec2f fScreenPos  = mEMView.simToScreen2D(fp, mSimView2D);
          Vec2f fCursorPos  = mEMView.simToScreen2D(fp + Vec2f(0.0f, mParams.cp.fs.y*mUnits.dL), mSimView2D);
          Vec2f fScreenSize = mEMView.simToScreen2D(makeV<CFV3>(mParams.cp.fs)*mUnits.dL, mSimView2D, true);
          Vec2f t0(0.0f, 1.0f); Vec2f t1(1.0f, 0.0f);
          mEMTex.bind();
          ImGui::SetCursorScreenPos(fCursorPos);
          ImGui::Image(reinterpret_cast<ImTextureID>(mEMTex.texId()), fScreenSize, t0, t1, ImColor(Vec4f(1,1,1,1)));
          mEMTex.release();

          if(!mParams.op.lockViews && !mForcePause && (mEMView.hovered || !mParams.vp.mouseRadius)) { drawVectorField2D(mEMView, mSimView2D); }
          draw2DOverlay(mEMView, mSimView2D);
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
          Vec2f fScreenPos  = mMatView.simToScreen2D(fp, mSimView2D);
          Vec2f fCursorPos  = mMatView.simToScreen2D(fp + Vec2f(0.0f, mParams.cp.fs.y*mUnits.dL), mSimView2D);
          Vec2f fScreenSize = mMatView.simToScreen2D(makeV<CFV3>(mParams.cp.fs)*mUnits.dL, mSimView2D, true);
          Vec2f t0(0.0f, 1.0f); Vec2f t1(1.0f, 0.0f);
          mMatTex.bind();
          ImGui::SetCursorScreenPos(fCursorPos);
          ImGui::Image(reinterpret_cast<ImTextureID>(mMatTex.texId()), fScreenSize, t0, t1, ImColor(Vec4f(1,1,1,1)));
          mMatTex.release();

          if(!mParams.op.lockViews && !mForcePause && mMatView.hovered) { drawVectorField2D(mMatView, mSimView2D); }
          draw2DOverlay(mMatView, mSimView2D);
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
          // // TODO: show output frames in interactive view instead of rendering twice?
          // if(mParams.op.active && id == "main")
          //   {
          //     aspect.x = 1.0f/(tex3D->size.x/(float)tex3D->size.y);
          //     tex3D = &m3DGlTex;
          //     aspect.x *= tex3D->size.x/(float)tex3D->size.y;
          //   }
          mFieldDrawList = ImGui::GetWindowDrawList();
          Vec2f vSize = view3D->r.size()*aspect;
          Vec2f diff = (view3D->r.size() - vSize);

          Vec2f t0(0.0f, 1.0f); Vec2f t1(1.0f, 0.0f);
          tex3D->bind();
          ImGui::SetCursorScreenPos(view3D->r.p1 + diff/2.0f);
          ImGui::Image(reinterpret_cast<ImTextureID>(tex3D->texId()), vSize, t0, t1, ImColor(Vec4f(1,1,1,1)));
          tex3D->release();

          // makeVectorField3D(); // rendered separately (after ImGui frame)
          draw3DOverlay(*view3D, mCamera);

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
  mInfo.fps  = mMainFps.update();

  ImGuiStyle &style = ImGui::GetStyle();

  //// draw (imgui)
  ImGuiWindowFlags wFlags = (ImGuiWindowFlags_NoTitleBar      | ImGuiWindowFlags_NoCollapse        |
                             ImGuiWindowFlags_NoMove          | ImGuiWindowFlags_NoResize          |
                             ImGuiWindowFlags_NoScrollbar     | ImGuiWindowFlags_NoScrollWithMouse |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus );
  ImGui::SetNextWindowPos(Vec2f(0,0));
  ImGui::SetNextWindowSize(Vec2f(mFrameSize.x, mFrameSize.y));
  ImGui::PushStyleColor(ImGuiCol_WindowBg, Vec4f(0.05f, 0.05f, 0.05f, 1.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,   0); // square windows by default
  ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,    0); // square frames by default
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,    Vec2f(0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, Vec2f(1, 1));
  if(ImGui::Begin("##mainView", nullptr, wFlags)) // covers full application window
  {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, Vec2f(10, 10));
    ImGui::PushFont(mainFont);

    // top menu bar
    Vec2f menuBarSize;
    if(ImGui::BeginMainMenuBar())
    {
      menuBarSize = ImGui::GetWindowSize();
      if(ImGui::BeginMenu("File"))
        {
          if(ImGui::MenuItem("Reset Sim"))       { resetSim(false); }
          if(ImGui::MenuItem("Reset Views"))     { resetViews(false); }
          if(ImGui::MenuItem("Reset EM"))        { resetSignals(false); }
          if(ImGui::MenuItem("Reset Fluids"))    { resetFluid(false); }
          if(ImGui::MenuItem("Reset Materials")) { resetMaterials(false); }
          ImGui::EndMenu();
        }
      if(ImGui::BeginMenu("Tools"))
        {
          if(ImGui::MenuItem("Key Bindings"))  { mKeyManager->togglePopup(); }
          if(ImGui::MenuItem("ImGui Demo"))    { toggleImGuiDemo(); }
          if(ImGui::MenuItem("Freetype Demo")) { toggleFontDemo();  }
          ImGui::EndMenu();
        }
      ImGui::EndMainMenuBar();
    }
    ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos()) + Vec2f(0, menuBarSize.y));
    
    // key bindings/shortcuts
    if(mKeyManager)
      {
        mCaptured = (ImGui::GetIO().WantCaptureKeyboard && // NOTE: WantCaptureKeyboard set to true when mouse clicked
                     !mEMView.clickBtns() && !mMatView.clickBtns() && !m3DView.clickBtns()); // --> enable shortcuts anyway if clicked
        mKeyManager->draw(mFrameSize);
        mForcePause = mKeyManager->popupOpen();
      }

    Vec2f sideBarSize   = mSideTabs->getSize();
    Vec2f bottomBarSize = mBottomTabs->getSize();
    Vec2f sigPenBarSize  = mSigPenBar->getSize();
    Vec2f matPenBarSize  = mMatPenBar->getSize();
    Vec2f displaySize = (mFrameSize - //Vec2f(style.WindowPadding) -
                         // Vec2f((mSideTabs->selected() >= 0 ? style.ItemSpacing.x : 0.0f),
                         //       (mBottomTabs->selected() >= 0 ? style.ItemSpacing.y : 0.0f)) -
                         style.ItemSpacing -
                         Vec2f(sideBarSize.x + style.ItemSpacing.x,
                               menuBarSize.y + sigPenBarSize.y + matPenBarSize.y + bottomBarSize.y + 3.0f*style.ItemSpacing.y));

    mSigPenBar->setLength(displaySize.x);
    mMatPenBar->setLength(displaySize.x);
    mBottomTabs->setLength(displaySize.x);
    mSideTabs->setLength(mFrameSize.y - // style.WindowPadding.y - 
                         menuBarSize.y - mBottomTabs->getBarWidth());// - style.ItemSpacing.y);
    
    ImGui::BeginGroup();
    { // draw views
      render(displaySize);
      handleInput(displaySize);
      mMouseSimPosLast = mMouseSimPos;
      if(mEMView.hovered)  { mMouseSimPos = Vec2f(mEMView.mposSim.x,  mEMView.mposSim.y);  }
      if(mMatView.hovered) { mMouseSimPos = Vec2f(mMatView.mposSim.x, mMatView.mposSim.y); }
      if(m3DView.hovered)  { mMouseSimPos = Vec2f(m3DView.mposSim.x,  m3DView.mposSim.y);  }

      // pen toolbars
      mSigPenBar->draw(); sigPenBarSize = mSigPenBar->getSize();
      mMatPenBar->draw(); matPenBarSize = mMatPenBar->getSize();
      
      // bottom tab bar
      mBottomTabs->draw("bottomTabs");
      bottomBarSize = mBottomTabs->getSize();
    }
    ImGui::EndGroup();
    ImGui::SameLine();

    // side tab bar
    mSideTabs->draw("sideTabs");
    sideBarSize = mSideTabs->getSize();

    // enforce a minimum size
    Vec2f minSize = (Vec2f(512, 512) + Vec2f(sideBarSize.x, bottomBarSize.y) +
                     style.ItemSpacing + Vec2f(style.WindowPadding) + Vec2f(style.FramePadding)*2.0f);
    glfwSetWindowSizeLimits(mWindow, (int)minSize.x, (int)minSize.y, GLFW_DONT_CARE, GLFW_DONT_CARE);
    ImGui::SetWindowSize(sideBarSize + Vec2f(style.ScrollbarSize, 0.0f));

    // debug overlay
    if(mParams.debug)
      {
        Vec2f aspect3D = Vec2f(m3DView.r.aspect(), 1.0);
        ImDrawList *drawList = ImGui::GetForegroundDrawList();
        ImGui::PushClipRect(Vec2f(0,0), mFrameSize, false);
        ImGui::SetCursorPos(Vec2f(10.0f, 10.0f + menuBarSize.y));
        ImGui::PushStyleColor(ImGuiCol_ChildBg, Vec4f(0.0f, 0.0f, 0.0f, 0.0f));
        if(ImGui::BeginChild("##debugOverlay", displaySize, false, wFlags))
        {
          ImGui::Indent();
          ImGui::PushFont(titleFontB);
          ImGui::Text("Base:   %.2f FPS", mUpdateFps.fps);
          ImGui::Text("Render: %.2f FPS", mRenderFps.fps);
          ImGui::Text("Update: %.2f FPS", mPhysicsFps.fps);
          ImGui::PopFont();
          ImGui::Spacing();
          // sim info
          ImGui::Text("    t =  %f", mInfo.t);
          ImGui::Text("frame =  %d  (sim time %.3fx real time)", mInfo.frame, mFieldUI->running ? (mMainFps.fps*mUnits.dt) : 0.0f);
          ImGui::Text("Mouse Sim Pos: <%f, %f>",                 mMouseSimPos.x,        mMouseSimPos.y);
          ImGui::Text("3D Click Pos:  <%f, %f>",                 m3DView.clickPos.x,    m3DView.clickPos.y);
          ImGui::Text("SimView:       < %f, %f> : < %f, %f >",   mSimView2D.p1.x,       mSimView2D.p1.y, mSimView2D.p2.x, mSimView2D.p2.y);
          ImGui::Text("SimView Size:  < %f, %f>",                mSimView2D.size().x,   mSimView2D.size().y);
          ImGui::Text("EM  2D View:   < %f, %f> : < %f, %f>",    mEMView.r.p1.x,  mEMView.r.p1.y,  mEMView.r.p2.x,  mEMView.r.p2.y);
          ImGui::Text("Mat 2D View:   < %f, %f> : < %f, %f>",    mMatView.r.p1.x, mMatView.r.p1.y, mMatView.r.p2.x, mMatView.r.p2.y);
          ImGui::Text("3D  EM View:   < %f, %f> : < %f, %f>",    m3DView.r.p1.x,  m3DView.r.p1.y,  m3DView.r.p2.x,  m3DView.r.p2.y);
          ImGui::Text("3D  Aspect:    < %f,  %f>",               aspect3D.x, aspect3D.y);
          ImGui::Unindent();
        }
        ImGui::EndChild();
        ImGui::PopClipRect();
        ImGui::PopStyleColor();
      }

    if(mFirstFrame || isinf(mCamera.pos.z) || isnan(mKeyFrameUI->view2DInit()) || isnan(mKeyFrameUI->view2DInit()))
      {
        resetViews(); mKeyFrameUI->setInitView3D(mCamera.desc); mKeyFrameUI->setInitView2D(mSimView2D);
        mFirstFrame = false;
      }

    

    if(mImGuiDemo)          // show imgui demo window    (Alt+Shift+D)
      { ImGui::ShowDemoWindow(&mImGuiDemo); }
    if(ftDemo && mFontDemo) // show FreeType test window (Alt+Shift+F)
      { ftDemo->ShowFontsOptionsWindow(&mFontDemo); mParams.op.lockViews = (mImGuiDemo || mFontDemo); }
    
    ImGui::PopFont();
    ImGui::PopStyleVar();
  }
  ImGui::End();
  ImGui::PopStyleVar(4);
  ImGui::PopStyleColor(1);

  if(mClosing) { glfwSetWindowShouldClose(mWindow, GLFW_TRUE); }
}


void SimWindow::postRender() // (CudaVbo vectors)
{
  // if(mVBuffer3D.allocated())
  //   {
  //     glDisable(GL_DEPTH_TEST);
  //     mVBuffer3D.glShader->setUniform("VP", mCamera.glVP);
  //     mVBuffer3D.draw();
  //   }
}







////////////////////////////////////////////////
//// RENDER TO FILE ////////////////////////////
////////////////////////////////////////////////

void SimWindow::initGL()
{
  if(mGlTexSize != mParams.op.outSize || mGlAlpha != mParams.op.writeAlpha)
    {
      std::cout << "==== GL tex size: " << mGlTexSize << " --> " << mParams.op.outSize <<  ")...\n";
      cleanupGL();

      mGlTexSize = mParams.op.outSize;
      mGlAlpha   = mParams.op.writeAlpha;

      std::cout << "== Initializing GL resources...\n";
      // create framebuffer
      glGenFramebuffers(1, &mRenderFB);
      glBindFramebuffer(GL_FRAMEBUFFER, mRenderFB);
      // create texture
      glGenTextures(1, &mRenderTex);
      glBindTexture(GL_TEXTURE_2D, mRenderTex);
      //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texSize.x, texSize.y, 0, GL_RGB, GL_FLOAT, 0);
      glTexImage2D(GL_TEXTURE_2D, 0, (mGlAlpha ? GL_RGBA : GL_RGB), mGlTexSize.x, mGlTexSize.y, 0, GL_RGB, GL_FLOAT, 0);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      // set as color attachment
      glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, mRenderTex, 0);
      glDrawBuffers(1, mDrawBuffers);
      if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        { std::cout << "====> ERROR: Failed to create framebuffer for rendering ImGui text/etc. onto texture.\n"; return; }
      else
        { std::cout << "== Framebuffer success!\n"; }

      // initialize texture
      glViewport(0, 0, mGlTexSize.x, mGlTexSize.y);
      glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      glBindTexture(GL_TEXTURE_2D, 0);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      std::cout << "== (done)\n";
    }
}
void SimWindow::cleanupGL()
{
  if(mGlTexSize > 0)
    {
      std::cout << "== Cleaning up GL resources...\n";
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glBindTexture(GL_TEXTURE_2D, 0);
      glDeleteFramebuffers(1, &mRenderFB); mRenderFB  = 0;
      glDeleteTextures(1, &mRenderTex); mRenderTex = 0;
      mGlTexSize = Vec2i(0,0);
      std::cout << "== (done)\n";
    }
}

void SimWindow::renderToFile()
{
  if(mParams.op.active && mNewFrameOut)
    {
      initGL();
      ImGui_ImplOpenGL3_NewFrame();
      ImGuiIO &io = ImGui::GetIO();
      io.DisplaySize             = Vec2f(mGlTexSize);
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
        ImGui::SetNextWindowSize(Vec2f(mGlTexSize));
        ImGui::PushFont(mainFont);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0); // square windows by default
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,  0);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,  Vec2f(0, 0));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, Vec4f(0.0f, 0.0f, 0.0f, 1.0f));
        if(ImGui::Begin("##renderView", nullptr, wFlags)); // ImGui window covering full application window
        {
          ImGui::SetCursorScreenPos(Vec2f(0.0f, 0.0f));
          ImGui::BeginGroup();
          render(Vec2f(mGlTexSize), "offline");
          ImGui::EndGroup();
        }
        ImGui::End(); ImGui::PopStyleVar(3); ImGui::PopStyleColor(); ImGui::PopFont();
      }

      // render to ImGui frame to separate framebuffer
      glUseProgram(0); ImGui::Render();
      glBindFramebuffer(GL_FRAMEBUFFER, mRenderFB);
      glViewport(0, 0, mGlTexSize.x, mGlTexSize.y);
      glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

      // load frame data for file writing
      if(!mFrameWriter) { mFrameWriter = new FrameWriter(&mParams.op); mFileOutUI->setWriter(mFrameWriter); }
      
      FrameOut *frame = mFrameWriter->get();
      if(frame)
        {
          // copy fluid texture data to host frame data
          glBindTexture(GL_TEXTURE_2D, mRenderTex);
          glReadPixels(0, 0, mGlTexSize.x, mGlTexSize.y, (mGlAlpha ? GL_RGBA : GL_RGB), GL_UNSIGNED_BYTE, (GLvoid*)frame->raw);
          glBindTexture(GL_TEXTURE_2D, 0);
          // push frame
          frame->f = mInfo.frame;
          mFrameWriter->push(frame);
          mNewFrameOut = false;
        }
      else
        { std::cout << "====> WARNING(SimWindow::renderToFile): Could not get frame from FrameWriter (frame may be skipped (?))\n"; }
      
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
}

