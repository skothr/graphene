#ifndef SIM_WINDOW_HPP
#define SIM_WINDOW_HPP

#include <chrono>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <map>
#include <deque>

#include "vector.hpp"
#include "rect.hpp"
#include "field.hpp"
#include "mathParser.hpp"
#include "units.hpp"
#include "physics.h"
#include "raytrace.h"


#define FPS_UPDATE_INTERVAL 0.1 // FPS update interval (seconds)
#define CLOCK_T std::chrono::steady_clock
#define RENDER_BASE_PATH "./rendered"

#define CFT float // field base type (float/double (/int?))
#define STATE_BUFFER_SIZE 4

// font settings
#define SMALL_FONT_HEIGHT 13.0f
#define MAIN_FONT_HEIGHT  14.0f
#define TITLE_FONT_HEIGHT 19.0f
#define SUPER_FONT_HEIGHT 11.0f
#define FONT_OVERSAMPLE   8
#define FONT_PATH "./res/fonts/"
#define FONT_NAME "UbuntuMono"
#define FONT_PATH_REGULAR     (FONT_PATH FONT_NAME "-R.ttf" )
#define FONT_PATH_ITALIC      (FONT_PATH FONT_NAME "-RI.ttf")
#define FONT_PATH_BOLD        (FONT_PATH FONT_NAME "-B.ttf" )
#define FONT_PATH_BOLD_ITALIC (FONT_PATH FONT_NAME "-BI.ttf")

// sizes
#define SETTINGS_LABEL_COL_W 180.0
#define SETTINGS_INPUT_COL_W 300.0
#define SETTINGS_TOTAL_W (SETTINGS_LABEL_COL_W + SETTINGS_INPUT_COL_W + 10)

#define SIM_VIEW_RESET_INTERNAL_PADDING 0.05f // ratio of field size in sim space view

// forward declarations
struct ImFont;
struct ImFontConfig;
struct ImDrawList;
class  SettingBase;
class  SettingForm;
class  TabMenu;

// parameters for overall simulation
struct SimParams
{
  ////////////////////////////////////////
  // flags
  ////////////////////////////////////////
  bool debug         = false;        // toggle debugging (e.g. syncs device between steps for error checking)
  bool verbose       = false;        // toggle verbose printout
  bool vsync         = false;        // vertical sync refresh
  bool running       = false;        // play/pause
  
  ////////////////////////////////////////
  // physics
  ////////////////////////////////////////
  bool  updateQ      = false;        // toggle charge field update step
  bool  updateE      = false;        // toggle electric field update step
  bool  updateB      = false;        // toggle magnetic field update step
  
  int3  fieldRes = int3{256, 256, 4}; // desired resolution (number of cells) of charge field
  int2  texRes   = int2{1024, 1024};  // desired resolution (number of cells) of rendered texture

  float3 fieldPos  = float3{0.0f, 0.0f, 0.0f}; // (m) 3D position of field (min index to max index)

  float dt = 0.01f; // physics timestep
  ChargeParams cp = ChargeParams(true,                      // boundary reflect
                                 dt,                        // dt (overwritten by SimParams.dt)
                                 float3{0.1f, 0.1f, 0.1f},  // cell size
                                 float3{0.0f, 0.0f, 0.0f}); // border size (?)
  ////////////////////////////////////////
  // microstepping
  ////////////////////////////////////////
  int      uSteps  = 1;  // number of microsteps performed between each rendered frame
  float    dtFrame = dt; // total timestep over one frame

  ////////////////////////////////////////
  // initial conditions for reset
  ////////////////////////////////////////
    
  std::string initQPStr  = "0"; // "1";
  std::string initQNStr  = "0"; // "1-cos((r^2)/500)";
  std::string initQPVStr = "0";
  std::string initQNVStr = "0";
  std::string initEStr   = "cos((len(r)^2)/512)";
  std::string initBStr   = "sin(t*137)";

  Expression<float>  *initQPExpr  = nullptr;
  Expression<float>  *initQNExpr  = nullptr;
  Expression<float3> *initQPVExpr = nullptr;
  Expression<float3> *initQNVExpr = nullptr;
  Expression<float3> *initEExpr   = nullptr;
  Expression<float3> *initBExpr   = nullptr;

  ////////////////////////////////////////
  // on-screen rendering
  ////////////////////////////////////////
  
  EmRenderParams rp;          // field render params
  bool showEMField    = true;
  bool showMatField   = false;
  bool show3DField    = true;
  bool drawAxes       = false; // 3D axes (WIP)
  
  bool drawVectors     = false; // draws vector field on screen
  bool borderedVectors = true;  // uses fancy bordered polygons instead of standard GL_LINES (NOTE: slower)   TODO: optimize with VBO
  bool smoothVectors   = true;  // uses bilinear interpolation, centering at samples mouse instead of exact cell centers

  int vecMRadius   = 64;    // draws vectors for cells within radius around mouse
  int vecSpacing   = 1;     // base spacing 
  int vecCRadius   = 1024;  // only draws maximum of this radius number of vectors, adding spacing
  
  float vecMultE   = 10.0f; // E length multiplier
  float vecMultB   = 10.0f; // B length multiplier

  float vecLineW   = 0.1f;  // line width
  float vecAlpha   = 0.25f;  // line opacity
  
  float vecBorderW = 0.0f;  // border width
  float vecBAlpha  = 0.0f;  // border opacity
  
  ////////////////////////////////////////
  // mouse interaction
  ////////////////////////////////////////
  SignalPen<float>   signalPen;
  MaterialPen<float> materialPen;
  int zLayer2D = 0;
  
  ////////////////////////////////////////
  // params for rendering to files
  ////////////////////////////////////////
  // RenderParams render;             // handles flags and multipliers dictating how the simulation state is rendered
  bool outAlpha       = false;     // if true, outputs an alpha channel if supported by file format
  int  pngCompression = 4;         // PNG compression level (if path ends with .png)
  int2 outSize = int2{1920, 1080}; // output video size
  std::string simName = "unnamed"; // directory name/file prefix
  std::string outExt  = ".png";    // image file extension
};


struct SimInfo
{
  float t     = 0.0f; // simulation time passed since initial state
  float fps   = 0.0f; // render fps
  int   frame = 0;    // number of frames rendered since initial state
  int   uStep = 0;    // keeps track of microsteps betweeen frames
};

struct ScreenView
{
  Rect2f r;
  bool   hovered      = false;
  bool   clicked      = false;
  bool   rightClicked = false;
  Vec2f  clickPos;
  Vec3f  mposSim;
};


class SimWindow
{
private:
  bool mInitialized = false;

  CLOCK_T::time_point tNow;  // FPS current frame time
  CLOCK_T::time_point tLast; // FPS last frame time
  double dt      = 0.0;      // FPS time difference
  double tDiff   = 0.0;      // FPS time difference accumulator
  int    nFrames = 0;        // number of frames this interval
  double fpsLast = 0.0;      // previous FPS value

  float mSingleStepMult = 0.0f; // for single stepping while paused

  GLFWwindow *mWindow    = nullptr;
  bool        mClosing   = false;
  Vec2f mFrameSize;
  Vec2f mDisplaySize;
  Vec2f mSettingsSize;

  bool mImGuiDemo       = false;
  bool mFileRendering   = false;
  bool mNewFrame        = false;
  bool mSimHovered      = false;
  
  bool mEMClicked       = false;
  bool mEMRightClicked  = false;
  bool mMatClicked      = false;
  bool mMatRightClicked = false;
  bool m3DClicked       = false;
  bool m3DRightClicked  = false;
    
  // simulation
  SimParams  mParams;
  SimInfo    mInfo;
  Units<CFT> mUnits;
  std::deque<FieldBase*> mStates; // N-buffered states (e.g. for Runge-Kutta integration)
  CudaTexture mEMTex;
  CudaTexture mMatTex;
  CudaTexture m3DTex;

  ImDrawList *mFieldDrawList = nullptr;

  //                          1/R    1/R^2  theta  sin(t)  cos(t)
  std::vector<bool> Qopt   = {false, false, false, false,  false};
  std::vector<bool> QPVopt = {false, false, false, false,  false};
  std::vector<bool> QNVopt = {false, false, false, false,  false};
  std::vector<bool> Eopt   = {false, false, false, true,   false};
  std::vector<bool> Bopt   = {false, false, false, false,  true };
  std::vector<SettingBase*> mSettings;
  SettingForm *mSettingFormOld = nullptr;
  UnitsInterface<CFT> *mUnitsForm = nullptr;
  TabMenu *mTabs = nullptr;

  Camera<double> mCamera;
  Rect2f mSimView2D; // view in sim space
  ScreenView mEMView;
  ScreenView mMatView;
  ScreenView m3DView;
  
  Vec2f  mMouseSimPos;

  ChargeField<float> mSignalField;
  bool mNewSignal = false;
  
  Vec2f simToScreen2D(const Vec2f &pSim,    const Rect2f &simView, const Rect2f &screenView, bool vector=false);
  Vec2f screenToSim2D(const Vec2f &pScreen, const Rect2f &simView, const Rect2f &screenView, bool vector=false);
  // (to use X and Y components of 3D vectors)
  Vec2f simToScreen2D(const Vec3f &pSim,    const Rect2f &simView, const Rect2f &screenView, bool vector=false);
  Vec2f screenToSim2D(const Vec3f &pScreen, const Rect2f &simView, const Rect2f &screenView, bool vector=false);

  // ?
  // Vec3f simToPhysical(const Vec3f &fp, const Vec3f *p0=nullptr); // if p0 is null, treated as a vector (e.g. size)
  // Vec3f physicalToSim(const Vec3f &pp, const Vec3f *p0=nullptr); // if p0 is null, treated as a vector (e.g. size)

  void resetViews();
  
  void handleInput    (ScreenView &view);
  void handleInput3D  (ScreenView &view);
  void drawVectorField(const Rect2f &sr);
    
  std::map<int, bool>        mKeysDown;
  std::map<int, std::string> mKeyParamNames;

  static void windowCloseCallback(GLFWwindow *window);
  static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    
  double calcFps();

  // rendering to file
  // images output as: [mBaseDir]/[mSimName]/[mSimName]-[mParams.frame].png
  std::string mBaseDir     = RENDER_BASE_PATH;
  std::string mImageDir    = mBaseDir + "/" + mParams.simName;
    
  GLuint mRenderFB       = 0;
  GLuint mRenderTex      = 0;
  GLenum mDrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
  Vec2i  mGlTexSize      = Vec2i(0,0);
  bool   mGlAlpha        = true;
  unsigned char *mTexData  = nullptr;
  unsigned char *mTexData2 = nullptr;

  // CUDA fill expressions
  CudaExpression<float>  *mFillQP  = nullptr;
  CudaExpression<float>  *mFillQN  = nullptr;
  CudaExpression<float3> *mFillQPV = nullptr;
  CudaExpression<float3> *mFillQNV = nullptr;
  CudaExpression<float3> *mFillE   = nullptr;
  CudaExpression<float3> *mFillB   = nullptr;
  
  void initGL(const Vec2i &texSize);
  void cleanupGL();

  bool checkBaseRenderPath();
  bool checkSimRenderPath();
    
public:
  // fonts (public)
  ImFontConfig *fontConfig = nullptr;
  ImFont *smallFont  = nullptr; ImFont *smallFontB  = nullptr; ImFont *smallFontI = nullptr; ImFont *smallFontBI = nullptr;
  ImFont *mainFont   = nullptr; ImFont *mainFontB   = nullptr; ImFont *mainFontI  = nullptr; ImFont *mainFontBI  = nullptr;
  ImFont *titleFont  = nullptr; ImFont *titleFontB  = nullptr; ImFont *titleFontI = nullptr; ImFont *titleFontBI = nullptr;
  ImFont *superFont  = nullptr; ImFont *superFontB  = nullptr; ImFont *superFontI = nullptr; ImFont *superFontBI = nullptr;
  
  void keyPress(int mods, int key, int action);
  
  SimWindow(GLFWwindow *window);
  ~SimWindow();
    
  bool init();  //const SimParams<float2> &p=SimParams<float2>());
  void cleanup();
  void quit()          { mClosing = true; }
  bool closing() const { return mClosing; }

  bool resizeFields(const Vec3i &sz);

  void resetSignals();
  void resetMaterials();
  void resetSim();
  void togglePause();
  //SimParams<float2>* params() { return &mParams; }
    
  void draw(const Vec2f &frameSize=Vec2f(0,0));
  void update();
    
  void renderToFile();
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
////
////   1-SCREEN <--  <view rect>  <-- 2-SIM(0.0, 1.0) <-- 3-PHYSICAL(relative to some point + basis vector)
////                                         |
////                                        vvv
////                                 2.5-FIELD(0,fSize)
////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
// Conversion between screen space and sim space
//// p0 --> optional screen-space offset
inline Vec2f SimWindow::simToScreen2D(const Vec2f &pSim, const Rect2f &simView, const Rect2f &screenView, bool vector)
{
  Vec2f pScreen = (pSim-simView.p1*(vector?0:1)) * (screenView.size()/simView.size());
  if(!vector){ pScreen = Vec2f(screenView.p1.x + pScreen.x, screenView.p2.y - pScreen.y); }
  return pScreen;  
}
/////////////////////////////////////////////////////////////////////////////////////////////////
inline Vec2f SimWindow::screenToSim2D(const Vec2f &pScreen, const Rect2f &simView, const Rect2f &screenView, bool vector)
{
  Vec2f pSim = (pScreen-screenView.p1*(vector?0:1)) * (simView.size()/screenView.size());
  if(!vector){ pSim = Vec2f(simView.p1.x + pSim.x, simView.p2.y - pSim.y); }
  return pSim;
}
/////////////////////////////////////////////////////////////////////////////////////////////////

// (for 3D vectors, treated as 2D)
inline Vec2f SimWindow::simToScreen2D(const Vec3f &pSim,    const Rect2f &simView, const Rect2f &screenView, bool vector)
{ return simToScreen2D(Vec2f(pSim.x,    pSim.y),    simView, screenView, vector); }
inline Vec2f SimWindow::screenToSim2D(const Vec3f &pScreen, const Rect2f &simView, const Rect2f &screenView, bool vector)
{ return screenToSim2D(Vec2f(pScreen.x, pScreen.y), simView, screenView, vector); }













#endif // SIM_WINDOW_HPP
