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
#include "render.cuh"
#include "display.hpp"
#include "cuda-vbo.h"

#define FPS_UPDATE_INTERVAL 0.1 // FPS update interval (seconds)
#define CLOCK_T std::chrono::steady_clock
#define RENDER_BASE_PATH "./rendered"

#define CFT float // field base type (float/double (/int?))
#define STATE_BUFFER_SIZE 2

// font settings
#define SMALL_FONT_HEIGHT 13.0f
#define MAIN_FONT_HEIGHT  14.0f
#define TITLE_FONT_HEIGHT 19.0f
#define SUPER_FONT_HEIGHT 10.5f
#define FONT_OVERSAMPLE   1
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
template<typename T> class DrawInterface;
template<typename T> class DisplayInterface;

// parameters for overall simulation
struct SimParams
{
  ////////////////////////////////////////
  // flags
  ////////////////////////////////////////
  bool debug         = false;        // toggle debugging (e.g. syncs device between steps for error checking)
  bool verbose       = false;        // toggle verbose printout

  ////////////////////////////////////////
  // initial conditions for reset
  ////////////////////////////////////////
  FieldParams<CFT> cp;   // cuda field params

  ////////////////////////////////////////
  // on-screen rendering
  ////////////////////////////////////////
  RenderParams<CFT>     rp; // field render params
  VectorFieldParams<CFT> vp; // vector draw params


  
  ////////////////////////////////////////
  // microstepping
  ////////////////////////////////////////
  int uSteps  = 1;    // number of microsteps performed between each rendered frame
  CFT dtFrame = 0.1; // total timestep over one frame
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
  CFT t     = 0.0f; // simulation time passed since initial state
  CFT fps   = 0.0f; // render fps
  int frame = 0;    // number of frames rendered since initial state
  int uStep = 0;    // keeps track of microsteps betweeen frames
};

struct ScreenView
{
  Rect2f r;
  bool   hovered       = false;
  bool   leftClicked   = false;
  bool   rightClicked  = false;
  bool   middleClicked = false;
  bool   shiftClick    = false;
  bool   ctrlClick     = false;
  bool   altClick      = false;
  Vec2f           clickPos;
  Vector<CFT, 3>  mposSim;
};


class SimWindow
{
private:
  bool mInitialized = false;
  bool mFirstFrame  = true;

  CLOCK_T::time_point tNow;  // FPS current frame time
  CLOCK_T::time_point tLast; // FPS last frame time
  double dt      = 0.0;      // FPS time difference
  double tDiff   = 0.0;      // FPS time difference accumulator
  int    nFrames = 0;        // number of frames this interval
  double fpsLast = 0.0;      // previous FPS value

  
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
  SimParams    mParams;
  SimInfo      mInfo;
  
  Units<CFT> mUnits;

  float mSingleStepMult = 0.0f; // for single stepping with up/down arrow keys while paused
  
  CudaTexture  mEMTex;
  CudaTexture  mMatTex;
  CudaTexture  m3DTex;
  CudaVBO      mVecBuffer;
  std::deque<FieldBase*> mStates; // N-buffered states (e.g. for Runge-Kutta integration)

  ImDrawList *mFieldDrawList = nullptr;

  std::vector<SettingBase*> mSettings;
  FieldInterface<CFT>   *mFieldUI    = nullptr;
  UnitsInterface<CFT>   *mUnitsUI    = nullptr;
  DrawInterface<CFT>    *mDrawUI     = nullptr;
  DisplayInterface<CFT> *mDisplayUI  = nullptr;
  SettingForm           *mFlagUI     = nullptr;
  TabMenu *mTabs = nullptr;

  Camera<CFT> mCamera;
  Rect2f mSimView2D; // view in sim space
  ScreenView mEMView;
  ScreenView mMatView;
  ScreenView m3DView;
  
  Vec2f  mMouseSimPos;
  float3 mSigMPos; // 3D pos of active signal pen
  float3 mMatMPos; // 3D pos of active material pen

  EMField<CFT> mSignalField;
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
