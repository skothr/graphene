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
#define CFV2 typename DimType<CFT, 2>::VECTOR_T
#define CFV3 typename DimType<CFT, 3>::VECTOR_T
#define CFV4 typename DimType<CFT, 4>::VECTOR_T
#define STATE_BUFFER_SIZE 1

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

#define SIM_VIEW_RESET_INTERNAL_PADDING 0.1f // ratio of field size in sim space view

// colors
#define SIM_BG_COLOR  Vec4f(0.08, 0.08, 0.08, 1.0) // color of background behind field

// overlay colors
#define X_COLOR Vec4f(0.0f, 1.0f, 0.0f, 0.5f)
#define Y_COLOR Vec4f(1.0f, 0.0f, 0.0f, 0.5f)
#define Z_COLOR Vec4f(0.1f, 0.1f, 1.0f, 0.8f) // (bumped up -- hard to see over dark background)
#define OUTLINE_COLOR Vec4f(1,1,1,0.5f) // field outline
#define GUIDE_COLOR   Vec4f(1,1,1,0.3f) // pen input guides
#define RADIUS_COLOR  Vec4f(1,1,1,0.4f) // intersecting radii ghosts


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
  bool debug   = false;        // toggle debugging (e.g. syncs device between steps for error checking)
  bool verbose = false;        // toggle verbose printout

  FieldParams<CFT>       cp; // cuda field params
  RenderParams<CFT>      rp; // cuda render params
  VectorFieldParams<CFT> vp; // vector draw params

  ////////////////////////////////////////
  // microstepping
  ////////////////////////////////////////
  int uSteps  = 1;    // number of microsteps performed between each rendered frame
  CFT dtFrame = 0.1; // total timestep over one frame

  ////////////////////////////////////////
  // params for rendering to files
  ////////////////////////////////////////
  bool outAlpha       = false;     // if true, outputs an alpha channel if supported by file format
  int  pngCompression = 10;        // PNG compression level (if path ends with .png)
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

  // window callbacks
  std::map<int, bool>        mKeysDown;
  std::map<int, std::string> mKeyParamNames;
  static void windowCloseCallback(GLFWwindow *window);
  static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
  
  GLFWwindow *mWindow    = nullptr;
  bool        mClosing   = false;
  Vec2f mFrameSize;   // size of last/current window framebuffer (changes on resize)
  Vec2f mDisplaySize; // 

  bool mImGuiDemo     = false; // run the ImGui demo popup (provides working widget examples and debug tools)
  bool mFileRendering = false; // rendering simulation frames to image files
  bool mLockViews     = false; // prevent any user input while rendering to file
  bool mNewFrame      = false; // new rendered frame available for output
  
  // simulation
  SimParams  mParams;
  SimInfo    mInfo;
  Units<CFT> mUnits;

  CFT mSingleStepMult = 0.0; // used to single step with up/down arrow keys while paused
  
  CudaTexture  mEMTex;
  CudaTexture  mMatTex;
  CudaTexture  m3DTex;
  CudaVBO      mVecBuffer;
  std::deque<FieldBase*> mStates; // N-buffered states (e.g. for Runge-Kutta integration)

  ImDrawList *mFieldDrawList = nullptr;

  // UI
  std::vector<SettingBase*> mSettings;
  FieldInterface<CFT>   *mFieldUI   = nullptr; // base field settings
  UnitsInterface<CFT>   *mUnitsUI   = nullptr; // global units
  DrawInterface<CFT>    *mDrawUI    = nullptr; // settings for drawing in signals/materials
  DisplayInterface<CFT> *mDisplayUI = nullptr; // settings for rendering/displaying simulation
  SettingForm           *mFileOutUI = nullptr; // settings for outputting simulation frames to image files
  SettingForm           *mOtherUI   = nullptr; // other settings (misc)
  TabMenu *mTabs = nullptr;

  Camera<CFT> mCamera;
  Rect2f mSimView2D; // view in sim space
  ScreenView mEMView;
  ScreenView mMatView;
  ScreenView m3DView;
  ScreenView m3DGlView;
  
  Vec2f mMouseSimPos;
  CFV3  mSigMPos; // 3D pos of active signal pen
  CFV3  mMatMPos; // 3D pos of active material pen

  EMField<CFT> mSignalField;
  bool mNewSignal = false;
  
  Vec2f simToScreen2D(const Vec2f &pSim,    const Rect2f &simView, const Rect2f &screenView, bool vector=false);
  Vec2f screenToSim2D(const Vec2f &pScreen, const Rect2f &simView, const Rect2f &screenView, bool vector=false);
  // (to use X and Y components of 3D vectors)
  Vec2f simToScreen2D(const Vec3f &pSim,    const Rect2f &simView, const Rect2f &screenView, bool vector=false);
  Vec2f screenToSim2D(const Vec3f &pScreen, const Rect2f &simView, const Rect2f &screenView, bool vector=false);

  void drawRect3D(ScreenView &view, const Vec3f &p0, const Vec3f &p1, const Vec4f &color);
  void drawEllipse3D(ScreenView &view, const Vec3f &center, const Vec3f &radius, const Vec4f &color);
  
  void drawRect2D(ScreenView &view, const Vec2f &p0, const Vec2f &p1, const Vec4f &color);
  void drawEllipse2D(ScreenView &view, const Vec2f &center, const Vec2f &radius, const Vec4f &color);
  
  void resetViews();
  void cudaRender(FieldParams<CFT> &cp);
  
  void handleInput2D(ScreenView &view);
  void handleInput3D(ScreenView &view);
  void drawVectorField2D(const Rect2f &sr);
  
  void draw2DOverlay(ScreenView &view);
  void draw3DOverlay(ScreenView &view);
      
  double calcFps();

  // rendering to file
  // images output as: [mBaseDir]/[mSimName]/[mSimName]-[mParams.frame].png
  std::string mBaseDir     = RENDER_BASE_PATH;
  std::string mImageDir    = mBaseDir + "/" + mParams.simName;
  
  SettingForm *mFileOutSettings = nullptr;
  CudaTexture  m3DGlTex;
  
  GLuint mRenderFB       = 0;
  GLuint mRenderTex      = 0;
  GLenum mDrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
  Vec2i  mGlTexSize      = Vec2i(0,0);
  bool   mGlAlpha        = true;
  unsigned char *mTexData  = nullptr;
  unsigned char *mTexData2 = nullptr;
  
  void initGL(const Vec2i &texSize);
  void cleanupGL();
  void render(const Vec2f &frameSize=Vec2f(0,0), const std::string &id="");
  void handleInput(const Vec2f &frameSize=Vec2f(0,0), const std::string &id="");

  bool checkBaseRenderPath();
  bool checkSimRenderPath();
  bool fileOutInterface();
  
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
    
  bool init();
  void cleanup();
  void quit()          { mClosing = true; }
  bool closing() const { return mClosing; }

  bool resizeFields(const Vec3i &sz);

  void resetSignals();
  void resetMaterials();
  void resetSim();
  void togglePause();
    
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









// shape helper functions

//  (NOTE: hacky transformations... TODO: improve/optimize)
inline void SimWindow::drawRect3D(ScreenView &view, const Vec3f &p0, const Vec3f &p1, const Vec4f &color)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  Vec2f aspect = Vec2f(view.r.aspect(), 1.0); if(aspect.x < 1.0) { aspect.y = 1.0/aspect.x; aspect.x = 1.0; }
  Vec2f vSize = view.r.size();
  Vec2f aOffset = (aspect.x > 1 ?
                   Vec2f(vSize.x/aspect.x - vSize.x, 0.0f) :
                   Vec2f(0.0f, vSize.y/aspect.y - vSize.y))/2.0f;

  // rect points
  Vec3f W000  = p0;
  Vec3f W001  = Vec3f(p0.x, p0.y, p1.z);
  Vec3f W010  = Vec3f(p0.x, p1.y, p0.z);
  Vec3f W011  = Vec3f(p0.x, p1.y, p1.z);
  Vec3f W100  = Vec3f(p1.x, p0.y, p0.z);
  Vec3f W101  = Vec3f(p1.x, p0.y, p1.z);
  Vec3f W110  = Vec3f(p1.x, p1.y, p0.z);
  Vec3f W111  = p1;
  // transform (NOTE: hacky... TODO: improve/optimize)
  bool C000 = false; bool C001 = false; bool C010 = false; bool C011 = false;
  bool C100 = false; bool C101 = false; bool C110 = false; bool C111 = false;
  Vec2f S000 = view.r.p1 + (Vec2f(1,1) - mCamera.worldToView(W000, aspect, &C000)) * vSize/aspect - aOffset;
  Vec2f S001 = view.r.p1 + (Vec2f(1,1) - mCamera.worldToView(W001, aspect, &C001)) * vSize/aspect - aOffset;
  Vec2f S010 = view.r.p1 + (Vec2f(1,1) - mCamera.worldToView(W010, aspect, &C010)) * vSize/aspect - aOffset;
  Vec2f S011 = view.r.p1 + (Vec2f(1,1) - mCamera.worldToView(W011, aspect, &C011)) * vSize/aspect - aOffset;
  Vec2f S100 = view.r.p1 + (Vec2f(1,1) - mCamera.worldToView(W100, aspect, &C100)) * vSize/aspect - aOffset;
  Vec2f S101 = view.r.p1 + (Vec2f(1,1) - mCamera.worldToView(W101, aspect, &C101)) * vSize/aspect - aOffset;
  Vec2f S110 = view.r.p1 + (Vec2f(1,1) - mCamera.worldToView(W110, aspect, &C110)) * vSize/aspect - aOffset;
  Vec2f S111 = view.r.p1 + (Vec2f(1,1) - mCamera.worldToView(W111, aspect, &C111)) * vSize/aspect - aOffset;
                
  // XY plane (front -- 0XX)
  if(!C000 || !C001) { drawList->AddLine(S000, S001, ImColor(color), 1.0f); }
  if(!C001 || !C011) { drawList->AddLine(S001, S011, ImColor(color), 1.0f); }
  if(!C011 || !C010) { drawList->AddLine(S011, S010, ImColor(color), 1.0f); }
  if(!C010 || !C000) { drawList->AddLine(S010, S000, ImColor(color), 1.0f); }
  // XY plane (back  -- 1XX)
  if(!C100 || !C101) { drawList->AddLine(S100, S101, ImColor(color), 1.0f); }
  if(!C101 || !C111) { drawList->AddLine(S101, S111, ImColor(color), 1.0f); }
  if(!C111 || !C110) { drawList->AddLine(S111, S110, ImColor(color), 1.0f); }
  if(!C110 || !C100) { drawList->AddLine(S110, S100, ImColor(color), 1.0f); }
  // Z connections   -- (0XX - 1XX)
  if(!C000 || !C100) { drawList->AddLine(S000, S100, ImColor(color), 1.0f); }
  if(!C001 || !C101) { drawList->AddLine(S001, S101, ImColor(color), 1.0f); }
  if(!C011 || !C111) { drawList->AddLine(S011, S111, ImColor(color), 1.0f); }
  if(!C010 || !C110) { drawList->AddLine(S010, S110, ImColor(color), 1.0f); }
}


inline void SimWindow::drawEllipse3D(ScreenView &view, const Vec3f &center, const Vec3f &radius, const Vec4f &color)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  float S = 2.0*tan(mCamera.fov/2.0f*M_PI/180.0f);
  Vec2f aspect = Vec2f(view.r.aspect(), 1.0); if(aspect.x < 1.0) { aspect.y = 1.0/aspect.x; aspect.x = 1.0; }
  Vec2f vSize = view.r.size();
  Vec2f aOffset = (aspect.x > 1 ?
                   Vec2f(vSize.x/aspect.x - vSize.x, 0.0f) :
                   Vec2f(0.0f, vSize.y/aspect.y - vSize.y))/2.0f;
  for(int i = 0; i < 32; i++)
    {
      float a0 = 2.0f*M_PI*(i/32.0f);
      float a1 = 2.0f*M_PI*((i+1)/32.0f);
      Vec3f Wp0 = center + ((mCamera.right*cos(a0) - mCamera.up*sin(a0))*S*to_cuda(radius));
      Vec3f Wp1 = center + ((mCamera.right*cos(a1) - mCamera.up*sin(a1))*S*to_cuda(radius));
      bool Cp0 = false; bool Cp1 = false;
      Vec2f Sp0 = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(Wp0, aspect, &Cp0)) * view.r.size()/aspect - aOffset;
      Vec2f Sp1 = view.r.p1 + (Vec2f(1,1)-mCamera.worldToView(Wp1, aspect, &Cp1)) * view.r.size()/aspect - aOffset;
      drawList->AddLine(Sp0, Sp1, ImColor(color), 1);
    }
}


inline void SimWindow::drawRect2D(ScreenView &view, const Vec2f &p0, const Vec2f &p1, const Vec4f &color)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();  
  Vec2f Wp00 = Vec2f(p0.x, p0.y);
  Vec2f Wp01 = Vec2f(p0.x, p1.y);
  Vec2f Wp10 = Vec2f(p1.x, p0.y);
  Vec2f Wp11 = Vec2f(p1.x, p1.y);
  Vec2f Sp00 = simToScreen2D(Wp00, mSimView2D, view.r);
  Vec2f Sp01 = simToScreen2D(Wp01, mSimView2D, view.r);
  Vec2f Sp10 = simToScreen2D(Wp10, mSimView2D, view.r);
  Vec2f Sp11 = simToScreen2D(Wp11, mSimView2D, view.r);
  drawList->AddLine(Sp00, Sp01, ImColor(color), 1);
  drawList->AddLine(Sp01, Sp11, ImColor(color), 1);
  drawList->AddLine(Sp11, Sp10, ImColor(color), 1);
  drawList->AddLine(Sp10, Sp00, ImColor(color), 1);
}
inline void SimWindow::drawEllipse2D(ScreenView &view, const Vec2f &center, const Vec2f &radius, const Vec4f &color)
{
  ImDrawList *drawList = ImGui::GetWindowDrawList();  
  for(int i = 0; i < 32; i++)
    {
      float a0 = 2.0f*M_PI*(i/32.0f);
      float a1 = 2.0f*M_PI*((i+1)/32.0f);
      Vec2f Wp0 = center + Vec2f(cos(a0), -sin(a0))*radius;
      Vec2f Wp1 = center + Vec2f(cos(a1), -sin(a1))*radius;
      Vec2f Sp0 = simToScreen2D(Wp0, mSimView2D, view.r);
      Vec2f Sp1 = simToScreen2D(Wp1, mSimView2D, view.r);
      drawList->AddLine(Sp0, Sp1, ImColor(color), 1);
    }
}



#endif // SIM_WINDOW_HPP
