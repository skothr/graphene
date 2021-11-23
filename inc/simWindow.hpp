#ifndef SIM_WINDOW_HPP
#define SIM_WINDOW_HPP

#include <chrono>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <deque>
#include <filesystem>
namespace fs = std::filesystem;

#include <misc/freetype/imgui_freetype_test.h>

#include "vector.hpp"
#include "rect.hpp"
#include "display.hpp"
#include "screenView.hpp"

#include "keyBinding.hpp"
#include "mathParser.hpp"
#include "field.hpp"
#include "fluid.cuh"
#include "units.hpp"
#include "output.hpp"
#include "camera.hpp"
#include "physics.h"
#include "raytrace.h"
#include "render.cuh"
#include "cuda-texture.cuh"
#include "cuda-vbo.cuh"

#define SETTINGS_SAVE_FILE (fs::current_path() / ".settings.conf")
#define JSON_SPACES 4

#define FPS_UPDATE_INTERVAL 0.1 // FPS update interval (seconds)
#define CLOCK_T std::chrono::steady_clock
#define RENDER_BASE_PATH (fs::current_path() / "rendered")

#define CFT float // field base type (float/double (/int?))
#define CFV2 typename DimType<CFT, 2>::VEC_T
#define CFV3 typename DimType<CFT, 3>::VEC_T
#define CFV4 typename DimType<CFT, 4>::VEC_T
#define STATE_BUFFER_SIZE  2
#define DESTROY_LAST_STATE true //(STATE_BUFFER_SIZE <= 1)

// font settings
#define MAIN_FONT_HEIGHT  14.0f
#define SMALL_FONT_HEIGHT 13.0f
#define TITLE_FONT_HEIGHT 19.0f
#define SUPER_FONT_HEIGHT 10.5f
#define TINY_FONT_HEIGHT   9.0f
#define FONT_OVERSAMPLE   1 //4
#define FONT_PATH (fs::current_path() / "res/fonts/")
#define FONT_NAME "UbuntuMono"
#define FONT_PATH_REGULAR     (FONT_PATH / (FONT_NAME "-R.ttf" ))
#define FONT_PATH_ITALIC      (FONT_PATH / (FONT_NAME "-RI.ttf"))
#define FONT_PATH_BOLD        (FONT_PATH / (FONT_NAME "-B.ttf" ))
#define FONT_PATH_BOLD_ITALIC (FONT_PATH / (FONT_NAME "-BI.ttf"))

// sizes
#define SETTINGS_W 555
#define TIMELINE_H 333

#define SIM_VIEW_RESET_INTERNAL_PADDING 0.1f // ratio of field size in sim space view

// colors
#define SIM_BG_COLOR  Vec4f(0.08, 0.08, 0.08, 1.0) // color of background behind field

// overlay colors
#define X_COLOR Vec4f(0.0f, 1.0f, 0.0f, 0.5f)
#define Y_COLOR Vec4f(1.0f, 0.0f, 0.0f, 0.5f)
#define Z_COLOR Vec4f(0.1f, 0.1f, 1.0f, 0.8f) // (slightly brighter -- hard to see pure blue over dark background)
#define OUTLINE_COLOR Vec4f(1,1,1,0.5f) // field outline
#define GUIDE_COLOR   Vec4f(1,1,1,0.3f) // pen input guides
#define RADIUS_COLOR  Vec4f(1,1,1,0.4f) // intersecting radii ghosts

// forward declarations
struct ImFont;
struct ImFontConfig;
struct ImDrawList;
class  KeyManager;
class  KeyFrameWidget;
class  SettingForm;
class  TabMenu;
class  Toolbar;
class  FrameWriter;
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

  bool limitPhysicsFps = false;
  bool limitRenderFps  = false;
  float maxPhysicsFps  = 24.0f;
  float maxRenderFps   = 24.0f;
  
  bool vsync = false;        // vertical sync
  
  FluidParams<CFT>       cp; // cuda field params
  RenderParams<CFT>      rp; // cuda render params
  VectorFieldParams<CFT> vp; // vector draw params
  OutputParams           op; // file output params

  ////////////////////////////////////////
  // microstepping (not implemented)
  ////////////////////////////////////////
  int uSteps  = 1;   // number of microsteps performed between each rendered frame
  CFT dtFrame = 0.1; // total timestep over one frame
};


struct SimInfo
{
  CFT t     = 0.0f; // simulation time passed since initial state
  CFT fps   = 0.0f; // render fps
  int frame = 0;    // number of frames rendered since initial state
  int uStep = 0;    // keeps track of microsteps betweeen frames
};

struct FVector
{
  Vec3f p0;   // cell sim pos
  Vec3f sp;   // sample position
  Vec3f vV;   // V  sim vector
  Vec3f vQnv; // Qnv sim vector
  Vec3f vQpv; // Qpv sim vector
  Vec3f vE;   // E  sim vector
  Vec3f vB;   // B  sim vector
};


struct FpsCounter
{
  CLOCK_T::time_point tLast;   // last frame time
  double updateInterval = 0.1; // seconds over which to average
  int    nFrames        = 0;   // number of frames this interval
  double dtAcc          = 0.0; // dt accumulator
  double fps            = 0.0; // most recent FPS value

  double update()
  {
    auto tNow  = CLOCK_T::now();
    double dt = std::chrono::duration_cast<std::chrono::nanoseconds>(tNow - tLast).count()/1000000000.0;
    tLast = tNow;

    dtAcc += dt;
    nFrames++;
    if(dtAcc > FPS_UPDATE_INTERVAL)
      {
        fps     = nFrames / dtAcc;
        dtAcc   = 0.0;
        nFrames = 0;
      }
    return fps;
  }

  bool update(double limit, CLOCK_T::time_point tNow=CLOCK_T::now())
  {
    //auto tNow  = CLOCK_T::now();
    double dt = std::chrono::duration_cast<std::chrono::nanoseconds>(tNow - tLast).count()/1000000000.0;
    tLast = tNow;

    int    framesPassed = 0;
    double frameTime = 1.0/limit;
    dtAcc += dt;
    if(limit >= 0.0)
      {
        if(dtAcc >= frameTime)
          {
            framesPassed = 1;
            fps = 1.0 / dtAcc;
            dtAcc = fmod(dtAcc, frameTime);
          }
      }
    else
      {
        update();
        framesPassed = 1;
        // framesPassed = 1;
        // fps = 1.0 / dt;
        // dtAcc = 0.0;
      }
    return framesPassed > 0;
  }
};


class SimWindow
{
private:
  bool mInitialized = false;
  bool mFirstFrame  = true;
  
  // window callbacks
  static void windowCloseCallback(GLFWwindow *window);
  static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
  
  GLFWwindow *mWindow    = nullptr;
  bool        mClosing   = false;
  Vec2f mFrameSize;   // size of last/current window framebuffer (changes on resize)

  bool mImGuiDemo     = false; // run the ImGui demo popup (provides working widget examples and debug tools)
  bool mFontDemo      = false; // run a simple comparison test / config demo for freetype (https://gist.github.com/ocornut/b3a9ecf13502fd818799a452969649ad)
  bool mLockViews     = false; // prevent any user input (for render to file)
  bool mForcePause    = false; // pause all physics and prevent any user input (for open popup focus)
  bool mNewFrameOut   = false; // new simulation frame available
  bool mNewSimFrame   = false; // new rendered frame available for output

  FpsCounter mMainFps;
  FpsCounter mPhysicsFps;
  FpsCounter mRenderFps;
  FpsCounter mUpdateFps;
  
  CFT mSingleStepMult = 0.0; // used to single step with up/down arrow keys while paused
  
  // parameters
  SimParams  mParams;
  Units<CFT> mUnits;
  SimInfo    mInfo;
  FluidParams<CFT> cpCopy;
  
  CudaTexture mEMTex; CudaTexture mMatTex; CudaTexture m3DTex;
  std::deque<FieldBase*> mStates; // N-buffered states (e.g. for RK4 integration (TODO))
  FluidField<CFT> *mTempState = nullptr; // temp state (avoids destroying input source state)

  // accumulation fields for user-added inputs
  Field<float3> *mInputV   = nullptr;
  Field<float>  *mInputP   = nullptr;
  Field<float>  *mInputQn  = nullptr;
  Field<float>  *mInputQp  = nullptr;
  Field<float3> *mInputQnv = nullptr;
  Field<float3> *mInputQpv = nullptr;
  Field<float3> *mInputE   = nullptr;
  Field<float3> *mInputB   = nullptr;

  std::vector<SignalPen<CFT>*> mSigPens;
  std::vector<MaterialPen<CFT>*> mMatPens;
  
  bool mNeedResetViews     = false; // notify CUDA thread to reset views next frame                (default F1)
  bool mNeedResetSim       = false; // notify CUDA thread to reset full sim next frame             (default F5)
  bool mNeedResetSignals   = false; // notify CUDA thread to reset only E/M states next frame      (default F6) Q / E / B
  bool mNeedResetMaterials = false; // notify CUDA thread to reset only material states next frame (default F7) mat 
  bool mNeedResetFluid     = false; // notify CUDA thread to reset only fluid states next frame    (default F8) v / p
  Vec3i mNewResize; // notify CUDA thread to resize fields (<0,0,0> otherwise)
  
  // UI
  ImDrawList *mFieldDrawList = nullptr;
  FieldInterface<CFT>   *mFieldUI    = nullptr; // base field settings
  UnitsInterface<CFT>   *mUnitsUI    = nullptr; // global units
  DrawInterface<CFT>    *mDrawUI     = nullptr; // settings for drawing in signals/materials
  DisplayInterface<CFT> *mDisplayUI  = nullptr; // settings for rendering/displaying simulation
  OutputInterface       *mFileOutUI  = nullptr; // settings for outputting simulation frames to image files
  SettingForm           *mOtherUI    = nullptr; // other settings (misc)
  KeyFrameWidget        *mKeyFrameUI = nullptr; // keyframing widget (timeline/list/etc.)
  
  Toolbar *mSigPenBar  = nullptr; // toolbar for selecting active signal pen   (CTRL)
  Toolbar *mMatPenBar  = nullptr; // toolbar for selecting active material pen (ALT)
  
  TabMenu *mSideTabs   = nullptr; // settings at right side of window
  TabMenu *mBottomTabs = nullptr; // settings at bottom of window

  // input
  KeyManager *mKeyManager = nullptr;
  bool mCaptured = false;

  // views
  Camera<CFT> mCamera;
  Camera<CFT> mGlCamera;
  Rect2f mSimView2D; // view in sim space
  ScreenView<CFT> mEMView;
  ScreenView<CFT> mMatView;
  ScreenView<CFT> m3DView;
  ScreenView<CFT> m3DGlView;

  // TODO: refine...?
  Vec2f mMouseSimPos;
  Vec2f mMouseSimPosLast;
  CFV3  mSigMPos; // 3D pos of active signal pen
  CFV3  mMatMPos; // 3D pos of active material pen


  FrameWriter *mFrameWriter = nullptr;

  
  SignalPen<CFT>*   activeSigPen();
  MaterialPen<CFT>* activeMatPen();
  void addSigPen(SignalPen<CFT> *pen);
  void addMatPen(MaterialPen<CFT> *pen);
  
  void loadSettings(const fs::path &path=SETTINGS_SAVE_FILE);
  void saveSettings(const fs::path &path=SETTINGS_SAVE_FILE);
  
  void cudaRender(FluidParams<CFT> &cp);
  void render(const Vec2f &frameSize=Vec2f(0,0), const std::string &id="main");

  // input
  void handleInput(const Vec2f &frameSize=Vec2f(0,0));
  void handleInput2D(ScreenView<CFT> &view, Rect<CFT>   &simView); // 2D
  void handleInput3D(ScreenView<CFT> &view, Camera<CFT> &camera);  // 3D
  // overlays (debug text, axes, frame boundaries, etc.)
  void draw2DOverlay(const ScreenView<CFT> &view, const Rect<CFT>   &simView); // 2D
  void draw3DOverlay(const ScreenView<CFT> &view, const Camera<CFT> &camera);  // 3D

  ////////////////////////////////////////////////////////////////////////
  //// TODO: separate output rendering class
  ////////////////////////////////////////////////////////////////////////
  CudaTexture m3DGlTex;
  GLuint mRenderFB       = 0;
  GLuint mRenderTex      = 0;
  GLenum mDrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
  Vec2i  mGlTexSize      = Vec2i(0,0);
  bool   mGlAlpha        = true;
  void initGL();
  void cleanupGL();
  ////////////////////////////////////////////////////////////////////////
  
  ////////////////////////////////////////////////////////////////////////
  //// TODO: separate vector drawing class
  ////////////////////////////////////////////////////////////////////////
  std::vector<FVector> mVectorField2D;
  CudaVBO mVBuffer2D; // TODO: load vectors into separate VBO via CUDA
  CudaVBO mVBuffer3D; // TODO: load vectors into separate VBO via CUDA
  //bool mNewFrameVec   = false; // new updated sim frame (recalculate vector field)
  void drawVectorField2D(ScreenView<CFT> &view, const Rect<CFT> &simView);
  void makeVectorField3D();
  ////////////////////////////////////////////////////////////////////////
  
public:
  // fonts (public)
  ImVector<ImWchar>        fontRanges;
  ImFontGlyphRangesBuilder fontBuilder;
  ImFontConfig            *fontConfig = nullptr;
  ImFont *mainFont   = nullptr; ImFont *mainFontB   = nullptr; ImFont *mainFontI  = nullptr; ImFont *mainFontBI  = nullptr;
  ImFont *smallFont  = nullptr; ImFont *smallFontB  = nullptr; ImFont *smallFontI = nullptr; ImFont *smallFontBI = nullptr;
  ImFont *titleFont  = nullptr; ImFont *titleFontB  = nullptr; ImFont *titleFontI = nullptr; ImFont *titleFontBI = nullptr;
  ImFont *superFont  = nullptr; ImFont *superFontB  = nullptr; ImFont *superFontI = nullptr; ImFont *superFontBI = nullptr;
  ImFont *tinyFont   = nullptr; ImFont *tinyFontB   = nullptr; ImFont *tinyFontI  = nullptr; ImFont *tinyFontBI  = nullptr;
  FreeTypeTest *ftDemo = nullptr;
  
  void keyPress(int mods, int key, int action);
  
  SimWindow(GLFWwindow *window);
  ~SimWindow();
  
  bool init();
  void cleanup();
  void quit()          { mClosing = true; }
  bool closing() const { return mClosing; }

  bool isCaptured() const { return mCaptured; }
  bool verbose() const    { return mParams.verbose; }
  bool debug() const      { return mParams.debug; }

  bool resizeFields(const Vec3i &sz, bool cudaThread=true);

  void resetViews(bool cudaThread=true); // NOTE: cudaThread --> if true, assumes this is the main thread with CUDA initialized
  void resetSignals(bool cudaThread=true);
  void resetMaterials(bool cudaThread=true);
  void resetFluid(bool cudaThread=true);
  void resetSim(bool cudaThread=true);
  
  void singleStepField(CFT mult);
  void togglePause(); void toggleDebug(); void toggleVerbose();
  void toggleKeyBindings(); void toggleImGuiDemo(); void toggleFontDemo();

  bool preFrame();   // call in main loop before ImGui::NewFrame() --> updates GPU font textures for font demo
  void draw(const Vec2f &frameSize=Vec2f(0,0));
  void postRender(); // call in main loop after ImGui::EndFrame()  --> cuda VBO (TODO)
  void update();

  bool fileRendering() const { return mParams.op.active; }
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



#endif // SIM_WINDOW_HPP
