#ifndef KEYFRAME_WIDGET_HPP
#define KEYFRAME_WIDGET_HPP

#include <list>
#include <functional>
#include <nlohmann/json_fwd.hpp> // json forward declarations
using json = nlohmann::json;

#include "keyFrame.hpp"

#ifndef CFT
#define CFT float
#endif

struct ImGuiWindow;

// #define KEYFRAME_FRAME_MINSIZE Vec2f(42.0f, 8.0f) // minimum displayed width of one sim frame (x axis, before scaling)
// minimum displayed width of one second in sim time (x axis, before scaling)
#define KEYFRAME_SECOND_MINSIZE Vec2f(60.0f, ImGui::CalcTextSize("X").y+style.FramePadding.y*2.0f) 


struct KeyEventWidget
{
  KeyEventBase *eventStart = nullptr;
  KeyEventBase *eventEnd   = nullptr;
  double clickPos  = -1.0;
  double lastStart = -1.0;
  double length    = -1.0;
  KeyEventWidget(KeyEventBase *es, KeyEventBase *ee=nullptr)
    : eventStart(es), eventEnd(ee) { }
};


// represents a persistent signal source
struct SignalSource
{
  typedef typename cuda_vec<CFT,3>::VT VT3;
  VT3 pos;
  SignalPen<CFT> pen;
  double start  = -1.0;
  double length =  0.0;
  
  // typedef std::pair<double, double> Range;
  // std::vector<Range> onTimes;
};
  
// represents placement of material at some location
struct MaterialPlaced
{
  typedef typename cuda_vec<CFT,3>::VT VT3;
  VT3 pos;
  MaterialPen<CFT> pen;
};




class KeyFrameWidget
{
  typedef typename cuda_vec<CFT,3>::VT VT3;
  
private:
  // SignalPen<CFT>   *mSigPen = nullptr; // global signal pen
  // MaterialPen<CFT> *mMatPen = nullptr; // global material pen
  std::string mActiveSigPen = "";
  std::string mActiveMatPen = "";
  
  // current states
  std::map<std::string, std::vector<SignalSource>> mSources;    // active signal sources
  std::map<std::string, SignalPen<CFT>*>    mSigPens;    // signal pens
  std::map<std::string, MaterialPen<CFT>*>  mMatPens;    // material pens
  std::vector<MaterialPlaced>               mPlaced;     // material placements for next frame
  Rect2f                                    mView2D;     // 2D view for next frame
  CameraDesc<CFT>                           mView3D;     // 3D view for next frame
  std::unordered_map<std::string, std::any> mSettings;   // global settings (TODO)
  
  // initial states (frame 0)
  Rect2f                                    mView2DInit;
  CameraDesc<CFT>                           mView3DInit;
  std::unordered_map<std::string, std::any> mSettingsInit; // TODO

  // event structures
  std::map<KeyEventType, std::vector<KeyEventBase*>> mEvents;
  std::unordered_map<KeyEventBase*, KeyEventWidget*> mEventWidgets;

  KeyEventBase *mLastHovered = nullptr;
  bool mContextOpen = false;
  
  double mMaxTime = 0.0; // (sim time)
  double mCursor  = 0.0; // (sim time)
  double mFps     = 24.0; // TODO: make adjustable

  ImGuiWindow *mWindow = nullptr;
  float mCursorStart = 0.0f;
  float mScroll      = 0.0f;

  Vec2f mTimelineSize = Vec2i(-1,-1);
  float mScale        = 1.0f;
  
  bool  mRecording    = false; // if true, records user inputs, adding to any existing events
  bool  mOverwrite    = false; // if true, records over any existing events
  bool  mApply        = false; // if true, applies keyframe events to the simulation
  bool  mFollowCursor = true;  // if true, follows the cursor as time progresses
  
  float mUpdateScroll = 0;     // counts added time to update timeline size
  bool  mReset = false;        // set to true when sim is reset (scrolls to start)

  KeyEventBase* getEventEnd(KeyEventType type, int iStart);
  
public:
  KeyFrameWidget()
  {
    for(int i = 0; i < (int)KEYEVENT_COUNT; i++)
     { mEvents[(KeyEventType)i].emplace({}); }
  }
  ~KeyFrameWidget()
  {
    for(auto &iter : mEvents)
      {
        for(auto e : iter.second) { delete e; }
        iter.second.clear();
      }
  }

  json toJSON();// const;
  bool fromJSON(const json &js);

  std::vector<std::string> orderedEvents();
  
  bool recording() const   { return mRecording; }
  bool overwriting() const { return mOverwrite; }
  bool active() const      { return mApply;     }
  double cursor() const    { return mCursor; }

  void addPen(const std::string &id, Pen<CFT> *pen);
  void removePen(const std::string &id);
  void setSignalPen(const std::string &id);
  void setMaterialPen(const std::string &id);
  
  void setScale(float scale)   { mScale    = scale; }
  
  void setMaxTime(double tMax)  { mMaxTime = tMax; }
  void setCursor(double cursor) { mCursor = std::min(cursor, mMaxTime); }

  void setInitView2D(const Rect2f &r)            { mView2DInit = r; }
  void setInitView3D(const CameraDesc<CFT> &cam) { mView3DInit = cam; }
  
  void addEvent(KeyEventBase *e, bool force=false);
  bool nextFrame(double dt);    // returns true if event occurred
  bool processEvents(double t0, double t1); // returns true if event occurred

  void reset();
  void clear();
  
  // callbacks
  std::function<void(const Rect2f &r)>            view2DCallback = nullptr;
  std::function<void(const CameraDesc<CFT> &cam)> view3DCallback = nullptr;
  
  // access
  std::map<std::string, std::vector<SignalSource>>&       sources()        { return mSources;  }
  const std::map<std::string, std::vector<SignalSource>>& sources() const  { return mSources;  }
  std::vector<SignalSource> sources(double t) const
  {
    std::vector<SignalSource> s;
    for(const auto &iter : mSources)
      for(const auto &iter2 : iter.second)
        {
          if((iter2.start <= t && iter2.start+iter2.length > t) || iter2.length < 0)
            { s.push_back(iter2); }
        }
    return s;
  }
  std::vector<MaterialPlaced>&                            placed()         { return mPlaced;   }
  const std::vector<MaterialPlaced>&                      placed() const   { return mPlaced;   }
  std::unordered_map<std::string, std::any>&              settings()       { return mSettings; }
  const std::unordered_map<std::string, std::any>&        settings() const { return mSettings; }
  Rect2f&                                                 view2D()         { return mView2D;   }
  const Rect2f&                                           view2D() const   { return mView2D;   }
  CameraDesc<CFT>&                                        view3D()         { return mView3D;   }
  const CameraDesc<CFT>&                                  view3D() const   { return mView3D;   }
  
  std::unordered_map<std::string, std::any>&       settingsInit()       { return mSettingsInit; }
  const std::unordered_map<std::string, std::any>& settingsInit() const { return mSettingsInit; }
  Rect2f&                                          view2DInit()         { return mView2DInit;   }
  const Rect2f&                                    view2DInit() const   { return mView2DInit;   }
  CameraDesc<CFT>&                                 view3DInit()         { return mView3DInit;   }
  const CameraDesc<CFT>&                           view3DInit() const   { return mView3DInit;   }
  
  void drawTimeline();
  void drawEventList();
  void drawSources();
};




#endif // KEYFRAME_WIDGET_HPP
