#ifndef KEYFRAME_WIDGET_HPP
#define KEYFRAME_WIDGET_HPP

#include <list>

#include "keyFrame.hpp"

// #define KEYFRAME_FRAME_MINSIZE Vec2f(42.0f, 8.0f) // minimum displayed width of one sim frame (x axis, before scaling)
// minimum displayed width of one second in sim time (x axis, before scaling)
#define KEYFRAME_SECOND_MINSIZE Vec2f(60.0f, ImGui::CalcTextSize("X").y+style.FramePadding.y*2.0f) 


struct KeyEventWidget
{
  KeyEventBase *event = nullptr;
  double clickPos  = -1.0;
  double lastStart = -1.0;
  double length    = -1.0;
  KeyEventWidget(KeyEventBase *e) : event(e) { }
};


class KeyFrameWidget
{
private:
  SignalPen<float>   *mSigPen = nullptr; // global signal pen
  MaterialPen<float> *mMatPen = nullptr; // global material pen
  
  // current states
  std::map<std::string, SignalSource>       mSources;    // active signal sources
  std::vector<MaterialPlaced>               mPlaced;     // material placements for next frame
  Rect2f                                    mView2D;     // 2D view for next frame
  CameraDesc<float>                         mView3D;     // 3D view for next frame
  std::unordered_map<std::string, std::any> mSettings;   // global settings (TODO)
  
  // initial states (frame 0)
  Rect2f                                    mView2DInit;
  CameraDesc<float>                         mView3DInit;
  std::unordered_map<std::string, std::any> mSettingsInit; // TODO

  // event structures
  std::map<KeyEventType, std::vector<KeyEventBase*>> mEvents;
  std::unordered_map<KeyEventBase*, KeyEventWidget*> mEventWidgets;

  KeyEventBase *mLastHovered = nullptr;
  bool mContextOpen = false;
  
  double mMaxTime = 0.0; // (sim time)
  double mCursor  = 0.0; // (sim time)
  double mFps     = 24.0; // TODO: make adjustable

  Vec2f mTimelineSize = Vec2i(-1,-1);
  float mScale        = 1.0f;
  
  bool  mRecording    = false; // if true, records user inputs, adding to any existing events at each frame
  bool  mOverwrite    = false; // if true, records user inputs, replacing any existing events at each frame
  bool  mApply        = false; // if true, applies keyframe events to the simulation
  bool  mFollowCursor = true;  // if true, follows the cursor as time progresses
  
  float mUpdateScroll = 0;     // counts added time to update timeline size
  bool  mReset = false;        // set to true when sim is reset (scrolls to start)

  KeyEventBase* getEventEnd(KeyEventType type, int iStart)//std::list<KeyEventBase*>::iterator &iter)
  {
    if(type == KEYEVENT_SIGNAL_ADD)
      {
        if(!mEvents[KEYEVENT_SIGNAL_ADD][iStart]) { return nullptr; }
        KeyEvent<KEYEVENT_SIGNAL_ADD> *e = mEvents[KEYEVENT_SIGNAL_ADD][iStart]->sub<KEYEVENT_SIGNAL_ADD>();
        // double nextAdd = e->t;
        // for(int i = 0; i < mEvents[KEYEVENT_SIGNAL_ADD].size(); i++)
        //   {
        //     if(!mEvents[KEYEVENT_SIGNAL_ADD][i]) { continue; }
        //     KeyEvent<KEYEVENT_SIGNAL_ADD> *e2 = mEvents[KEYEVENT_SIGNAL_ADD][i]->sub<KEYEVENT_SIGNAL_ADD>();
        //     if(e && e2 && e2->id == e->id)
        //       {
        //         if(e2->t > e->t) { nextAdd = e2->t; break; }
        //       }
        //   }
        
        for(int i = 0; i < mEvents[KEYEVENT_SIGNAL_REMOVE].size(); i++)
          {
            if(!mEvents[KEYEVENT_SIGNAL_REMOVE][i]) { continue; }
            KeyEvent<KEYEVENT_SIGNAL_REMOVE> *e2 = mEvents[KEYEVENT_SIGNAL_REMOVE][i]->sub<KEYEVENT_SIGNAL_REMOVE>();
            if(!e2)       { continue; }
            if(!e || !e2) { continue; }
            if(e && e2 && e2->id == e->id)
              {
                // if(e2->t > e->t) { if(e2->t < nextAdd) { return e2->t; } else { return nextAdd; } }
                if(e2->t > e->t) { return mEvents[KEYEVENT_SIGNAL_REMOVE][i]; } ///e2; }
              }
          }
      }
    return nullptr; // mEvents[type][iStart]->t + 1.0;
  }

  
public:
  KeyFrameWidget(SignalPen<float> *sigPen, MaterialPen<float> *matPen)
    : mSigPen(sigPen), mMatPen(matPen)
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

  std::vector<std::string> orderedEvents() const;
  
  bool recording() const { return mRecording; }
  bool active() const    { return mApply;     }
  double cursor() const  { return mCursor; }
  
  void setScale(float scale)   { mScale    = scale; }
  
  void setMaxTime(double tMax)  { mMaxTime = tMax; }
  void setCursor(double cursor) { mCursor = std::min(cursor, mMaxTime); }

  void setInitView2D(const Rect2f &r)              { mView2DInit = r; }
  void setInitView3D(const CameraDesc<float> &cam) { mView3DInit = cam; }
  
  void addEvent(KeyEventBase *e, bool force=false);
  bool nextFrame(double dt);    // returns true if event occurred
  bool processEvents(double t0, double t1); // returns true if event occurred

  void reset();
  void clear();
  
  // callbacks
  std::function<void(const Rect2f &r)>              view2DCallback = nullptr;
  std::function<void(const CameraDesc<float> &cam)> view3DCallback = nullptr;
  
  // access
  std::map<std::string, SignalSource>&             sources()        { return mSources;  }
  const std::map<std::string, SignalSource>&       sources() const  { return mSources;  }
  std::vector<MaterialPlaced>&                     placed()         { return mPlaced;   }
  const std::vector<MaterialPlaced>&               placed() const   { return mPlaced;   }
  std::unordered_map<std::string, std::any>&       settings()       { return mSettings; }
  const std::unordered_map<std::string, std::any>& settings() const { return mSettings; }
  Rect2f&                                          view2D()         { return mView2D;   }
  const Rect2f&                                    view2D() const   { return mView2D;   }
  CameraDesc<float>&                               view3D()         { return mView3D;   }
  const CameraDesc<float>&                         view3D() const   { return mView3D;   }
  
  std::unordered_map<std::string, std::any>&       settingsInit()       { return mSettingsInit; }
  const std::unordered_map<std::string, std::any>& settingsInit() const { return mSettingsInit; }
  Rect2f&                                          view2DInit()         { return mView2DInit;   }
  const Rect2f&                                    view2DInit() const   { return mView2DInit;   }
  CameraDesc<float>&                               view3DInit()         { return mView3DInit;   }
  const CameraDesc<float>&                         view3DInit() const   { return mView3DInit;   }
  
  void draw(); //const Vec2f &size);
};




#endif // KEYFRAME_WIDGET_HPP
