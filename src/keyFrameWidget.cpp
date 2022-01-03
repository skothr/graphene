#include "keyFrameWidget.hpp"

#include <sstream>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

#include <imgui.h>
#include <imgui_internal.h>
#include <nlohmann/json.hpp> // json implementation
using json = nlohmann::json;

#include "vector-operators.h"
#include "tools.hpp"


#define MOD_COLUMNS "  r  | r^2 |  θ  | sin | cos "

template<typename T>
inline json viewToJSON(const Rect<T> &view)
{
  json js = nlohmann::ordered_json();
  js["p1"] = to_string(view.p1, 12);
  js["p2"] = to_string(view.p2, 12);
  return js;
}
template<typename T>
inline bool viewFromJSON(const json &js, Rect<T> &viewOut)
{
  using VT2 = typename cuda_vec<CFT,2>::VT;
  bool success = true;
  if(js.contains("pos"))   { viewOut.p1 = from_string<VT2>(js["p1"]);   }
  if(js.contains("dir"))   { viewOut.p2 = from_string<VT2>(js["p2"]);   }
  return success;
}


json KeyFrameWidget::toJSON()
{
  json js = nlohmann::ordered_json();

  // TODO: save initial state
  // json iSources = json::object();
  // for(const auto &iter : mSourcesInit)
  //   {
  //     json s = json::object();
  //     s["id"]  = iter.second.id;
  //     s["pos"] = to_string(iter.second.pos, 12);
  //     s["pen"] = penToJSON(iter.second.pen);
  //     if(iter.second.start > 0) { s["start"] = iter.second.start; }
  //     if(iter.second.end > 0)   { s["end"  ] = iter.second.end;   }
  //     iSources[iter.second.id] = s;
  //   } js["initSources"]   = iSources;
  // json iPlaced = json::object();
  // for(int i = 0; i < mPlacedInit.size(); i++)
  //   {
  //     const auto &m = mPlacedInit[i];
  //     std::string id = std::to_string(i, 12);
  //     json m = json::object();
  //     m["pos"] = to_string(iter.second.pos, 12);
  //     m["pen"] = penToJSON(iter.second.pen);
  //     iPlaced[iter.first] = m;
  //   } js["initMatPlaced"] = iPlaced;

  // events
  json events = nlohmann::ordered_json();
  for(auto iter : mEvents)
    {
      json jse = json::array();
      std::stringstream ss; ss << iter.first;
      for(auto e : iter.second)
        {
          //nlohmann::ordered_map<std::string, std::any> kmap;
          json je = nlohmann::ordered_json();
          if(e)
            {
              je["t"] = e->t;
              switch(iter.first)
                {
                case KEYEVENT_SIGNAL_ADD:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_ADD>();
                    je["id"]  = e2->id;
                    je["pos"] = to_string(e2->pos, 12);
                    je["pen"] = penToJSON(e2->pen);
                  } break;
                case KEYEVENT_SIGNAL_REMOVE:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_REMOVE>();
                    je["id"] =  e2->id;
                  } break;
                case KEYEVENT_SIGNAL_MOVE:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_MOVE>();
                    je["id"]  = e2->id;
                    je["pos"] = to_string(e2->pos, 12);
                  } break;
                case KEYEVENT_SIGNAL_PEN:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_PEN>();
                    je["id"]  = e2->id;
                    je["pen"] = penToJSON(e2->pen);
                  } break;
                case KEYEVENT_MATERIAL:
                  {
                    auto e2 = e->sub<KEYEVENT_MATERIAL>();
                    je["pos"] = to_string(e2->pos, 12);
                    // je["pen"] = penToJSON(e2->pen));
                  } break;
                case KEYEVENT_VIEW2D:
                  {
                    auto e2 = e->sub<KEYEVENT_VIEW2D>();
                    je["view"] = viewToJSON(e2->view);
                  } break;
                case KEYEVENT_VIEW3D:
                  {
                    auto e2 = e->sub<KEYEVENT_VIEW3D>();
                    je["view"] = cameraToJSON(e2->view);
                  } break;
                case KEYEVENT_SETTING:
                  {
                    auto e2 = e->sub<KEYEVENT_SETTING>();
                    je["status"] = "UNIMPLEMENTED";
                  } break;
                case KEYEVENT_INVALID: je["status"] = "INVALID"; break;
                default:               je["status"] = "UNKNOWN"; break;
                }
            }
          else { je["status"] = "NULL"; }
              jse.push_back(je);
        }
      events[ss.str()] = jse;
    }

  js["events"] = events;
  
  js["initView2D"] = viewToJSON(mView2DInit);
  js["initView3D"] = cameraToJSON(mView3DInit);
  // js["initSettings"]  = penToJSON(mSettingsInit);
  return js;
}

bool KeyFrameWidget::fromJSON(const json &js)
{
  using VT3 = typename cuda_vec<CFT,3>::VT;
  bool success = true;
  // TODO: load initial state
  // if(js.contains("initSources")
  //   {
  //     json iSources = js["initSources"];
  //     for(const auto &s : iSources)
  //       {
  //         from_string(s["pos"].get<std::string>());
  //         s["pen"] = penToJSON(iter.second.pen);
  //         iSources[iter.first] = s;
  //         mSourcesInit.emplace(id, );
  //       }
  //   } js["initSources"]   = iSources;
  // json iPlaced = json::object();
  // for(int i = 0; i < mPlacedInit.size(); i++)
  //   {
  //     const auto &m = mPlacedInit[i];
  //     std::string id = std::to_string(i, 12);
  //     iPlaced[id] = json::object();
  //     iPlaced[id]["pos"] = to_string(m.pos, 12);
  //     iPlaced[id]["pen"] = penToJSON(m.pen);
  //   } js["initMatPlaced"] = iPlaced;  
  
  if(js.contains("events"))
    {
      json events = js["events"];
      for(auto iter : mEvents)
        {
          std::stringstream ss; ss << iter.first;
          json jse; if(jse.contains(ss.str())) { jse = events[ss.str()]; }
          
          for(auto &e : iter.second)
            {
              json je = json::object();
              if(e)
                {
                  if(je.contains("t")) { e->t = je["t"].get<double>(); }
                  switch(iter.first)
                    {
                    case KEYEVENT_SIGNAL_ADD:
                      {
                        auto e2 = e->sub<KEYEVENT_SIGNAL_ADD>();
                        if(je.contains("id"))  { e2->id  = je["id"].get<std::string>(); }
                        else                   { if(e) { delete e; } e = nullptr; }
                        
                        if(je.contains("pos"))
                        {
                          VT3 newPos = from_string<VT3>(je["pos"].get<std::string>());
                          if(isnan(newPos) || newPos == e2->pos)
                            { if(e) { delete e; } e = nullptr; }
                          else
                            { penFromJSON(je["pen"], e2->pen); }
                        }
                        else  { if(e) { delete e; } e = nullptr; }
                      } break;
                    case KEYEVENT_SIGNAL_REMOVE:
                      {
                        auto e2 = e->sub<KEYEVENT_SIGNAL_REMOVE>();
                        if(je.contains("id"))  { e2->id = je["id"].get<std::string>(); }
                        else                   { if(e) { delete e; } e = nullptr;  }
                      } break;
                    case KEYEVENT_SIGNAL_MOVE:
                      {
                        auto e2 = e->sub<KEYEVENT_SIGNAL_MOVE>();
                        if(je.contains("id"))  { e2->id  = je["id"].get<std::string>(); }
                        else                   { if(e) { delete e; } e = nullptr;  }
                        if(je.contains("pos"))
                          {
                            VT3 newPos = from_string<VT3>(je["pos"].get<std::string>());
                            if(isnan(newPos) || newPos == e2->pos)
                              { if(e) { delete e; } e = nullptr;  }
                            else
                              { e2->pos = newPos; }
                          }
                      } break;
                    case KEYEVENT_SIGNAL_PEN:
                      {
                        auto e2 = e->sub<KEYEVENT_SIGNAL_PEN>();
                        if(je.contains("id"))
                          { e2->id = je["id"].get<std::string>(); }
                        else
                          { if(e) { delete e; } e = nullptr;  }
                        
                        penFromJSON(je["pen"], e2->pen);
                      } break;
                    case KEYEVENT_MATERIAL:
                      {
                        auto e2 = e->sub<KEYEVENT_MATERIAL>();
                        if(je.contains("pos"))
                          {
                            VT3 newPos = from_string<VT3>(je["pos"].get<std::string>());
                            if(isnan(newPos) || newPos == e2->pos)
                              { if(e) { delete e; } e = nullptr; }
                            else { e2->pos = newPos; }
                          }
                        else { if(e) { delete e; } e = nullptr;  }
                        
                        if(je.contains("pen"))
                          { penFromJSON(je["pen"], e2->pen); }
                        else
                          { if(e) { delete e; } e = nullptr;  }
                        
                      } break;
                    case KEYEVENT_VIEW2D:
                      {
                        auto e2 = e->sub<KEYEVENT_VIEW2D>();
                        if(je.contains("view"))
                          {
                            Rect2f newView;
                            if(!viewFromJSON(je["view"], newView) || newView == e2->view)
                              { if(e) { delete e; } e = nullptr;  }
                            else
                              { e2->view = newView; }
                          }
                        else
                          { if(e) { delete e; } e = nullptr;  }
                      } break;
                    case KEYEVENT_VIEW3D:
                      {
                        auto e2 = e->sub<KEYEVENT_VIEW3D>();
                        if(je.contains("view"))
                          {
                            CameraDesc<CFT> newView;
                            if(!cameraFromJSON(je["view"], newView) || newView == e2->view)
                              { if(e) { delete e; } e = nullptr;  }
                            else
                              { e2->view = newView; }
                          }
                        else
                          { if(e) { delete e; } e = nullptr; }
                      } break;
                    case KEYEVENT_SETTING:
                      {
                        if(e) { delete e; } e = nullptr;
                        // auto e2 = e->sub<KEYEVENT_SETTING>();
                        // je["status"] = "UNIMPLEMENTED";
                      } break;
                    case KEYEVENT_INVALID: std::cout << "====> WARNING(KeyFrameWidget::fromJSON()): INVALID EVENT\n"; if(e) { delete e; } e = nullptr; break; 
                    default:               std::cout << "====> WARNING(KeyFrameWidget::fromJSON()): UNKNOWN EVENT\n"; if(e) { delete e; } e = nullptr; break;
                    }
                }
              else { std::cout << "====> WARNING(KeyFrameWidget::fromJSON()): NULL EVENT\n"; }
            }
        }
    }
  if(js.contains("initView2D")) { success |= viewFromJSON(js["initView2D"],   mView2DInit); mView2D = mView2DInit; } else { success = false; }
  if(js.contains("initView3D")) { success |= cameraFromJSON(js["initView3D"], mView3DInit); mView3D = mView3DInit; } else { success = false; }
  // //js["initSettings"]  = penToJSON(mSettingsInit);
  
  return success;
}



std::vector<std::string>  KeyFrameWidget::orderedEvents()
{
  int eTotal = 0;
  for(const auto &iter : mEvents) { eTotal += iter.second.size(); }

  std::vector<std::string> strs; strs.reserve(eTotal);
  std::unordered_map<std::string, KeyEventBase*> events; events.reserve(eTotal);

  // remove invalid events  
  for(auto &iter : mEvents)
    {
      for(int i = 0; i < iter.second.size(); i++)
        {
          KeyEventBase *e = iter.second[i];
          if(!e || e->type() == KEYEVENT_INVALID)
            {
              std::cout << "====> WARNING(KeyFrameWidget::orderedEvents()): Null key event while creating ordered event list ("
                        << (KeyEventType)i << ") removing...\n";
              iter.second.erase(iter.second.begin()+i--);
            }
        }
    }

  for(const auto &iter : mEvents)
    {
      std::stringstream ss; 
      for(const auto &e : iter.second)
        {
          if(e)
            {
              ss << "[" << std::setw(24) << iter.first << " : " // type/timestamp
                 << std::fixed << std::setprecision(4) << std::setw(12) << e->t << "] --> ";
              
              switch(iter.first)
                {
                case KEYEVENT_SIGNAL_ADD:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_ADD>();
                    ss << "   id=" << std::setw(24) << e2->id;
                    ss << "  pos=" << std::setw(24) << to_string(e2->pos, 12);
                  } break;
                case KEYEVENT_SIGNAL_REMOVE:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_REMOVE>();
                    ss << "   id=" << std::setw(24) << e2->id;
                  } break;
                case KEYEVENT_SIGNAL_MOVE:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_MOVE>();
                    ss << "   id=" << std::setw(24) << e2->id;
                    ss << "  pos=" << std::setw(24) << to_string(e2->pos, 12);
                  } break;
                case KEYEVENT_SIGNAL_PEN:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_PEN>();
                    ss << "   id=" << std::setw(24) << e2->id;
                  } break;
                case KEYEVENT_MATERIAL:
                  {
                    auto e2 = e->sub<KEYEVENT_MATERIAL>();
                    ss << "  pos=" << std::setw(24) << to_string(e2->pos, 12);
                  } break;
                case KEYEVENT_VIEW2D:
                  {
                    auto e2 = e->sub<KEYEVENT_VIEW2D>();
                    ss << "  [VIEW 2D]";
                  } break;
                case KEYEVENT_VIEW3D:
                  {
                    auto e2 = e->sub<KEYEVENT_VIEW3D>();
                    ss << "  [VIEW 3D]";
                  } break;
                case KEYEVENT_SETTING:
                  {
                    auto e2 = e->sub<KEYEVENT_SETTING>();
                    ss << "  <UNIMPLEMENTED>";
                  } break;
                  break;
                case KEYEVENT_INVALID: ss << "  <INVALID>"; break;
                default:               ss << "  <UNKNOWN>"; break;
                }
            }
          else { ss << "====> WARNING: <NULL>"; }
          ss << "\n";
          strs.push_back(ss.str());
          events.emplace(ss.str(), e);
        }
    }

  std::sort(strs.begin(), strs.end(), [&](const std::string &s1, const std::string &s2) -> bool
  {
    const auto &iter1 = events.find(s1);
    const auto &iter2 = events.find(s2);
    return (iter1 != events.end() && iter2 != events.end() && iter1->second && iter2->second &&
            iter1->second->t <= iter2->second->t);
  });
  return strs;
}













KeyEventBase* KeyFrameWidget::getEventEnd(KeyEventType type, int iStart)//std::list<KeyEventBase*>::iterator &iter)
{
  if(type == KEYEVENT_SIGNAL_ADD)
    {
      if(!mEvents[KEYEVENT_SIGNAL_ADD][iStart]) { return nullptr; }
      KeyEvent<KEYEVENT_SIGNAL_ADD> *e = mEvents[KEYEVENT_SIGNAL_ADD][iStart]->sub<KEYEVENT_SIGNAL_ADD>();
      
      for(int i = 0; i < mEvents[KEYEVENT_SIGNAL_REMOVE].size(); i++)
        {
          if(!mEvents[KEYEVENT_SIGNAL_REMOVE][i]) { continue; }
          KeyEvent<KEYEVENT_SIGNAL_REMOVE> *e2 = mEvents[KEYEVENT_SIGNAL_REMOVE][i]->sub<KEYEVENT_SIGNAL_REMOVE>();
          if(!e2)       { continue; }
          if(!e || !e2) { continue; }
          if(e && e2 && e2->id == e->id)
            {
              if(e2->t > e->t) { return mEvents[KEYEVENT_SIGNAL_REMOVE][i]; } ///e2; }
            }
        }
    }
  return nullptr;
}



void KeyFrameWidget::addEvent(KeyEventBase *e, bool force)
{
  if(mRecording || force)
    {
      std::cout << "====> NEW KEYFRAME EVENT [" << e->t << "] ";
      if(!e)              { std::cout << "========> WARNING: skipping NULL event\n"; return; }
      else if(e->t < 0.0) { std::cout << "========> WARNING: skipping event with invalid timestamp\n"; delete e; e = nullptr; return; }
      
      if(mOverwrite)
        { // delete overlapping events of the same kind (TODO: differentiate e.g. multiple sources on same frame)
          // auto tEvents = mEvents[e->type()];
          // for(int i = 0; i < tEvents.size(); i++)
          //   {
          //     auto &ev = tEvents[i];
          //     if(ev->start == e->end && (ev->type() != KEYEVENT_SIGNAL_ADD ||
          //                                static_cast<KeyEvent<KEYEVENT_SIGNAL_ADD>*>(ev)->id == static_cast<KeyEvent<KEYEVENT_SIGNAL_ADD>*>(e)->id))
          //       { tEvents.erase(tEvents.begin() + i--); break; }
          //   }
        }

      switch(e->type())
        {
        case KEYEVENT_SIGNAL_ADD:
          {
            auto e2 = e->sub<KEYEVENT_SIGNAL_ADD>();
            std::string id = e2->id;
            std::cout << "(" << e->type() << ") --> '" << id << "' / " << e2->pos << "\n";

            // bool eStart = false;
            // bool eEnd   = false;
            // if(e->start >= 0.0 && e->end < 0.0)       { std::cout << "========> Source " << id << " [START]\n"; eStart = true; }
            // else if(e->start < 0.0 && e->end >= 0.0)  { std::cout << "========> Source " << id << " [END]\n";   eEnd   = true; }
            // else if(e->start >= 0.0 && e->end >= 0.0) { std::cout << "========> Source " << id << " [FULL]\n"; }
            // else                                      { std::cout << "========> Source " << id << " [UNKNOWN/INVALID(?)]\n";  }
            // check if overlapping
            // auto &addEvents = mEvents.at(KEYEVENT_SIGNAL_ADD);
            // auto &remEvents = mEvents.at(KEYEVENT_SIGNAL_REMOVE);
            // KeyEvent<KEYEVENT_SIGNAL_ADD>    *prev = nullptr; // most recent add event with same id
            // KeyEvent<KEYEVENT_SIGNAL_REMOVE> *next = nullptr; // next closest remove event with same id
            // int iPrev = -1; int iNext = -1;
            // // find most recent previous add event
            // for(int i = 0; i < addEvents.size(); i++)
            //   {
            //     KeyEventBase *sb = addEvents[i];
            //     auto *s2 = sb->sub<KEYEVENT_SIGNAL_ADD>();
            //     if(sb && s2 && s2->id == e2->id)
            //       {
            //         if(s2->t <= e->t) { prev  = s2; }
            //         else              { iPrev = i; break; }
            //       }
            //   }
            // if(prev)
            //   { // find next remove after previous add
            //     for(int i = remEvents.size()-1; i >= 0; i--)
            //       {
            //         KeyEventBase *sb = remEvents[i];
            //         auto *s2 = sb->sub<KEYEVENT_SIGNAL_REMOVE>();
            //         if(sb && s2 && s2->id == e2->id)
            //           {
            //             if(s2->t >= prev->t) { next  = s2; }
            //             else                 { iNext = i; break; }
            //           }
            //       }
            //   }
            // if(prev && (!next || (next->t > e->t)))
            //   { // add additional remove event to splice overlapped note
            //     std::cout << "====> OVERLAP --> inserting KEYEVENT_SIGNAL_ADD at t=" << e->t + 0.0001 << "\n";
            //     remEvents.insert(remEvents.begin()+iNext, new KeyEvent<KEYEVENT_SIGNAL_REMOVE>(e->t - 0.0001, e2->id));
            //   }
            // mEventWidgets.emplace(e, new KeyEventWidget(e));
            
            auto iter = mSources.find(e2->id);
            if(iter == mSources.end()) { mSources.emplace(e2->id, std::vector<SignalSource>{}); }
            mSources[e2->id].push_back(SignalSource{ e2->pos, e2->pen, e2->t, -1.0 });
            mEventWidgets.emplace(e, new KeyEventWidget(e));
            
          } break;
          
        case KEYEVENT_SIGNAL_REMOVE:
          {
            auto e2 = e->sub<KEYEVENT_SIGNAL_REMOVE>();
            std::cout << "(" << e->type() << ") --> '" << e2->id << "'\n";
            auto iter = mSources.find(e2->id);
            if(iter != mSources.end())
              {
                for(int i = 0; i < iter->second.size(); i++)
                  {
                    auto &r = iter->second[i];
                    if(r.start <= e2->t && r.length < 0)
                      {
                        r.length = e2->t - r.start;
                        auto w = mEventWidgets.find(e);
                        if(w != mEventWidgets.end()) { w->second->eventEnd = e; w->second->length = r.length; }
                        break;
                      }
                  }
              }

            // // check if overlapping
            // auto &addEvents = mEvents.at(KEYEVENT_SIGNAL_ADD);
            // auto &remEvents = mEvents.at(KEYEVENT_SIGNAL_REMOVE);
            // KeyEvent<KEYEVENT_SIGNAL_ADD>    *prev = nullptr; // next closest add event with same id
            // KeyEvent<KEYEVENT_SIGNAL_REMOVE> *next = nullptr; // most recent remove event with same id
            // int iPrev = -1; int iNext = -1;
            // // find next remove
            // for(int i = 0; i < remEvents.size(); i++)
            //  { 
            //     KeyEventBase *sb = remEvents[i];
            //     auto *s2 = sb->sub<KEYEVENT_SIGNAL_REMOVE>();
            //     if(sb && s2 && s2->id == e2->id)
            //       {
            //         if(s2->t >= e->t) { next  = s2; }
            //         else              { iNext = i; break; }
            //       }
            //   }
            // if(next)
            //   { // find most recent previous add
            //     for(int i = addEvents.size()-1; i >= 0; i--)
            //       {
            //         KeyEventBase *sb = addEvents[i];
            //         auto *s2 = sb->sub<KEYEVENT_SIGNAL_ADD>();
            //         if(sb && s2 && s2->id == e2->id)
            //           {
            //             if(s2->t <= next->t) { prev  = s2; }
            //             else                 { iPrev = i; break; }
            //           }
            //       }
            //   }
            // if(prev && (!next || (next->t > e->t)) && iNext >= 0)
            //   { // add additional remove event to splice overlapped note
            //     std::cout << "====> OVERLAP --> inserting KEYEVENT_SIGNAL_ADD at t=" << e->t + 0.0001 << "\n";
            //     remEvents.insert(addEvents.begin()+iNext, new KeyEvent<KEYEVENT_SIGNAL_ADD>(e->t + 0.0001, e2->id, prev->pos, prev->pen));
            //   }

            // KeyEventBase *last  = nullptr;
            // KeyEventBase *found = nullptr;
            // for(auto iter : mEventWidgets)
            //   {
            //     KeyEventBase *s = iter.first;
            //     if(iter.first->type() == KEYEVENT_SIGNAL_ADD)
            //       {
            //         KeyEvent<KEYEVENT_SIGNAL_ADD> *s2 = s->sub<KEYEVENT_SIGNAL_ADD>();
            //         if(s && s2 && s2->id == e2->id)
            //           {
            //             last  = found;
            //             found = s2;
            //             if(found->t > e->t) { break; }
            //           }
            //       }
            //   }
            // if(!last) { std::cout << "========> WARNING: Source doesn't exist -- skipping\n"; delete e; e = nullptr; }
            // else       { mEventWidgets.erase(last); }
          } break;
        case KEYEVENT_SIGNAL_MOVE:
          {
            auto e2 = e->sub<KEYEVENT_SIGNAL_MOVE>();
            std::cout << "(" << e->type() << ") --> '" << e2->id << "' / " << e2->pos << "\n";
            
            // KeyEvent<KEYEVENT_SIGNAL_MOVE> *found = nullptr;
            // for(auto s : mEvents[KEYEVENT_SIGNAL_MOVE])
            //   {
            //     KeyEvent<KEYEVENT_SIGNAL_MOVE> *s2 = s->sub<KEYEVENT_SIGNAL_MOVE>();
            //     if(s && s2 && s2->id == e2->id) { found = s2; break; }
            //   }
            const SignalSource* found = nullptr;
            for(const auto &iter : mSources)
              {
                if(iter.first == e2->id)
                  {
                    for(int i = 0; i < iter.second.size(); i++)
                      {
                        auto &r = iter.second[i];
                        if(r.start <= e2->t && (r.length < 0 || r.start+r.length > e2->t))
                          { found = &r; break; }
                      }
                  }
              }
            if(!found) { std::cout << "========> WARNING: Source doesn't exist -- skipping\n"; delete e; e = nullptr; }
          } break;
        case KEYEVENT_SIGNAL_PEN:
          {
            auto e2 = e->sub<KEYEVENT_SIGNAL_PEN>();
            std::cout << "(" << e->type() << ") --> '" << e2->id << "\n";
            
            // KeyEvent<KEYEVENT_SIGNAL_PEN> *found = nullptr;
            // for(auto s : mEvents[KEYEVENT_SIGNAL_PEN])
            //   {
            //     KeyEvent<KEYEVENT_SIGNAL_PEN> *s2 = s->sub<KEYEVENT_SIGNAL_PEN>();
            //     if(s && s2 && s2->id == e2->id) { found = s2; break; }
            //   }
            const SignalSource* found = nullptr;
            for(const auto &iter : mSources)
              {
                if(iter.first == e2->id)
                  {
                    for(int i = 0; i < iter.second.size(); i++)
                      {
                        auto &r = iter.second[i];
                        if(r.start <= e2->t && (r.length < 0 || r.start+r.length >= e2->t))
                          { found = &r; break; }
                      }
                  }
              }
            if(!found) { std::cout << "========> WARNING: Source doesn't exist -- skipping\n"; delete e; e = nullptr; }
            
          } break;
        case KEYEVENT_MATERIAL:
          {
            auto e2 = e->sub<KEYEVENT_MATERIAL>();
            std::cout << "(" << e->type() << ") --> " << e2->pos << "\n";
            if(mPlaced.size() > 0 && e2->pos == mPlaced.back().pos && e2->pen == mPlaced.back().pen)
              { delete e; e = nullptr; } // skip repeated material placement
          } break;
        case KEYEVENT_VIEW2D:
          {
            auto e2 = e->sub<KEYEVENT_VIEW2D>();
            std::cout << "(" << e->type() << ") --> " << e2->view << "\n";
          } break;
        case KEYEVENT_VIEW3D:
          {
            auto e2 = e->sub<KEYEVENT_VIEW3D>();
            std::cout << "(" << e->type() << ") --> pos = " << e2->view.pos << " / dir = " << e2->view.dir << "\n";
          } break;
        case KEYEVENT_SETTING:
          {
            auto e2 = e->sub<KEYEVENT_SETTING>();
            std::cout << "(" << e->type() << ") --> " << e2->id << " = " << std::any_cast<typename cuda_vec<CFT,3>::VT>(e2->value) << " (" <<  e2->value.type().name()<< ")\n";
          } break;
          
        case KEYEVENT_INVALID:
          std::cout << "\n========> INVALID KEYFRAME\n";
          if(e) delete e; e = nullptr;
          break;
        default:
          std::cout << "\n========>  KEYFRAME UNKNOWN\n";
          if(e) delete e; e = nullptr;
          break;
        }
      if(e) { mEvents[e->type()].push_back(e); }
    }
}

bool KeyFrameWidget::processEvents(double t0, double t1)
{
  using VT3 = typename cuda_vec<CFT,3>::VT;
  // mPlaced.clear();

  // remove invalid events
  for(auto &iter : mEvents)
    for(int i = 0; i < iter.second.size(); i++)
      {
        KeyEventBase *e = iter.second[i];
        if(!e)
          {
            std::cout << "====> WARNING(KeyFrameWidget::processEvents): Null key event while processing (" << (KeyEventType)i << ") removing...\n";
            iter.second.erase(iter.second.begin()+i--);
          }
      }

  if(!mApply) { return false; }
  
  bool changed = false;
  for(auto &iter : mEvents)
    {
      for(auto &e : iter.second)
        {
          if(!e)
            {
              std::cout << "====> WARNING(KeyFrameWidget::processEvents): Null event (?)\n";
            }
          else if(e->t >= t0 && e->t < t1)
            {
              std::cout << "====> APPLYING KEYFRAME EVENT [" << e->t << "] ";
              switch(e->type())
                {
                case KEYEVENT_SIGNAL_ADD:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_ADD>();
                    std::cout << "(" << e->type() << ") --> '" << e2->id << "' / " << e2->pos << "\n";
                    auto iter = mSources.find(e2->id);
                    if(iter != mSources.end())
                      {
                        std::cout << "========> WARNING(KeyFrameWidget::processEvents): Source already exist -- skipping\n";
                        // iter->second.pos = e2->pos; iter->second.pen = e2->pen;
                      }
                    else
                      {
                        // auto iter = mSources.find(e2->id);
                        // if(iter == mSources.end()) { mSources.emplace(e2->id, std::vector<SignalSource>{}); }
                        // mSources[e2->id].push_back(SignalSource{ e2->pos, e2->pen, e2->t, -1.0 });
                        std::cout << "========> Source active\n";
                        changed = true;
                      }
                  } break;
                
                case KEYEVENT_SIGNAL_REMOVE:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_REMOVE>();
                    std::cout << "(" << e->type() << ") --> '" << e2->id << "'\n";
                    auto iter = mSources.find(e2->id);
                    if(iter == mSources.end())
                      { std::cout << "========> WARNING(KeyFrameWidget::processEvents): Source doesn't exist -- skipping\n"; }
                    else
                      {
                        // for(auto &iter2 : iter->second)
                        //   {
                        //     if(iter2.start <= e->t && (iter2.length < 0 || iter2.start+iter2.length >= e->t))
                        //       {
                        //         iter2.length = e->t - iter2.start;;
                        //         std::cout << "========> Deleted source\n";
                        //         changed = true;
                        //       }
                        //   }
                      }
                  } break;
                
                case KEYEVENT_SIGNAL_MOVE:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_MOVE>();
                    std::cout << "(" << e->type() << ") --> '" << e2->id << "' / " << e2->pos << "\n";
                    auto iter = mSources.find(e2->id);
                    if(iter == mSources.end())
                      {
                        std::cout << "========> WARNING(KeyFrameWidget::processEvents): Source doesn't exist -- skipping\n";
                        // mSources.emplace(e2->id, SignalSource{ e2->pos, *mSigPen });
                      }
                    else
                      {
                        for(auto &iter2 : iter->second)
                          {
                            if(iter2.start <= e->t && (iter2.length < 0 || iter2.start+iter2.length >= e->t))
                              {
                                iter2.pos = e2->pos;
                                std::cout << "========> Moved source (" << iter2.pos << ")\n";
                                changed = true;
                              }
                          }
                      }
                  } break;
                
                case KEYEVENT_SIGNAL_PEN:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_PEN>();
                    std::cout << "(" << e->type() << ") --> '" << e2->id << "\n";
                    auto iter = mSources.find(e2->id);
                    if(iter == mSources.end())
                      {
                        std::cout << "========> WARNING(KeyFrameWidget::processEvents): Source doesn't exist -- adding (" << VT3{0,0,0} << ")\n";
                        mSources.emplace(e2->id, std::vector<SignalSource>{SignalSource{ VT3{0,0,0}, e2->pen }});
                      }
                    else
                      {
                        for(auto &iter2 : iter->second)
                          {
                            if(iter2.start <= e->t && (iter2.length < 0 || iter2.start+iter2.length >= e->t))
                              {
                                iter2.pen = e2->pen;
                                std::cout << "========> Set source pen\n";
                                changed = true;
                              }
                          }
                      }
                  } break;
                
                case KEYEVENT_MATERIAL:
                  {
                    auto e2 = e->sub<KEYEVENT_MATERIAL>();
                    std::cout << "(" << e->type() << ") --> " << e2->pos << "\n";
                    mPlaced.push_back(MaterialPlaced{ e2->pos, e2->pen });
                    std::cout << "========> Placed material\n";
                    changed = true;
                  } break;
                
                case KEYEVENT_VIEW2D:
                  {
                    auto e2 = e->sub<KEYEVENT_VIEW2D>();
                    std::cout << "(" << e->type() << ") --> " << e2->view << "\n";
                    mView2D = e2->view;
                    if(view2DCallback) { view2DCallback(mView2D); }
                    std::cout << "========> Set 2D view\n";
                    changed = true;
                  } break;
                
                case KEYEVENT_VIEW3D:
                  {
                    auto e2 = e->sub<KEYEVENT_VIEW3D>();
                    std::cout << "(" << e->type() << ") --> pos = " << e2->view.pos << " / dir = " << e2->view.dir << "\n";
                    mView3D = e2->view;
                    if(view3DCallback) { view3DCallback(mView3D); }
                    std::cout << "========> Set 3D view\n";
                    changed = true;
                  } break;
                
                case KEYEVENT_SETTING:
                  {
                    auto e2 = e->sub<KEYEVENT_SETTING>();
                    std::cout << "(" << e->type() << ") --> " << e2->id << " = " << std::any_cast<VT3>(e2->value)
                              << " (" <<  e2->value.type().name()<< ")\n";
                    mSettings[e2->id] = e2->value;
                    std::cout << "========> Updated settings\n";
                  } break;
                
                case KEYEVENT_INVALID: std::cout << "\n========> WARNING(KeyFrameWidget::processEvents): INVALID KEYFRAME\n";      break;
                default:               std::cout << "\n========> WARNING(KeyFrameWidget::processEvents): UNKNOWN KEYFRAME TYPE\n"; break;
                }
            }
        }
    }
  return changed;
}



void KeyFrameWidget::addPen(const std::string &id, Pen<CFT> *pen)
{
  if(pen->isSignal())        { mSigPens.emplace(id, static_cast<SignalPen<CFT>*>(pen)); }
  else if(pen->isMaterial()) { mMatPens.emplace(id, static_cast<MaterialPen<CFT>*>(pen)); }
}

void KeyFrameWidget::removePen(const std::string &id)
{
  auto iter = mSigPens.find(id);
  if(iter != mSigPens.end()) { mSigPens.erase(iter); }
  else
    {
      auto iter = mMatPens.find(id);
      if(iter != mMatPens.end()) { mMatPens.erase(iter); }
    }
}

void KeyFrameWidget::setSignalPen(const std::string &id)   { mActiveSigPen = id; }
void KeyFrameWidget::setMaterialPen(const std::string &id) { mActiveMatPen = id; }

bool KeyFrameWidget::nextFrame(double dt)
{
  ImGuiStyle &style = ImGui::GetStyle();
  CFT tSize = mScale*KEYFRAME_SECOND_MINSIZE.x;
  
  mCursor += dt;
  if(mFollowCursor)
    {
      if(mWindow) { // mScroll = mWindow->Scroll.x; 
        ImGui::SetScrollFromPosX(mWindow, mCursorStart + mCursor*tSize, 0.5f); }
    }
  return processEvents(mCursor-dt, mCursor);
}

void KeyFrameWidget::reset()
{
  mCursor = 0;
  if(mWindow) { mWindow->Scroll.x = 0; }

  mSources.clear();
  mPlaced.clear();
  
  if(mApply)
    {
      mView2D = mView2DInit;
      mView3D = mView3DInit;
      if(view2DCallback) { view2DCallback(mView2D); }
      if(view3DCallback) { view3DCallback(mView3D); }
    }
  mReset = true;
}

void KeyFrameWidget::clear()
{
  reset();
  for(auto &iter : mEvents) { for(auto e : iter.second) { if(e) { delete e; } } iter.second.clear(); }
  for(auto &iter : mEventWidgets) { if(iter.second) { delete iter.second; } } mEventWidgets.clear();
}

void KeyFrameWidget::drawTimeline()
{
  using VT3 = typename cuda_vec<CFT,3>::VT;
  ImGuiIO    &io    = ImGui::GetIO();
  ImGuiStyle &style = ImGui::GetStyle();

  Vec2f size = ImGui::GetContentRegionMax();
  
  ImGui::BeginGroup(); // first column
  {
    ImGui::Checkbox("Record##kf",       &mRecording   );
    ImGui::Checkbox("Overwrite##kf",    &mOverwrite   );
    ImGui::Checkbox("Apply##kf",        &mApply       );
    ImGui::Checkbox("Folow Cursor##kf", &mFollowCursor);

    const fs::path fileJSON = fs::current_path() / "events.json";
    const fs::path fileORD  = fs::current_path() / "events.ord";
    
    if(ImGui::Button("Clear##kb")) { clear(); }
    if(ImGui::Button("Save JSON"))
      {
        std::ofstream f(fileJSON);
        if(f.is_open()) { f << std::setw(2) << toJSON(); }
      }
    if(ImGui::Button("Load JSON"))
      {
        std::ifstream f(fileJSON);
        if(f.is_open())
          {
            json js; f >> js;
            if(!fromJSON(js)) { std::cout << "====> ERROR: Failed to parse JSON file '" << fileJSON << "'\n"; }
          }
        else { std::cout << "====> ERROR: Failed to load JSON file '" << fileJSON << "'\n"; }
      }
    if(ImGui::Button("Save List"))
      {
        std::ofstream f(fileORD);
        if(f.is_open())
          {
            auto lines = orderedEvents();
            for(const auto &l : lines) { if(!l.empty()) { f << l; } }
          }
        else { std::cout << "====> ERROR: Failed to load ORD file '" << fileORD << "'\n"; }
      }
  }
  ImGui::EndGroup();
  ImGui::SameLine();

  // scrollable inner timeline
  Vec2f childSize = (size - Vec2f((ImGui::GetItemRectMax().x-ImGui::GetItemRectMin().x) + style.ItemSpacing.x, 0.0f) - style.WindowPadding);
  Vec2f p0 = ImGui::GetCursorScreenPos();

  ImGuiWindowFlags wFlags = ImGuiWindowFlags_HorizontalScrollbar;
  ImGui::PushStyleColor(ImGuiCol_ChildBg, Vec4f(0.24f, 0.24f, 0.24f, 1.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, Vec2f(0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, Vec2f(4, 4));
  if(ImGui::BeginChild("kfChild", childSize, true, wFlags))
    { ImGui::PopStyleVar(2);

      mWindow      = ImGui::GetCurrentWindow();
      mCursorStart = ImGui::GetCursorStartPos().x;
      
      ImDrawList *drawList = ImGui::GetWindowDrawList();
      Vec2f       p1       = ImGui::GetCursorScreenPos();
      CFT         scrollX  = ImGui::GetScrollX();
      Vec2f       mp       = ImGui::GetMousePos();

      Vec2f  tSize   = Vec2f(mScale*KEYFRAME_SECOND_MINSIZE.x, KEYFRAME_SECOND_MINSIZE.y); // scaled width of 1 second
      Vec2f  fSize   = tSize/mFps; // scaled width of 1 frame
      double tOffset = scrollX/tSize.x;
      double numSecs = childSize.x/tSize.x; // total number of seconds visible
      Vec2d  tRange  = Vec2d(tOffset, tOffset + numSecs);
      
      double mtime   = (scrollX + (mp.x - p0.x))/tSize.x; // time at mouse position
      int    mframe  = (scrollX + (mp.x - p0.x))/fSize.x; // frame at mouse position

      bool eventHovered = false;
      KeyEventBase *hoveredEvent = nullptr;
      ImGui::BeginGroup();
      {
        // draw column header
        ImGui::BeginGroup();
        {
          drawList->AddRectFilled(p0, p0+Vec2f(childSize.x, tSize.y), ImColor(0.0f, 0.0f, 0.0f, 0.15f)); // darken header
          for(int t = std::floor(tRange.x); t <= std::ceil(tRange.y); t++)
            {
              // double t2 = tOffset+t-fmod(tOffset, 1.0);
              double tp = t*tSize.x;
              // frame number label
              std::stringstream ss; ss << std::round(t);
              std::string label = ss.str();
              CFT lW = ImGui::CalcTextSize(label.c_str()).x;
              ImGui::SetCursorPos(Vec2f(tp + (tSize.x - lW)/2.0f, 0.0f));
              ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted(label.c_str());
              ImGui::SameLine();

              if(tp >= -scrollX && tp <= childSize.x+scrollX)
                { // column separator (left of column)
                  drawList->AddLine(p1+Vec2f(tp, 0.0f),    p1+Vec2f(tp, childSize.y),     ImColor(Vec4f(1, 1, 1, 0.3f)), 1.0f);
                }
            }
          // header separator
          drawList->AddLine(p1+Vec2f(scrollX, tSize.y),      p1+Vec2f(scrollX+childSize.x, tSize.y), ImColor(Vec4f(1, 1, 1, 0.6f)), 2.0f);
          // draw cursor
          drawList->AddLine(p1+Vec2f(mCursor*tSize.x, 0.0f), p1+Vec2f(mCursor*tSize.x, childSize.y), ImColor(Vec4f(1, 0, 0, 0.5f)), 3.0f);
        }
        ImGui::Dummy(Vec2f((numSecs+2.0f)*tSize.x, 1.0f)); // fill out full size
        ImGui::EndGroup();
        if(ImGui::IsItemClicked(ImGuiMouseButton_Left))
          { mUpdateScroll = mtime-mCursor; mCursor = mtime;}

        Vec2i headerSize = Vec2f(ImGui::GetItemRectMax())-ImGui::GetItemRectMin();
        float keyH = (ImGui::GetContentRegionMax().y - headerSize.y)/(float)KEYEVENT_COUNT;

        // draw keyed events
        for(auto &iter : mEvents)
          {
            // type dividers
            float ety = headerSize.y + keyH*((float)iter.first);
            drawList->AddLine(p1+Vec2f(scrollX, ety), p1+Vec2f(scrollX+childSize.x, ety), ImColor(Vec4f(1, 1, 1, 0.5f)), 1.0f);
            
            if(iter.first == KEYEVENT_SIGNAL_REMOVE) { continue; } // remove events handled by KEYEVENT_SIGNAL_ADD
            
            for(int i = 0; i < iter.second.size(); i++)
              {
                //auto e = iter.second[i];
                KeyEventBase *e  = iter.second[i];             if(!e)  { continue; }
                KeyEventBase *e2 = getEventEnd(iter.first, i); //if(!e2) { continue; }

                double start  = e->t;
                double end    = e2 ? e2->t : (e->type() == KEYEVENT_SIGNAL_ADD ? mCursor : start+0.1);
                double sStart = start*tSize.x;
                double sEnd   = end*tSize.x;
                double mpos   = (mp.x - p1.x)/tSize.x;

                ImGui::SetCursorScreenPos(p1 + Vec2f(sStart, headerSize.y + keyH*(float)iter.first));
                std::string typeStr = to_string(e->type());

                const Vec4f moveCol = Vec4f(1.0f, 0.3f, 0.3f, 1.0f);
                bool color = (mLastHovered == e);
                if(color)
                  {
                    ImGui::PushStyleColor(ImGuiCol_Button,        moveCol);
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, moveCol);
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  moveCol);
                  }
                ImGui::Button(std::to_string(i).c_str(), Vec2f(sEnd-sStart, keyH));
                if(color) { ImGui::PopStyleColor(3); }
              
                bool hovered = ImGui::IsItemHovered();
                eventHovered |= hovered;
                //if(!eventHovered && !mContextOpen) { mLastHovered = nullptr; }
                
                if(hovered || (mLastHovered == e && !mContextOpen && ImGui::IsMouseDragging(ImGuiMouseButton_Left)))
                  {
                    hoveredEvent = e;
                    mLastHovered = e;
                    ImGui::SetTooltip("t = %f / Frame %d \n\nEvent(%s): %s (%f -> %f)", mtime, mframe, typeStr.c_str(),
                                      (e->type() == KEYEVENT_SIGNAL_ADD ? static_cast<KeyEvent<KEYEVENT_SIGNAL_ADD>*>(e)->id.c_str() : ""), start, end);
                  }

                auto wIter = mEventWidgets.find(e);
                if(wIter != mEventWidgets.end())
                  {
                    auto *w = wIter->second;
                    if(hovered && w->clickPos < 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                      {
                        std::cout << " EVENT CLICKED: " << typeStr << " --> mpos=" << mpos << " / start=" << start << ", end=" << end << "\n";
                        w->clickPos  = mpos;
                        w->lastStart = start;
                        w->length    = end-start;
                      }
                    else if(w->clickPos >= 0.0f && ImGui::IsMouseReleased(ImGuiMouseButton_Left))
                      {
                        std::cout << " EVENT RELEASED: " << typeStr << "\n";
                        w->clickPos  = -1.0;
                        w->lastStart = -1.0;
                        w->length    = -1.0;
                      }
              
                    if(w->clickPos >= 0 && ImGui::IsMouseDragging(ImGuiMouseButton_Left))
                      {
                        Vec2f  dmp = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left); ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
                        double dt = (mpos - w->clickPos);
                        double newStart = w->lastStart + dt;
                        if(newStart != start)
                          { std::cout << "MOVING EVENT: (" << typeStr << ")  " << w->lastStart << " / " << dt << " / mpos=" << mpos << " --> " << newStart << "\n"; }
                        e->t  = newStart;
                        if(e2) { e2->t = newStart + w->length; }
                      }
                  }
                if(hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Middle) &&
                   !ImGui::IsMouseDown(ImGuiMouseButton_Left) && !ImGui::IsMouseDown(ImGuiMouseButton_Right))
                  { // delete event (test)
                    std::cout << "DELETING EVENT: " << typeStr << "\n";
                    if(iter.first == KEYEVENT_SIGNAL_ADD && e2)
                      { mEventWidgets.erase(e2); }
                    iter.second.erase(iter.second.begin() + i);
                    mEventWidgets.erase(e);
                    delete e;
                  }
              }
          }
      }
      ImGui::EndGroup();

      // event context menu
      bool bgHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) && !eventHovered;
      if(ImGui::BeginPopup("kfeContext"))
        {
          if(mLastHovered)
            {
              std::stringstream ss; ss << "Event  (" << mLastHovered->type() << ") --> " << mLastHovered->t << "\n";
              if(mLastHovered->type() == KEYEVENT_SIGNAL_ADD)
                {
                  KeyEvent<KEYEVENT_SIGNAL_ADD> *last2 = mLastHovered->sub<KEYEVENT_SIGNAL_ADD>();
                  
                  auto iter = mSources.find(last2->id);
                  if(iter != mSources.end())
                    {
                      auto wIter = mEventWidgets.find(mLastHovered);
                      for(int i = 0; i < iter->second.size(); i++)
                        {
                          auto &s = iter->second[i];
                          if(s.start <= mLastHovered->t)
                            {
                              if(ImGui::InputDouble("start##ev",  &s.start))
                                {
                                  if(wIter != mEventWidgets.end() && wIter->second) { wIter->second->lastStart = s.start; }
                                  KeyEventBase *e2 = getEventEnd(KEYEVENT_SIGNAL_ADD, i);
                                  if(last2) { last2->t = s.start; }
                                  if(e2)    { e2->t = s.start+s.length; }
                                }
                              if(ImGui::InputDouble("length##ev", &s.length))
                                {
                                  if(wIter != mEventWidgets.end() && wIter->second) { wIter->second->length = s.length; }
                                  KeyEventBase *e2 = getEventEnd(KEYEVENT_SIGNAL_ADD, i);
                                  if(e2) { e2->t = s.start+s.length; }
                                }
                              break;
                            }
                        }
                    }
                  else
                    {
                      ImGui::InputDouble("start##ev", &mLastHovered->t);
                      // ImGui::InputDouble("end##ev",   &mLastHovered->t);
                    }
                }
              else
                {
                  ImGui::InputDouble("t##ev", &mLastHovered->t);
                }
              
            }
          ImGui::EndPopup();
        }

      mContextOpen = false;

      // event context menu
      if(eventHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) { ImGui::OpenPopup("kfeContext"); }
      if(ImGui::BeginPopup("kfContext"))
        {
          if(ImGui::BeginMenu("Edit"))
            {
              if(ImGui::MenuItem("Add Signal"))
                {
                  const auto &iter = mSigPens.find(mActiveSigPen);
                  if(iter != mSigPens.end())
                    {
                      addEvent(new KeyEvent<KEYEVENT_SIGNAL_ADD>   (mCursor,     iter->first, VT3{0.0f, 0.0f, 0.0f}, *iter->second), true);
                      addEvent(new KeyEvent<KEYEVENT_SIGNAL_REMOVE>(mCursor+1.0, iter->first), true);
                    }
                }
              if(ImGui::MenuItem("Add Material"))
                {
                  const auto &iter = mMatPens.find(mActiveMatPen);
                  if(iter != mMatPens.end())
                    {
                      addEvent(new KeyEvent<KEYEVENT_MATERIAL>     (mCursor, VT3{0.0f, 0.0f, 0.0f}, *iter->second), true);
                    }
                }
              ImGui::EndMenu();
            }
          mContextOpen = true;
          ImGui::EndPopup();
        }

      // background context menu
      if(bgHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) { ImGui::OpenPopup("kfContext"); }
      if(bgHovered)
        {
          // Scale timeline horizontally -- Ctrl+Alt+Scroll
          if(std::abs(io.MouseWheel) > 0.0f && io.KeyCtrl && io.KeyAlt)
            {
              double mf1 = (mp.x - p0.x + scrollX)/tSize.x;
              
              double vel = 1.08; // scale velocity
              mScale *= (io.MouseWheel > 0.0f ? vel : 1.0/vel);

              // correct scrolling so mouse position stays the same
              double newW = mScale * tRange.x;
              double mf2 = (mp.x - p0.x + scrollX)/newW;
              float newScroll = scrollX + (mf1 - mf2);
              ImGui::SetScrollX(newScroll);
            }

          if(ImGui::IsMouseDown(ImGuiMouseButton_Left) && io.KeyCtrl)
            {
              mCursor = mtime;
              Vec2i cr = ImGui::GetContentRegionMax();
              ImGui::SetTooltip("t = %f / Frame %d \n (cursor: %f) \n content region: %d x %d", mtime, mframe, mCursor, cr.x, cr.y);
            }
          else
            { ImGui::SetTooltip("t = %f / Frame %d \n (cursor: %f)", mtime, mframe, mCursor); }
        }
    }
  else { ImGui::PopStyleVar(2); }
  ImGui::EndChild();
  ImGui::PopStyleColor();
}



void KeyFrameWidget::drawEventList()
{
  for(const auto &eStr : orderedEvents())
    { ImGui::TextUnformatted(eStr.c_str()); }
}

void KeyFrameWidget::drawSources()
{
  auto iter = mEvents.find(KEYEVENT_SIGNAL_ADD);
  if(iter != mEvents.end())
    {
      for(int i = 0; i < iter->second.size(); i++)
        {
          auto e = iter->second[i];
          if(e)
            {
              ImGui::Text("%2.4f --> Source Added", e->t);
              KeyEventBase *e2 = getEventEnd(KEYEVENT_SIGNAL_ADD, i);
              if(e2)
                {
                  ImGui::SameLine();
                  ImGui::Text("  (removed %2.4f)", e2->t);
                }
            }
        }
    }
}
