#ifndef KEY_FRAME_HPP
#define KEY_FRAME_HPP

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <map>
#include <any>

#include <nlohmann/json_fwd.hpp> // json forward declarations
using json = nlohmann::json;

#include "vector.hpp"
#include "rect.hpp"
#include "draw.cuh"
#include "drawPens.hpp"
#include "camera.hpp"


enum KeyEventClass
  {
   EVENTCLASS_INVALID = -1,
   EVENTCLASS_SIGNAL =  0, // signal placement
   EVENTCLASS_MATERIAL,    // material placement
   EVENTCLASS_VIEW,        // change view
   EVENTCLASS_SETTING,     // change a global setting
   EVENTCLASS_COUNT
  };

enum KeyEventType
  {
   KEYEVENT_INVALID    = -1,
   KEYEVENT_SIGNAL_ADD =  0,  // add a persistent signal source (start/stop)
   KEYEVENT_SIGNAL_REMOVE,    // remove a signal source         (start/stop)
   KEYEVENT_SIGNAL_MOVE,      // move source location           (single)
   KEYEVENT_SIGNAL_PEN,       // change source properties       (single)
   KEYEVENT_MATERIAL,         // place material                 (single)
   KEYEVENT_VIEW2D,           // set 2D view rect               (single)
   KEYEVENT_VIEW3D,           // set 3D view camera             (single)
   KEYEVENT_SETTING,          // change a (global?) setting     (single)
   KEYEVENT_COUNT
  };
inline std::ostream& operator<<(std::ostream &os, const KeyEventType t)
{
  switch(t)
    {
    case KEYEVENT_INVALID:       os << "<INVALID>";                break;
    case KEYEVENT_SIGNAL_ADD:    os << "<KEYEVENT_SIGNAL_ADD>";    break;
    case KEYEVENT_SIGNAL_REMOVE: os << "<KEYEVENT_SIGNAL_REMOVE>"; break;
    case KEYEVENT_SIGNAL_MOVE:   os << "<KEYEVENT_SIGNAL_MOVE>";   break;
    case KEYEVENT_SIGNAL_PEN:    os << "<KEYEVENT_SIGNAL_PEN>";    break;
    case KEYEVENT_MATERIAL:      os << "<KEYEVENT_MATERIAL>";      break;
    case KEYEVENT_VIEW2D:        os << "<KEYEVENT_VIEW2D>";        break;
    case KEYEVENT_VIEW3D:        os << "<KEYEVENT_VIEW3D>";        break;
    case KEYEVENT_SETTING:       os << "<KEYEVENT_SETTING>";       break;
    default:                     os << "<UNKNOWN>";                break;
    }
  return os;
}
template<typename T>
std::enable_if_t<std::is_same_v<T, KeyEventType>, std::string> to_string(T e) { std::stringstream ss; ss << e; return ss.str(); }

template<KeyEventType T> struct KeyEvent;

// base class for keyframe events (similar to MIDI)
struct KeyEventBase
{
  double t = -1.0;  // timestamp (sim time)
  KeyEventBase(double t_) : t(t_) { }
  virtual constexpr KeyEventType type() const = 0;
  template<KeyEventType T> KeyEvent<T>* sub() { return static_cast<KeyEvent<T>*>(this); }
};

// add signal source
template<> struct KeyEvent<KEYEVENT_SIGNAL_ADD> : public KeyEventBase
{
  std::string      id; // signal source id
  float3           pos;
  SignalPen<float> pen;
  KeyEvent(double t0, const std::string &id_, const float3 &pos_, const SignalPen<float> &pen_)
    : KeyEventBase(t0),  id(id_), pos(pos_), pen(pen_) { }
  virtual constexpr KeyEventType type() const override { return KEYEVENT_SIGNAL_ADD; }
};

// remove signal source
template<> struct KeyEvent<KEYEVENT_SIGNAL_REMOVE> : public KeyEventBase
{
  std::string id; // signal source id
  KeyEvent(double t1, const std::string &id_) : KeyEventBase(t1), id(id_) { }
  virtual constexpr KeyEventType type() const override { return KEYEVENT_SIGNAL_REMOVE; }
};

// move signal source
template<> struct KeyEvent<KEYEVENT_SIGNAL_MOVE> : public KeyEventBase
{
  std::string id; // signal source id
  float3      pos;
  KeyEvent(double t, const std::string &id_, const float3 &pos_) : KeyEventBase(t), id(id_), pos(pos_) { }
  virtual constexpr KeyEventType type() const override { return KEYEVENT_SIGNAL_MOVE; }
};

// set signal source pen
template<> struct KeyEvent<KEYEVENT_SIGNAL_PEN> : public KeyEventBase
{
  std::string      id; // signal source id
  SignalPen<float> pen;
  KeyEvent(double t, const std::string &id_, const SignalPen<float> &pen_) : KeyEventBase(t), id(id_), pen(pen_) { }
  virtual constexpr KeyEventType type() const override { return KEYEVENT_SIGNAL_PEN; }
};

// place material
template<> struct KeyEvent<KEYEVENT_MATERIAL> : public KeyEventBase
{
  MaterialPen<float> pen;
  float3             pos;
  KeyEvent(double t, const float3 &pos_, const MaterialPen<float> &pen_) : KeyEventBase(t), pos(pos_), pen(pen_) { }
  virtual constexpr KeyEventType type() const override { return KEYEVENT_MATERIAL; }
};

// set 2D view rect
template<> struct KeyEvent<KEYEVENT_VIEW2D> : public KeyEventBase
{
  Rect2f view;
  KeyEvent(double t, const Rect2f &view_) : KeyEventBase(t), view(view_) { }
  virtual constexpr KeyEventType type() const override { return KEYEVENT_VIEW2D; }
};

// set 3D view camera
template<> struct KeyEvent<KEYEVENT_VIEW3D> : public KeyEventBase
{
  CameraDesc<float> view;
  KeyEvent(double t, const CameraDesc<float> &view_) : KeyEventBase(t), view(view_) { }
  virtual constexpr KeyEventType type() const override { return KEYEVENT_VIEW3D; }
};

// adjust another setting
template<> struct KeyEvent<KEYEVENT_SETTING> : public KeyEventBase
{
  std::string id; // setting id
  std::any value; // setting value
  KeyEvent(double t, const std::string &id_, const std::any &val) : KeyEventBase(t), id(id_), value(val) { }
  virtual constexpr KeyEventType type() const override { return KEYEVENT_SETTING; }
};




#endif // KEY_FRAME_HPP
