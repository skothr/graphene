#ifndef SETTING_HPP
#define SETTING_HPP

#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <imgui.h>
#include <imgui_internal.h>
#include <nlohmann/json.hpp> // full json headers
using json = nlohmann::json;

#include "vector.hpp"
#include "matrix.hpp"
#include "imtools.hpp"
#include "glfwKeys.hpp"
#include "ui.hpp"

#define DEFAULT_STEP       1
#define DEFAULT_STEP_FAST 10
#define DEFAULT_FORMAT "%.6f"

#define HELP_ICON      "?"
#define JSON_SPACES    4


// callback types
typedef std::function<void(void)> SettingUpdateCB;
typedef std::function<bool(void)> SettingEnabledCB;
typedef std::function<bool(void)> SettingDrawCB;

class SettingGroup;

////////////////////////////////
//// SETTING  -- BASE CLASS ////
////////////////////////////////
class SettingBase
{
protected:
  SettingGroup *mParent = nullptr;
  std::string mName;
  std::string mId;
  std::string mHelpText;
  bool  mHorizontal = true; // if true, contents are organized horizontally
  bool *mToggle  = nullptr; // if not null, adds an extra checkbox to enable/disable setting
  bool  mToggled = false;   // set to true if setting was toggled (since drawToggle() is called by parent group)
  
  float mBodyWidth = -1.0f; // width of setting body (-1.0f ==> remaining space in area)
  float mMinLabelW = 0.0f;  // minimum width of label column
  Vec2f mSize = Vec2f(0,0); // measured screen size

  // callbacks
  SettingUpdateCB  updateCallback  = nullptr; // called when value is updated
  SettingEnabledCB enabledCallback = nullptr; // custom draw callback (instead of using template overload below) 
  SettingEnabledCB visibleCallback = nullptr; // custom draw callback (instead of using template overload below)
  
public:
  SettingBase(const std::string &name_, const std::string &id_) : mName(name_), mId(id_) { }
  virtual ~SettingBase() = default;

  void setParent(SettingGroup *parent) { mParent = parent; }
  virtual bool isGroup() const { return false; }  // (overloaded by SettingGroup)
  
  // JSON (for file saving/loading)
  virtual json toJSON() const           { return json::object(); }
  virtual bool fromJSON(const json &js) { return true; }

  virtual std::string toString() const { std::stringstream ss; ss << name() << ": <?>"; return ss.str(); }
    
  virtual bool onDraw()    { return false; }
  virtual void updateAll() { if(updateCallback) { updateCallback(); } }

  void setUpdateCallback (const SettingUpdateCB  &cb) { updateCallback  = cb; }
  void setEnabledCallback(const SettingEnabledCB &cb) { enabledCallback = cb; }
  void setVisibleCallback(const SettingEnabledCB &cb) { visibleCallback = cb; }
  void setHelp(const std::string &text) { mHelpText = text; }  
  void setToggle(bool *toggle) { mToggle = toggle; } // adds an additional checkbox to enable/disable setting

  std::string& name()                 { return mName; }
  std::string& id()                   { return mId; }
  std::string& helpText()             { return mHelpText; }
  const std::string& name() const     { return mName; }
  const std::string& id() const       { return mId; }
  const std::string& helpText() const { return mHelpText; }
  
  bool horizontal() const    { return mHorizontal; }
  void setHorizontal(bool h) { mHorizontal = h; }
  float labelW() const       { return mMinLabelW; }
  void setLabelW(float w)    { mMinLabelW = w; }
  void setBodyW(float w)     { mBodyWidth = w; }

  Vec2f size() const { return mSize; }
  
  bool visible() const { return (!visibleCallback || visibleCallback()); }
  bool enabled() const { return (!enabledCallback || enabledCallback()) && (!mToggle || *mToggle); }
  
  bool draw();
  void drawHelp();
  bool drawToggle();
  std::ostream& print(std::ostream &os) const;

  friend std::ostream& operator<<(std::ostream &os, const SettingBase &s);
};

inline bool SettingBase::draw()
{
  bool changed = false;
  bool sVisible = visible();
  if(sVisible)
    {
      ImGui::SetNextItemWidth(mBodyWidth);
      
      if(!isGroup()) { ImGui::BeginDisabled(!enabled()); }
      changed |= onDraw();
      mSize = Vec2f(ImGui::GetItemRectMax()) - ImGui::GetItemRectMin();
      if(!isGroup()) { ImGui::EndDisabled(); }
    }
  if((changed || mToggled) && updateCallback) { updateCallback(); mToggled = false; }
  return changed;
}

inline void SettingBase::drawHelp()
{
  ImGuiStyle &style = ImGui::GetStyle();
  ImGui::AlignTextToFramePadding();
  if(!mHelpText.empty() || !name().empty())
    {
      Vec2f tSize = ImGui::CalcTextSize(HELP_ICON);
      ImGui::TextColored(Vec4f(1, 1, 1, 0.5f), HELP_ICON);
      if(ImGui::IsItemHovered())
        {
          ImGui::BeginTooltip();
          {
            std::stringstream ss; ss << name() << " (" << id() << ")";
            TextPhysics(ss.str()); //"%s (%s)", name().c_str(), id().c_str());
            if(!mHelpText.empty()) { ImGui::Separator(); ImGui::Spacing(); ImGui::Spacing(); TextPhysics(mHelpText.c_str()); }
          }
          ImGui::EndTooltip();
        }
      ImGui::SameLine();
    }
}

inline bool SettingBase::drawToggle()
{
  bool changed = false;
  if(mToggle)
    {
      bool t = *mToggle;
      changed |= ImGui::Checkbox(("##"+mId+"-toggle").c_str(), &t); ImGui::SameLine();
      mToggled = (*mToggle != t);
      *mToggle = t;
    }
  return changed;
}

inline std::ostream& operator<<(std::ostream &os, const SettingBase &s) { os << s.id() << " = " << s.toJSON(); return os; }



////////////////////////////////////
//// SETTING --  TEMPLATE CLASS ////
////////////////////////////////////
template<typename T>
class Setting : public SettingBase
{
protected:
  T *mData   = nullptr;
  T  mDefault;
  T  mStep    = T(); T mFastStep = T(); std::string mFormat = "";
  T  mMinVal  = T(); T mMaxVal   = T(); bool mMinSet = false; bool mMaxSet = false;
  bool mLiveEdit = true; // if false, waits until editing is complete to apply changes
  
public:
  typedef T type;
  Setting(const std::string &name, const std::string &id, T *pData, const T &defaultVal)
    : SettingBase(name, id), mData(pData), mDefault(defaultVal) { fixFormat(); }
  Setting(const std::string &name, const std::string &id, T *pData)
    : Setting(name, id, pData, (pData ? (*pData) : T())) // SettingBase(name, id), mData(pData), mDefault(pData ? (*pData) : T())
  { }
  ~Setting() = default;

  virtual std::string toString() const override { std::stringstream ss; ss << value(); return ss.str(); }
  
  const T& value() const { return *mData; }
  T& value()             { return *mData; }
  const T* data() const  { return *mData; }
  T* data()              { return *mData; }
  
  void setFormat(const T &step, const T &step_fast=T(), const std::string &format="");
  void setMin   (const T &minVal) { mMinVal = minVal; mMinSet = true; }
  void setMax   (const T &maxVal) { mMaxVal = maxVal; mMaxSet = true; }
  void setLiveEdit(bool live)     { mLiveEdit = live; }
  void fixFormat();
  
  // JSON
  virtual json toJSON() const override;
  virtual bool fromJSON(const json &js) override;
  
  virtual bool onDraw() { return false; } // (NOTE: should be protected)
};

///////////////////
//// SAVE/LOAD ////
///////////////////
template<typename T>
inline json Setting<T>::toJSON() const
{
  std::stringstream ss;
  if(mData)  { ss << std::fixed << (*mData); }
  json js = json::object();
  js["value"] = ss.str();
  if(mToggle) { js["active"] = *mToggle; }
  return js;
}
template<typename T>
inline bool Setting<T>::fromJSON(const json &js)
{
  bool success = true;
  std::stringstream ss;
  if(js.contains("value"))
    {
      ss.str(js["value"].get<std::string>());
      ss >> (*mData);
    }
  else { success = false; std::cout << "======> WARNING(Setting<T>::fromJSON): Couldn't find setting '" << name() << "' (" << id() << ") -- using default\n"; }
  if(mToggle)
    {
      if(js.contains("active"))
        {
          try { *mToggle = (js["active"].get<bool>()); }
          catch(const json::type_error &e)
            {
              std::cout << "======> WARNING(Setting<T>::fromJSON): Invalid type found for setting '" << name() << "' (" << id() << ") -- using default\n";
              std::cout << "========> " << e.what() << "\n";
              success = false;
            }
          catch(...)
            {
              std::cout << "======> WARNING(Setting<T>::fromJSON): Unhandled exception loading setting '" << name() << "' (" << id() << ") -- using default\n";
              success = false;
            }
        }
      else
        { success = false; std::cout << "======> WARNING(Setting<T>::fromJSON): Setting (togglable) couldn't find 'active'\n"; }
    }
  return success;
}


template<typename T>
inline void Setting<T>::setFormat(const T &step, const T &step_fast, const std::string &format)
{
  if constexpr(std::is_arithmetic_v<typename cuda_vec<T>::BASE>)
    { mStep = step; mFastStep = step_fast; mFormat = format; }
  fixFormat();
}
template<typename T>
inline void Setting<T>::fixFormat()
{
  // step sizes only needed for intN/floatN/doubleN
  if constexpr(std::is_arithmetic_v<T>)
    { // scalar types
      if(mStep == T(0)) { mStep = T(1); } if(mFastStep == T(0)) { mFastStep = T(10); }
    }
  else if constexpr(std::is_arithmetic_v<typename cuda_vec<T>::BASE>)
    { // vector types
      for(int i = 0; i < cuda_vec<T>::N; i++)
        {
          if(VElement<T>::get(mStep,     i) == 0) { VElement<T>::get(mStep,     i) = DEFAULT_STEP;      }
          if(VElement<T>::get(mFastStep, i) == 0) { VElement<T>::get(mFastStep, i) = DEFAULT_STEP_FAST; }
        }
    }
  else
    {
      for(int i = 0; i < cuda_vec<T>::N; i++)
        {
          if(mStep[i]     == 0) { mStep[i]     = DEFAULT_STEP;      }
          if(mFastStep[i] == 0) { mFastStep[i] = DEFAULT_STEP_FAST; }
        }
    }
  // printf format only needed for floatN/doubleN
  if constexpr(std::is_floating_point_v<typename cuda_vec<T>::BASE>)
    { if(mFormat.empty()) { mFormat = DEFAULT_FORMAT; } }
}







  
//// SETTING DRAW SPECIALIZATIONS (BY TYPE) ////

//// BOOLEAN -- Checkbox ////
template<> inline bool Setting<bool>::onDraw() { return ImGui::Checkbox(("##"+mId).c_str(), mData); }

//// basic single types ////
template<> inline bool Setting<int>::onDraw()         // INT
{
  int v = *mData; bool changed = false;
  if(ImGui::InputInt(("##"+mId).c_str(), &v, mStep, mFastStep))
    { if(mMaxSet) { v = std::min(mMaxVal, v); } if(mMinSet) { v = std::max(mMinVal, v); } *mData = v; changed = true; }
  return changed;
}
template<> inline bool Setting<float>::onDraw()       // FLOAT
{
  float v = *mData; bool changed = false;
  if(ImGui::InputFloat(("##"+mId).c_str(), &v, mStep, mFastStep, mFormat.c_str()))
    { if(mMaxSet) { v = std::min(mMaxVal, v); } if(mMinSet) { v = std::max(mMinVal, v); } *mData = v; changed = true; }
  return changed;
}
template<> inline bool Setting<double>::onDraw()      // DOUBLE
{
  double v = *mData; bool changed = false;
  if(ImGui::InputDouble(("##"+mId).c_str(), &v, 1.0f, 10.0f))
    { if(mMaxSet) { v = std::min(mMaxVal, v); } if(mMinSet) { v = std::max(mMinVal, v); } *mData = v; changed = true; }
  return changed;
}
template<> inline bool Setting<std::string>::onDraw() // STRING
{
  char str[1024] = {0}; std::copy(mData->begin(), mData->end(), str); bool changed = false;
  if(ImGui::InputText(("##"+mId).c_str(), str, 1024)) { *mData = str; changed = true; }
  return changed;
}



////////////////////////////////////////
//// onDraw() vector specialization ////
////////////////////////////////////////
// (helper macros)
#define DATA_STEP_SET  mId, mData, mStep, mFastStep
#define MAX_MIN_SET    (mMinSet ? &mMinVal : nullptr), (mMaxSet ? &mMaxVal : nullptr)
#define HORIZ_LIVE_EXTEND_SET horizontal(), mLiveEdit, (mToggle == nullptr)
// INT vectors
template<> inline bool Setting<int2 >::onDraw()   { return InputIntV   (DATA_STEP_SET, MAX_MIN_SET,          HORIZ_LIVE_EXTEND_SET); }
template<> inline bool Setting<int3 >::onDraw()   { return InputIntV   (DATA_STEP_SET, MAX_MIN_SET,          HORIZ_LIVE_EXTEND_SET); }
template<> inline bool Setting<int4 >::onDraw()   { return InputIntV   (DATA_STEP_SET, MAX_MIN_SET,          HORIZ_LIVE_EXTEND_SET); }
template<> inline bool Setting<Vec2i>::onDraw()   { return InputIntV   (DATA_STEP_SET, MAX_MIN_SET,          HORIZ_LIVE_EXTEND_SET); }
template<> inline bool Setting<Vec3i>::onDraw()   { return InputIntV   (DATA_STEP_SET, MAX_MIN_SET,          HORIZ_LIVE_EXTEND_SET); }
template<> inline bool Setting<Vec4i>::onDraw()   { return InputIntV   (DATA_STEP_SET, MAX_MIN_SET,          HORIZ_LIVE_EXTEND_SET); }
// FLOAT vectors
template<> inline bool Setting<float2>::onDraw()  { return InputFloatV (DATA_STEP_SET, MAX_MIN_SET, mFormat, HORIZ_LIVE_EXTEND_SET); }
template<> inline bool Setting<float3>::onDraw()  { return InputFloatV (DATA_STEP_SET, MAX_MIN_SET, mFormat, HORIZ_LIVE_EXTEND_SET); }
template<> inline bool Setting<float4>::onDraw()  { return InputFloatV (DATA_STEP_SET, MAX_MIN_SET, mFormat, HORIZ_LIVE_EXTEND_SET); }
template<> inline bool Setting<Vec2f >::onDraw()  { return InputFloatV (DATA_STEP_SET, MAX_MIN_SET, mFormat, HORIZ_LIVE_EXTEND_SET); }
template<> inline bool Setting<Vec3f >::onDraw()  { return InputFloatV (DATA_STEP_SET, MAX_MIN_SET, mFormat, HORIZ_LIVE_EXTEND_SET); }
template<> inline bool Setting<Vec4f >::onDraw()  { return InputFloatV (DATA_STEP_SET, MAX_MIN_SET, mFormat, HORIZ_LIVE_EXTEND_SET); }
// DOUBLE vectors
template<> inline bool Setting<double2>::onDraw() { return InputDoubleV(DATA_STEP_SET, MAX_MIN_SET, mFormat, HORIZ_LIVE_EXTEND_SET); }
template<> inline bool Setting<double3>::onDraw() { return InputDoubleV(DATA_STEP_SET, MAX_MIN_SET, mFormat, HORIZ_LIVE_EXTEND_SET); }
template<> inline bool Setting<double4>::onDraw() { return InputDoubleV(DATA_STEP_SET, MAX_MIN_SET, mFormat, HORIZ_LIVE_EXTEND_SET); }
template<> inline bool Setting<Vec2d  >::onDraw() { return InputDoubleV(DATA_STEP_SET, MAX_MIN_SET, mFormat, HORIZ_LIVE_EXTEND_SET); }
template<> inline bool Setting<Vec3d  >::onDraw() { return InputDoubleV(DATA_STEP_SET, MAX_MIN_SET, mFormat, HORIZ_LIVE_EXTEND_SET); }
template<> inline bool Setting<Vec4d  >::onDraw() { return InputDoubleV(DATA_STEP_SET, MAX_MIN_SET, mFormat, HORIZ_LIVE_EXTEND_SET); }
////////////////////////////////////////////////////////////////////////////////////////////////////




// just displays text
class TextSetting : public Setting<std::string>
{
protected:
  virtual bool onDraw() override;

public:
  TextSetting(const std::string &name, const std::string &id, std::string *text)
    : Setting<std::string>(name, id, text) { }
  ~TextSetting() = default;

  // JSON (don't store)
  virtual json toJSON() const override { return nullptr; }
  virtual bool fromJSON(const json &js) override { return true; }
};

inline bool TextSetting::onDraw()
{ // STATIC TEXT

  TextPhysics(mData->c_str());
  return false;
}




////////////////////////////////
//// COMBOBOX SETTING (int) ////
////////////////////////////////
class ComboSetting : public Setting<int>
{
protected:
  std::vector<std::string> mChoices;
  virtual bool onDraw() override;

public:
  ComboSetting(const std::string &name, const std::string &id)
    : Setting<int>(name, id, nullptr, 0) { }
  ComboSetting(const std::string &name, const std::string &id, int *selection, const std::vector<std::string> &choices)
    : Setting<int>(name, id, selection), mChoices(choices) { }
  ComboSetting(const std::string &name, const std::string &id, int *selection, const std::vector<std::string> &choices, int defaultVal)
    : Setting<int>(name, id, selection, defaultVal), mChoices(choices) { }
  ~ComboSetting() = default;
    
  // JSON
  virtual json toJSON() const override;
  virtual bool fromJSON(const json &js) override;
};

inline json ComboSetting::toJSON() const
{
  std::stringstream ss;
  json combo = json::object();
  if(mData)  { ss << mChoices[*mData]; }
  combo["selection"] = ss.str();
  return combo;
}
inline bool ComboSetting::fromJSON(const json &js)
{
  bool success = true;
  if(js.contains("selection"))
    {
      std::stringstream ss; ss << js["selection"].get<std::string>();
      const auto &iter = std::find(mChoices.begin(), mChoices.end(), ss.str());
      if(iter != mChoices.end())
        { *mData = (iter - mChoices.begin()); }
      else
        {
          success = false;
          std::cout << "======> WARNING(ComboSetting::fromJSON): Couldn't find '" << ss.str() << "' (" << (*mData) << ") in choices! (" << id() << ")\n";
        }
    }
  else { success = false; std::cout << "======> WARNING(ComboSetting::fromJSON): Couldn't find setting 'selection'\n"; }
  return success;
}
inline bool ComboSetting::onDraw()
{ // COMBOBOX
  bool changed = false;
  if(ImGui::BeginCombo(("##"+mId).c_str(), mChoices[*mData].c_str()))
    {
      for(int i = 0; i < mChoices.size(); i++)
        {
          std::string &s = mChoices[i];
          if(ImGui::Selectable(((i == *mData ? "* " : "") + mChoices[i]).c_str())) { changed = true; *mData = i; }
        }
      ImGui::EndCombo();
    }
  return changed;
}


////////////////////////////////
//// COLOR SETTING (float4) ////
////////////////////////////////
class ColorSetting : public Setting<float4>
{
public:
  ColorSetting(const std::string &name, const std::string &id, float4 *val)
    : Setting<float4>(name, id, val) { }
  ColorSetting(const std::string &name, const std::string &id, float4 *val, float4 defaultVal)
    : Setting<float4>(name, id, val, defaultVal) { }
  
  virtual bool onDraw() override;
};

inline bool ColorSetting::onDraw()
{
  static float4 lastColor; // save previous color in case user cancels
  static std::string editId = "";
  std::string buttonName = "##" + mId + "btn";
  std::string popupName  = "##" + mId + "pop";
  std::string pickerName = "##" + mId + "pick";

  bool changed = false;
  // choose graph background color
  ImGuiColorEditFlags cFlags = (ImGuiColorEditFlags_DisplayRGB |
                                ImGuiColorEditFlags_HDR        |
                                ImGuiColorEditFlags_Float      |
                                ImGuiColorEditFlags_AlphaBar);
  if(ImGui::ColorButton(buttonName.c_str(), Vec4f(*mData), cFlags, ImVec2(20, 20)))
    {
      lastColor = *mData;
      ImGui::OpenPopup(popupName.c_str());
    }
  ImGuiWindowFlags wFlags = (ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoMove           |
                             ImGuiWindowFlags_NoTitleBar       |
                             ImGuiWindowFlags_NoResize );
  if(ImGui::BeginPopup(popupName.c_str(), wFlags))
    {
      bool hover = ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows);
      float4 d = *mData; float4 last = lastColor;
      if(ImGui::ColorPicker4(pickerName.c_str(), arr(d), cFlags, arr(last))) { *mData = d; changed = true; }
      
      hover |= ImGui::IsItemHovered();   
      if(ImGui::Button("Select") || ImGui::IsKeyPressed(GLFW_KEY_ENTER) || (!hover && ImGui::IsMouseClicked(ImGuiMouseButton_Left)))
        { ImGui::CloseCurrentPopup(); }                                     // select color
      
      ImGui::SameLine();
      if(ImGui::Button("Cancel") || ImGui::IsKeyPressed(GLFW_KEY_ESCAPE))
        { *mData = lastColor; changed = true; ImGui::CloseCurrentPopup(); } // cancel current selection
      
      ImGui::SameLine();
      if(ImGui::Button("Reset")) { *mData = mDefault; changed = true; }     // reset to default
      
      ImGui::EndPopup();
    }
  return changed;
}





/////////////////////////////////////////////
//// SLIDER SETTING (int/float/double-2) ////
/////////////////////////////////////////////
// TODO: rename RangeSetting
// TODO: SliderSetting --> single slider position (OR: combine? multi-sliders)?
template<typename T, typename VT2=typename cuda_vec<T,2>::VT>
class SliderSetting : public Setting<VT2>
{
// public:
//   typedef Setting<VT2> s;
private:
  using Setting<VT2>::mData;
  using Setting<VT2>::id;
  T *mRangeMin = nullptr;
  T *mRangeMax = nullptr;
public:
  SliderSetting(const std::string &name, const std::string &id, VT2 *range, T *rMin=nullptr, T *rMax=nullptr)
    : Setting<VT2>(name, id, range),               mRangeMin(rMin), mRangeMax(rMax) { }
  SliderSetting(const std::string &name, const std::string &id, VT2 *range, const VT2 &defaultRange, T *rMin=nullptr, T *rMax=nullptr)
    : Setting<VT2>(name, id, range, defaultRange), mRangeMin(rMin), mRangeMax(rMax) { }
  virtual bool onDraw() override;
};

template<typename T, typename VT2>
inline bool SliderSetting<T, VT2>::onDraw()
{
  return RangeSlider(("##slider-"+id()).c_str(), &mData->x, &mData->y, (mRangeMin ? *mRangeMin : 0), (mRangeMax ? *mRangeMax-1 : 0));
}





//// MATRIX SETTING ////
template<typename T, int N, int M=N>
class MatrixSetting : public Setting<Matrix<T, N, M>>
{
private:
  using Setting<Matrix<T,N,M>>::mData;
  using Setting<Matrix<T,N,M>>::mToggle;
  using Setting<Matrix<T,N,M>>::name;
  using Setting<Matrix<T,N,M>>::id;
  
protected:
  virtual bool onDraw() override;
  
public:
  typedef Matrix<T, N, M> Mat;
  
  MatrixSetting(const std::string &name_, const std::string &id_, Mat *mat)
    : Setting<Matrix<T, N, M>>(name_, id_, mat) { }
  ~MatrixSetting() = default;
  
  // JSON
  virtual json toJSON() const override;
  virtual bool fromJSON(const json &js) override;
};

template<typename T, int N, int M>
inline json MatrixSetting<T,N,M>::toJSON() const
{
  json jr = json::array();
  if(mData)
    {
      for(int r = 0; r < 4; r++)
        {
          json row = json::array();
          for(int c = 0; c < 4; c++) { row.push_back((*mData)[r][c]); }
          jr.push_back(row);
        }
    }
  json js = json::object();
  js["data"] = jr;
  if(mToggle) { js["active"] = *mToggle; }
  return js;
}

template<typename T, int N, int M>
inline bool MatrixSetting<T,N,M>::fromJSON(const json &js)
{
  bool success = true;
  if(mData && js.contains("data"))
    {
      json jr = js["data"];
      for(int r = 0; r < 4; r++)
        for(int c = 0; c < 4; c++)
          { (*mData)[r][c] = jr[r][c]; }
    }
  else { success = false; }
  if(mToggle)
    {
      if(js.contains("active")) { *mToggle = (js["active"].get<bool>()); }
      else { success = false; }
    }
  return success;
}

template<typename T, int N, int M>
inline bool MatrixSetting<T,N,M>::onDraw()
{
  bool changed = false;
  ImGuiTableFlags tableFlags     = (ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_PreciseWidths |
                                    ImGuiTableFlags_NoPadOuterX | ImGuiTableFlags_Borders | ImGuiTableFlags_NoClip);
  ImGuiTableColumnFlags colFlags = ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_NoClip;
  float hSpace = ImGui::GetContentRegionMax().y - ImGui::GetCursorPos().y;
  float colW  = hSpace/4.0f;
  if(ImGui::BeginTable(id().c_str(), M, tableFlags))
    {
      for(int i = 0; i < M; i++) { ImGui::TableSetupColumn(std::to_string(i).c_str(), colFlags, colW); }
      for(int r = 0; r < N; r++)
        {
          for(int c = 0; c < M; c++)
            { // display matrix contents
              std::stringstream ss; ss << std::right << std::setw(10) << std::fixed << std::setprecision(4) << (*mData)[r][c];
              ImGui::TableNextColumn(); ImGui::SetNextItemWidth(-1);
              TextPhysics(ss.str());
              if(c <= M-1) { ImGui::SameLine(colW*(c+1), 0); }
            }
        }
      ImGui::EndTable();
    }
  return changed;
}









////////////////////////
//// SETTING GROUP  ////
////////////////////////
class SettingGroup : public SettingBase
{
protected:
  std::vector<SettingBase*> mContents;
  bool mCollapsible = false; // if true, group is collapsible (collapsing header vs. text title)
  bool mTree        = false; // if true, group is collapsible as an ImGui TreeNode
  bool mIndent      = true;  // if true, group contents are indented
  bool mDrawLabels  = true;  // if true, child labels are drawn
  bool mCenter      = false; // if true, children are centered within columns
  bool mOpen        = false; // false if group is collapsed
  
  int  mNumColumns  = 1;     // number of setting columns for contents
  std::vector<std::string> mRowLabels;
  std::vector<std::string> mColLabels;
  
  bool drawContents(); // draws settings arranged in columns
  virtual bool onDraw() override;
  
public:
  SettingGroup(const std::string &name_, const std::string &id_, const std::vector<SettingBase*> &contents={ })
    : SettingBase(name_, id_), mContents(contents)
  { setHorizontal(false); for(auto s : mContents) { s->setParent(this); } }
  ~SettingGroup() { clear(); }
  
  virtual bool isGroup() const override { return true; }
  
  // JSON
  virtual json toJSON() const override;
  virtual bool fromJSON(const json &js) override;

  std::vector<SettingBase*>&       contents()       { return mContents; }
  const std::vector<SettingBase*>& contents() const { return mContents; }

  void setCollapsible(bool collapsible) { mCollapsible = collapsible; }
  void setTree(bool tree)     { mTree       = tree; }
  void setColumns(int nCols)  { mNumColumns = nCols; }
  void setCenter(int center)  { mCenter     = center; }
  void setIndent(bool indent) { mIndent     = indent; }
  void setDrawLabels(bool dl) { mDrawLabels = dl; }
  
  void setColumnLabels(const std::vector<std::string> &cols) { mColLabels = cols; }
  void setRowLabels   (const std::vector<std::string> &rows) { mRowLabels = rows; }

  void add(SettingBase *setting) { setting->setParent(this); mContents.push_back(setting); }
  void clear() { for(auto &s : mContents) { if(s) { delete s; s = nullptr; } } mContents.clear(); }
    
  const bool& open() const { return mOpen; }
  bool& open()             { return mOpen; }
  void setOpen(bool gopen) { mOpen = gopen; }

  virtual void updateAll() override
  {
    for(auto s : mContents) { s->updateAll(); }
    if(updateCallback) { updateCallback(); }
  }
  
};
  
//// SETTING GROUP SAVE/LOAD ////
inline json SettingGroup::toJSON() const
{
  json js = json::object();
  if(mCollapsible) { js["open"]   =  mOpen; }
  if(mToggle)      { js["active"] = *mToggle; }
  json contents = json::object();
  for(auto s : mContents) { contents[s->id()] = s->toJSON(); }
  js["contents"] = contents;
  return js;
}
inline bool SettingGroup::fromJSON(const json &js)
{
  bool success = true;
  if(mCollapsible)
    {
      if(js.contains("open"))   { mOpen = js["open"].get<bool>(); }
      else                      { success = false; std::cout << "======> WARNING(SettingGroup::fromJSON): SettingGroup (collapsible) couldn't find 'open'\n"; }
    }
  if(mToggle)
    {
      if(js.contains("active")) { *mToggle = js["active"].get<bool>(); }
      else                      { success = false; std::cout << "======> WARNING(SettingGroup::fromJSON): SettingGroup (togglable)Couldn't find 'active'\n"; }
    }
  if(js.contains("contents"))
    {
      json contents = js["contents"];
      if(js["contents"].size() == mContents.size())
        {
          for(int i = 0; i < mContents.size(); i++)
            {
              if(js["contents"].contains(mContents[i]->id()))
                { success &= mContents[i]->fromJSON(js["contents"][mContents[i]->id()]); }
              else
                { success = false; std::cout << "======> WARNING(SettingGroup::fromJSON): Couldn't find '" << mContents[i]->id() << "'\n"; }
            }
        }
    }
  else { success = false; std::cout << "======> WARNING(SettingGroup::fromJSON): Couldn't find 'contents'\n"; success = false; }
  return success;
}

//// SETTING GROUP DRAW //// 
inline bool SettingGroup::drawContents()
{
  ImGuiStyle &style = ImGui::GetStyle();
  bool changed = false;

  Vec2f p0 = ImGui::GetCursorPos();
  
  bool rlabel = false;
  float maxLabelW = mMinLabelW; // width  of row labels
  for(int i = 0; i < (int)std::ceil(mContents.size()/(float)mNumColumns); i++)
    {
      const char *label = (i < mRowLabels.size() ? mRowLabels[i] : mContents[i*mNumColumns]->name()).c_str();
      if(!std::string(label).empty()) { rlabel = true; }
      maxLabelW = std::max(maxLabelW, ImGui::CalcTextSize(label).x);
    }
  float paddedLabelW = maxLabelW;
  if(mHorizontal) { paddedLabelW += ImGui::CalcTextSize("XX").x; }
  else            { paddedLabelW += ImGui::CalcTextSize(HELP_ICON "XXXX").x; }
  
  int rCol = (mDrawLabels && rlabel ? 1 : 0);
  int nCols = mNumColumns + rCol;

  int tableNum = 0; // used if multiple sub-tables needed
  ImGuiTableFlags tFlags = (ImGuiTableFlags_SizingStretchProp |
                            ImGuiTableFlags_PreciseWidths |
                            ImGuiTableFlags_NoPadOuterX |
                            ImGuiTableFlags_NoClip);
  ImGuiTableColumnFlags lColFlags = ImGuiTableColumnFlags_WidthFixed   | ImGuiTableColumnFlags_NoClip;
  ImGuiTableColumnFlags sColFlags = ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_NoClip;

  bool inTable = false; // keeps track of whether table needs to be ended
  ImGui::SetNextItemWidth(mBodyWidth);
  if(ImGui::BeginTable(mId.c_str(), nCols, tFlags))
    {
      inTable = true;
      for(int i = 0; i < nCols; i++)
        {
          if(rCol && i == 0) { ImGui::TableSetupColumn("rowLabels", lColFlags, paddedLabelW); }
          else
            {
              if(i-rCol < mColLabels.size()) { ImGui::TableSetupColumn(mColLabels[i-rCol].c_str(), sColFlags, maxLabelW); }
              else                           { ImGui::TableSetupColumn("",                         sColFlags); }
            }
        }
      ImGui::TableNextColumn();

      // column labels
      for(int i = 0; i < mNumColumns; i++)
        {
          ImGui::TableSetColumnIndex(i + rCol);
          if(i < mColLabels.size())
            {
              Vec2f tSize = ImGui::CalcTextSize(mColLabels[i].c_str());
              const ImGuiTableColumn &column = ImGui::GetCurrentTable()->Columns[ImGui::GetCurrentTable()->CurrentColumn];
              float columnW = column.WorkMaxX - column.WorkMinX; // ~hack
              float textW = ImGui::CalcTextSize(mColLabels[i].c_str()).x;
              ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos()) + Vec2f(std::floor((columnW - textW)/2.0f), 0));
              TextPhysics(mColLabels[i].c_str());
            }
        }
      if(mColLabels.size() > 0) { ImGui::TableNextRow(); }
      ImGui::TableSetColumnIndex(rCol);

      // draw row labels and children
      bool gEnabled = enabled();
      ImGui::BeginDisabled(!gEnabled);
      for(int i = 0; i < mContents.size(); i++)
        {
          bool newTable      = false; // set to true if child is a non-horizontal group (so expand to full group width)
          int  col = (i % mNumColumns); int row = (i / mNumColumns);
          if(col == 0)
            { // skip row if nothing visible
              bool rowVisible = false;
              for(int j = 0; j < mNumColumns && (i+j) < mContents.size(); j++)
                { rowVisible |= mContents[i+j]->visible(); }
              if(!rowVisible) { i += mNumColumns; i--; continue; }

              float colH  = 0.0f; // (height of child column labels)
              if(mContents[i]->isGroup())
                {
                  SettingGroup *g = static_cast<SettingGroup*>(mContents[i]);
                  newTable = !g->horizontal();                  
                  // offset toggle/help by height of columns
                  for(int j = 0; j < g->mNumColumns; j++)
                    { if(j < g->mColLabels.size()) { colH = std::max(colH, ImGui::CalcTextSize(g->mColLabels[j].c_str()).y); } }
                  if(colH > 0.0f) { colH += style.CellPadding.y*2.0f; } // padding around label text
                  colH += style.CellPadding.y;                          // table padding offset for alignment
                }

              // start of row
              if(i != 0) { ImGui::TableNextRow(); }
              ImGui::TableSetColumnIndex(0);

              if(rCol && (!mContents[i]->isGroup() || reinterpret_cast<SettingGroup*>(mContents[i])->horizontal()))
                { // draw row label column
                  ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos()) + Vec2f(0, colH));
                  if(mContents[i]->visible()) { mContents[i]->drawHelp(); }
                  
                  if(mRowLabels.size() > row && !mRowLabels[row].empty()) { TextPhysics(mRowLabels[row].c_str());      ImGui::SameLine(); }
                  else if(!mContents[i]->name().empty())                  { TextPhysics(mContents[i]->name().c_str()); ImGui::SameLine(); }
                  
                  ImGui::TableNextColumn();
                  ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos()) + Vec2f(0, colH));
                  if(mContents[i]->visible()) { changed |= mContents[i]->drawToggle(); }
                  ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos()) - Vec2f(0, colH));
                }
            }
          else if(mContents[i]->visible())
            { // still draw help and toggle
              ImGui::TableNextColumn();
              mContents[i]->drawHelp();
              changed |= mContents[i]->drawToggle();
              if(!mContents[i]->name().empty()) { TextPhysics(mContents[i]->name().c_str()); ImGui::SameLine(); }
            }
          ImGui::TableSetColumnIndex(col + rCol);
          float offset = ImGui::GetCursorPos().x - p0.x;

          // split main table if necessary
          if(newTable && inTable)
            {
              ImGui::EndTable(); inTable = false;
              ImGui::SetCursorPos(Vec2f(p0.x, ImGui::GetCursorPos().y));
              reinterpret_cast<SettingGroup*>(mContents[i])->setLabelW(offset - 3*style.IndentSpacing - style.CellPadding.x/2.0f);
            }

          // draw
          if(mCenter)
            {
              // center contents in column (~hack)
              const ImGuiTableColumn &column = ImGui::GetCurrentTable()->Columns[ImGui::GetCurrentTable()->CurrentColumn];
              float columnW = column.WorkMaxX - column.WorkMinX; // ~hack
              ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos()) + Vec2f((columnW - mContents[i]->size().x)/2.0f, 0));
            }
          changed |= mContents[i]->draw();
          
          if(newTable && !inTable)
            {
              if(i < mContents.size()-1)
                { // begin new table for remaining contents
                  if(ImGui::BeginTable((mId+"-"+std::to_string(tableNum)).c_str(), nCols, tFlags))
                    {
                      tableNum++; inTable = true;
                      for(int j = 0; j < nCols; j++)
                        {
                          if(rCol && j == 0) { ImGui::TableSetupColumn("rowLabels", lColFlags, paddedLabelW); }
                          else
                            {
                              if(j-rCol < mColLabels.size()) { ImGui::TableSetupColumn(mColLabels[j-rCol].c_str(), sColFlags); }
                              else                           { ImGui::TableSetupColumn("",                         sColFlags); }
                            }
                        }
                      ImGui::TableNextColumn();
                    }
                }
            }
        }
      ImGui::EndDisabled();
      if(inTable) { ImGui::EndTable(); }
    }
  mMinLabelW = maxLabelW;
  // propagate label column width
  for(auto s : mContents) { if(s->isGroup() && !s->horizontal()) { mMinLabelW = std::max(mMinLabelW, s->labelW()); } }
  for(auto s : mContents) { if(s->isGroup() && !s->horizontal()) { s->setLabelW(mMinLabelW); } }
  return changed;
}

inline bool SettingGroup::onDraw()
{ // SETTING GROUP
  bool changed = false;
  if(mCollapsible || mTree)
    {
      ImGuiTreeNodeFlags treeFlags = (ImGuiTreeNodeFlags_FramePadding |
                                      ImGuiTreeNodeFlags_SpanAvailWidth);
      ImGui::SetNextTreeNodeOpen(mOpen); ImGui::SetNextItemWidth(mBodyWidth);
      if((mCollapsible && ImGui::CollapsingHeader(mName.c_str(), nullptr, treeFlags)) ||
         (mTree && ImGui::TreeNodeEx(mName.c_str(), treeFlags)))
        {
          mOpen = true;
          if(mIndent && !mHorizontal) { ImGui::Indent(); }
          changed |= drawContents();
          if(mIndent && !mHorizontal) { ImGui::Unindent(); }
          if(mTree) { ImGui::TreePop(); }
        }
      else { mOpen = false; }
    }
  else
    {
      mOpen = true; // no collapse -- always open
      if(!mHorizontal && !mName.empty())
        {
          drawHelp();
          TextPhysics(mName.c_str()); ImGui::SameLine(); // draw title
          changed |= drawToggle();
          TextPhysics(""); ImGui::Separator(); ImGui::Spacing();
        }

      if(mIndent && !mHorizontal && !mName.empty()) { ImGui::Indent(); }
      changed |= drawContents();
      if(mIndent && !mHorizontal && !mName.empty()) { ImGui::Unindent(); }
    }
  if(changed && updateCallback) { updateCallback(); }

  return changed;
}










// helper -- makes a group of settings referencing a vector of values
template<typename T>
inline SettingGroup* makeSettingGroup(const std::string &name, const std::string &id, std::vector<T> *contentData)
{
  if(!contentData) { return nullptr; }
  std::vector<SettingBase*> contents;
  for(int i = 0; i < contentData->size(); i++)
    {
      std::string index = std::to_string(i);
      contents.push_back(new Setting<T>(name+index, id+index, &contentData->at(i)));
    }
  SettingGroup *g = new SettingGroup(name, id, contents);
  return g;
}
// helper -- makes a group of settings from an array of values  
template<typename T, int N>
inline SettingGroup* makeSettingGroup(const std::string &name, const std::string &id, std::array<T, N> *contentData)
{
  if(!contentData) { return nullptr; }
  std::vector<SettingBase*> contents;
  for(int i = 0; i < N; i++)
    {
      std::string index = std::to_string(i);
      contents.push_back(new Setting<T>(/*name+" "+index*/ "", id+index, &contentData->at(i)));
    }
  SettingGroup *g = new SettingGroup(name, id, contents);
  return g;
}

// helper -- makes a group of settings from an array of values
template<typename T, int COLS>
inline SettingGroup* makeSettingGrid(const std::string &name, const std::string &id, const std::vector<std::array<T, COLS>*> &contentData,
                                     const std::vector<std::string> &rowLabels={})
{
  std::vector<SettingBase*> contents;
  for(int r = 0; r < contentData.size(); r++)
    {
      std::string ri = std::to_string(r);
      if(contentData[r])
        {
          for(int c = 0; c < COLS; c++)
            {
              std::string ci = std::to_string(c);
              contents.push_back(new Setting<T>("", id+"("+ri+","+ci+")", &contentData[r]->at(c)));
            }
        }
    }
  SettingGroup *g = new SettingGroup(name, id, contents);
  g->setRowLabels(rowLabels);
  g->setColumns(COLS);
  g->setHorizontal(true);
  return g;
}

#endif // SETTING_HPP
