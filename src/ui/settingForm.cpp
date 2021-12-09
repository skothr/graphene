#include "settingForm.hpp"

#include <imgui.h>
#include <algorithm>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "setting.hpp"
#include "glfwKeys.hpp"


SettingForm::SettingForm(const std::string &title)
  : mTitle(title)
{
  mOther = new SettingGroup("Other", "other");
}
SettingForm::~SettingForm() { cleanup(); if(mOther) { delete mOther; mOther = nullptr; } }

void SettingForm::cleanup()
{
  for(auto s : mSettings) { if(s) { delete s; } } mSettings.clear();
  mOther->clear();
}

json SettingForm::toJSON() const
{
  json js = json::object();
  for(auto s : mSettings) { js[s->id()] = s->toJSON(); }
  js[mOther->id()] = mOther->toJSON();
  return js;
}

bool SettingForm::fromJSON(const json &js)
{
  bool success = true;
  for(auto s : mSettings)
    {
      if(js.contains(s->id())) { s->fromJSON(js[s->id()]); }
      else { std::cout << "====> WARNING: SettingForm couldn't find '" << s->id() << "'\n"; success = false; }
    }
  if(js.contains(mOther->id()))
    {
      if(!mOther->fromJSON(js[mOther->id()])) { success = false; std::cout << "====> WARNING: SettingForm couldn't find '" << mOther->id() << "' (Other)\n"; }
    }
  else { std::cout << "====> WARNING: SettingForm couldn't find main settings\n"; success = false; }
  return success;
}

void SettingForm::add(SettingBase *setting, bool other)
{
  if(setting)
    {
      if(!other && setting->isGroup() && !setting->horizontal()) { mSettings.push_back(setting); }
      else                                                       { mOther->add(setting); } // extra group for any non-group settings
    }
}

bool SettingForm::draw()
{
  ImGuiStyle &style = ImGui::GetStyle();  
  Vec2f p0 = ImGui::GetCursorPos();
  
  // unify label column widths
  float labelW = 0.0f;
  for(auto s : mSettings) { if(s->isGroup() && !s->horizontal()) { labelW = std::max(labelW, s->labelW()); } }
  if(mOther->contents().size() > 0) { labelW = std::max(labelW, mOther->labelW()); }
  // propagate label column widths
  for(auto s : mSettings) { if(s->isGroup() && !s->horizontal()) { s->setLabelW(labelW); } }
  if(mOther->contents().size() > 0) { mOther->setLabelW(labelW); }
  
  bool changed = false;
  ImGui::BeginGroup();
  {
    for(auto s : mSettings)           { changed |= s->draw(); }      // setting groups
    if(mOther->contents().size() > 0) { changed |= mOther->draw(); } // other settings (no group)
  }
  ImGui::EndGroup();
  mSize = Vec2f(ImGui::GetItemRectMax()) - ImGui::GetItemRectMin(); // update size

  return changed;
}

void SettingForm::updateAll()
{
  for(int i = 0; i < mSettings.size(); i++) { mSettings[i]->updateAll(); }
  mOther->updateAll();
}
