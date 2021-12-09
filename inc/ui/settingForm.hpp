#ifndef SETTINGS_FORM_HPP
#define SETTINGS_FORM_HPP

#include <vector>
#include <string>
#include <nlohmann/json_fwd.hpp> // json forward declarations
using json = nlohmann::json;
#include "vector.hpp"

// forward declarations
class SettingBase;
class SettingGroup;

class SettingForm
{
private:
  SettingForm(const SettingForm &other) = delete; // : SettingForm(other.mTitle) { }
  SettingForm& operator=(const SettingForm &other) = delete; // { cleanup(); mTitle = other.mTitle; return *this; }
  
protected:
  std::vector<SettingBase*> mSettings;
  SettingGroup *mOther = nullptr;
  std::string mTitle = "";
  Vec2f mSize = Vec2f(888, 444);
  
public:
  SettingForm(const std::string &title);
  SettingForm() : SettingForm("") { }
  virtual ~SettingForm();
  
  void cleanup();
  
  json toJSON() const;
  bool fromJSON(const json &js);
  Vec2f getSize() const { return mSize; }
  void add(SettingBase *setting, bool other=false);
  void updateAll();
  
  virtual bool draw();
};

#endif // SETTINGS_FORM_HPP
