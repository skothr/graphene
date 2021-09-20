#include "keyManager.hpp"

#include <bitset>

#include <imgui.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "simWindow.hpp"


KeyManager::KeyManager(SimWindow *parent, const std::vector<KeyBinding> &bindings, const std::vector<KeyBindingGroup> &groups)
  : mParent(parent), mDefaultKeyBindings(bindings), mKeyBindings(bindings), mKeyBindingGroups(groups)
{
  // set group ids -- maps binding in group to actual binding
  std::vector<std::string> miscNames;
  std::vector<int>         miscIds;
  for(int i = 0; i < mKeyBindings.size(); i++)
    {
      // set key popup binding
      if(mKeyBindings[i].name.find("Key Bindings") != std::string::npos)
        { mKeyPopupBinding = &mKeyBindings[i]; }

      mMaxNameLength = std::max(mMaxNameLength, (int)mKeyBindings[i].name.length());
      mMaxKeyLength  = std::max(mMaxKeyLength,  (int)mKeyBindings[i].toString().length());
      
      // find containing group
      bool found = false;
      for(auto &g : mKeyBindingGroups)
        {
          auto iter = std::find(g.bindings.begin(), g.bindings.end(), mKeyBindings[i].name);
          if(iter != g.bindings.end())
            {
              g.ids.push_back(i);
              found = true; break;
            }
        }
      if(!found)
        { // add to misc group
          std::cout << "====> NOTE: Node type " << mKeyBindings[i].name << " was not found in defined Keybindings (adding to Misc group)\n";
          miscNames.push_back(mKeyBindings[i].name);
          miscIds.push_back(i);
        }
    }
  // add ungrouped bindings to "misc" group
  KeyBindingGroup miscGroup{ "Misc", {}, {} };
  for(int i = 0; i < miscNames.size(); i++)
    {
      miscGroup.bindings.push_back(miscNames[i]);
      miscGroup.ids.push_back(miscIds[i]);
    }
  if(miscGroup.bindings.size() > 0) { mKeyBindingGroups.push_back(miscGroup); }
}


json KeyManager::toJSON() const
{
  json keyBindings = json::object();
  for(auto &k : mKeyBindings) { keyBindings[k.name]  = k.toString(); }
  // js["KeyBindings"] = keyBindings;
  return keyBindings;
}
bool KeyManager::fromJSON(const json &js)
{
  for(auto &k : mKeyBindings)
    {
      if(js.contains(k.name))
        {
          k.fromString(js[k.name]);
          // if(k.name == "Cancel") { mCancelKey = k.sequence.back().key; }
        }
      else
        { std::cout << "====> WARNING: Settings file missing key binding '" << k.toString() << "' (skipping)\n"; }
    }
  return true;
}


void KeyManager::keyPress(int mods, int key, int action)
{
  // (for managed key bindings)
  if(key != GLFW_KEY_LEFT_CONTROL && key != GLFW_KEY_RIGHT_CONTROL && // (don't count modifiers as presses)
     key != GLFW_KEY_LEFT_SHIFT   && key != GLFW_KEY_RIGHT_SHIFT   &&
     key != GLFW_KEY_LEFT_ALT     && key != GLFW_KEY_RIGHT_ALT     &&
     key != GLFW_KEY_LEFT_SUPER   && key != GLFW_KEY_RIGHT_SUPER   &&
     key != GLFW_KEY_CAPS_LOCK    && key != GLFW_KEY_NUM_LOCK)
    {
      bool press    = (action == GLFW_PRESS);
      bool repeat   = (action == GLFW_REPEAT);
      bool release  = (action == GLFW_RELEASE);
      bool anyPress = (press || repeat);

      if(anyPress)     // key pressed -- add to sequence
        {
          if(repeat && mKeySequence.size() > 0) // don't keep growing list of repeats
            { mKeySequence.back() = KeyPress(mods, key, repeat); }
          else
            { mKeySequence.emplace_back(mods, key, repeat); }
        }
      else if(release) // key released --> clear sequence
        {
          for(int i = 0; i < mKeySequence.size(); i++)
            {
              const KeyPress &k = mKeySequence[i];
              if(k.key == key)
                { // key released --> clear sequence after this key
                  mKeySequence.erase(mKeySequence.begin()+i, mKeySequence.end());
                  break;
                }
            }
        }
    }
}

void KeyManager::update(bool captured, bool verbose)
{
  if(mBindingEdit)
    { // user setting new key binding
      mBindingEdit->sequence = mKeySequence;
      if((mBindingEdit->sequence.size() > 0 && mBindingEdit->sequence.back().key != GLFW_KEY_UNKNOWN))
        {
          if(!(mBindingEdit->sequence.back().key == GLFW_KEY_ESCAPE &&
               mBindingEdit->sequence.back().mods == 0)) // pressing only escape cancels binding 
            {
              // check if binding is available
              for(auto &s : mKeyBindings)
                {
                  if(&s != mBindingEdit && s == *mBindingEdit)
                    {
                      std::cout << "Key binding is already in use! (" << s.name << ")\n";
                      mBindingEdit->sequence = mOldBinding.sequence;
                      break;
                    }
                }
            }
          else
            { mBindingEdit->sequence = mOldBinding.sequence; }
          // sequence complete -- reset
          mKeySequence.clear();
          mBindingEdit = nullptr;
        }
    }
  else
    {
      if(mKeySequence.size() > 0) // && mKeySequence.back().key != GLFW_KEY_UNKNOWN)
        {
          if(mPopupOpen && mKeyPopupBinding)
            { // popup open -- just handle popup binding (to close)
              if(mKeyPopupBinding->check(mKeySequence, verbose)) { mKeySequence.clear(); }
              mKeyPopupBinding->update(mKeySequence, verbose);
              // close popup on escape
              if(mKeySequence.size() > 0 && mKeySequence.back().key == GLFW_KEY_ESCAPE)
                { mPopupOpen = false; mKeySequence.clear(); }
            }
          else
            { // handle all bindings
              for(auto &k : mKeyBindings)
                {
                  if(((k.flags & KEYBINDING_GLOBAL) || !captured) && k.check(mKeySequence, verbose))
                    {
                      k.update(mKeySequence, verbose);
                      if(k.flags & KEYBINDING_REPEAT) { mKeySequence.pop_back(); }
                      else                            { mKeySequence.clear();    }
                      break;
                    }
                }
              // for(auto &k : mKeyBindings) { k.update(mKeySequence, verbose); }
            }
        }
    }
}

void KeyManager::togglePopup()
{
  mPopupOpen = !mPopupOpen;
}

void KeyManager::draw(const Vec2f &frameSize)
{
  if(mPopupOpen) { ImGui::OpenPopup("keyPopup"); }

  Vec2f       padding = KEY_POPUP_PADDING;
  std::string pName   = "Key Bindings";
  Vec2f wPos;     // popup window position
  Vec2f wSize;    // popup window size
  
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, padding);
  ImGui::PushStyleColor(ImGuiCol_ModalWindowDimBg, Vec4f(0.1, 0.1, 0.1, 0.6));
  if(ImGui::BeginPopupModal("keyPopup", &mPopupOpen, (ImGuiWindowFlags_NoMove           |
                                                      ImGuiWindowFlags_NoResize         |
                                                      ImGuiWindowFlags_NoTitleBar       |
                                                      ImGuiWindowFlags_AlwaysAutoResize |   // resize based on child window size
                                                      ImGuiWindowFlags_NoScrollbar      |   // dont scroll outer popup window
                                                      ImGuiWindowFlags_NoScrollWithMouse)))
    {
      //padding = (p.childBorder ? padding : Vec2f(0,0)); // if no border, dont add more padding to child
      Vec2f textSize;
      Vec2f sizeDiff;
      // popup title
      ImGui::PushFont(mParent->titleFontB);
      {
        textSize = ImGui::CalcTextSize(pName.c_str());
        sizeDiff = Vec2f(0, textSize.y) + 2.0f*padding;
        ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos())+Vec2f((mPopupSize.x + sizeDiff.x - textSize.x)/2.0f, 0));
        ImGui::TextUnformatted(pName.c_str());
      }
      ImGui::PopFont();
      ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
      
      if(mPopupSize.x > 0.0f && mPopupSize.y > 0.0f)
        {
          wPos = (Vec2f(frameSize) - (mPopupSize+sizeDiff))/2.0f;
          ImGui::SetWindowPos(wPos);
        }

      bool hover = false;
      Vec2f p0 = ImGui::GetCursorScreenPos();
      if(ImGui::BeginChild("##keyPopupChild", mPopupSize, true, (ImGuiWindowFlags_NoMove     |
                                                                 ImGuiWindowFlags_NoResize   |
                                                                 ImGuiWindowFlags_NoTitleBar)))
        {
          ImGui::BeginGroup();
          drawKeyBindings();
          ImGui::EndGroup();

          Vec2f contentSize = Vec2f(ImGui::GetItemRectMax()) - ImGui::GetItemRectMin();
          if(mPopupSize.x <= 0.0f || mPopupSize.y <= 0.0f) // if size(x/y) < 0, automatic resizing based on contents
            {
              if(mPopupSize.x <= 0.0f) { mPopupSize.x = contentSize.x + 2.0f*padding.x; }
              if(mPopupSize.y <= 0.0f) { mPopupSize.y = contentSize.y + 2.0f*padding.y; }
              ImGui::SetWindowSize("##keyPopupChild", mPopupSize);
            }
          hover |= (ImGui::IsItemHovered() || ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows) ||
                    Rect2f(p0, p0+mPopupSize).contains(ImGui::GetMousePos())); // without rect.contains() test, popup is closed when scrollbar is clicked (?)
          wSize = (Vec2f(ImGui::GetItemRectMax()) - ImGui::GetItemRectMin()) + sizeDiff;
        }
      ImGui::EndChild();
      
      hover |= ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows);
      wPos   = ImGui::GetWindowPos();

      // cancel if clicked outside of window
      if(!mBindingEdit && (!hover && ImGui::IsMouseClicked(ImGuiMouseButton_Left)))
        { mPopupOpen = false; mKeySequence.clear(); }
      
      ImGui::EndPopup();
    }
  ImGui::PopStyleColor();
  ImGui::PopStyleVar();
}

// draw key binding popup
void KeyManager::drawKeyBindings()
{
  for(auto &g : mKeyBindingGroups)
    {
      if(g.bindings.size() > 0)
        {
          ImGui::Indent(); ImGui::TextUnformatted(g.name.c_str());
          ImGui::Unindent(); ImGui::Separator(); ImGui::Indent(); 
          ImGui::Spacing(); ImGui::Indent();
          for(int i = 0; i < g.bindings.size(); i++)
            {
              KeyBinding *kb  = (g.ids[i] < mKeyBindings.size()        ? &mKeyBindings[g.ids[i]]        : nullptr);
              KeyBinding *kbd = (g.ids[i] < mDefaultKeyBindings.size() ? &mDefaultKeyBindings[g.ids[i]] : nullptr);
              if(kb && kbd) { drawKeyBinding(*kb, *kbd); }
              else { std::cout << "====> WARNING: Missing key binding! --> " << g.ids[i] << " / " << g.bindings[i] << "\n"; }
            }
          ImGui::Unindent(); ImGui::Unindent();
          ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos()) + Vec2f(0.0f, 16.0f));
          ImGui::Separator();
        }
    }
}

// draw single key binding
void KeyManager::drawKeyBinding(KeyBinding &kb, const KeyBinding &defaultKb)
{
  ImGuiStyle &style = ImGui::GetStyle();
  ImGui::BeginGroup();
  {
    char info[256];
    sprintf(info, "%*s   %*s   ", -mMaxNameLength, kb.name.c_str(), -mMaxKeyLength, kb.toString().c_str());

    // find maximum binding width (approximated for monospace font)
    float tWidth = (ImGui::CalcTextSize((std::string(info)+"   <Press Keys>   ").c_str()).x +
                    style.WindowPadding.x + style.ItemSpacing.x + 2*style.FramePadding.x);
    
    ImGui::TextUnformatted(info);
    ImGui::SameLine();
    if(mBindingEdit == &kb)
      { // currently editing this binding
        if(ImGui::Button("<Press Keys>")) { kb.sequence = mOldBinding.sequence; mBindingEdit = nullptr; } // click button to cancel
      }
    else
      {
        if(ImGui::Button(("Set##"+kb.name).c_str()) && !mBindingEdit)
          {
            mOldBinding = kb;
            kb.sequence.clear();
            mBindingEdit = &kb;
          }
        if(kb != defaultKb)
          { // reset button 
            ImGui::SameLine();
            if(ImGui::Button(("Reset##"+kb.name).c_str()) && !mBindingEdit)
              { kb = defaultKb; }
          }
      }
    // enforce maximum width
    ImGui::SetCursorPos(Vec2f(tWidth, ImGui::GetCursorPos().y));
  }
  ImGui::EndGroup();
  if(ImGui::IsItemHovered())
    { ImGui::SetTooltip("%s", kb.description.c_str()); }
}
