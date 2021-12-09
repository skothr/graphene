#include "keyBinding.hpp"

#include <sstream>
#include <iostream>
#include <algorithm>
#include <bitset>
#include <cassert>

#include "glfwKeys.hpp"

KeyBinding::KeyBinding(const std::string &name_, const std::string &default_, const std::string &desc_, KeyBindingFlags flags_,
                       const std::function<void(float)> &action_, float shift, float ctrl, float alt)
  : name(name_), defaultBinding(default_), description(desc_), action(action_), flags(flags_), shiftMult(shift), ctrlMult(ctrl), altMult(alt)
{ fromString(default_); }

KeyBinding::KeyBinding(const std::string &name_, const std::string &default_, const std::string &desc_, KeyBindingFlags flags_,
                       const std::function<void(int)> &action_, bool dummy)
  : name(name_), defaultBinding(default_), description(desc_), flags(flags_), action([action_](float mult){ action_((int)mult); })
{ fromString(default_); }

KeyBinding::KeyBinding(const std::string &name_, const std::string &default_, const std::string &desc_, KeyBindingFlags flags_,
                       const std::function<void()> &action_)
  : name(name_), defaultBinding(default_), description(desc_), flags(flags_), action([action_](float mult){ action_(); })
{ fromString(default_); }

KeyBinding& KeyBinding::operator=(const KeyBinding &other)
{
  flags          = other.flags;
  sequence       = other.sequence;
  name           = other.name;
  defaultBinding = other.defaultBinding;
  description    = other.description;
  action         = other.action;
  shiftMult = other.shiftMult;
  ctrlMult  = other.ctrlMult;
  altMult   = other.altMult;
  kNumeric  = other.kNumeric;
  return *this;
}

// comparison
bool KeyBinding::operator==(const KeyBinding &other) const
{
  if(other.sequence.size() != sequence.size()) { return false; }
  for(auto i = 0; i < sequence.size(); i++)
    {
      if(other.sequence[i].key != sequence[i].key || other.sequence[i].mods != sequence[i].mods)
        { return false; }
    }
  return true;
}
bool KeyBinding::operator!=(const KeyBinding &other) const { return !(other == *this); }




int KeyBinding::getMod(const std::string &str) const
{
  if     (str == "CTRL" || str == "CONTROL")  { return GLFW_MOD_CONTROL;   }
  else if(str == "SHIFT")                     { return GLFW_MOD_SHIFT;     }
  else if(str == "ALT")                       { return GLFW_MOD_ALT;       }
  else if(str == "SUPER")                     { return GLFW_MOD_SUPER;     }
  else if(str == "CAPS" || str == "CAPSLOCK") { return GLFW_MOD_CAPS_LOCK; }
  else if(str == "NUM"  || str == "NUMLOCK")  { return GLFW_MOD_NUM_LOCK;  }
  else                                        { return 0; }
}

std::string KeyBinding::getModString(int mod) const
{
  std::string modStr = "";
  if(mod & GLFW_MOD_CONTROL)   { modStr += (std::string(modStr.empty() ? "" : "+") + "Ctrl");  }
  if(mod & GLFW_MOD_SHIFT)     { modStr += (std::string(modStr.empty() ? "" : "+") + "Shift"); }
  if(mod & GLFW_MOD_ALT)       { modStr += (std::string(modStr.empty() ? "" : "+") + "Alt");   }
  if(mod & GLFW_MOD_SUPER)     { modStr += (std::string(modStr.empty() ? "" : "+") + "Super"); }
  if(mod & GLFW_MOD_CAPS_LOCK) { modStr += (std::string(modStr.empty() ? "" : "+") + "Caps");  }
  if(mod & GLFW_MOD_NUM_LOCK)  { modStr += (std::string(modStr.empty() ? "" : "+") + "Num");   }
  return modStr;
}


int KeyBinding::getKey(const std::string &keyStr) const
{
  return (((flags & KEYBINDING_NUMERIC) && keyStr == "<#>") ? KEY_NUMERIC : stringToGlfwKey(keyStr));
}
std::string KeyBinding::getKeyString(int key) const
{
  return (((flags & KEYBINDING_NUMERIC) && key == KEY_NUMERIC) ? "<#>" : glfwKeyToString(key));
}
  
bool KeyBinding::check(const std::vector<KeyPress> &seq, bool verbose)
{
  if(seq.size() != sequence.size()) { return false; }
  for(auto i = 0; i < seq.size(); i++)
    {
      int modOverlap = (seq[i].mods & sequence[i].mods);
      int k = seq[i].key;

      bool noMatch = (((!(flags & KEYBINDING_GLOBAL) || sequence[i].mods != 0) &&
                       (((flags & KEYBINDING_MOD_MULT) ? (sequence[i].mods != modOverlap) : (seq[i].mods != sequence[i].mods && !(flags & KEYBINDING_EXTRA_MODS))) ||
                        (seq.back().repeat && !(flags & KEYBINDING_REPEAT)))));
      
      // check for numeric key if enabled
      if((flags & KEYBINDING_NUMERIC) && !noMatch && k >= GLFW_KEY_0 && k <= GLFW_KEY_9)
        {
          kNumeric = (k - GLFW_KEY_0);
          if(verbose) { std::cout << "======> KNUMERIC --> [" << k << "] ==> " << kNumeric << "\n"; }
        }
      
      // check if key matches
      else if(k != sequence[i].key || noMatch) { pressed = false; return false; }
    }
  pressed = true;
  return pressed;
}

bool KeyBinding::update(const std::vector<KeyPress> &seq, bool verbose)
{
  bool activated = false;
  if(pressed && action)
    {
      float mult = 1.0f;
      if(flags & KEYBINDING_MOD_MULT)
        {
          int m = seq.back().mods & ~sequence.back().mods; // exclude mods defined as part of base sequence
          mult *= (((m & GLFW_MOD_SHIFT)   ? shiftMult : 1.0f) *
                   ((m & GLFW_MOD_CONTROL) ? ctrlMult  : 1.0f) *
                   ((m & GLFW_MOD_ALT)     ? altMult   : 1.0f));
        }
      if(verbose)
        {
          std::cout << "\n==== " << toString() << (flags & KEYBINDING_NUMERIC ? (" ("+std::to_string(kNumeric)+")") : "")
                    << " --> " << name << ((flags & KEYBINDING_MOD_MULT) ? (" (x" + std::to_string(mult) + ")") : "")
                    << " | " << description << "\n\n";
        }

      // set to numeric key if flag enabled
      if(flags & KEYBINDING_NUMERIC) // && kNumeric >= 0)
        { mult = (float)kNumeric; }
      
      action(mult);
      activated = true;
    }
  pressed = false;
  return activated;
}

void KeyBinding::reset() { fromString(defaultBinding); }
void KeyBinding::setModMults(float shift, float ctrl, float alt) { shiftMult = shift; ctrlMult = ctrl; altMult = alt; }
void KeyBinding::setShiftMult(float shift) { shiftMult = shift; }
void KeyBinding::setCtrlMult(float ctrl)   { ctrlMult  = ctrl;  }
void KeyBinding::setAltMult(float alt)     { altMult   = alt;   }

void KeyBinding::fromString(const std::string &keyStr)
{
  // convert string to uppercase (values match GLFW keys)
  std::string keyStrUpper = keyStr;
  std::transform(keyStrUpper.begin(), keyStrUpper.end(), keyStrUpper.begin(), [](unsigned char c) { return std::toupper(c); });
  // tokenize string into a sequence of key names
  std::vector<std::string> tokens;
  std::stringstream ss(keyStrUpper);
  std::string token;
  while(std::getline(ss, token, '+')) { tokens.push_back(token); }
      
  // parse tokens to create sequence
  sequence.clear();
  for(auto k : tokens)
    {
      int mod = getMod(k);
      if(mod != 0)
        { // token is a modifier key
          if(sequence.size() > 0 && sequence.back().key == GLFW_KEY_UNKNOWN) { sequence.back().mods |= mod; } // combine
          else { sequence.push_back(KeyPress(mod, GLFW_KEY_UNKNOWN, false)); } // new keypress
        }
      else
        {
          int key = getKey(k);
          if(key != GLFW_KEY_UNKNOWN)
            {
              if(sequence.size() > 0 && sequence.back().key == GLFW_KEY_UNKNOWN) { sequence.back().key = key; } // combine
              else { sequence.push_back(KeyPress(0, key, false)); } // new keypress
            }
        }
    }
}

std::string KeyBinding::toString() const
{
  std::string keyStr = "";
  for(auto &k : sequence)
    {
      keyStr += std::string(keyStr.empty() ? "" : "+") + getModString(k.mods);
      std::string kStr = getKeyString(k.key);
      if(!kStr.empty())
        {
          std::transform(kStr.begin(), kStr.end(), kStr.begin(), [](unsigned char c) { return std::tolower(c); });
          kStr[0] = std::toupper(kStr[0]); // first character uppercase
          keyStr += std::string(keyStr.empty() ? "" : "+") + kStr;
        }
    }
  return keyStr;
}
