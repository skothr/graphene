#ifndef KEY_BINDING_HPP
#define KEY_BINDING_HPP

#include <functional>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <iostream>
#include <bitset>
#include "glfwKeys.hpp"
#include "tools.hpp"

// for categorizing key bindings
struct KeyBindingGroup
{
  std::string name = "";             // group name
  std::vector<std::string> bindings; // binding names
  std::vector<int>         ids;      // binding indices
};

// binding callback (mult will be 1 unless KEYBINDING_MOD_MULT is set and a separate 
typedef std::function<void(float mult)> KeyAction;

// single key press info
struct KeyPress
{
  int  mods    = 0;
  int  key     = GLFW_KEY_UNKNOWN;
  bool repeat  = false;
  KeyPress(int m, int k, bool r=false) : mods(m), key(k), repeat(r) { }
};

enum KeyBindingFlags
  {
   KEYBINDING_NONE     = 0x00, // (no flags set)
   KEYBINDING_GLOBAL   = 0x01, // binding triggered even when interacting with UI (e.g. editing textbox -- assumes key pattern isn't relevant)
   KEYBINDING_REPEAT   = 0x02, // binding triggered on key repeat actions (press/hold non-mod keys)
   KEYBINDING_MOD_MULT = 0x04, // additional mod keys act as multipliers for the binding action (generally: SHIFT --> x0.1 | CTRL --> x2 | ALT --> x10)
  };
ENUM_FLAG_OPERATORS(KeyBindingFlags)



// defines a sequence of keys as a string, and a triggered callback 
class KeyBinding
{
private:
  int getMod(const std::string &str) const
  {
    if     (str == "CTRL" || str == "CONTROL")  { return GLFW_MOD_CONTROL;   }
    else if(str == "SHIFT")                     { return GLFW_MOD_SHIFT;     }
    else if(str == "ALT")                       { return GLFW_MOD_ALT;       }
    else if(str == "SUPER")                     { return GLFW_MOD_SUPER;     }
    else if(str == "CAPS" || str == "CAPSLOCK") { return GLFW_MOD_CAPS_LOCK; }
    else if(str == "NUM"  || str == "NUMLOCK")  { return GLFW_MOD_NUM_LOCK;  }
    else                                        { return 0; }
  }
  std::string getModString(int mod) const
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
    
  int getKey(const std::string &keyStr) const { return stringToGlfwKey(keyStr); }
  std::string getKeyString(int key) const     { return glfwKeyToString(key); }
    
public:
  bool pressed = false; // set to true if activated
  
  std::vector<KeyPress> sequence;
  std::string name;
  std::string defaultBinding;
  std::string description;
  KeyAction   action = nullptr;
  KeyBindingFlags flags = KEYBINDING_NONE;

  // used to scale action if KEYBIDING_MOD_MULT is set
  float shiftMult =  0.1f;
  float ctrlMult  =  2.0f;
  float altMult   = 10.0f;
  
  KeyBinding() { }
  
  // NOTE: shift/ctrl/alt multipliers only relevant if KEYBIDING_MOD_MULT is set
  explicit KeyBinding(const std::string &name_, const std::string &default_, const std::string &desc_, std::function<void(float mult)> action_,
                      KeyBindingFlags flags_=KEYBINDING_NONE, float shift=0.1f, float ctrl=2.0f, float alt=10.0f)
    : name(name_), defaultBinding(default_), description(desc_), action(action_), flags(flags_), shiftMult(shift), ctrlMult(ctrl), altMult(alt)
  { fromString(default_); }

  // wraps void() action callback with unused argument (for compatibility, KEYBINDING_MOD_MULT)
  //    NOTE: shift/ctrl/alt multipliers only relevant if KEYBIDING_MOD_MULT is set
  explicit KeyBinding(const std::string &name_, const std::string &default_, const std::string &desc_, std::function<void()> action_,
                      KeyBindingFlags flags_=KEYBINDING_NONE, float shift=0.1f, float ctrl=2.0f, float alt=10.0f)
    : KeyBinding(name_, default_, desc_, [action_](float mult){ action_(); }, flags_, shift, ctrl, alt) { }

  // copying
  KeyBinding(const KeyBinding &other)
    : name(other.name), defaultBinding(other.defaultBinding), description(other.description), action(other.action), sequence(other.sequence),
      flags(other.flags), shiftMult(other.shiftMult), ctrlMult(other.ctrlMult), altMult(other.altMult) { }
  KeyBinding& operator=(const KeyBinding &other)
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
    return *this;
  }
  // comparison
  bool operator==(const KeyBinding &other) const
  {
    if(other.sequence.size() != sequence.size()) { return false; }
    for(auto i = 0; i < sequence.size(); i++)
      {
        if(other.sequence[i].key != sequence[i].key || other.sequence[i].mods != sequence[i].mods)
          { return false; }
      }
    return true;
  }
  bool operator!=(const KeyBinding &other) const
  { return !(other == *this); }
  
  bool check(const std::vector<KeyPress> &seq, bool verbose=false)
  {
    if(seq.size() != sequence.size()) { return false; }
    for(auto i = 0; i < seq.size(); i++)
      {
        int modOverlap = (seq[i].mods & sequence[i].mods);
        if(seq[i].key != sequence[i].key || ((flags & KEYBINDING_MOD_MULT) ?
                                             (sequence[i].mods != modOverlap) :
                                             (seq[i].mods != sequence[i].mods)) ||
           (seq.back().repeat && !(flags & KEYBINDING_REPEAT)))
          { return false; }
      }
    pressed = true;
    return pressed;
  }

  bool update(const std::vector<KeyPress> &seq, bool verbose=false)
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
        std::cout << "\n==== " << toString() << " --> "
                  << name << ((flags & KEYBINDING_MOD_MULT) ? (" (x" + std::to_string(mult) + ")") : "")
                  << " | " << description << "\n";
        action(mult); activated = true;
      }
    pressed = false;
    return activated;
  }

  void reset() { fromString(defaultBinding); }

  void setModMults(float shift=0.1f, float ctrl=2.0f, float alt=10.0f) { shiftMult = shift; ctrlMult = ctrl; altMult = alt; }
  void setShiftMult(float shift) { shiftMult = shift; }
  void setCtrlMult(float ctrl)   { ctrlMult  = ctrl;  }
  void setAltMult(float alt)     { altMult   = alt;   }

  void fromString(const std::string &keyStr)
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
            else { sequence.push_back(KeyPress(mod, GLFW_KEY_UNKNOWN)); } // new keypress
          }
        else
          {
            int key = getKey(k);
            if(key != GLFW_KEY_UNKNOWN)
              {
                if(sequence.size() > 0 && sequence.back().key == GLFW_KEY_UNKNOWN) { sequence.back().key = key; } // combine
                else { sequence.push_back(KeyPress(0, key)); } // new keypress
              }
          }
      }
  }
  std::string toString() const
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
};


#endif // KEY_BINDING_HPP
