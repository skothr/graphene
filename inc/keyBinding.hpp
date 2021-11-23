#ifndef KEY_BINDING_HPP
#define KEY_BINDING_HPP

#include <vector>
#include <string>
#include <functional>

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
  int  key     = -1;
  bool repeat  = false;
  KeyPress(int m, int k, bool r) : mods(m), key(k), repeat(r) { }
};

enum KeyBindingFlags
  {
    KEYBINDING_NONE       = 0x00, // (no flags set)
    KEYBINDING_GLOBAL     = 0x01, // binding triggered even when interacting with UI (e.g. editing textbox -- assumes key pattern isn't relevant)
    KEYBINDING_REPEAT     = 0x02, // binding triggered on key repeat actions (press/hold non-mod keys)
    KEYBINDING_EXTRA_MODS = 0x04, // additional mod keys will still trigger this binding
    KEYBINDING_MOD_MULT   = 0x08, // additional mod keys act as multipliers for the binding action (generally: SHIFT --> x0.1 | CTRL --> x2 | ALT --> x10)
  };
ENUM_FLAG_OPERATORS(KeyBindingFlags)



// defines a sequence of keys as a string, and a triggered callback 
class KeyBinding
{
private:
  int getMod(const std::string &str) const;
  std::string getModString(int mod) const;
  
  int getKey(const std::string &keyStr) const;
  std::string getKeyString(int key) const;
    
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
  
  KeyBinding() = default;
  
  // NOTE: shift/ctrl/alt multipliers only relevant if KEYBIDING_MOD_MULT is set
  KeyBinding(const std::string &name_, const std::string &default_, const std::string &desc_, KeyBindingFlags flags_,
             const std::function<void(float mult)> &action_, float shift=0.1f, float ctrl=2.0f, float alt=10.0f);

  // wraps void() action callback with unused argument (for compatibility, KEYBINDING_MOD_MULT)
  //    NOTE: shift/ctrl/alt multipliers only relevant if KEYBIDING_MOD_MULT is set
  KeyBinding(const std::string &name_, const std::string &default_, const std::string &desc_, KeyBindingFlags flags_,
             const std::function<void()> &action_, float shift=0.1f, float ctrl=2.0f, float alt=10.0f);
  KeyBinding(const KeyBinding &other) = default;
  KeyBinding& operator=(const KeyBinding &other);
  
  bool operator==(const KeyBinding &other) const;
  bool operator!=(const KeyBinding &other) const;
  
  bool check(const std::vector<KeyPress> &seq, bool verbose=false);
  bool update(const std::vector<KeyPress> &seq, bool verbose=false);
  void reset();

  void setModMults(float shift=0.1f, float ctrl=2.0f, float alt=10.0f);
  void setShiftMult(float shift);
  void setCtrlMult(float ctrl);
  void setAltMult(float alt);

  void fromString(const std::string &keyStr);
  std::string toString() const;
};


#endif // KEY_BINDING_HPP
