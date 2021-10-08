#ifndef KEY_MANAGER_HPP
#define KEY_MANAGER_HPP

#include <vector>
#include <thread>
#include <mutex>
#include <nlohmann/json_fwd.hpp> // json forward declarations
using json = nlohmann::json;

#include "vector.hpp"
#include "keyBinding.hpp"

// forward declarations
class SimWindow;

#define KEY_POPUP_PADDING Vec2f( 10,  10)
#define KEY_POPUP_SIZE    Vec2f(  0, 555)

class KeyManager
{
private:
  std::vector<KeyBinding>      mDefaultKeyBindings; // default key bindings (hard-coded)
  std::vector<KeyBinding>      mKeyBindings;        // current key bindings
  std::vector<KeyBindingGroup> mKeyBindingGroups;   // named groups for displaying
  
  std::vector<KeyPress>  mKeySequence;  // sequence of key presses (live)
  KeyBinding *mBindingEdit = nullptr;   // points to binding currently being edited
  KeyBinding  mOldBinding;              // previous binding for mBindingEdit (in case cancelled/taken)

  int mMaxNameLength = 0; // max length of binding name
  int mMaxKeyLength  = 0; // max length of shortcut text
  
  SimWindow *mParent = nullptr;
  KeyBinding *mKeyPopupBinding = nullptr; // pointer to key popup binding (closes popup if open, default Alt+K)
  bool  mPopupOpen = false;
  Vec2f mPopupSize = KEY_POPUP_SIZE;

  bool mUpdating = false;
  std::thread mUpdateThread;
  std::mutex  mUpdateLock;
  
public:
  KeyManager() { }
  KeyManager(SimWindow *parent, const std::vector<KeyBinding> &bindings, const std::vector<KeyBindingGroup> &groups={});
  ~KeyManager();
  
  json toJSON() const;
  bool fromJSON(const json &js);

  void togglePopup();
  bool popupOpen() const { return mPopupOpen; }
  
  void keyPress(int mods, int key, int action);
  void update(bool captured, bool verbose=false);
  void draw(const Vec2f &frameSize);

  void drawKeyBinding(KeyBinding &kb, const KeyBinding &defaultKb);
  void drawKeyBindings();

  void updateLoop();
};


#endif // KEY_MANAGER_HPP
