#ifndef TAB_MENU_HPP
#define TAB_MENU_HPP

#include <vector>
#include <string>
#include <functional>
#include "vector.hpp"

#define TABMENU_TAB_PADDING  Vec2f(10.0f, 10.0f)
#define TABMENU_MENU_PADDING 4.0f

// forward declarations
struct ImFont;

typedef std::function<void()> TabCallback;
struct TabDesc
{
  TabCallback drawMenu;           // draw callback
  std::string label;              // tab label
  std::string title;              // tab title over child window
  float       width;              // width of contents when open (measured)
  float       minWidth   = -1.0f; // minimum width
  float       fixedWidth = -1.0f; // constant width
  ImFont     *titleFont  = nullptr;
};

class TabMenu
{
private:
  std::vector<TabDesc> mTabs;
  int mSelected = 0;

  float mLength       = 0.0f; // size of area along direction of text/tabs
  float mBarWidth     = 0.0f; // thickness of bar (if <= 0, based on font size and padding)
  bool  mCollapsible  = false;
  bool  mHorizontal   = false;
  
public:
  TabMenu() { }
  TabMenu(int barWidth, int length);
  
  Vec2f getSize()      const;
  float getBarWidth()  const;
  float getTabLength() const;

  int selected() const { return mSelected; }
  
  void setItemWidth(int index, int w)      { mTabs[index].width      = w; }
  void setItemFixedWidth(int index, int w) { mTabs[index].fixedWidth = w; }
  
  void setBarWidth(int w);
  void setLength(int l);
  void setCollapsible(bool collapsible);
  void setHorizontal(bool horizontal);
  
  int add(TabDesc desc);
  void remove(const std::string &label);
  void remove(int index);
  
  void select(int index); // -1 or index of open tab will collapse bar
  void collapse();
  void prev();
  void next();
  
  void draw(const std::string &id);
};


#endif //TAB_MENU_HPP
