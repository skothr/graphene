#ifndef TAB_MENU_HPP
#define TAB_MENU_HPP

#include <vector>
#include <string>
#include <functional>
#include "vector.hpp"

#define TABMENU_TAB_PADDING  6.0f
#define TABMENU_MENU_PADDING 5.0f

// forward declarations
struct ImFont;

typedef std::function<void()> TabCallback;
struct TabDesc
{
  std::string label;           // tab label
  std::string title;           // tab title over child window
  TabCallback drawMenu;        // draw callback
  int         width;           // width of contents when open (measured)
  int         minWidth;        // minimum width
  int         fixedWidth = -1; // constant width
  ImFont     *titleFont = nullptr;
};

class TabMenu
{
private:
  std::vector<TabDesc> mTabs;
  int mSelected = 0;

  float mLength       = 0.0f; // direction of text/tabs
  float mBarWidth     = 0.0f; // thickness of bar
  bool  mCollapsible  = false;
  
public:
  TabMenu() { }
  TabMenu(int width, int length, bool collapsible=false);
  
  Vec2f getSize()      const;
  float getBarWidth()  const;
  float getTabLength() const;

  void setItemWidth(int index, int w) { mTabs[index].width = w; }
  void setItemFixedWidth(int index, int w) { mTabs[index].fixedWidth = w; }
  
  void setBarWidth(int w);
  void setLength(int l);
  void setCollapsible(bool collapsible=true);
  
  int add(TabDesc desc);
  void remove(const std::string &label);
  void remove(int index);
  
  void select(int index); // -1 or index of open tab will collapse bar
  void collapse();
  void draw();
};


#endif //TAB_MENU_HPP
