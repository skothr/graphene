#ifndef TOOLBAR_HPP
#define TOOLBAR_HPP

#ifndef __NVCC__
#define __NVCC__

#include <string>
#include <vector>
#include <functional>

#include "image.hpp"
#include "vector.hpp"

// forward declarations
class ImFont;
class ImDrawList;


typedef std::function<void(int i, ImDrawList *drawList, const Vec2f &p0, const Vec2f &p1)> DrawCallback;

struct ToolButton
{
  std::string name;
  DrawCallback imgDraw = nullptr; // callback for drawing image/icon (1)
  std::string imgPath;            // path to an image/icon file      (2)
  std::string label;              // label text shown                (3)
  ImFont *font = nullptr;         //   label font

  ToolButton() = default;
  ToolButton(const std::string name_, const std::string label_, const std::string &img, ImFont *font_=nullptr)
    : name(name_), label(label_), imgPath(img), imgDraw(nullptr),   font(font_) { }
  ToolButton(const std::string name_, const std::string label_, const DrawCallback &imgDrawCb=nullptr, ImFont *font_=nullptr)
    : name(name_), label(label_), imgPath(""),  imgDraw(imgDrawCb), font(font_) { }
  ToolButton(const ToolButton &other) = default;
  ~ToolButton() = default;
};


class Toolbar
{
protected:
  std::vector<ToolButton> mTools;
  ToolButton mDefaultTool;
  int mSelected = -1;

  std::unordered_map<std::string, Image*> mImgMap; // maps image path to image data
  std::unordered_map<std::string, GLuint> mTexMap; // maps image path to texture ID
  
  bool  mHorizontal  = true;
  bool  mCollapsible = true;
  bool  mCollapsed   = false;
  float mLength = 200.0f;   // (horizontal ? xSize : ySize)
  float mWidth  = 20.0f;    // (horizontal ? ySize : xSize)
  
  std::string mTitle;
  ImFont *mHotkeyFont = nullptr;
  ImFont *mTitleFont  = nullptr;

  std::function<void(int)> mAddCallback;
  std::function<void(int)> mRemoveCallback;
  std::function<void(int)> mSelectCallback;

  bool drawToolButton(int i, const Vec2f &bSize);

public:
  Toolbar(ImFont *hotkeyFont=nullptr) : mHotkeyFont(hotkeyFont) { }
  virtual ~Toolbar();

  void setHotkeyFont(ImFont *font) { mHotkeyFont = font; }
  void setDefault(const ToolButton &tool) { mDefaultTool = tool; }

  void setAddCallback(const std::function<void(int)> &cb)    { mAddCallback    = cb; }
  void setRemoveCallback(const std::function<void(int)> &cb) { mRemoveCallback = cb; }
  void setSelectCallback(const std::function<void(int)> &cb) { mSelectCallback = cb; }
  
  void setHorizontal(bool horizontal);
  void setLength(float length);
  void setWidth(float width);
  float getLength() const;
  float getWidth() const;
  Vec2f getSize() const;
  
  Vec2f toolSize(int i);
  
  void setTitle(const std::string &title, ImFont *font);
  void add(const ToolButton &tool);
  void remove(int i);
  void select(int i) { if(i < mTools.size()) { if(mSelectCallback) { mSelectCallback(i); } mSelected = i; } }
  void clear() { for(int i = mTools.size()-1; i >= 0; i--) { remove(i); } }
  
  int count() const    { return mTools.size(); }
  int selected() const { return mSelected; }
  
  bool draw();
};

#endif // __NVCC__

#endif // TOOLBAR_HPP
