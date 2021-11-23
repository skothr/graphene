#include "toolbar.hpp"

#include <imgui.h>

#include "rect.hpp"

#define TOOLBAR_BG_COLOR            Vec4f(0.1, 0.1, 0.1, 1.0)
#define TOOLBAR_BUTTON_COLOR        Vec4f(0.05, 0.05, 0.05, 1.0)
#define TOOLBAR_BUTTON_HOVER_COLOR  Vec4f(0.2, 0.4, 0.5, 1.0)
#define TOOLBAR_BUTTON_ACTIVE_COLOR Vec4f(0.4, 0.5, 0.8, 1.0)
#define TOOLBAR_SELECTED_COLOR      Vec4f(0.8, 0.2, 0.2, 1.0)

#define TOOLBAR_DELETE_SIZE Vec2f(13,13)
#define TOOLBAR_COLLAPSE_SIZE Vec2f(16,16)
#define TOOLBAR_COLLAPSE_COLOR Vec4f(0.4f, 0.4f, 0.4f, 1.0f)

Toolbar::~Toolbar()
{
  for(auto iter : mTexMap) { if(iter.second) { destroyTexture(iter.second); } }
  mTexMap.clear();
  for(auto iter : mImgMap)
    {
      if(iter.second)
        {
          if(iter.second->data) { delete iter.second->data; iter.second->data = nullptr; }
          delete iter.second;
        }
    }
  mImgMap.clear();
}


void Toolbar::add(const ToolButton &tool)
{
  mTools.push_back(tool);
  if(!tool.imgPath.empty())
    {
      Image *image = loadImageData(tool.imgPath);
      mImgMap.emplace(tool.imgPath, image);
      GLuint texId = createTexture(image);
      mTexMap.emplace(tool.imgPath, texId);
    }
  if(mAddCallback) { mAddCallback(mTools.size()-1); }
  select(mTools.size()-1);
}

void Toolbar::remove(int i)
{
  // if(i == mTools.size()-1) { select(mTools.size()-2); }
  ToolButton &t = mTools[i];
  auto imgIter = mImgMap.find(t.imgPath);
  if(imgIter != mImgMap.end()) { delete imgIter->second->data; delete imgIter->second; mImgMap.erase(imgIter); }
  auto texIter = mTexMap.find(t.imgPath);
  if(texIter != mTexMap.end()) { destroyTexture(texIter->second); mTexMap.erase(texIter); }
  
  mTools.erase(mTools.begin()+i);
  if(mRemoveCallback) { mRemoveCallback(i); }
  
  if(i <= mSelected) { select(mSelected - 1); }
}


void Toolbar::setTitle(const std::string &title, ImFont *font) { mTitle = title; mTitleFont = font; }
void Toolbar::setHorizontal(bool horizontal) { mHorizontal = horizontal; }
void Toolbar::setLength(float length)        { mLength = length; }
void Toolbar::setWidth(float width)          { mWidth  = width;  }

float Toolbar::getLength() const { return mLength; }
float Toolbar::getWidth() const
{
  float w = mWidth;
  if(mCollapsible && mCollapsed)
    {
      ImGuiStyle &style = ImGui::GetStyle();
      if(mTitleFont) { ImGui::PushFont(mTitleFont); }
      w = (mHorizontal ?
           ImGui::CalcTextSize(mTitle.empty() ? " " : mTitle.c_str()).y + 2.0*style.WindowPadding.x :
           ImGui::CalcTextSize(mTitle.empty() ? " " : mTitle.c_str()).x + 2.0*style.WindowPadding.y);
      if(mTitleFont) { ImGui::PopFont(); }
    }
  return w;
}

Vec2f Toolbar::getSize() const
{
  return (mHorizontal ? Vec2f(mLength, getWidth()) : Vec2f(getWidth(), mLength));
}

bool Toolbar::drawToolButton(int i, const Vec2f &bSize)
{
  ImGuiStyle &style = ImGui::GetStyle();
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  Vec2f p1 = ImGui::GetCursorPos();
  int wd = (mHorizontal ? 0 : 1); // (width dimension  --> 0=x, 1=y)
  int ld = (mHorizontal ? 1 : 0); // (length dimension --> 0=x, 1=y)
  bool changed = false;

  const ToolButton &tool = mTools[i];
  // size calculations
  const auto &imgExtIter = mImgMap.find(tool.imgPath);   // external image (loaded from ToolButton::imgPath)
  const auto &imgTexIter = mTexMap.find(tool.imgPath);   // texture id of external image
  bool hasDrawImg = (tool.imgDraw != nullptr);         // defines draw image via callback
  bool hasExtImg  = (imgExtIter != mImgMap.end() && // defines loaded icon file
                     imgTexIter != mTexMap.end());
  bool hasLabel   = (!tool.label.empty());             // defines text label
  
  Vec2f imgSize; // size of external image to be drawn on-screen
  Vec2f tSize;   // size of label text
  if(hasExtImg)
    { // icon
      float imgH = (bSize.y - 2.5f*Vec2f(style.ItemInnerSpacing)[ld]);
      float imgAspect = imgExtIter->second->width / imgExtIter->second->height;
      imgSize = Vec2f(imgH, imgH*imgAspect);
    }
  else if(hasDrawImg)
    { // icon
      float imgH = (bSize.y - 2.5f*Vec2f(style.ItemInnerSpacing)[ld]);
      imgSize = Vec2f(imgH, imgH);
    }
  if(hasLabel)
    {
      if(tool.font) { ImGui::PushFont(tool.font); }
      tSize = Vec2f(ImGui::CalcTextSize(tool.label.c_str()));
      if(tool.font) { ImGui::PopFont(); }
    }
  Vec2f buttonSize = bSize;
  if(mHorizontal)
    {
      buttonSize.x += ((hasDrawImg ? imgSize.x : 0) + // actual button size
                       (hasExtImg ? (imgSize.x + (hasDrawImg ? style.ItemSpacing.x : 0)) : 0) +
                       (hasLabel ? (tSize.x + ((hasDrawImg || hasExtImg) ? style.ItemSpacing.x : 0)) : 0));
      if(!hasDrawImg && !hasExtImg && !hasLabel) // (no image or label) --> square button
        { buttonSize.x = buttonSize.y; }
    }
  else { buttonSize.x = bSize.x; }

  // draw
  bool deleted = false;
  ImGui::BeginGroup();
  {    
    // draw tool button
    ImGui::PushStyleColor(ImGuiCol_Button,        TOOLBAR_BUTTON_COLOR);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, TOOLBAR_BUTTON_HOVER_COLOR);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  TOOLBAR_BUTTON_ACTIVE_COLOR);
    if(ImGui::Button(("##tool"+std::to_string(i)).c_str(), buttonSize)) { changed = true; select(i); }
    ImGui::PopStyleColor(3);
    
    if(ImGui::BeginPopupContextItem(("toolContext-"+std::to_string(i)).c_str(), ImGuiMouseButton_Right))
      {
        if(ImGui::MenuItem("delete")) { deleted = true; changed = true; }
        ImGui::EndPopup();
      }
    
    if(mSelected == i)
      {
        ImGui::GetWindowDrawList()->AddRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(),
                                            ImColor(TOOLBAR_SELECTED_COLOR), 0, 0, 2.0f);
      }

    if(hasDrawImg)
      { // draw icon
        ImGui::SetCursorPos(p1 + style.ItemInnerSpacing);
        Vec2f sp = ImGui::GetCursorScreenPos();
        
        ImGui::PushClipRect(sp, sp + imgSize, true);
        tool.imgDraw(i, drawList, sp, sp + imgSize);
        ImGui::PopClipRect();
        if(hasExtImg || hasLabel) { ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos())+Vec2f(imgSize.x+style.ItemSpacing.x, 0)); }
      }

    if(hasExtImg)
      { // external icon
        if(!hasDrawImg) { ImGui::SetCursorPos(p1 + style.ItemInnerSpacing); }
        ImGui::SetCursorPos(p1 + style.ItemInnerSpacing);
        ImGui::Image(reinterpret_cast<ImTextureID*>(imgTexIter->second), imgSize, Vec2f(0,0), Vec2f(1,1), Vec4f(1,1,1,1), Vec4f(1,1,1,1));
        if(hasLabel) { ImGui::SameLine(); }
      }
            
    if(hasLabel)
      { // tool label
        Vec2f spacing = (buttonSize-tSize)/2.0f;
        if(!hasDrawImg && !hasExtImg) { ImGui::SetCursorPos(p1 + spacing); }
        else                          { ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos().x, p1.y+spacing.y)); }
        if(tool.font) { ImGui::PushFont(tool.font); }
        ImGui::TextUnformatted(tool.label.c_str());
        if(tool.font) { ImGui::PopFont(); }
      }

    // draw index number
    ImGui::SetCursorPos(p1 + 2.0f);
    if(mHotkeyFont) { ImGui::PushFont(mHotkeyFont); }
    ImGui::TextUnformatted(std::to_string(i+1).c_str());
    if(mHotkeyFont) { ImGui::PopFont(); }
  }
  ImGui::EndGroup();
  if(mHorizontal) { ImGui::SameLine(); }
  
  if(deleted) { remove(i); }
  
  return changed;
}



bool Toolbar::draw()
{
  bool changed = false;
  ImGuiStyle &style = ImGui::GetStyle();
  ImGuiIO &io = ImGui::GetIO();

  float maxW = 0.0f;
  if(!mHorizontal)
    {
      for(const auto &t : mTools)
        {
          if(!t.label.empty())
            {
              if(t.font) { ImGui::PushFont(t.font); }
              maxW = std::max(maxW, ImGui::CalcTextSize(t.label.c_str()).x);
              if(t.font) { ImGui::PopFont(); }
            }
        }
      maxW += style.ItemSpacing.x;
    }

  Vec2f barSize = getSize() + Vec2f(maxW, 0);
  
  ImGui::PushStyleColor(ImGuiCol_ChildBg, TOOLBAR_BG_COLOR);
  if(ImGui::BeginChild(("##toolBar-"+mTitle).c_str(), barSize, true, (ImGuiWindowFlags_AlwaysUseWindowPadding |
                                                                      ImGuiWindowFlags_NoScrollWithMouse |
                                                                      ImGuiWindowFlags_NoDecoration)))
    {
      Vec2f p0 = ImGui::GetCursorPos();
      if(!mTitle.empty())
        {
          if(mTitleFont) { ImGui::PushFont(mTitleFont); }
          Vec2f tSize = ImGui::CalcTextSize(mTitle.c_str());
          ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos()) +
                              (mHorizontal ?
                               Vec2f(0, barSize.y-2.0f*style.WindowPadding.y-tSize.y) :
                               Vec2f(barSize.x-2.0f*style.WindowPadding.x-tSize.x, 0)) / 2.0f);
          ImGui::TextUnformatted(mTitle.c_str());
          if(mTitleFont)  { ImGui::PopFont();  }
          if(mHorizontal) { ImGui::SameLine(); }
          ImGui::SetCursorPos(mHorizontal ? Vec2f(ImGui::GetCursorPos().x, p0.y) : Vec2f(p0.x, ImGui::GetCursorPos().y));
        }
      
      Vec2f bSize; // base button size
      float barWidth = getWidth();
      if(mHorizontal) { bSize = Vec2f(2.5f*style.ItemInnerSpacing.x,         barWidth - 2.0*style.WindowPadding.y); }
      else            { bSize = Vec2f(barSize.x - 2.0*style.WindowPadding.x, barWidth - 2.0*style.WindowPadding.x); }

      // draw tools
      for(int i = 0; i < mTools.size(); i++) { changed |= drawToolButton(i, bSize); }

      // draw new tool "+" button
      ImGui::PushStyleColor(ImGuiCol_Button,        TOOLBAR_BUTTON_COLOR);
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, TOOLBAR_BUTTON_HOVER_COLOR);
      ImGui::PushStyleColor(ImGuiCol_ButtonActive,  TOOLBAR_BUTTON_ACTIVE_COLOR);
      Vec2f tSize  = ImGui::CalcTextSize("+");
      Vec2f baSize = tSize + 2.0f*Vec2f(style.FramePadding.x);
      Vec2f baPos = Vec2f(ImGui::GetCursorScreenPos()) + Vec2f(0, (barWidth-2*style.WindowPadding.y-baSize.y)/2.0f);
      ImGui::SetCursorScreenPos(baPos);
      if(ImGui::Button("##add", baSize))
        {
          changed = true;
          ToolButton newTool = (io.KeyShift ? mTools.back() : mDefaultTool);
          newTool.name += std::to_string(mTools.size()-1);
          add(newTool);
        }
      ImGui::GetWindowDrawList()->AddText(baPos + (baSize-tSize)/2.0f, ImColor(Vec4f(1,1,1,1)), "+");

      // draw collapse button
      if(mCollapsible)
        {
          ImGui::SetCursorPos(p0 + (mHorizontal ?
                                    (barSize - TOOLBAR_COLLAPSE_SIZE - style.WindowPadding) :
                                    Vec2f(0, barSize.y - 2*TOOLBAR_COLLAPSE_SIZE.y)) - style.WindowPadding);
          Vec2f spC = ImGui::GetCursorScreenPos();
          if(ImGui::Button("##collapse", TOOLBAR_COLLAPSE_SIZE)) { mCollapsed = !mCollapsed; }
          if(mCollapsed)
            { ImGui::GetWindowDrawList()->AddTriangleFilled(spC+Vec2f(TOOLBAR_COLLAPSE_SIZE.x/2, style.FramePadding.y+1),
                                                            spC+Vec2f(style.FramePadding.x+1, TOOLBAR_COLLAPSE_SIZE.y-style.FramePadding.y-1),
                                                            spC+TOOLBAR_COLLAPSE_SIZE - style.FramePadding-1,
                                                            ImColor(TOOLBAR_COLLAPSE_COLOR)); }
          else
            { ImGui::GetWindowDrawList()->AddTriangleFilled(spC+Vec2f(TOOLBAR_COLLAPSE_SIZE.x/2, TOOLBAR_COLLAPSE_SIZE.y-style.FramePadding.y),
                                                            spC+style.FramePadding,
                                                            spC+Vec2f(TOOLBAR_COLLAPSE_SIZE.x - style.FramePadding.x, style.FramePadding.y),
                                                            ImColor(TOOLBAR_COLLAPSE_COLOR)); }
        }      
      ImGui::PopStyleColor(3);
    }
  ImGui::EndChild(); ImGui::PopStyleColor();
  
  return changed;
}
  
