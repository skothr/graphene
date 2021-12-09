#include "tabMenu.hpp"

#include <imgui.h>
#include "imtools.hpp"


TabMenu::TabMenu(int barWidth, int length)
  : mBarWidth(barWidth), mLength(length) { }

Vec2f TabMenu::getSize() const
{
  ImGuiStyle &style = ImGui::GetStyle();
  float tSize = 0.0f;
  if(mHorizontal && mSelected >= 0)
    {
      const TabDesc &tab = mTabs[mSelected];
      if(!tab.title.empty())
        {
          if(tab.titleFont) { ImGui::PushFont(tab.titleFont); }
          tSize = ImGui::CalcTextSize(tab.label.c_str()).y + style.ItemSpacing.y;
          if(tab.titleFont) { ImGui::PopFont(); }
        }
    }
  
  float width  = getBarWidth() + (mSelected >= 0 ? (mTabs[mSelected].width +
                                                    (mHorizontal ? tSize : 0) +
                                                    2.0f*(mHorizontal ? style.ItemSpacing.y : style.ItemSpacing.x) +
                                                    4.0f*(mHorizontal ? style.WindowPadding.y : style.WindowPadding.x))
                                  : 0);
  return (mHorizontal ? Vec2f(mLength, width) : Vec2f(width, mLength));
}

float TabMenu::getBarWidth() const
{
  if(mBarWidth == 0.0f)
    {
      float barW = 0.0f;
      for(auto &tab : mTabs) { barW = std::max(barW, ImGui::CalcTextSize(tab.label.c_str()).y); }
      return barW + (mSelected >= 0 ? TABMENU_MENU_PADDING : 0.0f);
    }
  else { return mBarWidth; }
}

float TabMenu::getTabLength() const
{
  float tabL = 0.0f;
  for(auto &tab : mTabs) { tabL = std::max(tabL, ImGui::CalcTextSize(tab.label.c_str()).x); }
  return tabL + 2.0f*TABMENU_TAB_PADDING.x;
}

void TabMenu::setBarWidth(int w) { mBarWidth = w; }
void TabMenu::setLength(int l)   { mLength   = l; }
void TabMenu::setCollapsible(bool collapsible) { mCollapsible = collapsible; }
void TabMenu::setHorizontal(bool horizontal)   { mHorizontal  = horizontal;  }

int TabMenu::add(TabDesc desc)
{
  int index = mTabs.size();
  mTabs.push_back(desc);
  return index;
}

// remove by label
void TabMenu::remove(const std::string &label)
{
  for(int i = 0; i < mTabs.size(); i++)
    {
      if(mTabs[i].label == label)
        { mTabs.erase(mTabs.begin() + i); break; }
    }
}
// remove by index
void TabMenu::remove(int index) { if(mTabs.size() > index) { mTabs.erase(mTabs.begin() + index); } }
void TabMenu::select(int index) { if(index >= 0 || mCollapsible) { mSelected = ((mCollapsible && index == mSelected) ? -1 : index); } }
void TabMenu::collapse()        { select(-1); }

void TabMenu::prev() { mSelected = (mSelected - 1 + mTabs.size()) % mTabs.size(); }
void TabMenu::next() { mSelected = (mSelected + 1)                % mTabs.size(); }

void TabMenu::draw(const std::string &id)
{
  ImGuiStyle &style = ImGui::GetStyle();

  // update tab menu widths
  for(auto &tab : mTabs)
    {
      if(tab.fixedWidth > 0)       { tab.width = tab.fixedWidth; }
      if(tab.width < tab.minWidth) { tab.width = tab.minWidth;   }
    }
   
  // tab colors
  Vec4f inactiveColor     (0.2f,  0.2f,  0.2f,  1.0f); Vec4f activeColor       (0.5f, 0.5f, 0.5f, 1.0f);
  Vec4f hoveredColor      (0.35f, 0.35f, 0.35f, 1.0f); Vec4f clickedColor      (0.8f, 0.8f, 0.8f, 1.0f);
  Vec4f activeHoveredColor(0.65f, 0.65f, 0.65f, 1.0f); Vec4f activeClickedColor(0.8f, 0.8f, 0.8f, 1.0f);

  float barW = getBarWidth();
  float tabL = getTabLength();
 
  ImGuiWindowFlags outerFlags = (ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse); // outer child flags
  ImGuiWindowFlags innerFlags = ImGuiWindowFlags_None;                                               // inner child flags
  ImGuiWindowFlags tabFlags   = (ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse); // tab bar child flags

  ImGui::BeginGroup();
  {
    // draw contents
    ImGui::BeginGroup();
    {
      if(mSelected >= 0)
        {
          TabDesc &tab = mTabs[mSelected];
        
          Vec2f tSize;
          if(!tab.title.empty())
            {
              if(tab.titleFont) { ImGui::PushFont(tab.titleFont); }
              tSize = Vec2f(ImGui::CalcTextSize(tab.title.c_str())) + (mHorizontal ? Vec2f(0, style.ItemSpacing.y) : Vec2f(style.ItemSpacing.x, 0));
              if(tab.titleFont) { ImGui::PopFont(); }
            }
          
          Vec2f outerSize = getSize() - (mHorizontal ? Vec2f(0, getBarWidth()) : Vec2f(getBarWidth(), 0));
        
          ImGui::BeginChild(("##tabOuter"+id).c_str(), outerSize, true, outerFlags);
          {
            // draw title
            if(!tab.title.empty())
              {
                if(tab.titleFont) { ImGui::PushFont(tab.titleFont); }
                Vec2f offset = Vec2f(outerSize.x-tSize.x, 0.0f) / 2.0f;
                ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos()) + offset);
                ImGui::TextUnformatted(tab.title.c_str());
                if(tab.titleFont) { ImGui::PopFont(); }
              }
        
            // scrollable inner child
            Vec2f innerSize = outerSize - Vec2f(style.WindowPadding)*2.0f;
            ImGui::BeginChild(("##tabInner"+id).c_str(), innerSize, true, innerFlags);
            {
              ImGui::BeginGroup(); tab.drawMenu(); ImGui::EndGroup();
              if(tab.fixedWidth < 0) { tab.width = ImGui::GetItemRectMax().x - ImGui::GetItemRectMin().x; }
              if(tab.minWidth  >= 0) { tab.width = std::max(tab.width, tab.minWidth); }
              innerSize = (mHorizontal ?
                           Vec2f(mLength - 2.0f*style.WindowPadding.x, tab.width) :
                           Vec2f(tab.width, mLength - 2.0f*style.WindowPadding.y));
              ImGui::SetWindowSize(innerSize);
            }
            ImGui::EndChild();
          }
          ImGui::EndChild();
          if(!mHorizontal) { ImGui::SameLine(); }
        }
    }
    ImGui::EndGroup();
    if(!mHorizontal) { ImGui::SameLine(); }
  
    // draw tabs
    Vec2f barSize = (mHorizontal ?
                     Vec2f(std::max(mLength, (style.ItemSpacing.x) + tabL*mTabs.size()), barW) :
                     Vec2f(barW, std::max(mLength, (style.ItemSpacing.y + tabL)*mTabs.size())));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, Vec2f(0,0));
    ImGui::BeginChild(("##tabBar"+id).c_str(), barSize, false, tabFlags);
    {
      ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos()) + (mHorizontal ? Vec2f(TABMENU_TAB_PADDING.x, 0) : Vec2f(0, TABMENU_TAB_PADDING.y))/2.0f);
      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, mHorizontal ? TABMENU_TAB_PADDING.yx() : TABMENU_TAB_PADDING);
      for(int i = 0; i < mTabs.size(); i++)
        {
          TabDesc &tab = mTabs[i];
          Vec2f lSize = ImGui::CalcTextSize(tab.label.c_str()); if(!mHorizontal) { std::swap(lSize.x,   lSize.y); }
          Vec2f tabSize = Vec2f(tabL, barW);                    if(!mHorizontal) { std::swap(tabSize.x, tabSize.y); }
          Vec2f padding = (tabSize - lSize)/2.0f;
          Vec2f sp = ImGui::GetCursorScreenPos();

          // draw tab button
          ImGui::PushStyleColor(ImGuiCol_Button,        (mSelected == i ? activeColor        : inactiveColor));
          ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (mSelected == i ? activeHoveredColor : hoveredColor ));
          ImGui::PushStyleColor(ImGuiCol_ButtonActive,  (mSelected == i ? activeClickedColor : clickedColor ));
          if(ImGui::Button(((mHorizontal ? tab.label : "") + "##tab" + std::to_string(i) + id).c_str(), tabSize))
            { select(i); }
          ImGui::PopStyleColor(3);
        
          if(!mHorizontal) // add vertical button text manually
            { AddTextVertical(ImGui::GetWindowDrawList(), tab.label.c_str(), sp + padding, Vec4f(1.0f, 1.0f, 1.0f, 1.0f)); }
          else // horizontal tabs
            { ImGui::SameLine(); }
        }
      ImGui::PopStyleVar();
    }
    ImGui::EndChild(); ImGui::PopStyleVar();
  }
  ImGui::EndGroup();
}
