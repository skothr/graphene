#include "tabMenu.hpp"

#include <imgui.h>
#include "imtools.hpp"


TabMenu::TabMenu(int barWidth, int length, bool collapsible)
  : mBarWidth(barWidth), mLength(length) { }

Vec2f TabMenu::getSize() const
{
  ImGuiStyle &style = ImGui::GetStyle();
  float width  = mBarWidth + (mSelected >= 0 ? (mTabs[mSelected].width + 2.0f*style.WindowPadding.x + TABMENU_MENU_PADDING) : 0);
  float length = mLength;
  return Vec2f(width, length);
}

float TabMenu::getBarWidth() const
{
  if(mBarWidth == 0.0f)
    {
      float barW = 0.0f;
      for(auto &tab : mTabs) { barW = std::max(barW, ImGui::CalcTextSize(tab.label.c_str()).y); }
      return barW + 2.0f*TABMENU_TAB_PADDING;
    }
  else { return mBarWidth; }
}

float TabMenu::getTabLength() const
{
  float tabL = 0.0f;
  for(auto &tab : mTabs) { tabL = std::max(tabL, ImGui::CalcTextSize(tab.label.c_str()).x); }
  return tabL + 2.0f*TABMENU_TAB_PADDING;
}

void TabMenu::setBarWidth(int w) { mBarWidth = w; }
void TabMenu::setLength(int l)   { mLength   = l; }
void TabMenu::setCollapsible(bool collapsible)          { mCollapsible = collapsible; }

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

void TabMenu::draw()
{
  ImGuiStyle &style = ImGui::GetStyle();
  // tab colors
  Vec4f inactiveColor     (0.2f,  0.2f,  0.2f,  1.0f); Vec4f activeColor       (0.5f, 0.5f, 0.5f, 1.0f);
  Vec4f hoveredColor      (0.35f, 0.35f, 0.35f, 1.0f); Vec4f clickedColor      (0.8f, 0.8f, 0.8f, 1.0f);
  Vec4f activeHoveredColor(0.65f, 0.65f, 0.65f, 1.0f); Vec4f activeClickedColor(0.8f, 0.8f, 0.8f, 1.0f);

  float barW = mBarWidth;
  float tabL = getTabLength();
  Vec2f spacing = ImGui::GetStyle().ItemSpacing;

  ImGuiWindowFlags outerFlags = (ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse); // outer child flags
  ImGuiWindowFlags innerFlags = ImGuiWindowFlags_None;                                               // inner child flags
  ImGuiWindowFlags tabFlags   = (ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse); // tab bar child flags
  
  Vec2f sp0 = ImGui::GetCursorScreenPos();
  ImGui::BeginGroup();
  {
    // update tab menu widths
    for(auto &tab : mTabs)
      {
        if(tab.fixedWidth >= 0)      { tab.width = tab.fixedWidth; }
        if(tab.width < tab.minWidth) { tab.width = tab.minWidth; }
      }
    
    // draw open menu
    if(mSelected >= 0)
      {
        TabDesc &tab = mTabs[mSelected];
        Vec2f outerSize = Vec2f(tab.width+2.0f*style.WindowPadding.x, std::max(mLength, style.ItemSpacing.y + tabL*mTabs.size()));
        ImGui::BeginChild("##tabMenuOuter", outerSize, true, outerFlags);
        {
          // draw title
          Vec2f tSize;
          if(!tab.title.empty())
            {
              if(tab.titleFont) { ImGui::PushFont(tab.titleFont); }
              tSize = Vec2f(ImGui::CalcTextSize(tab.title.c_str()));
              ImGui::SetCursorPos(Vec2f(ImGui::GetCursorPos()) + Vec2f((outerSize.x-tSize.x)/2.0f, 0.0f));
              ImGui::TextUnformatted(tab.title.c_str());
              if(tab.titleFont) { ImGui::PopFont(); }
            }
          // scrollable inner child
          Vec2f innerSize = Vec2f(tab.width, mLength - tSize.y - 2.0f*style.WindowPadding.y - style.ItemSpacing.y);
          ImGui::BeginChild("##tabMenuInner", innerSize, true, innerFlags);
          {
            ImGui::BeginGroup(); tab.drawMenu(); ImGui::EndGroup();
            if(tab.fixedWidth < 0) { tab.width = ImGui::GetItemRectMax().x - ImGui::GetItemRectMin().x; }
            if(tab.minWidth  >= 0) { tab.width = std::min(tab.width, tab.minWidth); }
            innerSize = Vec2f(tab.width, mLength - tSize.y - 2.0f*style.WindowPadding.y);
            ImGui::SetWindowSize(innerSize);
          }
          ImGui::EndChild();
        }
        ImGui::EndChild();
        ImGui::SameLine();
      }
    // draw tabs
    ImGui::BeginChild("##tabBar", Vec2f(barW, std::max(mLength, style.ItemSpacing.y + tabL*mTabs.size())), false, tabFlags);
    {
      for(int i = 0; i < mTabs.size(); i++)
        {
          TabDesc &tab = mTabs[i];
          Vec2f lSize = ImGui::CalcTextSize(tab.label.c_str()); std::swap(lSize.x, lSize.y);
          Vec2f tabSize = Vec2f(barW, tabL);
          Vec2f padding = (tabSize - lSize)/2.0f;

          Vec2f sp = ImGui::GetCursorScreenPos();
          ImGui::PushStyleColor(ImGuiCol_Button,        (mSelected == i ? activeColor        : inactiveColor));
          ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (mSelected == i ? activeHoveredColor : hoveredColor ));
          ImGui::PushStyleColor(ImGuiCol_ButtonActive,  (mSelected == i ? activeClickedColor : clickedColor ));
          if(ImGui::Button(("##tab" + std::to_string(i)).c_str(), tabSize)) { select(i); }
          ImGui::PopStyleColor(3);
          AddTextVertical(ImGui::GetWindowDrawList(), tab.label.c_str(), sp + padding, Vec4f(1.0f, 1.0f, 1.0f, 1.0f));
        }
    }
    ImGui::EndChild();

  }
  ImGui::EndGroup();
}
