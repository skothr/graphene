#ifndef SCREEN_VIEW_HPP
#define SCREEN_VIEW_HPP

#include "tools.hpp"
#include "vector.hpp"
#include "rect.hpp"
#include "settingForm.hpp"
#include "render.cuh"

// simple flag type for mouse buttons 
enum MouseButton
  { // NOTE: Values should match ImGuiButtonFlags_ in imgui.h
   MOUSEBTN_NONE   =  0,         // ImGuiButtonFlags_None
   MOUSEBTN_LEFT   =  1 << 0,    // ImGuiButtonFlags_MouseButtonLeft
   MOUSEBTN_RIGHT  =  1 << 1,    // ImGuiButtonFlags_MouseButtonRight
   MOUSEBTN_MIDDLE =  1 << 2,    // ImGuiButtonFlags_MouseButtonMiddle
   MOUSEBTN_ALL    = (1 << 3)-1, // All flags set (next available flag minus one (e.g. (1 << 4) --> 0x10000 --> 0x10000-1 = 0x01111)
  };
ENUM_FLAG_OPERATORS(MouseButton)

template<typename T>
struct ScreenView
{
  Rect2f r;
  Rect2f rSim;
  bool         hovered  = false;
  MouseButton  clicked  = MOUSEBTN_NONE; // mouse buttons that were clicked
  int          mods     = 0;             // modifier keys that were held when clicked (GLFW_MOD_XXXX)
  Vec2f        clickPos;                 // screen position of click
  Vector<T, 3> mposSim;                  // sim position of click
  Vector<T, 3> mposFace;                 // face of cube mouse is over (e.g. <1,0,0> for +X face)
  
  RenderParams<T> rp;
  SettingForm     renderSettings;
  bool            settingsOpen = false;
  std::string     name;

  ScreenView();
  
  MouseButton clickBtns(MouseButton mask=MOUSEBTN_ALL) const { return (clicked & mask); }
  int         clickMods(int mask=0)                    const { return (mods    & mask); }

  bool operator==(const ScreenView &other) const { return  (r == other.r);   }
  bool operator!=(const ScreenView &other) const { return !(*this == other); }

  // scales output from 3D Camera<T> transform (P0 --> camera.nearClip(camera.worldToScreen(P0), camera.worldToScreen(P1)))
  Vec2f toScreen(const Vec2f &p) const { return r.p1 + p*r.size(); }

  Vec2f simToScreen2D(const Vec2f &pSim,    const Rect2f &simView, bool vector=false) const;
  Vec2f screenToSim2D(const Vec2f &pScreen, const Rect2f &simView, bool vector=false) const;
  // (to use X and Y components of 3D vectors)
  Vec2f simToScreen2D(const Vec3f &pSim,    const Rect2f &simView, bool vector=false) const;
  Vec2f screenToSim2D(const Vec3f &pScreen, const Rect2f &simView, bool vector=false) const;

  bool drawMenu();
  
};


template<typename T>
ScreenView<T>::ScreenView()
{
  // visualized render data
  auto *sRSMP = new Setting<bool> ("Simple", "simple", &rp.simple); renderSettings.add(sRSMP);
  sRSMP->setHelp("Hide extra options for a more compact interface and slightly faster performance");
  
  auto *sRCOF  = new Setting<float> ("Opacity",    "fOpacity",       &rp.fOpacity);      sRCOF->setFormat (MULT_SMALLSTEP, MULT_BIGSTEP, MULT_FORMAT);
  auto *sRCBRF = new Setting<float> ("Brightness", "fBrightness",    &rp.fBrightness);   sRCBRF->setFormat(MULT_SMALLSTEP, MULT_BIGSTEP, MULT_FORMAT);
  auto *sRCOS  = new Setting<float> ("Opacity",     "emOpacity",     &rp.emOpacity);     sRCOS->setFormat (MULT_SMALLSTEP, MULT_BIGSTEP, MULT_FORMAT);
  auto *sRCBRS = new Setting<float> ("Brightness",  "emBrightness",  &rp.emBrightness);  sRCBRS->setFormat(MULT_SMALLSTEP, MULT_BIGSTEP, MULT_FORMAT);
  auto *sRCOM  = new Setting<float> ("Opacity",     "matOpacity",    &rp.matOpacity);    sRCOM->setFormat (MULT_SMALLSTEP, MULT_BIGSTEP, MULT_FORMAT);
  auto *sRCBRM = new Setting<float> ("Brightness",  "matBrightness", &rp.matBrightness); sRCBRM->setFormat(MULT_SMALLSTEP, MULT_BIGSTEP, MULT_FORMAT);
  SettingGroup *fDGroup   = new SettingGroup("Fluid",    "fRender",   { sRCBRF, sRCOF }); renderSettings.add(fDGroup);
  fDGroup->setColumns(1);   fDGroup->setCollapsible(true);
  SettingGroup *emDGroup  = new SettingGroup("EM",       "emRender",  { sRCBRS, sRCOS }); renderSettings.add(emDGroup);
  emDGroup->setColumns(1);  emDGroup->setCollapsible(true);
  SettingGroup *matDGroup = new SettingGroup("Material", "matRender", { sRCBRM, sRCOM }); renderSettings.add(matDGroup);
  matDGroup->setColumns(1); matDGroup->setCollapsible(true);
  
  auto *g = fDGroup;
  SettingGroup *subgroup = nullptr;
  for(long long i = 0LL; i < RENDER_FLAG_COUNT; i++)
    {
      RenderFlags f = (RenderFlags)(1LL << i);
      float4     *c = rp.getColor(f);
      T          *m = rp.getMult(f);
      
      if     (f == FLUID_RENDER_EMOFFSET)  { g = emDGroup;  subgroup = nullptr; }
      else if(f == FLUID_RENDER_MATOFFSET) { g = matDGroup; subgroup = nullptr; }
      if(!c || !m) { std::cout << "====> WARNING: DisplayInterface skipping RenderFlag " << renderFlagName(f) << " (2^" << i << ")\n"; continue; }
      
      std::string name  = renderFlagName(f);
      std::string id    = name + "(2^"+std::to_string(i)+")";
      std::string idC   = id + "-color";
      std::string idM   = id + "-mult";
      
      std::string gName = renderFlagGroupName(f);
      if(gName.find("INVALID") == std::string::npos) // create new tree group (TODO --> tree)
        { subgroup = new SettingGroup(gName, gName); g->add(subgroup); }
      
      auto sRFC = new ColorSetting("", idC, c); sRFC->setFormat(COLOR_SMALLSTEP, COLOR_BIGSTEP, COLOR_FORMAT);
      auto sRFM = new Setting<T>  ("", idM, m); sRFM->setFormat(MULT_SMALLSTEP,  MULT_BIGSTEP,  MULT_FORMAT);
      
      SettingGroup *rfg = new SettingGroup(name, id, { sRFC, sRFM });
      rfg->setHorizontal(true); rfg->setColumns(2); rfg->setToggle(rp.getToggle(f));
      rfg->setVisibleCallback([f, this]() -> bool { return ((RenderParams<T>::MAIN & f) || !rp.simple); });
      (subgroup ? subgroup : g)->add(rfg);
    }
}

template<typename T>
bool ScreenView<T>::drawMenu()
{
  bool changed = false;

  std::string menuName = "##popup-" + name;
  if(ImGui::Button(("Settings##"+name).c_str()))
    {
      ImGui::OpenPopup(menuName.c_str());
      ImGui::SetNextWindowPos(ImGui::GetMousePos());
      ImGui::SetNextWindowSize(renderSettings.getSize());
    }

  ImGuiWindowFlags popupFlags = (ImGuiWindowFlags_NoDecoration |
                                 ImGuiWindowFlags_NoCollapse   |
                                 ImGuiWindowFlags_NoMove);
  if(ImGui::BeginPopup(menuName.c_str(), popupFlags))
    {
      changed |renderSettings.draw();
      settingsOpen = true;
      ImGui::EndPopup();
    }
  else { settingsOpen = false; }
  
  return changed;
}


/////////////////////////////////////////////////////////////////////////////////////////////////
// Conversion between screen space and sim space
//// p0 --> optional screen-space offset
template<typename T>
Vec2f ScreenView<T>::simToScreen2D(const Vec2f &pSim, const Rect2f &simView, bool vector) const
{
  Vec2f pScreen = (pSim-simView.p1*(vector?0:1)) * (r.size()/simView.size());
  if(!vector) { pScreen = Vec2f(r.p1.x + pScreen.x, r.p2.y - pScreen.y); }
  return pScreen;
}
template<typename T>
Vec2f ScreenView<T>::screenToSim2D(const Vec2f &pScreen, const Rect2f &simView, bool vector) const
{
  Vec2f pSim = (pScreen-r.p1*(vector?0:1)) * (simView.size()/r.size());
  if(!vector) { pSim = Vec2f(simView.p1.x + pSim.x, simView.p2.y - pSim.y); }
  return pSim;
}

// for 3D vectors, uses X/Y components (for convenience)
template<typename T>
Vec2f ScreenView<T>::simToScreen2D(const Vec3f &pSim, const Rect2f &simView, bool vector) const
{ return simToScreen2D(Vec2f(pSim.x,    pSim.y),    simView, vector); }
template<typename T>
Vec2f ScreenView<T>::screenToSim2D(const Vec3f &pScreen, const Rect2f &simView, bool vector) const
{ return screenToSim2D(Vec2f(pScreen.x, pScreen.y), simView, vector); }


#endif // SCREEN_VIEW_HPP
