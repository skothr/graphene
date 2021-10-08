#include "keyFrameWidget.hpp"



#include <sstream>
#include <fstream>
#include <imgui.h>
#include <nlohmann/json.hpp> // json implementation
using json = nlohmann::json;

#include "vector-operators.h"
#include "tools.hpp"


// PEN BASE CLASS
template<typename T>
inline json penToJSON(const Pen<T> *pen)
{
  json js = json::object(); std::stringstream ss;

  // ordered_map 
  js["active"]    = pen->active;
  js["cellAlign"] = pen->cellAlign;
  js["square"]    = pen->square;
  js["radial"]    = pen->radial;
  js["speed"]     = pen->speed;
  js["depth"]     = pen->depth;
  js["radius0"]   = to_string(pen->radius0);
  js["radius1"]   = to_string(pen->radius1);
  js["rDist"]     = to_string(pen->rDist);
  js["mult"]      = pen->mult;
  js["sizeMult"]  = to_string(pen->sizeMult);
  js["xyzMult"]   = to_string(pen->xyzMult);
  js["speedMult"] = pen->speedMult;
  return js;
}
template<typename T>
inline bool penFromJSON(const json &js, SignalPen<T> *penOut)
{
  using VT3 = float3;
  bool success = true;
  if(js.contains("active"))    { penOut->active    = js["active"];    }
  if(js.contains("cellAlign")) { penOut->cellAlign = js["cellAlign"]; }
  if(js.contains("square"))    { penOut->square    = js["square"];    }
  if(js.contains("radial"))    { penOut->radial    = js["radial"];    }
  if(js.contains("speed"))     { penOut->speed     = js["speed"];     }
  if(js.contains("depth"))     { penOut->depth     = js["depth"];     }
  if(js.contains("mult"))      { penOut->mult      = js["mult"];      }
  if(js.contains("speedMult")) { penOut->speedMult = js["speedMult"]; }
  if(js.contains("radius0"))   { penOut->radius0   = from_string<VT3>(js["radius0" ].get<std::string>());  }
  if(js.contains("radius1"))   { penOut->radius1   = from_string<VT3>(js["radius1" ].get<std::string>());  }
  if(js.contains("rDist"))     { penOut->rDist     = from_string<VT3>(js["rDist"   ].get<std::string>());  }
  if(js.contains("sizeMult"))  { penOut->sizeMult  = js["sizeMult"].get<float>();  }
  if(js.contains("xyzMult"))   { penOut->xyzMult   = from_string<VT3>(js["xyzMult" ].get<std::string>());  }
  return success;
}


#define MOD_COLUMNS "  r  | r^2 |  θ  | sin | cos "

// SIGNAL PEN
template<typename T>
inline json penToJSON(const SignalPen<T> &pen)
{
  json js = penToJSON(&pen);
  js["wavelength"] = pen.wavelength;
  js["frequency"]  = pen.frequency;
  
  std::stringstream ss;
  js["pV" ]["base"] = to_string(pen.pV.base );
  js["pP" ]["base"] = to_string(pen.pP.base );
  js["pQn"]["base"] = to_string(pen.pQn.base);
  js["pQp"]["base"] = to_string(pen.pQp.base);
  js["pQv"]["base"] = to_string(pen.pQv.base);
  js["pE" ]["base"] = to_string(pen.pE.base );
  js["pB" ]["base"] = to_string(pen.pB.base );
  for(int i = 0; i < pen.pV.modArr.size();  i++) { ss << (pen.pV.mods [i] ? "1":"0") << " "; } js["pV" ]["mods"] = ss.str(); ss.str(""); ss.clear();
  for(int i = 0; i < pen.pP.modArr.size();  i++) { ss << (pen.pP.mods [i] ? "1":"0") << " "; } js["pP" ]["mods"] = ss.str(); ss.str(""); ss.clear();
  for(int i = 0; i < pen.pQn.modArr.size(); i++) { ss << (pen.pQn.mods[i] ? "1":"0") << " "; } js["pQn"]["mods"] = ss.str(); ss.str(""); ss.clear();
  for(int i = 0; i < pen.pQp.modArr.size(); i++) { ss << (pen.pQp.mods[i] ? "1":"0") << " "; } js["pQp"]["mods"] = ss.str(); ss.str(""); ss.clear();
  for(int i = 0; i < pen.pQv.modArr.size(); i++) { ss << (pen.pQv.mods[i] ? "1":"0") << " "; } js["pQv"]["mods"] = ss.str(); ss.str(""); ss.clear();
  for(int i = 0; i < pen.pE.modArr.size();  i++) { ss << (pen.pE.mods [i] ? "1":"0") << " "; } js["pE" ]["mods"] = ss.str(); ss.str(""); ss.clear();
  for(int i = 0; i < pen.pB.modArr.size();  i++) { ss << (pen.pB.mods [i] ? "1":"0") << " "; } js["pB" ]["mods"] = ss.str(); ss.str(""); ss.clear();
  
  // js["pV"]  = std::vector<std::pair<std::string, std::string>>
  //   {{" r ", (pen.pV.multR   ?"1":"0")}, {"r^2", (pen.pV.multR_2 ?"1":"0")}, {" θ ", (pen.pV.multT ?"1":"0")},
  //    {"sin", (pen.pV.multSin ?"1":"0")}, {"cos", (pen.pV.multCos ?"1":"0")}};
  // js["pP"]  = std::vector<std::pair<std::string, std::string>>
  //   {{" r ", (pen.pP.multR   ?"1":"0")}, {"r^2", (pen.pP.multR_2 ?"1":"0")}, {" θ ", (pen.pP.multT ?"1":"0")},
  //    {"sin", (pen.pP.multSin ?"1":"0")}, {"cos", (pen.pP.multCos ?"1":"0")}};
  // js["pQn"] = std::vector<std::pair<std::string, std::string>>
  //   {{" r ", (pen.pQn.multR  ?"1":"0")}, {"r^2", (pen.pQn.multR_2?"1":"0")}, {" θ ", (pen.pQn.multT?"1":"0")},
  //    {"sin", (pen.pQn.multSin?"1":"0")}, {"cos", (pen.pQn.multCos?"1":"0")}};
  // js["pQp"] = std::vector<std::pair<std::string, std::string>>
  //   {{" r ", (pen.pQp.multR  ?"1":"0")}, {"r^2", (pen.pQp.multR_2?"1":"0")}, {" θ ", (pen.pQp.multT?"1":"0")},
  //    {"sin", (pen.pQp.multSin?"1":"0")}, {"cos", (pen.pQp.multCos?"1":"0")}};
  // js["pQv"] = std::vector<std::pair<std::string, std::string>>
  //   {{" r ", (pen.pQv.multR  ?"1":"0")}, {"r^2", (pen.pQv.multR_2?"1":"0")}, {" θ ", (pen.pQv.multT?"1":"0")},
  //    {"sin", (pen.pQv.multSin?"1":"0")}, {"cos", (pen.pQv.multCos?"1":"0")}};
  // js["pE"]  = std::vector<std::pair<std::string, std::string>>
  //   {{" r ", (pen.pE.multR   ?"1":"0")}, {"r^2", (pen.pE.multR_2 ?"1":"0")}, {" θ ", (pen.pE.multT ?"1":"0")},
  //    {"sin", (pen.pE.multSin ?"1":"0")}, {"cos", (pen.pE.multCos ?"1":"0")}};
  // js["pB"]  = std::vector<std::pair<std::string, std::string>>
  //   {{" r ", (pen.pB.multR   ?"1":"0")}, {"r^2", (pen.pB.multR_2 ?"1":"0")}, {" θ ", (pen.pB.multT ?"1":"0")},
  //    {"sin", (pen.pB.multSin ?"1":"0")}, {"cos", (pen.pB.multCos ?"1":"0")}};
  // TODO?: add multipliers for each option
  // js["pV"]  = {{"base"   : to_string(pen.pV.base)},   {"multR"   : to_string(pen.pV.multR)},    {"multR^2" : to_string(pen.pV.multR)},
  //              {"multT"  : to_string(pen.pV.multT)},  {"multSin" : to_string(pen.pV.multSin)},  {"multCos" : to_string(pen.pV.multCos)}  };
  // js["pP"]  = {{"base"   : to_string(pen.pP.base)},   {"multR"   : to_string(pen.pP.multR)},    {"multR^2" : to_string(pen.pP.multR)},
  //              {"multT"  : to_string(pen.pP.multT)},  {"multSin" : to_string(pen.pP.multSin)},  {"multCos" : to_string(pen.pP.multCos)}  };
  // js["pQn"] = {{"base"   : to_string(pen.pQn.base)},  {"multR"   : to_string(pen.pQn.multR)},   {"multR^2" : to_string(pen.pQn.multR)},
  //              {"multT"  : to_string(pen.pQn.multT)}, {"multSin" : to_string(pen.pQn.multSin)}, {"multCos" : to_string(pen.pQn.multCos)} };
  // js["pQp"] = {{"base"   : to_string(pen.pQp.base)},  {"multR"   : to_string(pen.pQp.multR)},   {"multR^2" : to_string(pen.pQp.multR)},
  //              {"multT"  : to_string(pen.pQp.multT)}, {"multSin" : to_string(pen.pQp.multSin)}, {"multCos" : to_string(pen.pQp.multCos)} };
  // js["pQv"] = {{"base"   : to_string(pen.pQv.base)},  {"multR"   : to_string(pen.pQv.multR)},   {"multR^2" : to_string(pen.pQv.multR)},
  //              {"multT"  : to_string(pen.pQv.multT)}, {"multSin" : to_string(pen.pQv.multSin)}, {"multCos" : to_string(pen.pQv.multCos)} };
  // js["pE"]  = {{"base"   : to_string(pen.pE.base)},   {"multR"   : to_string(pen.pE.multR)},    {"multR^2" : to_string(pen.pE.multR)},
  //              {"multT"  : to_string(pen.pE.multT)},  {"multSin" : to_string(pen.pE.multSin)},  {"multCos" : to_string(pen.pE.multCos)}  };
  // js["pB"]  = {{"base"   : to_string(pen.pB.base)},   {"multR"   : to_string(pen.pB.multR)},    {"multR^2" : to_string(pen.pB.multR)},
  //              {"multT"  : to_string(pen.pB.multT)},  {"multSin" : to_string(pen.pB.multSin)},  {"multCos" : to_string(pen.pB.multCos)}  };
  
  return js;
}
template<typename T>
inline bool penFromJSON(const json &js, SignalPen<T> &penOut)
{
  using VT3 = float3;
  bool success = true;
  penFromJSON(js, &penOut);
  if(js.contains("signal"))
    {
      json jss = js["signal"];
      if(jss.contains("wavelength")) { penOut.wavelength = jss["wavelength"]; }
      if(jss.contains("frequency"))  { penOut.frequency  = jss["frequency"];  }

      std::stringstream ss; int b;
      if(jss.contains("pV")  && jss["pV"].contains("mods"))
        {
          ss.str(jss["pV"]["mods"].get<std::string>());
          for(int i = 0; i < penOut.pV.modArr.size();   i++) { ss >> b; penOut.pV.mods[i]  = (b != 0); }
          ss.str(""); ss.clear();
        }
      if(jss.contains("pP")  && jss["pP"].contains("mods"))
        {
          ss.str(jss["pP"]["mods"].get<std::string>());
          for(int i = 0; i < penOut.pP.modArr.size();   i++) { ss >> b; penOut.pP.mods[i]  = (b != 0); }
          ss.str(""); ss.clear();
        }
      if(jss.contains("pQn") && jss["pQn"].contains("mods"))
        {
          ss.str(jss["pQn"]["mods"].get<std::string>());
          for(int i = 0; i < penOut.pQn.modArr.size();  i++) { ss >> b; penOut.pQn.mods[i] = (b != 0); }
          ss.str(""); ss.clear();
        }
      if(jss.contains("pQp") && jss["pQp"].contains("mods"))
        {
          ss.str(jss["pQp"]["mods"].get<std::string>());
          for(int i = 0; i < penOut.pQp.modArr.size();  i++) { ss >> b; penOut.pQp.mods[i] = (b != 0); }
          ss.str(""); ss.clear();
        }
      if(jss.contains("pQv") && jss["pQv"].contains("mods"))
        {
          ss.str(jss["pQv"]["mods"].get<std::string>());
          for(int i = 0; i < penOut.pQv.modArr.size();  i++) { ss >> b; penOut.pQv.mods[i] = (b != 0); }
          ss.str(""); ss.clear();
        }
      if(jss.contains("pE")  && jss["pE"].contains("mods"))
        {
          ss.str(jss["pE"]["mods"].get<std::string>());
          for(int i = 0; i < penOut.pE.modArr.size();   i++) { ss >> b; penOut.pE.mods[i]  = (b != 0); }
          ss.str(""); ss.clear();
        }
      if(jss.contains("pB")  && jss["pB"].contains("mods"))
        {
          ss.str(jss["pB"]["mods"].get<std::string>());
          for(int i = 0; i < penOut.pB.modArr.size();   i++) { ss >> b; penOut.pB.mods[i]  = (b != 0); }
          ss.str(""); ss.clear();
        }


      ////
      // accessing direct variables --> (first iteration)
      ////
      // if(jss.contains("pV")) { json jsp = jss["pV"];
      //     if(jsp.contains("base"))  { penOut.pV.base dt=from_string<VT3>(jsp["base"].get<std::string>()); }
      //     if(jsp.contains(" r ")) { penOut.pV.mult     = jsp[" r "].get<std::string>(); }
      //     if(jsp.contains("r^2")) { penOut.pV.multR_2  = sp["r^2"].get<std::string>(); }
      //     if(jsp.contains(" θ ")) { penOut.pV.multT    = sp[" θ "].get<std::string>(); }
      //     if(jsp.contains("sin")) { penOut.pV.multSin   = jsp["sin"].get<std::string>(); }
      //     if(jsp.contains("cos")) { penOut.pV.multCos   = jsp["cos"].get<std::string>(); } }
      // if(jss.contains("pP")) { json jsp = jss["pP"];
      //     if(jsp.contains("base")) { penOut.pP.base     = jsp["base"].get<float>();      }
      //     if(jsp.contains(" r "))  { penOut.pP.mult     = jsp[" r "].get<std::string>(); }
      //     if(jsp.contains("r^2"))  { penOut.pP.multR_2  = jsp["r^2"].get<std::string>(); }
      //     if(jsp.contains(" θ "))  { penOut.pP.multT    = jsp[" θ "].get<std::string>(); }
      //     if(jsp.contains("sin"))  { penOut.pP.multSin  = jsp["sin"].get<std::string>(); }
      //     if(jsp.contains("cos"))  { penOut.pP.multCos  = jsp["cos"].get<std::string>(); } }
      // if(jss.contains("pQn")) { json jsp = jss["pQn"];
      //     if(jsp.contains("base")) { penOut.pQn.base    = jsp["base"].get<float>();      }
      //     if(jsp.contains(" r "))  { penOut.pQn.mult    = jsp[" r "].get<std::string>(); }
      //     if(jsp.contains("r^2"))  { penOut.pQn.multR_2 = jsp["r^2"].get<std::string>(); }
      //     if(jsp.contains(" θ "))  { penOut.pQn.multT   = jsp[" θ "].get<std::string>(); }
      //     if(jsp.contains("sin"))  { penOut.pQn.multSin = jsp["sin"].get<std::string>(); }
      //     if(jsp.contains("cos"))  { penOut.pQn.multCos = jsp["cos"].get<std::string>(); } }
      // if(jss.contains("pQp")) { json jsp = jss["pQp"];
      //     if(jsp.contains("base")) { penOut.pQp.base    = jsp["base"].get<float>();      }
      //     if(jsp.contains(" r "))  { penOut.pQp.mult    = jsp[" r "].get<std::string>(); }
      //     if(jsp.contains("r^2"))  { penOut.pQp.multR_2 = jsp["r^2"].get<std::string>(); }
      //     if(jsp.contains(" θ "))  { penOut.pQp.multT   = jsp[" θ "].get<std::string>(); }
      //     if(jsp.contains("sin"))  { penOut.pQp.multSin = jsp["sin"].get<std::string>(); }
      //     if(jsp.contains("cos"))  { penOut.pQp.multCos = jsp["cos"].get<std::string>(); } }
      // if(jss.contains("pQv")) { json jsp = jss["pQv"];
      //     if(jsp.contains("base")) { penOut.pQv.base    = from_string<VT3>(jsp["base"].get<std::string>()); }
      //     if(jsp.contains(" r "))  { penOut.pQv.mult    = jsp[" r "].get<std::string>(); }
      //     if(jsp.contains("r^2"))  { penOut.pQv.multR_2 = jsp["r^2"].get<std::string>(); }
      //     if(jsp.contains(" θ "))  { penOut.pQv.multT   = jsp[" θ "].get<std::string>(); }
      //     if(jsp.contains("sin"))  { penOut.pQv.multSin = jsp["sin"].get<std::string>(); }
      //     if(jsp.contains("cos"))  { penOut.pQv.multCos = jsp["cos"].get<std::string>(); } }
      // if(jss.contains("pE")) { json jsp = jss["pE"];
      //     if(jsp.contains("base")) { penOut.pE.base    = from_string<VT3>(jsp["base"].get<std::string>()); }
      //     if(jsp.contains(" r "))  { penOut.pE.mult    = jsp[" r "].get<std::string>(); }
      //     if(jsp.contains("r^2"))  { penOut.pE.multR_2 = jsp["r^2"].get<std::string>(); }
      //     if(jsp.contains(" θ "))  { penOut.pE.multT   = jsp[" θ "].get<std::string>(); }
      //     if(jsp.contains("sin"))  { penOut.pE.multSin = jsp["sin"].get<std::string>(); }
      //     if(jsp.contains("cos"))  { penOut.pE.multCos = jsp["cos"].get<std::string>(); } }
      // if(jss.contains("pB")) { json jsp = jss["pB"];
      //     if(jsp.contains("base")) { penOut.pB.base    = from_string<VT3>(jsp["base"].get<std::string>()); }
      //     if(jsp.contains(" r "))  { penOut.pB.mult    = jsp[" r "].get<std::string>(); }
      //     if(jsp.contains("r^2"))  { penOut.pB.multR_2 = jsp["r^2"].get<std::string>(); }
      //     if(jsp.contains(" θ "))  { penOut.pB.multT   = jsp[" θ "].get<std::string>(); }
      //     if(jsp.contains("sin"))  { penOut.pB.multSin = jsp["sin"].get<std::string>(); }
      //     if(jsp.contains("cos"))  { penOut.pB.multCos = jsp["cos"].get<std::string>(); } }
    }
  return success;
}

// MATERIAL PEN
template<typename T>
inline json penToJSON(const MaterialPen<T> &pen)
{
  json js  = penToJSON(&pen);
  json jsm = json::object();
  jsm["vacuum"] = pen.mat.vacuum();
  jsm["permittivity"], pen.mat.ep;
  jsm["permeability"], pen.mat.mu;
  jsm["conductivity"], pen.mat.sig;
  js["material"] = jsm;
  return js;
}

template<typename T>
inline bool penFromJSON(const json &js, MaterialPen<T> &penOut)
{
  using VT3 = typename DimType<T, 3>::VEC_T;
  bool success = true;
  penFromJSON(js, &penOut);
  if(js.contains("material"))
    {
      json jsm = js["material"];
      if(jsm.contains("permittivity")) { penOut.material.ep  = jsm["permittivity"]; }
      if(jsm.contains("permeability")) { penOut.material.mu  = jsm["permeability"]; }
      if(jsm.contains("conductivity")) { penOut.material.sig = jsm["conductivity"]; }
      if(jsm.contains("vacuum"))       { penOut.material.setVacuum(jsm["vacuum"]);  }
    }
  return success;
}

template<typename T>
inline json view2DToJSON(const Rect<T> &view)
{
  json js = json::object();
  js["p1"] = to_string(view.p1);
  js["p2"] = to_string(view.p2);
  return js;
}
template<typename T>
inline bool view2DFromJSON(const json &js, Rect<T> &viewOut)
{
  using VT2 = float2;
  bool success = true;
  if(js.contains("pos"))   { viewOut.p1 = from_string<VT2>(js["p1"]);   }
  if(js.contains("dir"))   { viewOut.p2 = from_string<VT2>(js["p2"]);   }
  return success;
}

template<typename T>
inline json view3DToJSON(const CameraDesc<T> &cam)
{
  json js = json::object();
  js["pos"]   = to_string(cam.pos);
  js["dir"]   = to_string(cam.dir);
  js["right"] = to_string(cam.right);
  js["up"]    = to_string(cam.up);
  js["fov"]   = cam.fov;
  js["near"]  = cam.near;
  js["far"]   = cam.far;
  return js;
}
template<typename T>
inline bool view3DFromJSON(const json &js, CameraDesc<T> &camOut)
{
  using VT3 = float3;
  bool success = true;
  if(js.contains("pos"))   { camOut.pos   = from_string<VT3>(js["pos"]);   }
  if(js.contains("dir"))   { camOut.dir   = from_string<VT3>(js["dir"]);   }
  if(js.contains("right")) { camOut.right = from_string<VT3>(js["right"]); }
  if(js.contains("up"))    { camOut.up    = from_string<VT3>(js["up"]);    }
  if(js.contains("fov"))   { camOut.fov   = js["fov"];  }
  if(js.contains("near"))  { camOut.near  = js["near"]; }
  if(js.contains("far"))   { camOut.far   = js["far"];  }
  return success;
}


json KeyFrameWidget::toJSON()
{
  json js = json::object();

  // // initial state(s)
  // json iSources = json::object();
  // for(const auto &iter : mSourcesInit)
  //   {
  //     json s = json::object();
  //     s["id"]  = iter.second.id;
  //     s["pos"] = to_string(iter.second.pos);
  //     s["pen"] = penToJSON(iter.second.pen);
  //     if(iter.second.start > 0) { s["start"] = iter.second.start; }
  //     if(iter.second.end > 0)   { s["end"  ] = iter.second.end;   }
  //     iSources[iter.second.id] = s;
  //   } js["initSources"]   = iSources;
  // json iPlaced = json::object();
  // for(int i = 0; i < mPlacedInit.size(); i++)
  //   {
  //     const auto &m = mPlacedInit[i];
  //     std::string id = std::to_string(i);
  //     json m = json::object();
  //     m["pos"] = to_string(iter.second.pos);
  //     m["pen"] = penToJSON(iter.second.pen);
  //     iPlaced[iter.first] = m;
  //   } js["initMatPlaced"] = iPlaced;

  // events
  json events = json::object();
  for(auto iter : mEvents)
    {
      json jse = json::array();
      std::stringstream ss; ss << iter.first;
      for(auto e : iter.second)
        {
          json je = json::object();
          if(e)
            {
              je["t"] = e->t;
              switch(iter.first)
                {
                case KEYEVENT_SIGNAL_ADD:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_ADD>();
                    je["id"]  = e2->id;
                    je["pos"] = to_string(e2->pos);
                    je["pen"] = penToJSON(e2->pen);
                  } break;
                case KEYEVENT_SIGNAL_REMOVE:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_REMOVE>();
                    je["id"]  = e2->id;
                  } break;
                case KEYEVENT_SIGNAL_MOVE:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_MOVE>();
                    je["id"]  = e2->id;
                    je["pos"] = to_string(e2->pos);
                  } break;
                case KEYEVENT_SIGNAL_PEN:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_PEN>();
                    je["id"]  = e2->id;
                    je["pen"] = penToJSON(e2->pen);
                  } break;
                case KEYEVENT_MATERIAL:
                  {
                    auto e2 = e->sub<KEYEVENT_MATERIAL>();
                    je["pos"] = to_string(e2->pos);
                    // je["pen"] = penToJSON(e2->pen);
                  } break;
                case KEYEVENT_VIEW2D:
                  {
                    auto e2 = e->sub<KEYEVENT_VIEW2D>();
                    je["view"] = view2DToJSON(e2->view);
                  } break;
                case KEYEVENT_VIEW3D:
                  {
                    auto e2 = e->sub<KEYEVENT_VIEW3D>();
                    je["view"] = view3DToJSON(e2->view);
                  } break;
                case KEYEVENT_SETTING:
                  {
                    auto e2 = e->sub<KEYEVENT_SETTING>();
                    je["status"] = "UNIMPLEMENTED";
                  } break;
                  break;
                case KEYEVENT_INVALID: je["status"] = "INVALID"; break;
                default:               je["status"] = "UNKNOWN"; break;
                }
            }
          else { je["status"] = "NULL"; }
          jse.push_back(je);
        }
      events[ss.str()] = jse;
    }

  js["events"] = events;
  
  js["initView2D"] = view2DToJSON(mView2DInit);
  js["initView3D"] = view3DToJSON(mView3DInit);
  // js["initSettings"]  = penToJSON(mSettingsInit);
  return js;
}

bool KeyFrameWidget::fromJSON(const json &js)
{
  using VT3 = float3;
  bool success = true;
  // if(js.contains("initSources")
  //   {
  //     json iSources = js["initSources"];;
  //     for(const auto &s : iSources)
  //       {
  //         from_string(s["pos"].get<std::string>());
  //         s["pen"] = penToJSON(iter.second.pen);
  //         iSources[iter.first] = s;
  //         mSourcesInit.emplace(id, );
  //       }
  //   } js["initSources"]   = iSources;
  // json iPlaced = json::object();
  // for(int i = 0; i < mPlacedInit.size(); i++)
  //   {
  //     const auto &m = mPlacedInit[i];
  //     std::string id = std::to_string(i);
  //     iPlaced[id] = json::object();
  //     iPlaced[id]["pos"] = to_string(m.pos);
  //     iPlaced[id]["pen"] = penToJSON(m.pen);
  //   } js["initMatPlaced"] = iPlaced;  
  
  if(js.contains("events"))
    {
      json events = js["events"];
      for(auto iter : mEvents)
        {
          std::stringstream ss; ss << iter.first;
          json jse; if(jse.contains(ss.str())) { jse = events[ss.str()]; }
          
          for(auto e : iter.second)
            {
              json je = json::object();
              if(e)
                {
                  if(je.contains("t")) { e->t = je["t"].get<double>(); }
                  switch(iter.first)
                    {
                    case KEYEVENT_SIGNAL_ADD:
                      {
                        auto e2 = e->sub<KEYEVENT_SIGNAL_ADD>();
                        if(je.contains("id"))  { e2->id  = je["id"].get<std::string>(); }
                        if(je.contains("pos")) { e2->pos = from_string<VT3>(je["pos"].get<std::string>()); }
                        penFromJSON(je["pen"], e2->pen);
                      } break;
                    case KEYEVENT_SIGNAL_REMOVE:
                      {
                        auto e2 = e->sub<KEYEVENT_SIGNAL_REMOVE>();
                        if(je.contains("id")) { e2->id = je["id"].get<std::string>(); }
                      } break;
                    case KEYEVENT_SIGNAL_MOVE:
                      {
                        auto e2 = e->sub<KEYEVENT_SIGNAL_MOVE>();
                        if(je.contains("id"))  { e2->id  = je["id"].get<std::string>(); }
                        if(je.contains("pos")) { e2->pos = from_string<VT3>(je["pos"].get<std::string>()); }
                      } break;
                    case KEYEVENT_SIGNAL_PEN:
                      {
                        auto e2 = e->sub<KEYEVENT_SIGNAL_PEN>();
                        if(je.contains("id")) { e2->id = je["id"].get<std::string>(); }
                        penFromJSON(je["pen"], e2->pen);
                      } break;
                    case KEYEVENT_MATERIAL:
                      {
                        auto e2 = e->sub<KEYEVENT_MATERIAL>();
                        if(je.contains("pos")) { e2->pos = from_string<VT3>(je["pos"].get<std::string>()); }
                        // penFromJSON(je["pen"], e2->pen);
                      } break;
                    case KEYEVENT_VIEW2D:
                      {
                        auto e2 = e->sub<KEYEVENT_VIEW2D>();
                        if(je.contains("view")) { view2DFromJSON(je["view"], e2->view); }
                      } break;
                    case KEYEVENT_VIEW3D:
                      {
                        auto e2 = e->sub<KEYEVENT_VIEW3D>();
                        if(je.contains("view")) { view3DFromJSON(je["view"], e2->view); }
                      } break;
                    case KEYEVENT_SETTING:
                      {
                        // auto e2 = e->sub<KEYEVENT_SETTING>();
                        // je["status"] = "UNIMPLEMENTED";
                      } break;
                    case KEYEVENT_INVALID: std::cout << "====> WARNING: INVALID EVENT (fromJSON())\n"; break;
                    default:               std::cout << "====> WARNING: UNKNOWN EVENT (fromJSON())\n"; break;
                    }
                }
              else { std::cout << "====> WARNING: NULL EVENT (fromJSON())\n"; }
            }
        }
    }
  if(js.contains("initView2D")) { success |= view2DFromJSON(js["initView2D"], mView2DInit); } else { success = false; }
  if(js.contains("initView3D")) { success |= view3DFromJSON(js["initView3D"], mView3DInit); } else { success = false; }
  if(js.contains("initView2D")) { success |= view2DFromJSON(js["initView2D"], mView2DInit); } else { success = false; }
  if(js.contains("initView3D")) { success |= view3DFromJSON(js["initView3D"], mView3DInit); } else { success = false; }
  // //js["initSettings"]  = penToJSON(mSettingsInit);
  
  return success;
}



std::vector<std::string>  KeyFrameWidget::orderedEvents() const
{
  int eTotal = 0;
  for(const auto &iter : mEvents) { for(const auto &e : iter.second) { eTotal += iter.second.size(); } }

  std::vector<std::string> strs; strs.reserve(eTotal);
  std::unordered_map<std::string, KeyEventBase*> events; events.reserve(eTotal);

  for(const auto &iter : mEvents)
    {
      std::stringstream ss; 
      for(const auto &e : iter.second)
        {
          if(e)
            {
              ss << "[" << std::setw(24) << iter.first << " : " // type/timestamp
                 << std::fixed << std::setprecision(4) << std::setw(12) << e->t << "] --> ";
              
              switch(iter.first)
                {
                case KEYEVENT_SIGNAL_ADD:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_ADD>();
                    ss << "  id=" << e2->id;
                    ss << "  pos=" << to_string(e2->pos);
                  } break;
                case KEYEVENT_SIGNAL_REMOVE:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_REMOVE>();
                    ss << "  id=" << e2->id;
                  } break;
                case KEYEVENT_SIGNAL_MOVE:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_MOVE>();
                    ss << "  id=" << e2->id;
                    ss << "  pos=" << to_string(e2->pos);
                  } break;
                case KEYEVENT_SIGNAL_PEN:
                  {
                    auto e2 = e->sub<KEYEVENT_SIGNAL_PEN>();
                    ss << "  id=" << e2->id;
                  } break;
                case KEYEVENT_MATERIAL:
                  {
                    auto e2 = e->sub<KEYEVENT_MATERIAL>();
                    ss << "  pos=" << to_string(e2->pos);
                  } break;
                case KEYEVENT_VIEW2D:
                  {
                    auto e2 = e->sub<KEYEVENT_VIEW2D>();
                    ss << "  [VIEW 2D]";
                  } break;
                case KEYEVENT_VIEW3D:
                  {
                    auto e2 = e->sub<KEYEVENT_VIEW3D>();
                    ss << "  [VIEW 3D]";
                  } break;
                case KEYEVENT_SETTING:
                  {
                    auto e2 = e->sub<KEYEVENT_SETTING>();
                    ss << "  <UNIMPLEMENTED>";
                  } break;
                  break;
                case KEYEVENT_INVALID: ss << "  <INVALID>"; break;
                default:               ss << "  <UNKNOWN>"; break;
                }
            }
          else { ss << "====> WARNING: <NULL>"; }
          ss << "\n";
          strs.push_back(ss.str());
          events.emplace(ss.str(), e);
        }
    }

  std::sort(strs.begin(), strs.end(), [&](const std::string &s1, const std::string &s2) -> bool
                                      { return(events[s1]->t > events[s2]->t); });
  return strs;
}
















void KeyFrameWidget::addEvent(KeyEventBase *e, bool force)
{
  if((mRecording || force) && e)
    {
      std::cout << "====> NEW KEYFRAME EVENT [" << e->t << "] ";
      if(e->t < 0.0) { std::cout << "========> WARNING: skipping event with invalid timestamp\n"; delete e; return; }
      
      if(mOverwrite)
        { // delete overlapping events of the same kind (TODO: differentiate e.g. multiple sources on same frame)
          // auto tEvents = mEvents[e->type()];
          // for(int i = 0; i < tEvents.size(); i++)
          //   {
          //     auto &ev = tEvents[i];
          //     if(ev->start == e->end && (ev->type() != KEYEVENT_SIGNAL ||
          //                                static_cast<KeyEvent<KEYEVENT_SIGNAL>*>(ev)->id == static_cast<KeyEvent<KEYEVENT_SIGNAL>*>(e)->id))
          //       { tEvents.erase(tEvents.begin() + i--); break; }
          //   }
        }

      switch(e->type())
        {
        case KEYEVENT_INVALID:
          std::cout << "\n========> INVALID KEYFRAME\n";
          break;
        case KEYEVENT_SIGNAL_ADD:
          {
            auto e2 = e->sub<KEYEVENT_SIGNAL_ADD>();
            if(!e2) { return; }
            std::string id = e2->id;
            std::cout << "(" << e->type() << ") --> '" << id << "' / " << e2->pos << "\n";

            // bool eStart = false;
            // bool eEnd   = false;
            // if(e->start >= 0.0 && e->end < 0.0)       { std::cout << "========> Source " << id << " [START]\n"; eStart = true; }
            // else if(e->start < 0.0 && e->end >= 0.0)  { std::cout << "========> Source " << id << " [END]\n";   eEnd   = true; }
            // else if(e->start >= 0.0 && e->end >= 0.0) { std::cout << "========> Source " << id << " [FULL]\n"; }
            // else                                      { std::cout << "========> Source " << id << " [UNKNOWN/INVALID(?)]\n";  }

            // check if overlapping
            auto &addEvents = mEvents.at(KEYEVENT_SIGNAL_ADD);
            auto &remEvents = mEvents.at(KEYEVENT_SIGNAL_REMOVE);
            KeyEvent<KEYEVENT_SIGNAL_ADD>    *prev = nullptr; // most recent add event with same id
            KeyEvent<KEYEVENT_SIGNAL_REMOVE> *next = nullptr; // next closest remove event with same id
            int iPrev = -1; int iNext = -1;
            
            // find most recent previous add event
            for(int i = 0; i < addEvents.size(); i++)
              {
                KeyEventBase *sb = addEvents[i];
                auto *s2 = sb->sub<KEYEVENT_SIGNAL_ADD>();
                if(sb && s2 && s2->id == e2->id)
                  {
                    if(s2->t <= e->t) { prev  = s2; }
                    else              { iPrev = i; break; }
                  }
              }
            if(prev)
              { // find next remove after previous add
                for(int i = remEvents.size()-1; i >= 0; i--)
                  {
                    KeyEventBase *sb = remEvents[i];
                    auto *s2 = sb->sub<KEYEVENT_SIGNAL_REMOVE>();
                    if(sb && s2 && s2->id == e2->id)
                      {
                        if(s2->t >= prev->t) { next  = s2; }
                        else                 { iNext = i; break; }
                      }
                  }
              }
            if(prev && (!next || (next->t > e->t)))
              { // add additional remove event to splice overlapped note
                std::cout << "====> OVERLAP --> inserting KEYEVENT_SIGNAL_ADD at t=" << e->t + 0.0001 << "\n";
                remEvents.insert(remEvents.begin()+iNext, new KeyEvent<KEYEVENT_SIGNAL_REMOVE>(e->t - 0.0001, e2->id));
              }




            
          } break;
        case KEYEVENT_SIGNAL_REMOVE:
          {
            auto e2 = e->sub<KEYEVENT_SIGNAL_REMOVE>();
            std::cout << "(" << e->type() << ") --> '" << e2->id << "'\n";

            // check if overlapping
            auto &addEvents = mEvents.at(KEYEVENT_SIGNAL_ADD);
            auto &remEvents = mEvents.at(KEYEVENT_SIGNAL_REMOVE);
            KeyEvent<KEYEVENT_SIGNAL_ADD>    *prev = nullptr; // next closest add event with same id
            KeyEvent<KEYEVENT_SIGNAL_REMOVE> *next = nullptr; // most recent remove event with same id
            int iPrev = -1; int iNext = -1;
            
            // find next remove
            for(int i = 0; i < remEvents.size(); i++)
             { 
                KeyEventBase *sb = remEvents[i];
                auto *s2 = sb->sub<KEYEVENT_SIGNAL_REMOVE>();
                if(sb && s2 && s2->id == e2->id)
                  {
                    if(s2->t >= e->t) { next  = s2; }
                    else              { iNext = i; break; }
                  }
              }
            if(next)
              { // find most recent previous add
                for(int i = addEvents.size()-1; i >= 0; i--)
                  {
                    KeyEventBase *sb = addEvents[i];
                    auto *s2 = sb->sub<KEYEVENT_SIGNAL_ADD>();
                    if(sb && s2 && s2->id == e2->id)
                      {
                        if(s2->t <= next->t) { prev  = s2; }
                        else                 { iPrev = i; break; }
                      }
                  }
              }
            
            if(prev && (!next || (next->t > e->t)) && iNext >= 0)
              { // add additional remove event to splice overlapped note
                std::cout << "====> OVERLAP --> inserting KEYEVENT_SIGNAL_ADD at t=" << e->t + 0.0001 << "\n";
                remEvents.insert(addEvents.begin()+iNext, new KeyEvent<KEYEVENT_SIGNAL_ADD>(e->t + 0.0001, e2->id, prev->pos, prev->pen));
              }


            
          } break;
        case KEYEVENT_SIGNAL_MOVE:
          {
            auto e2 = e->sub<KEYEVENT_SIGNAL_MOVE>();
            std::cout << "(" << e->type() << ") --> '" << e2->id << "' / " << e2->pos << "\n";
            
            // KeyEvent<KEYEVENT_SIGNAL_MOVE> *found = nullptr;
            // for(auto s : mEvents[KEYEVENT_SIGNAL_MOVE])
            //   {
            //     KeyEvent<KEYEVENT_SIGNAL_MOVE> *s2 = s->sub<KEYEVENT_SIGNAL_MOVE>();
            //     if(s && s2 && s2->id == e2->id) { found = s2; break; }
            //   }

            // if(!found)
            //   {
            //     std::cout << "========> WARNING: Source doesn't exist -- skipping\n";
            //     delete e; e = nullptr;
            //   }
          } break;
        case KEYEVENT_SIGNAL_PEN:
          {
            auto e2 = e->sub<KEYEVENT_SIGNAL_PEN>();
            std::cout << "(" << e->type() << ") --> '" << e2->id << "\n";
            
            // KeyEvent<KEYEVENT_SIGNAL_PEN> *found = nullptr;
            // for(auto s : mEvents[KEYEVENT_SIGNAL_PEN])
            //   {
            //     KeyEvent<KEYEVENT_SIGNAL_PEN> *s2 = s->sub<KEYEVENT_SIGNAL_PEN>();
            //     if(s && s2 && s2->id == e2->id) { found = s2; break; }
            //   }
            // if(!found)
            //   {
            //     std::cout << "========> WARNING: Source doesn't exist -- skipping\n";
            //     delete e; e = nullptr;
            //   }
          } break;
        case KEYEVENT_MATERIAL:
          {
            auto e2 = e->sub<KEYEVENT_MATERIAL>();
            std::cout << "(" << e->type() << ") --> " << e2->pos << "\n";
          } break;
        case KEYEVENT_VIEW2D:
          {
            auto e2 = e->sub<KEYEVENT_VIEW2D>();
            std::cout << "(" << e->type() << ") --> " << e2->view << "\n";
          } break;
        case KEYEVENT_VIEW3D:
          {
            auto e2 = e->sub<KEYEVENT_VIEW3D>();
            std::cout << "(" << e->type() << ") --> pos = " << e2->view.pos << " / dir = " << e2->view.dir << "\n";
          } break;
        case KEYEVENT_SETTING:
          {
            auto e2 = e->sub<KEYEVENT_SETTING>();
            std::cout << "(" << e->type() << ") --> " << e2->id << " = " << std::any_cast<float3>(e2->value) << " (" <<  e2->value.type().name()<< ")\n";
          } break;
        default:
          std::cout << "\n========>  KEYFRAME UNKNOWN\n";
          break;
        }
      if(e)
        {
          mEvents[e->type()].push_back(e);
          mEventWidgets.emplace(e, new KeyEventWidget(e));
        }
    }
}

bool KeyFrameWidget::processEvents(double t0, double t1)
{
  if(!mApply) { return false; }
  // mPlaced.clear();

  // remove invalid events  
  for(auto &iter : mEvents)
    for(int i = 0; i < iter.second.size(); i++)
      {
        KeyEventBase *e = iter.second[i];
        if(!e)
          {
            std::cout << "====> WARNING: Null key event while processing (" << (KeyEventType)i << ") removing...\n";
            iter.second.erase(iter.second.begin()+i--);
          }
      }

  bool changed = false;
  for(auto &iter : mEvents)
    for(auto &e : iter.second)
      {
        if(!e)
          {
            std::cout << "====> WARNING: Null event (?) --> KeyFrameWidget::processEvents()\n";
          }
        else if(e->t >= t0 && e->t < t1)
          {
            std::cout << "====> APPLYING KEYFRAME EVENT [" << e->t << "] ";
            switch(e->type())
              {
              case KEYEVENT_INVALID:
                std::cout << "\n========> INVALID KEYFRAME\n";
                break;
                
              case KEYEVENT_SIGNAL_ADD:
                {
                  auto e2 = e->sub<KEYEVENT_SIGNAL_ADD>();
                  std::cout << "(" << e->type() << ") --> '" << e2->id << "' / " << e2->pos << "\n";
                  auto iter = mSources.find(e2->id);
                  if(iter != mSources.end())
                    {
                      std::cout << "========> WARNING: Source already exists!\n";
                      iter->second.pos = e2->pos;
                      iter->second.pen = e2->pen;
                    }
                  else
                    {
                      mSources.emplace(e2->id, SignalSource{ e2->pos, e2->pen });
                      std::cout << "========> Source active\n";
                    }
                } break;
                
              case KEYEVENT_SIGNAL_REMOVE:
                {
                  auto e2 = e->sub<KEYEVENT_SIGNAL_REMOVE>();
                  std::cout << "(" << e->type() << ") --> '" << e2->id << "'\n";
                  auto iter = mSources.find(e2->id);
                  if(iter == mSources.end())
                    { std::cout << "========> WARNING: Source doesn't exist -- skipping\n"; }
                  else
                    {
                      mSources.erase(e2->id);
                      std::cout << "========> Deleted source\n";
                    }
                } break;
                
              case KEYEVENT_SIGNAL_MOVE:
                {
                  auto e2 = e->sub<KEYEVENT_SIGNAL_MOVE>();
                  std::cout << "(" << e->type() << ") --> '" << e2->id << "' / " << e2->pos << "\n";
                  auto iter = mSources.find(e2->id);
                  if(iter == mSources.end())
                    {
                      std::cout << "========> WARNING: Source doesn't exist -- adding (" << e2->pos << ")\n";
                      mSources.emplace(e2->id, SignalSource{ e2->pos, *mSigPen });
                    }
                  else
                    {
                      iter->second.pos = e2->pos;
                      std::cout << "========> Moved source (" << iter->second.pos << ")\n";
                    }
                } break;
                
              case KEYEVENT_SIGNAL_PEN:
                {
                  auto e2 = e->sub<KEYEVENT_SIGNAL_PEN>();
                  std::cout << "(" << e->type() << ") --> '" << e2->id << "\n";
                  auto iter = mSources.find(e2->id);
                  if(iter == mSources.end())
                    {
                      std::cout << "========> WARNING: Source doesn't exist -- adding (" << float3{0,0,0} << ")\n";
                      mSources.emplace(e2->id, SignalSource{ float3{0,0,0}, e2->pen });
                    }
                  else
                    {
                      iter->second.pen = e2->pen;
                      std::cout << "========> Set source pen\n";
                    }
                } break;
                
              case KEYEVENT_MATERIAL:
                {
                  auto e2 = e->sub<KEYEVENT_MATERIAL>();
                  std::cout << "(" << e->type() << ") --> " << e2->pos << "\n";
                  mPlaced.push_back(MaterialPlaced{ e2->pos, e2->pen });
                  std::cout << "========> Placed material\n";
                } break;
                
              case KEYEVENT_VIEW2D:
                {
                  auto e2 = e->sub<KEYEVENT_VIEW2D>();
                  std::cout << "(" << e->type() << ") --> " << e2->view << "\n";
                  mView2D = e2->view;
                  if(view2DCallback) { view2DCallback(mView2D); }
                  std::cout << "========> Set 2D view\n";
                } break;
                
              case KEYEVENT_VIEW3D:
                {
                  auto e2 = e->sub<KEYEVENT_VIEW3D>();
                  std::cout << "(" << e->type() << ") --> pos = " << e2->view.pos << " / dir = " << e2->view.dir << "\n";
                  mView3D = e2->view;
                  if(view3DCallback) { view3DCallback(mView3D); }
                  std::cout << "========> Set 3D view\n";
                } break;
                
              case KEYEVENT_SETTING:
                {
                  auto e2 = e->sub<KEYEVENT_SETTING>();
                  std::cout << "(" << e->type() << ") --> " << e2->id << " = " << std::any_cast<float3>(e2->value)
                            << " (" <<  e2->value.type().name()<< ")\n";
                  mSettings[e2->id] = e2->value;
                  std::cout << "========> Updated settings\n";
                } break;
                
              default:
                std::cout << "\n========>  KEYFRAME UNKNOWN\n";
                break;
              }
          }
      }
  return changed;
}

bool KeyFrameWidget::nextFrame(double dt)
{
  mCursor += dt;
  //mMaxTime = std::max(mMaxTime, mCursor);
  if(mFollowCursor) { mUpdateScroll += dt; }
  return processEvents(mCursor-dt, mCursor);
}

void KeyFrameWidget::reset()
{
  mCursor = 0;
  mSources.clear();
  mPlaced.clear();
  mView2D = mView2DInit;
  mView3D = mView3DInit;
  if(view2DCallback) { view2DCallback(mView2D); }
  if(view3DCallback) { view3DCallback(mView3D); }
  mReset = true;
}

void KeyFrameWidget::clear()
{
  reset();
  for(auto &iter : mEvents) { for(auto e : iter.second) { if(e) { delete e; } } iter.second.clear(); }
  for(auto &iter : mEventWidgets) { if(iter.second) { delete iter.second; } } mEventWidgets.clear();
}

void KeyFrameWidget::draw() //const Vec2f &size)
{
  ImGuiIO    &io    = ImGui::GetIO();
  ImGuiStyle &style = ImGui::GetStyle();

  Vec2f size = ImGui::GetContentRegionMax();
  
  ImGui::BeginGroup(); // first column
  {
    ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Frame:");
    ImGui::Separator();
    ImGui::Checkbox("Record##kf",       &mRecording   );
    ImGui::Checkbox("Apply##kf",        &mApply       );
    ImGui::Checkbox("Folow Cursor##kf", &mFollowCursor);
    
    if(ImGui::Button("Clear##kb")) { clear(); }
    if(ImGui::Button("Save JSON"))
      {
        std::ofstream f("./events.json");
        f << std::setw(2) << toJSON();
      }
    if(ImGui::Button("Load JSON"))
      {
        std::ifstream f("./events.json");
        json js; f >> js;
        fromJSON(js);
      }
  }
  if(ImGui::Button("Save List"))
    {
      std::ofstream f("./events.ord");
      auto lines = orderedEvents();
      for(const auto &l : lines) { f << l; }
    }
  ImGui::EndGroup();
  ImGui::SameLine();

  // scrollable inner timeline
  
  Vec2f childSize = (size - Vec2f((ImGui::GetItemRectMax().x-ImGui::GetItemRectMin().x) + style.ItemSpacing.x, 0.0f));
  Vec2f p0 = ImGui::GetCursorScreenPos();

  ImGuiWindowFlags wFlags = ImGuiWindowFlags_HorizontalScrollbar;
  ImGui::PushStyleColor(ImGuiCol_ChildBg, Vec4f(0.24f, 0.24f, 0.24f, 1.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, Vec2f(0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, Vec2f(4, 4));
  if(ImGui::BeginChild("kfChild", childSize, true, wFlags))
    {
      ImGui::PopStyleVar(2);

      Vec2f tSize = Vec2f(mScale*KEYFRAME_SECOND_MINSIZE.x, KEYFRAME_SECOND_MINSIZE.y); // scaled width of 1 second
      Vec2f fSize = tSize/mFps; // scaled width of 1 frame
      
      if(mUpdateScroll != 0)
        {
          if(ImGui::GetScrollMaxX() > 0.0f) { ImGui::SetScrollFromPosX(mCursor*tSize.x); }
          mUpdateScroll = 0;
        }
      if(mReset) { ImGui::SetScrollX(0); mReset = false; }
    
      ImDrawList *drawList = ImGui::GetWindowDrawList();
      Vec2f       p1       = ImGui::GetCursorScreenPos();
      float       scrollX  = ImGui::GetScrollX();
      Vec2f       mp       = ImGui::GetMousePos();

      
      // int numSeconds = mMaxTime;
      // std::stringstream ss; ss << mMaxTime;
      // Vec2f maxSecondSize = std::max(Vec2f(ImGui::CalcTextSize(ss.str().c_str())) + Vec2f(style.FramePadding)*2.0f,
      //                                KEYFRAME_SECOND_MINSIZE);
  

      double tOffset = scrollX/tSize.x;
      double numSecs = childSize.x/tSize.x; // total number of seconds visible
      Vec2d  tRange  = Vec2d(tOffset, tOffset + numSecs);
      
      double mtime   = (scrollX + (mp.x - p0.x))/tSize.x; // time at mouse position
      int    mframe  = (scrollX + (mp.x - p0.x))/fSize.x; // frame at mouse position

      bool eventHovered = false;
      KeyEventBase *hoveredEvent = nullptr;
      ImGui::BeginGroup();
      {
        // draw column header (frame numbers) (TODO: better time axis labelling)
        ImGui::BeginGroup();
        {
          for(int t = std::floor(tRange.x); t <= std::ceil(tRange.y); t++)
            {
              // double t2 = tOffset+t-fmod(tOffset, 1.0);
              double tp = t*tSize.x;
              // frame number label
              std::stringstream ss; ss << std::round(t);
              std::string label = ss.str();
              float lW = ImGui::CalcTextSize(label.c_str()).x;
              ImGui::SetCursorPos(Vec2f(tp + (tSize.x - lW)/2.0f, 0.0f));
              ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted(label.c_str());
              // if(t < numSeconds) 
                { ImGui::SameLine(); }

              if(tp >= -scrollX && tp <= childSize.x+scrollX)
                { // left column separator
                  drawList->AddLine(p1+Vec2f(tp, 0.0f), p1+Vec2f(tp, childSize.y), ImColor(Vec4f(1, 1, 1, 0.5f)), 1.0f);
                  if(t == numSecs) // final right separator
                    { drawList->AddLine(p1+Vec2f(tp+tSize.x, 0.0f), p1+Vec2f(tp+tSize.x, childSize.y), ImColor(Vec4f(1, 1, 1, 0.5f)), 1.0f); }
                }
            }
          // header separator
          drawList->AddLine(p1+Vec2f(scrollX, tSize.y),      p1+Vec2f(scrollX+childSize.x, tSize.y), ImColor(Vec4f(1, 1, 1, 0.5f)), 1.0f);
          // draw cursor
          drawList->AddLine(p1+Vec2f(mCursor*tSize.x, 0.0f), p1+Vec2f(mCursor*tSize.x, childSize.y), ImColor(Vec4f(1, 0, 0, 0.5f)), 3.0f);
        }
        ImGui::Dummy(Vec2f((numSecs+2.0f)*tSize.x, 1.0f)); // fill out full size
        ImGui::EndGroup();
      
        Vec2i headerSize = Vec2f(ImGui::GetItemRectMax())-ImGui::GetItemRectMin();
        float keyH = (ImGui::GetContentRegionMax().y - headerSize.y)/(float)KEYEVENT_COUNT;

        // draw keyed events
        for(auto &iter : mEvents)
          {
            if(iter.first == KEYEVENT_SIGNAL_REMOVE) { continue; } // remove events handled by KEYEVENT_SIGNAL_ADD
            for(int i = 0; i < iter.second.size(); i++)
              {
                //auto e = iter.second[i];
                KeyEventBase *e  = iter.second[i];             if(!e)  { continue; }
                KeyEventBase *e2 = getEventEnd(iter.first, i); if(!e2) { continue; }

                double start  = e->t;
                double end    = e2->t;
                double sStart = start*tSize.x;
                double sEnd   = end*tSize.x;
                double mpos   = (mp.x - p1.x)/tSize.x;

                ImGui::SetCursorScreenPos(p1 + Vec2f(sStart, headerSize.y + keyH*(float)iter.first));
                std::string typeStr = to_string(e->type());

                const Vec4f moveCol = Vec4f(1.0f, 0.3f, 0.3f, 1.0f);
                if(mLastHovered == e) // ImGui::PushStyleColor(ImGuiCol_Button, Vec4f(1.0f, 0.3f, 0.3f, 1.0f));
                  {
                    ImGui::PushStyleColor(ImGuiCol_Button,        moveCol);
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, moveCol);
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  moveCol);
                  }
                ImGui::Button(to_string(i).c_str(), Vec2f(sEnd-sStart, keyH));
                if(mLastHovered == e) { ImGui::PopStyleColor(3); }
              
                bool hovered = ImGui::IsItemHovered();
                eventHovered |= hovered;
                //if(!eventHovered && !mContextOpen) { mLastHovered = nullptr; }
                
                if(hovered || (mLastHovered == e && !mContextOpen && ImGui::IsMouseDragging(ImGuiMouseButton_Left)))
                  {
                    hoveredEvent = e;
                    mLastHovered = e;
                    ImGui::SetTooltip("t = %f / Frame %d \n\nEvent(%s): %s (%f -> %f)", mtime, mframe, typeStr.c_str(),
                                      (e->type() == KEYEVENT_SIGNAL_ADD ? static_cast<KeyEvent<KEYEVENT_SIGNAL_ADD>*>(e)->id.c_str() : ""), start, end);
                  }

                auto w = mEventWidgets[e];              
                if(hovered && w->clickPos < 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                  {
                    std::cout << " EVENT CLICKED: " << typeStr << " --> mpos=" << mpos << " / start=" << start << ", end=" << end << "\n";
                    w->clickPos  = mpos;
                    w->lastStart = start;
                    w->length    = end-start;
                  }
                else if(w->clickPos >= 0.0f && ImGui::IsMouseReleased(ImGuiMouseButton_Left))
                  {
                    std::cout << " EVENT RELEASED: " << typeStr << "\n";
                    w->clickPos  = -1.0;
                    w->lastStart = -1.0;
                    w->length    = -1.0;
                  }
              
                if(w->clickPos >= 0 && ImGui::IsMouseDragging(ImGuiMouseButton_Left))
                  {
                    Vec2f  dmp = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left); ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
                    double dt = (mpos - w->clickPos);///tSize.x;
                    double newStart = w->lastStart + dt;
                    if(newStart != start)
                      { std::cout << "MOVING EVENT: (" << typeStr << ")  " << w->lastStart << " / " << dt << " / mpos=" << mpos << " --> " << newStart << "\n"; }
                    e->t  = newStart;
                    e2->t = newStart + w->length;
                  }

                if(hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Middle) &&
                   !ImGui::IsMouseDown(ImGuiMouseButton_Left) && !ImGui::IsMouseDown(ImGuiMouseButton_Right))
                  { // delete event (test)
                    std::cout << "DELETING EVENT: " << typeStr << "\n";
                    iter.second.erase(iter.second.begin() + i);
                    mEventWidgets.erase(e);
                    delete e;
                  }
              }
            // event type divider
            float ety = headerSize.y + keyH*((float)iter.first);
            drawList->AddLine(p1+Vec2f(scrollX, ety), p1+Vec2f(scrollX+childSize.x, ety), ImColor(Vec4f(1, 1, 1, 0.5f)), 1.0f);
          }
      }
      ImGui::EndGroup();

      // event context menu
      bool bgHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) && !eventHovered;
      if(ImGui::BeginPopup("kfeContext"))
        {
          if(mLastHovered)
            {
              std::stringstream ss; ss << "Event  (" << mLastHovered->type() << ") --> " << mLastHovered->t << "\n";
              // if(ImGui::BeginMenu(ss.str().c_str()))
              //   {
              //     if(ImGui::MenuItem("TEST!!!")) // TODO: Popup to adjust placement/etc.
              //       { } //addEvent(new KeyEvent<KEYEVENT_SIGNAL>(mCursor, mCursor+1.0, "Added Signal", float3{0.0f, 0.0f, 0.0f}, *mSigPen), true); }
              //     ImGui::EndMenu();
              //   }

              //if(mLastHovered->type() == KEYEVENT_SIGNAL_ADD)
                {
                  KeyEvent<KEYEVENT_SIGNAL_ADD> *last2 = mLastHovered->sub<KEYEVENT_SIGNAL_ADD>();
                  
                  SignalSource *sig = &mSources[last2->id];
                  // for(int i = 0; i < mSources.size(); i++)
                  //   {
                  //     if(mSources[i].id == last2->id){ sig = &mSources[i]; } ]


                  if(sig)
                    {
                      ImGui::InputDouble("start##ev", &sig->start);
                      ImGui::InputDouble("end##ev",   &sig->end);
                    }
                  else
                    {
                      ImGui::InputDouble("start##ev", &mLastHovered->t);
                      ImGui::InputDouble("end##ev",   &mLastHovered->t);
                    }
                }
              
            }
          ImGui::EndPopup();
        }

      mContextOpen = false;
      
      if(eventHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) { ImGui::OpenPopup("kfeContext"); }
      // background context menu
      if(ImGui::BeginPopup("kfContext"))
        {
          if(ImGui::BeginMenu("Edit"))
            {
              if(ImGui::MenuItem("Add Signal"))
                {
                  addEvent(new KeyEvent<KEYEVENT_SIGNAL_ADD>   (mCursor,     "Signal1", float3{0.0f, 0.0f, 0.0f}, *mSigPen), true);
                  addEvent(new KeyEvent<KEYEVENT_SIGNAL_REMOVE>(mCursor+1.0, "Signal1"), true);
                }
              ImGui::EndMenu();
            }
          mContextOpen = true;
          ImGui::EndPopup();
        }
      if(bgHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) { ImGui::OpenPopup("kfContext"); }

        
      // popups
      if(bgHovered)
        {
          // Scale timeline horizontally -- Ctrl+Alt+Scroll
          if(std::abs(io.MouseWheel) > 0.0f && io.KeyCtrl && io.KeyAlt)
            {
              double mf1 = (mp.x - p0.x + scrollX)/tSize.x;
              
              double vel = 1.11; //(io.KeyAlt ? 1.36 : (io.KeyShift ? 1.0011 : 1.0055)); // scale velocity
              mScale *= (io.MouseWheel > 0.0f ? vel : 1.0/vel);

              // correct scrolling so mouse position stays the same
              double newW = mScale * tRange.x;
              double mf2 = (mp.x - p0.x + scrollX)/newW;
              float newScroll = scrollX + (mf1 - mf2);
              // if(newScroll > ImGui::GetScrollMaxX()) { ImGui::SetCursorPos(Vec2f(newScroll, ImGui::GetCursorPos().y)); }
              ImGui::SetScrollX(newScroll);
            }

          if(!eventHovered)
            {
              if(ImGui::IsMouseDown(ImGuiMouseButton_Left) && io.KeyCtrl)
                {
                  mCursor = mtime;
                  Vec2i cr = ImGui::GetContentRegionMax();
                  ImGui::SetTooltip("t = %f / Frame %d \n (cursor: %f) \n content region: %d x %d", mtime, mframe, mCursor, cr.x, cr.y);
                }
              else
                { ImGui::SetTooltip("t = %f / Frame %d \n (cursor: %f)", mtime, mframe, mCursor); }
            }
        }
    
    }
  ImGui::EndChild();
  ImGui::PopStyleColor();
}
