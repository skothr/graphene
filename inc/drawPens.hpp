#ifndef DRAW_PENS_H
#define DRAW_PENS_H

#include "vector-operators.h"
// #include "raytrace.h"
#include "material.h"
//#include "drawPens.hpp"

#include <array>
#include <vector>


// defines a "pen" used to add to a field
template<typename T>
struct Pen
{
  using VT3 = typename DimType<T, 3>::VEC_T;
  bool active    = true;  // pen always active
  bool cellAlign = false; // snap offset to center of cell
  bool square    = false; // draw with square pen
  bool radial    = false; // multiply by normal (from pen center)
  bool speed     = true;  // scale by mouse speed
  
  int depth      = 0;                     // depth of pen center from view surface
  VT3 radius0    = VT3{10.0, 10.0, 10.0}; // base pen size in fluid cells
  VT3 radius1    = VT3{ 0.0,  0.0,  0.0}; // if x,y,z > 0, pen shape will be the intersection of spheres 0/1
  VT3 rDist      = VT3{ 0.0,  0.0,  0.0}; // positional difference between intersecting spheres
  T   mult       = 1.0;                   // multiplier (amplitude if signal)
  T   sizeMult   = 1.0;                   // multiplier (pen size)
  VT3 xyzMult    = VT3{1.0, 1.0, 1.0};    // multiplier (pen size, each dimension)
  T   speedMult  = 1.0;                   // multiplier for mouse speed

  T startTime  = -1.0; // time of initial mouse click (< 0 if inactive)
  T mouseSpeed =  0.0; // current mouse speed
  virtual ~Pen() = default;

  virtual bool isSignal() const { return false; }
  virtual bool isMaterial() const { return false; }

  bool operator==(const Pen &other) const
  {
    return (active == other.active && cellAlign == other.cellAlign && square == other.square && radial == other.radial && speed == other.speed &&
            depth == other.depth && radius0 == other.radius0 && radius1 == other.radius1 && rDist == other.rDist && mult == other.mult &&
            sizeMult == other.sizeMult &&  xyzMult == other.xyzMult &&  speedMult == other.speedMult);
  }
  bool operator==(const Pen *other) const
  {
    return (active == other->active && cellAlign == other->cellAlign && square == other->square && radial == other->radial && speed == other->speed &&
            depth == other->depth && radius0 == other->radius0 && radius1 == other->radius1 && rDist == other->rDist && mult == other->mult &&
            sizeMult == other->sizeMult &&  xyzMult == other->xyzMult &&  speedMult == other->speedMult);
  }
  bool operator!=(const Pen &other) const { return !(*this == other); }
};


#define SIG_PARAM_COUNT 6

// derived -- signal
template<typename T>
struct SigFieldParams
{
  T base;
  union
  {
    struct
    {
      bool multR;   // multiplies signal by 1/r
      bool multR_2; // multiplies signal by 1/r^2
      bool multT;   // multiplies signal by theta
      bool multT_2; // multiplies signal by theta^2
      bool multSin; // multiplies signal by sin(2*pi*f*t) [t => simTime, f => frequency]
      bool multCos; // multiplies signal by cos(2*pi*f*t) [t => simTime, f => frequency]
    };
    bool mods[SIG_PARAM_COUNT] = { false };
    
#ifndef __NVCC__ // NOTE: std::array not compatible with CUDA kernels
    std::array<bool, SIG_PARAM_COUNT> modArr;
#endif // __NVCC__
  };
  
#ifndef __NVCC__  
  bool operator==(const SigFieldParams &other) const
  {
    for(int i = 0; i < modArr.size(); i++)
      { if(modArr[i] != other.modArr[i]) { return false; } }
    return (base == other.base);
  }
  bool operator==(const SigFieldParams *other) const
  {
    for(int i = 0; i < modArr.size(); i++)
      { if(modArr[i] != other->modArr[i]) { return false; } }
    return (base == other.base);
  }
  bool operator!=(const SigFieldParams &other) const { return !(*this == other); }
#endif // __NVCC__

  SigFieldParams(const T &m) : base(m) { }
};

// for drawing signal in with mouse
template<typename T>
struct SignalPen : public Pen<T>
{
  using VT3 = typename DimType<T, 3>::VEC_T;
  T wavelength = 12.0;           // in dL units (cells per period)
  T frequency  = 1.0/wavelength; // Hz(c/t in sim time) for sin/cos mult flags (c --> speed in vacuum = vMat.c())
  SigFieldParams<VT3> pV;
  SigFieldParams<T>   pP;
  SigFieldParams<T>   pQn;
  SigFieldParams<T>   pQp;
  SigFieldParams<VT3> pQnv;
  SigFieldParams<VT3> pQpv;
  SigFieldParams<VT3> pE;
  SigFieldParams<VT3> pB;
  
  SignalPen()
    : pV(VT3{0,0,0}), pP(T(0)), pQn(T(0)), pQp(T(0)), pQnv(VT3{0,0,0}), pQpv(VT3{0,0,0}), pE(VT3{10,10,10}), pB(VT3{0,0,0})
  {
    this->mult = 20.0f;
    pE.multSin = true; pB.multSin = true;
  }
  ~SignalPen() = default;

  virtual bool isSignal() const override { return true; }
  
  bool operator==(const SignalPen &other) const
  {
    return (Pen<T>::operator==(static_cast<const Pen<T>*>(&other)) && wavelength == other.wavelength && frequency == other.frequency &&
            pV == other.pV && pP == other.pP && pQn == other.pQn && pQp == other.pQp &&
            pQpv == other.pQpv && pQnv == other.pQnv && pE == other.pE && pB == other.pB);
  }
  bool operator!=(const SignalPen &other) const { return !(*this == other); }
};

template<typename T>
struct MaterialPen : public Pen<T>
{
  Material<T> mat = Material<T>(2.4, 2.0, 0.0001, false);
  bool vacuumErase = false;
  MaterialPen()  = default;
  ~MaterialPen() = default;
  
  virtual bool isMaterial() const override { return true; }
  bool operator==(const MaterialPen &other) const
  {
    return (Pen<T>::operator==(static_cast<const Pen<T>*>(&other)) &&
            mat.ep == other.mat.ep && mat.mu == other.mat.mu && mat.sig == other.mat.sig && vacuumErase == other.vacuumErase);
  }
  bool operator!=(const MaterialPen &other) const { return !(*this == other); }
};



#ifndef __NVCC__

#include <nlohmann/json.hpp> // json implementation
using json = nlohmann::json;
#include "tools.hpp"

//// JSON helpers ////

// PEN BASE CLASS
template<typename T>
inline json penToJSON(const Pen<T> *pen)
{
  json js = nlohmann::ordered_json(); std::stringstream ss;
  // ordered_map 
  js["active"]    = pen->active;
  js["cellAlign"] = pen->cellAlign;
  js["square"]    = pen->square;
  js["radial"]    = pen->radial;
  js["speed"]     = pen->speed;
  js["depth"]     = pen->depth;
  js["mult"]      = pen->mult;
  js["speedMult"] = pen->speedMult;
  js["sizeMult"]  = pen->sizeMult;
  js["radius0"]   = to_string(pen->radius0,  12);
  js["radius1"]   = to_string(pen->radius1,  12);
  js["rDist"]     = to_string(pen->rDist,    12);
  js["xyzMult"]   = to_string(pen->xyzMult,  12);
  return js;
}
template<typename T>
inline bool penFromJSON(const json &js, Pen<T> *penOut)
{
  using VT3 = float3;
  bool success = true;
  if(js.contains("active"))    { penOut->active    = js["active"];    } else { success = false; }
  if(js.contains("cellAlign")) { penOut->cellAlign = js["cellAlign"]; } else { success = false; }
  if(js.contains("square"))    { penOut->square    = js["square"];    } else { success = false; }
  if(js.contains("radial"))    { penOut->radial    = js["radial"];    } else { success = false; }
  if(js.contains("speed"))     { penOut->speed     = js["speed"];     } else { success = false; }
  if(js.contains("depth"))     { penOut->depth     = js["depth"];     } else { success = false; }
  if(js.contains("mult"))      { penOut->mult      = js["mult"];      } else { success = false; }
  if(js.contains("speedMult")) { penOut->speedMult = js["speedMult"]; } else { success = false; }
  if(js.contains("sizeMult"))  { penOut->sizeMult  = js["sizeMult"];  } else { success = false; }
  if(js.contains("radius0"))   { penOut->radius0   = from_string<VT3>  (js["radius0"].get<std::string>()); } else { success = false; }
  if(js.contains("radius1"))   { penOut->radius1   = from_string<VT3>  (js["radius1"].get<std::string>()); } else { success = false; }
  if(js.contains("rDist"))     { penOut->rDist     = from_string<VT3>  (js["rDist"  ].get<std::string>()); } else { success = false; }
  if(js.contains("xyzMult"))   { penOut->xyzMult   = from_string<VT3>  (js["xyzMult"].get<std::string>()); } else { success = false; }
  return success;
}

// SIGNAL PEN
template<typename T>
inline json sigPenToJSON(const SignalPen<T> &pen)
{
  json js  = penToJSON(&pen);
  json jss = nlohmann::ordered_json();
  jss["wavelength"] = pen.wavelength;
  jss["frequency"]  = pen.frequency;
  std::stringstream ss;
  jss["pV" ] ["base"] = to_string(pen.pV.base, 12 );
  jss["pP" ] ["base"] = to_string(pen.pP.base, 12 );
  jss["pQn"] ["base"] = to_string(pen.pQn.base, 12);
  jss["pQp"] ["base"] = to_string(pen.pQp.base, 12);
  jss["pQnv"]["base"] = to_string(pen.pQnv.base, 12);
  jss["pQpv"]["base"] = to_string(pen.pQpv.base, 12);
  jss["pE" ] ["base"] = to_string(pen.pE.base, 12 );
  jss["pB" ] ["base"] = to_string(pen.pB.base, 12 );
  for(int i = 0; i < pen.pV.modArr.size();   i++) { ss << (pen.pV.mods  [i] ? "1":"0") << " "; } jss["pV"]  ["mods"] = ss.str(); ss.str(""); ss.clear();
  for(int i = 0; i < pen.pP.modArr.size();   i++) { ss << (pen.pP.mods  [i] ? "1":"0") << " "; } jss["pP"]  ["mods"] = ss.str(); ss.str(""); ss.clear();
  for(int i = 0; i < pen.pQn.modArr.size();  i++) { ss << (pen.pQn.mods [i] ? "1":"0") << " "; } jss["pQn"] ["mods"] = ss.str(); ss.str(""); ss.clear();
  for(int i = 0; i < pen.pQp.modArr.size();  i++) { ss << (pen.pQp.mods [i] ? "1":"0") << " "; } jss["pQp"] ["mods"] = ss.str(); ss.str(""); ss.clear();
  for(int i = 0; i < pen.pQnv.modArr.size(); i++) { ss << (pen.pQnv.mods[i] ? "1":"0") << " "; } jss["pQnv"]["mods"] = ss.str(); ss.str(""); ss.clear();
  for(int i = 0; i < pen.pQpv.modArr.size(); i++) { ss << (pen.pQpv.mods[i] ? "1":"0") << " "; } jss["pQpv"]["mods"] = ss.str(); ss.str(""); ss.clear();
  for(int i = 0; i < pen.pE.modArr.size();   i++) { ss << (pen.pE.mods  [i] ? "1":"0") << " "; } jss["pE"]  ["mods"] = ss.str(); ss.str(""); ss.clear();
  for(int i = 0; i < pen.pB.modArr.size();   i++) { ss << (pen.pB.mods  [i] ? "1":"0") << " "; } jss["pB"]  ["mods"] = ss.str(); ss.str(""); ss.clear();
  js["signal"] = jss;
  return js;
}
template<typename T>
inline bool sigPenFromJSON(const json &js, SignalPen<T> &penOut)
{
  using VT3 = float3;
  bool success = penFromJSON(js, &penOut);
  if(js.contains("signal"))
    {
      json jss = js["signal"];
      if(jss.contains("wavelength")) { penOut.wavelength = jss["wavelength"]; } else { std::cout << "====> wavelength\n"; success = false; }
      if(jss.contains("frequency"))  { penOut.frequency  = jss["frequency"];  } else { std::cout << "====> frequency\n";  success = false; }
      std::stringstream ss; int b;
      if(jss.contains("pV")  && jss["pV"].contains("mods"))
        {
          penOut.pV.base = from_string<VT3>(jss["pV"]["base"].get<std::string>());
          ss.clear(); ss.str(jss["pV"]["mods"].get<std::string>());
          for(int i = 0; i < penOut.pV.modArr.size();  i++) { ss >> b; penOut.pV.mods[i]  = (b != 0); }
        } else { std::cout << "====> pV\n"; success = false; }
      if(jss.contains("pP")  && jss["pP"].contains("mods"))
        {
          penOut.pP.base = from_string<T>(jss["pP"]["base"].get<std::string>());
          ss.clear(); ss.str(jss["pP"]["mods"].get<std::string>());
          for(int i = 0; i < penOut.pP.modArr.size();  i++) { ss >> b; penOut.pP.mods[i]  = (b != 0); }
        } else { std::cout << "====> pP\n"; success = false; }
      if(jss.contains("pQn") && jss["pQn"].contains("mods"))
        {
          penOut.pQn.base = from_string<T>(jss["pQn"]["base"].get<std::string>());
          ss.clear(); ss.str(jss["pQn"]["mods"].get<std::string>());
          for(int i = 0; i < penOut.pQn.modArr.size(); i++) { ss >> b; penOut.pQn.mods[i] = (b != 0); }
        } else { std::cout << "====> pQn\n"; success = false; }
      if(jss.contains("pQp") && jss["pQp"].contains("mods"))
        {
          penOut.pQp.base = from_string<T>(jss["pQp"]["base"].get<std::string>());
          ss.clear(); ss.str(jss["pQp"]["mods"].get<std::string>());
          for(int i = 0; i < penOut.pQp.modArr.size(); i++) { ss >> b; penOut.pQp.mods[i] = (b != 0); }
        } else { std::cout << "====> pQp\n"; success = false; }
      if(jss.contains("pQnv") && jss["pQnv"].contains("mods"))
        {
          penOut.pQnv.base = from_string<VT3>(jss["pQnv"]["base"].get<std::string>());
          ss.clear(); ss.str(jss["pQnv"]["mods"].get<std::string>());
          for(int i = 0; i < penOut.pQnv.modArr.size(); i++) { ss >> b; penOut.pQnv.mods[i] = (b != 0); }
        } else { std::cout << "====> pQnv\n"; success = false; }
      if(jss.contains("pQpv") && jss["pQpv"].contains("mods"))
        {
          penOut.pQpv.base = from_string<VT3>(jss["pQpv"]["base"].get<std::string>());
          ss.clear(); ss.str(jss["pQpv"]["mods"].get<std::string>());
          for(int i = 0; i < penOut.pQpv.modArr.size(); i++) { ss >> b; penOut.pQpv.mods[i] = (b != 0); }
        } else { std::cout << "====> pQpv\n"; success = false; }
      if(jss.contains("pE")  && jss["pE"].contains("mods"))
        {
          penOut.pE.base = from_string<VT3>(jss["pE"]["base"].get<std::string>());
          ss.clear(); ss.str(jss["pE"]["mods"].get<std::string>());
          for(int i = 0; i < penOut.pE.modArr.size();  i++) { ss >> b; penOut.pE.mods[i]  = (b != 0); }
        } else { std::cout << "====> pE\n"; success = false; }
      if(jss.contains("pB")  && jss["pB"].contains("mods"))
        {
          penOut.pB.base = from_string<VT3>(jss["pB"]["base"].get<std::string>());
          ss.clear(); ss.str(jss["pB"]["mods"].get<std::string>());
          for(int i = 0; i < penOut.pB.modArr.size();  i++) { ss >> b; penOut.pB.mods[i]  = (b != 0); }
        } else { std::cout << "====> pB\n"; success = false; }
    }
  else { std::cout << "====> 'signal' group\n"; success = false; }
  return success;
}

// MATERIAL PEN
template<typename T>
inline json matPenToJSON(const MaterialPen<T> &pen)
{
  json js  = penToJSON(&pen);
  json jsm = nlohmann::ordered_json();
  jsm["vacuum"] = pen.mat.vacuum();
  jsm["permittivity"] = pen.mat.ep;
  jsm["permeability"] = pen.mat.mu;
  jsm["conductivity"] = pen.mat.sig;
  js["material"] = jsm;
  return js;
}

template<typename T>
inline bool matPenFromJSON(const json &js, MaterialPen<T> &penOut)
{
  using VT3 = typename DimType<T, 3>::VEC_T;
  bool success = penFromJSON(js, &penOut);
  if(js.contains("material"))
    {
      json jsm = js["material"];
      if(jsm.contains("permittivity")) { penOut.mat.ep  = jsm["permittivity"]; } else { success = false; }
      if(jsm.contains("permeability")) { penOut.mat.mu  = jsm["permeability"]; } else { success = false; }
      if(jsm.contains("conductivity")) { penOut.mat.sig = jsm["conductivity"]; } else { success = false; }
      if(jsm.contains("vacuum"))       { penOut.mat.setVacuum(jsm["vacuum"]);  } else { success = false; }
    }
  else { return false; }
  return success;
}


template<typename T> inline bool penFromJSON(const json &js, MaterialPen<T> &penOut) { return matPenFromJSON(js, penOut); }
template<typename T> inline bool penFromJSON(const json &js, SignalPen<T>   &penOut) { return sigPenFromJSON(js, penOut); }
template<typename T> inline json penToJSON(MaterialPen<T> &penOut) { return matPenToJSON(penOut); }
template<typename T> inline json penToJSON(SignalPen<T>   &penOut) { return sigPenToJSON(penOut); }


#endif // __NVCC__


#endif // DRAW_PENS_H
