#ifndef FIELD_HPP
#define FIELD_HPP

#include "fluid.cuh"
#include "field.cuh"
#include "setting.hpp"
#include "settingForm.hpp"
#include "mathParser.hpp"

#define FIELD_RES_SMALL_STEP int3{  8,   8,   8}
#define FIELD_RES_LARGE_STEP int3{128, 128,  64}
#define FIELD_RES_MIN        int3{  1,   1,   1}
#define FIELD_RES_MAX        int3{2048, 2048, 512} // probably higher than it could reasonably be set, unless only in one dimension
#define TEX_RES_SMALL_STEP   int2{  8,   8, }
#define TEX_RES_LARGE_STEP   int2{128, 128, }
#define TEX_RES_MIN          int2{  1,   1, }
#define TEX_RES_MAX          int2{4096, 4096}

// wraps resources for initialization via recursive expression
template<typename T>
struct FieldInit
{
  bool               active  = false;   // if false, initializes to 0
  std::string        str     = "0";     // expression string
  Expression<T>     *hExpr   = nullptr; // host   expression (necessary?)
  CudaExpression<T> *dExpr   = nullptr; // device expression
};

template<typename T>
struct FieldInterface : public SettingForm
{
  typedef typename DimType<T, 3>::VEC_T VT3;

  int2 texRes2D = int2{1024, 1024};  // desired resolution (number of cells/pixels) of rendered texture
  int2 texRes3D = int2{1024, 1024};  // desired resolution (number of cells/pixels) of rendered texture
  bool texRes3DMatch = true;         // if true, overrides to match current view resolution (depends on size of window)

  FluidParams<T> *cp;       // cuda field params

  // physics toggles
  bool running       = false; // toggle all physics (play/pause)
  bool inputDecay    = true;  // toggle input signal decay (for E/B weirdness (?))
  bool applyGravity  = false; // toggle gravity

  bool updateEM      = true;  // toggle EM updates (Maxwell's equations)
  bool updateQ       = true;  //   toggle charge update step
  bool updateCoulomb = true;  //   toggle Coulomb force calculation
  bool updateQpot    = false; //   toggle charge potential step (WIP)
  bool updateE       = true;  //   toggle electric update step
  bool updateB       = true;  //   toggle magnetic update step

  bool updateFluid   = true; // toggle fluid update steps
  bool updateP1      = true; //    pre-advect pressure solve
  bool updateAdvect  = true; //    parameter advection
  bool updateVisc    = true; //    viscosity solve
  bool updateP2      = true; //    post-advect pressure solve

  // initialization expressions
  FieldInit<VT3>   initV;   // fluid velocity field
  FieldInit<float> initP;   // fluid pressure field
  FieldInit<float> initQn;  // - charge field
  FieldInit<float> initQp;  // + charge field
  FieldInit<VT3>   initQnv; // - charge velocity field
  FieldInit<VT3>   initQpv; // + charge velocity field
  FieldInit<VT3>   initE;   // electric field   (Maxwell's equations)
  FieldInit<VT3>   initB;   // magnetic field   (Maxwell's equations)
  FieldInit<float> initEp;  // material epsilon (permittivity)
  FieldInit<float> initMu;  // material mu      (permeability)
  FieldInit<float> initSig; // material sigma   (conductivity)

  std::function<void()> fieldResCallback; // callback when field resolution is changed
  std::function<void()> texRes2DCallback; // callback when 2D texture resolution is changed
  std::function<void()> texRes3DCallback; // callback when 3D texture resolution is changed

  FieldInterface(FluidParams<T> *cp_, const std::function<void()> &fieldResCb,
                 const std::function<void()> &texRes2DCb, const std::function<void()> &texRes3DCb);
  ~FieldInterface() = default; // { if(mForm) { delete mForm; mForm = nullptr; } }

  // update callback for when an init expression string is changed
  template<typename U> void initStrUpdateCb(FieldInit<U> *init)
  {
    std::cout << "INIT UPDATE\n";
    if(init->str.empty()) { init->str = "0"; } // empty string defaults to 0 (cleared field)
    if(init->dExpr) { cudaFree(init->dExpr); init->dExpr = nullptr; } // destroy cuda expression (reinitialized with new string by SimWindow)
  }
};



template<typename T>
FieldInterface<T>::FieldInterface(FluidParams<T> *cp_, const std::function<void()> &frCb, const std::function<void()> &tr2DCb, const std::function<void()> &tr3DCb)
  : cp(cp_), fieldResCallback(frCb), texRes2DCallback(tr2DCb), texRes3DCallback(tr3DCb)
{
  initEp.str = "1"; initMu.str = "1";

  //// param group
  SettingGroup *paramGroup = new SettingGroup("Field Parameters", "fieldParams", { }); add(paramGroup);
  paramGroup->setHelp("TEST");

  // flags
  auto *sRUN   = new Setting<bool>("Running",              "running",    &running);  paramGroup->add(sRUN);
  sRUN->setHelp("If checked, step physics each frame");
  // field/texture sizes
  auto *sFRES = new Setting<int3> ("Field Resolution",     "fieldRes",   &cp->fs);   paramGroup->add(sFRES);
  sFRES->setUpdateCallback(fieldResCallback);
  sFRES->setHelp("Size of field (number of cells) in each dimension");
  sFRES->setFormat(FIELD_RES_SMALL_STEP, FIELD_RES_LARGE_STEP);
  sFRES->setMin(FIELD_RES_MIN); sFRES->setMax(FIELD_RES_MAX);
  auto *sTRES2 = new Setting<int2> ("Tex Resolution (2D)", "texRes2D",   &texRes2D); paramGroup->add(sTRES2);
  sTRES2->setUpdateCallback(texRes2DCallback);
  sTRES2->setHelp("Size of 2D rendered texture (EM and Material views)");
  sTRES2->setFormat(TEX_RES_SMALL_STEP, TEX_RES_LARGE_STEP);
  sTRES2->setMin(TEX_RES_MIN); sTRES2->setMax(TEX_RES_MAX);
  auto *sTRES3 = new Setting<int2> ("Tex Resolution (3D)", "texRes3D",   &texRes3D); paramGroup->add(sTRES3);
  sTRES3->setUpdateCallback(texRes3DCallback);
  sTRES3->setHelp("Size of 3D rendered texture (3D view)");
  sTRES3->setFormat(TEX_RES_SMALL_STEP, TEX_RES_LARGE_STEP);
  sTRES3->setMin(TEX_RES_MIN); sTRES3->setMax(TEX_RES_MAX);
  auto *sTRES3M = new Setting<bool>("   Match Viewport", "texRes3DMatch", &texRes3DMatch); paramGroup->add(sTRES3M);
  sTRES3M->setHelp("Dynamically fit 3D texture size to actual resolution (changes if window is resized)");

  auto *sFPP = new Setting<VT3>  ("Position",               "fPos",      &cp->fp);      paramGroup->add(sFPP);
  sFPP->setHelp("3D position of field within sim");
  auto *sREF = new Setting<bool> ("Reflective Bounds (EM)", "reflectEM", &cp->reflect); paramGroup->add(sREF);
  sREF->setHelp("Reflect signals at boundary (some reflections are still apparent even if disabled)");

  std::vector<std::string> edgeNames {"WRAP", "VOID", "FREESLIP"};
  auto *sREENX = new ComboSetting("-X edge", "edgeNX", (int*)&cp->edgeNX, edgeNames);
  sREENX->setHelp("-X axis edge conditions"); paramGroup->add(sREENX);
  auto *sREEPX = new ComboSetting("+X edge", "edgePX", (int*)&cp->edgePX, edgeNames);
  sREEPX->setHelp("+X axis edge conditions"); paramGroup->add(sREEPX);
  auto *sREENY = new ComboSetting("-Y edge", "edgeNY", (int*)&cp->edgeNY, edgeNames);
  sREENY->setHelp("-Y axis edge conditions"); paramGroup->add(sREENY);
  auto *sREEPY = new ComboSetting("+Y edge", "edgePY", (int*)&cp->edgePY, edgeNames);
  sREEPY->setHelp("+Y axis edge conditions"); paramGroup->add(sREEPY);
  auto *sREENZ = new ComboSetting("-Z edge", "edgeNZ", (int*)&cp->edgeNZ, edgeNames);
  sREENZ->setHelp("-Z axis edge conditions"); paramGroup->add(sREENZ);
  auto *sREEPZ = new ComboSetting("+Z edge", "edgePZ", (int*)&cp->edgePZ, edgeNames);
  sREEPZ->setHelp("+Z axis edge conditions"); paramGroup->add(sREEPZ);
  auto *sSD   = new Setting<T>  ("Input Decay",              "decay",  &cp->decay);  paramGroup->add(sSD);
  sSD->setHelp("Input signal field is multiplied by decay^dt each frame to prevent stuck vectors");
  sSD->setToggle(&inputDecay);
  sSD->setFormat(0.01f, 0.1f); sSD->setMin(-2.0f); sSD->setMax(2.0f);

  auto *sFG = new Setting<VT3>  ("Gravity", "gravity", &cp->gravity); paramGroup->add(sFG);
  sFG->setFormat(VT3{0.01, 0.01, 0.01}, VT3{0.1, 0.1, 0.1}, "%4f");
  sFG->setToggle(&applyGravity);
  sFG->setHelp("(vector) Force of gravity applied to fluid");

  //// step group
  SettingGroup *stepGroup  = new SettingGroup("Physics Steps", "physicsSteps", { }); add(stepGroup);
  stepGroup->setHelp("TEST");

  // --> Maxwell's equations
  auto *sUEM = new Setting<bool>("Update EM",            "updateEM", &updateEM); stepGroup->add(sUEM);
  sUEM->setHelp("EM field update (Maxwell's equations)");
  // Q/E --> E (Coulomb force calcualtion)
  auto *qcGroup = new SettingGroup  (" --> Coulomb Force", "qCoulomb", {}); stepGroup->add(qcGroup); qcGroup->setHorizontal(true); qcGroup->setColumns(3);
  qcGroup->setHelp("Charge field divergence calculation (corrects electric potential as divergence of charge)");
  qcGroup->setEnabledCallback([this]() { return updateEM; });
  auto *sUQcE = new Setting<bool>("",       "qcEnable", &updateCoulomb);   qcGroup->add(sUQcE);
  auto *sUQcR = new Setting<int> ("Radius", "qcRad",    &cp->rCoulomb);    qcGroup->add(sUQcR);
  sUQcR->setHelp("Effective radius of Coulomb force calculations (NOTE: O(r^3) time complexity)"); sUQcR->setFormat(1, 10); sUQcR->setMin(0);
  auto *sUQcM = new Setting<T>   ("Mult",   "qcMult",   &cp->coulombMult); qcGroup->add(sUQcM);
  sUQcM->setHelp("Coulomb force multiplier"); sUQcM->setFormat(0.1f, 1.0f, "%.4f");

  // Q/E --> E (experimental potential correction based on vector calculus)
  // auto *qpGroup = new SettingGroup  (" --> Correct Potential", "qPotential", {}); stepGroup->add(qpGroup); qpGroup->setHorizontal(true); qpGroup->setColumns(2);
  // qpGroup->setHelp("Charge field divergence calculation (corrects electric potential as divergence of charge)");
  // qpGroup->setEnabledCallback([this]() { return updateEM; });
  // auto *sUQpE = new Setting<bool>("",           "qpEnable", &updateQpot); qpGroup->add(sUQpE);
  // auto *sUQpI = new Setting<int> ("Iterations", "qpIter",   &cp->qIter);  qpGroup->add(sUQpI);
  // sUQpI->setHelp("Number of iterations (Jacobi method)"); sUQpI->setFormat(1, 10); sUQpI->setMin(0);

  // E/B/Qv/v --> Q/Qv (apply forces)
  auto *sUQ  = new Setting<bool> (" --> Update Q", "updateQ", &updateQ); stepGroup->add(sUQ);
  sUQ->setHelp("Charge field physics update (applies forces from E/B field to Q/Qv and steps Q via Qv)");
  sUQ->setEnabledCallback([this]() { return updateEM; });
  // Qv/E/B --> E
  auto *sUE  = new Setting<bool> (" --> Update E", "updateE", &updateE); stepGroup->add(sUE);
  sUE->setHelp("Electric field physics update");
  sUE->setEnabledCallback([this]() { return updateEM; });
  // E/B --> B
  auto *sUB  = new Setting<bool> (" --> Update B", "updateB", &updateB); stepGroup->add(sUB);
  sUB->setHelp("Magnetic field physics update");
  sUB->setEnabledCallback([this]() { return updateEM; });

  // --> fluid dynamics
  auto *fGroup = new SettingGroup("Fluid Dynamics", "fluidUpdate", {}); stepGroup->add(fGroup); fGroup->setHorizontal(true); fGroup->setColumns(2);
  fGroup->setHelp("Fluid physics update");
  auto *sUF  = new Setting<bool>("",        "updateFluid", &updateFluid); fGroup->add(sUF);
  auto *sUFD = new Setting<T>   ("Density", "fDensity",    &cp->density); fGroup->add(sUFD);
  sUFD->setFormat(0.1f, 1.0f, "%.4f");
  sUFD->setHelp("Fluid density");

  // pre-advect pressure
  auto *fp1Group = new SettingGroup(" --> Pressure(pre)", "fluidP1", {}); stepGroup->add(fp1Group); fp1Group->setHorizontal(true); fp1Group->setColumns(2);
  fp1Group->setHelp("Fluid pre-advect pressure update");
  fp1Group->setEnabledCallback([this]() { return updateFluid; });
  auto *sUFP1E = new Setting<bool>("",           "fp1Enable", &updateP1); fp1Group->add(sUFP1E);
  auto *sUFP1I = new Setting<int> ("Iterations", "fp1Iter",   &cp->pIter1);   fp1Group->add(sUFP1I); // iterations
  sUFP1I->setMin(0); sUFP1I->setMax(11111); sUFP1I->setFormat(1, 10);
  sUFP1I->setHelp("Number of iterations (Jacobi method)");
  // advection
  auto *faGroup = new SettingGroup(" --> Advect", "fluidAdvect", {}); stepGroup->add(faGroup); faGroup->setHorizontal(true); faGroup->setColumns(2);
  faGroup->setHelp("Fluid advection (move cell contents by velocity)");
  faGroup->setEnabledCallback([this]() { return updateFluid; });
  auto *sUFAE = new Setting<bool>("", "faEnable", &updateAdvect); faGroup->add(sUFAE);
  auto *sUVAL = new Setting<T>   ("Limit V","fluidMaxV", &cp->maxV);  faGroup->add(sUVAL); sUVAL->setToggle(&cp->limitV); // velocity limit
  sUVAL->setMin(0); sUFP1I->setMax(11111); sUVAL->setFormat(0.1f, 1.0f, "%.4f");
  sUVAL->setHelp("Limit fluid velocity (V) to this magnitude\n --> V = normalize(V)*max(length(V), <limit>)");
  // viscosity
  auto *fvGroup = new SettingGroup(" --> Viscosity", "fluidVisc", {}); stepGroup->add(fvGroup); fvGroup->setHorizontal(true); fvGroup->setColumns(3);
  fvGroup->setHelp("Fluid viscosity update");
  fvGroup->setEnabledCallback([this]() { return updateFluid; });
  auto *sUFVE = new Setting<bool> ("", "fvEnable", &updateVisc);    fvGroup->add(sUFVE);
  auto *sUFVV = new Setting<T>    ("", "fvVisc",   &cp->viscosity); fvGroup->add(sUFVV); // viscosity
  sUFVV->setFormat(0.1f, 1.0f, "%.4f");
  sUFVV->setHelp("Fluid viscosity");
  auto *sUFVI = new Setting<int>  ("", "fvIter",   &cp->vIter);  fvGroup->add(sUFVI); // iterations
  sUFVI->setMin(0); sUFVI->setMax(11111); sUFVI->setFormat(0.1f, 1.0f);
  sUFVI->setHelp("Number of iterations (Jacobi method)");
  // post-advect pressure
  auto *fp2Group = new SettingGroup(" --> Pressure(post)", "fluidP2", {}); stepGroup->add(fp2Group); fp2Group->setHorizontal(true); fp2Group->setColumns(2);
  fp2Group->setHelp("Fluid pre-advect pressure update");
  fp2Group->setEnabledCallback([this]() { return updateFluid; });
  auto *sUFP2E = new Setting<bool>("",           "fp2Enable", &updateP2);   fp2Group->add(sUFP2E);
  auto *sUFP2I = new Setting<int> ("Iterations", "fp2Iter",   &cp->pIter2); fp2Group->add(sUFP2I); // iterations
  sUFP2I->setFormat(1, 10); sUFP2I->setMin(0); sUFP2I->setMax(11111);
  sUFP2I->setHelp("Number of iterations (Jacobi method)");


  //// initial condition group
  SettingGroup *initGroup  = new SettingGroup("Initial Conditions", "initialConditions", { }); add(initGroup);
  initGroup->setHelp("Each component field can be set parametrically\n"
                     "  Vector field:\n"
                     "   --> r:  distance from center of field in each dimension (len(r) ==> magnitude / norm(r) ==> direction)\n"
                     "   --> t:  some weird components based off angle relative to center of field\n"
                     "   [...?]\n"
                     "  Scalar field:\n"
                     ">>TODO: improve parameters available, finish help [...]");
  auto *sVINIT   = new Setting<std::string>("V  init (fluid)",    "VInit",   &initV.str);   initGroup->add(sVINIT);
  sVINIT->setUpdateCallback(  [&]() { initStrUpdateCb(&initV);   });
  sVINIT->setToggle(&initV.active);
  auto *sPINIT   = new Setting<std::string>("P  init (fluid)",    "PInit",   &initP.str);   initGroup->add(sPINIT);
  sPINIT->setUpdateCallback(  [&]() { initStrUpdateCb(&initP);   });
  sPINIT->setToggle(&initP.active);
  auto *sQNINIT  = new Setting<std::string>("Q- init (EM)",       "QnInit",  &initQn.str);  initGroup->add(sQNINIT);
  sQNINIT->setUpdateCallback( [&]() { initStrUpdateCb(&initQn);  });
  sQNINIT->setToggle(&initQn.active);
  auto *sQPINIT  = new Setting<std::string>("Q+ init (em)",       "QpInit",  &initQp.str);  initGroup->add(sQPINIT);
  sQPINIT->setUpdateCallback( [&]() { initStrUpdateCb(&initQp);  });
  sQPINIT->setToggle(&initQp.active);
  auto *sQNVINIT = new Setting<std::string>("Qnv init (em)",      "QnvInit", &initQnv.str); initGroup->add(sQNVINIT);
  sQNVINIT->setUpdateCallback([&]() { initStrUpdateCb(&initQnv); });
  sQNVINIT->setToggle(&initQnv.active);
  auto *sQPVINIT = new Setting<std::string>("Qpv init (em)",      "QpvInit", &initQpv.str); initGroup->add(sQPVINIT);
  sQPVINIT->setUpdateCallback([&]() { initStrUpdateCb(&initQpv); });
  sQPVINIT->setToggle(&initQpv.active);
  auto *sEINIT   = new Setting<std::string>("E  init (em)",       "EInit",   &initE.str);   initGroup->add(sEINIT);
  sEINIT->setUpdateCallback(  [&]() { initStrUpdateCb(&initE);   });
  sEINIT->setToggle(&initE.active);
  auto *sBINIT   = new Setting<std::string>("B  init (em)",       "BInit",   &initB.str);   initGroup->add(sBINIT);
  sBINIT->setUpdateCallback(  [&]() { initStrUpdateCb(&initB);   });
  sBINIT->setToggle(&initB.active);
  auto *sEPINIT  = new Setting<std::string>("ε  init (material)", "epInit",  &initEp.str);  initGroup->add(sEPINIT);
  sEPINIT->setUpdateCallback( [&]() { initStrUpdateCb(&initEp);  });
  sEPINIT->setToggle(&initEp.active);
  auto *sMUINIT  = new Setting<std::string>("μ  init (material)", "muInit",  &initMu.str);  initGroup->add(sMUINIT);
  sMUINIT->setUpdateCallback( [&]() { initStrUpdateCb(&initMu);  });
  sMUINIT->setToggle(&initMu.active);
  auto *sSIGINIT = new Setting<std::string>("σ  init (material)", "sigInit", &initSig.str); initGroup->add(sSIGINIT);
  sSIGINIT->setUpdateCallback([&]() { initStrUpdateCb(&initSig); });
  sSIGINIT->setToggle(&initSig.active);
}

#endif // FIELD_HPP
