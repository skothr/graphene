#include "output.hpp"

#include <iomanip>

#include "setting.hpp"
#include "frameWriter.hpp"

OutputInterface::OutputInterface(OutputParams *params, FrameWriter *writer)
  : op(params), mWriter(writer)
{
  auto *g = new SettingGroup("", "outParams", { }); add(g, false);
  auto *sOR  = new Setting<int2>       ("Resolution",     "outSize",      &op->outSize);     g->add(sOR);
  
  auto *sBN  = new Setting<std::string>("Project Name",   "projectName",  &op->projectName);
  auto *sEX  = new Setting<std::string>("Ext", "extension", &op->extension); sEX->setBodyW(69.0f);
  auto *extGroup = new SettingGroup("", "extSettings", { sBN, sEX }); extGroup->setColumns(2); g->add(extGroup);
  
  auto *sPA  = new Setting<bool>("Alpha Channel",  "outAlpha",     &op->writeAlpha);
  auto *sPC  = new Setting<int> ("Compression",    "pngCompress",  &op->pngCompress); sPC->setMin(0); sPC->setMax(10);
  sPC->setVisibleCallback([this]() -> bool { return op ? (op->extension.find(".png") != std::string::npos) : false; });
  auto *optGroup = new SettingGroup("", "optSettings", { sPA, sPC }); optGroup->setColumns(2); g->add(optGroup);
  
  auto *sFD  = new Setting<int> ("Frame Digits",  "frameDigits", &op->frameDigits); g->add(sFD); sFD->setMin(1);
  auto *sPT  = new Setting<int> ("Threads",       "nThreads",    &op->nThreads);    g->add(sPT); sPT->setMin(1);
  sPT->setUpdateCallback([&]() { if(mWriter) { mWriter->setThreads(op->nThreads); } });

  auto *bufferGroup = new SettingGroup("", "bufferParams", { }); add(bufferGroup, false);
  bufferGroup->setColumns(2); //bufferGroup->setHorizontal(true);
  auto *sBS  = new Setting<int> ("Frame Buffer",  "bufferSize",  &op->bufferSize); bufferGroup->add(sBS); sBS->setMin(0);
  sBS->setUpdateCallback([&]()
  {
    if(mWriter) { mWriter->setBufferSize(op->bufferSize); }
    std::stringstream ss; ss << "(~";
    ss << std::fixed << std::setprecision(2)
       << ((unsigned long)op->bufferSize*(unsigned long)op->outSize.x*(unsigned long)op->outSize.y*(op->writeAlpha ? 4UL : 3UL)*sizeof(float))/1000000000.0
       << "GB RAM)";
    op->estMemory = ss.str();
  });
  sOR->setUpdateCallback([sBS]() { sBS->updateAll(); }); // update memory estimate when resolution is changed 
  sPA->setUpdateCallback([sBS]() { sBS->updateAll(); }); // update memory estimate when writeAlpha is changed 
  auto *sBSm = new TextSetting  ("",              "estMemory",   &op->estMemory); bufferGroup->add(sBSm);
  
  auto *sLV  = new Setting<bool>("Lock Views",    "lockViews",   &op->lockViews); g->add(sLV);
  auto *sOA  = new Setting<bool>("Write to File", "outActive",   &op->active);    g->add(sOA);
}
