#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include "settingForm.hpp"

// forward declarations
class FrameWriter;

// data for a single output frame
struct FrameOut
{
  int f = 0;                     // frame number
  int channels = 0;              // number of channels to write
  Vec2i size = Vec2i(0,0);       // size of frame (W x H pixels)
  unsigned char *raw  = nullptr; // raw texture data
  unsigned char *data = nullptr; // final image data to write

  FrameOut() = default;
  FrameOut(const Vec2i &sz, int numChannels) { create(sz, numChannels); }
  ~FrameOut() { destroy(); }

  bool create(const Vec2i &sz, int numChannels)
  {
    if(size != sz || channels != numChannels)
      {
        destroy();
        if(sz.x > 0 && sz.y > 0 && numChannels > 0)
          {
            raw  = new unsigned char[numChannels*sz.x*sz.y];
            data = new unsigned char[numChannels*sz.x*sz.y];
            size = sz; channels = numChannels;
          }
      }
    return true;
  }

  void destroy()
  {
    if(raw)  { delete[] raw;  raw  = nullptr; }
    if(data) { delete[] data; data = nullptr; }
    size = Vec2i(0,0); channels = 0;
  }
};

// parameters for writing rendered frames to files (to be rendered into video)
struct OutputParams
{
  bool active    = false;
  bool lockViews = false;
  std::string projectName = "unnamed"; // project directory name/file prefix
  std::string extension   = ".png";    // file extension (supported:  .png | .hdr)
  int  pngCompress = 10;               // PNG compression level (only applicable for PNG files)
  bool writeAlpha  = false;            // if true, includes alpha channel if supported by file format
  int2 outSize     = int2{1920, 1080}; // output video resolution
  
  int frameDigits = 5;   // number of digits to pad frame number (with zeros) in file name
  int nThreads    = 4;   // number of worker threads
  int bufferSize  = 256; // max number of frames to buffer
  std::string estMemory = ""; // max number of frames to buffer
};


// UI for file writing parameters
class OutputInterface : public SettingForm
{
private:
  OutputParams *op      = nullptr;
  FrameWriter  *mWriter = nullptr;
  
  OutputInterface(const OutputInterface &other) = delete;
  OutputInterface& operator=(const OutputInterface &other) = delete;
  
public:
  OutputInterface(OutputParams *params, FrameWriter *writer);
  virtual ~OutputInterface() = default;

  void setWriter(FrameWriter *writer) { mWriter = writer; }
  
  OutputParams* params()             { return op; }
  const OutputParams* params() const { return op; }
};


#endif // OUTPUT_HPP
