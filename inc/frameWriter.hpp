#ifndef FRAME_WRITER_HPP
#define FRAME_WRITER_HPP

#include <filesystem>
namespace fs = std::filesystem;
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "output.hpp"
#include "vector.hpp"
#include "cuda-texture.cuh"


#define RENDER_BASE_PATH (fs::current_path() / "rendered")

// multithreaded interface for writing frames to image files
class FrameWriter
{
private:
  OutputParams *op = nullptr;
  
  // file paths (images output as: [mBaseDir]/[mSimName]/[mSimName]-[mParams.frame].png)
  fs::path mBaseDir  = RENDER_BASE_PATH;
  fs::path mImageDir = mBaseDir / "unnamed";

  int mNumFrames = 0;
  int mMaxFrames = 256;
  std::queue<FrameOut*> mNewFrames; // image data for source frames to be written to files
  std::queue<FrameOut*> mFramePool; // recycled frames (avoids uneccesary reallocation)

  // threading
  bool mActive   = false;
  bool mFlushing = false;
  int mNumThreads    = 0; // number of worker threads
  int mMaxThreads    = 0; // max number of threads
  int mActiveThreads = 0; // number of active threads
  std::vector<std::thread> mThreads;
  std::mutex mPoolLock;  // used to lock frame queues
  std::mutex mFlushLock; // used to lock get()/push() while flushing
  std::condition_variable  mPoolCv;  // workerLoop()
  std::condition_variable  mPushCv;  // push()

  bool checkBaseRenderPath();
  bool checkSimRenderPath();
  
  void workerLoop(int tid);
  void process(FrameOut *f, int tid=-1); // TODO: avoid processing? (flips Y)
  void writeToFile(FrameOut *f, int tid=-1);
  
public:
  FrameWriter(OutputParams *params);
  ~FrameWriter();

  void setThreads(int nThreads);
  void setBufferSize(int nFrames);
  const int& bufferSize() const { return mMaxFrames; } // direct access
  int& bufferSize()             { return mMaxFrames; }
  
  FrameOut* get();            // pop from from pool, or create
  void push(FrameOut *frame); // push new rendered frame
  void clear();               // discard all queued frames
  void flush();               // flush all queued frames
};


#endif // FRAME_WRITER_HPP

