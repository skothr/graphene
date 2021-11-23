#include "frameWriter.hpp"

#include <iostream>
#include <iomanip>
#include <functional>
#include <chrono>
using namespace std::literals::chrono_literals;

#include "image.hpp"

FrameWriter::FrameWriter(OutputParams *params)
  : op(params)
{
  mActive = true;
  setBufferSize(op->bufferSize);
  setThreads(op->nThreads);
}

FrameWriter::~FrameWriter()
{
  std::cout << "====== Destroying FrameWriter...\n";
  flush(); // flush existing frames
  {
    std::lock_guard<std::mutex> pLock(mPoolLock);
    mActive = false;
  }
  mPoolCv.notify_all(); // signal threads to exit
  
  // join threads
  for(int i = 0; i < mThreads.size(); i++)
    {
      std::cout << "======== [Waiting for thread " << i << "...]\n";
      mThreads[i].join();
    }
  mThreads.clear();

  // delete frames
  while(!mFramePool.empty()) { delete mFramePool.front(); mFramePool.pop(); };
  while(!mNewFrames.empty()) { delete mNewFrames.front(); mNewFrames.pop(); };
  std::cout << "====== (done)\n";
}

void FrameWriter::setThreads(int nThreads)
{
  std::unique_lock<std::mutex> pLock(mPoolLock);
  int diff = nThreads - mNumThreads;
  mMaxThreads = nThreads;

  if(diff > 0)
    {
      std::cout << "====== FrameWriter starting " << diff << " worker threads...\n";
      mThreads.reserve(mMaxThreads);
      for(int i = mNumThreads; i < mMaxThreads; i++) { mThreads.emplace_back(std::bind(&FrameWriter::workerLoop, this, i)); }
      std::cout << "====== (done)\n";
    }
  else if(diff < 0)
    {
      diff = std::abs(diff);
      std::cout << "====== FrameWriter stopping " << diff << " worker threads (tid " << mMaxThreads << " - " << mActiveThreads  << ")...\n";
      pLock.unlock(); mPoolCv.notify_all();
      for(int i = mMaxThreads; i < mNumThreads; i++) { std::cout << "[waiting for T" << i << "]...\n"; mThreads[i].join(); } // join extra threads
      mThreads.erase(mThreads.begin()+mMaxThreads, mThreads.end());          // erase extra threads
      std::cout << "====== (done)\n";
    }
  mNumThreads = mThreads.size();
}

void FrameWriter::setBufferSize(int nFrames)
{
  if(!mNewFrames.size() > nFrames)
    {
      std::lock_guard<std::mutex>  fLock(mFlushLock); // blocks external get()/push() calls until excess frames are flushed
      std::unique_lock<std::mutex> pLock(mPoolLock);
      mMaxFrames = nFrames;
      while(mNewFrames.size() >= mMaxFrames)
        { // dispatch threads until complete
          pLock.unlock();
          std::this_thread::sleep_for(50ms);
          mPoolCv.notify_all();
          pLock.lock();
        }
    }
}


void FrameWriter::workerLoop(int tid)
{
  mActiveThreads++;
  std::cout << "==== [THREAD " << std::setw(2) << tid << " STARTED]\n";
  while(mActive)
    {
      FrameOut *frame = nullptr;
      {
        std::unique_lock<std::mutex> pLock(mPoolLock);
        if(!mNewFrames.empty()) // (no wait)
          { frame = mNewFrames.front(); mNewFrames.pop(); }
        else if(mActive && tid < mMaxThreads)
          { // wait for new source frame
            mPoolCv.wait(pLock, [tid, this]() { return (!mNewFrames.empty() || !mActive || tid >= mMaxThreads); });
            if(tid >= mMaxThreads) { break; } // number of threads reduced -- exit loop
            else if(!mNewFrames.empty())      // acquire new source frame
              { frame = mNewFrames.front(); mNewFrames.pop(); }
          }
        else { break; }
      }
      
      if(frame)
        {
          // process/write data
          process(frame, tid); writeToFile(frame, tid);
          { // recycle frame
            std::lock_guard<std::mutex> pLock(mPoolLock);
            mFramePool.push(frame);
          }
          mPushCv.notify_one(); // notify any blocked threads calling FrameWriter::get()
        }
    }
  std::cout << "===== [THREAD " << std::setw(2) << tid << " STOPPED]\n";
  mActiveThreads--;
}

FrameOut* FrameWriter::get()
{
  FrameOut *frame = nullptr;
  std::lock_guard<std::mutex>  fLock(mFlushLock); // NOTE: blocks while flushing
  std::unique_lock<std::mutex> pLock(mPoolLock);
  if(!mFramePool.empty()) // use recycled frame if available
    {
      frame = mFramePool.front(); mFramePool.pop();
      std::cout << "====== [GET() --> OLD frame (" << frame->size << " | allocated " << mNumFrames << "/" << mMaxFrames << ")]\n";
    }
  else if(mNumFrames < mMaxFrames)
    { // allocate new frame
      frame = new FrameOut(); mNumFrames++;
      std::cout << "====== [GET() --> NEW frame (" << frame->size << " | allocated " << mNumFrames << "/" << mMaxFrames << ")]\n";
    }
  else
    { // wait until allocated frame is recycled
      std::cout << "====== [GET() --> Waiting for recycled frame... (allocated " << mNumFrames << "/" << mMaxFrames << ")]\n";
      while(mActive && mFramePool.empty())
        {
          auto now = std::chrono::system_clock::now();
          mPushCv.wait_until(pLock, now+50ms, [this](){ return !mFramePool.empty(); });
        }
      
      if(!mFramePool.empty())
        {
          frame = mFramePool.front(); mFramePool.pop();
          std::cout << "====== [GET() --> mFramePool (" << (long long)((void*)frame) << " / " << frame->size << ")]\n";
        }
    }
  if(frame) { frame->create(op->outSize, (op->writeAlpha ? 4 : 3)); }
  return frame;
}

void FrameWriter::push(FrameOut *frame)
{
  if(!frame) { return; }
  std::lock_guard<std::mutex> fLock(mFlushLock); // NOTE: blocks while flushing
  if(mActive)
    {
      {
        std::lock_guard<std::mutex> pLock(mPoolLock);
        while(!mNewFrames.empty() && mNewFrames.front()->f == frame->f)
          { // remove previous duplicate frames (overwritten anyway)
            mFramePool.push(mNewFrames.front()); mNewFrames.pop();
          }
        mNewFrames.push(frame);
        std::cout << "====== [Pushed frame (" << frame->f << ")]\n";
      }
      mPoolCv.notify_one();
    }
}

void FrameWriter::clear()
{
  std::lock_guard<std::mutex> fLock(mFlushLock); // NOTE: blocks while flushing
  std::lock_guard<std::mutex> pLock(mPoolLock);
  while(!mNewFrames.empty()) // recycle all queued frames
    { mFramePool.push(mNewFrames.front()); mNewFrames.pop(); }
}

void FrameWriter::flush()
{
  std::lock_guard<std::mutex>  fLock(mFlushLock); // blocks external get()/push() calls until flush is complete
  std::unique_lock<std::mutex> pLock(mPoolLock);
  while(!mNewFrames.empty())
    { // dispatch threads until complete
      pLock.unlock();
      std::this_thread::sleep_for(50ms);
      mPoolCv.notify_all();
      pLock.lock();
    }
}

void FrameWriter::process(FrameOut *frame, int tid)
{
  if(!frame) { return; }
  // process frame and write to file (TODO: improve data handling to minimize processing)
  int channels = (op->writeAlpha ? 4 : 3);
  for(int y = 0; y < frame->size.y; y++)
    { // flip y --> copy each row of fluid into opposite row of mTexData2
      std::copy(&frame->raw [y*frame->size.x*channels],                    // row y
                &frame->raw [(y+1)*frame->size.x*channels],                // row y+1
                &frame->data[(frame->size.y-1-y)*frame->size.x*channels]); // flipped row
    }
}

void FrameWriter::writeToFile(FrameOut *frame, int tid)
{
  if(!checkSimRenderPath() || !frame) { return; }
  // flip y --> copy each row of fluid into opposite row of mTexData2
  for(int y = 0; y < frame->size.y; y++)
    {
      std::copy(&frame->raw [y*frame->size.x*frame->channels],                    // row y
                &frame->raw [(y+1)*frame->size.x*frame->channels],                // row y+1
                &frame->data[(frame->size.y-1-y)*frame->size.x*frame->channels]); // flipped row
    }
  // write data to file
  std::stringstream ss; ss << "-" << std::setfill('0') << std::setw(op->frameDigits) << frame->f << std::setfill(' ');
  fs::path imagePath = mBaseDir / op->projectName / op->projectName;
  imagePath += ss.str() + op->extension;
  setPngCompression(op->pngCompress);
  
  std::cout << "====== ["; if(tid >= 0) { std::cout << "THREAD " << std::setw(2) << tid << " "; }
  std::cout << "Writing frame " << std::setw(op->frameDigits) << frame->f << " ==>" << imagePath << "]\n";
  writeTexture(imagePath, (const void*)frame->data, op->outSize, op->writeAlpha);
}

bool FrameWriter::checkBaseRenderPath()
{
  // check base render directory
  if(!fs::exists(mBaseDir) || !fs::is_directory(mBaseDir))
    {
      std::cout << "====== Creating base directory for rendered simulations (" << mBaseDir << ")...\n";
      if(fs::create_directory(mBaseDir)) { std::cout << "====== Successfully created directory\n"; }
      else                               { std::cout << "========> ERROR: Could not create directory\n";  return false; }
    }
  return true;
}

bool FrameWriter::checkSimRenderPath()
{
  if(!checkBaseRenderPath()) { return false; } // make sure base directory exists
  else
    { // check named image sub-directory for this simulation render
      mImageDir = mBaseDir / op->projectName;
      if(!fs::exists(mImageDir) || !fs::is_directory(mBaseDir))
        {
          std::cout << "====== Creating directory for simulation '" << op->projectName << "' --> (" << mImageDir << ")...\n";
          if(fs::create_directory(mImageDir)) { std::cout << "====== Successfully created directory\n"; }
          else                                { std::cout << "========> ERROR: Could not create directory\n"; return false; }
        }
    }
  return true;
}

