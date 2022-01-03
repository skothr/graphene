#ifndef FPS_COUNTER
#define FPS_COUNTER

#include <chrono>
#include <limits>

#define FPS_DEFAULT_INTERVAL 0.1

template<typename T, typename CLOCK_T>
struct FpsCounter
{
public:
  typedef CLOCK_T::time_point      time_t;
  typedef std::chrono::duration<T> duration_t; // (T seconds)
  
private:  
  T       interval;     // seconds over which to average fps
  T       dtAcc = T(0); // dt accumulator
  time_t  tLast;        // previous time
  int     nFrames = 0;  // number of frames over previous interval
  
public:
  T fps = T(0); // most recent FPS value
  
  FpsCounter(T interval_=0.1) : interval(interval_) { }

  // update --> accumulate time passed, and recalculate fps if acuumulated interval has passed
  // void update(const time_t &t=CLOCK_T::now());
  //   if maxFPS >= 0: returns false if frame should be skipped (fps > maxFPS)
  bool update(const time_t &t=CLOCK_T::now(), T maxFPS=T(-1));
};

template<typename T, typename CLOCK_T>
bool FpsCounter<T, CLOCK_T>::update(const time_t &t, T maxFPS)
{
  const T maxFrameTime = (maxFPS >= 0 ? (T)(1 / maxFPS) : (T)2); // maximum time for one frame
  
  duration_t dt = t - tLast;
  tLast = t;
  dtAcc += dt.count();
  
  int frames = (maxFPS >= 0 ? std::floor(dtAcc/maxFrameTime) : 1);
  nFrames += frames;
  if(dtAcc >= std::max(interval, maxFrameTime))
    {
      fps     = T(nFrames) / dtAcc;
      dtAcc   = 0.0;
      nFrames = 0;
    }
  
  return (frames > 0);
}

#endif // FPS_COUNTER
