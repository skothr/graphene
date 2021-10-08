#include "field.cuh"

#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>

#include "physics.h"
#include "raytrace.cuh"
#include "vector-operators.h"
#include "cuda-tools.cuh"
#include "mathParser.hpp"

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8
#define BLOCKDIM_Z 8


//// FILL FIELD ////

// fills field with constant value
template<typename T> __global__ void fillFieldValue_k(Field<T> dst, T val)
{
  long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long iy = blockIdx.y*blockDim.y + threadIdx.y;
  long iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix >= 0 && iy >= 0 && ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      unsigned long i = dst.idx(ix, iy, iz);
      dst[i] = val;
    }
}

// fills field with constant value
template<typename T>
__global__ void fillFieldMaterial_k(Field<Material<T>> dst, CudaExpression<T> *dExprEp, CudaExpression<T> *dExprMu, CudaExpression<T> *dExprSig)
{
  long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long iy = blockIdx.y*blockDim.y + threadIdx.y;
  long iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix >= 0 && iy >= 0 && ix < dst.size.x && iy < dst.size.y && iz < dst.size.z && dExprEp && dExprMu && dExprSig)
    {
      unsigned long i = dst.idx(ix, iy, iz);
      Material<T> M = dst[i];
      
      // allowed expression variables
      float3 s = float3{(float)dst.size.x, (float)dst.size.y, (float)dst.size.z}; // (s --> size)
      float3 p = float3{(float)ix, (float)iy, (float)iz};                         // (p --> position)
      float3 c = p - s/2.0;              // offset from field center
      float3 n = normalize(c);           // unit vector from field center
      float  r = length(c);              // (r --> radius) distance from field center
      float  t = (float)atan2(n.y, n.x); // (t --> theta)  angle measured from field center

      // run expression
      const int nVars = 8;
      T vars[nVars]; // {"px", "py", "pz", "sx", "sy", "sz", "r", "t"}
      vars[0] = p.x; vars[1] = p.y; vars[2] = p.z; // "px" / "py" / "pz"
      vars[3] = s.x; vars[4] = s.y; vars[5] = s.z; // "sx" / "sy" / "sz"
      vars[6] = r;   vars[7] = t;                  // "r" / "t"
      
      M.ep   = dExprEp->calculate(vars);
      M.mu   = dExprMu->calculate(vars);
      M.sig  = dExprSig->calculate(vars);
      M.nonVacuum = true;
      dst[i] = M;
    }
}

// fills field via given math expression
template<typename T> __global__ void fillField_k(Field<T> dst, CudaExpression<T> *expr);
template<typename T> __global__ void fillFieldChannel_k(Field<T> dst, CudaExpression<typename Dim<T>::BASE_T> *expr, int channel=-1);
template<> __global__ void fillField_k<float>(Field<float> dst, CudaExpression<float> *expr)
{
  unsigned long ix = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned long iy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned long iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      unsigned long i = dst.idx(ix, iy, iz);

      // allowed expression variables
      float3 s = float3{(float)dst.size.x, (float)dst.size.y, (float)dst.size.z}; // (s --> size)
      float3 p = float3{(float)ix, (float)iy, (float)iz};                         // (p --> position)
      float3 c = p - s/2.0;              // offset from field center
      float3 n = normalize(c);           // unit vector from field center
      float  r = length(c);              // (r --> radius) distance from field center
      float  t = (float)atan2(n.y, n.x); // (t --> theta)  angle measured from field center

      // run expression
      const int nVars = 8;
      float vars[nVars]; // {"px", "py", "pz", "sx", "sy", "sz", "r", "t"}
      vars[0] = p.x; vars[1] = p.y; vars[2] = p.z; // "px" / "py" / "pz"
      vars[3] = s.x; vars[4] = s.y; vars[5] = s.z; // "sx" / "sy" / "sz"
      vars[6] = r;   vars[7] = t;                  // "r" / "t"
      dst[i] = expr->calculate(vars);
    }
}

template<> __global__ void fillField_k<float3>(Field<float3> dst, CudaExpression<float3> *expr)
{
  long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long iy = blockIdx.y*blockDim.y + threadIdx.y;
  long iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      unsigned long i = dst.idx(ix, iy,iz);

      // allowed expression variables
      float3 s    = makeV<float3>(dst.size); //float3{(float)dst.size.x, (float)dst.size.y, 1.0}; // (s --> size)
      float3 p    = float3{(float)ix, (float)iy, (float)iz};  // (p --> position)
      float3 c    = p - s/2.0;       // offset from field center
      float3 n    = normalize(c);    // unit vector from field center
      float  r    = length(c);       // (r --> radius) distance from field center
      float  t    = atan2(n.y, n.x); // (t --> theta)  angle measured from field center (cylindrical)
      
      const int nVars = 5;
      float3 vars[nVars]; // {"p", "s", "r", "n", "t"}
      vars[0] = p;               // "p" -- position from origin (field index 0)
      vars[1] = s;               // "s" -- size
      vars[2] = c;               // "r" -- radius (position from center)
      vars[3] = n;               // "n" -- normalized radius
      //vars[4] = float3{t, t, t}; // "t" -- theta from center
      vars[4] = float3{atan2(n.z, n.y),
                       atan2(n.x, n.z),
                       atan2(n.y, n.x)}; // t alternative?
      // calculate value
      dst[i] = expr->calculate(vars);
    }
}
// only set one component/channel of each cell (used for setting +/- charge in Q.x/y
template<> __global__ void fillFieldChannel_k<float2>(Field<float2> dst, CudaExpression<float> *expr, int channel)
{
  unsigned long ix = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned long iy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned long iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      unsigned long i = dst.idx(ix, iy, iz);
      // allowed expression variables
      float3 s    = float3{(float)dst.size.x, (float)dst.size.y, (float)dst.size.z}; // (s --> size)
      float3 p    = float3{(float)ix, (float)iy, (float)iz};           // (p --> position)
      float3 c    = p - s/2.0;                                         // offset from field center
      float3 n    = normalize(c);                                      // unit vector from field center
      float  r    = length(c);                                         // (r --> radius) distance from field center
      float  t    = atan2(n.y, n.x);                                   // (t --> theta)  angle measured from field center
      
      // run expression
      const int nVars = 8;
      float vars[nVars]; // {"px", "py", "pz", "sx", "sy", "sz", "r", "t"}
      vars[0] = p.x; vars[1] = p.y; vars[2] = p.z; // "px" / "py" / "pz"
      vars[3] = s.x; vars[4] = s.y; vars[5] = s.z; // "sx" / "sy" / "sz"
      vars[6] = r;   vars[7] = t;                  // "r" / "t"
      float val = expr->calculate(vars);

      // write to channel
      if     (channel == 0) { dst[i].x = val; }
      else if(channel == 1) { dst[i].y = val; }
      else                  { dst[i] = float2{val, val}; }
    }
}

template<typename T>
void fillFieldValue(Field<T> &dst, const T &val)
{
  if(dst.size.x > 0 && dst.size.y > 0 && dst.size.z > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      fillFieldValue_k<<<grid, threads>>>(dst, val);
    }
  else { std::cout << "Skipped Field<float> fill --> " << dst.size << " --> " << val << "\n"; }
}

// wrapppers
template<typename T>
void fillField(Field<T> &dst, CudaExpression<T> *dExpr)
{
  if(dst.size.x > 0 && dst.size.y > 0 && dst.size.z > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      if(dExpr) { fillField_k<<<grid, threads>>>(dst, dExpr); }
      else { std::cout << "====> WARNING: fillField skipped -- null expression pointer ("
                       << "dExpr: "<< (long)((void*)dExpr) << ")\n"; }
    }
  else { std::cout << "Skipped Field<float> fill --> " << dst.size << " --> " << (long)((void*)dExpr) << "\n"; }
}

template<typename T>
void fillFieldMaterial<T>(Field<Material<T>> &dst, CudaExpression<T> *dExprEp, CudaExpression<T> *dExprMu, CudaExpression<T> *dExprSig)
{
  if(dst.size.x > 0 && dst.size.y > 0 && dst.size.z > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      if(dExprEp && dExprMu && dExprSig) { fillFieldMaterial_k<<<grid, threads>>>(dst, dExprEp, dExprMu, dExprSig); }
      else
        {
          std::cout << "====> WARNING: fillFieldMateral skipped -- null expression pointer ("
                    << "ε: "<< (long)((void*)dExprEp) << "|μ: " << (long)((void*)dExprMu) << "|σ: " << (long)((void*)dExprSig) << ")\n";
        }
    }
  else { std::cout << "Skipped Field<float> Material fill --> " << dst.size << " \n"; }
}
template<typename T>
void fillFieldChannel(Field<T> &dst, CudaExpression<typename Dim<T>::BASE_T> *dExpr, int channel)
{
  if(dst.size.x > 0 && dst.size.y > 0 && dst.size.z > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      if(dExpr) { fillFieldChannel_k<<<grid, threads>>>(dst, dExpr, channel); }
    }
  else { std::cout << "Skipped Field<float> fill --> " << dst.size << " \n"; }
}

// template instantiation
template void fillFieldMaterial<float >         (Field<Material<float>> &dst,
                                                 CudaExpression<float> *dExprEp, CudaExpression<float> *dExprMu, CudaExpression<float> *dExprSig);
template void fillFieldValue   <Material<float>>(Field<Material<float>> &dst, const Material<float> &val);
template void fillField        <float >         (Field<float>  &dst, CudaExpression<float > *dExpr);
template void fillField        <float3>         (Field<float3> &dst, CudaExpression<float3> *dExpr);
template void fillFieldChannel <float2>         (Field<float2> &dst, CudaExpression<float>  *dExpr, int channel);



