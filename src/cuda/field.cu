#include "field.hpp"

#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>

#include "physics.h"
#include "raytrace.cuh"
#include "vector-operators.h"
#include "cuda-tools.cuh"
#include "mathParser.hpp"

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16
#define BLOCKDIM_Z 1


//// FILL FIELD ////

// fills field with constant value
template<typename T> __global__ void fieldFillValue_k(Field<T> dst, T val)
{
  long long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long long iy = blockIdx.y*blockDim.y + threadIdx.y;
  long long iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix >= 0 && iy >= 0 && ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      unsigned long long i = dst.idx(ix, iy, iz);
      dst[i] = val;
    }
}
// fills field via given math expression
template<typename T> __global__ void fillField_k(Field<T> dst, CudaExpression<T> *expr);
template<typename T> __global__ void fillFieldChannel_k(Field<T> dst, CudaExpression<typename Dims<T>::BASE> *expr, int channel=-1);
template<> __global__ void fillField_k<float>(Field<float> dst, CudaExpression<float> *expr)
{
  unsigned long long ix = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned long long iy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned long long iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      unsigned long long i = dst.idx(ix, iy, iz);

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
  long long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long long iy = blockIdx.y*blockDim.y + threadIdx.y;
  long long iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      unsigned long long i = dst.idx(ix, iy,iz);

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
      vars[4] = float3{atan2(n.x, n.y),
                       atan2(n.y, n.z),
                       atan2(n.z, n.x)}; // t alternative?
      
      
      // calculate value
      dst[i] = expr->calculate(vars);
    }
}
// only set one component/channel of each cell (used for setting +/- charge in Q.x/y
template<> __global__ void fillFieldChannel_k<float2>(Field<float2> dst, CudaExpression<float> *expr, int channel)
{
  unsigned long long ix = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned long long iy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned long long iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      unsigned long long i = dst.idx(ix, iy, iz);
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

// wrapppers
template<typename T>
void fieldFillValue(Field<T> &dst, const T &val)
{
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      fieldFillValue_k<<<grid, threads>>>(dst, val);
      getLastCudaError("====> ERROR: fieldFillValue_k failed!");
    }
  else { std::cout << "Skipped Field<float> fill --> " << dst.size << " --> " << val << "\n"; }
}
template<typename T>
void fieldFill(Field<T> &dst, CudaExpression<T> *dExpr)
{
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      if(dExpr) { fillField_k<<<grid, threads>>>(dst, dExpr); }
      getLastCudaError("====> ERROR: fillField_k failed!");
    }
  else { std::cout << "Skipped Field<float> fill --> " << dst.size << " \n"; }
}
template<typename T>
void fieldFillChannel(Field<T> &dst, CudaExpression<typename Dims<T>::BASE> *dExpr, int channel)
{
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      if(dExpr) { fillFieldChannel_k<<<grid, threads>>>(dst, dExpr, channel); }
      getLastCudaError("====> ERROR: fillFieldChannel_k failed!");
    }
  else { std::cout << "Skipped Field<float> fill --> " << dst.size << " \n"; }
}

// template instantiation
template void fieldFillValue  <Material<float>>(Field<Material<float>> &dst, const Material<float> &val);
template void fieldFill       <float>          (Field<float>  &dst, CudaExpression<float > *dExpr);
template void fieldFill       <float3>         (Field<float3> &dst, CudaExpression<float3> *dExpr);
template void fieldFillChannel<float2>         (Field<float2> &dst, CudaExpression<float>  *dExpr, int channel);





//// SIGNALS ////

// inline __device__ float3 pointInCircle(const float3 cp, const float3 lp, float cr) { }
// // returns intersections
// template<typename T> inline __device__ typename Dims<T, 4>::VECTOR_T
// lineIntersectCircle(T cRad, const typename Dims<T, 2>::VECTOR_T cp, const typename Dims<T, 2>::VECTOR_T lp1, const typename Dims<T, 2>::VECTOR_T lp2)
// {
//   using VT2 = typename Dims<T, 2>::VECTOR_T;
//   using VT2 = typename Dims<T, 4>::VECTOR_T;
//   VT2 lDiff = lp2 - lp1;
//   T lDist2  = lDiff.x*lDiff.x + lDiff.y*lDiff.y;
//   T lD      = lp1.x*lp2.y - lp2.x*lp1.y;
//   T discrim = cRad*cRad*lDist_2 - lD*lD;Q
//   if(discrim <= 0) { return 0.0f; }
// }


// add field containing signals
template<typename T> __global__ void addSignal_k(ChargeField<T> signal, ChargeField<T> dst, ChargeParams cp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      int i = dst.idx(ix, iy, iz);
      dst.Q[i]   += cp.dt*signal.Q[i];
      dst.QPV[i] += cp.dt*signal.QPV[i];
      dst.QNV[i] += cp.dt*signal.QNV[i];
      dst.E[i]   += cp.dt*signal.E[i];
      dst.B[i]   += cp.dt*signal.B[i];
    }
}

// draw in signals based on pen location and parameters
template<typename T> __global__ void addSignal_k(typename DimType<T, 3>::VECTOR_T pSrc, ChargeField<T> dst, SignalPen<T> pen, ChargeParams cp)
{
  typedef typename DimType<T, 2>::VECTOR_T VT2;
  typedef typename DimType<T, 3>::VECTOR_T VT3;
  long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long iy = blockIdx.y*blockDim.y + threadIdx.y;
  long iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      unsigned long long i = dst.idx(ix, iy, iz);
      VT3 pCell = VT3{(T)ix,(T)iy,(T)iz} + VT3{0.5, 0.5, 0.5};
      if(pen.cellAlign) { pCell = floor(pCell); pSrc = floor(pSrc); }
      
      VT3 diff  = (pCell - pSrc);// + (pen.cellAlign ? 1.0f : 0.0f);
      T   dist2 = dot(diff, diff);

      T rad = pen.radius;///sum(cp.cs/3);
      //T rMax = pen.radius + 1/(T)sqrt(2.0); // circle radius plus maximum possible intersection radius from cell (center to corner)
      //if(dist2 <= rMax*rMax)
      if((!pen.square && dist2 <= rad*rad) || (pen.square && abs(diff.x) <= rad+0.5f &&
                                                             abs(diff.y) <= rad+0.5f &&
                                                             abs(diff.z) <= rad+0.5f))
        {
          T dist = sqrt(dist2);
          VT3 n  = (dist == 0 ? diff : diff/dist);

          T rMult   = (dist  != 0.0f ? 1.0f / dist  : 1.0f);
          T r2Mult  = (dist2 != 0.0f ? 1.0f / dist2 : 1.0f);
          T cosMult = cos(2.0f*M_PI*pen.frequency*cp.t);
          T sinMult = sin(2.0f*M_PI*pen.frequency*cp.t);
          T tMult   = atan2(n.y, n.x);

          T   Qmult   =   pen.mult*((pen.Qopt   & IDX_R   ? rMult   : 1)*(pen.Qopt   & IDX_R2  ? r2Mult  : 1)*(pen.Qopt   & IDX_T ? tMult : 1) *
                                    (pen.Qopt   & IDX_COS ? cosMult : 1)*(pen.Qopt   & IDX_SIN ? sinMult : 1));
          VT3 QPVmult = n*pen.mult*((pen.QPVopt & IDX_R   ? rMult   : 1)*(pen.QPVopt & IDX_R2  ? r2Mult  : 1)*(pen.QPVopt & IDX_T ? tMult : 1) *
                                    (pen.QPVopt & IDX_COS ? cosMult : 1)*(pen.QPVopt & IDX_SIN ? sinMult : 1));
          VT3 QNVmult = n*pen.mult*((pen.QNVopt & IDX_R   ? rMult   : 1)*(pen.QNVopt & IDX_R2  ? r2Mult  : 1)*(pen.QNVopt & IDX_T ? tMult : 1) *
                                    (pen.QNVopt & IDX_COS ? cosMult : 1)*(pen.QNVopt & IDX_SIN ? sinMult : 1));
          T   Emult   =   pen.mult*((pen.Eopt   & IDX_R   ? rMult   : 1)*(pen.Eopt   & IDX_R2  ? r2Mult  : 1)*(pen.Eopt   & IDX_T ? tMult : 1) *
                                    (pen.Eopt   & IDX_COS ? cosMult : 1)*(pen.Eopt   & IDX_SIN ? sinMult : 1));
          T   Bmult   =   pen.mult*((pen.Bopt   & IDX_R   ? rMult   : 1)*(pen.Bopt   & IDX_R2  ? r2Mult  : 1)*(pen.Bopt   & IDX_T ? tMult : 1) *
                                    (pen.Bopt   & IDX_COS ? cosMult : 1)*(pen.Bopt   & IDX_SIN ? sinMult : 1));

          dst.Q  [i] += pen.Q   * Qmult   * cp.dt;
          dst.QPV[i] += pen.QPV * QPVmult * cp.dt / cp.cs;
          dst.QNV[i] += pen.QNV * QNVmult * cp.dt / cp.cs;
          dst.E  [i] += pen.E   * Emult   * cp.dt / cp.cs;
          dst.B  [i] += pen.B   * Bmult   * cp.dt / cp.cs;
        }
    }
}

// wrappers
template<typename T> void addSignal(ChargeField<T> &signal, ChargeField<T> &dst, const ChargeParams &cp)
{
  if(dst.size.x > 0 && dst.size.y > 0 && signal.size == dst.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      addSignal_k<<<grid, threads>>>(signal, dst, cp);
      getLastCudaError("====> ERROR: addSignal_k failed!");
    }
  else { std::cout << "==> WARNING: Skipped addSignal(srcField) (" << signal.size << " / " << dst.size << ")\n"; }
}
template<typename T> void addSignal(const typename DimType<T,3>::VECTOR_T &pSrc, ChargeField<T> &dst, const SignalPen<T> &pen, const ChargeParams &cp)
{
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      addSignal_k<<<grid, threads>>>(pSrc, dst, pen, cp);
      getLastCudaError("====> ERROR: addSignal_k failed!");
    }
  else { std::cout << "==> WARNING: Skipped addSignal(srcPoint) (" << dst.size << ")\n"; }
}

// template instantiation
template void addSignal<float>(ChargeField<float> &signal, ChargeField<float> &dst, const ChargeParams &cp);
template void addSignal<float>(const float3 &pSrc, ChargeField<float> &dst, const SignalPen<float> &pen, const ChargeParams &cp);




// MATERIAL
template<typename T> __global__ void addMaterial_k(typename DimType<T, 3>::VECTOR_T pSrc, ChargeField<T> dst, MaterialPen<T> pen, ChargeParams cp)
{
  typedef typename DimType<T, 2>::VECTOR_T VT2;
  typedef typename DimType<T, 3>::VECTOR_T VT3;
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < dst.size.x && iy < dst.size.y && iz < dst.size.z)
    {
      int i = dst.idx(ix, iy, iz);
      //T rMax = pen.radius + 1/(T)sqrt(2.0); // circle radius plus maximum possible intersection radius from cell (center to corner)

      //T   rad = pen.radius;// / sum(cp.cs/3);
      VT3 pCell = VT3{(T)ix+0.5f, (T)iy+0.5f, (T)iz+0.5f};
      VT3 diff  = pCell - pSrc;
      T   dist2 = dot(diff, diff);
      if((!pen.square && dist2 <= pen.radius*pen.radius) || (pen.square && (abs(diff.x) <= pen.radius+0.5f &&
                                                                            abs(diff.y) <= pen.radius+0.5f &&
                                                                            abs(diff.z) <= pen.radius+0.5f)))
        { dst.mat[i] = Material<T>(pen.mult*pen.permittivity, pen.mult*pen.permeability, pen.mult*pen.conductivity, pen.vacuum); }
    }
}







// wrapper functions
template<typename T>
void addMaterial(const typename DimType<T,3>::VECTOR_T &pSrc, ChargeField<T> &dst, const MaterialPen<T> &pen, const ChargeParams &cp)
{
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y),
                (int)ceil(dst.size.z/(float)BLOCKDIM_Z));
      addMaterial_k<<<grid, threads>>>(pSrc, dst, pen, cp);
      getLastCudaError("====> ERROR: addMaterial_k failed!");
    }
  else { std::cout << "==> WARNING: Skipped addMaterial(srcPoint) (" << dst.size << ")\n"; }
}

// template instantiation
template void addMaterial<float>(const float3 &pSrc, ChargeField<float> &dst, const MaterialPen<float> &pen, const ChargeParams &cp);








//// PHYSICS UPDATES ////

template<typename T>
__global__ void updateCharge_k(ChargeField<T> src, ChargeField<T> dst, ChargeParams cp)
{
  using VT2 = typename DimType<T, 2>::VECTOR_T;
  using VT3 = typename DimType<T, 3>::VECTOR_T;
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;

  if(ix < src.size.x && iy < src.size.y && iz < dst.size.z)
    {
      int3 p0 = int3{ix, iy, iz};
      int  i0 = src.idx(p0.x, p0.y, p0.z);
      // if(p0.x >= 0 && p0.x < src.size.x && p0.y >= 0 && p0.y < src.size.y)
      //   { atomicAdd(&tex[src.idx(p0], val); }

      // float3 lastP = p0 - cp.dt*
      // if(p0.x >= 0 && p0.x < s.x && p0.y >= 0 && p0.y < s.y)
      //   { atomicAdd(&tex[p0.y*s.x + p0.x], val); }

      int iPX = dst.Q.idx(ix+(ix<src.size.x-1?1:0), iy, iz);
      int iNX = dst.Q.idx(ix-(ix>0?1:0), iy, iz);
      int iPY = dst.Q.idx(ix, iy+(iy<src.size.y-1?1:0), iz);
      int iNY = dst.Q.idx(ix, iy-(iy>0?1:0), iz);
      int iPZ = dst.Q.idx(ix, iy, iz+(iz<src.size.z-1?1:0));
      int iNZ = dst.Q.idx(ix, iy, iz-(iz>0?1:0));

      VT2 Q0 = src.Q[i0];
      VT3 pv = src.QPV[i0];
      VT3 nv = src.QNV[i0];
      T   q0 = Q0.x - Q0.y;
      
      // update velocities based on charge gradient
      VT2 Q01 = VT2{0.0,0.0};
      if(iPX < src.size.x) { VT2 Q1 = src.Q[iPX];  Q01 -= (q0-(Q1.x-Q1.y)); } //pv.x += (q0-(Q1.x-Q1.y)*(Q1.x+Q1.y))*cp.dt;
      if(iNX >= 0)         { VT2 Q1 = src.Q[iNX];  Q01 -= (q0-(Q1.x-Q1.y)); } //nv.x += (q0-(Q1.x-Q1.y)*(Q1.x+Q1.y))*cp.dt;
      if(iPY < src.size.y) { VT2 Q1 = src.Q[iPY];  Q01 -= (q0-(Q1.x-Q1.y)); } //pv.y += (q0-(Q1.x-Q1.y)*(Q1.x+Q1.y))*cp.dt;
      if(iNY >= 0)         { VT2 Q1 = src.Q[iNY];  Q01 -= (q0-(Q1.x-Q1.y)); } //nv.y += (q0-(Q1.x-Q1.y)*(Q1.x+Q1.y))*cp.dt;
      if(iPZ < src.size.z) { VT2 Q1 = src.Q[iPZ];  Q01 -= (q0-(Q1.x-Q1.y)); } //pv.z += (q0-(Q1.x-Q1.y)*(Q1.x+Q1.y))*cp.dt;
      if(iNZ >= 0)         { VT2 Q1 = src.Q[iNZ];  Q01 -= (q0-(Q1.x-Q1.y)); } //nv.z += (q0-(Q1.x-Q1.y)*(Q1.x+Q1.y))*cp.dt;     
      
      dst.E[i0]   = src.E[i0];
      dst.B[i0]   = src.B[i0];
      dst.Q[i0]   = Q0 + Q01/4.0;
      dst.QPV[i0] = pv;
      dst.QNV[i0] = nv;
      dst.mat[i0] = src.mat[i0];
    }
}



//// SIMULATION -- MAXWELL'S EQUATIONS ////

// electric field E
template<typename T>
__global__ void updateElectric_k(ChargeField<T> src, ChargeField<T> dst, ChargeParams cp)
{
  using VT2 = typename DimType<T, 2>::VECTOR_T;
  using VT3 = typename DimType<T, 3>::VECTOR_T;
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;

  if(ix < src.size.x && iy < src.size.y && iz < dst.size.z)
    {
      int3 ip0 = int3{ix, iy, iz};
      int  i0  = src.idx(ix, iy, iz);
      VT3  p0  = makeV<VT3>(ip0)+VT3{0.5f, 0.5f, 0.5f};

      // check for boundary
      if(!cp.boundReflect)
        {
          int xOffset = src.size.x > 1 ? (ip0.x < 1 ? 1 : (ip0.x >= src.size.x-1 ? -1 : 0)) : 0;
          int yOffset = src.size.y > 1 ? (ip0.y < 1 ? 1 : (ip0.y >= src.size.y-1 ? -1 : 0)) : 0;
          int zOffset = src.size.z > 1 ? (ip0.z < 1 ? 1 : (ip0.z >= src.size.z-1 ? -1 : 0)) : 0;
          // int xOffset = (ip0.x < 1 ? 1 : (ip0.x >= src.size.x-1 ? -1 : 0));
          // int yOffset = (ip0.y < 1 ? 1 : (ip0.y >= src.size.y-1 ? -1 : 0));
          // int zOffset = (ip0.z < 1 ? 1 : (ip0.z >= src.size.z-1 ? -1 : 0));

          int3 p = ip0;
          if((xOffset != 0 && src.size.x > 1) || (yOffset != 0 && src.size.y > 1) || (zOffset != 0 && src.size.z > 1))
            {
              p += int3{xOffset, yOffset, zOffset};
              p.x = max(0, min(src.size.x-1, p.x));
              p.y = max(0, min(src.size.y-1, p.y));
              p.z = max(0, min(src.size.z-1, p.z));
              
              int i = src.idx(p.x, p.y, p.z);
              dst.E[i0]   = src.E[i];
              dst.B[i0]   = src.B[i0];
              dst.Q[i0]   = src.Q[i0];
              dst.QPV[i0] = src.QPV[i0];
              dst.QNV[i0] = src.QNV[i0];
              dst.mat[i0] = src.mat[i0];
              return;
            }
        }
      else
        {
          ip0.x = min(src.size.x-(int)cp.bs.x, max((int)cp.bs.x-1, ip0.x));
          ip0.y = min(src.size.y-(int)cp.bs.y, max((int)cp.bs.y-1, ip0.y));
          ip0.z = min(src.size.z-(int)cp.bs.z, max((int)cp.bs.z-1, ip0.z));
          ip0 = int3{ max(0, min(src.size.x-1, ip0.x)),
                      max(0, min(src.size.y-1, ip0.y)),
                      max(0, min(src.size.z-1, ip0.z))};
        }
      
      int3 ip1 = int3{min(src.size.x-1, max(0, ip0.x-1)), min(src.size.y-1, max(0, ip0.y-1)), min(src.size.z-1, max(0, ip0.z-1))};
      int i = src.E.idx(ip0.x, ip0.y, ip0.z);
      VT3 E00 = src.E[i];
      VT3 B00 = src.B[i];
      
      VT2 Q00         = src.Q[i0];
      VT3 QPV00       = src.QPV[i0];
      VT3 QNV00       = src.QNV[i0];
      Material<T> M00 = src.mat[i0];
      if(M00.vacuum()) { M00 = cp.material; } // check if vacuum
      
      VT3 Bxn = src.B[src.B.idx(ip1.x, ip0.y, ip0.z)];
      VT3 Byn = src.B[src.B.idx(ip0.x, ip1.y, ip0.z)];
      VT3 Bzn = src.B[src.B.idx(ip0.x, ip0.y, ip1.z)];

      VT3 dS = float3{cp.cs.y, cp.cs.y, cp.cs.y}*((src.size.x > 1 && src.size.y > 1 && src.size.y > 1) ? cp.cs : VT3{1.0, 1.0, 1.0});
      
      VT3 J = (QPV00 - QNV00)*(Q00.x - Q00.y) / dS / cp.dt;
      
      Material<T> mat = src.mat[i];
      if(mat.vacuum()) { mat = cp.material; } // check if vacuum (use user-defined properties in ChargeCp)
      typename Material<T>::Factors f = mat.getFactors(cp.dt, cp.cs.x);

      VT3 dEdt = VT3{  (B00.z-Byn.z) - (B00.y-Bzn.y),   // dBz/dY - dBy/dZ
                       (B00.x-Bzn.x) - (B00.z-Bxn.z),   // dBx/dZ - dBz/dX
                       (B00.y-Bxn.y) - (B00.x-Byn.x) }; // dBy/dX - dBx/dY
      dEdt -= J/mat.permittivity;
      
      VT3 newE = f.alphaE*E00 + f.betaE*dEdt;

      // ?
      //dEdt += cp.dt*coulombForce(-1.0f, 1.0f/T(M_Ke), makeV<VT3>(ip0 - src.size/4)/makeV<VT3>(src.size));
      //newE += pforce;//*(src.Q[i0].x - src.Q[i0].y);

      
      if(isnan(newE.x) || isinf(newE.x) || isnan(newE.y) || isinf(newE.y) || isnan(newE.z) || isinf(newE.z)) { newE = normalize(E00);   }
      if(isnan(newE.x) || isinf(newE.x) || isnan(newE.y) || isinf(newE.y) || isnan(newE.z) || isinf(newE.z)) { newE = VT3{0.0,0.0,0.0}; }
      
      // lorentz (E)
      VT3 newQPV = QPV00 + (Q00.x-Q00.y)*newE*cp.dt;
      if(isnan(newQPV.x) || isinf(newQPV.x) || isnan(newQPV.y) || isinf(newQPV.y) || isnan(newQPV.z) || isinf(newQPV.z)) { newQPV = normalize(QPV00); }
      if(isnan(newQPV.x) || isinf(newQPV.x) || isnan(newQPV.y) || isinf(newQPV.y) || isnan(newQPV.z) || isinf(newQPV.z)) { newQPV = VT3{0.0,0.0,0.0}; }
      VT3 newQNV = QNV00 - (Q00.x-Q00.y)*newE*cp.dt;
      if(isnan(newQNV.x) || isinf(newQNV.x) || isnan(newQNV.y) || isinf(newQNV.y) || isnan(newQNV.z) || isinf(newQNV.z)) { newQNV = normalize(QNV00); }
      if(isnan(newQNV.x) || isinf(newQNV.x) || isnan(newQNV.y) || isinf(newQNV.y) || isnan(newQNV.z) || isinf(newQNV.z)) { newQNV = VT3{0.0,0.0,0.0}; }
      // VT3 newQPV = src.QPV[i0]; VT3 newQNV = src.QNV[i0];
      
      dst.E[i0]   = newE;
      dst.QPV[i0] = newQPV;
      dst.QNV[i0] = newQNV;

      dst.Q[i0]   = Q00;
      dst.B[i0]   = B00;
      dst.mat[i0] = M00;
    }
}

// magnetic field B
template<typename T>
__global__ void updateMagnetic_k(ChargeField<T> src, ChargeField<T> dst, ChargeParams cp)
{
  using VT2 = typename DimType<T, 2>::VECTOR_T;
  using VT3 = typename DimType<T, 3>::VECTOR_T;
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;

  if(ix < src.size.x && iy < src.size.y && iz < dst.size.z)
    {
      int3  ip0 = int3{ix, iy, iz};
      int    i0 = src.idx(ix, iy, iz);
      VT3    p0  = makeV<VT3>(ip0)+VT3{0.5f, 0.5f, 0.5f};

      // check for boundary
      if(!cp.boundReflect)
        {
          int xOffset = src.size.x > 1 ? (ip0.x < 1 ? 1 : (ip0.x >= src.size.x-1 ? -1 : 0)) : 0;
          int yOffset = src.size.y > 1 ? (ip0.y < 1 ? 1 : (ip0.y >= src.size.y-1 ? -1 : 0)) : 0;
          int zOffset = src.size.z > 1 ? (ip0.z < 1 ? 1 : (ip0.z >= src.size.z-1 ? -1 : 0)) : 0;
          // int xOffset = (ip0.x < 1 ? 1 : (ip0.x >= src.size.x-1 ? -1 : 0));
          // int yOffset = (ip0.y < 1 ? 1 : (ip0.y >= src.size.y-1 ? -1 : 0));
          // int zOffset = (ip0.z < 1 ? 1 : (ip0.z >= src.size.z-1 ? -1 : 0));

          int3 p = ip0;
          if((xOffset != 0 && src.size.x > 1) || (yOffset != 0 && src.size.y > 1) || (zOffset != 0 && src.size.z > 1))
            {
              p += int3{xOffset, yOffset, zOffset};
              p.x = max(0, min(src.size.x-1, p.x));
              p.y = max(0, min(src.size.y-1, p.y));
              p.z = max(0, min(src.size.z-1, p.z));
              
              int i = src.idx(p.x, p.y, p.z);
              dst.B[i0] = src.B[i];
              dst.E[i0]   = src.E[i0];
              dst.Q[i0]   = src.Q[i0];
              dst.QPV[i0] = src.QPV[i0];
              dst.QNV[i0] = src.QNV[i0];
              dst.mat[i0] = src.mat[i0];
              return;
            }
        }
      else
        {
          // ip0.x = min(src.size.x-1-(int)cp.bs.x, max((int)cp.bs.x, ip0.x));
          // ip0.y = min(src.size.y-1-(int)cp.bs.y, max((int)cp.bs.y, ip0.y));
          // ip0.z = min(src.size.z-1-(int)cp.bs.z, max((int)cp.bs.z, ip0.z));
          ip0.x = max((int)cp.bs.x, min(src.size.x-(int)cp.bs.x-1, ip0.x));
          ip0.y = max((int)cp.bs.y, min(src.size.y-(int)cp.bs.y-1, ip0.y));
          ip0.z = max((int)cp.bs.z, min(src.size.z-(int)cp.bs.z-1, ip0.z));
          ip0 = int3{ max(0, min(src.size.x-1, ip0.x)),
                      max(0, min(src.size.y-1, ip0.y)),
                      max(0, min(src.size.z-1, ip0.z))};
        }
      
      int3 ip1 = int3{max(0, min(src.size.x-1, ip0.x+1)),
                      max(0, min(src.size.y-1, ip0.y+1)),
                      max(0, min(src.size.z-1, ip0.z+1)) };
      int i = src.B.idx(ip0.x, ip0.y, ip0.z);
      
      VT3 B00 = src.B[i];
      VT3 E00 = src.E[i];
      VT2 Q00         = src.Q[i0];
      VT3 QPV00       = src.QPV[i0];
      VT3 QNV00       = src.QNV[i0];
      Material<T> M00 = src.mat[i0];
      if(M00.vacuum()) { M00 = cp.material; } // check if vacuum

      VT3 Exp = src.E[src.E.idx(ip1.x, ip0.y, ip0.z)];
      VT3 Eyp = src.E[src.E.idx(ip0.x, ip1.y, ip0.z)];
      VT3 Ezp = src.E[src.E.idx(ip0.x, ip0.y, ip1.z)];

      typename Material<T>::Factors f = M00.getFactors(cp.dt, cp.cs.x);
      VT3 dBdt = VT3{  (Eyp.z-E00.z) - (Ezp.y-E00.y),   // dEz/dY - dEy/dZ
                       (Ezp.x-E00.x) - (Exp.z-E00.z),   // dEx/dZ - dEz/dX
                       (Exp.y-E00.y) - (Eyp.x-E00.x) }; // dEy/dX - dEx/dY
      VT3 newB = f.alphaB*B00 - f.betaB*dBdt;
      if(isnan(newB.x) || isinf(newB.x) || isnan(newB.y) || isinf(newB.y) || isnan(newB.z) || isinf(newB.z)) { newB = normalize(B00);   }
      if(isnan(newB.x) || isinf(newB.x) || isnan(newB.y) || isinf(newB.y) || isnan(newB.z) || isinf(newB.z)) { newB = VT3{0.0,0.0,0.0}; }
      
      // lorentz (v x B)
      VT3 newQPV = QPV00 + (Q00.x-Q00.y)*cross(QPV00, newB)*cp.dt;
      if(isnan(newQPV.x) || isinf(newQPV.x) || isnan(newQPV.y) || isinf(newQPV.y) || isnan(newQPV.z) || isinf(newQPV.z)) { newQPV = normalize(QPV00); }
      if(isnan(newQPV.x) || isinf(newQPV.x) || isnan(newQPV.y) || isinf(newQPV.y) || isnan(newQPV.z) || isinf(newQPV.z)) { newQPV = VT3{0.0,0.0,0.0}; }
      VT3 newQNV = QNV00 - (Q00.x-Q00.y)*cross(QNV00, newB)*cp.dt;
      if(isnan(newQNV.x) || isinf(newQNV.x) || isnan(newQNV.y) || isinf(newQNV.y) || isnan(newQNV.z) || isinf(newQNV.z)) { newQNV = normalize(QNV00); }
      if(isnan(newQNV.x) || isinf(newQNV.x) || isnan(newQNV.y) || isinf(newQNV.y) || isnan(newQNV.z) || isinf(newQNV.z)) { newQNV = VT3{0.0,0.0,0.0}; }
      // VT3 newQPV = src.QPV[i0]; VT3 newQNV = src.QNV[i0];
        
      dst.B[i0]   = newB;
      dst.QPV[i0] = newQPV;
      dst.QNV[i0] = newQNV;
    
      dst.Q[i0]   = Q00;
      dst.E[i0]   = E00;
      dst.mat[i0] = M00;
    }
}

// wrappers
template<typename T> void updateCharge(ChargeField<T> &src, ChargeField<T> &dst, ChargeParams &cp)
{
  if(src.size.x > 0 && src.size.y > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      //dst.Q.clear(); // advection via atomic writes
      updateCharge_k<<<grid, threads>>>(src, dst, cp);
      getLastCudaError("====> ERROR: updateCharge_k failed!");
    }
  else { std::cout << "==> WARNING: Skipped updateCharge (" << src.size << " / " << dst.size << ")\n"; }
}
template<typename T> void updateElectric(ChargeField<T> &src, ChargeField<T> &dst, ChargeParams &cp)
{
  if(src.size.x > 0 && src.size.y > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      updateElectric_k<<<grid, threads>>>(src, dst, cp);
      getLastCudaError("====> ERROR: updateElectric_k failed!");
    }
  else { std::cout << "==> WARNING: Skipped updateElectric (" << src.size << " / " << dst.size << ")\n"; }
}
template<typename T> void updateMagnetic(ChargeField<T> &src, ChargeField<T> &dst, ChargeParams &cp)
{
  if(src.size.x > 0 && src.size.y > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      updateMagnetic_k<<<grid, threads>>>(src, dst, cp);
      getLastCudaError("====> ERROR: updateMagnetic2D_k failed!");
    }
  else { std::cout << "==> WARNING: Skipped updateMagnetic2D (src: " << src.size << " / dst: " << dst.size << ")\n"; }
}

// template instantiation
template void updateCharge  <float> (ChargeField<float> &src, ChargeField<float>  &dst, ChargeParams &cp);
template void updateElectric<float> (ChargeField<float> &src, ChargeField<float>  &dst, ChargeParams &cp);
template void updateMagnetic<float> (ChargeField<float> &src, ChargeField<float>  &dst, ChargeParams &cp);
// template void updateCharge  <double>(ChargeField<double> &src, ChargeField<double> &dst, ChargeParams &cp);
// template void updateElectric<double>(ChargeField<double> &src, ChargeField<double> &dst, ChargeParams &cp);











//// RENDERING ////

template<typename T>
__global__ void renderFieldEM_k(ChargeField<T> src, CudaTexture dst, EmRenderParams rp)
{
  long long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long long iy = blockIdx.y*blockDim.y + threadIdx.y;
  if(ix < dst.size.x && iy < dst.size.y)
    {
      int ti = ix + iy*dst.size.x;
      int2 fp = int2{(int)(ix*(src.size.x/(T)dst.size.x)),  // scale texture index to field index
                     (int)(iy*(src.size.y/(T)dst.size.y))};
      float bScale = float(src.size.z-1)/(rp.numLayers2D+1);
      float4 color = float4{0.0f, 0.0f, 0.0f, 0.0f};
      for(int i = min(src.size.z-1, rp.numLayers2D-1); i >= 0; i--)
        {
         int fi = src.idx(fp.x, fp.y, i);
         T qLen = (src.Q[fi].x - src.Q[fi].y); T eLen = length(src.E[fi]); T bLen = length(src.B[fi]);
         float4 col = rp.brightness*rp.opacity*(qLen*rp.Qmult*rp.Qcol + eLen*rp.Emult*rp.Ecol + bLen*rp.Bmult*rp.Bcol);
         col.x *= bScale; col.y *= bScale; col.z *= bScale;
         fluidBlend(color, col, rp);
         //if(color.x >= 1.0f || color.y >= 1.0f || color.z >= 1.0f) { break; }
        }
      float a = color.w;
      color += float4{BG_COLOR.x, BG_COLOR.y, BG_COLOR.z, 0.0} * BG_COLOR.w*(1-a*rp.brightness);
      color.w += BG_COLOR.w*(1-color.w)*(rp.opacity);
      dst[ti] += float4{ max(0.0f, min(1.0f, color.x)), max(0.0f, min(1.0f, color.y)), max(0.0f, min(1.0f, color.z)), 1.0f };
    }
}

template<typename T>
__global__ void renderFieldMat_k(Field<Material<T>> src, CudaTexture dst, EmRenderParams rp)
{
  long long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long long iy = blockIdx.y*blockDim.y + threadIdx.y;
  if(ix < dst.size.x && iy < dst.size.y)
    {
      int ti = ix + iy*dst.size.x;
      int fi = src.idx((int)(ix*(src.size.x/(T)dst.size.x)),  // scale texture index to field index
                       (int)(iy*(src.size.y/(T)dst.size.y)), 0);
      
      Material<T> mat = src[fi];
      float4 color = (mat.vacuum() ? float4{00.f, 0.0f, 0.0f, 1.0f} :
                      (mat.permittivity * rp.epMult  * rp.epCol +    // epsilon
                       mat.permeability * rp.muMult  * rp.muCol +    // mu
                       mat.conductivity * rp.sigMult * rp.sigCol )); // sigma
      dst[ti] += float4{ color.x, color.y, color.z, 1.0f };
    }
}


template<typename T>
__global__ void rtRenderFieldEM_k(ChargeField<T> src, CudaTexture dst, CameraDesc<double> cam, EmRenderParams rp, ChargeParams cp, double aspect)
{
  long long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long long iy = blockIdx.y*blockDim.y + threadIdx.y;
  if(ix < dst.size.x && iy < dst.size.y)
    {
      double tf2 = tan(cam.fov/2.0*(M_PI/180.0));
      Ray<double> ray;
      ray.pos = cam.pos;
      ray.dir = normalize(cam.dir +
                          cam.right * 2.0*tf2*(ix/double(dst.size.x)-0.5)*aspect +  //*double(2.0*ix/dst.size.x - 1.0)*aspect +
                          cam.up    * 2.0*tf2*(iy/double(dst.size.y)-0.5)); //*double(2.0*iy/dst.size.y - 1.0));
      float4 color = rayTraceField(src, ray, rp, cp);
      
      long long ti = ix + iy*dst.size.x;
      dst.dData[ti] = (color.w < 0.0f ? float4{0.0f, 0.0f, 0.0f, 1.0f} : color);
    }
}
template<typename T>
__global__ void rtRenderFieldMat_k(ChargeField<T> src, CudaTexture dst, CameraDesc<double> cam, EmRenderParams rp, ChargeParams cp, double aspect)
{
  long long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long long iy = blockIdx.y*blockDim.y + threadIdx.y;
  if(ix < dst.size.x && iy < dst.size.y)
    {
      double tf2 = tan(M_PI/180.0*cam.fov/2.0);
      Ray<double> ray;
      ray.pos = cam.pos;
      ray.dir = normalize(cam.dir +
                          cam.right * tf2*double((ix-dst.size.x/2.0)/dst.size.x)*aspect +
                          cam.up    * tf2*double((iy-dst.size.y/2.0)/dst.size.y));
      float4 color = rayTraceField(src, ray, rp, cp);
      
      long long ti = ix + iy*dst.size.x;
      dst.dData[ti] = (color.w < 0.0f ? float4{0.0f, 0.0f, 0.0f, 1.0f} : float4{color.x, color.y, color.z, 1.0f});
    }
}

// wrappers
template<typename T>
void renderFieldEM(ChargeField<T> &src, CudaTexture &dst, const EmRenderParams &rp)
{
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y)); // 2D -- thread texture pixels
      bool mapped = dst.mapped;
      if(!mapped) { dst.map(); }
      renderFieldEM_k<<<grid, threads>>>(src, dst, rp);
      // cudaDeviceSynchronize(); getLastCudaError("====> ERROR: renderFieldEM_k failed!");
      if(!mapped) { dst.unmap(); }
    }
  else { std::cout << "Skipped ChargeField render --> " << src.size << " / " << dst.size << " \n"; }
}
template<typename T>
void renderFieldMat(Field<Material<T>> &src, CudaTexture &dst, const EmRenderParams &rp)
{
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y)); // 2D -- thread texture pixels
      bool mapped = dst.mapped;
      if(!mapped) { dst.map(); }
      renderFieldMat_k<<<grid, threads>>>(src, dst, rp);
      // cudaDeviceSynchronize(); getLastCudaError("====> ERROR: renderFieldMat_k failed!");
      if(!mapped) { dst.unmap(); }
    }
  else { std::cout << "Skipped ChargeField render --> " << src.size << " / " << dst.size << " \n"; }
}

template<typename T>
void raytraceFieldEM(ChargeField<T> &src, CudaTexture &dst, const Camera<double> &camera, const EmRenderParams &rp, const ChargeParams &cp, double aspect)
{
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y)); // 2D -- thread texture pixels
      bool mapped = dst.mapped;
      if(!mapped) { dst.map(); }
      rtRenderFieldEM_k<<<grid, threads>>>(src, dst, camera.desc, rp, cp, aspect);
      // cudaDeviceSynchronize(); getLastCudaError("====> ERROR: raytraceFieldEM_k failed!");
      if(!mapped) { dst.unmap(); }
    }
  else { std::cout << "Skipped ChargeField render (RT) --> " << src.size << " / " << dst.size << " \n"; }
}

template<typename T>
void raytraceFieldMat(ChargeField<T> &src, CudaTexture &dst, const Camera<double> &camera, const EmRenderParams &rp, const ChargeParams &cp, double aspect)
{
  if(dst.size.x > 0 && dst.size.y > 0)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(dst.size.x/(float)BLOCKDIM_X),
                (int)ceil(dst.size.y/(float)BLOCKDIM_Y)); // 2D -- thread texture pixels
      bool mapped = dst.mapped;
      if(!mapped) { dst.map(); }
      rtRenderFieldMat_k<<<grid, threads>>>(src, dst, camera.desc, rp, cp, aspect);
      // cudaDeviceSynchronize(); getLastCudaError("====> ERROR: raytraceFieldEM_k failed!");
      if(!mapped) { dst.unmap(); }
    }
  else { std::cout << "Skipped ChargeField render (RT) --> " << src.size << " / " << dst.size << " \n"; }
}

// template instantiation
template void renderFieldEM   <float>(ChargeField<float>          &src, CudaTexture &dst, const EmRenderParams &rp);
template void renderFieldMat  <float>(Field<Material<float>> &src, CudaTexture &dst, const EmRenderParams &rp);
template void raytraceFieldEM <float>(ChargeField<float> &src, CudaTexture &dst, const Camera<double> &camera,
                                      const EmRenderParams &rp, const ChargeParams &cp, double aspect);
template void raytraceFieldMat<float>(ChargeField<float> &src, CudaTexture &dst, const Camera<double> &camera,
                                      const EmRenderParams &rp, const ChargeParams &cp, double aspect);
