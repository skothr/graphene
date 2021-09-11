#include "field.cuh"

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
      vars[0] = p;        // "p" -- position from origin (field index 0)
      vars[1] = s;        // "s" -- size
      vars[2] = c;        // "r" -- radius (position from center)
      vars[3] = n;        // "n" -- normalized radius
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
      cudaDeviceSynchronize(); getLastCudaError("====> ERROR: fieldFillValue_k failed!");
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
      cudaDeviceSynchronize(); getLastCudaError("====> ERROR: fillField_k failed!");
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
      cudaDeviceSynchronize(); getLastCudaError("====> ERROR: fillFieldChannel_k failed!");
    }
  else { std::cout << "Skipped Field<float> fill --> " << dst.size << " \n"; }
}

// template instantiation
template void fieldFillValue  <Material<float>>(Field<Material<float>> &dst, const Material<float> &val);
template void fieldFill       <float>          (Field<float>  &dst, CudaExpression<float > *dExpr);
template void fieldFill       <float3>         (Field<float3> &dst, CudaExpression<float3> *dExpr);
template void fieldFillChannel<float2>         (Field<float2> &dst, CudaExpression<float>  *dExpr, int channel);








//// PHYSICS UPDATES ////

template<typename T>
__global__ void updateCharge_k(EMField<T> src, EMField<T> dst, FieldParams<T> cp)
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

      VT3 E0  = src.E[i0];
      VT2 Q0  = VT2{src.QP[i0], src.QN[i0]};
      VT3 QVp0 = VT3{(ix < src.size.x-1 ? src.QVx[src.QVx.idx(ix, iy, iz)] : (T)0),
                     (iy < src.size.y-1 ? src.QVy[src.QVy.idx(ix, iy, iz)] : (T)0),
                     (iz < src.size.z-1 ? src.QVz[src.QVz.idx(ix, iy, iz)] : (T)0)};
      VT3 QVn0 = VT3{(ix > 0 ? src.QVx[src.QVx.idx(ix-1, iy, iz)] : (T)0),
                     (iy > 0 ? src.QVy[src.QVy.idx(ix, iy-1, iz)] : (T)0),
                     (iz > 0 ? src.QVz[src.QVz.idx(ix, iy, iz-1)] : (T)0)};
      Material<T> M0 = src.mat[i0]; if(M0.vacuum()) { M0 = cp.u.vacuum(); }
      
      //T   q0  = Q0.x - Q0.y; // total charge in cell (TODO: combine?)
      
      // // update velocities based on charge gradient
      VT2 newQ = Q0;
      // if(iPX < src.size.x) { T QV1 = src.Q[iPX]; newQ -= (q0-(Q1.x-Q1.y)); } //pv.x += (q0-(Q1.x-Q1.y)*(Q1.x+Q1.y))*cp.u.dt;
      // if(iNX >= 0)         { T QV1 = src.Q[iNX];  newQ -= (q0-(Q1.x-Q1.y)); } //nv.x += (q0-(Q1.x-Q1.y)*(Q1.x+Q1.y))*cp.u.dt;
      // if(iPY < src.size.y) { T QV1 = src.Q[iPY];  newQ -= (q0-(Q1.x-Q1.y)); } //pv.y += (q0-(Q1.x-Q1.y)*(Q1.x+Q1.y))*cp.u.dt;
      // if(iNY >= 0)         { T QV1 = src.Q[iNY];  newQ -= (q0-(Q1.x-Q1.y)); } //nv.y += (q0-(Q1.x-Q1.y)*(Q1.x+Q1.y))*cp.u.dt;
      // if(iPZ < src.size.z) { T QV1 = src.Q[iPZ];  newQ -= (q0-(Q1.x-Q1.y)); } //pv.z += (q0-(Q1.x-Q1.y)*(Q1.x+Q1.y))*cp.u.dt;
      // if(iNZ >= 0)         { T QV1 = src.Q[iNZ];  newQ -= (q0-(Q1.x-Q1.y)); } //nv.z += (q0-(Q1.x-Q1.y)*(Q1.x+Q1.y))*cp.u.dt;

      // TEST -- treat Q as fully derived from divergence of E (TODO: conserve?)
      int iPX = dst.E.idx(ix+(ix<src.size.x-1?1:0), iy, iz);
      int iNX = dst.E.idx(ix-(ix>0?1:0), iy, iz);
      int iPY = dst.E.idx(ix, iy+(iy<src.size.y-1?1:0), iz);
      int iNY = dst.E.idx(ix, iy-(iy>0?1:0), iz);
      int iPZ = dst.E.idx(ix, iy, iz+(iz<src.size.z-1?1:0));
      int iNZ = dst.E.idx(ix, iy, iz-(iz>0?1:0));
      VT3 gradE = VT3{0.0,0.0,0.0};
      if(ix < src.size.x-1) { gradE.x = src.E[iPX].x - E0.x;   }
      if(iy < src.size.y-1) { gradE.y = src.E[iPY].y - E0.y;   }
      if(iz < src.size.z-1) { gradE.z = src.E[iPZ].z - E0.z;   }

      newQ.x = sum(gradE) * cp.u.e0; // (~Gauss's law)
      //newQ.y = 0;
      
      dst.QP[i0]  = newQ.x;
      dst.QN[i0]  = newQ.y;
      if(ix < src.size.x-1) { dst.QVx[i0]  = QVp0.x; }
      if(iy < src.size.y-1) { dst.QVy[i0]  = QVp0.y; }
      if(iz < src.size.z-1) { dst.QVz[i0]  = QVp0.z; }
      dst.E[i0]   = E0;
      dst.B[i0]   = src.B[i0];
      dst.mat[i0] = src.mat[i0];
    }
}



//// SIMULATION -- MAXWELL'S EQUATIONS ////

// electric field E
template<typename T>
__global__ void updateElectric_k(EMField<T> src, EMField<T> dst, FieldParams<T> cp)
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

      // check for boundary (TODO: not working? -- reflects)
      if(!cp.reflect) 
        {
          const int bs = 2;
          int xOffset = (src.size.x != 2*bs ? ((ip0.x < bs ? 1 : 0) + (ip0.x+bs >= src.size.x ? -1 : 0)) : 0);
          int yOffset = (src.size.y != 2*bs ? ((ip0.y < bs ? 1 : 0) + (ip0.y+bs >= src.size.y ? -1 : 0)) : 0);
          int zOffset = (src.size.z != 2*bs ? ((ip0.z < bs ? 1 : 0) + (ip0.z+bs >= src.size.z ? -1 : 0)) : 0);
          if(xOffset != 0 || yOffset != 0 || zOffset != 0)
            {
              int i = src.idx(max(0, min(src.size.x-1, ip0.x + xOffset)),
                              max(0, min(src.size.y-1, ip0.y + yOffset)),
                              max(0, min(src.size.z-1, ip0.z + zOffset)));
              dst.E[i0]   = src.E[i];  // use updated index for E
              dst.B[i0]   = src.B[i0]; // just copy everything else
              dst.QP[i0]  = src.QP[i0];
              dst.QN[i0]  = src.QN[i0];
              dst.QVx[dst.QVx.idx(ix,iy,iz)] = src.QVx[dst.QVx.idx(ix,iy,iz)];
              dst.QVy[dst.QVy.idx(ix,iy,iz)] = src.QVy[dst.QVy.idx(ix,iy,iz)];
              dst.QVz[dst.QVz.idx(ix,iy,iz)] = src.QVz[dst.QVz.idx(ix,iy,iz)];
              dst.mat[i0] = src.mat[i0];
              return;
            }
        }
      
      VT3 E0 = src.E[i0];
      VT3 B0 = src.B[i0];
      
      VT2 Q0  = VT2{src.QP[i0], src.QN[i0]};
      VT3 QVp0 = VT3{(ix < src.size.x-1 ? src.QVx[src.QVx.idx(ix, iy, iz)] : (T)0),
                     (iy < src.size.y-1 ? src.QVy[src.QVy.idx(ix, iy, iz)] : (T)0),
                     (iz < src.size.z-1 ? src.QVz[src.QVz.idx(ix, iy, iz)] : (T)0)};
      VT3 QVn0 = VT3{(ix > 0 ? src.QVx[src.QVx.idx(ix-1, iy, iz)] : (T)0),
                     (iy > 0 ? src.QVy[src.QVy.idx(ix, iy-1, iz)] : (T)0),
                     (iz > 0 ? src.QVz[src.QVz.idx(ix, iy, iz-1)] : (T)0)};
      Material<T> M0 = src.mat[i0];
      if(M0.vacuum()) { M0 = cp.u.vacuum(); } // check if vacuum
      typename Material<T>::Factors f = M0.getFactors(cp.u.dt, cp.u.dL);
      
      int3 ip1  = int3{min(src.size.x-1, max(0, ip0.x-1)), min(src.size.y-1, max(0, ip0.y-1)), min(src.size.z-1, max(0, ip0.z-1))};
      VT3  Bxn  = src.B[src.B.idx(ip1.x, ip0.y, ip0.z)]; // -1 in x direction
      VT3  Byn  = src.B[src.B.idx(ip0.x, ip1.y, ip0.z)]; // -1 in y direction
      VT3  Bzn  = src.B[src.B.idx(ip0.x, ip0.y, ip1.z)]; // -1 in z direction
      VT3  dEdt = VT3{  (B0.z-Byn.z) - (B0.y-Bzn.y),   // dBz/dY - dBy/dZ
                        (B0.x-Bzn.x) - (B0.z-Bxn.z),   // dBx/dZ - dBz/dX
                        (B0.y-Bxn.y) - (B0.x-Byn.x) }; // dBy/dX - dBx/dY

      // apply effect of electric current (TODO: improve)
      VT3 dS = VT3{cp.u.dL*cp.u.dL, cp.u.dL*cp.u.dL, cp.u.dL*cp.u.dL};
      VT3 J = (QVp0-QVn0)*(Q0.x - Q0.y) / cp.u.dt / dS;
      dEdt -= J / M0.permittivity;
      
      VT3 newE = f.alphaE*E0 + f.betaE*dEdt;
      // TODO: solve for divergence?
      if(isnan(newE.x) || isinf(newE.x) || isnan(newE.y) || isinf(newE.y) || isnan(newE.z) || isinf(newE.z) ||
         abs(newE.x) > 1e24 ||abs(newE.y) > 1e24 ||abs(newE.z) > 1e24) { newE = VT3{0.0,0.0,0.0}; }
      
      // // lorentz (E)
      VT3 newQV = QVp0 + (Q0.x-Q0.y)*newE*cp.u.dt;
      if(isnan(newQV.x) || isinf(newQV.x) || isnan(newQV.y) || isinf(newQV.y) || isnan(newQV.z) || isinf(newQV.z)) { newQV = VT3{0.0,0.0,0.0}; }
      
      dst.E[i0]   = newE; // updated values
      if(ix > src.size.x) { dst.QVx[dst.QVx.idx(ix,iy,iz)] = newQV.x; }
      if(iy > src.size.y) { dst.QVy[dst.QVy.idx(ix,iy,iz)] = newQV.y; }
      if(iz > src.size.z) { dst.QVz[dst.QVz.idx(ix,iy,iz)] = newQV.z; }
    
      dst.QP[i0]  = Q0.x; dst.QN[i0] = Q0.y; // copied values (unchanged)
      
      dst.B[i0]   = B0;
      dst.mat[i0] = M0;
    }
}

// magnetic field B
template<typename T>
__global__ void updateMagnetic_k(EMField<T> src, EMField<T> dst, FieldParams<T> cp)
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

      // check for boundary (TODO: remove/fi)x
      if(!cp.reflect)
        {
          const int bs = 2;
          int xOffset = (src.size.x != 2*bs ? ((ip0.x < bs ? 1 : 0) + (ip0.x+bs >= src.size.x ? -1 : 0)) : 0);
          int yOffset = (src.size.y != 2*bs ? ((ip0.y < bs ? 1 : 0) + (ip0.y+bs >= src.size.y ? -1 : 0)) : 0);
          int zOffset = (src.size.z != 2*bs ? ((ip0.z < bs ? 1 : 0) + (ip0.z+bs >= src.size.z ? -1 : 0)) : 0);
          if(xOffset != 0 || yOffset != 0 || zOffset != 0)
            {
              int i = src.idx(max(0, min(src.size.x-1, ip0.x + xOffset)),
                              max(0, min(src.size.y-1, ip0.y + yOffset)),
                              max(0, min(src.size.z-1, ip0.z + zOffset)));
              dst.B[i0]   = src.B[i];  // use updated index for B
              dst.E[i0]   = src.E[i0]; // just copy everything else
              dst.QP[i0]  = src.QP[i0];
              dst.QN[i0]  = src.QN[i0];
              dst.QVx[dst.QVx.idx(ix,iy,iz)] = src.QVx[dst.QVx.idx(ix,iy,iz)];
              dst.QVy[dst.QVy.idx(ix,iy,iz)] = src.QVy[dst.QVy.idx(ix,iy,iz)];
              dst.QVz[dst.QVz.idx(ix,iy,iz)] = src.QVz[dst.QVz.idx(ix,iy,iz)];
              dst.mat[i0] = src.mat[i0];
              return;
            }
        }
      
      VT3 B0         = src.B[i0];
      VT3 E0         = src.E[i0];
      VT2 Q0  = VT2{src.QP[i0], src.QN[i0]};
      VT3 QVp0 = VT3{(ix < src.size.x-1 ? src.QVx[src.QVx.idx(ix, iy, iz)] : (T)0),
                     (iy < src.size.y-1 ? src.QVy[src.QVy.idx(ix, iy, iz)] : (T)0),
                     (iz < src.size.z-1 ? src.QVz[src.QVz.idx(ix, iy, iz)] : (T)0)};
      VT3 QVn0 = VT3{(ix > 0 ? src.QVx[src.QVx.idx(ix-1, iy, iz)] : (T)0),
                     (iy > 0 ? src.QVy[src.QVy.idx(ix, iy-1, iz)] : (T)0),
                     (iz > 0 ? src.QVz[src.QVz.idx(ix, iy, iz-1)] : (T)0)};
      Material<T> M0 = src.mat[i0];
      if(M0.vacuum()) { M0 = cp.u.vacuum(); } // check if vacuum
      typename Material<T>::Factors f = M0.getFactors(cp.u.dt, cp.u.dL);

      int3 ip1  = int3{max(0, min(src.size.x-1, ip0.x+1)), max(0, min(src.size.y-1, ip0.y+1)), max(0, min(src.size.z-1, ip0.z+1)) };
      VT3  Exp  = src.E[src.E.idx(ip1.x, ip0.y, ip0.z)]; // +1 in x direction
      VT3  Eyp  = src.E[src.E.idx(ip0.x, ip1.y, ip0.z)]; // +1 in y direction
      VT3  Ezp  = src.E[src.E.idx(ip0.x, ip0.y, ip1.z)]; // +1 in z direction
      VT3  dBdt = VT3{  (Eyp.z-E0.z) - (Ezp.y-E0.y),   // dEz/dY - dEy/dZ
                        (Ezp.x-E0.x) - (Exp.z-E0.z),   // dEx/dZ - dEz/dX
                        (Exp.y-E0.y) - (Eyp.x-E0.x) }; // dEy/dX - dEx/dY
      VT3 newB = f.alphaB*B0 - f.betaB*dBdt;
      if(isnan(newB.x) || isinf(newB.x) || isnan(newB.y) || isinf(newB.y) || isnan(newB.z) || isinf(newB.z) ||
         abs(newB.x) > 1e24 ||abs(newB.y) > 1e24 ||abs(newB.z) > 1e24) { newB = VT3{0.0,0.0,0.0}; }
      
      // // lorentz (v x B)
      VT3 newQV = QVp0 + (Q0.x-Q0.y)*cross(QVp0, newB)*cp.u.dt;
      if(isnan(newQV.x) || isinf(newQV.x) || isnan(newQV.y) || isinf(newQV.y) || isnan(newQV.z) || isinf(newQV.z)) { newQV = VT3{0.0,0.0,0.0}; }
        
      dst.B[i0]   = newB; // updated values
      if(ix > src.size.x) { dst.QVx[dst.QVx.idx(ix,iy,iz)] = newQV.x; }
      if(iy > src.size.y) { dst.QVy[dst.QVy.idx(ix,iy,iz)] = newQV.y; }
      if(iz > src.size.z) { dst.QVz[dst.QVz.idx(ix,iy,iz)] = newQV.z; }
    
      dst.QP[i0]  = Q0.x; // copied values (unchanged)
      dst.QN[i0]  = Q0.y;
      dst.E[i0]   = E0;
      dst.mat[i0] = M0;
    }
}

// wrappers
template<typename T> void updateCharge(EMField<T> &src, EMField<T> &dst, FieldParams<T> &cp)
{
  if(src.size.x > 0 && src.size.y > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      updateCharge_k<<<grid, threads>>>(src, dst, cp);
      cudaDeviceSynchronize(); getLastCudaError("====> ERROR: updateCharge_k failed!");
    }
  else { std::cout << "==> WARNING: Skipped updateCharge (" << src.size << " / " << dst.size << ")\n"; }
}
template<typename T> void updateElectric(EMField<T> &src, EMField<T> &dst, FieldParams<T> &cp)
{
  if(src.size.x > 0 && src.size.y > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      updateElectric_k<<<grid, threads>>>(src, dst, cp);
      cudaDeviceSynchronize(); getLastCudaError("====> ERROR: updateElectric_k failed!");
    }
  else { std::cout << "==> WARNING: Skipped updateElectric (" << src.size << " / " << dst.size << ")\n"; }
}
template<typename T> void updateMagnetic(EMField<T> &src, EMField<T> &dst, FieldParams<T> &cp)
{
  if(src.size.x > 0 && src.size.y > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      updateMagnetic_k<<<grid, threads>>>(src, dst, cp);
      cudaDeviceSynchronize(); getLastCudaError("====> ERROR: updateMagnetic2D_k failed!");
    }
  else { std::cout << "==> WARNING: Skipped updateMagnetic2D (src: " << src.size << " / dst: " << dst.size << ")\n"; }
}

// template instantiation
template void updateCharge  <float> (EMField<float> &src, EMField<float>  &dst, FieldParams<float> &cp);
template void updateElectric<float> (EMField<float> &src, EMField<float>  &dst, FieldParams<float> &cp);
template void updateMagnetic<float> (EMField<float> &src, EMField<float>  &dst, FieldParams<float> &cp);
// template void updateCharge  <double>(EMField<double> &src, EMField<double> &dst, FieldParams<double> &cp);
// template void updateElectric<double>(EMField<double> &src, EMField<double> &dst, FieldParams<double> &cp);






