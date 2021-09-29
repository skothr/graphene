#include "field.cuh"

#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>

#include "physics.h"
#include "fluid.cuh"
#include "raytrace.cuh"
#include "vector-operators.h"
#include "cuda-tools.cuh"
#include "mathParser.hpp"

#define BLOCKDIM_X 10
#define BLOCKDIM_Y 10
#define BLOCKDIM_Z 8

//// PHYSICS UPDATES ////

// electric charge Q (NOTE: unimplemented)
template<typename T>
__global__ void updateCharge_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  typedef typename DimType<T, 2>::VEC_T VT2;
  typedef typename DimType<T, 3>::VEC_T VT3;
  typedef int3 IT;
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;

  if(ix < src.size.x && iy < src.size.y && iz < dst.size.z)
    {
      int i  = src.idx(ix, iy, iz);
      VT3 p  = VT3{T(ix), T(iy), T(iz)};

      VT3 v0 = src.v[i];
      T   p0 = src.p[i];
      T   d0 = src.div[i];
      T   Qn = src.Qn[i];
      T   Qp = src.Qp[i];
      VT3 Qv = src.Qv[i];
      VT3 E0 = src.E[i];
      VT3 B0 = src.B[i];
      Material<T> mat = src.mat[i];
      
      IT  ip = IT{int(ix), int(iy), int(iz)};
      if(slipPlane(ip-IT{1,0,0}, cp)) { Qv.x = 0; } // abs(Qv.x); }
      if(slipPlane(ip+IT{1,0,0}, cp)) { Qv.x = 0; } //-abs(Qv.x); }
      if(slipPlane(ip-IT{0,1,0}, cp)) { Qv.y = 0; } // abs(Qv.y); }
      if(slipPlane(ip+IT{0,1,0}, cp)) { Qv.y = 0; } //-abs(Qv.y); }
      if(slipPlane(ip-IT{0,0,1}, cp)) { Qv.z = 0; } // abs(Qv.z); }
      if(slipPlane(ip+IT{0,0,1}, cp)) { Qv.z = 0; } //-abs(Qv.z); }
      
      // Lorentz forces
      Qv += (Qp-Qn)*cross((Qp>Qn?1:-1)*Qv + v0, B0)*cp.u.dt;
      Qv += (Qp-Qn)*E0*cp.u.dt;
      
      if(isnan(Qn)   || isinf(Qn))   { Qn   = 0.0; } if(isnan(Qp)   || isinf(Qp))   { Qp   = 0.0; }
      if(isnan(Qv.x) || isinf(Qv.x)) { Qv.x = 0.0; } if(isnan(Qv.y) || isinf(Qv.y)) { Qv.y = 0.0; } if(isnan(Qv.z) || isinf(Qv.z)) { Qv.z = 0.0; }
      
      // use forward Euler method
      VT3 p2 = integrateForwardEuler(src.Qv.dData, p, Qv, cp.u.dt);
      // add actively to next point in texture
      int4 tiX = texPutIX(p2, cp);
      int4 tiY = texPutIY(p2, cp);
      int4 tiZ = texPutIZ(p2, cp);
      float4 mults0 = texPutMults0<float>(p2);
      float4 mults1 = texPutMults1<float>(p2);
      IT p000 = IT{tiX.x, tiY.x, tiZ.x}; IT p100 = IT{tiX.y, tiY.y, tiZ.x};
      IT p010 = IT{tiX.z, tiY.z, tiZ.x}; IT p110 = IT{tiX.w, tiY.w, tiZ.x};
      IT p001 = IT{tiX.x, tiY.x, tiZ.z}; IT p101 = IT{tiX.y, tiY.y, tiZ.z};
      IT p011 = IT{tiX.z, tiY.z, tiZ.z}; IT p111 = IT{tiX.w, tiY.w, tiZ.z};
      // scale value by grid overlap and store in each location      
      texAtomicAdd<float,3>(dst.Qp.dData, Qp*mults0.x, p000, cp); texAtomicAdd<float,3>(dst.Qp.dData, Qp*mults0.z,  p010, cp);
      texAtomicAdd<float,3>(dst.Qp.dData, Qp*mults0.y, p100, cp); texAtomicAdd<float,3>(dst.Qp.dData, Qp*mults0.w,  p110, cp);
      texAtomicAdd<float,3>(dst.Qp.dData, Qp*mults1.x, p001, cp); texAtomicAdd<float,3>(dst.Qp.dData, Qp*mults1.z,  p011, cp);
      texAtomicAdd<float,3>(dst.Qp.dData, Qp*mults1.y, p101, cp); texAtomicAdd<float,3>(dst.Qp.dData, Qp*mults1.w,  p111, cp);
      texAtomicAdd<float,3>(dst.Qv.dData, Qv*mults0.x, p000, cp); texAtomicAdd<float,3>(dst.Qv.dData, Qv*mults0.z,  p010, cp);
      texAtomicAdd<float,3>(dst.Qv.dData, Qv*mults0.y, p100, cp); texAtomicAdd<float,3>(dst.Qv.dData, Qv*mults0.w,  p110, cp);
      texAtomicAdd<float,3>(dst.Qv.dData, Qv*mults1.x, p001, cp); texAtomicAdd<float,3>(dst.Qv.dData, Qv*mults1.z,  p011, cp);
      texAtomicAdd<float,3>(dst.Qv.dData, Qv*mults1.y, p101, cp); texAtomicAdd<float,3>(dst.Qv.dData, Qv*mults1.w,  p111, cp);
      
      // update for negative charge
      p2 = integrateForwardEuler(src.Qv.dData, p, -Qv, cp.u.dt);
      tiX = texPutIX(p2, cp);
      tiY = texPutIY(p2, cp);
      tiZ = texPutIZ(p2, cp);
      mults0 = texPutMults0<float>(p2);
      mults1 = texPutMults1<float>(p2);
      p000 = IT{tiX.x, tiY.x, tiZ.x}; p100 = IT{tiX.y, tiY.y, tiZ.x};
      p010 = IT{tiX.z, tiY.z, tiZ.x}; p110 = IT{tiX.w, tiY.w, tiZ.x};
      p001 = IT{tiX.x, tiY.x, tiZ.z}; p101 = IT{tiX.y, tiY.y, tiZ.z};
      p011 = IT{tiX.z, tiY.z, tiZ.z}; p111 = IT{tiX.w, tiY.w, tiZ.z};
      // scale value by grid overlap and store in each location
      texAtomicAdd<float,3>(dst.Qn.dData, Qn*mults0.x, p000, cp); texAtomicAdd<float,3>(dst.Qn.dData, Qn*mults0.z,  p010, cp);
      texAtomicAdd<float,3>(dst.Qn.dData, Qn*mults0.y, p100, cp); texAtomicAdd<float,3>(dst.Qn.dData, Qn*mults0.w,  p110, cp);
      texAtomicAdd<float,3>(dst.Qn.dData, Qn*mults1.x, p001, cp); texAtomicAdd<float,3>(dst.Qn.dData, Qn*mults1.z,  p011, cp);
      texAtomicAdd<float,3>(dst.Qn.dData, Qn*mults1.y, p101, cp); texAtomicAdd<float,3>(dst.Qn.dData, Qn*mults1.w,  p111, cp);
      
      dst.v[i]   = v0;
      dst.p[i]   = p0;
      dst.div[i] = d0;
      dst.E[i]   = E0;
      dst.B[i]   = B0;
      dst.mat[i] = mat;
    }
}


//// SIMULATION -- MAXWELL'S EQUATIONS ////

// electric field E
template<typename T>
__global__ void updateElectric_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  using VT2 = typename DimType<T, 2>::VEC_T;
  using VT3 = typename DimType<T, 3>::VEC_T;
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;

  if(ix < src.size.x && iy < src.size.y && iz < dst.size.z)
    {
      int3 ip0 = int3{ix, iy, iz};
      int  i0  = src.idx(ix, iy, iz);

      // check for boundary (TODO: improve(?) -- still some reflections)
      if(!cp.reflect) 
        {
          const int bs = 1;
          int xOffset = src.size.x <= 2*bs ? 0 : ((ip0.x < bs ? 1 : 0) + (ip0.x >= src.size.x-2*bs ? -1 : 0));
          int yOffset = src.size.y <= 2*bs ? 0 : ((ip0.y < bs ? 1 : 0) + (ip0.y >= src.size.y-2*bs ? -1 : 0));
          int zOffset = src.size.z <= 2*bs ? 0 : ((ip0.z < bs ? 1 : 0) + (ip0.z >= src.size.z-2*bs ? -1 : 0));
          if(xOffset != 0 || yOffset != 0 || zOffset != 0)
            {
              int i = src.idx(max(0, min(src.size.x-1, ip0.x + xOffset)),
                              max(0, min(src.size.y-1, ip0.y + yOffset)),
                              max(0, min(src.size.z-1, ip0.z + zOffset)));
              dst.E[i0]   = src.E[i];   // use updated index for E
              dst.B[i0]   = src.B[i0];  // copy other data from original index
              dst.v[i0]   = src.v[i0];
              dst.p[i0]   = src.p[i0];
              dst.Qn[i0]  = src.Qn[i0];
              dst.Qp[i0]  = src.Qp[i0];
              dst.Qv[i0]  = src.Qv[i0];
              dst.mat[i0] = src.mat[i0];
              return;
            }
        }
      
      VT3 v0  = src.v[i0];
      T   p0  = src.p[i0];
      T   Qn0 = src.Qn[i0];
      T   Qp0 = src.Qp[i0];
      VT3 Qv0 = src.Qv[i0];
      VT3 E0  = src.E[i0];
      VT3 B0  = src.B[i0];
      Material<T> M0 = src.mat[i0];
      if(M0.vacuum()) { M0 = cp.u.vacuum(); } // check if vacuum
      typename Material<T>::Blend ab = M0.getBlendE(cp.u.dt, cp.u.dL);
      
      int3 ip1  = int3{max(0, ip0.x-1), max(0, ip0.y-1), max(0, ip0.z-1)};
      VT3  Bxn  = src.B[src.B.idx(ip1.x, ip0.y, ip0.z)]; // -1 in x direction
      VT3  Byn  = src.B[src.B.idx(ip0.x, ip1.y, ip0.z)]; // -1 in y direction
      VT3  Bzn  = src.B[src.B.idx(ip0.x, ip0.y, ip1.z)]; // -1 in z direction
      VT3  dEdt = VT3{  (B0.z-Byn.z) - (B0.y-Bzn.y),   // dBz/dY - dBy/dZ
                        (B0.x-Bzn.x) - (B0.z-Bxn.z),   // dBx/dZ - dBz/dX
                        (B0.y-Bxn.y) - (B0.x-Byn.x) }; // dBy/dX - dBx/dY

      // // apply effect of electric current (TODO: improve)
      VT3 J = (Qp0-Qn0)*normalize((Qp0 > Qn0 ? 1 : -1)*Qv0 + v0) / cp.u.dL/cp.u.dL/cp.u.dL; // (per unit volume)
      if(isnan(J.x) || isinf(J.x)) { J.x = 0.0; } if(isnan(J.y) || isinf(J.y)) { J.y = 0.0; } if(isnan(J.z) || isinf(J.z)) { J.z = 0.0; }
      dEdt -= J/cp.u.e0 * cp.u.dt;
      
      VT3 newE = ab.alpha*E0 + ab.beta*dEdt;
      if(isnan(newE.x) || isinf(newE.x)) { newE.x = 0.0; } if(isnan(newE.y) || isinf(newE.y)) { newE.y = 0.0; } if(isnan(newE.z) || isinf(newE.z)) { newE.z = 0.0; }
      
      // TODO: calculate ∇·E, and (somehow) correct so ∇·E = ρ/ε₀
      
      // // lorentz (E)
      // VT3 newQv = Qv0 + (Qp0-Qn0)*newE*cp.u.dt;
      // if(isnan(newQPV.x) || isinf(newQPV.x) || isnan(newQPV.y) || isinf(newQPV.y) || isnan(newQPV.z) || isinf(newQPV.z)) { newQPV = VT3{0.0,0.0,0.0}; }
      
      dst.v[i0]   = v0;
      dst.p[i0]   = p0;
      dst.Qn[i0]  = Qn0;
      dst.Qp[i0]  = Qp0;
      dst.Qv[i0]  = Qv0;
      dst.E[i0]   = newE;   // updated values
      dst.B[i0]   = B0;
      dst.mat[i0] = M0;
    }
}

// magnetic field B
template<typename T>
__global__ void updateMagnetic_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  using VT2 = typename DimType<T, 2>::VEC_T;
  using VT3 = typename DimType<T, 3>::VEC_T;
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;

  if(ix < src.size.x && iy < src.size.y && iz < dst.size.z)
    {
      int3  ip0 = int3{ix, iy, iz};
      int    i0 = src.idx(ix, iy, iz);

      // check for boundary (TODO: improve(?) -- still some reflections)
      if(!cp.reflect)
        {
          const int bs = 1;
          int xOffset = src.size.x <= 2*bs ? 0 : ((ip0.x < bs ? 1 : 0) + (ip0.x >= src.size.x-2*bs ? -1 : 0));
          int yOffset = src.size.y <= 2*bs ? 0 : ((ip0.y < bs ? 1 : 0) + (ip0.y >= src.size.y-2*bs ? -1 : 0));
          int zOffset = src.size.z <= 2*bs ? 0 : ((ip0.z < bs ? 1 : 0) + (ip0.z >= src.size.z-2*bs ? -1 : 0));
          if(xOffset != 0 || yOffset != 0 || zOffset != 0)
            {
              int i = src.idx(max(0, min(src.size.x-1, ip0.x + xOffset)),
                              max(0, min(src.size.y-1, ip0.y + yOffset)),
                              max(0, min(src.size.z-1, ip0.z + zOffset)));
              dst.B[i0]   = src.B[i];   // use updated index for B
              dst.E[i0]   = src.E[i0];  // copy other data from original index
              dst.v[i0]   = src.v[i0];
              dst.p[i0]   = src.p[i0];
              dst.Qn[i0]  = src.Qn[i0];
              dst.Qp[i0]  = src.Qp[i0];
              dst.Qv[i0]  = src.Qv[i0];
              dst.mat[i0] = src.mat[i0];
              return;
            }
        }
      
      VT3 v0  = src.v[i0];
      T   p0  = src.p[i0];
      T   Qn0 = src.Qn[i0];
      T   Qp0 = src.Qp[i0];
      VT3 Qv0 = src.Qv[i0];
      VT3 B0  = src.B[i0];
      VT3 E0  = src.E[i0];
      Material<T> M0 = src.mat[i0];
      if(M0.vacuum()) { M0 = cp.u.vacuum(); } // check if vacuum
      typename Material<T>::Blend ab = M0.getBlendB(cp.u.dt, cp.u.dL);

      int3 ip1  = int3{min(src.size.x-1, ip0.x+1),
                       min(src.size.y-1, ip0.y+1),
                       min(src.size.z-1, ip0.z+1) };
      VT3  Exp  = src.E[src.E.idx(ip1.x, ip0.y, ip0.z)]; // +1 in x direction
      VT3  Eyp  = src.E[src.E.idx(ip0.x, ip1.y, ip0.z)]; // +1 in y direction
      VT3  Ezp  = src.E[src.E.idx(ip0.x, ip0.y, ip1.z)]; // +1 in z direction
      VT3  dBdt = VT3{  (Eyp.z-E0.z) - (Ezp.y-E0.y),   // dEz/dY - dEy/dZ
                        (Ezp.x-E0.x) - (Exp.z-E0.z),   // dEx/dZ - dEz/dX
                        (Exp.y-E0.y) - (Eyp.x-E0.x) }; // dEy/dX - dEx/dY
      
      VT3 newB = ab.alpha*B0 - ab.beta*dBdt;
      if(isnan(newB.x) || isinf(newB.x)) { newB.x = 0.0; }
      if(isnan(newB.y) || isinf(newB.y)) { newB.y = 0.0; }
      if(isnan(newB.z) || isinf(newB.z)) { newB.z = 0.0; }

      // TODO: correct so ∇·B = 0 (like fluid pressure, hmm...)
      
      // // lorentz (v x B)
      // VT3 newQv = Qv0 + (Qp0-Qn0)*cross(Qv0, newB)*cp.u.dt;
      // if(isnan(newQPV.x) || isinf(newQPV.x) || isnan(newQPV.y) || isinf(newQPV.y) || isnan(newQPV.z) || isinf(newQPV.z)) { newQPV = VT3{0.0,0.0,0.0}; }
        
      dst.v[i0]   = v0;
      dst.p[i0]   = p0;
      dst.Qn[i0]  = Qn0;
      dst.Qp[i0]  = Qp0;
      dst.Qv[i0]  = Qv0;
      dst.B[i0]   = newB; // updated value
      dst.E[i0]   = E0;
      dst.mat[i0] = M0;
    }
}

// wrappers
template<typename T> void updateCharge(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> &cp)
{
  if(src.size.x > 0 && src.size.y > 0 && src.size.z > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      dst.Qn.clear();
      dst.Qp.clear();
      dst.Qv.clear();
      updateCharge_k<<<grid, threads>>>(src, dst, cp);
    }
  else { std::cout << "==> WARNING: Skipped updateCharge (" << src.size << " / " << dst.size << ")\n"; }
}
template<typename T> void updateElectric(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> &cp)
{
  if(src.size.x > 0 && src.size.y > 0 && src.size.z > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      updateElectric_k<<<grid, threads>>>(src, dst, cp);
    }
  else { std::cout << "==> WARNING: Skipped updateElectric (" << src.size << " / " << dst.size << ")\n"; }
}
template<typename T> void updateMagnetic(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> &cp)
{
  if(src.size.x > 0 && src.size.y > 0 && src.size.z > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      updateMagnetic_k<<<grid, threads>>>(src, dst, cp);
    }
  else { std::cout << "==> WARNING: Skipped updateMagnetic2D (src: " << src.size << " / dst: " << dst.size << ")\n"; }
}

// template instantiation
template void updateCharge  <float> (FluidField<float> &src, FluidField<float>  &dst, FluidParams<float> &cp);
template void updateElectric<float> (FluidField<float> &src, FluidField<float>  &dst, FluidParams<float> &cp);
template void updateMagnetic<float> (FluidField<float> &src, FluidField<float>  &dst, FluidParams<float> &cp);
// template void updateCharge  <double>(FluidField<double> &src, FluidField<double> &dst, FluidParams<double> &cp);
// template void updateElectric<double>(FluidField<double> &src, FluidField<double> &dst, FluidParams<double> &cp);






