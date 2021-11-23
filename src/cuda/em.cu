#include "em.cuh"

#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>

#include "cuda-tools.cuh"
#include "field-operators.cuh"
#include "vector-operators.h"
#include "physics.h"
#include "fluid.cuh"
#include "mathParser.hpp"

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8
#define BLOCKDIM_Z 8


template<typename T, typename VT3=typename DimType<T, 3>::VEC_T>
__device__ T getCharge(const FluidField<T> &src, unsigned long i) { return (src.Qp[i] - src.Qn[i]); }


//// PHYSICS UPDATES ////

// update electric charge (Q)
template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT=typename Dim<VT3>::SIZE_T>
__global__ void updateCharge_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;

  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      int i  = src.idx(ix, iy, iz);
      VT3 p  = VT3{T(ix), T(iy), T(iz)};

      VT3 v0  = src.v[i];   T   p0  = src.p[i];  T div0 = src.divE[i];
      T   Qn  = src.Qn[i];  T   Qp  = src.Qp[i]; T Q    = (Qp-Qn);
      VT3 Qnv = src.Qnv[i]; VT3 Qpv = src.Qpv[i];
      VT3 E0  = src.E[i];   VT3 B0  = src.B[i];

      VT3 E = E0;
      VT3 B = B0;

      // average surrounding cells (corners/faces(?))  ==> TODO: do for B field instead(?)
      // IT pE100 = applyBounds(IT{ix+1, iy,   iz  }, src.size, cp); IT pE010 = applyBounds(IT{ix,   iy+1, iz  }, src.size, cp);
      // IT pE110 = applyBounds(IT{ix+1, iy+1, iz  }, src.size, cp); IT pE001 = applyBounds(IT{ix,   iy,   iz+1}, src.size, cp);
      // IT pE101 = applyBounds(IT{ix+1, iy,   iz+1}, src.size, cp); IT pE011 = applyBounds(IT{ix,   iy+1, iz+1}, src.size, cp);
      // IT pE111 = applyBounds(IT{ix+1, iy+1, iz+1}, src.size, cp);
      // bool n100 = (pE100 < 0); bool n010 = (pE010 < 0); bool n110 = (pE110 < 0);
      // bool n001 = (pE001 < 0); bool n101 = (pE101 < 0); bool n011 = (pE011 < 0); bool n111 = (pE111 < 0);
      // E = E + ((!n001 ? src.E[src.idx(pE001)] : VT3{0,0,0}) +
      //          (!n010 ? src.E[src.idx(pE010)] : VT3{0,0,0}) +
      //          (!n011 ? src.E[src.idx(pE011)] : VT3{0,0,0}) +
      //          (!n100 ? src.E[src.idx(pE100)] : VT3{0,0,0}) +
      //          (!n101 ? src.E[src.idx(pE101)] : VT3{0,0,0}) +
      //          (!n110 ? src.E[src.idx(pE110)] : VT3{0,0,0}) +
      //          (!n111 ? src.E[src.idx(pE111)] : VT3{0,0,0})) / (1+int(n001)+int(n010)+int(n011)+int(n100)+int(n101)+int(n110)+int(n111));      

      // enforce boundaries
      IT ip = IT {ix, iy, iz};
      VT3 eQpv = VT3{0,0,0}; VT3 eQnv = VT3{0,0,0}; int count = 0;
      if(ip.x == 0 && ip.x < src.size.x-1 && cp.edgeNX == EDGE_FREESLIP) // -X bound
        { eQpv.x -= src.Qpv[src.idx(ip+IT{1,0,0})].x; eQnv.x -= src.Qnv[src.idx(ip+IT{1,0,0})].x; count++; }
      if(ip.y == 0 && ip.y < src.size.y-1 && cp.edgeNY == EDGE_FREESLIP) // -Y bound
        { eQpv.y -= src.Qpv[src.idx(ip+IT{0,1,0})].y; eQnv.y -= src.Qnv[src.idx(ip+IT{0,1,0})].y; count++; }
      if(ip.z == 0 && ip.z < src.size.z-1 && cp.edgeNZ == EDGE_FREESLIP) // -Z bound
        { eQpv.z -= src.Qpv[src.idx(ip+IT{0,0,1})].z; eQnv.z -= src.Qnv[src.idx(ip+IT{0,0,1})].z; count++; }
      if(ip.x == src.size.x-1 && ip.x > 0 && cp.edgePX == EDGE_FREESLIP) // +X bound
        { eQpv.x -= src.Qpv[src.idx(ip-IT{1,0,0})].x; eQnv.x -= src.Qnv[src.idx(ip-IT{1,0,0})].x; count++; }
      if(ip.y == src.size.y-1 && ip.y > 0 && cp.edgePY == EDGE_FREESLIP) // +Y bound
        { eQpv.y -= src.Qpv[src.idx(ip-IT{0,1,0})].y; eQnv.y -= src.Qnv[src.idx(ip-IT{0,1,0})].y; count++; }
      if(ip.z == src.size.z-1 && ip.z > 0 && cp.edgePZ == EDGE_FREESLIP) // +Z bound
        { eQpv.z -= src.Qpv[src.idx(ip-IT{0,0,1})].z; eQnv.z -= src.Qnv[src.idx(ip-IT{0,0,1})].z; count++; }
      Qpv = (VT3{(count == 0 ? Qpv.x : eQpv.x/(T)count),
                 (count == 0 ? Qpv.y : eQpv.y/(T)count),
                 (count == 0 ? Qpv.z : eQpv.z/(T)count)});
      Qnv = (VT3{(count == 0 ? Qnv.x : eQnv.x/(T)count),
                 (count == 0 ? Qnv.y : eQnv.y/(T)count),
                 (count == 0 ? Qnv.z : eQnv.z/(T)count)});

      // Lorentz forces
      Qnv += cp.u.dt*(-Qn*E + cross(-Qn*(Qnv) + v0, B)); // Fe = qE + q(v×B)
      Qpv += cp.u.dt*( Qp*E + cross( Qp*(Qpv) + v0, B)); // Fb = qE + q(v×B)

      
      // check result values
      if(!isvalid(Qnv)) { Qnv = VT3{0,0,0}; } if(!isvalid(Qnv.x)) { Qnv.x = 0.0; } if(!isvalid(Qnv.y)) { Qnv.y = 0.0; } if(!isvalid(Qnv.z)) { Qnv.z = 0.0; }
      if(!isvalid(Qpv)) { Qpv = VT3{0,0,0}; } if(!isvalid(Qpv.x)) { Qpv.x = 0.0; } if(!isvalid(Qpv.y)) { Qpv.y = 0.0; } if(!isvalid(Qpv.z)) { Qpv.z = 0.0; }      

      VT3 p2 = integrateForwardEuler(src.Qpv.dData, p, Qpv, cp.u.dt);
      putTex2DD(dst.Qp.dData,   Qp,  p2, cp);
      putTex2DD(dst.Qpv.dData,  Qpv, p2, cp);
      // VT3 p2 = integrateRK4(src.Qpv, p, cp);
      // p2 = (VT3{max(1.0f, min(T(src.size.x-2), p2.x)), max(1.0f, min(T(src.size.y-2), p2.y)), max(1.0f, min(T(src.size.z-2), p2.z))});

      // // (forward Euler integration)
      // int4 tiX = texPutIX(p2, cp); int4 tiY = texPutIY(p2, cp); int4 tiZ = texPutIZ(p2, cp);
      // float4 mults0 = texPutMults0<float>(p2); float4 mults1 = texPutMults1<float>(p2);
      // IT p000 = IT{tiX.x, tiY.x, tiZ.x}; IT p100 = IT{tiX.y, tiY.y, tiZ.x}; IT p010 = IT{tiX.z, tiY.z, tiZ.x}; IT p110 = IT{tiX.w, tiY.w, tiZ.x};
      // IT p001 = IT{tiX.x, tiY.x, tiZ.z}; IT p101 = IT{tiX.y, tiY.y, tiZ.z}; IT p011 = IT{tiX.z, tiY.z, tiZ.z}; IT p111 = IT{tiX.w, tiY.w, tiZ.z};
      // // scale value by grid overlap and store in each location
      // texAtomicAdd<float,3>(dst.Qp.dData,  Qp*mults0.x,  p000, cp); texAtomicAdd<float,3>(dst.Qp.dData,  Qp*mults0.z,  p010, cp); // Qp
      // texAtomicAdd<float,3>(dst.Qp.dData,  Qp*mults0.y,  p100, cp); texAtomicAdd<float,3>(dst.Qp.dData,  Qp*mults0.w,  p110, cp);
      // texAtomicAdd<float,3>(dst.Qp.dData,  Qp*mults1.x,  p001, cp); texAtomicAdd<float,3>(dst.Qp.dData,  Qp*mults1.z,  p011, cp);
      // texAtomicAdd<float,3>(dst.Qp.dData,  Qp*mults1.y,  p101, cp); texAtomicAdd<float,3>(dst.Qp.dData,  Qp*mults1.w,  p111, cp);
      // texAtomicAdd<float,3>(dst.Qpv.dData, Qpv*mults0.x, p000, cp); texAtomicAdd<float,3>(dst.Qpv.dData, Qpv*mults0.z, p010, cp); // Qpv
      // texAtomicAdd<float,3>(dst.Qpv.dData, Qpv*mults0.y, p100, cp); texAtomicAdd<float,3>(dst.Qpv.dData, Qpv*mults0.w, p110, cp);
      // texAtomicAdd<float,3>(dst.Qpv.dData, Qpv*mults1.x, p001, cp); texAtomicAdd<float,3>(dst.Qpv.dData, Qpv*mults1.z, p011, cp);
      // texAtomicAdd<float,3>(dst.Qpv.dData, Qpv*mults1.y, p101, cp); texAtomicAdd<float,3>(dst.Qpv.dData, Qpv*mults1.w, p111, cp);

      p2 = integrateForwardEuler(src.Qnv.dData, p, Qnv, cp.u.dt);
      putTex2DD(dst.Qn.dData,   Qn,  p2, cp);
      putTex2DD(dst.Qnv.dData,  Qnv, p2, cp);
      // p2 = integrateRK4(src.Qnv, p, cp);
      // p2 = (VT3{max(1.0f, min(T(src.size.x-2), p2.x)), max(1.0f, min(T(src.size.y-2), p2.y)), max(1.0f, min(T(src.size.z-2), p2.z))});
      
      // // (forward Euler integration)
      // tiX = texPutIX(p2, cp); tiY = texPutIY(p2, cp); tiZ = texPutIZ(p2, cp);
      // mults0 = texPutMults0<float>(p2); mults1 = texPutMults1<float>(p2);
      // p000 = IT{tiX.x, tiY.x, tiZ.x}; p100 = IT{tiX.y, tiY.y, tiZ.x}; p010 = IT{tiX.z, tiY.z, tiZ.x}; p110 = IT{tiX.w, tiY.w, tiZ.x};
      // p001 = IT{tiX.x, tiY.x, tiZ.z}; p101 = IT{tiX.y, tiY.y, tiZ.z}; p011 = IT{tiX.z, tiY.z, tiZ.z}; p111 = IT{tiX.w, tiY.w, tiZ.z};
      // // scale value by grid overlap and store in each location
      // texAtomicAdd<float,3>(dst.Qn.dData,  Qn*mults0.x,  p000, cp); texAtomicAdd<float,3>(dst.Qn.dData,  Qn*mults0.z,  p010, cp); // Qn
      // texAtomicAdd<float,3>(dst.Qn.dData,  Qn*mults0.y,  p100, cp); texAtomicAdd<float,3>(dst.Qn.dData,  Qn*mults0.w,  p110, cp);
      // texAtomicAdd<float,3>(dst.Qn.dData,  Qn*mults1.x,  p001, cp); texAtomicAdd<float,3>(dst.Qn.dData,  Qn*mults1.z,  p011, cp);
      // texAtomicAdd<float,3>(dst.Qn.dData,  Qn*mults1.y,  p101, cp); texAtomicAdd<float,3>(dst.Qn.dData,  Qn*mults1.w,  p111, cp);
      // texAtomicAdd<float,3>(dst.Qnv.dData, Qnv*mults0.x, p000, cp); texAtomicAdd<float,3>(dst.Qnv.dData, Qnv*mults0.z, p010, cp); // Qnv
      // texAtomicAdd<float,3>(dst.Qnv.dData, Qnv*mults0.y, p100, cp); texAtomicAdd<float,3>(dst.Qnv.dData, Qnv*mults0.w, p110, cp);
      // texAtomicAdd<float,3>(dst.Qnv.dData, Qnv*mults1.x, p001, cp); texAtomicAdd<float,3>(dst.Qnv.dData, Qnv*mults1.z, p011, cp);
      // texAtomicAdd<float,3>(dst.Qnv.dData, Qnv*mults1.y, p101, cp); texAtomicAdd<float,3>(dst.Qnv.dData, Qnv*mults1.w, p111, cp);

      dst.v[i]   = v0;
      dst.p[i]   = src.p[i];
      dst.E[i]   = E0;
      dst.B[i]   = B0;
      dst.mat[i] = src.mat[i];
      //dst.divE[i] = src.divE[i];
      //dst.divB[i] = src.divB[i];
      //dst.Ep[i]   = src.Ep;
      //dst.Bp[i]   = src.Bp[i];
    }
}


///////////////////////////////////////////
//// SIMULATION -- MAXWELL'S EQUATIONS ////
///////////////////////////////////////////
//// Uses a Yee grid / Yee's method (more info: https://empossible.net/wp-content/uploads/2019/08/Lecture-4b-Maxwells-Equations-on-a-Yee-Grid.pdf)
//// - E and H are staggered -- E is measured at cell origins, and B/H at cell face centers

// electric field E
template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT=typename Dim<VT3>::SIZE_T>
__global__ void updateElectric_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;

  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT  ip0 = IT{ix, iy, iz};
      int i0  = src.idx(ip0);

      // check for boundary (TODO: improve(?) -- still some reflections)
      if(!cp.reflect) 
        {
          const int bs = 1;
          int xOffset = (((cp.edgeNX == EDGE_WRAP || cp.edgePX == EDGE_WRAP)) ? 0 :
                         (src.size.x <= bs ? 0 : ((ip0.x < bs ? 1 : 0) + (ip0.x >= src.size.x-1-bs ? -1 : 0))));
          int yOffset = (((cp.edgeNY == EDGE_WRAP || cp.edgePY == EDGE_WRAP)) ? 0 :
                         (src.size.y <= bs ? 0 : ((ip0.y < bs ? 1 : 0) + (ip0.y >= src.size.y-1-bs ? -1 : 0))));
          int zOffset = (((cp.edgeNZ == EDGE_WRAP || cp.edgePZ == EDGE_WRAP)) ? 0 :
                         (src.size.z <= bs ? 0 : ((ip0.z < bs ? 1 : 0) + (ip0.z >= src.size.z-1-bs ? -1 : 0))));
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
              dst.Qnv[i0] = src.Qnv[i0];
              dst.Qpv[i0] = src.Qpv[i0];
              dst.mat[i0] = src.mat[i0];
              return;
            }
        }
      
      VT3 v0   = src.v[i0];
      T   p0   = src.p[i0];
      T   Qn0  = src.Qn[i0];  T   Qp0  = src.Qp[i0];  T Q  = (Qp0-Qn0);
      VT3 Qnv0 = src.Qnv[i0]; VT3 Qpv0 = src.Qpv[i0];
      VT3 E0   = src.E[i0];
      VT3 B0   = src.B[i0];
      Material<T> M0 = src.mat[i0];
      if(M0.vacuum()) { M0 = cp.u.vacuum(); } // check if vacuum
      typename Material<T>::Blend ab = M0.getBlendE(cp.u.dt, cp.u.dL);
      
      // IT  ipN1  = applyBounds(ip0-1, src.size, cp); IT{max(0, ip0.x-1), max(0, ip0.y-1), max(0, ip0.z-1)};
      IT ip1 = applyBounds(ip0-1, src.size, cp);
      if(ip1.x < 0) { ip1.x = max(0, ip0.x-1); }
      if(ip1.y < 0) { ip1.y = max(0, ip0.y-1); }
      if(ip1.z < 0) { ip1.z = max(0, ip0.z-1); }
      VT3  Bxn  = src.B[src.B.idx(ip1.x, ip0.y, ip0.z )]; // -1 in x direction
      VT3  Byn  = src.B[src.B.idx(ip0.x, ip1.y, ip0.z )]; // -1 in y direction
      VT3  Bzn  = src.B[src.B.idx(ip0.x, ip0.y, ip1.z)];  // -1 in z direction
      VT3  dEdt = (VT3{(B0.z-Byn.z) - (B0.y-Bzn.y),    // dBz/dY - dBy/dZ
                       (B0.x-Bzn.x) - (B0.z-Bxn.z),    // dBx/dZ - dBz/dX
                       (B0.y-Bxn.y) - (B0.x-Byn.x) }); // dBy/dX - dBx/dY
          
      // apply effect of electric current (TODO: improve)
      VT3 J = (Qp0*Qpv0 - Qn0*Qnv0 + Q*v0) / (6.0f*cp.u.dL*cp.u.dL); // (averaged over cell faces (?))
      if(!isvalid(J)) { J = VT3{0,0,0}; } if(!isvalid(J.x)) { J.x = 0.0; } if(!isvalid(J.y)) { J.y = 0.0; } if(!isvalid(J.z)) { J.z = 0.0; }
      
      dEdt -= J/cp.u.e0 * cp.u.dt;
      if(!isvalid(dEdt)) { dEdt = VT3{0,0,0}; } if(!isvalid(dEdt.x)) { dEdt.x = 0; } if(!isvalid(dEdt.y)) { dEdt.y = 0; } if(!isvalid(dEdt.z)) { dEdt.z = 0; }
      
      VT3 newE = ab.alpha*E0 + ab.beta*dEdt;
      if(!isvalid(newE)) { newE = VT3{0,0,0}; } if(!isvalid(newE.x)) { newE.x = 0; } if(!isvalid(newE.y)) { newE.y = 0; } if(!isvalid(newE.z)) { newE.z = 0; }
      
      dst.v[i0]   = v0;
      dst.p[i0]   = p0;
      dst.Qn[i0]  = Qn0;
      dst.Qp[i0]  = Qp0;
      dst.Qnv[i0] = Qnv0;
      dst.Qpv[i0] = Qpv0;
      dst.E[i0]   = newE;   // updated values
      dst.B[i0]   = B0;
      dst.mat[i0] = M0;
    }
}

// magnetic field B
template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT=typename Dim<VT3>::SIZE_T>
__global__ void updateMagnetic_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;

  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT  ip0 = IT{ix, iy, iz};
      int i0  = src.idx(ip0);
      // check for boundary (TODO: improve(?) -- still some reflections)
      if(!cp.reflect)
        {
          const int bs = 1;
          int xOffset = src.size.x <= bs ? 0 : ((ip0.x < bs ? 1 : 0) + (ip0.x >= src.size.x-1-bs ? -1 : 0));
          int yOffset = src.size.y <= bs ? 0 : ((ip0.y < bs ? 1 : 0) + (ip0.y >= src.size.y-1-bs ? -1 : 0));
          int zOffset = src.size.z <= bs ? 0 : ((ip0.z < bs ? 1 : 0) + (ip0.z >= src.size.z-1-bs ? -1 : 0));
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
              dst.Qnv[i0] = src.Qnv[i0];
              dst.Qpv[i0] = src.Qpv[i0];
              dst.mat[i0] = src.mat[i0];
              return;
            }
        }
      
      VT3 v0  = src.v[i0];
      T   p0  = src.p[i0];
      T   Qn0 = src.Qn[i0];
      T   Qp0 = src.Qp[i0];
      VT3 Qnv0 = src.Qnv[i0];
      VT3 Qpv0 = src.Qpv[i0];
      VT3 B0  = src.B[i0];
      VT3 E0  = src.E[i0];
      Material<T> M0 = src.mat[i0];
      if(M0.vacuum()) { M0 = cp.u.vacuum(); } // check if vacuum
      typename Material<T>::Blend ab = M0.getBlendB(cp.u.dt, cp.u.dL);

      // IT ip1 = (IT{min(src.size.x-1, ip0.x+1), min(src.size.y-1, ip0.y+1), min(src.size.z-1, ip0.z+1) });
      IT ip1 = applyBounds(ip0+1, src.size, cp);
      if(ip1.x < 0) { ip1.x = min(src.size.x-1, ip0.x+1); }
      if(ip1.y < 0) { ip1.y = min(src.size.y-1, ip0.y+1); }
      if(ip1.z < 0) { ip1.z = min(src.size.z-1, ip0.z+1); }
      
      VT3  Exp  = src.E[src.E.idx(ip1.x, ip0.y, ip0.z)]; // +1 in x direction
      VT3  Eyp  = src.E[src.E.idx(ip0.x, ip1.y, ip0.z)]; // +1 in y direction
      VT3  Ezp  = src.E[src.E.idx(ip0.x, ip0.y, ip1.z)]; // +1 in z direction
      VT3  dBdt = (VT3{(Eyp.z-E0.z) - (Ezp.y-E0.y),    // dEz/dY - dEy/dZ
                       (Ezp.x-E0.x) - (Exp.z-E0.z),    // dEx/dZ - dEz/dX
                       (Exp.y-E0.y) - (Eyp.x-E0.x) }); // dEy/dX - dEx/dY
      if(!isvalid(dBdt)) { dBdt = VT3{0,0,0}; } if(!isvalid(dBdt.x)) { dBdt.x = 0; } if(!isvalid(dBdt.y)) { dBdt.y = 0; } if(!isvalid(dBdt.z)) { dBdt.z = 0; }
      
      VT3 newB = ab.alpha*B0 - ab.beta*dBdt;
      if(!isvalid(newB)) { newB = VT3{0,0,0}; } if(!isvalid(newB.x)) { newB.x = 0; } if(!isvalid(newB.y)) { newB.y = 0; } if(!isvalid(newB.z)) { newB.z = 0; }

      // TODO: ∇·B = 0 (?)
      
      dst.v[i0]   = v0;
      dst.p[i0]   = p0;
      dst.Qn[i0]  = Qn0;
      dst.Qp[i0]  = Qp0;
      dst.Qnv[i0] = Qnv0;
      dst.Qpv[i0] = Qpv0;
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
      dst.Qn.clear(); // clear relevant dst fields (advected using forward Euler method)
      dst.Qp.clear();
      dst.Qnv.clear();
      dst.Qpv.clear();
      updateCharge_k<<<grid, threads>>>(src, dst, cp);
    }
  else { std::cout << "==> WARNING: skipped updateCharge (" << src.size << " / " << dst.size << ")\n"; }
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
  else { std::cout << "==> WARNING: skipped updateElectric (" << src.size << " / " << dst.size << ")\n"; }
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
  else { std::cout << "==> WARNING: skipped updateMagnetic2D (src: " << src.size << " / dst: " << dst.size << ")\n"; }
}

// template instantiation
template void updateCharge  <float> (FluidField<float> &src, FluidField<float>  &dst, FluidParams<float> &cp);
template void updateElectric<float> (FluidField<float> &src, FluidField<float>  &dst, FluidParams<float> &cp);
template void updateMagnetic<float> (FluidField<float> &src, FluidField<float>  &dst, FluidParams<float> &cp);




//// COULOMB FORCES --> [...] ////
//////// discrete

template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT=typename Dim<VT3>::SIZE_T>
__global__ void updateCoulomb_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;

  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT  ip = IT{ix, iy, iz};
      int i  = src.idx(ip);

      VT3 F = VT3{0,0,0};
      for(int x = -cp.rCoulomb; x <= cp.rCoulomb; x++)
        for(int y = -cp.rCoulomb; y <= cp.rCoulomb; y++)
          for(int z = -cp.rCoulomb; z <= cp.rCoulomb; z++)
            {
              IT ip1   = ip + IT{x,y,z} ;
              VT3 dp   = -(makeV<VT3>(IT{x,y,z}));// + VT3{0.5, 0.5, 0.5});
              T dist_2 = length2(dp);
              if(dist_2 > 0 && dist_2 <= cp.rCoulomb*cp.rCoulomb)
                {
                  IT ip2 = applyBounds(ip1, src.size, cp);
                  if(ip2.x >= 0 && ip2.y >= 0 && ip2.z >= 0)
                    {
                      int i2 = src.idx(ip2);
                      T Q2 = src.Qp[i2] - src.Qn[i2]; // charge at other point
                      F += Q2*dp / (dist_2*sqrt(dist_2));
                    }
                }
            }
      F *= cp.coulombMult;
      
      // dst.E[i]   = src.E[i]*(1.0-cp.coulombBlend) + (F*cp.coulombMult)*cp.coulombBlend;
      dst.E[i]   = src.E[i] + F*cp.u.dt;
      dst.B[i]   = src.B[i];
      dst.v[i]   = src.v[i];
      dst.p[i]   = src.p[i];
      dst.Qn[i]  = src.Qn[i];
      dst.Qp[i]  = src.Qp[i];
      dst.Qnv[i] = src.Qnv[i];
      dst.Qpv[i] = src.Qpv[i];
      dst.mat[i] = src.mat[i];
    }
}

// wrapper
template<typename T> void updateCoulomb(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> &cp)
{
  if(src.size > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
          
      updateCoulomb_k<<<grid, threads>>>(src, dst, cp);
    }
}

// template instantiation
template void updateCoulomb<float>(FluidField<float> &src, FluidField<float> &dst, FluidParams<float> &cp);







//// CHARGE POTENTIAL --> ∇·E = Q/ε₀  ==> ∫(E*dS) = ρ/ε₀ ==> (1/ε₀)*∫(ρ*dV) ////
//////// continuous

template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT=typename Dim<VT3>::SIZE_T>
__global__ void chargePotentialPre_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT  ip   = makeI<IT>(ix, iy, iz);
      int i = src.idx(ip);
      IT p00 = applyBounds(ip,   src.size, cp); // current cell
      IT pn1 = applyBounds(ip-1, src.size, cp); // cell - (1,1)
      IT pp1 = applyBounds(ip+1, src.size, cp); // cell + (1,1)
      
      // calculate divergence of E
      T   d = 0;
      VT3 q = VT3{0,0,0};
      if(p00 >= 0 && pp1 >= 0 && pn1 >= 0)
        {
          VT3 h = 1.0/makeV<VT3>(src.size);
          // d  = (h.x*(src.E[src.idx(pp1.x, p00.y, p00.z)].x - src.E[src.idx(pn1.x, p00.y, p00.z)].x) +
          //       h.y*(src.E[src.idx(p00.x, pp1.y, p00.z)].y - src.E[src.idx(p00.x, pn1.y, p00.z)].y) +
          //       h.z*(src.E[src.idx(p00.x, p00.y, pp1.z)].z - src.E[src.idx(p00.x, p00.y, pn1.z)].z)) / 3.0f;
          d = divergence(src.E, p00, pp1, pn1, cp.u.dL);
          q = gradient(src.Qp, p00, pp1, pn1, cp.u.dL) - gradient(src.Qn, p00, pp1, pn1, cp.u.dL);
          d = (isvalid(d) ? d : 0);
          q = VT3{(isvalid(q.x) ? q.x : 0), (isvalid(q.y) ? q.y : 0), (isvalid(q.z) ? q.z : 0)};
        }
      dst.divE[i]  = d;
      dst.gradQ[i] = q;
    }
}


template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT=typename Dim<VT3>::SIZE_T>
__global__ void chargePotentialIter_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT  ip   = makeI<IT>(ix, iy, iz);
      int i = src.idx(ip);
      IT p00 = applyBounds(ip,   src.size, cp); // current cell
      IT pp1 = applyBounds(ip+1, src.size, cp); // cell + (1,1,1)
      IT pn1 = applyBounds(ip-1, src.size, cp); // cell - (1,1,1)

      // iterate --> 
      if(p00 >= 0 && pp1 >= 0 && pn1 >= 0)
        {
          VT3 h = 1.0/makeV<VT3>(src.size);
          // VT3 gradE = gradient(src.divE, p00, pp1, pn1, cp.u.dL);
          // VT3 curlE = curl(src.E, p00, pp1, pn1, cp.u.dL); if(isnan(curlE) || isinf(curlE)) { curlE = VT3{0,0,0}; }
          
          VT3 dE = (((src.E[src.idx(pp1.x, p00.y, p00.z)] + src.E[src.idx(pn1.x, p00.y, p00.z)]) +
                     (src.E[src.idx(p00.x, pp1.y, p00.z)] + src.E[src.idx(p00.x, pn1.y, p00.z)]) +
                     (src.E[src.idx(p00.x, p00.y, pp1.z)] + src.E[src.idx(p00.x, p00.y, pn1.z)])) - src.gradQ[i])/6.0f; // - curlE (?)
          if(!isvalid(dE)) { dE = VT3{0,0,0}; }
          src.E[i] += dE*cp.u.dt;
        }
    }
}


template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT=typename Dim<VT3>::SIZE_T>
__global__ void chargePotentialPost_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT  ip   = makeI<IT>(ix, iy, iz);
      int i = src.idx(ip);
      IT p00 = applyBounds(ip,   src.size, cp); // current cell
      IT pp1 = applyBounds(ip+1, src.size, cp); // cell + (1,1,1)
      IT pn1 = applyBounds(ip-1, src.size, cp); // cell - (1,1,1)
      
      if(p00 >= 0 && pp1 >= 0 && pn1 >= 0)
        {
          VT3 h = 1.0/makeV<VT3>(src.size);
          // dst.E[i].x += cp.u.dt*(src.divE[src.idx(pn1.x, p00.y, p00.z)] - src.divE[src.idx(pp1.x, p00.y, p00.z)]) / 3.0f / h.x;
          // dst.E[i].y += cp.u.dt*(src.divE[src.idx(p00.x, pn1.y, p00.z)] - src.divE[src.idx(p00.x, pp1.y, p00.z)]) / 3.0f / h.y;
          // dst.E[i].z += cp.u.dt*(src.divE[src.idx(p00.x, p00.y, pn1.z)] - src.divE[src.idx(p00.x, p00.y, pp1.z)]) / 3.0f / h.z;
          // dst.E[i].x += (getCharge(src, src.idx(pn1.x, p00.y, p00.z)) - getCharge(src, src.idx(pp1.x, p00.y, p00.z))) / 3.0f / h.x;
          // dst.E[i].y += (getCharge(src, src.idx(p00.x, pn1.y, p00.z)) - getCharge(src, src.idx(p00.x, pp1.y, p00.z))) / 3.0f / h.y;
          // dst.E[i].z += (getCharge(src, src.idx(p00.x, p00.y, pn1.z)) - getCharge(src, src.idx(p00.x, p00.y, pp1.z))) / 3.0f / h.z;
          // dst.B[i].x -= cp.u.dt*cp.u.dt*(src.divE[src.idx(pp1.x, p00.y, p00.z)] - src.divE[src.idx(pn1.x, p00.y, p00.z)]) / 3.0f / h.x;
          // dst.B[i].y -= cp.u.dt*cp.u.dt*(src.divE[src.idx(p00.x, pp1.y, p00.z)] - src.divE[src.idx(p00.x, pn1.y, p00.z)]) / 3.0f / h.y;
          // dst.B[i].z -= cp.u.dt*cp.u.dt*(src.divE[src.idx(p00.x, p00.y, pp1.z)] - src.divE[src.idx(p00.x, p00.y, pn1.z)]) / 3.0f / h.z;
        }
    }
}


// wrapper
template<typename T> void chargePotential(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> &cp, int iter)
{
  if(src.size > 0 && dst.size == src.size)
    {
      src.copyTo(dst);
      if(iter > 0)
        {
          FluidField<T> *temp1 = &src; FluidField<T> *temp2 = &dst;
          dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
          dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                    (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                    (int)ceil(src.size.z/(float)BLOCKDIM_Z));
          
          chargePotentialPre_k<<<grid, threads>>>(*temp1, *temp2, cp); std::swap(temp1, temp2);
          for(int i = 0; i < iter; i++)
            { chargePotentialIter_k<<<grid, threads>>>(*temp1, *temp2, cp); std::swap(temp1, temp2); }
          if(temp1 == &dst) { temp1->copyTo(*temp2); std::swap(temp1, temp2); } // make sure temp2 == dst for post process
          chargePotentialPost_k<<<grid, threads>>>(*temp1, *temp2, cp);
        }
    }
}

// template instantiation
template void chargePotential<float>(FluidField<float> &src, FluidField<float> &dst, FluidParams<float> &cp, int iter);





//// MAGNETIC CURL --> ∇·B = 0 ////
template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT=typename Dim<VT3>::SIZE_T>
__global__ void magneticCurlPre_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT  ip   = makeI<IT>(ix, iy, iz);
      int i = src.idx(ip);
      VT3 h = 1.0/makeV<VT3>(src.size);
      
      IT p00 = applyBounds(ip,   src.size, cp); // current cell
      IT pn1 = applyBounds(ip-1, src.size, cp); // cell - (1,1)
      IT pp1 = applyBounds(ip+1, src.size, cp); // cell + (1,1)
      
      // calculate divergence of B
      T p = 0; T d = 0;
      if(p00 >= 0 && pp1 >= 0 && pn1 >= 0)
        {
          p = src.p[i]; p = (isvalid(p) ? p : 0);
          d = divergence(src.B, p00, pp1, pn1, cp.u.dL);
          d = (isvalid(d) ? d : 0);
        }
      dst.Bp[i]   = p;
      dst.divB[i] = d;
    }
}

template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT=typename Dim<VT3>::SIZE_T>
__global__ void magneticCurlIter_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT  ip   = makeI<IT>(ix, iy, iz);
      int i = src.idx(ip);
      IT p00 = applyBounds(ip,   src.size, cp); // current cell
      IT pp1 = applyBounds(ip+1, src.size, cp); // cell + (1,1,1)
      IT pn1 = applyBounds(ip-1, src.size, cp); // cell - (1,1,1)

      // iterate -- remove divergence (TODO)
      if(p00 >= 0 && pp1 >= 0 && pn1 >= 0)
        {
          dst.Bp[i] = ((src.Bp[src.idx(pp1.x, p00.y, p00.z)] + src.Bp[src.idx(pn1.x, p00.y, p00.z)]) +
                       (src.Bp[src.idx(p00.x, pp1.y, p00.z)] + src.Bp[src.idx(p00.x, pn1.y, p00.z)]) +
                       (src.Bp[src.idx(p00.x, p00.y, pp1.z)] + src.Bp[src.idx(p00.x, p00.y, pn1.z)]) - src.divB[i])/6.0f;
        }
    }
}

template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT=typename Dim<VT3>::SIZE_T>
__global__ void magneticCurlPost_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT  ip   = makeI<IT>(ix, iy, iz);
      int i = src.idx(ip);
      IT p00 = applyBounds(ip,   src.size, cp); // current cell
      IT pp1 = applyBounds(ip+1, src.size, cp); // cell + (1,1,1)
      IT pn1 = applyBounds(ip-1, src.size, cp); // cell - (1,1,1)
      
      // if(p00 >= 0 && pp1 >= 0 && pn1 >= 0)
      //   {
      //     VT3 h = 1.0/makeV<VT3>(src.size);
      //     dst.B[i].x -= (cp.u.dt*(src.Bp[src.idx(pp1.x, p00.y, p00.z) - src.Bp[src.idx(pn1.x, p00.y, p00.z)]])) / (3.0f*h.x);
      //     dst.B[i].y -= (cp.u.dt*(src.Bp[src.idx(p00.x, pp1.y, p00.z) - src.Bp[src.idx(p00.x, pn1.y, p00.z)]])) / (3.0f*h.y);
      //     dst.B[i].z -= (cp.u.dt*(src.Bp[src.idx(p00.x, p00.y, pp1.z) - src.Bp[src.idx(p00.x, p00.y, pn1.z)]])) / (3.0f*h.z);
      //   }
      
      // apply B pressure to B field
      if(p00 >= 0 && pp1 >= 0 && pn1 >= 0)
        {
          VT3 h = 1.0/makeV<VT3>(src.size);
          dst.B[i].x -= (src.Bp[src.idx(pp1.x, p00.y, p00.z)] - src.Bp[src.idx(pn1.x, p00.y, p00.z)]) / (3.0f * h.x);
          dst.B[i].y -= (src.Bp[src.idx(p00.x, pp1.y, p00.z)] - src.Bp[src.idx(p00.x, pn1.y, p00.z)]) / (3.0f * h.y);
          dst.B[i].z -= (src.Bp[src.idx(p00.x, p00.y, pp1.z)] - src.Bp[src.idx(p00.x, p00.y, pn1.z)]) / (3.0f * h.z);
        }
      T p = src.p[i];   dst.p[i]   = (isvalid(p) ? p : 0);
      T d = src.div[i]; dst.div[i] = (isvalid(d) ? d : 0);
    }
}


// wrapper
template<typename T> void magneticCurl(FluidField<T> &src, FluidField<T> &dst, FluidParams<T> &cp, int iter)
{
  if(src.size > 0 && dst.size == src.size)
    {
      src.copyTo(dst);
      if(iter > 0)
        {
          FluidField<T> *temp1 = &src; FluidField<T> *temp2 = &dst;
          dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
          dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                    (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                    (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      
          magneticCurlPre_k<<<grid, threads>>>(*temp1, *temp2, cp); std::swap(temp1, temp2);
          for(int i = 0; i < iter; i++)
            { magneticCurlIter_k<<<grid, threads>>>(*temp1, *temp2, cp); std::swap(temp1, temp2); }
          if(temp2 == &dst) { temp2->copyTo(*temp1); std::swap(temp1, temp2); } // make sure temp2 == dst for post process
          magneticCurlPost_k<<<grid, threads>>>(*temp1, *temp2, cp);
        }
    }
}

// template instantiation
template void magneticCurl<float>(FluidField<float> &src, FluidField<float> &dst, FluidParams<float> &cp, int iter);
