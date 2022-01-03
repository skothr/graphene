#include "em.cuh"

#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>

#include "cuda-tools.cuh"
#include "vector-calc.cuh"
#include "vector-operators.h"
#include "physics.h"
#include "fluid.cuh"
#include "mathParser.hpp"

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8
#define BLOCKDIM_Z 8

// update electric charge (Q)
template<typename T, typename VT3=typename cuda_vec<T, 3>::VT, typename IT=typename cuda_vec<VT3>::IT>
__global__ void updateCharge_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      int i = src.idx(ix, iy, iz);
      VT3 p = VT3{(T)ix, (T)iy, (T)iz};
      VT3 v0  = src.v[i];
      T Qn = src.Qn[i];
      T Qp = src.Qp[i];
      T Q  = (Qp-Qn);
      VT3 Qnv = src.Qnv[i];
      VT3 Qpv = src.Qpv[i];
      VT3 E   = src.E[i];
      VT3 B   = src.B[i];
      Material<float> mat = src.mat[i];

      VT3 p2;
      if(isImplicit(cp.qIntegration))
        { //// integrate for implicit methods (sample value from previous time point)
          // (+)
          if(cp.qIntegration == INTEGRATION_BACKWARD_EULER) { p2 = integrateBackwardEuler(src.Qpv.dData, p, Qpv, cp.u.dt); }
          Qp  = tex2DD(src.Qp.dData,  p2, cp);
          Qpv = tex2DD(src.Qpv.dData, p2, cp);
          // (-)
          if(cp.qIntegration == INTEGRATION_BACKWARD_EULER) { p2 = integrateBackwardEuler(src.Qnv.dData, p, Qnv, cp.u.dt); }
          Qn  = tex2DD(src.Qn.dData,  p2, cp);
          Qnv = tex2DD(src.Qnv.dData, p2, cp);
        }
      
      // calculate divergence of E --> ∇·E = q/ε₀
      // IT ip = makeV<IT>(ix, iy, iz);
      // IT p0 = applyBounds(ip,   src.size, cp); // current cell
      // IT pN = applyBounds(ip-1, src.size, cp); // cell - (1,1)
      // IT pP = applyBounds(ip+1, src.size, cp); // cell + (1,1)
      // Q = ((((pP.x < 0 || pP.x >= src.size.x) ? 0 : src.E[src.idx(pP.x, p0.y, p0.z)].x) -
      //       ((pN.x < 0 || pN.x >= src.size.x) ? 0 : src.E[src.idx(pN.x, p0.y, p0.z)].x)) +
      //      (((pP.y < 0 || pP.y >= src.size.y) ? 0 : src.E[src.idx(p0.x, pP.y, p0.z)].y) -
      //       ((pN.y < 0 || pN.y >= src.size.y) ? 0 : src.E[src.idx(p0.x, pN.y, p0.z)].y)) +
      //      (((pP.z < 0 || pP.z >= src.size.z) ? 0 : src.E[src.idx(p0.x, p0.y, pP.z)].z) -
      //       ((pN.z < 0 || pN.z >= src.size.z) ? 0 : src.E[src.idx(p0.x, p0.y, pN.z)].z))) / (2.0f * cp.u.dL);
      // Qp =  max((T)0, Q);
      // Qn = -min((T)0, Q);
      if(!isvalid(Qn)) { Qn = 0.0; } if(!isvalid(Qp)) { Qp = 0.0; }

      // lorentz forces
      Qnv += cp.u.dt*(-Qn*E + cross(-Qn*(Qnv) + v0, B)); // Fe = qE + q(v×B)
      Qpv += cp.u.dt*( Qp*E + cross( Qp*(Qpv) + v0, B)); // Fb = qE + q(v×B)
      
      // TODO: average surrounding cells (corners/faces(?))  ==> B field instead(?)
      // IT pE100 = applyBounds(IT{ix+1, iy,   iz  }, src.size, cp); IT pE010 = applyBounds(IT{ix,   iy+1, iz  }, src.size, cp);
      // IT pE110 = applyBounds(IT{ix+1, iy+1, iz  }, src.size, cp); IT pE001 = applyBounds(IT{ix,   iy,   iz+1}, src.size, cp);
      // IT pE101 = applyBounds(IT{ix+1, iy,   iz+1}, src.size, cp); IT pE011 = applyBounds(IT{ix,   iy+1, iz+1}, src.size, cp);
      // IT pE111 = applyBounds(IT{ix+1, iy+1, iz+1}, src.size, cp);
      // bool n100 = (pE100 < 0); bool n010 = (pE010 < 0); bool n110 = (pE110 < 0);
      // bool n001 = (pE001 < 0); bool n101 = (pE101 < 0); bool n011 = (pE011 < 0); bool n111 = (pE111 < 0);
      // E = E + ((!n001 ? src.E[src.idx(pE001)] : VT3{0,0,0}) + (!n010 ? src.E[src.idx(pE010)] : VT3{0,0,0}) +
      //          (!n011 ? src.E[src.idx(pE011)] : VT3{0,0,0}) + (!n100 ? src.E[src.idx(pE100)] : VT3{0,0,0}) +
      //          (!n101 ? src.E[src.idx(pE101)] : VT3{0,0,0}) + (!n110 ? src.E[src.idx(pE110)] : VT3{0,0,0}) +
      //          (!n111 ? src.E[src.idx(pE111)] : VT3{0,0,0})) / (1+int(n001)+int(n010)+int(n011)+int(n100)+int(n101)+int(n110)+int(n111));
      
      // check result values
      if(!isvalid(Qnv.x)) { Qnv.x = 0.0; } if(!isvalid(Qnv.y)) { Qnv.y = 0.0; } if(!isvalid(Qnv.z)) { Qnv.z = 0.0; }
      if(!isvalid(Qpv.x)) { Qpv.x = 0.0; } if(!isvalid(Qpv.y)) { Qpv.y = 0.0; } if(!isvalid(Qpv.z)) { Qpv.z = 0.0; }
      
      if(isImplicit(cp.qIntegration))
        { //// write final values for implicit methods (current cell)
          dst.Qp[i] = Qp; dst.Qpv[i] = Qpv;
          dst.Qn[i] = Qn; dst.Qnv[i] = Qnv;
        }
      else
        { //// integrate for explicit methods (write atomically to advected position)
          // (+)
          if(cp.qIntegration == INTEGRATION_FORWARD_EULER) { p2 = integrateForwardEuler(src.Qpv.dData, p, Qpv, cp.u.dt); }
          else if(cp.qIntegration == INTEGRATION_RK4)      { p2 = integrateRK4(src.Qpv, p, cp); }
          putTex2DD(dst.Qp.dData,  Qp,  p2, cp);
          putTex2DD(dst.Qpv.dData, Qpv, p2, cp);
          // (-)
          if(cp.qIntegration == INTEGRATION_FORWARD_EULER) { p2 = integrateForwardEuler(src.Qnv.dData, p, Qnv, cp.u.dt); }
          else if(cp.qIntegration == INTEGRATION_RK4)      { p2 = integrateRK4(src.Qnv, p, cp); }
          putTex2DD(dst.Qn.dData,  Qn,  p2, cp);
          putTex2DD(dst.Qnv.dData, Qnv, p2, cp);
        }
    }

  
  
}

///////////////////////////////////////////
//// SIMULATION -- MAXWELL'S EQUATIONS ////
///////////////////////////////////////////
//// Uses a Yee grid / Yee's method (more info: https://empossible.net/wp-content/uploads/2019/08/Lecture-4b-Maxwells-Equations-on-a-Yee-Grid.pdf)
//// - E and H are staggered -- E is measured at cell origins, and B/H at cell face centers

// electric field E
template<typename T, typename VT3=typename cuda_vec<T, 3>::VT, typename IT=typename cuda_vec<VT3>::IT>
__global__ void updateElectric_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;

  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT  ip0 = IT{ix, iy, iz};
      int i0  = src.idx(ip0);

      // check for boundary and cancel reflections
      //if(!cp.reflect)
        {
          const int bs = 1;
          int xOffset = (((cp.edgeNX == BOUND_WRAP || cp.edgePX == BOUND_WRAP)) ? 0 :
                         (src.size.x <= bs ? 0 : ((ip0.x < bs ? 1 : 0) + (ip0.x >= src.size.x-1-bs ? -1 : 0))));
          int yOffset = (((cp.edgeNY == BOUND_WRAP || cp.edgePY == BOUND_WRAP)) ? 0 :
                         (src.size.y <= bs ? 0 : ((ip0.y < bs ? 1 : 0) + (ip0.y >= src.size.y-1-bs ? -1 : 0))));
          int zOffset = (((cp.edgeNZ == BOUND_WRAP || cp.edgePZ == BOUND_WRAP)) ? 0 :
                         (src.size.z <= bs ? 0 : ((ip0.z < bs ? 1 : 0) + (ip0.z >= src.size.z-1-bs ? -1 : 0))));
          if(xOffset != 0 || yOffset != 0 || zOffset != 0)
            {
              int i = src.idx(max(0, min(src.size.x-1, ip0.x + xOffset)),
                              max(0, min(src.size.y-1, ip0.y + yOffset)),
                              max(0, min(src.size.z-1, ip0.z + zOffset)));
              dst.E[i0] = src.E[i];
              return;
            }
        }
      
      VT3 v0   = src.v[i0];
      T   Qn0  = src.Qn[i0];
      T   Qp0  = src.Qp[i0];
      VT3 Qnv0 = src.Qnv[i0];
      VT3 Qpv0 = src.Qpv[i0];
      VT3 E0   = src.E[i0];
      VT3 B0   = src.B[i0];
      Material<T> M0 = src.mat[i0];
      if(M0.vacuum()) { M0 = cp.u.vacuum(); } // check if vacuum
      typename Material<T>::Blend ab = M0.getBlendE(cp.u.dt, cp.u.dL);
      T Q = (Qp0-Qn0);
      
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
      VT3 J = (Qp0*Qpv0 - Qn0*Qnv0 + Q*v0)/(cp.u.dL*cp.u.dL); // (averaged over cell faces (?))
      if(!isvalid(J.x)) { J.x = 0.0; }
      if(!isvalid(J.y)) { J.y = 0.0; }
      if(!isvalid(J.z)) { J.z = 0.0; }
      if(!isvalid(J))   { J = VT3{0,0,0}; }
      
      dEdt -= (J/cp.u.e0)*cp.u.dt;
      if(!isvalid(dEdt.x)) { dEdt.x = 0; }
      if(!isvalid(dEdt.y)) { dEdt.y = 0; }
      if(!isvalid(dEdt.z)) { dEdt.z = 0; }
      if(!isvalid(dEdt))   { dEdt = VT3{0,0,0}; }
      
      VT3 newE = ab.alpha*E0 + ab.beta*dEdt; // blend 
      if(!isvalid(newE.x)) { newE.x = 0; }
      if(!isvalid(newE.y)) { newE.y = 0; }
      if(!isvalid(newE.z)) { newE.z = 0; }
      if(!isvalid(newE))   { newE = VT3{0,0,0}; }

      dst.E[i0] = newE;   // updated values
    }
}

// magnetic field B
template<typename T, typename VT3=typename cuda_vec<T, 3>::VT, typename IT=typename cuda_vec<VT3>::IT>
__global__ void updateMagnetic_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;

  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT  ip0 = IT{ix, iy, iz};
      int i0  = src.idx(ip0);
      // check for boundary and cancel reflections
      //if(!cp.reflect)
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
              dst.B[i0] = src.B[i];
              return;
            }
        }
      
      VT3 v0  = src.v[i0];
      VT3 B0  = src.B[i0];
      VT3 E0  = src.E[i0];
      Material<T> M0 = src.mat[i0];
      if(M0.vacuum()) { M0 = cp.u.vacuum(); } // check if vacuum
      typename Material<T>::Blend ab = M0.getBlendB(cp.u.dt, cp.u.dL);

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
      if(!isvalid(dBdt))   { dBdt = VT3{0,0,0}; }
      // if(!isvalid(dBdt.x)) { dBdt.x = 0; }
      // if(!isvalid(dBdt.y)) { dBdt.y = 0; }
      // if(!isvalid(dBdt.z)) { dBdt.z = 0; }
      
      VT3 newB = ab.alpha*B0 - ab.beta*dBdt;
      if(!isvalid(newB))   { newB = VT3{0,0,0}; }
      // if(!isvalid(newB.x)) { newB.x = 0; }
      // if(!isvalid(newB.y)) { newB.y = 0; }
      // if(!isvalid(newB.z)) { newB.z = 0; }
      dst.B[i0] = newB;
    }
}


//// COULOMB FORCES --> [...] ////
////  discrete point charge model, calculated by applying Coulomb's law based on neighboring cells/distances (within effective radius)
template<typename T, typename VT3=typename cuda_vec<T, 3>::VT, typename IT=typename cuda_vec<VT3>::IT>
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
              IT ip1   = ip + IT{x,y,z};
              VT3 dp   = VT3{(T)-x,(T)-y,(T)-z}; // vector from ip1 to ip  // + VT3{0.5, 0.5, 0.5}); // (alignment?)
              T dist_2 = length2(dp);
              if(dist_2 > 0 && dist_2 <= cp.rCoulomb*cp.rCoulomb)
                {
                  IT ip2 = applyBounds(ip1, src.size, cp);
                  if(ip2.x >= 0 && ip2.y >= 0 && ip2.z >= 0)
                    {
                      int i2 = src.idx(ip2);
                      T Q2 = src.Qp[i2] - src.Qn[i2]; // charge at other point
                      F += Q2 * dp / (dist_2*sqrt(dist_2));
                    }
                }
            }
      dst.E[i] = src.E[i] + cp.u.dt*(F * cp.u.e * cp.u.e);  // (e --> elementary charge)
    }
}


//// REMOVE DIVERGENCE OF B --> ∇·B = 0 ////
////// NOTE: very similar to fluid pressure... ////
template<typename T, typename VT3=typename cuda_vec<T, 3>::VT, typename IT=typename cuda_vec<VT3>::IT>
__global__ void updateDivBPre_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT ip = makeV<IT>(ix, iy, iz);
      int i = src.idx(ip);
      IT p0 = ip; // applyBounds(ip,   src.size, cp); // current cell
      IT pN = applyBounds(ip-1, src.size, cp); // cell - (1,1)
      IT pP = applyBounds(ip+1, src.size, cp); // cell + (1,1)
      
      // calculate divergence of B
      T Bd = ((((pP.x >= 0 && pP.x < src.size.x) ? src.B[src.idx(pP.x, p0.y, p0.z)].x : 0) -
               ((pN.x >= 0 && pN.x < src.size.x) ? src.B[src.idx(pN.x, p0.y, p0.z)].x : 0)) +
              (((pP.y >= 0 && pP.y < src.size.y) ? src.B[src.idx(p0.x, pP.y, p0.z)].y : 0) -
               ((pN.y >= 0 && pN.y < src.size.y) ? src.B[src.idx(p0.x, pN.y, p0.z)].y : 0)) +
              (((pP.z >= 0 && pP.z < src.size.z) ? src.B[src.idx(p0.x, p0.y, pP.z)].z : 0) -
               ((pN.z >= 0 && pN.z < src.size.z) ? src.B[src.idx(p0.x, p0.y, pN.z)].z : 0))) / (2.0*cp.u.dL);
      dst.divB[i] = (isvalid(Bd) ? Bd : 0);
    }
}

template<typename T, typename VT3=typename cuda_vec<T, 3>::VT, typename IT=typename cuda_vec<VT3>::IT>
__global__ void updateDivBIter_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT ip = makeV<IT>(ix, iy, iz);
      int i = src.idx(ip);
      IT p0 = ip; // applyBounds(ip,   src.size, cp); // current cell
      IT pP = applyBounds(ip+1, src.size, cp); // cell + (1,1,1)
      IT pN = applyBounds(ip-1, src.size, cp); // cell - (1,1,1)

      // iterate -- remove divergence (TODO)      
      T Bp = 0;
      int count = 0;
      if(pP.x > 0 && pP.x < src.size.x-1) { Bp += src.Bp[src.idx(pP.x, p0.y, p0.z)]; count++; }
      if(pP.y > 0 && pP.y < src.size.y-1) { Bp += src.Bp[src.idx(p0.x, pP.y, p0.z)]; count++; }
      if(pP.z > 0 && pP.z < src.size.z-1) { Bp += src.Bp[src.idx(p0.x, p0.y, pP.z)]; count++; }
      if(pN.x > 0 && pN.x < src.size.x-1) { Bp += src.Bp[src.idx(pN.x, p0.y, p0.z)]; count++; }
      if(pN.y > 0 && pN.y < src.size.y-1) { Bp += src.Bp[src.idx(p0.x, pN.y, p0.z)]; count++; }
      if(pN.z > 0 && pN.z < src.size.z-1) { Bp += src.Bp[src.idx(p0.x, p0.y, pN.z)]; count++; }
      Bp  = count > 0 ? ((Bp/(T)count - src.divB[i]/6.0)) : Bp;
      dst.Bp[i] = (isvalid(Bp) ? Bp : 0);
    }
}

template<typename T, typename VT3=typename cuda_vec<T, 3>::VT, typename IT=typename cuda_vec<VT3>::IT>
__global__ void updateDivBPost_k(FluidField<T> src, FluidField<T> dst, FluidParams<T> cp)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  if(ix < src.size.x && iy < src.size.y && iz < src.size.z)
    {
      IT ip = makeV<IT>(ix, iy, iz);
      int i = src.idx(ip);
      IT p0 = ip; // applyBounds(ip,   src.size, cp); // current cell
      IT pP = applyBounds(ip+1, src.size, cp); // cell + (1,1,1)
      IT pN = applyBounds(ip-1, src.size, cp); // cell - (1,1,1)

      
      // remove gradient of magnetic "pressure" from B 
      VT3 B  = src.B[i];
      VT3 dB = VT3{0,0,0};
      VT3 dL = VT3{2,2,2}*cp.u.dL;
      if(pP.x >= 0 && pP.x < src.size.x) { dB.x += src.Bp[src.idx(pP.x, p0.y, p0.z)]; }
      if(pP.y >= 0 && pP.y < src.size.y) { dB.y += src.Bp[src.idx(p0.x, pP.y, p0.z)]; }
      if(pP.z >= 0 && pP.z < src.size.z) { dB.z += src.Bp[src.idx(p0.x, p0.y, pP.z)]; }
      if(pN.x >= 0 && pN.x < src.size.x) { dB.x -= src.Bp[src.idx(pN.x, p0.y, p0.z)]; }
      if(pN.y >= 0 && pN.y < src.size.y) { dB.y -= src.Bp[src.idx(p0.x, pN.y, p0.z)]; }
      if(pN.z >= 0 && pN.z < src.size.z) { dB.z -= src.Bp[src.idx(p0.x, p0.y, pN.z)]; }
      dst.B[i] = B - dB/dL; // dB/(dL*cp.density); // <-- μ or μ₀ for "density"?
    }
}







// wrappers
template<typename T> void updateCharge(FluidField<T> &src, FluidField<T> &dst, const FluidParams<T> &cp)
{
  if(src.size.x > 0 && src.size.y > 0 && src.size.z > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      if(!isImplicit(cp.qIntegration))
        { // clear advected dst fields if using explitic integration method (written to atomically)
          dst.Qn.clear();
          dst.Qp.clear();
          dst.Qnv.clear();
          dst.Qpv.clear();
        }
      getLastCudaError("updateCharge() ==> copy cleared");
      // copy unmodified fields
      src.v.copyTo(dst.v);
      src.p.copyTo(dst.p);
      src.E.copyTo(dst.E);
      src.B.copyTo(dst.B);
      src.mat.copyTo(dst.mat);
      src.Bp.copyTo(dst.Bp);
      getLastCudaError("updateCharge() ==> copy unmodified");
      
      updateCharge_k<<<grid, threads>>>(src, dst, cp);
    }
  else { std::cout << "==> WARNING: skipped updateCharge (" << src.size << " / " << dst.size << ")\n"; }
  getLastCudaError("updateCharge()");
}

template<typename T> void updateElectric(FluidField<T> &src, FluidField<T> &dst, const FluidParams<T> &cp)
{
  if(src.size.x > 0 && src.size.y > 0 && src.size.z > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      // copy unmodified fields
      src.v.copyTo(dst.v);   src.p.copyTo(dst.p);
      src.Qn.copyTo(dst.Qn); src.Qp.copyTo(dst.Qp);   src.Qnv.copyTo(dst.Qnv); src.Qpv.copyTo(dst.Qpv);
      src.B.copyTo(dst.B);   src.mat.copyTo(dst.mat); src.Bp.copyTo(dst.Bp);
      // run kernel
      updateElectric_k<<<grid, threads>>>(src, dst, cp);
    }
  else { std::cout << "==> WARNING: skipped updateElectric (" << src.size << " / " << dst.size << ")\n"; }
  getLastCudaError("updateElectric()");
}

template<typename T> void updateMagnetic(FluidField<T> &src, FluidField<T> &dst, const FluidParams<T> &cp)
{
  if(src.size.x > 0 && src.size.y > 0 && src.size.z > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      // copy unmodified fields
      src.v.copyTo(dst.v);   src.p.copyTo(dst.p);
      src.Qn.copyTo(dst.Qn); src.Qp.copyTo(dst.Qp);   src.Qnv.copyTo(dst.Qnv); src.Qpv.copyTo(dst.Qpv);
      src.E.copyTo(dst.E);   src.mat.copyTo(dst.mat); src.Bp.copyTo(dst.Bp);
      getLastCudaError("updateMagnetic() ==> copy unmodified");
      // run kernel
      updateMagnetic_k<<<grid, threads>>>(src, dst, cp);
    }
  else { std::cout << "==> WARNING: skipped updateMagnetic2D (src: " << src.size << " / dst: " << dst.size << ")\n"; }
  getLastCudaError("updateMagnetic()");
}

template<typename T> void updateCoulomb(FluidField<T> &src, FluidField<T> &dst, const FluidParams<T> &cp)
{
  if(src.size > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                (int)ceil(src.size.z/(float)BLOCKDIM_Z));
      // copy unmodified fields
      src.v.copyTo(dst.v);   src.p.copyTo(dst.p);
      src.Qn.copyTo(dst.Qn); src.Qp.copyTo(dst.Qp);   src.Qnv.copyTo(dst.Qnv); src.Qpv.copyTo(dst.Qpv);
      src.B.copyTo(dst.B);   src.mat.copyTo(dst.mat); src.Bp.copyTo(dst.Bp);
      getLastCudaError("updateElectric() ==> copy unmodified");
      // run kernel
      updateCoulomb_k<<<grid, threads>>>(src, dst, cp);
    }
  getLastCudaError("updateCoulomb()");
}

template<typename T> void updateDivB(FluidField<T> &src, FluidField<T> &dst, const FluidParams<T> &cp, int iter)
{
  if(src.size > 0 && dst.size == src.size)
    {
      src.B.copyTo(dst.B); src.Bp.copyTo(dst.Bp);
      getLastCudaError("updateDivB() ==> copy src-->dst");
      if(iter > 0)
        {
          dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
          dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                    (int)ceil(src.size.y/(float)BLOCKDIM_Y),
                    (int)ceil(src.size.z/(float)BLOCKDIM_Z));
          FluidField<T> *temp1 = &src; FluidField<T> *temp2 = &dst;
          
          // pre
          updateDivBIter_k<<<grid, threads>>>(*temp1, *temp2, cp);     std::swap(temp1, temp2);
          getLastCudaError("updateDivB() ==> pre");
          temp1->divB.copyTo(temp2->divB); // copy updated divergence to both fields
          getLastCudaError("updateDivB() ==> pre-copy");
          
          // iteration
          for(int i = 0; i < iter; i++)
            {
              updateDivBIter_k<<<grid, threads>>>(*temp1, *temp2, cp); std::swap(temp1, temp2);
              getLastCudaError(("updateDivB() ==> iter " + std::to_string(i)).c_str());
            }
          
          // post
          updateDivBPost_k<<<grid, threads>>>(*temp1, *temp2, cp);     std::swap(temp1, temp2);
          getLastCudaError("updateDivB() ==> post");
          
          if(temp1 != &dst) // make sure final result is in dst
            {
              temp1->B.copyTo(dst.B);
              getLastCudaError("updateDivB() ==> post-copy B");
              temp1->Bp.copyTo(dst.Bp);
              getLastCudaError("updateDivB() ==> post-copy Bp");
            }
        }
    }
  getLastCudaError("updateDivB()");
}


// template instantiation
template void updateCharge  <float>(FluidField<float> &src, FluidField<float> &dst, const FluidParams<float> &cp);
template void updateElectric<float>(FluidField<float> &src, FluidField<float> &dst, const FluidParams<float> &cp);
template void updateMagnetic<float>(FluidField<float> &src, FluidField<float> &dst, const FluidParams<float> &cp);
template void updateCoulomb <float>(FluidField<float> &src, FluidField<float> &dst, const FluidParams<float> &cp);
template void updateDivB    <float>(FluidField<float> &src, FluidField<float> &dst, const FluidParams<float> &cp, int iter);
