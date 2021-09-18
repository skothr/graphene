#include "sim.cuh"
#include "sim.hpp"
using namespace grph;

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>

#include "states.h"
#include "params.h"
#include "fill.cuh"
#include "cuda-tools.cuh"
#include "physics.h"

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

#define FORWARD_EULER 1



// returns positive force vector at position p due to graphene Coulomb forces
template<typename T>
__host__ __device__ T GrapheneState<T>::forceAt(T p, const SimParams<T> &simParams)
{
  typedef T VT;
  using     ST = typename Dims<T>::BASE;
  using     IT = typename Dims<T>::SIZE_T;
  
  //IT fip = makeV<IT>(p); // location of 2D cell being affected
  
  // add electrostatic force from each cell of the graphene
  VT F = makeV<VT>(0.0f);
  for(int x = 0; x < this->size.x; x++)//-simParams.gEmRad; x < simParams.gEmRad; x++)
    //for(int x = (simParams.fp.edgeNX == EDGE_WRAP ? -this->size.x/2 : 0); x < this->size.x + (simParams.fp.edgePX == EDGE_WRAP ? this->size.x/2 : 0); x++)
    {
      IT ip = makeI<IT>((int)p.x + x, 0, 0);
      IT ip2 = applyBounds(ip, size, simParams);
      int i = idx(ip2);
      
      VT gp = (makeV<VT>(ip) + makeV<VT>(0.5f))/makeV<VT>(simParams.fp.fluidSize) * simParams.gp.simSize + simParams.gp.simPos; // apply graphene location/size
      //gp /= GRAPHENE_CARBON_DIST;
      VT dp = p - gp;
      
      ST dist_2 = dot(dp, dp);
      ST dTest  = sqrt(dist_2);
      if(dist_2 > 0.0f)// && dist_2 < simParams.gEmRad*simParams.gEmRad/simParams.gp.simSize.x/simParams.gp.simSize.x)
        {
          IT p2 = applyBounds(ip, size, simParams);
          if(ip2 >= 0 && ip2 < size)
            {
              int i2 = idx(ip2);
              F += coulombForce(1.0f, -qn[i2], dp); // q0 applied after loop (separately for qp and qn)
            }
        }
    }
  return F;
}

template<typename T>
__global__ void grapheneForce_k(FluidState<T> src, FluidState<T> dst, GrapheneState<T> gsrc, SimParams<T> params)
{
  typedef T VT;
  using     ST = typename Dims<VT>::BASE;
  using     IT = typename Dims<VT>::SIZE_T;
  
  int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  IT  ip   = makeI<IT>(ix, iy, iz);
  IT  size = src.size;
  if(ip < size)
    {
      int i = src.idx(ip);
      VT  p = makeV<VT>(ip);
      // VT  s = makeV<VT>(size);

      CellState<VT> fs = src[i];

      VT simP = (p / makeV<VT>(params.fp.fluidSize) * params.fp.simSize) + params.fp.simPos;
      //simP /= GRAPHENE_CARBON_DIST;
      VT F = gsrc.forceAt(simP, params) * params.gEmMult;
      
      VT pforce = F * fs.qp;
      VT nforce = F * -fs.qn;
      ST qnNew = fs.qn;
      ST qpNew = fs.qp;

      IT  gip = makeI<IT>(ix, 0, 0);
      int gi = gsrc.idx(gip);

      for(int x = 0; x < gsrc.size.x; x++)
        {
          IT gxi = makeI<IT>(x, 0, 0);
          VT gxP = (makeV<VT>(gxi) + makeV<VT>(0.5f))*params.gp.simSize + params.gp.simPos;
          //gxP /= GRAPHENE_CARBON_DIST;
          VT gF = coulombForce(-gsrc.qn[gsrc.idx(gxi)], src.qp[i]-src.qn[i],
                               gxP - simP)*params.gEmMult;
          texAtomicAdd(gsrc.qnv, gF, gxi, params);
        }
      
      dst.v[i] = fs.v;
      dst.d[i] = fs.d;
      dst.p[i] = fs.p;
      dst.div[i] = fs.div;
      
      dst.qn[i] = qnNew;
      dst.qp[i] = qpNew;
      dst.qnv[i] = fs.qnv + nforce*params.dt;
      dst.qpv[i] = fs.qpv + pforce*params.dt;
      
      dst.E[i] = pforce;
      dst.B[i] = fs.B;
    }
}



template<typename T>
__global__ void grapheneStep_k(GrapheneState<T> gsrc, GrapheneState<T> gdst, SimParams<T> params)
{
  typedef T VT;
  using     ST = typename Dims<VT>::BASE;
  using     IT = typename Dims<VT>::SIZE_T;
  
  int ix   = blockIdx.x*blockDim.x + threadIdx.x;
  int iy   = blockIdx.y*blockDim.y + threadIdx.y;
  int iz   = blockIdx.z*blockDim.z + threadIdx.z;
  IT  ip   = makeI<IT>(ix, iy, iz);
  IT  size = gsrc.size;
  if(ip < size)
    {
      int i = gsrc.idx(ip);
      gdst[i] = gsrc[i];
      
      VT  p  = VT{ST(ix), ST(iy)};
      ST  dt = params.dt;

      CellState<T> s;
      s.qn  = gsrc.qn [i];
      s.qnv = gsrc.qnv[i];
      
      // if(slipPlane(ip, params) || slipPlane(ip-1, params) || slipPlane(ip+1, params))
      //   { s.v = makeV<VT>(0); } // zero velocity on edges (assumes wall is static)
      
      // if(params.fp.edgeNX != EDGE_WRAP && ip.x == 0)        { s.qnv.x =  abs(s.qnv.x); } //abs(s.qnv.x); }
      // if(params.fp.edgePX != EDGE_WRAP && ip.x == size.x-1) { s.qnv.x = -abs(s.qnv.x); } //-abs(s.qnv.x); }
      s.qnv.y = 0.0f;

      VT F = makeV<VT>(0);

      // // add force from adjacent cells with charge
      // int emRad = params.emRad;
      // for(int x = -emRad; x <= emRad; x++)
      //     {
      //       IT dpi    = makeI<IT>(x, 0, 0); // TODO: z
      //       VT dp     = makeV<VT>(dpi) / makeV<VT>(params.gp.fluidSize) * gsrc.params.simSize + gsrc.params.simPos;
      //       ST dist_2 = dot(dp, dp);
      //       ST dTest  = sqrt(dist_2);
      //       if(dist_2 > 0.0f && dTest <= emRad)
      //        {
      //          IT p2 = applyBounds(ip + dpi, gsrc.size, params);
      //          if(p2.x >= 0 && p2.x < size.x)
      //            {
      //              int i2 = gsrc.idx(p2);
      //              ST q2 = gsrc.qn[i2];               // charge at other point
      //              F += coulombForce(1.0f, q2, -dp);  // q0 applied after loop (separately for qp and qn)
      //            }
      //         }
      //     }

      // EdgeType epxOld = params.fp.edgePX; EdgeType enxOld = params.fp.edgeNX;
      // if(epxOld == EDGE_WRAP) { params.fp.edgePX = EDGE_NOSLIP; } if(enxOld == EDGE_WRAP) { params.fp.edgeNX = EDGE_NOSLIP; }
        IT pp1 = applyBounds(ip+makeI<VT>( 1, 0, 0), gsrc.size, params); // adjacent cells
        IT pn1 = applyBounds(ip+makeI<VT>(-1, 0, 0), gsrc.size, params);
      // params.fp.edgePX = epxOld; params.fp.edgeNX = enxOld;
      
      int ppi = gsrc.idx(pp1); // x+1
      int pni = gsrc.idx(pn1); // x-1

      ST pq = gsrc.qn[ppi]; // adjacent charges
      ST nq = gsrc.qn[pni];
      
      VT pp1v = makeV<VT>(pp1-ip) / makeV<VT>(params.gp.fluidSize) * gsrc.params.simSize; // convert to physical vectors
      VT pn1v = makeV<VT>(pn1-ip) / makeV<VT>(params.gp.fluidSize) * gsrc.params.simSize;
      //pp1v /= GRAPHENE_CARBON_DIST; pn1v /= GRAPHENE_CARBON_DIST;
      
      F += coulombForce(s.qn, pq, pp1v); // add diffusion forces
      F += coulombForce(s.qn, nq, pn1v);

      s.qnv +=  F*-s.qn; // apply force to charge velocity
      //s.qnv.y = 0.0f;                    // constrain to 1D (TODO: 2D plane)
      
      // check for invalid values
      if(isnan(s.qn)    || isinf(s.qn))    { s.qn    = 0.0; }
      if(isnan(s.qnv.x) || isinf(s.qnv.x)) { s.qnv.x = 0.0; }
      
      // use forward Euler method
      VT     p2    = integrateForwardEuler(gsrc.qnv, p, s.qnv, dt);
      // add actively to next point in texture
      int4   tiX   = texPutIX   (p2, params);
      int4   tiY   = texPutIY   (p2, params);
      float4 mults = texPutMults(p2);
      IT     p00   = IT{tiX.x, tiY.x}; IT p10 = IT{tiX.y, tiY.y};
      IT     p01   = IT{tiX.z, tiY.z}; IT p11 = IT{tiX.w, tiY.w};

      // scale value by grid overlap and store in each location
      // // qn
      // texAtomicAdd(gdst.qn,  s.qn*(mults.x+mults.y),  p00+p10, params); texAtomicAdd(gdst.qn,  s.qn*(mults.z+mults.w),  p01+p11, params);
      // //texAtomicAdd(gdst.qn,  s.qn*mults.y,  p10, params); texAtomicAdd(gdst.qn,  s.qn*mults.w,  p11, params);
      // // qnv
      // texAtomicAdd(gdst.qnv, s.qnv*(mults.x+mults.y), p00+p10, params); texAtomicAdd(gdst.qnv, s.qnv*(mults.z+mults.w), p01+p11, params);
      // //texAtomicAdd(gdst.qnv, s.qnv*mults.y, p10, params); texAtomicAdd(gdst.qnv, s.qnv*mults.w, p11, params);
      texAtomicAdd(gdst.qn,  s.qn*mults.x,  p00, params); texAtomicAdd(gdst.qn,  s.qn*mults.z,  p01, params);
      texAtomicAdd(gdst.qn,  s.qn*mults.y,  p10, params); texAtomicAdd(gdst.qn,  s.qn*mults.w,  p11, params);
      texAtomicAdd(gdst.qnv, s.qnv*mults.x, p00, params); texAtomicAdd(gdst.qnv, s.qnv*mults.z, p01, params);
      texAtomicAdd(gdst.qnv, s.qnv*mults.y, p10, params); texAtomicAdd(gdst.qnv, s.qnv*mults.w, p11, params);
    }
}


#define SAMPLE_LINEAR 1
#define SAMPLE_POINT  0

template<typename T>
__global__ void grapheneRender_k(GrapheneState<T> gsrc, CudaTex<T> tex, SimParams<T> params)
{
  typedef T VT;
  using     ST = typename Dims<T>::BASE;
  using     IT = typename Dims<T>::SIZE_T;
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  int iz = blockIdx.z*blockDim.z + threadIdx.z;
  IT  ip = makeI<IT>(ix, iy, iz);
  if(ip >= 0 && ip < makeI<IT>(tex.size.x, tex.size.y, 0))
    {
      VT tp    = makeV<VT>(ip);
      VT tSize = makeV<VT>(IT{tex.size.x, tex.size.y});
      VT fSize = makeV<VT>(gsrc.size);
      VT fp    = ((tp + 0.5)/tSize) * fSize - 0.5;
      
      CellState<T> s;
#if   SAMPLE_LINEAR // linearly interpolated sampling
      VT fp0   = floor(fp); // lower index
      VT fp1   = fp0 + 1;   // upper index
      VT alpha = fp - fp0;  // fractional offset
      IT bp00 = applyBounds(makeV<IT>(fp0), gsrc.size, params);
      IT bp11 = applyBounds(makeV<IT>(fp1), gsrc.size, params);

      if(bp00 >= 0 && bp00 < gsrc.size && bp11 >= 0 && bp11 < gsrc.size)
        {
          IT bp01 = bp00; bp01.x = bp11.x; // x + 1
          IT bp10 = bp00; bp10.y = bp11.y; // y + 1
          
          s = lerp(lerp(gsrc[gsrc.idx(bp00)], gsrc[gsrc.idx(bp01)], alpha.x),
                   lerp(gsrc[gsrc.idx(bp10)], gsrc[gsrc.idx(bp11)], alpha.x), alpha.y);
        }
      else
        {
          int ti = iy*tex.size.x + ix;
          tex.data[ti] = float4{1.0f, 0.0f, 1.0f, 1.0f};
          return;
        }
      
#elif SAMPLE_POINT  // integer point sampling (render true cell areas)
      VT fp0 = floor(fp);  // lower index
      IT bp  = applyBounds(makeV<IT>(fp0), gsrc.size, params);
      if(bp >= 0 && bp < gsrc.size)
        { s = gsrc[gsrc.idx(ip)]; }
      else
        {
          int ti = iy*tex.size.x + ix;
          tex.data[ti] = float4{1.0f, 0.0f, 1.0f, 1.0f};
          return;
        }
#endif

      float4 color = float4{0.0f, 0.0f, 0.0f, 0.0f};
      
      ST vLen  = length(s.v);
      VT vn    = (vLen != 0.0f ? normalize(s.v) : makeV<VT>(0));
      ST nq    = s.qn;
      // ST pq    = s.qp;
      // ST q     = s.qn;
      VT qv    = s.qnv;
      ST qvLen = length(qv);
      VT qvn   = (qvLen != 0.0f ? normalize(qv) : makeV<VT>(0));
      
      ST Emag  = length(s.E);
      ST Bmag  = length(s.B);

      vn = abs(vn);
      
      // color += s.v.x * params.render.getParamMult(FLUID_RENDER_VX);
      // color += s.v.y * params.render.getParamMult(FLUID_RENDER_VY);
      //color += s.v.z * params.render.getParamMult(FLUID_RENDER_VZ);
      // color +=  vLen * params.render.getParamMult(FLUID_RENDER_VMAG);
      // color +=  vn.x * params.render.getParamMult(FLUID_RENDER_NVX);
      // color +=  vn.y * params.render.getParamMult(FLUID_RENDER_NVY);
      //color +=  vn.z * params.render.getParamMult(FLUID_RENDER_NVZ);
      // color += s.div * params.render.getParamMult(FLUID_RENDER_DIV);
      // color +=   s.d * params.render.getParamMult(FLUID_RENDER_D);
      // color +=   s.p * params.render.getParamMult(FLUID_RENDER_P);
      // color +=     q * params.render.getParamMult(FLUID_RENDER_Q);
      color +=    nq * params.render.getParamMult(FLUID_RENDER_NQ);
      // color +=    pq * params.render.getParamMult(FLUID_RENDER_PQ);
      // color +=  qv.x * params.render.getParamMult(FLUID_RENDER_QVX);
      // color +=  qv.y * params.render.getParamMult(FLUID_RENDER_QVY);
      //color +=  qv.z * params.render.getParamMult(FLUID_RENDER_QVZ);
      // color += qvLen * params.render.getParamMult(FLUID_RENDER_QVMAG);
      color += qvn.x * params.render.getParamMult(FLUID_RENDER_NQVX);
      // color += qvn.y * params.render.getParamMult(FLUID_RENDER_NQVY);
      //color += qvn.z * params.render.getParamMult(FLUID_RENDER_NQVZ);
      // color +=  Emag * params.render.getParamMult(FLUID_RENDER_E);
      // color +=  Bmag * params.render.getParamMult(FLUID_RENDER_B);      

      color = float4{ max(0.0f, min(color.x, 1.0f)), max(0.0f, min(color.y, 1.0f)),
                      max(0.0f, min(color.z, 1.0f)), max(0.0f, min(color.w, 1.0f)) };
      
      int ti = iy*tex.size.x + ix;
      tex.data[ti] = float4{color.x, color.y, color.z, 1.0f};
      // tex.data[ti] = float4{1,0,0, 1.0f};

    }
}

// template void grapheneForce <float2>(FluidState   <float2> &gsrc, FluidState   <float2> &gdst, GrapheneState<float2> &src, SimParams<float2> &params);

template<typename T>
void grapheneForce(FluidState<T> &src, FluidState<T> &dst, GrapheneState<T> &gsrc, SimParams<T> &params)
{
  if(src.size > 0 && dst.size == src.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(src.size.x/(float)BLOCKDIM_X),
                (int)ceil(src.size.y/(float)BLOCKDIM_Y));

      // NOTE: no atomic writes here -- 
// #if FORWARD_EULER
//       // set to zero for forward euler method -- kernel will re-add contents
      //grapheneClear_k<<<grid, threads>>>(gsrc);
// #endif // FORWARD_EULER
      
      grapheneForce_k<<<grid, threads>>>(src, dst, gsrc, params);
      getLastCudaError("====> ERROR: grapheneForce_k failed!");
    }
}

template<typename T>
void grapheneStep(GrapheneState<T> &gsrc, GrapheneState<T> &gdst, FluidState<T> &src, SimParams<T> &params)
{
  if(gsrc.size > 0 && gdst.size == gsrc.size)
    {
      dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
      dim3 grid((int)ceil(gsrc.size.x/(float)BLOCKDIM_X),
                (int)ceil(gsrc.size.y/(float)BLOCKDIM_Y));
#if FORWARD_EULER
      // set to zero for forward euler method -- kernel will re-add contents
      grapheneClear_k<<<grid, threads>>>(gdst);
#endif // FORWARD_EULER
      
      grapheneStep_k<<<grid, threads>>>(gsrc, gdst, params);
      getLastCudaError("====> ERROR: grapheneStep_k failed!");
    }
}

template<typename T>
void grapheneRender(GrapheneState<T> &gsrc, CudaTex<T> &tex, SimParams<T> &params)
{
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
  dim3 grid((int)ceil(tex.size.x/(float)BLOCKDIM_X),
            (int)ceil(tex.size.y/(float)BLOCKDIM_Y));
  float4 *texData = tex.map();
  grapheneRender_k<<<grid, threads>>>(gsrc, tex, params);
  cudaDeviceSynchronize();
  getLastCudaError("====> ERROR: grapheneRender_k failed!");
  tex.unmap();
}


template void grapheneForce <float2>(FluidState   <float2> &src,  FluidState   <float2> &dst,  GrapheneState<float2> &gsrc, SimParams<float2> &params);
template void grapheneStep  <float2>(GrapheneState<float2> &gsrc, GrapheneState<float2> &gdst, FluidState   <float2> &src,  SimParams<float2> &params);
template void grapheneRender<float2>(GrapheneState<float2> &gsrc, CudaTex<float2> &tex, SimParams<float2> &params);

template void grapheneForce <float3>(FluidState   <float3> &src,  FluidState   <float3> &dst,  GrapheneState<float3> &gsrc, SimParams<float3> &params);
template void grapheneStep  <float3>(GrapheneState<float3> &gsrc, GrapheneState<float3> &gdst, FluidState   <float3> &src,  SimParams<float3> &params);
template void grapheneRender<float3>(GrapheneState<float3> &gsrc, CudaTex<float3> &tex, SimParams<float3> &params);
