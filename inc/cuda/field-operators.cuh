#ifndef FIELD_OPERATORS_CUH
#define FIELD_OPERATORS_CUH

#ifdef __NVCC__

////////////////////////////////////////////////////////////////////////////////////////////////
//// vector calculus helpers
////////////////////////////////////////////////////////////////////////////////////////////////

//// GRADIENT --> ∇[T]
template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ VT3 gradient(const Field<T> &src, const IT3 &p, T dL)
{
  T vXn = src[src.idx(p.x, p.y, p.z)]; T vYn = src[src.idx(p.x, p.y, p.z)]; T vZn = src[src.idx(p.x, p.y, p.z)];
  T vXp = src[src.idx(p.x+1, p.y, p.z)]; T vYp = src[src.idx(p.x, p.y+1, p.z)]; T vZp = src[src.idx(p.x, p.y, p.z+1)];
  return VT3{(vXp-vXn), (vYp-vYn), (vZp-vZn)} / (2.0*dL);
}
template<typename T, typename VT3=typename DimType<T, 3>::VEC_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ VT3 gradient(const Field<T> &src, const IT3 &p0, const IT3 &pp, const IT3 &pn, T dL)
{
  T vXn = src[src.idx(p0.x, p0.y, p0.z)]; T vYn = src[src.idx(p0.x, p0.y, p0.z)]; T vZn = src[src.idx(p0.x, p0.y, p0.z)];
  T vXp = src[src.idx(pp.x, p0.y, p0.z)]; T vYp = src[src.idx(p0.x, pp.y, p0.z)]; T vZp = src[src.idx(p0.x, p0.y, pp.z)];
  return VT3{(vXp-vXn), (vYp-vYn), (vZp-vZn)} / (2.0*dL);
}



//// DIVERGENCE --> ∇·[VT3]
// (from cell center)
template<typename VT3, typename T=typename Dim<VT3>::BASE_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ T divergence(const Field<VT3> &src, const IT3 &p, T dL)
{
  T vxXn = src[src.idx(p.x-1, p.y, p.z)].x; T vyYn = src[src.idx(p.x, p.y-1, p.z)].y; T vzZn = src[src.idx(p.x, p.y, p.z-1)].z;
  T vxXp = src[src.idx(p.x+1, p.y, p.z)].x; T vyYp = src[src.idx(p.x, p.y+1, p.z)].y; T vzZp = src[src.idx(p.x, p.y, p.z+1)].z;
  return ((vxXp-vxXn) + (vyYp-vyYn) + (vzZp-vzZn)) / (2.0*dL);
}
template<typename VT3, typename T=typename Dim<VT3>::BASE_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ T divergence(const Field<VT3> &src, const IT3 &p0, const IT3 &pp, const IT3 &pn, T dL)
{
  T vxXn = src[src.idx(pn.x, p0.y, p0.z)].x; T vyYn = src[src.idx(p0.x, pn.y, p0.z)].y; T vzZn = src[src.idx(p0.x, p0.y, pn.z)].z;
  T vxXp = src[src.idx(pp.x, p0.y, p0.z)].x; T vyYp = src[src.idx(p0.x, pp.y, p0.z)].y; T vzZp = src[src.idx(p0.x, p0.y, pp.z)].z;
  return ((vxXp-vxXn) + (vyYp-vyYn) + (vzZp-vzZn)) / (2.0*dL);
}
// (from cell origin)
template<typename VT3, typename T=typename Dim<VT3>::BASE_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ T divergence2(const Field<VT3> &src, const IT3 &p, T dL)
{
  VT3 v0 = src[src.idx(p)];
  T vxXn = src[src.idx(p.x-1, p.y, p.z)].x; T vyYn = src[src.idx(p.x, p.y-1, p.z)].y; T vzZn = src[src.idx(p.x, p.y, p.z-1)].z;
  return ((v0.x-vxXn) + (v0.y-vyYn) + (v0.z-vzZn)) / dL;
}
template<typename VT3, typename T=typename Dim<VT3>::BASE_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ T divergence2(const Field<VT3> &src, const IT3 &p0, const IT3 &pn, T dL)
{
  VT3 v0 = src[src.idx(p0)];
  T vxXn = src[src.idx(pn.x, p0.y, p0.z)].x; T vyYn = src[src.idx(p0.x, pn.y, p0.z)].y; T vzZn = src[src.idx(p0.x, p0.y, pn.z)].z;
  return ((v0.x-vxXn) + (v0.y-vyYn) + (v0.z-vzZn)) / dL;
}



//// CURL --> ∇×[VT3]
template<typename VT3, typename T=typename Dim<VT3>::BASE_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ VT3 curl(const Field<VT3> &src, const IT3 &p, T dL)
{
  VT3 vXn = src[src.idx(p.x-1, p.y, p.z)]; VT3 vYn = src[src.idx(p.x, p.y-1, p.z)]; VT3 vZn = src[src.idx(p.x, p.y, p.z-1)];
  VT3 vpX = src[src.idx(p.x+1, p.y, p.z)]; VT3 vpY = src[src.idx(p.x, p.y+1, p.z)]; VT3 vpZ = src[src.idx(p.x, p.y, p.z+1)];
  return (VT3{(vpY.z-vYn.z)-(vpZ.y-vZn.y),
              (vpZ.x-vZn.x)-(vpX.z-vXn.z),
              (vpX.y-vXn.y)-(vpY.x-vYn.x)}) / (2.0*dL);
}
template<typename VT3, typename T=typename Dim<VT3>::BASE_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ VT3 curl(const Field<VT3> &src, const IT3 &p0, const IT3 &pp, const IT3 &pn, T dL)
{
  VT3 v0  = src[src.idx(p0)];
  VT3 vXn = src[src.idx(pn.x, p0.y, p0.z)]; VT3 vYn = src[src.idx(p0.x, pn.y, p0.z)]; VT3 vZn = src[src.idx(p0.x, p0.y, pn.z)];
  VT3 vpX = src[src.idx(pp.x, p0.y, p0.z)]; VT3 vpY = src[src.idx(p0.x, pp.y, p0.z)]; VT3 vpZ = src[src.idx(p0.x, p0.y, pp.z)];
  VT3 c = (VT3{((vpY.z-v0.z)+(v0.z-vYn.z)) - ((vpZ.y-v0.y)+(v0.y-vZn.y)),
               ((vpZ.x-v0.x)+(v0.x-vZn.x)) - ((vpX.z-v0.z)+(v0.z-vXn.z)),
               ((vpX.y-v0.y)+(v0.y-vXn.y)) - ((vpY.x-v0.x)+(v0.x-vYn.x))}) / (2.0*dL);
  return c;
}






// LAPLACIAN --> ∇²[VT3] --> ∇·∇[VT3] 
template<typename T, typename VT3=typename DimType<T,3>::VEC_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ T laplacian(const Field<T> &src, const IT3 &p, T dL)
{
  T v0   = src[src.idx(p)];
  T vxNX = src[src.idx(p.x-1, p.y, p.z)]; T vyNY = src[src.idx(p.x, p.y-1, p.z)]; T vzNZ = src[src.idx(p.x, p.y, p.z-1)];
  T vxPX = src[src.idx(p.x+1, p.y, p.z)]; T vyPY = src[src.idx(p.x, p.y+1, p.z)]; T vzPZ = src[src.idx(p.x, p.y, p.z+1)];
  return ((vxPX-vxNX) + (vyPY-vyNY) + (vzPZ-vzNZ))/(dL*dL);
}
template<typename T, typename VT3=typename DimType<T,3>::VEC_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ T laplacian(const Field<T> &src, const IT3 &p0, const IT3 &pp, const IT3 &pn, T dL)
{
  T v0   = src[src.idx(p0)];
  T vxNX = src[src.idx(pn.x, p0.y, p0.z)]; T vyNY = src[src.idx(p0.x, pn.y, p0.z)]; T vzNZ = src[src.idx(p0.x, p0.y, pn.z)];
  T vxPX = src[src.idx(pp.x, p0.y, p0.z)]; T vyPY = src[src.idx(p0.x, pp.y, p0.z)]; T vzPZ = src[src.idx(p0.x, p0.y, pp.z)];
  return ((vxPX-vxNX) + (vyPY-vyNY) + (vzPZ-vzNZ))/(dL*dL);
}
// ∇²[VT3]
template<typename VT3, typename T=typename Dim<VT3>::BASE_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ T laplacian(const Field<VT3> &src, const IT3 &p, T dL)
{
  VT3 v0 = src[src.idx(p)];
  T vxNX = src[src.idx(p.x-1, p.y, p.z)].x; T vyNY = src[src.idx(p.x, p.y-1, p.z)].y; T vzNZ = src[src.idx(p.x, p.y, p.z-1)].z;
  T vxPX = src[src.idx(p.x+1, p.y, p.z)].x; T vyPY = src[src.idx(p.x, p.y+1, p.z)].y; T vzPZ = src[src.idx(p.x, p.y, p.z+1)].z;
  return ((vxPX-vxNX) + (vyPY-vyNY) + (vzPZ-vzNZ))/(dL*dL);
}
template<typename VT3, typename T=typename Dim<VT3>::BASE_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ T laplacian(const Field<VT3> &src, const IT3 &p0, const IT3 &pp, const IT3 &pn, T dL)
{
  VT3 v0 = src[src.idx(p0)];
  T vxNX = src[src.idx(pn.x, p0.y, p0.z)].x; T vyNY = src[src.idx(p0.x, pn.y, p0.z)].y; T vzNZ = src[src.idx(p0.x, p0.y, pn.z)].z;
  T vxPX = src[src.idx(pp.x, p0.y, p0.z)].x; T vyPY = src[src.idx(p0.x, pp.y, p0.z)].y; T vzPZ = src[src.idx(p0.x, p0.y, pp.z)].z;
  return ((vxPX-vxNX) + (vyPY-vyNY) + (vzPZ-vzNZ))/(dL*dL);
}



// jacobi iteration
template<typename VT3, typename T=typename Dim<VT3>::BASE_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ VT3 jacobi(const Field<VT3> &src, const IT3 &p, T alpha, T beta, T dL, T dt)
{
  VT3 v0  = src[src.idx(p)];
  VT3 vNX = src[src.idx(p.x-1, p.y, p.z)]; VT3 vNY = src[src.idx(p.x, p.y-1, p.z)]; VT3 vNZ = src[src.idx(p.x, p.y, p.z-1)];
  VT3 vPX = src[src.idx(p.x+1, p.y, p.z)]; VT3 vPY = src[src.idx(p.x, p.y+1, p.z)]; VT3 vPZ = src[src.idx(p.x, p.y, p.z+1)];
  return (vNX+vPX + vNY+vPY + vNZ+vPZ + v0*alpha)/beta;
}
// jacobi iteration
template<typename VT3, typename T=typename Dim<VT3>::BASE_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ VT3 jacobi(const Field<VT3> &src, const IT3 &p0, const IT3 &p1p, const IT3 &p1n, T alpha, T beta)
{
  VT3 v0  = src[src.idx(p0)];
  VT3 vNX = src[src.idx(p1n.x, p0.y, p0.z)]; VT3 vNY = src[src.idx(p0.x, p1n.y, p0.z)]; VT3 vNZ = src[src.idx(p0.x, p0.y, p1n.z)];
  VT3 vPX = src[src.idx(p1p.x, p0.y, p0.z)]; VT3 vPY = src[src.idx(p0.x, p1p.y, p0.z)]; VT3 vPZ = src[src.idx(p0.x, p0.y, p1p.z)];
  return (vNX+vPX + vNY+vPY + vNZ+vPZ + v0*alpha)/beta;
}


// jacobi iteration
template<typename VT3, typename T=typename Dim<VT3>::BASE_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ T jacobi(const Field<T> &src, const IT3 &p, T alpha, T beta, T dL, T dt)
{
  VT3 v0  = src[src.idx(p)];
  VT3 vNX = src[src.idx(p.x-1, p.y, p.z)]; VT3 vNY = src[src.idx(p.x, p.y-1, p.z)]; VT3 vNZ = src[src.idx(p.x, p.y, p.z-1)];
  VT3 vPX = src[src.idx(p.x+1, p.y, p.z)]; VT3 vPY = src[src.idx(p.x, p.y+1, p.z)]; VT3 vPZ = src[src.idx(p.x, p.y, p.z+1)];
  return (vNX+vPX + vNY+vPY + vNZ+vPZ + v0*alpha)/beta;
}
// jacobi iteration
template<typename VT3, typename T=typename Dim<VT3>::BASE_T, typename IT3=typename Dim<VT3>::SIZE_T>
__device__ T jacobi(const Field<T> &src, const IT3 &p0, const IT3 &p1p, const IT3 &p1n, T alpha, T beta)
{
  VT3 v0  = src[src.idx(p0)];
  VT3 vNX = src[src.idx(p1n.x, p0.y, p0.z)]; VT3 vNY = src[src.idx(p0.x, p1n.y, p0.z)]; VT3 vNZ = src[src.idx(p0.x, p0.y, p1n.z)];
  VT3 vPX = src[src.idx(p1p.x, p0.y, p0.z)]; VT3 vPY = src[src.idx(p0.x, p1p.y, p0.z)]; VT3 vPZ = src[src.idx(p0.x, p0.y, p1p.z)];
  return (vNX+vPX + vNY+vPY + vNZ+vPZ + v0*alpha)/beta;
}

#endif // NVCC

#endif // FIELD_OPERATORS_CUH
