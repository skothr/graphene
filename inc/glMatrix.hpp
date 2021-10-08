#ifndef GL_MATRIX_HPP
#define GL_MATRIX_HPP

#include <array>
#include <iostream>
#include <iomanip>

#include "vector.hpp"

template<typename T, int N, int M=N> class glMatrix;

// shorthand types
typedef glMatrix<int,    2> Mat2i;
typedef glMatrix<int,    3> Mat3i;
typedef glMatrix<int,    4> Mat4i;
typedef glMatrix<float,  2> Mat2f;
typedef glMatrix<float,  3> Mat3f;
typedef glMatrix<float,  4> Mat4f;
typedef glMatrix<double, 2> Mat2d;
typedef glMatrix<double, 3> Mat3d;
typedef glMatrix<double, 4> Mat4d;

// glMatrix -- glMatrix<T,N> for NxN matrix, or glMatrix<T,N,M> for NxM matrix
template<typename T, int N, int M> // N --> ROWS, M --> COLS
class glMatrix
{
public: // (static)
  // static constexpr int ROWS = N; static constexpr int COLS = M; static constexpr int DATASIZE = N*M; typedef glMatrix<T,N,M> type;
  typedef Vector<T, M> RowVector; // row of the matrix
  typedef Vector<T, N> ColVector; // column of the matrix

private:

  union
  { // matrix data
    std::array<T,       N*M> mData;    // full data
    std::array<ColVector, M> mColumns; // as basis vectors (column major)
  };

  // utilities for calulating inverse matrix
  template<typename TT = glMatrix<T,N,M>>
  typename std::enable_if<(N == M), void>::type cofactor(glMatrix<T,N,M> &result, int row, int col, int dim) const
  {
    int i = 0; int j = 0;
    for(int x = 0; x < dim; x++)
      for(int y = 0; y < dim; y++)
        if(x != col && y != row)
          {
            result[j++][i] = mColumns[x][y];
            if(j == dim - 1) { j = 0; i++; }
          }
  }
  template<typename TT = glMatrix<T,N,M>>
  typename std::enable_if<(N == M), T>::type determinant(int dim) const
  {
    if(dim == 1) { return mColumns[0][0]; } // base case
    T det  = (T)0; T sign = (T)1; glMatrix<T,N,M> temp;
    for(int c = 0; c < dim; c++)
      {
        cofactor(temp, 0, c, dim);
        det += sign*mColumns[c][0] * temp.determinant(dim-1);
        sign = -sign;
      }
    return det;
  }
  template<typename TT = glMatrix<T,N,M>>
  typename std::enable_if<(N == M), TT>::type adjoint() const
  {
    T sign = 1; glMatrix<T,N,M> result, temp;
    if(N == 1) { result[0][0] = 1; return result; }
    for(int c = 0; c < N; c++)
      for(int r = 0; r < N; r++)
        {
          cofactor(temp, r, c, N);
          sign = ((r+c) % 2 == 0) ? 1 : -1;
          result[r][c] = sign*temp.determinant(N-1);
        }
    return result;
  }

public:
  glMatrix() { identity(); } // initialized to identity by default
  glMatrix(const std::array<ColVector, M> &colData) : mColumns(colData)  { }
  glMatrix(const std::array<T, N*M>       &matData) : mData(matData)     { }
  glMatrix(const glMatrix<T,N,M>          &other)   : mData(other.mData) { }

  // 
  template<typename TT = glMatrix<T,N,M>>
  static typename std::enable_if<(N == M), TT>::type makeIdentity()
  { return glMatrix<T,N,M>(); }
  template<typename TT = glMatrix<T,N,M>>
  static typename std::enable_if<(N == M), TT>::type makeTranslate(const Vector<T, N-1> &dPos)
  {
    glMatrix<T,N,M> result = makeIdentity();
    for(int i = 0; i < N-1; i++) { result[N-1][i] = dPos[i]; }
    return result;
  }
  template<typename TT = glMatrix<T,N,M>>
  static typename std::enable_if<(N == M), TT>::type makeScale(const Vector<T, N-1> &dScale)
  {
    glMatrix<T,N,M> result = makeIdentity();
    for(int i = 0; i < N-1; i++) { result[i][i] = dScale[i]; }
    return result;
  }

  template<typename TT = glMatrix<T,N,M>>
  static typename std::enable_if<(N == 4 && N == M), TT>::type makeProjection(float fov, float aspect, float near, float far)
  {
    T t = std::tan(fov/2)*near; // TOP
    T b = -t;                   // BOTTOM
    T r =  t * aspect;          // RIGHT
    T l = -t * aspect;          // LEFT
    glMatrix<T,N,M> mat;
    mat(0,0) = 2*near/(r-l);
    mat(1,1) = 2*near/(t-b);
    mat(2,2) = -(far+near)/(far-near);
    mat(0,2) = (r+l)/(r-l);
    mat(1,2) = (t+b)/(t-b);
    mat(3,2) = -1;
    mat(2,3) = -2*far*near/(far-near);
    mat(3,3) = 0;
    return mat;
  }

  template<typename TT = glMatrix<T,N,M>>
  static typename std::enable_if<(N == 4 && N == M), TT>::type makeLookAt(const Vector<T, 3> &pos, const Vector<T, 3> &focus,
                                                                          const Vector<T, 3> &upBasis=Vector<T, 3>(0,1,0))
  {
    // view vectors
    Vector<T, 3> eye   = normalize(pos-focus);
    Vector<T, 3> up    = upBasis;
    Vector<T, 3> right = normalize(cross(up, eye));
    up = cross(eye, right);
    // calculate lookat matrix
    glMatrix<T,N,M> basis;
    basis(0,0) = right.x; basis(0,1) = up.x; basis(0,2) = eye.x;
    basis(1,0) = right.y; basis(1,1) = up.y; basis(1,2) = eye.y;
    basis(2,0) = right.z; basis(2,1) = up.z; basis(2,2) = eye.z;
    glMatrix<T,N,M> posMat;
    posMat(0,3) = -pos.x; posMat(1,3) = -pos.y; posMat(2,3) = -pos.z;
    return (basis ^ posMat);
  }

  // data access
  T* data()             { return mData.data(); }
  const T* data() const { return mData.data(); }
  ColVector col(int c) const { return mColumns[c]; }
  RowVector row(int r) const { RowVector result; for(int c = 0; c < M; c++) { result[c] = mColumns[c][r]; } return result; }

  // matrix
  template<typename TT = glMatrix<T,N,M>> typename std::enable_if<(N == M), TT&>::type identity()
  { for(int c = 0; c < M; c++) for(int r = 0; r < N; r++) { mColumns[c][r] = (T)(r == c ? 1 : 0); }; return *this; }
  template<typename TT = glMatrix<T,N,M>> typename std::enable_if<(N == M), TT&>::type zero()
  { for(auto &d : mData) { d = (T)0.0; } return *this; }
  
  template<typename TT = glMatrix<T,N,M>> typename std::enable_if<(N == M), TT&>::type translate(const Vector<T, N-1> &dPos)
  { return ((*this) = glMatrix<T,N,M>::makeTranslate(dPos)^(*this)); }
  template<typename TT = glMatrix<T,N,M>> typename std::enable_if<(N == M), TT&>::type scale(const Vector<T, N-1> &dScale)
  { return ((*this) = glMatrix<T,N,M>::makeScale(dScale)^(*this)); }
  
  template<typename TT = glMatrix<T,N,M>> typename std::enable_if<(N == M), TT >::type translated(const Vector<T, N-1> &dPos) const
  { return (glMatrix<T,N,M>::makeTranslate(dPos)^(*this)); }
  template<typename TT = glMatrix<T,N,M>> typename std::enable_if<(N == M), TT >::type scaled(const Vector<T, N-1> &dScale) const
  { return (glMatrix<T,N,M>::makeScale(dScale)^(*this)); }

  // transpose
  glMatrix<T, M, N> transposed() const
  {
    glMatrix<T, M, N> result;
    for(int x = 0; x < M; x++) for(int y = 0; y < N; y++) { result[x][y] = mColumns[x][y]; }
    return result;
  }

public:

  // inverse
  template<typename TT = glMatrix<T,N,M>>
  typename std::enable_if<(N == M), TT>::type inverse() const
  {
    T det = determinant(N);
    if(det == 0) { std::cout << "====> WARNING: glMatrix doesn't have an inverse!\n"; return *this; } // no inverse
    return adjoint() / det;
  }

  // assignment
  glMatrix<T,N,M>& operator=(const glMatrix<T,N,M> &other) { mData = other.mData; return *this; }
  // comparison
  bool operator==(const glMatrix<T,N,M> &other) { for(int i = 0; i < N*M; i++) if(mData[i] != other.mData[i]) { return false; } return true;  }
  bool operator!=(const glMatrix<T,N,M> &other) { for(int i = 0; i < N*M; i++) if(mData[i] != other.mData[i]) { return true;  } return false; }
  // access (column vectors)
  const ColVector& operator[](int c) const { return mColumns[c]; }
  ColVector& operator[](int c)             { return mColumns[c]; }
  const T& operator()(int r, int c) const  { return mColumns[c][r]; }
  T& operator()(int r, int c)              { return mColumns[c][r]; }

  //////// MATH OPERATORS ////////
  // glMatrix + glMatrix
  glMatrix<T,N,M>& operator+=(const glMatrix<T,N,M> &rhs)       { for(int i = 0; i < N*M; i++) { mData[i] += rhs.mData[i]; } return *this; }
  glMatrix<T,N,M>  operator+ (const glMatrix<T,N,M> &rhs) const { glMatrix<T,N,M> result(*this); return (result += rhs); }
  // glMatrix - glMatrix
  glMatrix<T,N,M>& operator-=(const glMatrix<T,N,M> &rhs)       { for(int i = 0; i < N*M; i++) { mData[i] -= rhs.mData[i]; } return *this; }
  glMatrix<T,N,M>  operator- (const glMatrix<T,N,M> &rhs) const { glMatrix<T,N,M> result(*this); return (result -= rhs); }

  // ^= (matrix multiplication + assignment -- both must be square and the same size to overwrite)
  template<typename TT=glMatrix<T,N,M>>
  typename std::enable_if<(N == M), TT&>::type operator^=(const TT &rhs)
  {
    std::array<Vector<T, M>, N> result;
    for(int r = 0; r < N; r++) for(int c = 0; c < M; c++) { result[r][c] = dot(row(r), rhs.col(c)); }
    mColumns = result;
    return *this;
  }
  template<typename TT=glMatrix<T,N,M>>
  typename std::enable_if<(N == M), TT>::type operator^(const TT &rhs) const { TT result(*this); return (result ^= rhs); }
  
  // ^ (const matrix multiplication -- lhs rows)
  template<int NN=N, int MM=M, typename TT=glMatrix<T,NN,MM>>
  typename std::enable_if<((N != M && NN != MM) && M == NN), TT>::type operator^(const TT &rhs) const
  {
    glMatrix<T, N, MM> result;
    for(int r = 0; r < N; r++) for(int c = 0; c < MM; c++) { result[r][c] = dot(row(r), rhs.col(c)); }
    return result;
  }

  // % (matrix division -- multiply by inverse of rhs)
  template<int NN=N, int MM=M>
  typename std::enable_if<(NN == MM && M == NN), glMatrix<T,NN,MM>&>::type operator%=(const glMatrix<T,NN,MM> &rhs)
  { return (*this ^= rhs.inverse()); }
  template<int NN=N, int MM=M>
  typename std::enable_if<(NN == MM && M == NN), glMatrix<T,NN,MM> >::type operator% (const glMatrix<T,NN,MM> &rhs) const
  { glMatrix<T,NN,MM> result(*this); result %= rhs; return result; }
  
  // glMatrix / glMatrix (element-wise)
  glMatrix<T,N,M>& operator/=(const glMatrix<T,N,M> &rhs)       { for(int i = 0; i < N*M; i++) { mData[i] /= rhs; } return *this; }
  glMatrix<T,N,M>  operator/ (const glMatrix<T,N,M> &rhs) const { glMatrix<T, N, M> result(*this); return (result /= rhs); }
  // glMatrix / glMatrix (element-wise)
  glMatrix<T,N,M>& operator*=(const glMatrix<T,N,M> &rhs)       { for(int i = 0; i < N*M; i++) { mData[i] *= rhs; } return *this; }
  glMatrix<T,N,M>  operator* (const glMatrix<T,N,M> &rhs) const { glMatrix<T, N, M> result(*this); return (result *= rhs); }
  // glMatrix * Scalar
  glMatrix<T,N,M>& operator*=(const T &rhs)      { for(int i = 0; i < N*M; i++) { mData[i] *= rhs; } return *this; }
  glMatrix<T,N,M>  operator*(const T &rhs) const { glMatrix<T,N,M> result(*this); return (result *= rhs); }
  // glMatrix / Scalar
  glMatrix<T,N,M>& operator/=(const T &rhs)      { for(int i = 0; i < N*M; i++) { mData[i] /= rhs; } return *this; }
  glMatrix<T,N,M>  operator/(const T &rhs) const { glMatrix<T,N,M> result(*this); return (result /= rhs); }
  // scalar reverse order
  template<typename T2, int N2, int M2> friend glMatrix<T2,N2,M2> operator+(const T &lhs, const glMatrix<T2,N2,M2> &rhs);
  template<typename T2, int N2, int M2> friend glMatrix<T2,N2,M2> operator-(const T &lhs, const glMatrix<T2,N2,M2> &rhs);
  template<typename T2, int N2, int M2> friend glMatrix<T2,N2,M2> operator*(const T &lhs, const glMatrix<T2,N2,M2> &rhs);
  template<typename T2, int N2, int M2> friend glMatrix<T2,N2,M2> operator/(const T &lhs, const glMatrix<T2,N2,M2> &rhs);
  // printing
  template<typename T2, int N2, int M2> friend std::ostream& operator<<(std::ostream &os, const glMatrix<T2,N2,M2> &mat);
  std::string toString() const { std::stringstream ss; ss << *this; return ss.str(); }
};


// friend function definitions
template<typename T, int N, int M> inline glMatrix<T,N,M> operator+(const T &lhs, const glMatrix<T,N,M> &rhs) { return lhs + rhs; }
template<typename T, int N, int M>
inline glMatrix<T,N,M> operator-(const T &lhs, const glMatrix<T,N,M> &rhs)
{
  glMatrix<T,N,M> result = rhs;
  for(int i = 0; i < N*M; i++) { result.mData[i] = lhs - rhs.mData[i]; }
  return result;
}
template<typename T, int N, int M> inline glMatrix<T,N,M> operator*(const T &lhs, const glMatrix<T,N,M> &rhs) { return lhs * rhs; }
template<typename T, int N, int M> inline glMatrix<T,N,M> operator/(const T &lhs, const glMatrix<T,N,M> &rhs)
{
  glMatrix<T,N,M> result = rhs;
  for(int i = 0; i < N*M; i++) { result.mData[i] = lhs/rhs.mData[i]; }
  return result;
}

template<typename T, int N, int M>
inline std::ostream& operator<<(std::ostream &os, const glMatrix<T,N,M> &mat)
{
  os << "  ";
  for(int i = 0; i < 8*4+1; i++) { os << "="; }
  os << "\n  |";
  for(int r = 0; r < N; r++)
    for(int c = 0; c < M; c++)
      {
        os << std::right << std::internal << std::setprecision(5) << std::fixed << std::setw(10) << mat(r,c);
        os << ((r != N-1 || c != M-1) ? ((c == M-1) ? "|\n  |" : (" ")) : "|");
      }
  os << "\n  ";
  for(int i = 0; i < 8*4+1; i++) { os << "="; }
  os << "\n";
  return os;
}

#endif // MATRIX_HPP
