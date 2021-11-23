#ifndef MATH_PARSER_HPP
#define MATH_PARSER_HPP

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cctype>
#include <sstream>

#include "vector-operators.h"

#define M_PHI ((sqrt(5.0)+1.0)/2.0)

enum MathFunction
  {
   MATHFUNC_BAD  = -1,
   MATHFUNC_NONE =  0,
   // scalar or vector (element-wise)
   MATHFUNC_SIN  = 1,
   MATHFUNC_COS,
   MATHFUNC_TAN,
   MATHFUNC_LOG,
   MATHFUNC_EXP,
   MATHFUNC_ABS,
   MATHFUNC_SQRT,
   MATHFUNC_POS, // max(x, 0)
   MATHFUNC_NEG, // min(x, 0)
   // vector
   MATHFUNC_LEN,
   MATHFUNC_NORM,
   //
   MATHFUNC_COUNT
  };

enum MathConstant
  {
   MATHCONST_BAD  = -1,
   MATHCONST_NONE =  0,
   //
   MATHCONST_PI   =  1,
   MATHCONST_PHI,
   //
   MATHCONST_COUNT
  };

enum MathOperator
  {
   MATHOP_BAD  = -1,
   MATHOP_NONE =  0,
   MATHOP_ADD,
   MATHOP_SUB,
   MATHOP_MUL,
   MATHOP_DIV,
   MATHOP_POW,
   MATHOP_MOD,
   MATHOP_MIN,
   MATHOP_MAX,
   
   MATHOP_COUNT
  };



// apply math function
template<typename T>
__host__ __device__ inline T applyFunc(MathFunction f, T arg)
{
  switch(f)                
    {
    case MATHFUNC_SIN:  return (T)sin(arg);
    case MATHFUNC_COS:  return (T)cos(arg);
    case MATHFUNC_TAN:  return (T)tan(arg);
    case MATHFUNC_LOG:  return (T)log(arg);
    case MATHFUNC_EXP:  return (T)exp(arg);
    case MATHFUNC_ABS:  return (T)abs(arg);
    case MATHFUNC_SQRT: return (T)sqrt(arg);
    case MATHFUNC_POS:  return arg > (T)0 ? arg : (T)0; //(T)max((T)arg, (T)0);
    case MATHFUNC_NEG:  return arg < (T)0 ? arg : (T)0; //(T)min((T)arg, (T)0);
    default:            return arg;
    }
}
// apply math function
template<>
__host__ __device__ inline float3 applyFunc<float3>(MathFunction f, float3 arg)
{
  switch(f)                
    {
    case MATHFUNC_SIN:  return sin(arg);
    case MATHFUNC_COS:  return cos(arg);
    case MATHFUNC_TAN:  return tan(arg);
    case MATHFUNC_LOG:  return log(arg);
    case MATHFUNC_EXP:  return exp(arg);
    case MATHFUNC_ABS:  return abs(arg);
    case MATHFUNC_SQRT: return sqrt(arg);
    case MATHFUNC_POS:  return float3{(arg.x > 0 ? arg.x : 0), (arg.y > 0 ? arg.y : 0), (arg.z > 0 ? arg.z : 0)};
    case MATHFUNC_NEG:  return float3{(arg.x < 0 ? arg.x : 0), (arg.y < 0 ? arg.y : 0), (arg.z < 0 ? arg.z : 0)};
    case MATHFUNC_LEN:  return float3{1.0f, 1.0f, 1.0f}*length(arg);
    case MATHFUNC_NORM: return normalize(arg);
    default:            return arg;
    }
}


// get constant value
template<typename T>
__host__ __device__ inline T getConst(MathConstant c)
{
  switch(c)                
    {
    case MATHCONST_PI:  return makeV<T>((typename Dim<T>::BASE_T)(M_PI));
    case MATHCONST_PHI: return makeV<T>((typename Dim<T>::BASE_T)(M_PHI));
    default:            return T();
    }
}


//// HOST-ONLY ////

// naming
__host__ inline std::string getName(MathFunction f)
{
  switch(f)
    {
    case MATHFUNC_NONE: return "<NONE>";
    case MATHFUNC_SIN:  return "sin";
    case MATHFUNC_COS:  return "cos";
    case MATHFUNC_TAN:  return "tan";
    case MATHFUNC_LOG:  return "log";
    case MATHFUNC_EXP:  return "exp";
    case MATHFUNC_ABS:  return "abs";
    case MATHFUNC_SQRT: return "sqrt";
    case MATHFUNC_POS:  return "pos";
    case MATHFUNC_NEG:  return "neg";
    case MATHFUNC_LEN:  return "len";
    case MATHFUNC_NORM: return "norm";
    default:            return "<?>";
    }
}
__host__ inline std::string getName(MathConstant c)
{
  switch(c)
    {
    case MATHCONST_NONE: return "<NONE>";
    case MATHCONST_PI:   return "pi";
    case MATHCONST_PHI:  return "phi";
    default:             return "<?>";
    }
}

// operator overloads
__host__ inline std::ostream& operator<<(std::ostream &os, MathFunction f) { os << getName(f); return os; }
__host__ inline std::ostream& operator<<(std::ostream &os, MathConstant c) { os << getName(c); return os; }

__host__ inline std::ostream& printVerbose(int level)
{
  for(int i = 0; i < level; i++) { std::cout << "====| "; }
  return std::cout;
}

// find function (returns type and extracts argument from string)
__host__ inline MathFunction findFunc(std::string &str)
{
  auto openParen = str.find("(");
  if(openParen != std::string::npos)
    {
      auto closeParen = str.rfind(")");
      std::string fname = str.substr(0, openParen);
      for(int i = (int)MATHFUNC_NONE+1; i < (int)MATHFUNC_COUNT; i++)
        {
          if(fname == getName((MathFunction)i))
            {
              str = str.substr(openParen+1, closeParen-openParen-1);
              return (MathFunction)i;
            }
        }
    }
  return MATHFUNC_NONE;
}
// find constant (returns type)
__host__ inline MathConstant findConst(const std::string &str)
{
  for(int i = (int)MATHCONST_NONE+1; i < (int)MATHCONST_COUNT; i++)
    {
      if(str == getName((MathConstant)i))
        { return (MathConstant)i; }
    }
  return MATHCONST_NONE;
}

// toString
__host__ inline std::string toString(MathFunction f, const std::string &arg="") { return (f == MATHFUNC_NONE ? arg : (getName(f)+"("+arg+")")); }
__host__ inline std::string toString(MathConstant c)                            { return getName(c); }




template<typename T>
class Expression
{
public:
  MathFunction ftype  = MATHFUNC_NONE;
  MathConstant ctype  = MATHCONST_NONE;
  bool parentheses = false;
  
  virtual T calculate(const std::map<std::string, T> &vars={}, bool verbose=false, int level=0) const { return T(); }
  //__device__ virtual T calculate(int level=0) const { return T(); }
  virtual std::string toString(bool addParen=false) const = 0;
  virtual void release() { }
  virtual bool isTerm() const { return false; }
  void print() const { std::cout << toString(); }
};


template<typename T>
class Term : public Expression<T>
{
public:
  std::string valStr = "";
  T           value = T();
  
  Term(const std::string &val, bool paren=false)
    : valStr(val)
  {
    this->parentheses = paren;
    this->ftype = findFunc(valStr);
    this->ctype = findConst(valStr);
  }
  
  virtual std::string toString(bool addParen=false) const override
  { return (this->parentheses ? "(" : "") + ::toString(this->ftype, valStr) + (this->parentheses ? ")" : ""); }
  
  virtual T calculate(const std::map<std::string, T> &vars={}, bool verbose=false, int level=0) const override
  {
    T result = T();
    if(this->ctype != MATHCONST_NONE)
      {
        result = getConst<T>(this->ctype);
        if(verbose) { printVerbose(level) << " TERM CONSTANT --> " << getName(this->ctype) << " = " << result << "\n"; }
      }
    else
      {
        std::string str = toString();
        auto iter = vars.find(str);
        if(iter != vars.end())
          {
            result = iter->second;
            if(verbose) { printVerbose(level) << " TERM VARIABLE --> " << iter->first << " = " << result << "\n"; }
          }
        else
          { // try to convert to type T
            std::stringstream(valStr) >> result;
            if(verbose) { printVerbose(level) << " TERM NUMERIC(?) --> " << valStr << " = " << result << " (?)\n"; }
          }
      }
    return applyFunc(this->ftype, result);
  }
  virtual bool isTerm() const override { return true; }
};

template<typename T>
class ExpNode : public Expression<T>
{
public:
  Expression<T> *mLeft  = nullptr;
  Expression<T> *mRight = nullptr;
  char mOp = '\0';
  
  ExpNode(char op_, Expression<T> *leftExp, Expression<T> *rightExp, bool paren=false)
    : mOp(op_), mLeft(leftExp), mRight(rightExp) { this->parentheses = paren; }
  
  virtual T calculate(const std::map<std::string, T> &vars, bool verbose=false, int level=0) const override
  {
    if(verbose) { printVerbose(level) << " EXP LEFT (ftype: " << getName(this->ftype) << (mLeft ? " (" + getName(mLeft->ftype) + ")" : "") << ") --> \n"; }
    T leftVal  = (mLeft  ? mLeft->calculate(vars, verbose, level+1) : T());
    if(verbose)
      {
        printVerbose(level) << (mLeft ? "    (LEFT DONE)" : "    <LEFT NULL>") << "\n";
        printVerbose(level) << " EXP RIGHT (ftype=" << getName(this->ftype) << (mRight ? " (" + getName(mRight->ftype) + ")" : "") << ") --> \n";
      }
    T rightVal = (mRight ? mRight->calculate(vars, verbose, level+1) : T());
    if(verbose)
      {
        printVerbose(level) << (mRight ? "    (RIGHT DONE)" : "    <RIGHT NULL>") << "\n";
        printVerbose(level) << " EXP OP --> " << leftVal << " " << mOp << " " << rightVal << " ==> ";
      }
    T result = T();
    bool badOp = false;
    switch(mOp)
      {
      case '+':  result = leftVal + rightVal;      break;
      case '-':  result = leftVal - rightVal;      break;
      case '*':  result = leftVal * rightVal;      break;
      case '/':  result = leftVal / rightVal;      break;
      case '^':  result = pow(leftVal, rightVal);  break;
      case '%':  result = fmod(leftVal, rightVal); break;
      case '>':  result = (leftVal > rightVal ? leftVal: rightVal); break;//max(leftVal, rightVal);  break;
      case '<':  result = (leftVal < rightVal ? leftVal: rightVal); break; //min(leftVal, rightVal);  break;
      case '\0': result = (mLeft ? leftVal : (mRight ? rightVal : T())); break;
      default:  badOp = true;
      }
    if(verbose) { std::cout << result << "\n"; }

    if(badOp)
      {
        std::cout << "====> WARNING(ExpNode::calculate): Unknown operator --> '"
                  << std::string(mOp, 1) << (((int)mOp <= 20) ? " ("+std::to_string((int)mOp)+")" : "") << "\n";
      }
    
    if(verbose) { printVerbose(level) << "    (OP DONE)\n"; }
    return applyFunc(this->ftype, result);
  }

  virtual std::string toString(bool addParen=false) const override
  {
    std::stringstream ss;
    if(mLeft)       { ss << mLeft->toString(addParen); }
    if(mOp != '\0') { ss << mOp; }
    if(mRight)      { ss << mRight->toString(addParen); }
    
    return (this->parentheses||addParen ? "(" : "") + ::toString(this->ftype, ss.str()) + (this->parentheses||addParen ? ")" : "");
  }
  void release()
  {
    if(mLeft)  { mLeft->release();  delete mLeft;  mLeft  = nullptr; }
    if(mRight) { mRight->release(); delete mRight; mRight = nullptr; }
  }
};




// convert string to expression
template<typename T>
inline Expression<T>* toExpression(std::string str, bool paren=false, bool verbose=false)
{
  // remove whitespace
  str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
  // remove extra closing parentheses
  int numOpen = 0; int numClose = 0;
  int level = 0;
  for(int i = 0; i < str.size(); i++)
    {
      if(str[i] ==')')
        {
          if(level >= 0) { str.erase(i); }
          else           { level++; }
          numClose++;
        }
      else if (str[i] =='(') { numOpen++; level--; }
    }
  // for(int i = 0; i < numClose-numOpen; i++) { str.erase(str.rfind(")")); }
  for(int i = 0; i < numOpen-numClose; i++) { str += ")"; }
  
  if(str.size() == 0) { return nullptr; }
  
  level = 0;
  // + -
  for(int i = str.size()-1; i >= 0; i--)
    {
      char c = str[i];
      if     (c == ')') { level++; }
      else if(c == '(') { level--; }
      else if(level <= 0 && ((c == '+' || c == '-') && i != 0))
        {
          std::string l = str.substr(0,i); std::string r = str.substr(i+1); // split at operator
          bool lParen = (l[0] == '(' && l.back() == ')'); bool rParen = (r[0] == '(' && r.back() == ')');
          return new ExpNode<T>(c,
                                toExpression<T>(lParen ? l.substr(1, l.size()-1) : l, lParen),
                                toExpression<T>(rParen ? r.substr(1, r.size()-1) : r, rParen), paren);
        }
    }
  
  // * /
  for(int i = str.size()-1; i >= 0; i--)
    {
      char c = str[i];
      if     (c == ')') { level++; }
      else if(c == '(') { level--; }
      else if(level <= 0 && ((c == '*' || c == '/') && i != 0))
        {
          std::string l = str.substr(0,i); std::string r = str.substr(i+1); // split at operator
          bool lParen = (l[0] == '(' && l.back() == ')'); bool rParen = (r[0] == '(' && r.back() == ')');
          return new ExpNode<T>(c,
                                toExpression<T>(lParen ? l.substr(1, l.size()-1) : l, lParen),
                                toExpression<T>(rParen ? r.substr(1, r.size()-1) : r, rParen), paren);
        }
    }
  
  // ^ %
  for(int i = str.size()-1; i >= 0; i--)
    {
      char c = str[i];
      if(c == ')')      { level++; }
      else if(c == '(') { level--; }
      else if(level <= 0 && ((c == '^' || c == '%') && i != 0))
        {
          std::string l = str.substr(0,i); std::string r = str.substr(i+1); // split at operator
          bool lParen = (l[0] == '(' && l.back() == ')'); bool rParen = (r[0] == '(' && r.back() == ')');
          return new ExpNode<T>(c,
                                toExpression<T>(lParen ? l.substr(1, l.size()-1) : l, lParen),
                                toExpression<T>(rParen ? r.substr(1, r.size()-1) : r, rParen), paren);
        }
    }

  auto openParen  = str.find("(");
  auto closeParen = str.find(")");
  auto lastClose  = str.rfind(")");
  if(str[0]=='(')
    {
      for(int i=0; i<str.size(); i++)
        {
          if     (str[i]=='(') { level++; }
          else if(str[i]==')')
            {
              level--;
              if(level==0) { return toExpression<T>(str.substr(1, i-1), true, verbose); } //)); }
            }
        }
    }
  else if(openParen != std::string::npos && lastClose != std::string::npos)
    {
      std::string  arg   = str;
      MathFunction ftype = findFunc(arg);

      if(arg == str) // prevent stack overflow
        { return new Term<T>(str, paren); }
      
      ExpNode<T> *newExp = new ExpNode<T>('\0', toExpression<T>(arg, false, verbose), nullptr, false);
      newExp->ftype = ftype;
      return newExp;
    }
  else { return new Term<T>(str, paren); } // value
  
  return nullptr;
}





//// CUDA DEVICE IMPLEMENTATION WRAPPERS ////

template<typename T>
struct CudaExpression
{
  // if constant value
  bool constant   = false;
  T    constValue = T();
  
  // if single variable value
  int varIndex = -1;
  
  // if multiple-term expression
  CudaExpression<T> *dLeftExpr  = nullptr;
  CudaExpression<T> *dRightExpr = nullptr;
  MathOperator op = MATHOP_NONE; // operator to apply, if applicable

  // if function modifying expression
  MathFunction ftype = MATHFUNC_NONE;
  
  __device__ T calculate(T *vars) const
  {
    T result = T();
    if(constant)                  { result = constValue; }
    else if(vars && varIndex > 0) { result = vars[varIndex]; }
    else
      { // multiple terms, or invalid
        T leftVal = T(); T rightVal = T();
        if(dLeftExpr)  { leftVal  = dLeftExpr->calculate(vars); }
        if(dRightExpr) { rightVal = dRightExpr->calculate(vars); }
        switch(op)
          {
          case MATHOP_ADD: result = leftVal + rightVal;      break;
          case MATHOP_SUB: result = leftVal - rightVal;      break;
          case MATHOP_MUL: result = leftVal * rightVal;      break;
          case MATHOP_DIV: result = leftVal / rightVal;      break;
          case MATHOP_POW: result = pow(leftVal, rightVal);  break;
          case MATHOP_MOD: result = fmod(leftVal, rightVal); break;
          default:         result = leftVal + rightVal; // addition by default (one or both should be zero)
          }
      }
    return (ftype >= MATHFUNC_NONE ? applyFunc(ftype, result) : result);
  }

  ~CudaExpression() { release(); }
  
  bool releaseNeeded = false; // set to true once copied over to device
  void release()
  {
    if(releaseNeeded)
      {
        if(dLeftExpr)  { dLeftExpr->release();  cudaFree(dLeftExpr);  dLeftExpr  = nullptr; }
        if(dRightExpr) { dRightExpr->release(); cudaFree(dRightExpr); dRightExpr = nullptr; }
      }
    releaseNeeded = false;
  }  
};


template<typename T>
CudaExpression<T>* toCudaExpression(Expression<T> *expr, const std::vector<std::string> &varNames={})
{
  if(!expr) { return nullptr; }
  CudaExpression<T> cudaExpr;

  // check Expression type
  Term<T>    *term = reinterpret_cast<Term<T>*>(expr);
  ExpNode<T> *node = reinterpret_cast<ExpNode<T>*>(expr);
  if(term && expr->isTerm())
    {
      T result = T();
      if(std::isdigit(term->valStr[0]) || term->valStr[0] == '-') // check for explicit value (scalar)
        {
          typename Dim<T>::BASE_T scalar = 0;
          std::stringstream ss(term->valStr); ss >> scalar;
          result = makeV<T>((typename Dim<T>::BASE_T)1)*scalar;
        }
      else if(term->valStr[0] == '<')      // check for explicit value (scalar)
        { std::stringstream ss(term->valStr); ss >> result; }
      
      if(term->ctype != MATHCONST_NONE)
        { // constant value (predefined)
          cudaExpr.constant = true;
          cudaExpr.constValue = getConst<T>(term->ctype);
        }
      else if(result != T())
        { // constant value (custom)
          cudaExpr.constant = true;
          cudaExpr.constValue = result;
        }
      else
        { // check variable names
          for(int i = 0; i < varNames.size(); i++)
            {
              if(varNames[i] == term->toString())
                { cudaExpr.varIndex = i; break; }
            }
        }
    }
  else if(node && !expr->isTerm())
    {
      cudaExpr.dLeftExpr  = node->mLeft  ? toCudaExpression(node->mLeft,  varNames) : nullptr;
      cudaExpr.dRightExpr = node->mRight ? toCudaExpression(node->mRight, varNames) : nullptr;
      switch(node->mOp)
        {
        case '+': cudaExpr.op = MATHOP_ADD;  break;
        case '-': cudaExpr.op = MATHOP_SUB;  break;
        case '*': cudaExpr.op = MATHOP_MUL;  break;
        case '/': cudaExpr.op = MATHOP_DIV;  break;
        case '^': cudaExpr.op = MATHOP_POW;  break;
        case '%': cudaExpr.op = MATHOP_MOD;  break;
        case '>': cudaExpr.op = MATHOP_MAX;  break;
        case '<': cudaExpr.op = MATHOP_MIN;  break;
        default:  cudaExpr.op = MATHOP_NONE; break;
        }
    }
  
  cudaExpr.ftype = term->ftype; // applied function
  
  CudaExpression<T> *ptr = nullptr;
  // create device pointer
  cudaMalloc((void**)&ptr,   sizeof(CudaExpression<T>));
  // copy expression
  cudaExpr.releaseNeeded = true; // must be false on host copy when deconstructor runs
  cudaMemcpy(ptr, &cudaExpr, sizeof(CudaExpression<T>), cudaMemcpyHostToDevice);
  cudaExpr.releaseNeeded = false;
  return ptr;
}




// template<typename T>
// class CudaExpression
// {
// public:
//   int nVars = 0;
//   MathFunction ftype  = MATHFUNC_NONE;
//   MathConstant ctype  = MATHCONST_NONE;
//   bool parentheses = false;
//   __host__ __device__ virtual T calculate() const { return T(); }
//   __host__            virtual void release() { }
// };

// template<typename T>
// class CudaTerm : public CudaExpression<T>
// {
// private:
//   T value = T();
  
//   T *dVars = nullptr;

  
// public:
//   CudaTerm() { }
  
//   // setup CUDA device data from host
//   __host__ CudaTerm(Term &hTerm, const std::vector<std::string> &vNames)
//   {
//     nVars = vNames.size();
    
//     cudaMalloc((void**)&dVars, sizeof(T)*vNames.size());
//   }

//   __host__ __device__ virtual T calculate() const override
//   {
//     T result = T();
//     if(ctype != MATHCONST_NONE)
//       { result = getConst<T>(ctype); }
//     else
//       {
//         std::string str = toString();
//         bool found = false;
//         for(int i = 0; i < vNames.size(); i++)
//           {
//             if(vNames[i] == str)
//               {
//                 result = vVals[i];
//                 found = true;
//                 break;
//               }
//           }
//         if(!found) // try to convert to type T
//           { std::stringstream(valStr) >> result; }
//       }
//     return applyFunc(ftype, result);
//   }
// };

// template<typename T>
// class CudaExpNode : public CudaExpression<T>
// {
// public:
//   CudaExpression<T> *mLeft;
//   CudaExpression<T> *mRight;
//   char mOp;
  
//   CudaExpNode(char op_, CudaExpression<T> *leftExp, CudaExpression<T> *rightExp, bool paren=false)
//     : mOp(op_), mLeft(leftExp), mRight(rightExp) { parentheses = paren; }
  
//   __host__ __device__ virtual T calculate() const override
//   {
//     T leftVal  = (mLeft  ? mLeft->calculate(vNames, vVals, level+1) : T());
//     T rightVal = (mRight ? mRight->calculate(vNames, vVals, level+1) : T());
//     T result = T();
//     bool badOp = false;
//     switch(mOp)
//       {
//       case '+':  result = leftVal + rightVal;      break;
//       case '-':  result = leftVal - rightVal;      break;
//       case '*':  result = leftVal * rightVal;      break;
//       case '/':  result = leftVal / rightVal;      break;
//       case '^':  result = pow(leftVal, rightVal);  break;
//       case '%':  result = fmod(leftVal, rightVal); break;
//       case '\0': result = (mLeft ? leftVal :
//                            (mRight ? rightVal :
//                             (0)));                 break;
//       default:  badOp = true;
//       }
//     return applyFunc(ftype, result);
//   }
//   __host__ void release()
//   {
//     if(mLeft)  { mLeft->release();  delete mLeft;  mLeft  = nullptr; }
//     if(mRight) { mRight->release(); delete mRight; mRight = nullptr; }
//   }
// };




// CudaExpression toCudaExpression(Expression &hExpr)
// {
//   T result = T();
//   if(ctype != MATHCONST_NONE)
//     { result = getConst<T>(ctype); }
//   else
//     {
//       std::string str = toString();
//       auto iter = vars.find(str);
//       if(iter != vars.end())
//         {
//           {
//             result = iter->second;
//             break;
//           }
//         }
//       else
//         { // try to convert to type T
//           std::stringstream(valStr) >> result;
//         }
//     }
//   return applyFunc(ftype, result);  
// }



#endif // MATH_PARSER_HPP
