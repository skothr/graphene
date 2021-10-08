#ifndef RANGE_HPP
#define RANGE_HPP

template<typename T>
struct Range
{
  T lower = (T)0;
  T upper = (T)0;
  
  Range() = default;
  Range(const T &l, const T &u) : lower(l), upper(u) { }
  Range(const Range &other) : lower(other.lower), upper(other.upper) { }
  
  // assignment
  Range& operator= (const Range &other)       { lower = other.lower; upper = other.upper; return *this; }
  // equivalence
  bool   operator==(const Range &other) const { return (lower == other.lower && upper == other.upper); }
  bool   operator!=(const Range &other) const { return (lower != other.lower || upper != other.upper); }
  
  T     span() const                    { return upper - lower; }                                 // space covered
  bool  contains(const T &x) const      { return (x >= lower && x <= upper); }                    // (boundary inclusive)
  bool  clip    (const T &x) const      { return std::max(std::min(x, upper), lower); }           // closest value in range
  
  void  extend  (const T &amount)       { lower -= amount; upper += amount; }                     // expand or shrink this
  Range extended(const T &amount) const { return Range(lower-amount, upper+amount); }             // expand or shrink
  void  fit(const T &x)                 { lower = std::min(lower,x); upper = std::max(upper,x); } // extend to contain the given data point

  // template<typename T>
  // inline bool intersects(const Range<T> &r)
  // { return (r2.lower >= r1.upper && r1.lower <= r2.upper); }
  
  // inline bool intersects(const Range<T> &r1, const Range<T> &r2)
  // { return (r2.lower >= r1.upper && r1.lower <= r2.upper); }

  // const Range<T>& template<typename T>
  
  // template<typename T>
  // const Range<T> &r2
  // {
  //   Range<T> result = (ranges.size() > 0 ? ranges[0] : Range<T>());
  //   for(int i = 1; i < ranges.size(); i++) { result.fit(r.lower); result.fit(r.upper); }
  //   return result;
  // }
};

// union (NOTE: result is continuous, no holes)
template<typename T>
inline Range<T> combine(const std::vector<Range<T>> &ranges)
{
  Range<T> result = (ranges.size() > 0 ? ranges[0] : Range<T>());
  for(int i = 1; i < ranges.size(); i++) { result.fit(ranges[i].lower); result.fit(ranges[i].upper); }
  return result;
}

template<typename T>
inline bool checkIntersection(const Range<T> &r1, const Range<T> &r2)
{ return (r2.lower >= r1.upper && r1.lower <= r2.upper); }

template<typename T>
inline Range<T> intersection(const std::vector<Range<T>> &ranges)
{
  Range<T> result = (ranges.size() > 0 ? ranges[0] : Range<T>());
  for(int i = 1; i < ranges.size(); i++) { result.fit(ranges[i].lower); result.fit(ranges[i].upper); }
  return result;
}




#endif // RANGE_HPP
