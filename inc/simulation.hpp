#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include <vector>
#include "field.cuh"

class Simulation
{
private:
  std::vector<FieldBase*> mFields;
  
public:
  Simulation() = default;
  
};


#endif // SIMULATION_HPP
