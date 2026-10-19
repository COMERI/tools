#ifndef COMPUTE_NODAL_GRAD_TEMPERATURE_ALG_H
#define COMPUTE_NODAL_GRAD_TEMPERATURE_ALG_H

#include "CVFEMElementTraits.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <vector>

class ComputeNodalGradTemperatureAlg
{
public:

  ComputeNodalGradTemperatureAlg(
    stk::mesh::BulkData& bulk,
    stk::mesh::Part* part,
    stk::mesh::Field<double>& coordinates,
    stk::mesh::Field<double>& temperature,
    stk::mesh::Field<double>& gradTemperature);

  ~ComputeNodalGradTemperatureAlg() = default;

  void execute();

  std::vector<stk::mesh::Part*> partVec_;

private:

  template <typename Element>
  void execute_element(
    const stk::mesh::Selector& selector);

  stk::mesh::BulkData& bulk_;

  stk::mesh::Field<double>& coordinates_;
  stk::mesh::Field<double>& temperature_;
  stk::mesh::Field<double>& gradTemperature_;
};

#endif
