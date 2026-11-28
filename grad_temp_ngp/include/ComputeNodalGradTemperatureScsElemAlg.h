#ifndef COMPUTE_NODAL_GRAD_TEMPERATURE_SCS_ELEM_ALG_H
#define COMPUTE_NODAL_GRAD_TEMPERATURE_SCS_ELEM_ALG_H

#include "CVFEMElementTraits.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <vector>

class ComputeNodalGradTemperatureScsElemAlg
{
public:

  ComputeNodalGradTemperatureScsElemAlg(
    stk::mesh::BulkData& bulk,
    stk::mesh::Part* part,
    stk::mesh::Field<double>& coordinates,
    stk::mesh::Field<double>& dualNodalVolume,
    stk::mesh::Field<double>& temperature,
    stk::mesh::Field<double>& gradTemperature);

  ~ComputeNodalGradTemperatureScsElemAlg() = default;

  void execute();

  std::vector<stk::mesh::Part*> partVec_;

private:

  template <typename Element>
  void execute_element(
    const stk::mesh::Selector& selector);

  stk::mesh::BulkData& bulk_;

  // Field data is accessed and modified in the execution space used by this
  // algorithm (host execution space for CPU builds, device execution space for
  // GPU builds). The corresponding memory space is HostSpace or DeviceSpace,
  // respectively. Synchronization between host and device is intentionally
  // managed by the caller. Callers must synchronize before accessing the fields
  // from a different execution space, including host-side output or subsequent
  // host-only algorithms.

  stk::mesh::Field<double>& coordinates_;
  stk::mesh::Field<double>& temperature_;
  stk::mesh::Field<double>& gradTemperature_;
  stk::mesh::Field<double>& dualNodalVolume_;
};

#endif
