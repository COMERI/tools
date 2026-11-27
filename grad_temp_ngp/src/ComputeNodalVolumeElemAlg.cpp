#include "ComputeNodalVolumeElemAlg.h"

#include <Kokkos_Core.hpp>

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/FieldParallel.hpp>

#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/NgpParallelComm.hpp>
#include <stk_mesh/base/NgpTypes.hpp>
#include <stk_mesh/base/NgpFieldBLAS.hpp>

namespace
{

using ExecSpace = stk::ngp::ExecSpace;
using DeviceSpace = stk::ngp::DeviceSpace;

using TeamPolicy = stk::ngp::TeamPolicy<ExecSpace>;
using TeamHandle = typename TeamPolicy::member_type;

template <typename Element>
KOKKOS_FUNCTION
void compute_element_scvol(
  const TeamHandle& team,
  const stk::mesh::NgpMesh& ngpMesh,
  const auto& coordinatesData,
  auto& dualNodalVolumeData,
  const unsigned bucketId)
{
  // Number of elements in this element-rank bucket.
  const stk::mesh::EntityIdx numElements =
    ngpMesh.get_bucket(stk::topology::ELEM_RANK, bucketId).size();

  // Allocate per-thread scratch once; it is reused for every element this
  // thread processes.
  const std::size_t coordinatesBytes =
    Element::coordinates_scratch_size();

  const std::size_t scvVolumeBytes =
    Element::scv_volume_scratch_size();

  double* ptr =
    static_cast<double*>(
      team.thread_scratch(0).get_shmem(
        coordinatesBytes +
        scvVolumeBytes));

  // Map scratch memory to the element data structures.
  typename Element::CoordinatesView coordinates(
    ptr,
    Element::nodesPerElement,
    Element::spatialDim);

  ptr += coordinatesBytes / sizeof(double);

  typename Element::ScvVolumeView scvVolume(
    ptr,
    Element::numScvIp);

  Kokkos::parallel_for(
    Kokkos::TeamThreadRange(team, 0_entity, numElements),
    [&](const stk::mesh::EntityIdx elem)
    {
      // Get the element's connected nodes.
      stk::mesh::NgpMesh::ConnectedNodes nodes =
        ngpMesh
          .get_bucket(
            stk::topology::ELEM_RANK,
            bucketId)
          .get_nodes(
            static_cast<int>(elem));

      for (int node = 0; node < Element::nodesPerElement; ++node) {
        
        auto coord =
          coordinatesData.entity_values(nodes[node]);

        for (unsigned dim = 0; dim < Element::spatialDim; ++dim) {
          coordinates(node, dim) =
            coord(stk::mesh::ComponentIdx(dim));
        }
        
      }
      
      // Compute element geometry.
      Element::determinant(coordinates,
                           scvVolume);

      // Accumulate the element contribution at each
      // SCV integration point.
      double localDualNodalVolume[Element::nodesPerElement] = {};

      for (int ip = 0; ip < Element::numScvIp; ++ip) {
        const int node = Element::ip_node(ip);
        localDualNodalVolume[node] += scvVolume(ip);
      }
      
      // Accumulate the element contribution into the
      // nodal volume field.
      for (int node = 0; node < Element::nodesPerElement; ++node) {
        auto dnVolume =
          dualNodalVolumeData.entity_values(nodes[node]);
        
        Kokkos::atomic_add(&dnVolume(stk::mesh::ComponentIdx(0)),
                           localDualNodalVolume[node]);
      }
    });
}

} // namespace


ComputeNodalVolumeElemAlg::
ComputeNodalVolumeElemAlg(
  stk::mesh::BulkData& bulk,
  stk::mesh::Part* part,
  stk::mesh::Field<double>& coordinates,
  stk::mesh::Field<double>& dualNodalVolume)
  : bulk_(bulk),
    coordinates_(coordinates),
    dualNodalVolume_(dualNodalVolume)
{
  partVec_.push_back(part);
}


void
ComputeNodalVolumeElemAlg::execute()
{
  stk::mesh::NgpMesh& ngpMesh =
    stk::mesh::get_updated_ngp_mesh(bulk_);

  stk::mesh::MetaData& metaData = bulk_.mesh_meta_data();
  
  const unsigned nDim = metaData.spatial_dimension();
  
  // Select locally owned elements belonging to all parts
  // associated with this algorithm.
  stk::mesh::Selector s_locally_owned_union =
    metaData.locally_owned_part()
    & stk::mesh::selectUnion(partVec_);

  // Select all nodes on which dualNodal is defined.
  // Do not restrict this selector to locally owned nodes.
  stk::mesh::Selector s_all_nodes =
    stk::mesh::selectUnion(partVec_)
    & stk::mesh::selectField(dualNodalVolume_);

  // Initialize the output field on execution space
  stk::mesh::field_fill<DeviceSpace>(
    0.0,
    dualNodalVolume_,
    s_all_nodes);
  
  // Tet4 elements (add more topos if needed)
  stk::mesh::Selector s_tet4 =
    s_locally_owned_union
    & metaData.get_topology_root_part(
        stk::topology::TETRAHEDRON_4);

  execute_element<cvfem::Tet4SCVElement>(s_tet4);
  
  // Sum contributions across shared nodes using device execution space.
  stk::mesh::parallel_sum<DeviceSpace>(bulk_,
                                       {&dualNodalVolume_});
  
  // Computed field current in device memory; synchronize to host only for I/O.
}


template <typename Element>
void
ComputeNodalVolumeElemAlg::execute_element(
  const stk::mesh::Selector& selector)
{
  // Update the NGP mesh.
  stk::mesh::NgpMesh& ngpMesh =
    stk::mesh::get_updated_ngp_mesh(bulk_);

  // Input fields are read-only dependencies.
  auto coordinatesData =
    coordinates_.data<
      stk::mesh::ReadOnly,
      DeviceSpace>();

  // Output field is read-write.
  auto dualNodalVolumeData =
    dualNodalVolume_.data<
      stk::mesh::ReadWrite,
      DeviceSpace>();

  // Get the element bucket IDs selected by this topology.
  stk::NgpVector<unsigned> bucketIds =
    ngpMesh.get_bucket_ids(
      stk::topology::ELEM_RANK,
      selector);

  const unsigned numBuckets =
    bucketIds.size();

  // Allocate only the scratch required by this algorithm.
  const std::size_t scratchBytes =
    Element::coordinates_scratch_size()
    + Element::scv_volume_scratch_size();

  // One team corresponds to one bucket.
  Kokkos::parallel_for(
    TeamPolicy(
      numBuckets,
      Kokkos::AUTO)
      .set_scratch_size(
        0,
        Kokkos::PerThread(scratchBytes)),

    KOKKOS_LAMBDA(
      const TeamHandle& team) {
      const unsigned bucketId =
        bucketIds.get<ExecSpace>(
          team.league_rank());

      compute_element_scvol<Element>(
        team,
        ngpMesh,
        coordinatesData,
        dualNodalVolumeData,
        bucketId);
    });
}

// Explicit template instantiations.
template void
ComputeNodalVolumeElemAlg::execute_element<
  cvfem::Tet4SCVElement>(
    const stk::mesh::Selector& selector);
