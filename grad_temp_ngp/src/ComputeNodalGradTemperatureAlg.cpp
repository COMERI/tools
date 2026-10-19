#include "ComputeNodalGradTemperatureAlg.h"

#include <Kokkos_Core.hpp>

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>

#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/NgpTypes.hpp>

namespace
{

using ExecSpace = stk::ngp::ExecSpace;

using TeamPolicy =
  stk::ngp::TeamPolicy<ExecSpace>;

using TeamHandle =
  typename TeamPolicy::member_type;


template <typename Element>
KOKKOS_FUNCTION
void compute_element_gradient(
  const TeamHandle& team,
  const stk::mesh::NgpMesh& ngpMesh,
  const auto& coordinatesData,
  const auto& temperatureData,
  auto& gradTemperatureData,
  const unsigned bucketId)
{
  auto gradValues =
    gradTemperatureData.bucket_values(bucketId);

  if (!gradValues.is_field_defined()) {
    return;
  }

  const stk::mesh::EntityIdx numElements =
    gradValues.num_entities();

  Kokkos::parallel_for(
    Kokkos::TeamThreadRange(team, 0_entity, numElements),
    [&](const stk::mesh::EntityIdx elem)
    {
      //
      // Allocate scratch for this element.
      //
      const std::size_t coordinatesBytes =
        Element::coordinates_scratch_size();

      const std::size_t dndxBytes =
        Element::dndx_scratch_size();

      const std::size_t scvVolumeBytes =
        Element::scv_volume_scratch_size();

      double* ptr =
        static_cast<double*>(
          team.thread_scratch(0).get_shmem(
            coordinatesBytes +
            dndxBytes +
            scvVolumeBytes));


      //
      // Map scratch memory to the element data structures.
      //
      typename Element::CoordinatesView coordinates(
        ptr,
        Element::nodesPerElement,
        Element::spatialDim);

      ptr += coordinatesBytes / sizeof(double);


      typename Element::DndxView dndx(
        ptr,
        Element::numScvIp,
        Element::nodesPerElement,
        Element::spatialDim);

      ptr += dndxBytes / sizeof(double);


      typename Element::ScvVolumeView scvVolume(
        ptr,
        Element::numScvIp);


      //
      // Get the element's connected nodes.
      //
      stk::mesh::NgpMesh::ConnectedNodes nodes =
        ngpMesh
          .get_bucket(
            stk::topology::ELEM_RANK,
            bucketId)
          .get_nodes(
            static_cast<int>(elem));


      //
      // Gather element data.
      //
      double temperature[
        Element::nodesPerElement];

      for (int node = 0;
           node < Element::nodesPerElement;
           ++node)
      {
        auto coord =
          coordinatesData.entity_values(nodes[node]);

        coordinates(node, 0) = coord(0_comp);
        coordinates(node, 1) = coord(1_comp);
        coordinates(node, 2) = coord(2_comp);

        auto temp =
          temperatureData.entity_values(nodes[node]);

        temperature[node] = temp(0_comp);
      }


      //
      // Compute element geometry.
      //
      Element::compute_dndx(
        coordinates,
        dndx);

      Element::compute_scv_volume(
        coordinates,
        scvVolume);


      //
      // Accumulate the element contribution at each
      // SCV integration point.
      //
      double localGradient[
        Element::nodesPerElement]
        [Element::spatialDim];

      for (int node = 0;
           node < Element::nodesPerElement;
           ++node)
      {
        for (int dim = 0;
             dim < Element::spatialDim;
             ++dim)
        {
          localGradient[node][dim] = 0.0;
        }
      }


      for (int ip = 0;
           ip < Element::numScvIp;
           ++ip)
      {
        const int node =
          Element::ip_node(ip);

        for (int dim = 0;
             dim < Element::spatialDim;
             ++dim)
        {
          double gradient = 0.0;

          for (int j = 0;
               j < Element::nodesPerElement;
               ++j)
          {
            gradient +=
              temperature[j] *
              dndx(ip, j, dim);
          }

          localGradient[node][dim] +=
            gradient * scvVolume(ip);
        }
      }


      //
      // Accumulate the element contribution into the
      // nodal gradient field.
      //
      for (int node = 0;
           node < Element::nodesPerElement;
           ++node)
      {
        auto grad =
          gradTemperatureData.entity_values(nodes[node]);

        for (int dim = 0;
             dim < Element::spatialDim;
             ++dim)
        {
          Kokkos::atomic_add(
            &grad(stk::mesh::ComponentIdx(dim)),
            localGradient[node][dim]);
        }
      }
    });
}

} // namespace


ComputeNodalGradTemperatureAlg::
ComputeNodalGradTemperatureAlg(
  stk::mesh::BulkData& bulk,
  stk::mesh::Part* part,
  stk::mesh::Field<double>& coordinates,
  stk::mesh::Field<double>& temperature,
  stk::mesh::Field<double>& gradTemperature)
  : bulk_(bulk),
    coordinates_(coordinates),
    temperature_(temperature),
    gradTemperature_(gradTemperature)
{
  partVec_.push_back(part);
}


void
ComputeNodalGradTemperatureAlg::execute()
{
  stk::mesh::MetaData& meta_data =
    bulk_.mesh_meta_data();


  //
  // Select locally owned elements belonging to all parts
  // associated with this algorithm.
  //
  stk::mesh::Selector s_locally_owned_union =
    meta_data.locally_owned_part()
    & stk::mesh::selectUnion(partVec_);


  //
  // Select all nodes on which gradTemperature is defined.
  //
  // Do not restrict this selector to locally owned nodes.
  //
  stk::mesh::Selector s_all_nodes =
    stk::mesh::selectUnion(partVec_)
    & stk::mesh::selectField(gradTemperature_);


  //
  // Initialize the output field.
  //

  stk::mesh::field_fill(
    0.0,
    gradTemperature_,
    s_all_nodes);

  //
  // gradTemperature_ is owned by this algorithm, so make
  // the initialized field available on the device.
  //
  gradTemperature_.sync_to_device();


  //
  // Tet4 elements (add more topos if needed)
  //
  stk::mesh::Selector s_tet4 =
    s_locally_owned_union
    & meta_data.get_topology_root_part(
        stk::topology::TETRAHEDRON_4);

  execute_element<cvfem::Tet4SCVElement>(
    s_tet4);


  //
  // Make the device-computed output current on the host.
  //
  gradTemperature_.sync_to_host();


  //
  // Sum contributions to shared nodes across MPI ranks.
  //
  stk::mesh::parallel_sum(
    bulk_,
    {&gradTemperature_});
}


template <typename Element>
void
ComputeNodalGradTemperatureAlg::execute_element(
  const stk::mesh::Selector& selector)
{
  //
  // Update the NGP mesh.
  //
  stk::mesh::NgpMesh& ngpMesh =
    stk::mesh::get_updated_ngp_mesh(bulk_);

  //
  // Input fields are read-only dependencies.
  //
  auto coordinatesData =
    coordinates_.data<
      stk::mesh::ReadOnly,
      stk::ngp::DeviceSpace>();

  auto temperatureData =
    temperature_.data<
      stk::mesh::ReadOnly,
      stk::ngp::DeviceSpace>();


  //
  // Output field is read-write.
  //
  auto gradTemperatureData =
    gradTemperature_.data<
      stk::mesh::ReadWrite,
      stk::ngp::DeviceSpace>();


  //
  // Get the element bucket IDs selected by this topology.
  //
  stk::NgpVector<unsigned> bucketIds =
    ngpMesh.get_bucket_ids(
      stk::topology::ELEM_RANK,
      selector);

  const unsigned numBuckets =
    bucketIds.size();


  //
  // Allocate only the scratch required by this algorithm.
  //
  const std::size_t scratchBytes =
    Element::coordinates_scratch_size()
    + Element::dndx_scratch_size()
    + Element::scv_volume_scratch_size();


  //
  // One team corresponds to one bucket.
  //
  Kokkos::parallel_for(
    TeamPolicy(
      numBuckets,
      Kokkos::AUTO)
      .set_scratch_size(
        0,
        Kokkos::PerThread(scratchBytes)),

    KOKKOS_LAMBDA(
      const TeamHandle& team)
    {
      const unsigned bucketId =
        bucketIds.get<ExecSpace>(
          team.league_rank());

      compute_element_gradient<Element>(
        team,
        ngpMesh,
        coordinatesData,
        temperatureData,
        gradTemperatureData,
        bucketId);
    });
}


//
// Explicit template instantiations.
//
template void
ComputeNodalGradTemperatureAlg::execute_element<
  cvfem::Tet4SCVElement>(
    const stk::mesh::Selector& selector);
