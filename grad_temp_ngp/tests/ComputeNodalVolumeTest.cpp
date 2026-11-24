
#include "ComputeNodalVolumeAlg.h"

#include <Kokkos_Core.hpp>

#include <stk_io/StkMeshIoBroker.hpp>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

#include <stk_util/parallel/Parallel.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>

int
main(int argc, char** argv)
{
  // Initialize MPI/STK and Kokkos.
  stk::ParallelMachine comm =
    stk::parallel_machine_init(&argc, &argv);

  Kokkos::initialize(argc, argv);

  int result = 0;

  {
    const int myRank =
      stk::parallel_machine_rank(comm);

    // Analytical reference volume (pi is from my mesh, not precise pi).
    const double pi = 3.1415900000000008;
    const double analyticalVolume = (2.0*pi)*(2.0*pi)*(2.0*pi);

    // Read the mesh.
    std::string meshFile =
      std::string(CVFEM_SOURCE_DIR) + "/mesh/3d_tet4_taylor_green_0p4.exo";
    
    stk::io::StkMeshIoBroker io(comm);

    io.add_mesh_database(
      meshFile,
      stk::io::READ_MESH);

    io.create_input_mesh();

    stk::mesh::MetaData& metaData = io.meta_data();

    // Get the coordinate field supplied by the mesh.
    stk::mesh::Field<double>* coordinates =
      metaData.get_field<double>(
        stk::topology::NODE_RANK,
        "coordinates");

    if (coordinates == nullptr) {
      if (myRank == 0) {
        std::cerr
          << "ERROR: Could not find coordinates field.\n";
      }
      result = 1;
    }
    else {

      // Declare the nodal dual-volume field.
      stk::mesh::Field<double>& dualNodalVolume =
        metaData.declare_field<double>(
          stk::topology::NODE_RANK,
          "dual_nodal_volume");

      stk::mesh::put_field_on_entire_mesh(
        dualNodalVolume);

      // Populate the mesh.
      io.populate_bulk_data();

      stk::mesh::BulkData& bulk = io.bulk_data();

      // Select Tet4 elements.
      stk::mesh::Part& tet4Part =
        metaData.get_topology_root_part(
          stk::topology::TETRAHEDRON_4);

      // Calculate dual nodal volumes.
      ComputeNodalVolumeAlg dnvAlgorithm(
        bulk,
        &tet4Part,
        *coordinates,
        dualNodalVolume);

      dnvAlgorithm.execute();

      // Synchronize before host-side inspection.
      dualNodalVolume.sync_to_host();

      // Sum the dual nodal volumes on the host.
      double spdTotalVolume = 0.0;
      
      stk::mesh::Selector nodeSelector =
        metaData.universal_part()
        & stk::mesh::selectField(dualNodalVolume);

      const stk::mesh::BucketVector& nodeBuckets =
        bulk.get_buckets(
          stk::topology::NODE_RANK,
          nodeSelector);

      auto dualNodalVolumeData =
        dualNodalVolume.data<
          stk::mesh::ReadOnly,
          stk::ngp::HostSpace>();

      for (stk::mesh::Bucket* bucket : nodeBuckets) {
        for (size_t i = 0; i < bucket->size(); ++i) {
          stk::mesh::Entity node = (*bucket)[i];

          auto volume =
            dualNodalVolumeData.entity_values(node);

          spdTotalVolume +=
            volume(stk::mesh::ComponentIdx(0));
        }
      }

      // Sum contributions across MPI ranks.
      double globalTotalVolume = 0.0;

      stk::all_reduce_sum(
        comm,
        &spdTotalVolume,
        &globalTotalVolume,
        1);

      // Compare against the analytical reference.
      const double tolerance = 1.0e-10;

      const double error =
        std::abs(globalTotalVolume - analyticalVolume);

      const bool passed = error <= tolerance;

      if (myRank == 0) {
        std::cout << std::setprecision(12)
                  << "\nDual nodal volume unit test\n"
                  << "---------------------------\n"
                  << "Expected volume: " << analyticalVolume << "\n"
                  << "Computed volume: " << globalTotalVolume << "\n"
                  << "Absolute error:  " << error << "\n"
                  << "Tolerance:       " << tolerance << "\n"
                  << "Result:          "
                  << (passed ? "PASS" : "FAIL") << "\n";
      }

      if (!passed) {
        result = 1;
      }
    }
  }

  // Finalize Kokkos and MPI/STK.
  Kokkos::finalize();

  stk::parallel_machine_finalize();

  return result;
}
