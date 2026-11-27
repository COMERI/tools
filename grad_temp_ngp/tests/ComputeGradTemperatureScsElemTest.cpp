#include "ComputeNodalVolumeElemAlg.h"
#include "ComputeNodalGradTemperatureScsElemAlg.h"

#include <Kokkos_Core.hpp>

#include <stk_io/StkMeshIoBroker.hpp>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/NgpMesh.hpp>

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

      // Declare the nodal dual-volume field; place it on the mesh
      stk::mesh::Field<double>& dualNodalVolume =
        metaData.declare_field<double>(
          stk::topology::NODE_RANK,
          "dual_nodal_volume");

      stk::mesh::put_field_on_entire_mesh(
        dualNodalVolume);

      // Declare the nodal temperature field; place it on the mesh
      stk::mesh::Field<double>& temperature =
        metaData.declare_field<double>(
          stk::topology::NODE_RANK,
          "temperature");

      stk::mesh::put_field_on_entire_mesh(
        temperature);

      // Declare the nodal temperature-gradient field.    
      stk::mesh::Field<double>& gradTemperature =
        metaData.declare_field<double>(
          stk::topology::NODE_RANK,
          "scs_grad_temperature");

      stk::mesh::put_field_on_entire_mesh(
        gradTemperature,
        metaData.spatial_dimension());
      
      // Populate the mesh.
      io.populate_bulk_data();
      
      stk::mesh::BulkData& bulk = io.bulk_data();

      // extract ngpMesh
      stk::mesh::NgpMesh& ngpMesh = stk::mesh::get_updated_ngp_mesh(bulk);
      
      // extract nDim
      const unsigned nDim = metaData.spatial_dimension();
      
      // extract fields
      auto& ngpCoordinates =
        stk::mesh::get_updated_ngp_field<double>(*coordinates);
      
      auto& ngpTemperature =
        stk::mesh::get_updated_ngp_field<double>(temperature);

      // Synchronize mesh coordinates to the execution space.
      ngpCoordinates.sync_to_device();
 
      // Define the linear temperature field:
      //   T(x,y,z) = T0 + ax*x + ay*y + az*z
      //
      // Exact gradient:
      //   grad(T) = (ax, ay, az)
      //
      const double T0 = 100.0;
      const double ax[3] = {2.0, 3.0, 4.0};
      
      // Set temperature at the nodes.
      stk::mesh::Selector nodeSelector =
        metaData.universal_part()
        & stk::mesh::selectField(temperature);
      
      // Initialize temperature using the NGP device-side interface.
      stk::mesh::for_each_entity_run(ngpMesh,
                                     stk::topology::NODE_RANK,
                                     nodeSelector,
                                     KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& node)
      {
        auto xyz = ngpCoordinates(node);
        auto temp = ngpTemperature(node);
        
        double sum = T0;
        for (int dim = 0; dim < nDim; ++dim) {
          sum += ax[dim] * xyz[dim];
        }
        
        temp[0] = sum;
      });
      
      // Select Tet4 elements.
      stk::mesh::Part& tet4Part =
        metaData.get_topology_root_part(
          stk::topology::TETRAHEDRON_4);

      // Calculate dual nodal volumes.
      ComputeNodalVolumeElemAlg dnvElemAlgorithm(
        bulk,
        &tet4Part,
        *coordinates,
        dualNodalVolume);
      dnvElemAlgorithm.execute();

      // Synchronize before host-side inspection.
      dualNodalVolume.sync_to_host();

      // Construct the nodal-gradient algorithm.
      ComputeNodalGradTemperatureScsElemAlg gradElemAlgorithm(
        bulk,
        &tet4Part,
        *coordinates,
        dualNodalVolume,
        temperature,
        gradTemperature);
      gradElemAlgorithm.execute();

      // TBD: Add boundary contribution....
      
      // Synchronize before host-side inspection.
      gradTemperature.sync_to_host();

      //=============================================
      // Check the result using a traditional host-side
      // bucket iteration. Why? It is a simple diagnostic
      // operation and does not benefit materially from
      // device-side execution.
      //=============================================
               
      double l_error = 0.0;
      
      const stk::mesh::BucketVector& checkBuckets =
        bulk.get_buckets(stk::topology::NODE_RANK,
                         nodeSelector);

        auto gradTemperatureData =
          gradTemperature.data<stk::mesh::ReadOnly,
                               stk::ngp::HostSpace>();
        
        for (stk::mesh::Bucket* bucket : checkBuckets) {
          
          for (size_t i = 0; i < bucket->size(); ++i) {
            
            stk::mesh::Entity node = (*bucket)[i];

            auto grad = gradTemperatureData.entity_values(node);
            
            double error = 0.0;
            for (int dim = 0; dim < nDim; ++dim) {
              error = std::max(error,
                               std::abs(grad(stk::mesh::ComponentIdx(dim)) - ax[dim]));
            }
            
            l_error = std::max(l_error, error);
          }
        }
        
        // Reduce the error across MPI ranks.
        double g_error = 0.0;
	stk::all_reduce_max(comm,
			    &l_error,
			    &g_error,
			    1);

      // Compare against the analytical reference.
      const double tolerance = 1.0e-10;

      const bool passed = g_error <= tolerance;

      if (myRank == 0) {
        std::cout << std::setprecision(12)
                  << "\nDual nodal volume unit test\n"
                  << "---------------------------\n"
                  << "Absolute error:  " << g_error << "\n"
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
