#include "ComputeNodalGradTemperatureAlg.h"
#include "ComputeNodalVolumeAlg.h"

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
#include <algorithm>
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

    // Mesh filename.
    if (argc != 2) {

      if (myRank == 0) {
        std::cerr
          << "Usage: " << argv[0]
          << " mesh.exo\n";
      }

      result = 1;
    }
    else {

      const std::string meshFile = argv[1];

      // Create the STK mesh.
      stk::io::StkMeshIoBroker io(comm);

      io.add_mesh_database(
        meshFile,
        stk::io::READ_MESH);
      
      // Read mesh metadata, but do not populate bulk data yet.
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

        // Declare the nodal temperature field.
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
            "scv_grad_temperature");

        // and put it on the mesh
        stk::mesh::put_field_on_entire_mesh(
          gradTemperature,
          metaData.spatial_dimension());

        // Declare the nodal dual-nodal-gradient field.
        stk::mesh::Field<double>& dualNodalVolume =
          metaData.declare_field<double>(
            stk::topology::NODE_RANK,
            "dual_nodal_volume");

        // ... and put it on the mesh
        stk::mesh::put_field_on_entire_mesh(
          dualNodalVolume);
        
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

        // Mark device-side temperature updates for subsequent synchronization.
        ngpTemperature.modify_on_device();
        
        //=============================================
        // Algorithms operates on Tet4 elements.
        //=============================================

	stk::mesh::Part& tet4Part =
	  metaData.get_topology_root_part(
           stk::topology::TETRAHEDRON_4);

	// Construct the nodal-gradient algorithm.
	ComputeNodalVolumeAlg dnvAlgorithm(
          bulk,
	  &tet4Part,
	  *coordinates,
	  dualNodalVolume);
	
	// Execute the CVFEM calculation.
        // Alg owns zeroing of the fields.
        dnvAlgorithm.execute();
         
	// Construct the nodal-gradient algorithm.
	ComputeNodalGradTemperatureAlg gradAlgorithm(
          bulk,
	  &tet4Part,
	  *coordinates,
	  temperature,
	  gradTemperature,
	  dualNodalVolume);
	
	// Execute the CVFEM calculation.
        // Alg owns zeroing of the fields
	gradAlgorithm.execute();

        //=============================================
        // Sync all modified fields to prepare for output
        //=============================================
        temperature.sync_to_host();
        dualNodalVolume.sync_to_host();
        gradTemperature.sync_to_host();
        
	// Write the results to an Exodus database.
	const std::string outputFile =
	  "cvfem_dnv_gradt_test.exo";
	
	const size_t outputFileIndex =
	  io.create_output_mesh(outputFile,
				stk::io::WRITE_RESULTS);	
	
	// Output fields.
	io.add_field(outputFileIndex,
                     dualNodalVolume);

        io.add_field(outputFileIndex,
                     temperature);
	
	io.add_field(outputFileIndex,
		     gradTemperature);	
	
	// Write the mesh and the initial field definitions.
	io.write_output_mesh(outputFileIndex);
	
	// Write one transient result step.
	io.begin_output_step(outputFileIndex,
			     0.0);
	
	io.write_defined_output_fields(outputFileIndex);
	
	io.end_output_step(outputFileIndex);

        //=============================================
        // Check the result using a traditional host-side
        // bucket iteration. Why? It is a simple diagnostic
        // operation and does not benefit materially from
        // device-side execution.
        //=============================================
               
        double maxError = 0.0;
        unsigned long numNodes = 0;
        
        stk::mesh::Selector checkSelector =
          metaData.universal_part()
          & stk::mesh::selectField(gradTemperature);
        
        const stk::mesh::BucketVector& checkBuckets =
          bulk.get_buckets(stk::topology::NODE_RANK,
                           checkSelector);

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
            
            maxError = std::max(maxError, error);
            ++numNodes;
          }
        }
        
        // Reduce the error across MPI ranks.
        double globalMaxError = 0.0;
	stk::all_reduce_max(comm,
			    &maxError,
			    &globalMaxError,
			    1);
	
	unsigned long globalNumNodes = 0;
	stk::all_reduce_sum(comm,
			    &numNodes,
			    &globalNumNodes,
			    1);

        // Report the result.
	if (myRank == 0) {
	  
	  std::cout
	    << "\n"
	    << "CVFEM nodal temperature-gradient test\n"
	    << "--------------------------------------\n"
	    << "Temperature:\n"
	    << "  T = "
	    << T0
	    << " + "
	    << ax[0] << " x + "
	    << ax[1] << " y + "
	    << ax[2] << " z\n\n";
	  
	  std::cout
	    << "Expected gradient:\n"
	    << "  ("
	    << ax[0] << ", "
	    << ax[1] << ", "
	    << ax[2] << ")\n\n";
	  
	  std::cout
	    << "Nodes checked: "
	    << globalNumNodes
	    << "\n";
	  
	  std::cout
	    << std::setprecision(12)
	    << "Maximum error: "
	    << globalMaxError
	    << "\n\n";
	  
	  std::cout
	    << "Wrote results to: "
	    << outputFile
	    << "\n";
	}
      }
    }
  }

  // Finalize Kokkos and MPI/STK.
  Kokkos::finalize();

  stk::parallel_machine_finalize();

  return result;
}
