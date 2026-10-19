#include "ComputeNodalGradTemperatureAlg.h"

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
#include <algorithm>
#include <string>


int
main(int argc, char** argv)
{
  //
  // Initialize MPI/STK and Kokkos.
  //
  stk::ParallelMachine comm =
    stk::parallel_machine_init(&argc, &argv);

  Kokkos::initialize(argc, argv);

  int result = 0;

  {
    const int myRank =
      stk::parallel_machine_rank(comm);


    //
    // Mesh filename.
    //
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


      //
      // Create the STK mesh.
      //
      stk::io::StkMeshIoBroker io(comm);

      io.add_mesh_database(
        meshFile,
        stk::io::READ_MESH);

      //
      // Read mesh metadata, but do not populate bulk data yet.
      //
      io.create_input_mesh();

      stk::mesh::MetaData& metaData =
        io.meta_data();


      //
      // Get the coordinate field supplied by the mesh.
      //
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

        //
        // Declare the nodal temperature field.
        //
        stk::mesh::Field<double>& temperature =
          metaData.declare_field<double>(
            stk::topology::NODE_RANK,
            "temperature");

        stk::mesh::put_field_on_entire_mesh(
          temperature);


        //
        // Declare the nodal temperature-gradient field.
        //
        stk::mesh::Field<double>& gradTemperature =
          metaData.declare_field<double>(
            stk::topology::NODE_RANK,
            "grad_temperature");

        stk::mesh::put_field_on_entire_mesh(
          gradTemperature);


        //
        // Populate the mesh.
        //
        io.populate_bulk_data();
	stk::mesh::BulkData& bulk = io.bulk_data();
	
        //
        // Define the linear temperature field:
        //
        //   T(x,y,z) = T0 + ax*x + ay*y + az*z
        //
        // Exact gradient:
        //
        //   grad(T) = (ax, ay, az)
        //
        const double T0 = 100.0;
        const double ax =   2.0;
        const double ay =   3.0;
        const double az =   4.0;


        //
        // Set temperature at the nodes.
        //
        stk::mesh::Selector nodeSelector =
          metaData.universal_part()
          & stk::mesh::selectField(temperature);

        const stk::mesh::BucketVector& nodeBuckets =
          bulk.get_buckets(
            stk::topology::NODE_RANK,
            nodeSelector);


        for (stk::mesh::Bucket* bucket : nodeBuckets) {

          for (size_t i = 0; i < bucket->size(); ++i) {

            stk::mesh::Entity node =
              (*bucket)[i];

            const double* xyz =
              stk::mesh::field_data(
                *coordinates,
                node);

            double* temp =
              stk::mesh::field_data(
                temperature,
                node);

            temp[0] =
              T0
              + ax * xyz[0]
              + ay * xyz[1]
              + az * xyz[2];
          }
        }


        //
        // Temperature was modified on the host.
        // Make it available to the device.
        //
        temperature.sync_to_device();


        //
        // The algorithm operates on Tet4 elements.
        //
	stk::mesh::Part& tet4Part =
	  metaData.get_topology_root_part(
           stk::topology::TETRAHEDRON_4);

	//
	// Construct the nodal-gradient algorithm.
	//
	ComputeNodalGradTemperatureAlg algorithm(
          bulk,
	  &tet4Part,
	  *coordinates,
	  temperature,
	  gradTemperature);
	
	//
	// Execute the CVFEM calculation.
	//
	algorithm.execute();
	
	
	//
	// Check the result.
	//
	double maxError = 0.0;
	unsigned long numNodes = 0;
	
	stk::mesh::Selector checkSelector =
	  metaData.universal_part()
	  & stk::mesh::selectField(gradTemperature);
	
	const stk::mesh::BucketVector& checkBuckets =
	  bulk.get_buckets(
			   stk::topology::NODE_RANK,
			   checkSelector);
	
	
	for (stk::mesh::Bucket* bucket : checkBuckets) {
	  
	  for (size_t i = 0; i < bucket->size(); ++i) {
	    
	    stk::mesh::Entity node =
	      (*bucket)[i];
	    
	    const double* grad =
	      stk::mesh::field_data(
				    gradTemperature,
				    node);
	    
	    const double error =
	      std::max(
		       {
			 std::abs(grad[0] - ax),
			 std::abs(grad[1] - ay),
			 std::abs(grad[2] - az)
		       });
	    
	    maxError =
	      std::max(maxError, error);
	    
	    ++numNodes;
	  }
	}
	
	
	//
	// Reduce the error across MPI ranks.
	//
	double globalMaxError = 0.0;
	
	stk::all_reduce_max(
			    comm,
			    &maxError,
			    &globalMaxError,
			    1);
	
	
	unsigned long globalNumNodes = 0;
	
	stk::all_reduce_sum(
			    comm,
			    &numNodes,
			    &globalNumNodes,
			    1);
	
	
	//
	// Write the results to an Exodus database.
	//
	const std::string outputFile =
	  "gradient_test.exo";
	
	const size_t outputFileIndex =
	  io.create_output_mesh(
				outputFile,
				stk::io::WRITE_RESULTS);
	
	
	//
	// Output both fields.
	//
	io.add_field(
		     outputFileIndex,
		     temperature);
	
	io.add_field(
		     outputFileIndex,
		     gradTemperature);
	
	
	//
	// Write the mesh and the initial field definitions.
	//
	io.write_output_mesh(
			     outputFileIndex);
	
	
	//
	// Write one transient result step.
	//
	io.begin_output_step(
			     outputFileIndex,
			     0.0);
	
	io.write_defined_output_fields(
				       outputFileIndex);
	
	io.end_output_step(
			   outputFileIndex);
	
	
	//
	// Report the result.
	//
	if (myRank == 0) {
	  
	  std::cout
	    << "\n"
	    << "CVFEM nodal temperature-gradient test\n"
	    << "--------------------------------------\n"
	    << "Temperature:\n"
	    << "  T = "
	    << T0
	    << " + "
	    << ax << " x + "
	    << ay << " y + "
	    << az << " z\n\n";
	  
	  std::cout
	    << "Expected gradient:\n"
	    << "  ("
	    << ax << ", "
	    << ay << ", "
	    << az << ")\n\n";
	  
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


  //
  // Finalize Kokkos and MPI/STK.
  //
  Kokkos::finalize();

  stk::parallel_machine_finalize();

  return result;
}
