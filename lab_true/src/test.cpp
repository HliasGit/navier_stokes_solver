#include "NSSolver.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string  mesh_file_name = "../mesh/2dMeshCoarse.msh";
  const unsigned int degree_velocity = 2;
  const unsigned int degree_pressure = 1;

  NSSolver problem(mesh_file_name, degree_velocity, degree_pressure, 10);

  problem.run();
  problem.output();

  return 0;
}