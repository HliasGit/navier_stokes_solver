#include "NSSolverStationary.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string  mesh_file_name  = "../mesh/2dMesh.msh";
  const unsigned int degree_velocity = 2;
  const unsigned int degree_pressure = 1;

  NSSolverStationary problem(mesh_file_name, degree_velocity, degree_pressure);

  problem.setup();
  problem.solve_newton();
  problem.output();
  problem.compute_lift_drag();
  problem.print_lift();
  problem.print_drag();

  return 0;
}