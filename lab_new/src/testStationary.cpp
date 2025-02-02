#include "NSSolverStationary.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string  mesh_file_name  = "../mesh/2dMeshFine.msh";
  const unsigned int degree_velocity = 3;
  const unsigned int degree_pressure = 2;

  NSSolverStationary problem(mesh_file_name, degree_velocity, degree_pressure);

  problem.setup();
  problem.solve_newton();
  problem.output();
  problem.compute_lift_drag();
  problem.print_lift_coeff();
  problem.print_drag_coeff();

  return 0;
}