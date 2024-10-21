#include "NSSolverStationary.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string  mesh_file_name  = "../mesh/2dMeshCylinder.msh";
  const unsigned int degree_velocity = 2;
  const unsigned int degree_pressure = 1;

  NSSolverStationary problem1(mesh_file_name, degree_velocity, degree_pressure);

  std::ofstream outputFile("output.txt"); 

  double time1 = (double) clock();            /* get initial time */
  time1 = time1 / CLOCKS_PER_SEC;      /*    in seconds    */

  //Use the diagonal preconditioner
  problem1.setup();
  problem1.solve_newton(1);
  problem1.output();
  problem1.compute_lift_drag();
  problem1.print_lift_coeff();
  problem1.print_drag_coeff();

  double timedif = ( ((double) clock()) / CLOCKS_PER_SEC) - time1;
  outputFile << "first: " << timedif << std::endl; // write data to the file

  NSSolverStationary problem2(mesh_file_name, degree_velocity, degree_pressure);

  time1 = (double) clock();            /* get initial time */
  time1 = time1 / CLOCKS_PER_SEC; 

  //Use the triangular preconditioner
  problem2.setup();
  problem2.solve_newton(0);
  problem2.output();
  problem2.compute_lift_drag();
  problem2.print_lift_coeff();
  problem2.print_drag_coeff();

  timedif = ( ((double) clock()) / CLOCKS_PER_SEC) - time1;
  outputFile << "second: " << timedif; // write data to the file
  outputFile.close();


  NSSolverStationary problem3(mesh_file_name, degree_velocity, degree_pressure);

  time1 = (double) clock();            /* get initial time */
  time1 = time1 / CLOCKS_PER_SEC; 

  //Use the simple preconditioner
  problem3.setup();
  problem3.solve_newton(2);
  problem3.output();
  problem3.compute_lift_drag();
  problem3.print_lift_coeff();
  problem3.print_drag_coeff();

  timedif = ( ((double) clock()) / CLOCKS_PER_SEC) - time1;

  

  return 0;
}