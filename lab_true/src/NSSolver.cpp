#include "NSSolver.hpp"

void
NSSolver::setup()
{
  setup_mesh();
  print_line();
  setup_finite_element();
  print_line();
  setup_dofs();
  print_line();
  setup_system();
  print_line();
}

void
NSSolver::setup_mesh()
{
  // Create the mesh.
  pcout << "Initializing the mesh" << std::endl;

  // First we read the mesh from file into a serial (i.e. not parallel)
  // triangulation.
  Triangulation<dim> mesh_serial;

  {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(mesh_path);
    grid_in.read_msh(grid_in_file);
  }

  // Then, we copy the triangulation into the parallel one.
  {
    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);
  }

  // Notice that we write here the number of *global* active cells (across all
  // processes).
  pcout << "  Number of elements = " << mesh.n_global_active_cells()
        << std::endl;
}

void
NSSolver::setup_finite_element()
{
  // Initialize the finite element space. This is the same as in serial codes.
  pcout << "Initializing the finite element space" << std::endl;

  const FE_SimplexP<dim> fe_scalar_velocity(degree_velocity);
  const FE_SimplexP<dim> fe_scalar_pressure(degree_pressure);
  fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity,
                                       dim,
                                       fe_scalar_pressure,
                                       1);

  pcout << "  Velocity degree:           = " << fe_scalar_velocity.degree
        << std::endl;
  pcout << "  Pressure degree:           = " << fe_scalar_pressure.degree
        << std::endl;
  pcout << "  DoFs per cell              = " << fe->dofs_per_cell << std::endl;

  quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);

  pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;

  quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);

  pcout << "  Quadrature points per face = " << quadrature_face->size()
        << std::endl;
}

void
NSSolver::setup_dofs()
{
  // Init Dofs
  pcout << "Initializing the DoF Handler" << std::endl;

  // Reinit DoF Handler after mesh loading
  dof_handler.reinit(mesh);

  // Distribute dofs to the fe space
  dof_handler.distribute_dofs(*fe);

  // We want to reorder DoFs so that all velocity DoFs come first, and then
  // all pressure DoFs.
  std::vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise(dof_handler, block_component);

  // We retrieve the set of locally owned DoFs, which will be useful when
  // initializing linear algebra classes.
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  // Besides the locally owned and locally relevant indices for the whole
  // system (velocity and pressure), we will also need those for the
  // individual velocity and pressure blocks.
  std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
  const unsigned int n_u = dofs_per_block[0];
  const unsigned int n_p = dofs_per_block[1];

  block_owned_dofs.resize(2);
  block_relevant_dofs.resize(2);
  block_owned_dofs[0]    = locally_owned_dofs.get_view(0, n_u);
  block_owned_dofs[1]    = locally_owned_dofs.get_view(n_u, n_u + n_p);
  block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
  block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

  pcout << "  Number of DoFs: " << std::endl;
  pcout << "    velocity = " << n_u << std::endl;
  pcout << "    pressure = " << n_p << std::endl;
  pcout << "    total    = " << n_u + n_p << std::endl;
}

void
NSSolver::setup_system()
{
  pcout << "Initializing the linear system" << std::endl;

  pcout << "  Initializing the sparsity pattern" << std::endl;

  // Velocity DoFs interact with other velocity DoFs (the weak formulation has
  // terms involving u times v), and pressure DoFs interact with velocity DoFs
  // (there are terms involving p times v or u times q). However, pressure
  // DoFs do not interact with other pressure DoFs (there are no terms
  // involving p times q). We build a table to store this information, so that
  // the sparsity pattern can be built accordingly.
  Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
  for (unsigned int c = 0; c < dim + 1; ++c)
    {
      for (unsigned int d = 0; d < dim + 1; ++d)
        {
          if (c == dim && d == dim) // pressure-pressure term
            coupling[c][d] = DoFTools::none;
          else // other combinations
            coupling[c][d] = DoFTools::always;
        }
    }

  TrilinosWrappers::BlockSparsityPattern sparsity(block_owned_dofs,
                                                  MPI_COMM_WORLD);
  DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity);
  sparsity.compress();

  // We also build a sparsity pattern for the pressure mass matrix.
  for (unsigned int c = 0; c < dim + 1; ++c)
    {
      for (unsigned int d = 0; d < dim + 1; ++d)
        {
          if (c == dim && d == dim) // pressure-pressure term
            coupling[c][d] = DoFTools::always;
          else // other combinations
            coupling[c][d] = DoFTools::none;
        }
    }
  TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(
    block_owned_dofs, MPI_COMM_WORLD);

  DoFTools::make_sparsity_pattern(dof_handler,
                                  coupling,
                                  sparsity_pressure_mass);
  sparsity_pressure_mass.compress();

  // Then, we use the sparsity pattern to initialize the system matrix. Since
  // the sparsity pattern is partitioned by row, so will the matrix.
  pcout << "  Initializing the system matrix" << std::endl;
  system_matrix.reinit(sparsity);
  pressure_mass_matrix.reinit(sparsity_pressure_mass);

  // Finally, we initialize the right-hand side and solution vectors.
  pcout << "  Initializing the system right-hand side" << std::endl;
  system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
  pcout << "  Initializing the solution vector" << std::endl;
  solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
  solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
  newton_update.reinit(block_owned_dofs, MPI_COMM_WORLD);
}

void
NSSolver::assemble(const bool initial_step, const bool assemble_matrix)
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();
  const unsigned int n_q_face      = quadrature_face->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(*fe,
                                   *quadrature_face,
                                   update_values | update_normal_vectors |
                                     update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_pressure_mass_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  if (assemble_matrix)
    {
      system_matrix = 0.0;
      pressure_mass_matrix = 0.0;
    }

  system_rhs = 0.0;

  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);


  // We use these vectors to store the old solution (i.e. at previous Newton
  // iteration) and its gradient on quadrature nodes of the current cell.
  std::vector<Tensor<1, dim>> velocity_solution_loc(n_q);
  std::vector<Tensor<2, dim>> velocity_solution_gradient_loc(n_q);
  std::vector<double>         pressure_solution_loc(n_q);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_matrix               = 0.0;
      cell_rhs                  = 0.0;
      cell_pressure_mass_matrix = 0.0;

      // We need to compute the Jacobian matrix and the residual for current
      // cell. This requires knowing the value and the gradient of u^{(k)}
      // (stored inside solution) on the quadrature nodes of the current
      // cell. This can be accomplished through
      // FEValues::get_function_values and FEValues::get_function_gradients.
      fe_values[velocity].get_function_values(solution, velocity_solution_loc);
      fe_values[velocity].get_function_gradients(
        solution, velocity_solution_gradient_loc);
      fe_values[pressure].get_function_values(solution, pressure_solution_loc);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double nu_loc = viscosity;

          //   Vector<double> forcing_term_loc(dim);
          //   forcing_term.vector_value(fe_values.quadrature_point(q),
          //                             forcing_term_loc);
          //   Tensor<1, dim> forcing_term_tensor;
          //   for (unsigned int d = 0; d < dim; ++d)
          //     forcing_term_tensor[d] = forcing_term_loc[d];

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              // a(u)(delta,v)
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // Frechet derivative
                  // First term
                  cell_matrix(i, j) += fe_values[velocity].value(j, q) *
                                       velocity_solution_gradient_loc[q] *
                                       fe_values[velocity].value(i, q) *
                                       fe_values.JxW(q);

                  // Second term
                  cell_matrix(i, j) += velocity_solution_loc[q] *
                                       fe_values[velocity].gradient(j, q) *
                                       fe_values[velocity].value(i, q) *
                                       fe_values.JxW(q);

                  // Third term - viscosity
                  cell_matrix(i, j) +=
                    nu_loc *
                    scalar_product(fe_values[velocity].gradient(j, q),
                                   fe_values[velocity].gradient(i, q)) *
                    fe_values.JxW(q);

                  // Pressure term in the momentum equation.
                  cell_matrix(i, j) -= fe_values[pressure].value(j, q) *
                                       fe_values[velocity].divergence(i, q) *
                                       fe_values.JxW(q);

                  // Pressure term in the continuity equation.
                  cell_matrix(i, j) += fe_values[pressure].value(i, q) *
                                       fe_values[velocity].divergence(j, q) *
                                       fe_values.JxW(q);

                  // Augmented Lagrangian term
                  cell_matrix(i, j) -=
                    gamma * fe_values[velocity].divergence(i, q) *
                    fe_values[velocity].divergence(j, q) * fe_values.JxW(q);

                  // Pressure mass matrix
                  cell_pressure_mass_matrix(i, j) +=
                    fe_values[pressure].value(i, q) *
                    fe_values[pressure].value(j, q) / nu_loc * fe_values.JxW(q);
                }

              //-R(u,v)
              // a(u,v)
              cell_rhs(i) -=
                nu_loc *
                scalar_product(velocity_solution_gradient_loc[q],
                               fe_values[velocity].gradient(i, q)) *
                fe_values.JxW(q);
              // c(u;u,v)
              cell_rhs(i) -= velocity_solution_loc[q] *
                             velocity_solution_gradient_loc[q] *
                             fe_values[velocity].value(i, q) * fe_values.JxW(q);
              // b(v,p)
              cell_rhs(i) += pressure_solution_loc[q] *
                             fe_values[velocity].divergence(i, q) *
                             fe_values.JxW(q);

              double velocity_solution_divergence_loc =
                trace(velocity_solution_gradient_loc[q]);

              // b(u,q) - pressure contribution in the continuity equation
              cell_rhs(i) += velocity_solution_divergence_loc *
                             fe_values[pressure].value(i, q) * fe_values.JxW(q);

              // TODO Forcing term
              //   cell_rhs(i) += scalar_product(forcing_term_tensor,
              //                                 fe_values[velocity].value(i,
              //                                 q)) *
              //                  fe_values.JxW(q);

              // Augmented Lagrangian
              cell_rhs(i) -= gamma * velocity_solution_divergence_loc *
                             fe_values[velocity].divergence(i, q) *
                             fe_values.JxW(q);
            }
        }


      // 6 borders
      // 7 inlet
      // 8 outlet

      // Boundary integral for Neumann BCs.
      if (cell->at_boundary())
        {
          for (unsigned int f = 0; f < cell->n_faces(); ++f)
            {
              if (cell->face(f)->at_boundary() &&
                  cell->face(f)->boundary_id() == 8)
                {
                  fe_face_values.reinit(cell, f);

                  for (unsigned int q = 0; q < n_q_face; ++q)
                    {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          cell_rhs(i) -=
                            p_out *
                            scalar_product(fe_face_values.normal_vector(q),
                                           fe_face_values[velocity].value(i,
                                                                          q)) *
                            fe_face_values.JxW(q);
                        }
                    }
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      if (assemble_matrix)
        {
          system_matrix.add(dof_indices, cell_matrix);
          system_rhs.add(dof_indices, cell_rhs);
        }
      else
        {
          system_rhs.add(dof_indices, cell_rhs);
        }


      if (assemble_matrix)
        {
          pressure_mass_matrix.add(dof_indices, cell_pressure_mass_matrix);
        }
    }

  system_matrix.compress(VectorOperation::add);
  pressure_mass_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  apply_dirichlet(solution);
}

void
NSSolver::assemble_system(const bool initial_step)
{
  assemble(initial_step, true);
}

void
NSSolver::assemble_rhs(const bool initial_step)
{
  assemble(initial_step, false);
}

void
NSSolver::solve(const bool initial_step)
{
  SolverControl solver_control(1000000,
                               1e-4 * system_rhs.l2_norm(),
                               true);

  SolverFGMRES<TrilinosWrappers::MPI::BlockVector> gmres(solver_control);

  gmres.solve(system_matrix, newton_update, system_rhs, PreconditionIdentity());
  pcout << "FGMRES steps: " << solver_control.last_step() << std::endl;

  apply_dirichlet(newton_update);
}

void
NSSolver::apply_dirichlet(TrilinosWrappers::MPI::BlockVector solution_to_apply)
{
  // Dirichlet Boundary conditions.
  std::map<types::global_dof_index, double> boundary_values;

  std::map<types::boundary_id, const Function<dim> *> boundary_functions;
  Functions::ZeroFunction<dim>                        zero_function;

  // 6 borders
  // 7 inlet
  // 8 outlet

  boundary_functions[7] = &inlet_velocity;
  // boundary_functions[8] = &functionG;


  VectorTools::interpolate_boundary_values(dof_handler,
                                           boundary_functions,
                                           boundary_values,
                                           ComponentMask({true, true, false}));

  boundary_functions.clear();
  boundary_functions[6] = &zero_function;

  VectorTools::interpolate_boundary_values(dof_handler,
                                           boundary_functions,
                                           boundary_values,
                                           ComponentMask({true, true, false}));


  // Check that solution vector is the right one
  MatrixTools::apply_boundary_values(
    boundary_values, system_matrix, solution_to_apply, system_rhs, false);
}


void
NSSolver::newton_iteration(const double       tolerance,
                           const unsigned int max_n_line_searches,
                           const bool         is_initial_step,
                           const bool         output_result)
{
  bool         first_step    = is_initial_step;
  double       last_res      = 1.0;
  double       current_res   = 1.0;
  unsigned int line_search_n = 0;
  while ((first_step || (current_res > tolerance)) &&
         line_search_n < max_n_line_searches)
    {
      if (first_step)
        {
          // setup_dofs();
          setup();
          evaluation_point = solution;
          assemble_system(first_step);
          solve(first_step);
          solution = newton_update;
          apply_dirichlet(solution);
          first_step       = false;
          evaluation_point = solution;
          assemble_rhs(first_step);
          current_res = system_rhs.l2_norm();
          std::cout << "The residual of initial guess is " << current_res
                    << std::endl;
          last_res = current_res;
        }
      else
        {
          evaluation_point = solution;
          assemble_system(first_step);
          solve(first_step);

          for (double alpha = 1.0; alpha > 1e-5; alpha *= 0.5)
            {
              evaluation_point = solution;
              evaluation_point.add(alpha, newton_update);
              apply_dirichlet(evaluation_point);
              assemble_rhs(first_step);
              current_res = system_rhs.l2_norm();
              std::cout << "  alpha: " << std::setw(10) << alpha << std::setw(0)
                        << "  residual: " << current_res << std::endl;
              if (current_res < last_res)
                break;
            }
          {
            solution = evaluation_point;
            std::cout << "  number of line searches: " << line_search_n
                      << "  residual: " << current_res << std::endl;
            last_res = current_res;
          }
          ++line_search_n;
        }
    }
}

void
NSSolver::compute_initial_guess(double step_size)
{
  const double target_Re = 1.0 / viscosity;

  bool is_initial_step = true;

  for (double Re = 1000.0; Re < target_Re;
       Re        = std::min(Re + step_size, target_Re))
    {
      viscosity = 1.0 / Re;
      std::cout << "Searching for initial guess with Re = " << Re << std::endl;
      newton_iteration(1e-12, 50, is_initial_step, false);
      is_initial_step = false;
    }
}

void
NSSolver::run()
{
  const double Re = 1.0 / viscosity;
  solution = 0.0;
  newton_update = 0.0;

  if (Re > 1000.0)
    {
      std::cout << "Searching for initial guess ..." << std::endl;
      const double step_size = 2000.0;
      compute_initial_guess(step_size);
      std::cout << "Found initial guess." << std::endl;
      std::cout << "Computing solution with target Re = " << Re << std::endl;
      viscosity = 1.0 / Re;
      newton_iteration(1e-12, 50, false, true);
    }
  else
    {
      newton_iteration(1e-12, 50, true, true);
    }
}

void
NSSolver::output()
{
  print_line();

  DataOut<dim> data_out;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);
  std::vector<std::string> names = {"velocity", "velocity", "pressure"};

  data_out.add_data_vector(dof_handler,
                           solution,
                           names,
                           data_component_interpretation);

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::string output_file_name = "output-ns";
  data_out.write_vtu_with_pvtu_record("./",
                                      output_file_name,
                                      0,
                                      MPI_COMM_WORLD);

  pcout << "Output written to " << output_file_name << std::endl;
  print_line();
}

void
NSSolver::print_line()
{
  pcout << "-----------------------------------------------" << std::endl;
}