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

  const FE_Q<dim> fe_scalar_velocity(degree_velocity);
  const FE_Q<dim> fe_scalar_pressure(degree_pressure);
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

  // Clear system matrices
  system_matrix.clear();
  pressure_mass_matrix.clear();

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
  if (assemble_matrix)
    {
      system_matrix = 0;
    }

  system_rhs = 0;

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(*fe,
                                   *quadrature_face,
                                   update_values | update_normal_vectors |
                                     update_JxW_values);

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q_points    = quadrature->size();
  const unsigned int n_q_face      = quadrature_face->size();

  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_pressure_mass_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     local_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // We create temporary storage for present velocity and gradient, and present
  // pressure
  std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
  std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);
  std::vector<double>         present_pressure_values(n_q_points);

  std::vector<double>         div_phi_u(dofs_per_cell);
  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  std::vector<double>         phi_p(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      local_matrix = 0;
      local_rhs    = 0;

      fe_values[velocity].get_function_values(evaluation_point,
                                              present_velocity_values);

      fe_values[velocity].get_function_gradients(evaluation_point,
                                                 present_velocity_gradients);

      fe_values[pressure].get_function_values(evaluation_point,
                                              present_pressure_values);

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              div_phi_u[k]  = fe_values[velocity].divergence(k, q);
              grad_phi_u[k] = fe_values[velocity].gradient(k, q);
              phi_u[k]      = fe_values[velocity].value(k, q);
              phi_p[k]      = fe_values[pressure].value(k, q);
            }

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              if (assemble_matrix)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      local_matrix(i, j) +=
                        (phi_u[i] *
                         (present_velocity_gradients[q] * phi_u[j])) *
                        fe_values.JxW(q);

                      local_matrix(i, j) +=
                        (phi_u[i] *
                         (grad_phi_u[j] * present_velocity_values[q])) *
                        fe_values.JxW(q);

                      local_matrix(i, j) +=
                        viscosity *
                        scalar_product(grad_phi_u[i], grad_phi_u[j]) *
                        fe_values.JxW(q);

                      // Pressure term in the momentum equation.
                      local_matrix(i, j) -=
                        div_phi_u[i] * phi_p[j] * fe_values.JxW(q);

                      // Pressure term in the continuity equation.
                      local_matrix(i, j) -=
                        phi_p[i] * div_phi_u[j] * fe_values.JxW(q);

                      // Augmented Lagrangian term
                      local_matrix(i, j) +=
                        gamma * div_phi_u[i] * div_phi_u[j] * fe_values.JxW(q);

                      // Pressure mass
                      local_matrix(i, j) +=
                        phi_p[i] * phi_p[j] * fe_values.JxW(q);
                    }
                }

              // RHS -R(u,v)
              double present_velocity_divergence =
                trace(present_velocity_gradients[q]);

              // a(u,v)
              local_rhs(i) -=
                viscosity *
                scalar_product(grad_phi_u[i], present_velocity_gradients[q]) *
                fe_values.JxW(q);

              // c(u;u,v)
              local_rhs(i) -= (phi_u[i] * (present_velocity_gradients[q] *
                                           present_velocity_values[q])) *
                              fe_values.JxW(q);

              // b(v,p)
              local_rhs(i) +=
                div_phi_u[i] * present_pressure_values[q] * fe_values.JxW(q);

              // b(u,q) - pressure contribution in the continuity equation
              local_rhs(i) +=
                phi_p[i] * present_velocity_divergence * fe_values.JxW(q);

              // TODO: Forcing Term

              // Augmented Lagrangian
              local_rhs(i) += gamma * div_phi_u[i] *
                              present_velocity_divergence * fe_values.JxW(q);
            }
        }
    }
}

void
NSSolver::print_line()
{
  pcout << "-----------------------------------------------" << std::endl;
}