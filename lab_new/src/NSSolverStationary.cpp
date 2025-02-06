#include "NSSolverStationary.hpp"

void NSSolverStationary::setup()
{
  // Create the mesh
  if (!read_mesh_from_file) 
  {
    {
      constexpr unsigned int dim = 2;

      // First: Create a full rectangular mesh with quadrilaterals.
      Triangulation<dim> full_tria;
      const Point<dim> bottom_left(0.0, 0.0);
      const Point<dim> top_right(2.2, 0.41);

      // Use a subdivision that gives reasonable resolution.
      std::vector<unsigned int> subdivisions{mesh_size_x, mesh_size_y};

      GridGenerator::subdivided_hyper_rectangle(full_tria,
                                                  subdivisions,
                                                  bottom_left,
                                                  top_right);

      // Define the circle parameters.
      const Point<dim> circle_center((bottom_left[0] + 0.2),
                                    (bottom_left[1] + top_right[1]) / 2.0);
      const double circle_radius = 0.05;

      // Prepare vectors to store vertices and cell connectivity.
      std::vector<Point<dim>> vertices;
      std::vector<CellData<dim>> cells;
      SubCellData subcell_data;

      // Copy vertices from the full triangulation.
      vertices.resize(full_tria.n_vertices());
      for (unsigned int i = 0; i < full_tria.n_vertices(); ++i)
        vertices[i] = full_tria.get_vertices()[i];

      // For each cell not inside the circle, copy its vertex indices.
      for (auto cell = full_tria.begin_active(); cell != full_tria.end(); ++cell)
      {
        // Skip the cell if its center lies inside the circle.
        if ((cell->center() - circle_center).norm() < circle_radius)
          continue;
        
        // Create a CellData object.
        CellData<dim> cell_data;
        cell_data.vertices.resize(GeometryInfo<dim>::vertices_per_cell);
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          cell_data.vertices[v] = cell->vertex_index(v);
        // get all the elements which have a distance with the circle center than is comprised between 0.05 +- element_size/2
        // if the element is inside the circle, we assign it a material id of 10
        if((cell->center() - circle_center).norm() < circle_radius + cell->diameter()/2 &&
            (cell->center() - circle_center).norm() > circle_radius - cell->diameter()/2)
        {
          cell_data.material_id = 10;
        }
        else
        {
          cell_data.material_id = 0;
        }
        cells.push_back(cell_data);
      }

      // Remove vertices that are not used in any cell.
      GridTools::delete_unused_vertices(vertices, cells, subcell_data);

      // Create a new triangulation using the filtered vertices and cells.
      Triangulation<dim> mesh_serial;
      mesh_serial.create_triangulation(vertices, cells, subcell_data);

      // Mark boundaries.
      // Loop over all active cells and then over each face.
      // - Left side (x = bottom_left[0]) -> boundary id 7 (inlet)
      // - Right side (x = top_right[0]) -> boundary id 8 (outlet)
      // - All other boundaries -> boundary id 6.
      for (auto cell = mesh_serial.begin_active(); cell != mesh_serial.end(); ++cell)
      {
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
        {
          if (cell->face(face)->at_boundary())
          {
            const Point<dim> face_center = cell->face(face)->center();
            if (std::fabs(face_center[0] - bottom_left[0]) < 1e-12)
              cell->face(face)->set_boundary_id(7); // inlet
            else if (std::fabs(face_center[0] - top_right[0]) < 1e-12)
              cell->face(face)->set_boundary_id(8); // outlet
            // if the current cell has id 10, assign it boundary id 10
            else if (cell->material_id() == 10)
              cell->face(face)->set_boundary_id(10); // circle
            else
              cell->face(face)->set_boundary_id(6); // all other boundaries (top, bottom, circular)
          }
        }
      }

      {
        GridTools::partition_triangulation(mpi_size, mesh_serial);
        const auto construction_data = TriangulationDescription::Utilities::
            create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
        mesh.create_triangulation(construction_data);
      }

      pcout << "  Number of elements = " << mesh.n_global_active_cells()
            << std::endl;

      // Output the mesh to a VTU file.
      GridOut grid_out;
      std::ofstream output_file("mesh.msh");
      grid_out.write_msh(mesh, output_file);
      pcout << "Mesh written to mesh.msh" << std::endl;
    }

    // Initialize the finite element space. This is the same as in serial codes.
    {
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
      pcout << "  DoFs per cell              = " << fe->dofs_per_cell
            << std::endl;

      quadrature = std::make_unique<QGauss<dim>>(fe->degree + 1);

      pcout << "  Quadrature points per cell = " << quadrature->size()
            << std::endl;

      quadrature_face = std::make_unique<QGauss<dim - 1>>(fe->degree + 1);

      pcout << "  Quadrature points per face = " << quadrature_face->size()
            << std::endl;
    }
  }
  else 
  {
    // Create the mesh.
    {
      pcout << "Initializing the mesh" << std::endl;

      // First we read the mesh from file into a serial (i.e. not parallel)
      // triangulation.
      Triangulation<dim> mesh_serial;

      {
        GridIn<dim> grid_in;
        grid_in.attach_triangulation(mesh_serial);
        pcout << "Mesh file name = " << mesh_file_name << std::endl;

        std::ifstream grid_in_file(mesh_file_name);
        grid_in.read_msh(grid_in_file);
        pcout << "Here1" << std::endl;
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

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the finite element space. This is the same as in serial codes.
    {
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
      pcout << "  DoFs per cell              = " << fe->dofs_per_cell
            << std::endl;

      quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);

      pcout << "  Quadrature points per cell = " << quadrature->size()
            << std::endl;

      quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);

      pcout << "  Quadrature points per face = " << quadrature_face->size()
            << std::endl;
    }
  }
  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);

    dof_handler.distribute_dofs(*fe);

    // We want to reorder DoFs so that all velocity DoFs come first, and then
    // all pressure DoFs.
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

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
    block_owned_dofs[0] = locally_owned_dofs.get_view(0, n_u);
    block_owned_dofs[1] = locally_owned_dofs.get_view(n_u, n_u + n_p);
    block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
    block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

    pcout << "  Number of DoFs: " << std::endl;
    pcout << "    velocity = " << n_u << std::endl;
    pcout << "    pressure = " << n_p << std::endl;
    pcout << "    total    = " << n_u + n_p << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
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

    TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(
        block_owned_dofs, MPI_COMM_WORLD);

    // Do the same for the pressure mass term.
    TrilinosWrappers::BlockSparsityPattern pressure_mass_sparsity(
        block_owned_dofs, MPI_COMM_WORLD);
    for (unsigned int c = 0; c < dim + 1; ++c)
    {
      for (unsigned int d = 0; d < dim + 1; ++d)
      {
        if (c == dim && d == dim) // terms with only pressure
          coupling[c][d] = DoFTools::always;
        else // terms with velocity
          coupling[c][d] = DoFTools::none;
      }
    }

    DoFTools::make_sparsity_pattern(dof_handler,
                                    coupling,
                                    sparsity_pressure_mass);
    sparsity_pressure_mass.compress();

    pcout << "  Initializing the matrices" << std::endl;
    jacobian_matrix.reinit(sparsity);
    pressure_mass.reinit(sparsity_pressure_mass);

    pcout << "  Initializing the system right-hand side" << std::endl;
    residual_vector.reinit(block_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
    delta_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);

    solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
  }
}

void NSSolverStationary::assemble_system(bool global_first_iter, bool computing_stokes)
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();
  const unsigned int n_q_face = quadrature_face->size();

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
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  jacobian_matrix = 0.0;
  residual_vector = 0.0;
  pressure_mass = 0.0;

  // Extractors
  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  // We use the following vectors to store the old solution (i.e. at previous Newton
  // iteration) and its gradient on quadrature nodes of the current cell.
  std::vector<Tensor<1, dim>> velocity_loc(n_q);
  std::vector<Tensor<2, dim>> velocity_gradient_loc(n_q);
  std::vector<double> pressure_loc(n_q);

  // tensor used to store the nonlinear term contribution in each quadrature point
  // useful when computing (u . nabla) uv, corresponding to c(u;u,v) term
  Tensor<1, dim> nonlinear_term;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);

    cell_matrix = 0.0;
    cell_rhs = 0.0;
    cell_pressure_mass_matrix = 0.0;

    // We need to compute the Jacobian matrix and the residual for current
    // cell. This requires knowing the value and the gradient of u^{(k)}
    // (stored inside solution) on the quadrature nodes of the current
    // cell. This can be accomplished through
    // FEValues::get_function_values and FEValues::get_function_gradients.
    fe_values[velocity].get_function_values(solution, velocity_loc);
    fe_values[velocity].get_function_gradients(solution,
                                               velocity_gradient_loc);
    fe_values[pressure].get_function_values(solution, pressure_loc);

    for (unsigned int q = 0; q < n_q; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          if (global_first_iter || computing_stokes)
          {
            // Viscosity term.
            cell_matrix(i, j) +=
                nu *
                scalar_product(fe_values[velocity].gradient(i, q),
                               fe_values[velocity].gradient(j, q)) *
                fe_values.JxW(q);

            // Pressure term in the momentum equation.
            cell_matrix(i, j) -= fe_values[velocity].divergence(i, q) *
                                 fe_values[pressure].value(j, q) *
                                 fe_values.JxW(q);

            // Pressure term in the continuity equation.
            cell_matrix(i, j) -= fe_values[velocity].divergence(j, q) *
                                 fe_values[pressure].value(i, q) *
                                 fe_values.JxW(q);

            // Pressure mass matrix.
            cell_pressure_mass_matrix(i, j) +=
                fe_values[pressure].value(i, q) *
                fe_values[pressure].value(j, q) / nu * fe_values.JxW(q);
          }

          else
          {
            // Compute both terms associated with Newton linearization of (u . nabla) u
            // nabla u is a tensor, iterate over both dimensions
            for (unsigned int k = 0; k < dim; k++)
            {
              nonlinear_term[k] = 0.0;
              for (unsigned int l = 0; l < dim; l++)
              {
                // compute both terms yielded by the Frechet derivative
                // (u . nabla) u_old
                nonlinear_term[k] += velocity_loc[q][l] *
                                     fe_values[velocity].gradient(j, q)[k][l];

                // (u_old . nabla) u
                nonlinear_term[k] += fe_values[velocity].value(j, q)[l] *
                                     velocity_gradient_loc[q][k][l];
              }
            }

            // assemble the linearized convective term (u . nabla) uv
            cell_matrix(i, j) += scalar_product(nonlinear_term, fe_values[velocity].value(i, q)) * fe_values.JxW(q);

            // Third term - viscosity
            cell_matrix(i, j) +=
                nu *
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

            // Pressure mass matrix used for preconditioning
            cell_pressure_mass_matrix(i, j) +=
                fe_values[pressure].value(i, q) *
                fe_values[pressure].value(j, q) / nu * fe_values.JxW(q);
          }
        }

        if (global_first_iter || computing_stokes)
        {
          continue;
        }

        //-R(u,v)
        // a(u,v)
        cell_rhs(i) -=
            nu *
            scalar_product(velocity_gradient_loc[q],
                           fe_values[velocity].gradient(i, q)) *
            fe_values.JxW(q);

        // Compute residual associated with Newton linearization of (u . nabla) u
        // nabla u is a tensor, iterate over both dimensions
        for (unsigned int k = 0; k < dim; k++)
        {
          nonlinear_term[k] = 0.0;
          for (unsigned int l = 0; l < dim; l++)
          {
            // compute (u_old . nabla) u_old
            nonlinear_term[k] += velocity_loc[q][l] *
                                 velocity_gradient_loc[q][k][l];
          }
        }

        // assemble the residual associated with the nonlinear convective term
        cell_rhs(i) -= scalar_product(nonlinear_term, fe_values[velocity].value(i, q)) * fe_values.JxW(q);

        // b(v,p)
        cell_rhs(i) += pressure_loc[q] *
                       fe_values[velocity].divergence(i, q) *
                       fe_values.JxW(q);

        double velocity_divergence_loc = trace(velocity_gradient_loc[q]);

        // b(u,q) - pressure contribution in the continuity equation
        cell_rhs(i) += velocity_divergence_loc *
                       fe_values[pressure].value(i, q) * fe_values.JxW(q);
      }
    }

    // IDs for the mesh Neumann BCs
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

    jacobian_matrix.add(dof_indices, cell_matrix);
    residual_vector.add(dof_indices, cell_rhs);
    pressure_mass.add(dof_indices, cell_pressure_mass_matrix);
  }

  jacobian_matrix.compress(VectorOperation::add);
  residual_vector.compress(VectorOperation::add);
  pressure_mass.compress(VectorOperation::add);

  // Dirichlet Boundary conditions.
  {
    std::map<types::global_dof_index, double> boundary_values;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    Functions::ZeroFunction<dim> zero_function(dim + 1);

    boundary_values.clear();

    // Apply the non-homogeneous Dirichlet BCs to inlet_velocity only in the first step 
    if (global_first_iter)
    {
      boundary_functions[7] = &inlet_velocity;
    }
    else
    {
      boundary_functions[7] = &zero_function;
    }

    // Dirichlet conditions are not applied to pressure degrees of freedom
    // for this purpose use a component mask
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values,
                                             ComponentMask(
                                                 {true, true, false}));

    boundary_functions[6] = &zero_function;
    boundary_functions[10] = &zero_function;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values,
                                             ComponentMask(
                                                 {true, true, false}));

    MatrixTools::apply_boundary_values(
        boundary_values, jacobian_matrix, delta_owned, residual_vector, false);
  }
}

int NSSolverStationary::solve_system() {
  SolverControl solver_control(20000, tolerance);
  // Choose the correct preconditioner
  if (preconditioner_type == 0) {
      PreconditionBlockDiagonal preconditioner;
      preconditioner.initialize(jacobian_matrix.block(0, 0),
                                    pressure_mass.block(1, 1));

      if (solver_type == 0) {
          SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);
          solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
      }
      else if (solver_type == 1) {
          SolverFGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);
          solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
      }
      else if (solver_type == 2) {
          SolverBicgstab<TrilinosWrappers::MPI::BlockVector> solver(solver_control);
          solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
      }
  }
  else if (preconditioner_type == 1) {
      PreconditionBlockTriangular preconditioner;
      preconditioner.initialize(jacobian_matrix.block(0, 0),
                                    pressure_mass.block(1, 1),
                                    jacobian_matrix.block(1, 0));

      if (solver_type == 0) {
          SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);
          solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
      }
      else if (solver_type == 1) {
          SolverFGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);
          solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
      }
      else if (solver_type == 2) {
          SolverBicgstab<TrilinosWrappers::MPI::BlockVector> solver(solver_control);
          solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
      }
  }
  else if (preconditioner_type == 2) {
      double alpha = 0.5;
      PreconditionaSIMPLE preconditioner;
      preconditioner.initialize(jacobian_matrix.block(0, 0),
                                    jacobian_matrix.block(1, 0),
                                    jacobian_matrix.block(0, 1),
                                    solution_owned,
                                    alpha);

      if (solver_type == 0) {
          SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);
          solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
      }
      else if (solver_type == 1) {
          SolverFGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);
          solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
      }
      else if (solver_type == 2) {
          SolverBicgstab<TrilinosWrappers::MPI::BlockVector> solver(solver_control);
          solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
      }
  }
  else {
      throw std::invalid_argument("Invalid preconditioner type. Use 0: blockDiagonal, 1: blockTriangular, 2: aSIMPLE.");
  }
    
  pcout << "   " << solver_control.last_step() << " solver iterations" << std::endl;
  return solver_control.last_step();
}

void NSSolverStationary::solve_newton()
{
  pcout << "===============================================" << std::endl;

  const unsigned int n_max_iters = 15;
  const double residual_tolerance = 1e-9;
  double target_Re = Re;
  bool global_first_iter = true;
  bool computing_stokes = true;
  bool inlet_reached = false;
  pcout << "Target Re = " << target_Re << std::endl;

  // First solve the Stokes problem, then becomes a NS with an ingreasingly more dominant convective term.
  for (double current_Re = 10.0; current_Re <= target_Re; current_Re += 20.0)
  {
    pcout << "===============================================" << std::endl;
    nu = 1.0 / current_Re;
    inlet_reached = false;
    pcout << "Solving for nu = " << nu << ", Re = " << get_reynolds() << std::endl;

      while(!inlet_reached) {
        pcout << "Solving for inlet velocity: " << inlet_velocity.getVelocity() << std::endl;
        if(global_first_iter)
          pcout << "Solving Stokes adding BCs" << std::endl;
        else if(computing_stokes)
          pcout << "Solving Stokes without adding BCs" << std::endl;
        else
          pcout << "Solving NS" << std::endl;
        
        unsigned int n_iter = 0;
        double residual_norm = residual_tolerance + 1;
        double prev_residual;
        int GMRES_iter = 0;

        while (n_iter < n_max_iters && residual_norm > residual_tolerance)
        {
          if (global_first_iter)
          {
            global_first_iter = false;
            assemble_system(true, true);
          }
          else
          {
            if(computing_stokes)
              assemble_system(false, true);
            else
              assemble_system(false, false);
          }

          residual_norm = residual_vector.l2_norm();

          prev_residual = n_iter == 0 ? residual_norm + 1 : prev_residual;

          pcout << "Newton iteration " << n_iter << "/" << n_max_iters
                << " - ||r|| = " << std::scientific << std::setprecision(6)
                << residual_norm << std::flush;

          // We actually solve the system only if the residual is larger than the
          // tolerance.
          if (residual_norm > residual_tolerance)
          {
            GMRES_iter = solve_system();

            if(GMRES_iter == 0)
              break;           

            evaluation_point = solution;

            // Update the solution
            for (double alpha = 1; alpha > 1e-12; alpha *= 0.1)
            {
              solution_owned = evaluation_point;
              solution_owned.add(alpha, delta_owned);
              solution = solution_owned;

              if(computing_stokes)
                assemble_system(false, true);
              else
                assemble_system(false, false);

              residual_norm = residual_vector.l2_norm();

              pcout << "  Evaluating alpha=" << alpha << ", ||r||=" << residual_norm << std::endl;

              if (residual_norm < prev_residual)
                break;
            }

            prev_residual = residual_norm;
          }
          else
          {
            // newton method already converged for the current Re number, print tolerance and output
            pcout << " < tolerance" << std::endl;
            output();
            break;
          }
          output();
          ++n_iter;
        }

        // Increment inlet velocity
        inlet_reached = inlet_velocity.incrementVelocity(get_reynolds());
        if(inlet_reached)
          computing_stokes = false;
      }
      output();
  }
  pcout << "===============================================" << std::endl;
}

double NSSolverStationary::get_reynolds() const
{
  return get_avg_inlet_velocity() * 0.1 / nu;
}

void NSSolverStationary::output() const
{
  pcout << "===============================================" << std::endl;

  DataOut<dim> data_out;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
  std::vector<std::string> names = {"velocity",
                                    "velocity",
                                    "pressure"};

  data_out.add_data_vector(dof_handler,
                           solution,
                           names,
                           data_component_interpretation);

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::string output_file_name = "output-stokes";
  data_out.write_vtu_with_pvtu_record("./",
                                      output_file_name,
                                      0,
                                      MPI_COMM_WORLD);

  pcout << "Output written to " << output_file_name << std::endl;
  pcout << "===============================================" << std::endl;
}

void NSSolverStationary::compute_lift_drag()
{
  pcout << "===============================================" << std::endl;
  pcout << "Computing lift and drag forces" << std::endl;

  // variables to store lift and drag forces
  double local_lift_force = 0.0;
  double local_drag_force = 0.0;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q_face = quadrature_face->size();

  // need to iterate over all the cells corresponding to the cylindrical obstacle in order to compute the forces
  FEFaceValues<dim> fe_face_values(*fe,
                                   *quadrature_face,
                                   update_values | update_quadrature_points | update_gradients | update_normal_vectors |
                                       update_JxW_values);

  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // We use the following vectors to store the old solution (i.e. at previous Newton
  // iteration) and its gradient on quadrature nodes of the current cell.
  std::vector<Tensor<1, dim>> velocity_loc(n_q_face);
  std::vector<Tensor<2, dim>> velocity_gradient_loc(n_q_face);
  std::vector<double> pressure_loc(n_q_face);

  // declare shear stress tensor and force tensor
  Tensor<2, dim> shear_stress;
  Tensor<1, dim> force;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    for (unsigned int f = 0; f < cell->n_faces(); ++f)
    {
      if (cell->face(f)->at_boundary() &&
          cell->face(f)->boundary_id() == 10)
      {
        fe_face_values.reinit(cell, f);

        fe_face_values[velocity].get_function_values(solution, velocity_loc);

        fe_face_values[velocity].get_function_gradients(solution, velocity_gradient_loc);

        fe_face_values[pressure].get_function_values(solution, pressure_loc);

        for (unsigned int q = 0; q < n_q_face; ++q)
        {
          pcout << "Computing drag and lift forces" << std::endl;
          // Get the normal vector to the cylinder surface
          // note that the normal vector is pointing in the opposite direction with
          // respect to the one in the provided formulae
          const Tensor<1, dim> &negative_normal_vector = fe_face_values.normal_vector(q);

          // Calculate the shear stress tensor (which is coplanar with the cylinder cross section)
          // it is the component of the force vector parallel to the cylinder cross section
          // shear stress = nu * (grad u + grad u^T) - p * I
          shear_stress = velocity_gradient_loc[q];
          for (unsigned int i = 0; i < dim; i++)
          {
            for (unsigned int j = 0; j < dim; j++)
            {
              // sum the transpose of the velocity gradient tensor
              shear_stress[i][j] += velocity_gradient_loc[q][j][i];
            }
          }
          shear_stress *= nu;
          for (unsigned int i = 0; i < dim; i++)
          {
            shear_stress[i][i] -= pressure_loc[q];
          }

          // compute the force vector acting on the cylinder along both spatial directions
          // also invert the sign of the normal vector
          force = -shear_stress * negative_normal_vector *
                  fe_face_values.JxW(q);

          // Update drag and lift forces
          // drag force is the component of the force vector parallel to the flow direction
          local_drag_force += force[0];
          // lift force is the component of the force vector perpendicular to the flow direction
          local_lift_force += force[1];
        }
      }
    }
  }

  // Sum all the forces contributions that have been computed by each process in parallel
  lift_force = Utilities::MPI::sum(local_lift_force, MPI_COMM_WORLD);
  drag_force = Utilities::MPI::sum(local_drag_force, MPI_COMM_WORLD);
}

double NSSolverStationary::get_avg_inlet_velocity() const
{
  // U_avg = 2 * U(0, H/2) / 3
  return 2 * inlet_velocity.value(Point<dim>(0, 0.41 / 2.0)) / 3;
}

void NSSolverStationary::compute_lift_coeff()
{
  const double U_avg = get_avg_inlet_velocity();
  // lift coefficient = 2 * lift_force / (U_avg * U_avg * D)
  // where D is the diameter of the cylinder
  lift_coeff = 2 * lift_force / (U_avg * U_avg * 0.1);
}

void NSSolverStationary::compute_drag_coeff()
{
  const double U_avg = get_avg_inlet_velocity();
  // drag coefficient = 2 * drag_force / (U_avg * U_avg * D)
  // where D is the diameter of the cylinder
  drag_coeff = 2 * drag_force / (U_avg * U_avg * 0.1);
}

void NSSolverStationary::print_lift_coeff()
{
  pcout << "===============================================" << std::endl;
  compute_lift_coeff();
  pcout << "Lift coefficient: " << lift_coeff << std::endl;
}

void NSSolverStationary::print_drag_coeff()
{
  pcout << "===============================================" << std::endl;
  compute_drag_coeff();
  pcout << "Drag coefficient: " << drag_coeff << std::endl;
}