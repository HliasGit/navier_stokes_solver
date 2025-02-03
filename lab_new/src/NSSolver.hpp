#ifndef NSSOLVER_HPP
#define NSSOLVER_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// Class representing the non-linear diffusion problem.
class NSSolver
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;

  // Function for inlet velocity. This actually returns an object with three
  // components (one for each velocity component, and one for the pressure), but
  // then only the first one is really used (we have an inlent only along the x
  // axis). If we only return one component, however, we may get an error
  // message due to this function being incompatible with the finite element
  // space.
  class InletVelocity : public Function<dim>
  {
  public:
    InletVelocity()
        : Function<dim>(dim + 1)
    {
    }

    virtual void
    vector_value(const Point<dim> &p,
                 Vector<double> &values) const override
    {
      // in flow condition is: 4 * U_m * (H - y) / H^2
      values[0] = 4 * U_m * p[1] * (H - p[1]) / (H * H);
      // values[0] = 4 * U_m * p[1] * (H - p[1]) * sin(M_PI * get_time() / 8) / (H * H);

      for (unsigned int i = 1; i < dim + 1; ++i)
        values[i] = 0.0;
    }

    virtual double
    value(const Point<dim> &p,
          const unsigned int component = 0) const override
    {
      if (component == 0)
        return 4 * U_m * p[1] * (H - p[1]) / (H * H);
      // return 4 * U_m * p[1] * (H - p[1]) * sin(M_PI * get_time() / 8) / (H * H);
      else
        return 0.0;
    }
    const double U_m = 0.3;
    const double H = 0.41;
  };

  // Function for the forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    ForcingTerm()
        : Function<dim>(dim + 1)
    {
    }

    virtual void
    vector_value(const Point<dim> & /*p*/,
                 Vector<double> &values) const override
    {
      for (unsigned int i = 0; i < dim - 1; ++i)
        values[i] = 0.0;

      // values[0] = std::sin(get_time() * 2.0 * numbers::PI);
      values[0] = 0.0;
    }

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      if (component == 0)
        return 0.0;
      // return std::sin(get_time() * 2.0 * numbers::PI);
      else
        return 0.0;
    }
  };

  // Function for initial conditions.
  class FunctionU0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  // Preconditioner
  // Block-diagonal preconditioner.
  class PreconditionBlockDiagonal
  {
  public:
    // Initialize the preconditioner, given the velocity stiffness matrix, the
    // pressure mass matrix.
    void
    initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
               const TrilinosWrappers::SparseMatrix &pressure_mass_)
    {
      velocity_stiffness = &velocity_stiffness_;
      pressure_mass = &pressure_mass_;

      preconditioner_velocity.initialize(velocity_stiffness_);
      preconditioner_pressure.initialize(pressure_mass_);
    }

    // Application of the preconditioner.
    void
    vmult(TrilinosWrappers::MPI::BlockVector &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      SolverControl solver_control_velocity(1000,
                                            1e-1);
      SolverFGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_velocity(
          solver_control_velocity);
      solver_gmres_velocity.solve(*velocity_stiffness,
                                  dst.block(0),
                                  src.block(0),
                                  preconditioner_velocity);

      SolverControl solver_control_pressure(1000,
                                            1e-1);
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
          solver_control_pressure);
      solver_cg_pressure.solve(*pressure_mass,
                               dst.block(1),
                               src.block(1),
                               preconditioner_pressure);
    }

  protected:
    // Velocity stiffness matrix.
    const TrilinosWrappers::SparseMatrix *velocity_stiffness;

    // Preconditioner used for the velocity block.
    TrilinosWrappers::PreconditionILU preconditioner_velocity;

    // Pressure mass matrix.
    const TrilinosWrappers::SparseMatrix *pressure_mass;

    // Preconditioner used for the pressure block.
    TrilinosWrappers::PreconditionILU preconditioner_pressure;
  };

  // Block-triangular preconditioner.
  class PreconditionBlockTriangular
  {
  public:
    // Initialize the preconditioner, given the velocity stiffness matrix, the
    // pressure mass matrix.
    void
    initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
               const TrilinosWrappers::SparseMatrix &pressure_mass_,
               const TrilinosWrappers::SparseMatrix &B_)
    {
      velocity_stiffness = &velocity_stiffness_;
      pressure_mass = &pressure_mass_;
      B = &B_;

      preconditioner_velocity.initialize(velocity_stiffness_);
      preconditioner_pressure.initialize(pressure_mass_);
    }

    // Application of the preconditioner.
    void
    vmult(TrilinosWrappers::MPI::BlockVector &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      SolverControl solver_control_velocity(2000001,
                                            1e-4 * src.block(0).l2_norm());
      SolverFGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_velocity(
          solver_control_velocity);
      solver_gmres_velocity.solve(*velocity_stiffness,
                                  dst.block(0),
                                  src.block(0),
                                  preconditioner_velocity);

      tmp.reinit(src.block(1));
      B->vmult(tmp, dst.block(0));
      tmp.sadd(-1.0, src.block(1));

      SolverControl solver_control_pressure(2000000,
                                            1e-5 * src.block(1).l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
          solver_control_pressure);
      solver_cg_pressure.solve(*pressure_mass,
                               dst.block(1),
                               tmp,
                               preconditioner_pressure);
    }

  protected:
    // Velocity stiffness matrix.
    const TrilinosWrappers::SparseMatrix *velocity_stiffness;

    // Preconditioner used for the velocity block.
    TrilinosWrappers::PreconditionILU preconditioner_velocity;

    // Pressure mass matrix.
    const TrilinosWrappers::SparseMatrix *pressure_mass;

    // Preconditioner used for the pressure block.
    TrilinosWrappers::PreconditionILU preconditioner_pressure;

    // B matrix.
    const TrilinosWrappers::SparseMatrix *B;

    // Temporary vector.
    mutable TrilinosWrappers::MPI::Vector tmp;
  };

  class PreconditionaSIMPLE
  {
  public:
    // Initialize the preconditioner
    void
    initialize(const TrilinosWrappers::SparseMatrix &F_,
               const TrilinosWrappers::SparseMatrix &B_neg_,
               const TrilinosWrappers::SparseMatrix &B_t_,
               const TrilinosWrappers::MPI::BlockVector &vector_,
               const double &alpha_)
    {
      F_matrix = &F_;
      B_neg_matrix = &B_neg_;
      B_t_matrix = &B_t_;
      alpha = alpha_;

      D_vector.reinit(vector_.block(0));
      D_inv_vector.reinit(vector_.block(0));
      // compute diag(F) and diag(F)^{-1} and save it to the respective vector
      for (unsigned int i : D_vector.locally_owned_elements())
      {
        const double tmp = F_matrix->diag_element(i);
        D_vector[i] = tmp;
        D_inv_vector[i] = 1.0 / tmp;
      }

      // assemble approximate of Schur complement as S = B * D_inv_vector * B^T
      B_neg_matrix->mmult(S_neg_matrix, *B_t_matrix, D_inv_vector);

      // initialize preconditioners with corresponding matrices
      preconditioner_F.initialize(F_);
      preconditioner_S.initialize(S_neg_matrix);
    }

    // Application of the preconditioner.
    void
    vmult(TrilinosWrappers::MPI::BlockVector &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      // reinit temp vector to store intermediate results
      tmp.reinit(src);

      // compute multiplication [F^{-1} 0; 0 I] * src
      // solve linear system associated with F^{-1} * src.block(0) and store result in dst.block(0)

      // SolverControl solver_control_F(10000001,
      //                                1e-1 * src.block(0).l2_norm());
      // SolverFGMRES<TrilinosWrappers::MPI::Vector> solver_F(
      //     solver_control_F);

      // solver_F.solve(*F_matrix,
      //                dst.block(0),
      //                src.block(0),
      //                preconditioner_F);

      // std::cout << "Converged!" << std::endl;

      // // if do not want to use direct solver
      preconditioner_F.vmult(dst.block(0), src.block(0));

      // store src.block(1) in tmp.block(1)
      tmp.block(1) = src.block(1);

      // compute multiplication by [I 0; -B I]
      // compute -B * dst.block(0) + tmp.block(1) and store result in tmp.block(1)
      B_neg_matrix->vmult_add(tmp.block(1), dst.block(0));

      // compute multiplication by [I 0; 0 -S^{-1}]
      // solve linear system associated with the approximate Schur complement
      // SolverControl solver_control_S(10000000,
      //                                1e-1 * tmp.block(1).l2_norm());
      // SolverFGMRES<TrilinosWrappers::MPI::Vector> solver_S(
      //     solver_control_S);
      // solver_S.solve(S_neg_matrix,
      //                dst.block(1),
      //                tmp.block(1),
      //                preconditioner_S);

      // // if do not want to use direct solver
      preconditioner_S.vmult(dst.block(1), tmp.block(1));

      // compute multiplication by [D 0; 0 I*1/alpha]
      dst.block(0).scale(D_vector);
      dst.block(1) *= 1.0 / alpha;

      // compute multiplication by [I -B^T; 0 I]
      B_t_matrix->vmult(tmp.block(0), dst.block(1));
      dst.block(0) -= tmp.block(0);

      // compute multiplication by [D^{-1} 0; 0 I]
      dst.block(0).scale(D_inv_vector);
    }

  protected:
    // F = 1/delta_t * M + A + C, where M is the mass matrix, A is the stiffness matrix
    // and C is the matrix corresponding to the linearized convective term
    const TrilinosWrappers::SparseMatrix *F_matrix;

    // vector obtained from diag(F)
    TrilinosWrappers::MPI::Vector D_vector;

    // vector obtained from diag(F)^{-1}, thus inverse of the diag(F)
    TrilinosWrappers::MPI::Vector D_inv_vector;

    // approximation of the Schur complement -S=-BD^{-1}B^T
    TrilinosWrappers::SparseMatrix S_neg_matrix;

    // damping parameter alpha in [0,1]
    double alpha;

    // Preconditioner used to approximate F^{-1}
    TrilinosWrappers::PreconditionILU preconditioner_F;

    // Preconditioner used to approximate S^{-1}
    TrilinosWrappers::PreconditionILU preconditioner_S;

    // B matrix.
    const TrilinosWrappers::SparseMatrix *B_neg_matrix;

    // B transpose matrix, needed to compute S
    const TrilinosWrappers::SparseMatrix *B_t_matrix;

    // Temporary vector for intermediate results
    // mutable type is used for const-correctness 
    mutable TrilinosWrappers::MPI::BlockVector tmp;
  };

public:
  bool apply_first = true;

  // Constructor.
  NSSolver(const std::string &mesh_file_name_,
           const unsigned int &degree_velocity_,
           const unsigned int &degree_pressure_,
           const double &T_,
           const double &delta_t_)
      : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)), mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)), pcout(std::cout, mpi_rank == 0), mesh_file_name(mesh_file_name_), mesh(MPI_COMM_WORLD), degree_velocity(degree_velocity_), degree_pressure(degree_pressure_), T(T_), delta_t(delta_t_)
  {
  }

  // Initialization.
  void
  setup();

  // Solve the problem using Newton's method.
  void
  solve_newton();

  // Output.
  void
  output(const unsigned int &time_step) const;

  void
  solve();

protected:
  // Assemble the tangent problem.
  void
  assemble_system(bool first_iter);

  // Solve the tangent problem.
  int solve_system();

  double get_reynolds() const;

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  // Kinematic viscosity [m2/s]
  double nu = 1.0;

  // Inlet velocity
  InletVelocity inlet_velocity;

  // Initial conditions
  FunctionU0 u_0;

  // Pressure out [Pa]
  const double p_out = 1.0;

  // Forcing term
  ForcingTerm forcing_term;

  // Lagrangian
  // const double gamma = 1.0;

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh file name.
  const std::string &mesh_file_name;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Polynomial degrees.
  const unsigned int degree_velocity;
  const unsigned int degree_pressure;

  // current time
  double time;

  // final time
  double T;

  // Time step.
  const double delta_t;

  // Finite element space.
  std::unique_ptr<FESystem<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // Quadrature formula for face integrals.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs owned by current process in the velocity and pressure blocks.
  std::vector<IndexSet> block_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // DoFs relevant to current process in the velocity and pressure blocks.
  std::vector<IndexSet> block_relevant_dofs;

  // Jacobian matrix.
  TrilinosWrappers::BlockSparseMatrix jacobian_matrix;

  // Pressure mass matrix, needed for preconditioning. We use a block matrix for
  // convenience, but in practice we only look at the pressure-pressure block.
  TrilinosWrappers::BlockSparseMatrix pressure_mass;

  // Residual vector.
  TrilinosWrappers::MPI::BlockVector residual_vector;

  // Solution increment (without ghost elements).
  TrilinosWrappers::MPI::BlockVector delta_owned;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::BlockVector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::BlockVector solution;

  // store the solution at the previous iteration
  TrilinosWrappers::MPI::BlockVector solution_old;

  // Evaluation point, used to find an optimal update in the Newton iteration
  TrilinosWrappers::MPI::BlockVector evaluation_point;

  // Lift and Drag forces  ///////////////////////////////////////////////////////////
public:
  void compute_lift_drag();

  double get_avg_inlet_velocity() const;
  void print_lift_coeff();
  void print_drag_coeff();
  void compute_lift_coeff();
  void compute_drag_coeff();
  void write_lift_drag_to_file() const;

protected:
  double lift_force = 0.0;
  double drag_force = 0.0;
  double lift_coeff = 0.0;
  double drag_coeff = 0.0;
};

#endif