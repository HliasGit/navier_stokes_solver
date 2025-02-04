#ifndef NSSOLVERSTATIONARY_HPP
#define NSSOLVERSTATIONARY_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

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
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>


using namespace dealii;

// Class representing the non-linear diffusion problem.
class NSSolverStationary
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
  private: 
    double u = 0.1;
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
      values[0] = 4 * u * p[1] * (H - p[1]) / (H * H);

      for (unsigned int i = 1; i < dim + 1; ++i)
        values[i] = 0.0;
    }

    virtual double
    value(const Point<dim> &p,
          const unsigned int component = 0) const override
    {
      if (component == 0)
        return 4 * u * p[1] * (H - p[1]) / (H * H);
      else
        return 0.0;
    }

    double getVelocity() {
      return u;
    }

    bool incrementVelocity(double re) {
      if (u == U_m)
        return true;

      //u += 1.0 / re;
      u += 0.1;

      if (re == 0.0)
        u = 0.01;

      if (u > U_m)
        u = U_m;

      return false;
    }
    const double U_m = 1.0;
    const double H = 0.41;
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
      SolverControl solver_control_velocity(100001,
                                            1e-1 * src.block(0).l2_norm());
      SolverFGMRES<TrilinosWrappers::MPI::Vector> solver_cg_velocity(
          solver_control_velocity);
      solver_cg_velocity.solve(*velocity_stiffness,
                               dst.block(0),
                               src.block(0),
                               preconditioner_velocity);

      SolverControl solver_control_pressure(100000,
                                            1e-1 * src.block(1).l2_norm());
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
    TrilinosWrappers::PreconditionSSOR preconditioner_velocity;

    // Pressure mass matrix.
    const TrilinosWrappers::SparseMatrix *pressure_mass;

    // Preconditioner used for the pressure block.
    TrilinosWrappers::PreconditionSSOR preconditioner_pressure;
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
      SolverControl solver_control_velocity(10000001,
                                            1e-2 * src.block(0).l2_norm());

      SolverFGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_velocity(
          solver_control_velocity);
      solver_gmres_velocity.solve(*velocity_stiffness,
                                  dst.block(0),
                                  src.block(0),
                                  preconditioner_velocity);

      // UNCOMMENT if do not want to use direct solver
      // preconditioner_velocity.vmult(dst.block(0), tmp.block(0));

      tmp.reinit(src.block(1));
      B->vmult(tmp, dst.block(0));
      tmp.sadd(-1.0, src.block(1));

      SolverControl solver_control_pressure(100000,
                                            1e-2 * src.block(1).l2_norm());
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
    TrilinosWrappers::PreconditionAMG preconditioner_velocity;

    // Pressure mass matrix.
    const TrilinosWrappers::SparseMatrix *pressure_mass;

    // Preconditioner used for the pressure block.
    TrilinosWrappers::PreconditionILU preconditioner_pressure;

    // B matrix.
    const TrilinosWrappers::SparseMatrix *B;

    // Temporary vector.
    mutable TrilinosWrappers::MPI::Vector tmp;
  };

  class PreconditionaSIMPLE {
  public:
    void initialize(const TrilinosWrappers::SparseMatrix &F_,
                const TrilinosWrappers::SparseMatrix &B_,
                const TrilinosWrappers::SparseMatrix &B_t_,
                const TrilinosWrappers::MPI::BlockVector &vector_,
                const double &alpha_)
    {
        F_matrix = &F_;
        B_matrix = &B_;
        B_t_matrix = &B_t_;
        alpha = alpha_;

        // Ensure proper initialization of temporary vectors
        tmp_p.reinit(vector_.block(1)); // Use correct block for pressure
        delta_p.reinit(vector_.block(1));
        tmp_u.reinit(vector_.block(0)); // Use correct block for velocity

        // Extract diagonal of F_matrix (D) and its inverse (D^{-1})
        D_vector.reinit(vector_.block(0));
        D_inv_vector.reinit(vector_.block(0));
        for (unsigned int i : D_vector.locally_owned_elements()) {
            D_vector[i] = F_matrix->diag_element(i);
            D_inv_vector[i] = 1.0 / D_vector[i];
        }

        // Compute Schur complement approximation: S = B * D^{-1} * B^T
        S_matrix.clear();
        
        // Create S_matrix with appropriate sparsity pattern from B_t_
        dealii::TrilinosWrappers::SparsityPattern sp(B_t_.m(), B_t_.m(), B_t_.n());
        sp.compress();
        S_matrix.reinit(sp);

        // Compute Schur complement
        B_matrix->mmult(S_matrix, B_t_, D_inv_vector);

        // Initialize preconditioners with ILU for robustness
        preconditioner_F.initialize(*F_matrix);
        preconditioner_S.initialize(S_matrix);
    }

    void vmult(TrilinosWrappers::MPI::BlockVector &dst,
              const TrilinosWrappers::MPI::BlockVector &src) const
    {
      // Step 1: Solve F * ũ = src_u (velocity predictor)
      SolverControl solver_control_F(100000, 1e-1 * src.block(0).l2_norm());
      SolverFGMRES<TrilinosWrappers::MPI::Vector> solver_F(solver_control_F);
      solver_F.solve(*F_matrix, dst.block(0), src.block(0), preconditioner_F);

      // Step 2: Compute pressure residual: tmp_p = src_p - B * ũ
      tmp_p.reinit(src.block(1));
      B_matrix->vmult(tmp_p, dst.block(0));  // B * ũ
      tmp_p.sadd(-1.0, 1.0, src.block(1));   // tmp_p = src_p - B * ũ

      // Step 3: Solve S * δp = tmp_p (pressure correction)
      SolverControl solver_control_S(100000, 1e-1 * tmp_p.l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);
      solver_S.solve(S_matrix, delta_p, tmp_p, preconditioner_S);

      // Step 4: Apply under-relaxation: p = α * δp
      delta_p *= alpha;

      // Step 5: Correct velocity: u = ũ - D^{-1} * B^T * δp
      tmp_u.reinit(dst.block(0));
      B_t_matrix->vmult(tmp_u, delta_p);     // B^T * δp
      tmp_u.scale(D_inv_vector);             // D^{-1} * B^T * δp
      dst.block(0) -= tmp_u;                 // u = ũ - D^{-1} B^T δp

      // Store pressure correction
      dst.block(1) = delta_p;
    }

  protected:
    // Matrices
    const TrilinosWrappers::SparseMatrix *F_matrix;
    const TrilinosWrappers::SparseMatrix *B_matrix;      // Divergence (B)
    const TrilinosWrappers::SparseMatrix *B_t_matrix;    // Gradient (B^T)
    TrilinosWrappers::SparseMatrix S_matrix;             // Schur complement S = B D^{-1} B^T

    // Diagonal scaling vectors
    TrilinosWrappers::MPI::Vector D_vector;     // diag(F)
    TrilinosWrappers::MPI::Vector D_inv_vector; // diag(F)^{-1}

    // Preconditioners
    TrilinosWrappers::PreconditionILU preconditioner_F;  // For F-block
    TrilinosWrappers::PreconditionILU preconditioner_S;  // For S-block

    // Damping factor (α ∈ (0,1])
    double alpha;

    // Temporary vectors
    mutable TrilinosWrappers::MPI::Vector tmp_p;
    mutable TrilinosWrappers::MPI::Vector delta_p;
    mutable TrilinosWrappers::MPI::Vector tmp_u;
  };


  // Constructor.
  NSSolverStationary(const std::string &mesh_file_name_,
                     const unsigned int &degree_velocity_,
                     const unsigned int &degree_pressure_,
                     const unsigned int &mesh_size_x_,
                     const unsigned int &mesh_size_y_,
                     const unsigned int &solver_type_,
                     const double &tolerance_,
                     const unsigned int &preconditioner_type_)
      : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)), mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)), pcout(std::cout, mpi_rank == 0), mesh_file_name(mesh_file_name_), mesh(MPI_COMM_WORLD), degree_velocity(degree_velocity_), degree_pressure(degree_pressure_), solver_type(solver_type_), tolerance(tolerance_), preconditioner_type(preconditioner_type_), mesh_size_x(mesh_size_x_), mesh_size_y(mesh_size_y_)
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
  output() const;

  bool 
  decreaseViscosity();

protected:
  // Assemble the tangent problem.
  void
  assemble_system(bool first_iter);

  // Solve the tangent problem.
  int
  solve_system();

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
  double nu = 0.001;

  // Inlet velocity
  InletVelocity inlet_velocity;

  // Pressure out [Pa]
  const double p_out = 1.0;

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh file name.
  const std::string &mesh_file_name;
  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Polynomial degrees.
  const unsigned int degree_velocity;
  const unsigned int degree_pressure;
  const unsigned int solver_type;
  const double tolerance;
  const unsigned int preconditioner_type;
  const unsigned int mesh_size_x;
  const unsigned int mesh_size_y;

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

protected:
  double lift_force = 0.0;
  double drag_force = 0.0;
  double lift_coeff = 0.0;
  double drag_coeff = 0.0;
};

#endif