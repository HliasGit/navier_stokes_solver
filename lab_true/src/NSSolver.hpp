
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/tensor.h>
 
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
 
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
 
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
 
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
 
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
 
#include <deal.II/numerics/solution_transfer.h>
 
#include <deal.II/lac/sparse_direct.h>
 
#include <deal.II/lac/sparse_ilu.h>
 
 
#include <fstream>
#include <iostream>

using namespace dealii;

//Navier-Stokes problem class. Give us 6 points

class NSSolver{
    public:
        // Physical dimension
        static constexpr unsigned int dim = 2;

        NSSolver(
            const std::string &mesh_path_,
            const unsigned int &degree_velocity_,
            const unsigned int &degree_pressure_
            )        
        : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank == 0)
        , mesh_path(mesh_path_)
        , degree_velocity(degree_velocity_)
        , degree_pressure(degree_pressure_)
        , mesh(MPI_COMM_WORLD)
        {}

        void setup();

    protected:
    
    void setup_mesh();

    void setup_finite_element();

    void setup_dofs();

    void print_line();

    // Generic variables
    const std::string mesh_path;
    const unsigned int degree_velocity;
    const unsigned int degree_pressure;
    const unsigned int mpi_size;
    const unsigned int mpi_rank;

    // Parallel output stream.
    ConditionalOStream pcout;

    // Mesh
    parallel::fullydistributed::Triangulation<dim> mesh;

    // System matrices
    BlockSparseMatrix<double> system_matrix;
    SparseMatrix<double> pressure_mass_matrix;

    // Finite element space.
    std::unique_ptr<FiniteElement<dim>> fe;

    // DoF handler.
    DoFHandler<dim> dof_handler;

    // DoFs owned by current process.
    IndexSet locally_owned_dofs;

    // DoFs relevant to the current process (including ghost DoFs).
    IndexSet locally_relevant_dofs;

    // DoFs owned by current process in the velocity and pressure blocks.
    std::vector<IndexSet> block_owned_dofs;

    // DoFs relevant to current process in the velocity and pressure blocks.
    std::vector<IndexSet> block_relevant_dofs;

    // Quadrature formula.
    std::unique_ptr<Quadrature<dim>> quadrature;

    // Quadrature formula for face integrals.
    std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

};