# Navier Stoke Solver
### Giorgio Daneri, Jacopo Palumbo, Elia Vaglietti

## Overview

This project is a Navier-Stokes solver designed to simulate fluid dynamics. The solver is implemented using the ```deal.II``` library, which provides extensive tools for finite element analysis. The project includes both stationary and time-dependent solvers for the Navier-Stokes equations.

## Features

- **Stationary Solver**: Solves the steady-state Navier-Stokes equations.
- **Time-Dependent Solver**: Solves the transient Navier-Stokes equations.
- **Mesh Generation**: Supports both internal mesh generation and reading meshes from files.
- **Preconditioners**: Includes various preconditioners like block diagonal, block triangular, and aSIMPLE.
- **Solvers**: Supports multiple solvers including GMRES, FGMRES, and BiCGStab.

## Dependencies

- **deal.II**: A finite element library.
- **MPI**: For parallel computations.
- **CMake**: For building the project.

## Building the Project

1. **Clone the repository**:
    ```sh
    git clone https://github.com/HliasGit/navier_stokes_solver.git
    cd navier_stokes_solver
    ```

2. **Create a build directory**:
    ```sh
    module load dealii
    mkdir build
    cd build
    ```

3. **Run CMake**:
    ```sh
    cmake ..
    ```

4. **Build the project**:
    ```sh
    make
    ```

## Running the Solver

### Stationary Solver

To run the stationary solver, use the following command:
```sh
mpirun -n <number_of_processes> ./StationaryNSSolver [options]
```

### Options

- `-M, --read-mesh-from-file`: Read mesh from file instead of generating it inside the program.
- `-m, --mesh-size X,Y`: Set mesh size (two integers separated by a comma).
- `-r, --reynolds N`: Set Reynolds number (floating point value).
- `-s, --solver N`: Select solver (0: GMRES, 1: FGMRES, 2: BiCGStab).
- `-t, --tolerance D`: Set tolerance (floating point value).
- `-p, --preconditioner N`: Select preconditioner (0: blockDiagonal, 1: blockTriangular, 2: aSIMPLE).
- `-h, --help`: Display help message.

Only for the unsteady version:
- `-T, --time-span and time-step T,D`: Set time span and time step (two floating point values separated by a comma).

## Example

```sh
mpirun -n 4 ./StationaryNSSolver -m 300,100 -r 100 -s 1 -t 0.0000000001 -p 0
```

This command runs the stationary solver with a mesh size of 300x100, Reynolds number of 100, using the FGMRES solver, a tolerance of 1e-10, and the blockDiagonal preconditioner.