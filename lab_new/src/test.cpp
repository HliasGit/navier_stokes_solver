#include "NSSolver.hpp"
#include <getopt.h>
#include <iostream>
#include <cstdlib>

// Function to print help message
void print_help() {
    std::cout << "Usage: ./NSSolver [options]\n\n"
              << "Options:\n"
              << "  -T, --time-span and time-step T,D\n"
              << "  -M, --read-mesh-from-file  Read mesh from file instead or generate it inside the program\n"
              << "  -m, --mesh-size X,Y       Set mesh size (two integers separated by a comma)\n"
              << "  -r, --reynolds N         Set Reynolds number (floating point value)\n"
              << "  -s, --solver N            Select solver (valid values: 0: GMRES, 1: FGMRES, 2: Bicgstab)\n"
              << "  -t, --tolerance D         Set tolerance (floating point value)\n"
              << "  -p, --preconditioner N    Select preconditioner (valid values: 0: blockDiagonal, 1: blockTriangular, 2: aSIMPLE)\n"
              << "  -h, --help                Display this help message\n";
}

// Main function.
int main(int argc, char *argv[]) {
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    // Default parameters
    bool read_mesh_from_file = false;
    unsigned int degree_velocity = 3;
    unsigned int degree_pressure = 2;
    double Re = 100.0;
    int mesh_size_x = 100, mesh_size_y = 100;
    int solver_type = 1;
    double tolerance = 1e-6;
    int preconditioner = 0;
    double time_span = 1.0;
    double time_step = 0.01;

    // Define long options
    static struct option long_options[] = {
        {"timespan-step", required_argument, 0, 'T'}, 
        {"read-mesh-from-file", no_argument, 0, 'M'},
        {"mesh-size", required_argument, 0, 'm'},
        {"reynolds", required_argument, 0, 'r'},
        {"solver", required_argument, 0, 's'},
        {"tolerance", required_argument, 0, 't'},
        {"preconditioner", required_argument, 0, 'p'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    // Modified getopt_long string to match the new format
    while ((opt = getopt_long(argc, argv, "T:M:m:r:s:t:p:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'T': { 
                char* comma = strchr(optarg, ',');
                if (comma) {
                    *comma = '\0'; 
                    time_span = std::atof(optarg);
                    time_step = std::atof(comma + 1);
                } else {
                    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                        std::cerr << "Error: timespan-step requires two values separated by comma\n";
                    return 1;
                }
                break;
            }
            case 'M':
                read_mesh_from_file = true;
                degree_velocity = 2;
                degree_pressure = 1;
                break;
            case 'm': {
                char* comma = strchr(optarg, ',');
                if (comma) {
                    *comma = '\0';
                    mesh_size_x = std::atoi(optarg);
                    mesh_size_y = std::atoi(comma + 1);
                } else {
                    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                        std::cerr << "Error: mesh-size requires two values separated by comma\n";
                    return 1;
                }
                break;
            }
            case 'r':
                Re = std::atof(optarg);
                break;
            case 's':
                solver_type = std::atoi(optarg);
                break;
            case 't':
                tolerance = std::atof(optarg);
                break;
            case 'p':
                preconditioner = std::atoi(optarg);
                break;
            case 'h':
                if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                    print_help();
                return 0;
            default:
                if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                    print_help();
                return 1;
        }
    }

    // Add validation for parameters
    if (time_step <= 0 || time_span <= 0 || tolerance <= 0) {
        if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cerr << "Error: time_step, time_span, and tolerance must be positive\n";
        return 1;
    }

    // Print the parsed values
    // only the first MPI rank prints the values
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
        std::cout << "--------- CONFIGURATION PARAMETERS --------- \n";
        std::cout << "Time span: " << time_span << "\n";
        std::cout << "Time step: " << time_step << "\n";    
        std::cout << "Mesh size: " << mesh_size_x << "x" << mesh_size_y << "\n";
        std::cout << "Reynolds number: " << Re << "\n";
        std::cout << "Solver type: ";
        if (solver_type == 0) {
            std::cout << "GMRES\n";
        }
        else if (solver_type == 1) {
            std::cout << "FGMRES\n";
        }
        else if (solver_type == 2) {
            std::cout << "Bicgstab\n";
        }
        std::cout << "Tolerance: " << tolerance << "\n";
        std::cout << "Preconditioner: ";
        if(preconditioner == 0) {
            std::cout << "blockDiagonal\n";
        }
        else if(preconditioner == 1) {
            std::cout << "blockTriangular\n";
        }
        else if(preconditioner == 2) {
            std::cout << "aSIMPLE\n";
        }
        std::cout << "-----------------------------------------------\n";
    }
    
    const std::string mesh_file_name = "/home/users/gdaneri/navier_stokes_solver/lab_new/mesh/new_mesh.msh";
    
    NSSolver problem(mesh_file_name, degree_velocity, degree_pressure, time_span, time_step, mesh_size_x, mesh_size_y, solver_type, tolerance, preconditioner, Re, read_mesh_from_file);

    problem.setup();
    problem.solve();

    return 0;
}
