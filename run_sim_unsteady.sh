#!/bin/sh -l
#SBATCH --ntasks-per-node 128
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH --export=ALL
#SBATCH --mem=64GB
#SBATCH -J NSSimulation
#SBATCH -o results_sim_unsteady/sim%j.out

# Set variables for MPI processes and mesh dimensions
MPI_PROCS=128
MESH_DIMS="150,100"

# Performance log file
PERF_LOG="/home/users/gdaneri/navier_stokes_solver/performance_log.csv"

# Load modules
module load tools/Singularity
singularity -s exec /home/users/gdaneri/mk_latest.sif /bin/bash -c '
    source /u/sw/etc/profile && 
    module load gcc-glibc dealii && 
    
    # Capture start time
    start_time=$(date +%s.%N)
    
    # Run simulation
    mpiexec -n '"$MPI_PROCS"' /home/users/gdaneri/navier_stokes_solver/lab_new/build/NSSolver -m '"$MESH_DIMS"' -t 0.000000000001 -T 0.1,0.1 -p 1 -s 0
    
    # Capture end time and calculate duration
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    
    # Write to CSV (create if not exists)
    if [ ! -f '"$PERF_LOG"' ]; then
        echo "time,proc,dim_x,dim_y" > '"$PERF_LOG"'
    fi
    
    # Extract dimensions
    dim_x=$(echo '"$MESH_DIMS"' | cut -d, -f1)
    dim_y=$(echo '"$MESH_DIMS"' | cut -d, -f2)
    
    # Append performance data to CSV
    echo "$duration,$MPI_PROCS,$dim_x,$dim_y" >> '"$PERF_LOG"'
'