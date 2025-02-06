#!/bin/sh -l
#SBATCH --ntasks-per-node 128
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH --export=ALL
#SBATCH --mem=32GB
#SBATCH -J NSSimulation
#SBATCH -e /home/users/gdaneri/navier_stokes_solver/results_sim_unsteady/sim_unsteady%j.err
#SBATCH -o /home/users/gdaneri/navier_stokes_solver/results_sim_unsteady/sim_unsteady%j.out

# Export variables so they're accessible inside the singularity container
export MPI_PROCS=128
export MESH_DIMS="60,40"
export PERF_LOG="/home/users/gdaneri/navier_stokes_solver/performance_log.csv"

module load tools/Singularity
singularity -s exec /home/users/gdaneri/mk_latest.sif /bin/bash -c '
   source /u/sw/etc/profile && 
   module load gcc-glibc dealii && 
   
   start_time=$(date +%s.%N)
   
   mpiexec -n $MPI_PROCS /home/users/gdaneri/navier_stokes_solver/lab_new/build/NSSolver -M -T 0.03,0.01 -t 0.000000001 -m 60,40 -s 1 -r 1.0  -p 1
   end_time=$(date +%s.%N)
   duration=$(awk "BEGIN {print $end_time - $start_time}")
   
   if [ ! -f $PERF_LOG ]; then
       echo "time,proc,dim_x,dim_y" > $PERF_LOG
   fi
   
   dim_x=$(echo $MESH_DIMS | cut -d, -f1)
   dim_y=$(echo $MESH_DIMS | cut -d, -f2)
   
   echo "$duration,$MPI_PROCS,$dim_x,$dim_y" >> $PERF_LOG
'