#!/bin/sh -l
#SBATCH --ntasks-per-node 128
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH --export=ALL
#SBATCH --mem=64GB
#SBATCH -J NSSimulation
#SBATCH -o ../results_sim_steady/128_%j.out
#SBATCH -e ../results_sim_steady/128_%j.err

# Export variables so they're accessible inside the singularity container
export MPI_PROCS=128
export MESH_DIMS="100,70"
export SOLVER=1
export PRECONDITIONER=1
export PERF_LOG="/home/users/gdaneri/navier_stokes_solver/weak_scalability_log.csv"

module load tools/Singularity
singularity -s exec /home/users/gdaneri/mk_latest.sif /bin/bash -c '
   source /u/sw/etc/profile && 
   module load gcc-glibc dealii && 
   
   start_time=$(date +%s.%N)
   
   mpiexec -n $MPI_PROCS /home/users/gdaneri/navier_stokes_solver/lab_new/build/StationaryNSSolver -M -m $MESH_DIMS -r 10.0 -t 0.0000000001 -p $PRECONDITIONER -s $SOLVER
   
   end_time=$(date +%s.%N)
   duration=$(awk "BEGIN {print $end_time - $start_time}")
   
   if [ ! -f $PERF_LOG ]; then
       echo "time,proc,dim_x,dim_y,solver,prec" > $PERF_LOG
   fi
   
   dim_x=$(echo $MESH_DIMS | cut -d, -f1)
   dim_y=$(echo $MESH_DIMS | cut -d, -f2)
   
   echo "$duration,$MPI_PROCS,$dim_x,$dim_y,$SOLVER,$PRECONDITIONER" >> $PERF_LOG
'