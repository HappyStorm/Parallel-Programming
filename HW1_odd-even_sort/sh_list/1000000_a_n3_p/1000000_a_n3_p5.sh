#PBS -q batch
#PBS -N judge_sh_out
#PBS -r n
#PBS -l nodes=3:ppn=5
#PBS -l walltime=00:30:00

cd $PBS_O_WORKDIR
time mpiexec ./HW1_101062319_advanced_test 1000000 randnum 1000000_a_n3_p5
