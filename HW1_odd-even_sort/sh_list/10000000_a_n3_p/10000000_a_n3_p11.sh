#PBS -q batch
#PBS -N judge_sh_out
#PBS -r n
#PBS -l nodes=3:ppn=11
#PBS -l walltime=00:30:00

cd $PBS_O_WORKDIR
time mpiexec ./HW1_101062319_advanced_test 10000000 randnum 10000000_a_n3_p11
