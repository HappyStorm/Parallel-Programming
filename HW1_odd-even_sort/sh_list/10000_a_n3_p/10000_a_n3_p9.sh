#PBS -q batch
#PBS -N judge_sh_out
#PBS -r n
#PBS -l nodes=3:ppn=9
#PBS -l walltime=00:01:00

cd $PBS_O_WORKDIR
time mpiexec ./HW1_101062319_advanced_test 10000 randnum 10000_a_n3_p9
