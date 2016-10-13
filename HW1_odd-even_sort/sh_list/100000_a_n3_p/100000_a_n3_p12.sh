#PBS -q batch
#PBS -N judge_sh_out
#PBS -r n
#PBS -l nodes=3:ppn=12
#PBS -l walltime=00:01:00

cd $PBS_O_WORKDIR
time mpiexec ./HW1_101062319_advanced_test 100000 randnum 100000_a_n3_p12
