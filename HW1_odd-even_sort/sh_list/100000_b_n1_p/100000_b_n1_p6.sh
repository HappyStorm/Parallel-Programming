#PBS -q batch
#PBS -N judge_sh_out
#PBS -r n
#PBS -l nodes=1:ppn=6
#PBS -l walltime=00:01:00

cd $PBS_O_WORKDIR
time mpiexec ./HW1_101062319_basic_test 100000 randnum 100000_b_n1_p6
