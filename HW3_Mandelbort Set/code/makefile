all:
	make ms
	make md
	make os
	make od
	make hs
	make hd

seq:
	gcc -lX11 -O3 seq_MS.c -o seq_MS

ms:
	mpic++ -std=c++11 -lX11 -O3  MS_MPI_static.cpp -o MS_MPI_static

md:
	mpic++ -std=c++11 -lX11 -O3  MS_MPI_dynamic.cpp -o MS_MPI_dynamic

os:
	g++ -std=c++11 -fopenmp -lX11 -O3 MS_OpenMP_static.cpp -o MS_OpenMP_static

od:
	g++ -std=c++11 -fopenmp -lX11 -O3 MS_OpenMP_dynamic.cpp -o MS_OpenMP_dynamic

hs:
	mpic++ -std=c++11 -fopenmp -lX11 -O3 MS_Hybrid_static.cpp -o MS_Hybrid_static

hd:
	mpic++ -std=c++11 -fopenmp -lX11 -O3 MS_Hybrid_dynamic.cpp -o MS_Hybrid_dynamic

clean:
	rm -f MS_MPI_static MS_MPI_dynamic MS_OpenMP_static MS_OpenMP_dynamic MS_Hybrid_static MS_Hybrid_dynamic

bt:
	mpirun ./MS_MPI_dynamic 4 -2 2 -2 2 400 400 enable
	mpirun ./MS_MPI_dynamic 4 0 2 -2 2 800 400 enable
	mpirun ./MS_MPI_static 4 -2 2 -2 2 400 400 enable
	mpirun ./MS_MPI_static 4 0 2 -2 2 800 400 enable
	./MS_OpenMP_dynamic 4 -2 2 -2 2 400 400 enable
	./MS_OpenMP_dynamic 4 0 2 -2 2 800 400 enable
	./MS_OpenMP_static 4 -2 2 -2 2 400 400 enable
	./MS_OpenMP_static 4 0 2 -2 2 800 400 enable
	mpirun ./MS_Hybrid_dynamic 4 -2 2 -2 2 400 400 enable
	mpirun ./MS_Hybrid_dynamic 4 0 2 -2 2 800 400 enable
	mpirun ./MS_Hybrid_static 4 -2 2 -2 2 400 400 enable
	mpirun ./MS_Hybrid_static 4 0 2 -2 2 800 400 enable

at:
	mpirun -n 3 ./MS_MPI_dynamic 4 -2 2 -2 2 400 400 enable
	mpirun -n 3 ./MS_MPI_static 4 -2 2 -2 2 400 400 enable
	mpirun -n 3 ./MS_OpenMP_dynamic 4 -2 2 -2 2 400 400 enable
	mpirun -n 3 ./MS_OpenMP_static 4 -2 2 -2 2 400 400 enable
	mpirun -n 3 ./MS_Hybrid_dynamic 4 -2 2 -2 2 400 400 enable
	mpirun -n 3 ./MS_Hybrid_static 4 -2 2 -2 2 400 400 enable
	mpirun -n 3 ./MS_MPI_dynamic 4 0 2 -2 2 800 400 enable
	mpirun -n 3 ./MS_MPI_static 4 0 2 -2 2 800 400 enable
	mpirun -n 3 ./MS_OpenMP_dynamic 4 0 2 -2 2 800 400 enable
	mpirun -n 3 ./MS_OpenMP_static 4 0 2 -2 2 800 400 enable
	mpirun -n 3 ./MS_Hybrid_dynamic 4 0 2 -2 2 800 400 enable
	mpirun -n 3 ./MS_Hybrid_static 4 0 2 -2 2 800 400 enable

#	Strong scalability – scalability to number of cores (Problem size is fixed)
ss:
	mpirun ./MS_MPI_dynamic 1 -2 2 -2 2 400 400 disable
	mpirun ./MS_MPI_static 1 -2 2 -2 2 400 400 disable
	./MS_OpenMP_dynamic 4 -2 2 -2 2 400 400 enable

mpiclean:
	rm -f MPI.*

ompclean:
	rm -f OMP.*

hybclean:
	rm -f HYBRID.*

mpit1_odd:
	qsub mpi_n1_p1.sh
	qsub mpi_n1_p3.sh
	qsub mpi_n1_p5.sh
	qsub mpi_n1_p7.sh
	qsub mpi_n1_p9.sh
	qsub mpi_n1_p11.sh

mpit1_even:
	qsub mpi_n2_p2.sh
	qsub mpi_n2_p4.sh
	qsub mpi_n2_p6.sh
	qsub mpi_n2_p8.sh
	qsub mpi_n2_p10.sh
	qsub mpi_n2_p12.sh
	qsub mpi_n2_p14.sh
	qsub mpi_n2_p16.sh

ompt1_pre:
	qsub omp_n1_p1.sh
	qsub omp_n1_p2.sh
	qsub omp_n1_p3.sh
	qsub omp_n1_p4.sh
	qsub omp_n1_p5.sh
	qsub omp_n1_p6.sh

ompt1_las:
	qsub omp_n1_p7.sh
	qsub omp_n1_p8.sh
	qsub omp_n1_p9.sh
	qsub omp_n1_p10.sh
	qsub omp_n1_p11.sh
	qsub omp_n1_p12.sh

mpit2:
	qsub mpi_20.sh
	qsub mpi_40.sh
	qsub mpi_80.sh
	qsub mpi_160.sh
	qsub mpi_320.sh
	qsub mpi_640.sh
	qsub mpi_1280.sh
	qsub mpi_2560.sh

ompt2:
	qsub omp_20.sh
	qsub omp_40.sh
	qsub omp_80.sh
	qsub omp_160.sh
	qsub omp_320.sh
	qsub omp_640.sh
	qsub omp_1280.sh
	qsub omp_2560.sh

hybt2:
	qsub hyb_20.sh
	qsub hyb_40.sh
	qsub hyb_80.sh
	qsub hyb_160.sh
	qsub hyb_320.sh
	qsub hyb_640.sh
	qsub hyb_1280.sh
	qsub hyb_2560.sh

load:
	qsub mpi_load.sh
	qsub omp_load.sh
	qsub hyb_load.sh
	

