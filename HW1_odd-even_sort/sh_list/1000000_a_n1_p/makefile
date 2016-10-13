all:
	mpicc HW1_101062319_advanced_test.c -o HW1_101062319_advanced_test

ss:
	qstat -a

clean:
	-rm judge_sh_out.*

6:
	gcc rand.c -o rand
	./rand 1000000
	g++ sort.cpp -o sort
	./sort 1000000

6a1:
	qsub 1000000_a_n1_p1.sh
	qsub 1000000_a_n1_p2.sh
	qsub 1000000_a_n1_p3.sh
	qsub 1000000_a_n1_p4.sh
	qsub 1000000_a_n1_p5.sh
	qsub 1000000_a_n1_p6.sh

6a2:
	qsub 1000000_a_n1_p7.sh
	qsub 1000000_a_n1_p8.sh
	qsub 1000000_a_n1_p9.sh
	qsub 1000000_a_n1_p10.sh
	qsub 1000000_a_n1_p11.sh
	qsub 1000000_a_n1_p12.sh