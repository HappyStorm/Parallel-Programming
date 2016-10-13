all:
	mpicc HW1_101062319_basic_test.c -o HW1_101062319_basic_test

ss:
	qstat -a

clean:
	-rm judge_sh_out.*

7:
	gcc rand.c -o rand
	./rand 10000000
	g++ sort.cpp -o sort
	./sort 10000000

7b1:
	qsub 10000000_b_n1_p1.sh
	qsub 10000000_b_n1_p2.sh
	qsub 10000000_b_n1_p3.sh
	qsub 10000000_b_n1_p4.sh
	qsub 10000000_b_n1_p5.sh
	qsub 10000000_b_n1_p6.sh

7b2:
	qsub 10000000_b_n1_p7.sh
	qsub 10000000_b_n1_p8.sh
	qsub 10000000_b_n1_p9.sh
	qsub 10000000_b_n1_p10.sh
	qsub 10000000_b_n1_p11.sh
	qsub 10000000_b_n1_p12.sh
