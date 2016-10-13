all:
	mpicc HW1_101062319_basic_test.c -o HW1_101062319_basic_test

ss:
	qstat -a

clean:
	-rm judge_sh_out.*

6:
	gcc rand.c -o rand
	./rand 1000000
	g++ sort.cpp -o sort
	./sort 1000000

6b1:
	qsub 1000000_b_n3_p1.sh
	qsub 1000000_b_n3_p2.sh
	qsub 1000000_b_n3_p3.sh
	qsub 1000000_b_n3_p4.sh
	qsub 1000000_b_n3_p5.sh
	qsub 1000000_b_n3_p6.sh

6b2:
	qsub 1000000_b_n3_p7.sh
	qsub 1000000_b_n3_p8.sh
	qsub 1000000_b_n3_p9.sh
	qsub 1000000_b_n3_p10.sh
	qsub 1000000_b_n3_p11.sh
	qsub 1000000_b_n3_p12.sh
