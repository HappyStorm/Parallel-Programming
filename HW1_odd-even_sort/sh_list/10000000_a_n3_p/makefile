all:
	mpicc HW1_101062319_advanced_test.c -o HW1_101062319_advanced_test

ss:
	qstat -a

clean:
	-rm judge_sh_out.*

7:
	gcc rand.c -o rand
	./rand 10000000
	g++ sort.cpp -o sort
	./sort 10000000

7a1:
	qsub 10000000_a_n3_p1.sh
	qsub 10000000_a_n3_p2.sh
	qsub 10000000_a_n3_p3.sh
	qsub 10000000_a_n3_p4.sh
	qsub 10000000_a_n3_p5.sh
	qsub 10000000_a_n3_p6.sh

7a2:
	qsub 10000000_a_n3_p7.sh
	qsub 10000000_a_n3_p8.sh
	qsub 10000000_a_n3_p9.sh
	qsub 10000000_a_n3_p10.sh
	qsub 10000000_a_n3_p11.sh
	qsub 10000000_a_n3_p12.sh