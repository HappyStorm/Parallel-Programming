all:
	mpicc HW1_101062319_advanced_test.c -o HW1_101062319_advanced_test

ss:
	qstat -a

clean:
	-rm judge_sh_out.*

8:
	gcc rand.c -o rand
	./rand 600000000
	g++ sort.cpp -o sort
	./sort 600000000

8a1:
	qsub 600000000_a_n3_p1.sh
	qsub 600000000_a_n3_p2.sh
	qsub 600000000_a_n3_p3.sh
	qsub 600000000_a_n3_p4.sh
	qsub 600000000_a_n3_p5.sh
	qsub 600000000_a_n3_p6.sh

8a2:
	qsub 600000000_a_n3_p7.sh
	qsub 600000000_a_n3_p8.sh
	qsub 600000000_a_n3_p9.sh
	qsub 600000000_a_n3_p10.sh
	qsub 600000000_a_n3_p11.sh
	qsub 600000000_a_n3_p12.sh