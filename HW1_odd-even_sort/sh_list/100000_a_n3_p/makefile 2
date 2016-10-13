all:
	mpicc HW1_101062319_advanced_test_io.c -o HW1_101062319_advanced_test_io

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
	qsub 1000000_a_n3_p1.sh
	qsub 1000000_a_n3_p2.sh
	qsub 1000000_a_n3_p3.sh
	qsub 1000000_a_n3_p4.sh
	qsub 1000000_a_n3_p5.sh
	qsub 1000000_a_n3_p6.sh

6a2:
	qsub 1000000_a_n3_p7.sh
	qsub 1000000_a_n3_p8.sh
	qsub 1000000_a_n3_p9.sh
	qsub 1000000_a_n3_p10.sh
	qsub 1000000_a_n3_p11.sh
	qsub 1000000_a_n3_p12.sh

io6a1:
	qsub 1000000_a_n3_p1_io.sh
	qsub 1000000_a_n3_p2_io.sh
	qsub 1000000_a_n3_p3_io.sh
	qsub 1000000_a_n3_p4_io.sh
	qsub 1000000_a_n3_p5_io.sh
	qsub 1000000_a_n3_p6_io.sh

io6a2:
	qsub 1000000_a_n3_p7_io.sh
	qsub 1000000_a_n3_p8_io.sh
	qsub 1000000_a_n3_p9_io.sh
	qsub 1000000_a_n3_p10_io.sh
	qsub 1000000_a_n3_p11_io.sh
	qsub 1000000_a_n3_p12_io.sh