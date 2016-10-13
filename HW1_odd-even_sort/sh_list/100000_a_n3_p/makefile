all:
	mpicc HW1_101062319_advanced_test.c -o HW1_101062319_advanced_test
	mpicc HW1_101062319_advanced_test_io.c -o HW1_101062319_advanced_test_io

ss:
	qstat -a

clean:
	-rm judge_sh_out.*

5:
	gcc rand.c -o rand
	./rand 100000
	g++ sort.cpp -o sort
	./sort 100000

5a1:
	qsub 100000_a_n3_p1.sh
	qsub 100000_a_n3_p2.sh
	qsub 100000_a_n3_p3.sh
	qsub 100000_a_n3_p4.sh
	qsub 100000_a_n3_p5.sh
	qsub 100000_a_n3_p6.sh

5a2:
	qsub 100000_a_n3_p7.sh
	qsub 100000_a_n3_p8.sh
	qsub 100000_a_n3_p9.sh
	qsub 100000_a_n3_p10.sh
	qsub 100000_a_n3_p11.sh
	qsub 100000_a_n3_p12.sh

io5a1:
	qsub 100000_a_n3_p1_io.sh
	qsub 100000_a_n3_p2_io.sh
	qsub 100000_a_n3_p3_io.sh
	qsub 100000_a_n3_p4_io.sh
	qsub 100000_a_n3_p5_io.sh
	qsub 100000_a_n3_p6_io.sh

io5a2:
	qsub 100000_a_n3_p7_io.sh
	qsub 100000_a_n3_p8_io.sh
	qsub 100000_a_n3_p9_io.sh
	qsub 100000_a_n3_p10_io.sh
	qsub 100000_a_n3_p11_io.sh
	qsub 100000_a_n3_p12_io.sh