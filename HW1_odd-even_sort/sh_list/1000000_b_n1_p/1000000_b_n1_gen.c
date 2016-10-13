#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char const *argv[])
{
	int i, j;

	char nout[100];
	char tmp[2];

	// printf("Before for.\n");

	for(i=1; i<=12; ++i){
		memset(nout, '\0', sizeof(nout));
		memset(tmp, '\0', sizeof(tmp));

		// printf("Before concat i: %s\n", nout);
		strcat(nout, "1000000_b_n1_p");
		if(i<10){
			tmp[0] = i + '0';
			// printf("tmp:%s\n", tmp);
			strcat(nout, tmp);
			// printf("i<10:%s\n", nout);
		}
		else{
			strcat(nout, "1");
			tmp[0] = (i-10) + '0';
			strcat(nout, tmp);
		}
		strcat(nout, ".sh");

		// printf("After concat .sh: %s\n", nout);

		FILE *out = fopen(nout, "w");
		fprintf(out, "#PBS -q batch\n");
		fprintf(out, "#PBS -N judge_sh_out\n");
		fprintf(out, "#PBS -r n\n");
		fprintf(out, "#PBS -l nodes=3:ppn=%d\n", i);
		fprintf(out, "#PBS -l walltime=00:30:00\n\n");
		fprintf(out, "cd $PBS_O_WORKDIR\n");
		fprintf(out, "time mpiexec ./HW1_101062319_basic_test 1000000 randnum 1000000_b_n1_p%d\n", i);
		fclose(out);
	}
	return 0;
}