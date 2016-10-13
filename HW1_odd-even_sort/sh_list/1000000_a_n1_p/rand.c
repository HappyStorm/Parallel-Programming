#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main (int argc, char *argv[])
{
	int i, j, k;


	const int N = atoi(argv[1]);
	FILE *bfile = fopen("randnum", "wb");
	FILE *nfile = fopen("randnum.txt", "w");

	int *arr;
	arr = (int*) malloc(N*sizeof(int));
	srand(time(NULL));
	for(i=0; i<N; ++i)
		arr[i] = rand()%2147483647;

	for(i=0; i<N; ++i)
		fprintf(nfile, "%d\n", arr[i]);

	fwrite(arr, sizeof(int), N, bfile);
	fclose(bfile);
	fclose(nfile);
	return 0;
}