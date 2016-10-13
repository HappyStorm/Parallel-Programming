#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>
using namespace std;


int main (int argc, char *argv[])
{
	int i, j, k;
	const int N = atoi(argv[1]);

	FILE *in = fopen("randnum.txt", "r");
	FILE *nout = fopen("out.txt", "w");

	FILE *bout = fopen("out", "wb");
	int *num;
	num = (int*) malloc(N*sizeof(int));

	for(i=0; i<N; ++i)
		fscanf(in, "%d",&num[i]);
	
	sort(num, num+N);

	for(i=0; i<N; ++i)
		fprintf(nout, "%d\n", num[i]);
	
	fwrite(num, sizeof(int), N, bout);

	fclose(in);
	fclose(nout);
	fclose(bout);
	return 0;
}