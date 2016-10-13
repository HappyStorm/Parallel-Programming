#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
using namespace std;

int main(int argc, char const *argv[])
{
	int rank[12];
	double all[12], read[12], write[12], io[12], comm[12], compute[12];
	double max_all, max_read, max_write, max_io, max_comm, max_compute;
	double avg_all, avg_read, avg_write, avg_io, avg_comm, avg_compute;
	
	
	FILE *out = fopen("Result.txt", "w");
	for(int i=0; i<12; ++i){
		rank[i] = 0, all[i] = 0, read[i] = 0;
		write[i] = 0, io[i] = 0, comm[i] = 0;
		max_all=0, max_read=0, max_write=0, max_io=0, max_comm=0, max_compute=0;
		avg_all=0, avg_read=0, avg_write=0, avg_io=0, avg_comm=0, avg_compute=0;
		compute[i] = 0;
		const char *path = argv[i+1];
		FILE *in = fopen(path, "r");
		int ct = 0;
		while(fscanf(in, "Rank:%d (All:%lf, I:%lf, W:%lf, I/O:%lf, Comm:%lf, Compute:%lf)\n", 
			&rank[ct], &all[ct], &read[ct], &write[ct], &io[ct], &comm[ct], &compute[ct])!=EOF){
			avg_all += all[ct];
			avg_read += read[ct];
			avg_write += write[ct];
			avg_io += io[ct];
			avg_comm += comm[ct];
			avg_compute += compute[ct];
			max_all = max(max_all, all[ct]);
			max_read = max(max_read, read[ct]);
			max_write = max(max_write, write[ct]);
			max_io = max(max_io, io[ct]);
			max_comm = max(max_comm, comm[ct]);
			max_compute = max(max_compute, compute[ct]);
			++ct;
		}
		fclose(in);
		fprintf(out, "Proc:%d\n", i+1);
		fprintf(out, "\tMAX:\n");
		fprintf(out, "\t\tAll:%lf, I/O:%lf, Comm:%lf Compute:%lf\n", max_all, max_io, max_comm, max_compute);
		fprintf(out, "\tAVG:\n");
		fprintf(out, "\t\tAll:%lf, I/O:%lf, Comm:%lf Compute:%lf\n\n", avg_all/ct, avg_io/ct, avg_comm/ct, avg_compute/ct);
	}
	fclose(out);
	return 0;
}