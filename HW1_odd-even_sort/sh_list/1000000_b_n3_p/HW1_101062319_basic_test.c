#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#define ROOT 0
#define MAX_INT 2147483647

void swap (int*, int*);

int main (int argc, char *argv[]) {

	double all_start, all_end, all_total;
	double comm_start, comm_end, comm_total;
	double ior_start, ior_end, ior_total, iow_start, iow_end, iow_total, io_total;
	double compute_start, compute_end, compute_total;

	all_total = 0;
	comm_total = 0;
	ior_total = 0;
	iow_total = 0;
	io_total = 0;
	compute_total = 0;


	int i, j, k;
	//	Initial MPI environment
	/*
		rank: the ID of each process
		size: # total available process
	*/
	int rank, size;

	MPI_Init(&argc, &argv);
	all_start = MPI_Wtime();

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//	Argument exception handle
	if(argc < 4) {
		if (rank == ROOT) {
			fprintf(stderr, "Insufficient args\n");
			fprintf(stderr, "Usage: %s N input_file", argv[0]);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();
		return 0;
	}

	//	Argument assigned
	const int N = atoi(argv[1]);
	const char *input = argv[2];
	const char *output = argv[3];

	//	Part 1: Use MPI I/O to open the input file
	MPI_File ifh;
	MPI_Status istatus;
	
	//	Part 2: Use MPI I/O to read the file(N numbers) into the node-array, then close it
	int num_per_node, rank_last, *node_arr, work;
	work = 1;
	

	compute_start = MPI_Wtime();
	if(N<size){ //	N < #process
		num_per_node = 1;
		rank_last = N;
		if(rank>=N)
			work = 0, num_per_node = 0;
	}
	else{ //	N >= #process
		if(N%size){ //	If N can't be divided into the # process
			if(rank!=size-1)
				num_per_node = N/size;
			else
				num_per_node = N - ((N/size)*(size-1));
			rank_last = size;
		}
		else{ //	If N can be divided into # process
			num_per_node = N/size;
			rank_last = size;
		}
	}
	compute_end = MPI_Wtime();
	compute_total += (compute_end - compute_start);

	node_arr = (int*) calloc(num_per_node, sizeof(int)); // store the N/P numbers in each node
	
	ior_start = MPI_Wtime();
	MPI_File_open(MPI_COMM_WORLD, input, MPI_MODE_RDONLY, MPI_INFO_NULL, &ifh);
	if(N>=size && rank==rank_last-1)
		MPI_File_set_view(ifh, rank*(N/size)*sizeof(int), MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
	else
		MPI_File_set_view(ifh, rank*num_per_node*sizeof(int), MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
	MPI_File_read_all(ifh, node_arr, num_per_node, MPI_INT, &istatus);

	MPI_File_close(&ifh);
	ior_end = MPI_Wtime();
	ior_total = (ior_end - ior_start);
	io_total += ior_total;

	MPI_Barrier(MPI_COMM_WORLD);

	// Part 3: Start odd-even sort algorithm
	/*
		node_sorted: whether the node is sorted or not
		all_sorted:  whether all nodes are sorted or not
	*/
	MPI_Status status;
	int node_sorted, all_sorted, receive;
	
	node_sorted = 0, all_sorted = 0, receive = 0;
	while(!all_sorted){
		node_sorted = 1, all_sorted = 1;

		if(work){
			//	Even-phase
			if(num_per_node%2==0){ //	Even-list

				compute_start = MPI_Wtime();
				for(i=0; i<num_per_node; i+=2){
					if(node_arr[i]>node_arr[i+1]){
						swap(&node_arr[i], &node_arr[i+1]);
						node_sorted = 0;
					}
				}
				compute_end = MPI_Wtime();
				compute_total += (compute_end - compute_start);
			}
			else{ //	Odd-list
				if(rank%2==0){ //	Even-index process(0, 2, 4...)

					compute_start = MPI_Wtime();
					for(i=0; i<num_per_node-1; i+=2){
						if(node_arr[i]>node_arr[i+1]){
							swap(&node_arr[i], &node_arr[i+1]);
							node_sorted = 0;
						}
					}
					compute_end = MPI_Wtime();
					compute_total += (compute_end - compute_start);


					if(rank!=rank_last-1){
						comm_start = MPI_Wtime();
						MPI_Send(&node_arr[num_per_node-1], 1, MPI_INT, rank+1, 8, MPI_COMM_WORLD);
						comm_end = MPI_Wtime();
						comm_total += (comm_end - comm_start);

						comm_start = MPI_Wtime();
						MPI_Recv(&node_arr[num_per_node-1], 1, MPI_INT, rank+1, 8, MPI_COMM_WORLD, &status);
						comm_end = MPI_Wtime();
						comm_total += (comm_end - comm_start);
					}
				}
				else{ //	Odd-index process(1, 3, 5...)
					compute_start = MPI_Wtime();
					for(i=1; i<num_per_node; i+=2){
						if(node_arr[i]>node_arr[i+1]){
							swap(&node_arr[i], &node_arr[i+1]);
							node_sorted = 0;
						}
					}
					compute_end = MPI_Wtime();
					compute_total += (compute_end - compute_start);

					comm_start = MPI_Wtime();
					MPI_Recv(&receive, 1, MPI_INT, rank-1, 8, MPI_COMM_WORLD, &status);
					comm_end = MPI_Wtime();
					comm_total += (comm_end - comm_start);

					compute_start = MPI_Wtime();
					if(receive>node_arr[0]){
						swap(&receive, &node_arr[0]);
						node_sorted = 0;
					}
					compute_end = MPI_Wtime();
					compute_total += (compute_end - compute_start);

					comm_start = MPI_Wtime();
					MPI_Send(&receive, 1, MPI_INT, rank-1, 8, MPI_COMM_WORLD);
					comm_end = MPI_Wtime();
					comm_total += (comm_end - comm_start);
				}
			}
			//	Odd-phase
			if(num_per_node%2==0){ //	Even-list

				compute_start = MPI_Wtime();
				for(i=1; i<num_per_node-1; i+=2){
					if(node_arr[i]>node_arr[i+1]){
						swap(&node_arr[i], &node_arr[i+1]);
						node_sorted = 0;
					}
				}
				compute_end = MPI_Wtime();
				compute_total += (compute_end - compute_start);


				if(rank!=0){ //	Normal process, nut not the last

					comm_start = MPI_Wtime();
					MPI_Recv(&receive, 1, MPI_INT, rank-1, 8, MPI_COMM_WORLD, &status);
					comm_end = MPI_Wtime();
					comm_total += (comm_end - comm_start);

					compute_start = MPI_Wtime();
					if(receive>node_arr[0]){
						swap(&receive, &node_arr[0]);
						node_sorted = 0;
					}
					compute_end = MPI_Wtime();
					compute_total += (compute_end - compute_start);
					
					comm_start = MPI_Wtime();
					MPI_Send(&receive, 1, MPI_INT, rank-1, 8, MPI_COMM_WORLD);
					comm_end = MPI_Wtime();
					comm_total += (comm_end - comm_start);
				}
				if(rank!=rank_last-1){
					
					comm_start = MPI_Wtime();
					MPI_Send(&node_arr[num_per_node-1], 1, MPI_INT, rank+1, 8, MPI_COMM_WORLD);
					comm_end = MPI_Wtime();
					comm_total += (comm_end - comm_start);

					comm_start = MPI_Wtime();
					MPI_Recv(&node_arr[num_per_node-1], 1, MPI_INT, rank+1, 8, MPI_COMM_WORLD, &status);
					comm_end = MPI_Wtime();
					comm_total += (comm_end - comm_start);
				}
			}
			else{ //	Odd-list
				if(rank%2==0){

					compute_start = MPI_Wtime();
					for(i=1; i<num_per_node; i+=2){
						if(node_arr[i]>node_arr[i+1]){
							swap(&node_arr[i], &node_arr[i+1]);
							node_sorted = 0;
						}
					}
					compute_end = MPI_Wtime();
					compute_total += (compute_end - compute_start);

					if(rank!=0){ //	Normal process, nut not the last
						
						comm_start = MPI_Wtime();
						MPI_Recv(&receive, 1, MPI_INT, rank-1, 8, MPI_COMM_WORLD, &status);
						comm_end = MPI_Wtime();
						comm_total += (comm_end - comm_start);

						compute_start = MPI_Wtime();
						if(receive>node_arr[0]){
							swap(&receive, &node_arr[0]);
							node_sorted = 0;
						}
						compute_end = MPI_Wtime();
						compute_total += (compute_end - compute_start);


						comm_start = MPI_Wtime();
						MPI_Send(&receive, 1, MPI_INT, rank-1, 8, MPI_COMM_WORLD);
						comm_end = MPI_Wtime();
						comm_total += (comm_end - comm_start);
					}
				}
				else{
					compute_start = MPI_Wtime();
					for(i=0; i<num_per_node-1; i+=2){
						if(node_arr[i]>node_arr[i+1]){
							swap(&node_arr[i], &node_arr[i+1]);
							node_sorted = 0;
						}
					}
					compute_end = MPI_Wtime();
					compute_total += (compute_end - compute_start);
					

					if(rank!=rank_last-1){

						comm_start = MPI_Wtime();
						MPI_Send(&node_arr[num_per_node-1], 1, MPI_INT, rank+1, 8, MPI_COMM_WORLD);
						comm_end = MPI_Wtime();
						comm_total += (comm_end - comm_start);

						comm_start = MPI_Wtime();
						MPI_Recv(&node_arr[num_per_node-1], 1, MPI_INT, rank+1, 8, MPI_COMM_WORLD, &status);
						comm_end = MPI_Wtime();
						comm_total += (comm_end - comm_start);
					}
				}
			}
		}
		comm_start = MPI_Wtime();
		MPI_Allreduce(&node_sorted, &all_sorted, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
		comm_end = MPI_Wtime();
		comm_total += (comm_end - comm_start);

		MPI_Barrier(MPI_COMM_WORLD);
		if(all_sorted){
			break;
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);


	//	Part 4: Print the output by MPI I/O
	MPI_File ofh;
	MPI_Status ostatus;

	iow_start = MPI_Wtime();

	MPI_File_open(MPI_COMM_WORLD, output, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &ofh);
	
	if(N>=size && rank==rank_last-1)
		MPI_File_set_view(ofh, rank*(N/size)*sizeof(int), MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
	else
		MPI_File_set_view(ofh, rank*num_per_node*sizeof(int), MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
	MPI_File_write_all(ofh, node_arr, num_per_node, MPI_INT, &ostatus);
	MPI_File_close(&ofh);
	iow_end = MPI_Wtime();
	iow_total = (iow_end - iow_start);
	io_total += iow_total;


	all_end = MPI_Wtime();
	all_total = (all_end - all_start);

	printf("Rank:%d (All:%lf, I:%lf, W:%lf, I/O:%lf, Comm:%lf, Compute:%lf)\n", rank, all_total,
		 ior_total, iow_total, io_total, comm_total, compute_total);


	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}

void swap (int *a, int *b){
	int tmp = *a;
	*a = *b;
	*b = tmp;
}
