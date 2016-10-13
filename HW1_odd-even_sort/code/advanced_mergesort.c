#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#define ROOT 0
#define MAX_INT 2147483647

void swap (int*, int*);
void Mergesort(int, int*);
void my_merge(int, int, int*, int*, int*);

int main (int argc, char *argv[]) {
	int i, j, k;
	//	Initial MPI environment
	/*
		rank: the ID of each process
		size: # total available process
	*/
	int rank, size;
	MPI_Init(&argc, &argv);
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

	//	Part 1: Determine arguments of the I/O, and do MPI I/O
	/*
		num_per_node:	# number stored in each process
		rank_first_add:	the first process which need to add MAX_INT
		num_first_add:	# number that the first process which need to add
		read_file:		indicate the process if it need to read file
		*node_arr:		local number array of each process
	*/
	MPI_File ifh;
	MPI_Status istatus;	
	int num_per_node, rank_first_add, num_first_add, read_file, *node_arr;
	read_file=0;
	if(N<size){ //	N < #process
		num_per_node = 1;
		num_first_add = 1;
		rank_first_add = N;
		if(rank<rank_first_add)
			read_file = 1;
	}
	else{ //	N >= #process
		if(N%size){ //	If N can't be divided into the # process
			num_per_node = (N/size) + 1;
			rank_first_add = N/num_per_node; // # element to be add in the first rank
			num_first_add = num_per_node - (N%num_per_node); 
			if(rank<=rank_first_add)
				read_file = 1;
		}
		else{ //	If N can be divided into # process
			num_per_node = N/size;
			rank_first_add = size; // no need to add
			num_first_add = 0;
			read_file = 1;
		}
	}
	node_arr = (int*) malloc(num_per_node * sizeof(int)); // store the N/P numbers in each node
	MPI_File_open(MPI_COMM_WORLD, input, MPI_MODE_RDONLY, MPI_INFO_NULL, &ifh);
	if(read_file)
		MPI_File_set_view(ifh, rank*num_per_node*sizeof(int), MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
	else
		MPI_File_set_view(ifh, 0, MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
	if(read_file && rank!=rank_first_add)
		MPI_File_read_all(ifh, node_arr, num_per_node, MPI_INT, &istatus);
	else if(read_file && rank==rank_first_add){
		MPI_File_read_all(ifh, node_arr, (num_per_node-num_first_add), MPI_INT, &istatus);
		for(i=(num_per_node-num_first_add); i<num_per_node; ++i)
			node_arr[i] = MAX_INT;
	}
	else{
		MPI_File_read_all(ifh, node_arr, 0, MPI_INT, &istatus);
		for(i=0; i<num_per_node; ++i)
			node_arr[i] = MAX_INT;
	}
	MPI_File_close(&ifh);
	MPI_Barrier(MPI_COMM_WORLD);

	// Part 2: Start odd-even sort algorithm
	MPI_Status status;
	int *next_arr, *merge_arr, *ori_arr;
	next_arr = (int*) malloc(num_per_node * sizeof(int));
	Mergesort(num_per_node, node_arr);
	MPI_Barrier(MPI_COMM_WORLD);
	for(i=0; i<size; ++i){
		if(i%2==0){ //	Even-phase
			if(rank%2){ //	Odd-rank process: # 1, 3, 5...(Sender) => P -> P-1
				MPI_Send(node_arr, num_per_node, MPI_INT, rank-1, 8, MPI_COMM_WORLD);
				MPI_Recv(node_arr, num_per_node, MPI_INT, rank-1, 8, MPI_COMM_WORLD, &status);
			}
			else{ // Even-rank process: # 0, 2, 4...(Receiver) => P-1 <- P
				if(rank!=size-1){
					MPI_Recv(next_arr, num_per_node, MPI_INT, rank+1, 8, MPI_COMM_WORLD, &status);
					merge_arr = (int*) malloc(num_per_node * 2 * sizeof(int));
					for(j=0; j<num_per_node; ++j){
						merge_arr[j] = node_arr[j];
						merge_arr[j+num_per_node] = next_arr[j];
					}
					Mergesort(num_per_node*2, merge_arr);
					for(j=0; j<num_per_node; ++j){
						node_arr[j] = merge_arr[j];
						next_arr[j] = merge_arr[j+num_per_node];
					}
					free(merge_arr);
					MPI_Send(next_arr, num_per_node, MPI_INT, rank+1, 8, MPI_COMM_WORLD);
				}
			}
		}
		else{ //	Odd-phase
			if(rank%2==0){ //	Even-rank process: # 0, 2, 4... (Sender) => Q -> Q-1
				if(rank!=0){
					MPI_Send(node_arr, num_per_node, MPI_INT, rank-1, 8, MPI_COMM_WORLD);
					MPI_Recv(node_arr, num_per_node, MPI_INT, rank-1, 8, MPI_COMM_WORLD, &status);
				}
			}
			else{ //	Odd-rank process: # 1, 3, 5... (Receiver) => Q-1 <- Q
				if(rank!=size-1){
					MPI_Recv(next_arr, num_per_node, MPI_INT, rank+1, 8, MPI_COMM_WORLD, &status);
					merge_arr = (int*) malloc(num_per_node * 2 * sizeof(int));
					for(j=0; j<num_per_node; ++j){
						merge_arr[j] = node_arr[j];
						merge_arr[j+num_per_node] = next_arr[j];
					}
					Mergesort(num_per_node*2, merge_arr);
					for(j=0; j<num_per_node; ++j){
						node_arr[j] = merge_arr[j];
						next_arr[j] = merge_arr[j+num_per_node];
					}
					free(merge_arr);
					MPI_Send(next_arr, num_per_node, MPI_INT, rank+1, 8, MPI_COMM_WORLD);
				}
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	free(next_arr);

	MPI_File ofh;
	MPI_Status ostatus;
	MPI_File_open(MPI_COMM_WORLD, output, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &ofh);
	if(read_file)
		MPI_File_set_view(ofh, rank*num_per_node*sizeof(int), MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
	else
		MPI_File_set_view(ofh, 0, MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
	if(read_file && rank!=rank_first_add)
		MPI_File_write_all(ofh, node_arr, num_per_node, MPI_INT, &ostatus);
	else if(read_file && rank==rank_first_add)
		MPI_File_write_all(ofh, node_arr, (num_per_node-num_first_add), MPI_INT, &ostatus);
	else
		MPI_File_write_all(ofh, node_arr, 0, MPI_INT, &ostatus);

	MPI_File_close(&ofh);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}

void swap (int *a, int *b){
	int tmp = *a;
	*a = *b;
	*b = tmp;
}
void Mergesort(int num, int* data)
{
    if(num==1)
        return;
    int Lhs = num/2;
    int Rhs = num-Lhs;

    int *left = (int*) malloc(Lhs * sizeof(int)),
    	*right = (int*) malloc(Rhs * sizeof(int));

    //Copy data to local array.(to left[] and right[])
    int index;
    for(index=0 ; index<Lhs ; ++index)
        left[index] = data[index];
    for(index=0 ; index<Rhs ; ++index)
        right[index] = data[index+Lhs];
    //After copy, then cut the array until the length of array = 1.
    Mergesort(Lhs, left), Mergesort(Rhs, right);
    //Then, start to merge the arrays.
    my_merge(Lhs, Rhs, left, right, data);

    free(left);
    free(right);
}
void my_merge(int L_max, int R_max, int* left, int* right, int* data)
{
    int index=0, l=0, r=0;
    //While these two arrays are not out of its bound.
    while(l<L_max && r<R_max){
        if(left[l]<=right[r])
            data[index++] = left[l++];
        else
            data[index++] = right[r++];
    }
    while(l<L_max)
        data[index++] = left[l++];
    while(r<R_max)
        data[index++] = right[r++];
}
