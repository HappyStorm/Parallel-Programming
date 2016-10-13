#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <omp.h>
const int INF = 10000000;
void input(char *inFileName);
void output(char *outFileName);
void block_FW(int B);
void print_device(int);
int ceil(int a, int b);
__global__ void phase_1(int*, int, int, int, int, int);
__global__ void phase_2(int*, int, int, int, int, int);
__global__ void phase_2_row(int*, int, int, int, int, int);
__global__ void phase_2_col(int*, int, int, int, int, int);
__global__ void phase_3(int*, int, int, int, int, int);
__global__ void phase_3_top(int*, int, int, int, int, int);
__global__ void phase_3_bot(int*, int, int, int, int, int);
__device__ int cal(int, int, int);

int n, m, // Number of vertices, edges
	num_device;
int *h_Dist,
	*d_Dist;
float compute_time;

double  total_start, total_end, total_time=0.0,
		comm0_start, comm0_end, comm0_time=0.0,
		mem0_start, mem0_end, mem0_time=0.0,
		comm1_start, comm1_end, comm1_time=0.0,
		mem1_start, mem1_end, mem1_time=0.0,
		io_start, io_end, io_time=0.0;

int main(int argc, char* argv[])
{
	total_start = omp_get_wtime();
	input(argv[1]);
	int B = atoi(argv[3]);  // B = # of vertices
	block_FW(B);
	output(argv[2]);
	cudaFreeHost(h_Dist);
	total_end = omp_get_wtime();
	printf("Total: %lf\n", total_end - total_start);
	printf("\tMemcpy: %lf\n", mem0_time + mem1_time);
	printf("\tComm0:%lf Comm1:%lf Total Comm:%lf\n", comm0_time, comm1_time, comm0_time+comm1_time);
	printf("\tI/O: %lf\n", io_time);
	return 0;
}

void input(char *inFileName)
{
	io_start = omp_get_wtime();
	FILE *infile = fopen(inFileName, "r");
	fscanf(infile, "%d %d", &n, &m);
	
	if(cudaHostAlloc((void **)&h_Dist, n * n * sizeof(int), cudaHostAllocDefault)!=cudaSuccess)
		printf("cudaHostAlloc() failed\n");
	
	for (int i=0; i<n; ++i){
		for (int j=0; j<n; ++j){
			if (i==j)	h_Dist[i*n+j] = 0;
			else		h_Dist[i*n+j] = INF;
		}
	}
	while(--m >= 0){
		int a, b, v;
		fscanf(infile, "%d %d %d", &a, &b, &v);
		--a, --b;
		h_Dist[a*n+b] = v;
	}
	io_end = omp_get_wtime();
	io_time += io_end - io_start;
}

void output(char *outFileName)
{
	io_start = omp_get_wtime();
	FILE *outfile = fopen(outFileName, "w");
	for(int i=0; i<n; ++i){
		for(int j=0; j<n; ++j){
			if (h_Dist[i*n+j] >= INF)	fprintf(outfile, "INF ");
			else						fprintf(outfile, "%d ", h_Dist[i*n+j]);
		}
		fprintf(outfile, "\n");
	}
	io_end = omp_get_wtime();
	io_time += io_end - io_start;
}

int ceil(int a, int b)
{
	return (a + b -1)/b;
}

void block_FW(int B) // each phase will do B-iteration
{
	int round = ceil(n, B);	// round = the width and height of block matrix
	dim3 block(B, B);
	
	cudaGetDeviceCount(&num_device);
	omp_set_num_threads(num_device);
	// printf("#CUDA Device: %d\n", num_device);
	
	#pragma omp parallel
	{	
		unsigned int thread_id = omp_get_thread_num();
		if(cudaSetDevice(thread_id)!=cudaSuccess)
	    	printf("CPU %d set device %d failed\n", thread_id, thread_id%2);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
	    cudaEventCreate(&stop);

    	// print_device(thread_id);
	    
		cudaEventRecord(start, 0);
		for(int r=0; r<round; ++r){ // traverse block matrix
			/* Phase 1 => 1 primary block(pivot block)*/
			dim3 grid_1(1, 1);
			int sharedMemory = B * B * sizeof(int);
			if(thread_id==1){
				if(cudaSetDevice(thread_id)!=cudaSuccess)
	    			printf("CPU %d set device %d failed\n", thread_id, thread_id%2);
	    		phase_1<<<grid_1, block, sharedMemory>>>(h_Dist, B, r, n, round, num_device);
			}
			#pragma omp barrier

			dim3 grid_2(round, 1);
			sharedMemory = B * B * 2 * sizeof(int);
			if(thread_id==0){
				if(cudaSetDevice(thread_id)!=cudaSuccess)
					printf("CPU %d set device %d failed\n", thread_id, thread_id%2);
				phase_2_row<<<grid_2, block, sharedMemory>>>(h_Dist, B, r, n, round, num_device);
				cudaDeviceSynchronize();
			}
			else{
				if(cudaSetDevice(thread_id)!=cudaSuccess)
					printf("CPU %d set device %d failed\n", thread_id, thread_id%2);
				phase_2_col<<<grid_2, block, sharedMemory>>>(h_Dist, B, r, n, round, num_device);
				cudaDeviceSynchronize();
			}

			dim3 grid_3(round, round);
			sharedMemory = B * B * 3 * sizeof(int);
			if(thread_id==0){
				if(cudaSetDevice(thread_id)!=cudaSuccess)
					printf("CPU %d set device %d failed\n", thread_id, thread_id%2);
				phase_3_top<<<grid_3, block, sharedMemory>>>(h_Dist, B, r, n, round, num_device);
				cudaDeviceSynchronize();
			}
			else{
				if(cudaSetDevice(thread_id)!=cudaSuccess)
					printf("CPU %d set device %d failed\n", thread_id, thread_id%2);
				phase_3_bot<<<grid_3, block, sharedMemory>>>(h_Dist, B, r, n, round, num_device);
				cudaDeviceSynchronize();
			}
		}
		cudaEventRecord(stop, 0);
	    cudaEventSynchronize(stop);
	    cudaEventElapsedTime(&compute_time, start, stop);
		fprintf(stderr, "GPU %d Compute Time: %lf\n\n", thread_id, compute_time);
		cudaEventDestroy(start);
	    cudaEventDestroy(stop);
	}
}

__global__ void phase_1(int* d_Dist, int B, int r, int n, int Round, int num_device)
{
	int b_i = r,
		b_j = r,
		i = b_i * B + threadIdx.x,
		j = b_j * B + threadIdx.y;

	extern __shared__ int shared_Dist[];
	int *pri_Dist = &shared_Dist[0];
	
	if(i>=n || j>=n){
		return;
	}
	
	pri_Dist[threadIdx.x*B+threadIdx.y] = d_Dist[i*n+j];
	__syncthreads();

	for(int k=0; k<B; ++k){ // Round = ceil(N, B) = block matrix's dimention
		pri_Dist[threadIdx.x*B+threadIdx.y] = cal(pri_Dist[threadIdx.x*B+k], 
			pri_Dist[k*B+threadIdx.y], pri_Dist[threadIdx.x*B+threadIdx.y]);
		__syncthreads();
	}
	d_Dist[i*n+j] = pri_Dist[threadIdx.x*B+threadIdx.y];
}

__global__ void phase_2_row(int* d_Dist, int B, int r, int n, int Round, int num_device)
{
	int b_i = r, b_j = blockIdx.x,
		i = b_i * B + threadIdx.x,
		j = b_j * B + threadIdx.y,
		tpb_i = r * B + threadIdx.x,
		tpb_j = r * B + threadIdx.y;

	extern __shared__ int shared_Dist[];

	int *pri_Dist = &shared_Dist[0],
		*sel_Dist = &pri_Dist[B*B];

	if(b_i==b_j) return;
	if(i>=n || j>=n || tpb_i>=n || tpb_j>=n){
		return;
	}

	pri_Dist[threadIdx.x*B+threadIdx.y] = d_Dist[tpb_i*n+tpb_j];
	sel_Dist[threadIdx.x*B+threadIdx.y] = d_Dist[i*n+j];
	__syncthreads();

	for(int k=0; k<B; ++k){ // Round = ceil(N, B) = block matrix's dimention
		sel_Dist[threadIdx.x*B+threadIdx.y] = cal(pri_Dist[threadIdx.x*B+k],
			sel_Dist[k*B+threadIdx.y], sel_Dist[threadIdx.x*B+threadIdx.y]);
		__syncthreads();
	}
	d_Dist[i*n+j] = sel_Dist[threadIdx.x*B+threadIdx.y];
}

__global__ void phase_2_col(int* d_Dist, int B, int r, int n, int Round, int num_device)
{
	int b_i = Round-1-blockIdx.x, b_j = r,
		i = b_i * B + threadIdx.x,
		j = b_j * B + threadIdx.y,
		tpb_i = r * B + threadIdx.x,
		tpb_j = r * B + threadIdx.y;

	extern __shared__ int shared_Dist[];

	int *pri_Dist = &shared_Dist[0],
		*sel_Dist = &pri_Dist[B*B];

	if(b_i==b_j) return;
	if(i>=n || j>=n || tpb_i>=n || tpb_j>=n){
		return;
	}

	pri_Dist[threadIdx.x*B+threadIdx.y] = d_Dist[tpb_i*n+tpb_j];
	sel_Dist[threadIdx.x*B+threadIdx.y] = d_Dist[i*n+j];
	__syncthreads();

	for(int k=0; k<B; ++k){ // Round = ceil(N, B) = block matrix's dimention
		sel_Dist[threadIdx.x*B+threadIdx.y] = cal(sel_Dist[threadIdx.x*B+k],
			pri_Dist[k*B+threadIdx.y], sel_Dist[threadIdx.x*B+threadIdx.y]);
		__syncthreads();
	}
	d_Dist[i*n+j] = sel_Dist[threadIdx.x*B+threadIdx.y];
}

__global__ void phase_3_top(int* d_Dist, int B, int r, int n, int Round, int num_device)
{
	int b_i = r-1-blockIdx.x, b_j = blockIdx.y;
	int	i = b_i * B + threadIdx.x,
		j = b_j * B + threadIdx.y,
		trb_i = b_i * B + threadIdx.x,
		trb_j = r * B + threadIdx.y,
		tcb_i = r * B + threadIdx.x,
		tcb_j = b_j * B + threadIdx.y;

	extern __shared__ int shared_Dist[];
	int *row_Dist = &shared_Dist[0],
		*col_Dist = &row_Dist[B*B],
		*sel_Dist = &col_Dist[B*B];
	


	if(b_i==r || b_j==r || b_i<0) return;
	if(i>=n || j>=n || trb_i>=n || trb_j>=n || tcb_i>=n || tcb_j>=n){
		return;
	}
	row_Dist[threadIdx.x*B+threadIdx.y] = d_Dist[trb_i*n+trb_j];
	col_Dist[threadIdx.x*B+threadIdx.y] = d_Dist[tcb_i*n+tcb_j];
	sel_Dist[threadIdx.x*B+threadIdx.y] = d_Dist[i*n+j];
	__syncthreads();

	for(int k=0; k<B; ++k){ // Round = ceil(N, B) = block matrix's dimention
		sel_Dist[threadIdx.x*B+threadIdx.y] = cal(row_Dist[threadIdx.x*B+k],
			col_Dist[k*B+threadIdx.y], sel_Dist[threadIdx.x*B+threadIdx.y]);
		__syncthreads();
	}
	d_Dist[i*n+j] = sel_Dist[threadIdx.x*B+threadIdx.y];
}

__global__ void phase_3_bot(int* d_Dist, int B, int r, int n, int Round, int num_device)
{
	int b_i = r+1+blockIdx.x, b_j = blockIdx.y;
	int	i = b_i * B + threadIdx.x,
		j = b_j * B + threadIdx.y,
		trb_i = b_i * B + threadIdx.x,
		trb_j = r * B + threadIdx.y,
		tcb_i = r * B + threadIdx.x,
		tcb_j = b_j * B + threadIdx.y;

	extern __shared__ int shared_Dist[];
	int *row_Dist = &shared_Dist[0],
		*col_Dist = &row_Dist[B*B],
		*sel_Dist = &col_Dist[B*B];
	


	if(b_i==r || b_j==r) return;
	if(i>=n || j>=n || trb_i>=n || trb_j>=n || tcb_i>=n || tcb_j>=n){
		return;
	}
	row_Dist[threadIdx.x*B+threadIdx.y] = d_Dist[trb_i*n+trb_j];
	col_Dist[threadIdx.x*B+threadIdx.y] = d_Dist[tcb_i*n+tcb_j];
	sel_Dist[threadIdx.x*B+threadIdx.y] = d_Dist[i*n+j];
	__syncthreads();

	for(int k=0; k<B; ++k){ // Round = ceil(N, B) = block matrix's dimention
		sel_Dist[threadIdx.x*B+threadIdx.y] = cal(row_Dist[threadIdx.x*B+k],
			col_Dist[k*B+threadIdx.y], sel_Dist[threadIdx.x*B+threadIdx.y]);
		__syncthreads();
	}
	d_Dist[i*n+j] = sel_Dist[threadIdx.x*B+threadIdx.y];
}

__device__ int cal(int ik, int kj, int ij)
{		
	return (ij>ik+kj) ? ik+kj : ij;
}

void print_device(int thread_id)
{
	printf("Now thread_%d GPU_%d\n", thread_id, thread_id);
    cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, thread_id);
	printf("%s\n", prop.name);
	printf("CanMapHostMemory:%d\n", prop.canMapHostMemory);
	printf("MaxThread_PerBlock:%d\n", prop.maxThreadsPerBlock);
	printf("SharedMem_PerBlock:%d\n", prop.sharedMemPerBlock);
}
