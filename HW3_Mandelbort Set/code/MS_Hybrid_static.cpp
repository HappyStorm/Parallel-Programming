#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <mpi.h>
#include <omp.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
using namespace std;
#define SQUARE(x) (x)*(x)
#define CUBE(x) (x)*(x)*(x)
#define MAX(a,b) \
        ({ __typeof__ (a) _a = (a); \
            __typeof__ (b) _b = (b); \
            _a > _b ? _a : _b; })
#define MIN(a,b) \
        ({ __typeof__ (a) _a = (a); \
            __typeof__ (b) _b = (b); \
            _a > _b ? _b : _a; })
#define ROW_TAG_M2S 11
#define DATA_TAG_S2M 9
#define TER_TAH_M2S 10
#define DRAW_CONST_MUL 1048576
#define DRAW_CONST_MOD 256

typedef struct complextype{
	double real, imag;
} Compl;

/* program variables */
int num_thread, np_x, np_y, Xenable=0;
double L_x, R_x,
	   U_y, L_y;

/* Xwindow variables */
Window window;
Display *display;
GC gc;
int screen,
	width, height,
    black, white;
double  real_adj, real_div,
        imag_adj, imag_div;

void Init(int, char* []);
void InitWindow();

int main(int argc, char *argv[])
{
	Init(argc, argv);

    // Initialize OpenMP
    omp_set_num_threads(num_thread);

	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // double  st = MPI_Wtime(),
    //         et;

	MPI_Request request;
	MPI_Status status;

    int row_per_proc=0, row_last_proc=0;

    if(size>1){ // size > 1
        if((np_y%(size-1))!=0){ // np_y can't be divided by size-1(#proc-slave)
            row_per_proc = (np_y/(size-1)) + 1;
            row_last_proc = np_y - (row_per_proc * (size-2));
        }
        else{ // np can be divided by size-1(#proc-slave)
            row_per_proc = np_y/(size-1);
            row_last_proc = row_per_proc;
        }
        int *data_per_proc; // range of data, like rectangle
        data_per_proc = (int*) calloc(row_per_proc * np_x, sizeof(int));
    	if(rank==0){ // Master
            if(Xenable)
                InitWindow();
    		int ct = size-1; // remaining #task
    		while(ct){
    			MPI_Irecv(data_per_proc, row_per_proc * np_x, MPI_INT, MPI_ANY_SOURCE, DATA_TAG_S2M, MPI_COMM_WORLD, &request); // slave return the row_data
    			MPI_Wait(&request, &status);
                --ct;
    			int returnSlaveID = status.MPI_SOURCE; // get which slave come back
    			if(Xenable){
                    if(returnSlaveID!=size-1){ // not last process
                        #pragma omp parallel
                        {
                            #pragma for schedule(static)
                			for(int i=0; i<row_per_proc; ++i){
                                int real_i = (returnSlaveID-1) * row_per_proc + i;
                                #pragma omp critical
                                {
                                    for(int j=0; j<width; ++j){
                        				XSetForeground (display, gc, 
                                            DRAW_CONST_MUL * (data_per_proc[i*np_x+j] % DRAW_CONST_MOD));		
                                        XDrawPoint(display, window, gc, j, real_i);
                        			}
                                }
                            }
                        }
                    }
                    else{ // last process
                        #pragma omp parallel
                        {
                            #pragma omp for schedule(static)
                            for(int i=0; i<row_last_proc; ++i){
                                int real_i = (returnSlaveID-1) * row_per_proc + i;
                                #pragma omp critical
                                {
                                    for(int j=0; j<width; ++j){
                                        XSetForeground(display, gc, 
                                            DRAW_CONST_MUL * (data_per_proc[i*np_x+j] % DRAW_CONST_MOD));       
                                        XDrawPoint(display, window, gc, j, real_i);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if(Xenable){
                XFlush(display);
                sleep(10);
            }
    		free(data_per_proc);
            // et = MPI_Wtime();
            // printf("Rank: %d Thread: -1 Time: %lf\n", rank, et-st);
    	}
    	else{ // Slave
            // double *thread_time = (double*) calloc(num_thread, sizeof(double));
            // int *thread_point = (int*) calloc(num_thread, sizeof(int));
            if(rank!=size-1){ // do task(only one time), not the last process
                #pragma omp parallel
                {
                    #pragma omp for schedule(static)
                    for(int i=0; i<row_per_proc; ++i){
                        for(int j=0; j<width; ++j){
                            // double thread_stime = omp_get_wtime();
                            Compl z,c;
                            z.real = 0.0;
                            z.imag = 0.0;
                            c.real = ((double) j - real_adj) / real_div;    // X-axis
                            c.imag = ((double) (rank-1)*row_per_proc + i - imag_adj) / imag_div;// Y-axis
                            int repeats = 0;
                            double temp, lengthsq = 0.0;
                            while(repeats < 100000 && lengthsq < 4.0){
                                temp = z.real*z.real - z.imag*z.imag + c.real;
                                z.imag = 2.0*z.real*z.imag + c.imag;
                                z.real = temp;
                                lengthsq = z.real*z.real + z.imag*z.imag;
                                ++repeats;
                            }
                            data_per_proc[i*np_x+j] = repeats; // to store result in row_data
                            // double thread_etime = omp_get_wtime();
                            // thread_time[omp_get_thread_num()] += thread_etime - thread_stime;
                            // ++thread_point[omp_get_thread_num()];
                        }
                    }
                }
                MPI_Isend(data_per_proc, row_per_proc * np_x, MPI_INT, 0, DATA_TAG_S2M, MPI_COMM_WORLD, &request);
            }
            else{
                #pragma omp parallel
                {
                    #pragma omp for schedule(static)
                    for(int i=0; i<row_last_proc; ++i){
                        for(int j=0; j<width; ++j){
                            // double thread_stime = omp_get_wtime();
                            Compl z,c;
                            z.real = 0.0;
                            z.imag = 0.0;
                            c.real = ((double) j - real_adj) / real_div;    // X-axis
                            c.imag = ((double) (rank-1)*row_per_proc + i - imag_adj) / imag_div;// Y-axis
                            int repeats = 0;
                            double temp, lengthsq = 0.0;
                            while(repeats < 100000 && lengthsq < 4.0){
                                temp = z.real*z.real - z.imag*z.imag + c.real;
                                z.imag = 2.0*z.real*z.imag + c.imag;
                                z.real = temp;
                                lengthsq = z.real*z.real + z.imag*z.imag;
                                ++repeats;
                            }
                            data_per_proc[i*np_x+j] = repeats; // to store result in row_data
                            // double thread_etime = omp_get_wtime();
                            // thread_time[omp_get_thread_num()] += thread_etime - thread_stime;
                            // ++thread_point[omp_get_thread_num()];
                        }
                    }
                }
                MPI_Isend(data_per_proc, row_per_proc * np_x, MPI_INT, 0, DATA_TAG_S2M, MPI_COMM_WORLD, &request);
            }
            MPI_Wait(&request, &status);
            free(data_per_proc);
            // free(thread_time);
            // free(thread_point);
            // et = MPI_Wtime();
            // for(int i=0; i<num_thread; ++i)
            //     printf("Thread %d => %d points\n", i, thread_point[i]);
            //     printf("Rank: %d Thread: %d Time: %lf\n", rank, i, thread_time[i]);
            // printf("Rank: %d Thread: -1 Time: %lf\n", rank, et-st);
        }
    }
    else{ // size = 1
        if(Xenable)
            InitWindow();
        // double *thread_time = (double*) calloc(num_thread, sizeof(double));
        #pragma omp parallel for schedule(static)
        for(int i=0; i<height; ++i){
            for(int j=0; j<width; ++j){
                // double thread_stime = omp_get_wtime();
                Compl z, c;
                z.real = 0.0;
                z.imag = 0.0;
                c.real = ((double)j - real_adj) / real_div;
                c.imag = ((double)i - imag_adj) / imag_div;
                int repeats = 0;
                double temp, lengthsq = 0.0;
                while(repeats < 100000 && lengthsq < 4.0){
                    temp = z.real*z.real - z.imag*z.imag + c.real;
                    z.imag = 2*z.real*z.imag + c.imag;
                    z.real = temp;
                    lengthsq = z.real*z.real + z.imag*z.imag; 
                    repeats++;
                }
                // double thread_etime = omp_get_wtime();
                // thread_time[omp_get_thread_num()] += thread_etime - thread_stime;
                #pragma omp critical
                {
                    if(Xenable){
                        XSetForeground(display, gc, DRAW_CONST_MUL * (repeats % DRAW_CONST_MOD));       
                        XDrawPoint(display, window, gc, j, i);
                    }
                }
            }
        }
        // et = MPI_Wtime();
        // for(int i=0; i<num_thread; ++i)
        //     printf("Rank: %d Thread: %d Time: %lf\n", rank, i, thread_time[i]);
        // printf("Rank: %d Thread: -1 Time: %lf\n", rank, et-st);
        // free(thread_time);
        if(Xenable){
            XFlush(display);
            sleep(10);
        }
    }
    MPI_Finalize();
	return 0;
}


void Init(int argc, char* argv[])
{
    num_thread = stoi(argv[1]);
    L_x = stod(argv[2]), R_x = stod(argv[3]);
    L_y = stod(argv[4]), U_y = stod(argv[5]);
    np_x = stoi(argv[6]), np_y = stoi(argv[7]);
    Xenable = (argv[8]==string("enable")) ? 1 : 0;
    real_adj = (np_x * (R_x - L_x) - (R_x * np_x)) / (R_x - L_x);
    real_div = np_x / (R_x - L_x);
    imag_adj = (np_y * (U_y - L_y) - (U_y * np_y)) / (U_y - L_y);
    imag_div = np_y / (U_y - L_y);
    width = np_x;
    height = np_y;
}

void InitWindow()
{
    if((display=XOpenDisplay(NULL))==NULL){
        fprintf(stderr, "Cannot connect to X server %s\n",
                XDisplayName(NULL));
        exit(1);
    }
    screen = DefaultScreen(display);

    // set window position 
    int x = 0;
    int y = 0;

    // border width in pixels 
    int border_width = 0;

    // get color representation
    black = BlackPixel(display, screen);
    white = WhitePixel(display, screen);

    // create window
    window = XCreateSimpleWindow(display, RootWindow(display, screen),
        x, y, width, height, border_width, black, white);
    XStoreName(display, window, "Mandelbrot Set (Hybrid_Static)");

    // create graph 
    XGCValues values;
    long valuemask = 0;
    gc = XCreateGC(display, window, valuemask, &values);
    XSetForeground (display, gc, black);
	XSetBackground(display, gc, 0X0000FF00);
    XSetLineAttributes(display, gc, 1, LineSolid, CapRound, JoinRound);
    XMapWindow(display, window);
    XSync(display, 0);
}