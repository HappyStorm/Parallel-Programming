#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
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

int main(int argc, char* argv[])
{
    // double  st = omp_get_wtime(),
    //         et;
	Init(argc, argv);
    if(Xenable)
	   InitWindow();
	// Initialize OpenMP
    omp_set_num_threads(num_thread);

    int task_per_thread = np_y / num_thread;
    
    // double *thread_time = (double*) calloc(num_thread, sizeof(double));
    // int *thread_point = (int*) calloc(num_thread, sizeof(int));

	#pragma omp parallel
	{
		int *data_per_row = (int*) malloc(np_x * sizeof(int));
		// #pragma omp for schedule(dynamic, task_per_thread)
        #pragma omp for schedule(dynamic)
		for(int i=0; i<height; ++i){
			for(int j=0; j<width; ++j){
                // double thread_stime = omp_get_wtime();
				Compl z, c;
				z.real = 0.0;
				z.imag = 0.0;
				c.real = ((double)j - real_adj) / real_div;
				c.imag = ((double)i - imag_adj) / imag_div;
				int repeats = 0;
				double lengthsq = 0.0, temp;
				while(repeats < 100000 && lengthsq < 4.0){
					temp = z.real*z.real - z.imag*z.imag + c.real;
					z.imag = 2*z.real*z.imag + c.imag;
					z.real = temp;
					lengthsq = z.real*z.real + z.imag*z.imag; 
					repeats++;
				}
				data_per_row[j] = repeats;
                // double thread_etime = omp_get_wtime();
                // thread_time[omp_get_thread_num()] += thread_etime - thread_stime;
                // ++thread_point[omp_get_thread_num()];
			}
			#pragma omp critical
			{
				if(Xenable){
					for(int j=0; j<width; ++j){
						XSetForeground(display, gc, DRAW_CONST_MUL * (data_per_row[j] % DRAW_CONST_MOD));
						XDrawPoint(display, window, gc, j, i);
					}
				}
			}
		}
		free(data_per_row);
	}
	if(Xenable){
		XFlush(display);
        sleep(10);
	}
    // et =  omp_get_wtime();
    // for(int i=0; i<num_thread; ++i)
    //     printf("Thread %d => %d points\n", i, thread_point[i]);
        // printf("Thread:%d Time:%lf\n", i, thread_time[i]);
    // printf("Total elapsed Time:%lf (ms)\n", et-st);
    // free(thread_time);
    // free(thread_point);
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
    XStoreName(display, window, "Mandelbrot Set (OpenMP_Dynamic)");

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
