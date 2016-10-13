#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#define CUBE(x) (x)*(x)*(x)

/* program variables */
typedef struct Body{
    double x, y;
    double vx, vy;
} Body;
const double G = 6.67384e-11; // Gravitational constant
int num_thread,
    num_step,
    num_body,
    Xwindow;
double  mass,
        step_time,
        *vx_new,
        *vy_new,
        theta,
        Gmm,
        start_time,
        end_time,
        total_time;
FILE *input, *output;
Body *Nbody;

/* Xwindow variables */
Window win;
Display *display;
GC gc;
int screen,
    window_len,
    black,
    white;
double  xmin, ymin,
        coor_len,
        ratio;

void Init(int, char* []);
void InitWindow();
void moveBody(int);
void draw_points();

int main(int argc, char* argv[]){

    start_time = omp_get_wtime();

    int i, j, k;
    // Initialize environment(Xwindow)
    Init(argc, argv);
    fscanf(input, "%d", &num_body);

    // Initialize OpenMP
    omp_set_num_threads(num_thread);

    // N-Body variables
    Nbody = (Body*) malloc(num_body * sizeof(Body));
    vx_new = (double*) malloc(num_body * sizeof(double));
    vy_new = (double*) malloc(num_body * sizeof(double));

    int task_per_thread = num_body / num_thread / 10;
    // Initialize N-body
    for(i=0; i<num_body; ++i)
        fscanf(input, "%lf %lf %lf %lf",
                &Nbody[i].x, &Nbody[i].y, &Nbody[i].vx, &Nbody[i].vy);
    fclose(input);

    // Start simulation 
    for(i=0; i<num_step; ++i){
        if(Xwindow) draw_points();
        #pragma omp parallel for schedule(dynamic, task_per_thread)
        for(j=0; j<num_body; j++)
            moveBody(j);
    }
    end_time = omp_get_wtime();
    total_time = end_time - start_time;
    printf("Total run time = %lf\n", total_time);
    free(Nbody);
    free(vx_new);
    free(vy_new);
    return 0;
}

void Init(int argc, char* argv[])
{
    //  ./hw2_NB_{version} #threads m T t FILE 𝜃 enable/disable 𝑥𝑚𝑖𝑛 𝑦𝑚𝑖𝑛 length Length
    num_thread = atoi(argv[1]);
    mass = strtod(argv[2], NULL), Gmm = G * mass * mass;
    num_step = atoi(argv[3]);
    step_time = strtod(argv[4], NULL);
    input = fopen(argv[5], "r");
    theta = strtod(argv[6], NULL);
    Xwindow = (strcmp(argv[7], "enable")==0) ? 1 : 0;
    if(Xwindow){
        xmin = strtod(argv[8], NULL), ymin = strtod(argv[9], NULL);
        coor_len = strtod(argv[10], NULL), window_len = atoi(argv[11]);
        ratio = (double) window_len / coor_len;
        InitWindow();
    }
}

void InitWindow()
{
    char *window_name="N-Body Simulation(Sequential)", *display_name=NULL;
    if((display=XOpenDisplay(display_name))==NULL){
        fprintf(stderr, "Cannot connect to X server %s\n",
                XDisplayName (display_name) );
        exit(1);
    }
    screen = DefaultScreen(display);
    // set window position 
    int x = 0;
    int y = 0;

    // border width in pixels 
    int border_width = 0;

    black = BlackPixel(display, screen);
    white = WhitePixel(display, screen);

    // create window
    win = XCreateSimpleWindow(display, RootWindow(display, screen),
        x, y, window_len, window_len, border_width, black, white);
    XStoreName(display, win, window_name);

    // create graph 
    XGCValues values;
    long valuemask = 0;
    gc = XCreateGC(display, win, valuemask, &values);
    XSetLineAttributes(display, gc, 1, LineSolid, CapRound, JoinRound);

    XMapWindow(display, win);
}

void moveBody(int n)
{
    Body *old = &Nbody[n], temp;
    int i;
    double sum_fx = 0, sum_fy = 0;
    for(i=0; i<num_body; ++i) {
        if(i==n) continue;
        temp = Nbody[i];
        double delta_x = temp.x - old->x, 
               delta_y = temp.y - old->y,
               dis_cube = CUBE(sqrt(pow(delta_x, 2) + pow(delta_y, 2))) + 10e-7;
        if(dis_cube==0) continue;
        sum_fx += Gmm * delta_x / (dis_cube);
        sum_fy += Gmm * delta_y / (dis_cube);
    }
    // Update
    vx_new[n] = old->vx + sum_fx * step_time / mass;
    vy_new[n] = old->vy + sum_fy * step_time / mass;
    old->x = old->x + vx_new[n] * step_time;
    old->y = old->y + vy_new[n] * step_time;
    old->vx = vx_new[n];
    old->vy = vy_new[n];
}

void draw_points()
{
    int i;
    XClearWindow(display, win);
    XSetForeground(display, gc, black);
    XFillRectangle(display, win, gc, 0, 0, window_len, window_len);
    XSetForeground(display, gc, white);
    for(i=0; i<num_body; ++i)
        XDrawPoint(display, win, gc, (Nbody[i].x - xmin) * ratio, (Nbody[i].y - ymin) * ratio);
    XFlush(display);
}