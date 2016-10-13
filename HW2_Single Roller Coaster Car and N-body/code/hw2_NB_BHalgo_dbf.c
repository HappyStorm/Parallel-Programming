#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <omp.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
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

/* program variables */
typedef enum {
    NE, SE, SW, NW
} Quadrant;

typedef enum {
    Internal, External
} Type;

typedef struct Body{
    double x, y;
    double vx, vy;
} Body;

typedef struct Force{
    double sum_fx, sum_fy;
} Force;

typedef struct QuadTree
{
    int num_body;
    double CM_x, CM_y,
           mid_x, mid_y,
           ll_x, ll_y,
           ru_x, ru_y,
           sum_mass,
           r_width,
           theta;
    Body *childbody;
    struct QuadTree *childtree;
    Type node_type;
} QTree;

const double G = 6.67384e-11; // Gravitational constant
const int QUAD = 4;
int num_thread,
    num_step,
    num_body,
    Xwindow,
    finish = 0,
    left_job = 0,
    num_done = 0;
double  mass,
        step_time,
        theta,
        Gm,
        start_time,
        end_time,
        io_time=0, io_stime, io_etime,
        buildtree_time=0, buildtree_stime, buildtree_etime,
        compute_time=0, compute_stime, compute_etime,
        total_time,
        tmin_x = 0, tmin_y = 0,
        tmax_x = 0, tmax_y = 0;
FILE *input, *output;
Body *Nbody, *new_body;
QTree *root;
pthread_mutex_t mutex_waiting;
pthread_cond_t dealing, step_done;

/* Xwindow variables */
Window win;
Display *display;
GC gc;
XWindowAttributes wa;
Pixmap d_backBuf;
XEvent e;
int screen,
    window_len,
    black,
    white;
double  xmin, ymin,
        coor_len,
        ratio;

void Init(int, char* []);
void InitWindow();
void moveBody(int, Force*);
void* compute(void *data);
void force_compute();
void draw_back();
void draw_points();
void draw_lines_dark(double, double, double, double);
void draw_lines_light(double, double, double, double);
void init_QTree(QTree*, double, double, double, double);
void setQuadrant(QTree*);
void insertNode(QTree*, Body*);
void force_compute(QTree*, Body*, Force*);
void buildQTree(QTree*);
void force_value(double, double, Body*, double, Force*);
Quadrant getQuardant(QTree*, Body*);

int main(int argc, char* argv[]){

    start_time = omp_get_wtime();

    int i, j, k;
    // Initialize environment(Xwindow)
    Init(argc, argv);
    fscanf(input, "%d", &num_body);

    // N-Body variables
    Nbody = (Body*) malloc(num_body * sizeof(Body));
    new_body = (Body*) malloc(num_body * sizeof(Body));
    root = (QTree*) malloc(1 * sizeof(QTree));
    root->node_type = External, root->num_body = 0;
    root->childbody = NULL;
    pthread_t threads[num_thread];

    // Initialize N-body
    io_stime = omp_get_wtime();
    for(i=0; i<num_body; ++i)
        fscanf(input, "%lf %lf %lf %lf",
                &Nbody[i].x, &Nbody[i].y, &Nbody[i].vx, &Nbody[i].vy);
    fclose(input);
    io_etime = omp_get_wtime();
    io_time += io_etime - io_stime;

    for(i=0; i<num_thread; ++i)
        pthread_create(&threads[i], NULL, compute, NULL);

    // Start simulation 
    for(i=0; i<num_step; ++i){
        if(Xwindow)
            draw_back();
        buildtree_stime = omp_get_wtime();
        buildQTree(root);
        buildtree_etime = omp_get_wtime();
        buildtree_time += buildtree_etime - buildtree_stime;

        if(Xwindow){
            draw_points();
            if(XCheckWindowEvent(display, win, ExposureMask, &e)){
                while(XCheckWindowEvent(display, win, ExposureMask, &e));
                XCopyArea(display, d_backBuf, win, gc, 0, 0, wa.width, wa.height, 0, 0);
            }
            usleep(10);
            // XClearWindow(display, win); 
        }
        pthread_mutex_lock(&mutex_waiting);
        left_job = num_body, num_done = 0;
        pthread_cond_broadcast(&dealing);
        pthread_cond_wait(&step_done, &mutex_waiting);
        pthread_mutex_unlock(&mutex_waiting);
        Body *temp = new_body;
        new_body = Nbody; Nbody = temp;
    }
    finish = 1;
    pthread_mutex_lock(&mutex_waiting);
    pthread_cond_broadcast(&dealing);
    pthread_mutex_unlock(&mutex_waiting);
    for(k=0; k<num_thread; ++k) 
        pthread_join(threads[k], NULL);

    end_time = omp_get_wtime();
    total_time = end_time - start_time;

    io_stime = omp_get_wtime();
    printf("Total run time = %lf\n", total_time);
    printf("Total build_tree time = %lf\n", buildtree_time);
    printf("Total compute time = %lf\n", compute_time);
    io_etime = omp_get_wtime();
    io_time += io_etime - io_stime;
    printf("Total I/O time = %lf\n", io_time);

    free(Nbody);
    free(new_body);
    free(root);
    pthread_mutex_destroy(&mutex_waiting);
    pthread_exit(NULL);
    return 0;
}

void Init(int argc, char* argv[])
{
    //  ./hw2_NB_{version} #threads m T t FILE 𝜃 enable/disable 𝑥𝑚𝑖𝑛 𝑦𝑚𝑖𝑛 length Length
    io_stime = omp_get_wtime();

    num_thread = atoi(argv[1]);
    mass = strtod(argv[2], NULL), Gm = G * mass;
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
    io_etime = omp_get_wtime();
    io_time += io_etime - io_stime;

    pthread_mutex_init(&mutex_waiting, NULL);
    pthread_cond_init(&dealing, NULL);
    pthread_cond_init(&step_done, NULL);
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
    XGetWindowAttributes(display, win, &wa);
    XStoreName(display, win, window_name);

    // create graph 
    XGCValues values;
    long valuemask = 0;
    gc = XCreateGC(display, win, valuemask, &values);
    XSetLineAttributes(display, gc, 1, LineSolid, CapRound, JoinRound);
    XMapWindow(display, win);

    d_backBuf = XCreatePixmap(display, win, wa.width, wa.height, wa.depth);
    XSetForeground(display, gc, BlackPixelOfScreen(DefaultScreenOfDisplay(display)));
    XFillRectangle(display, d_backBuf, gc, 0, 0, wa.width, wa.height);
    XCopyArea(display, d_backBuf, win, gc, 0, 0, wa.width, wa.height, 0, 0);
    XSelectInput(display, win, ExposureMask);
}

void* compute(void *data)
{

    while(1){
        pthread_mutex_lock(&mutex_waiting);
        while(!finish && left_job <= 0)
            pthread_cond_wait(&dealing, &mutex_waiting);
        int index = --left_job;
        pthread_mutex_unlock(&mutex_waiting);
        
        if(finish) 
            break;
        Force *tforce = malloc(1 * sizeof(Force));
        tforce->sum_fy = 0.0, tforce->sum_fy = 0.0;

        compute_stime = omp_get_wtime();
        force_compute(root, &Nbody[index], tforce);
        compute_etime = omp_get_wtime();
        compute_time = compute_etime - compute_stime;


        moveBody(index, tforce);
        
        pthread_mutex_lock(&mutex_waiting);
        ++num_done;
        free(tforce);
        if(num_done >= num_body)
            pthread_cond_signal(&step_done);
        pthread_mutex_unlock(&mutex_waiting);
    }
}

void moveBody(int n, Force *f)
{
    Body *oldb = &Nbody[n], *newb = &new_body[n];
    // Update
    newb->vx = oldb->vx + f->sum_fx * step_time / mass;
    newb->vy = oldb->vy + f->sum_fy * step_time / mass;
    newb->x = oldb->x + newb->vx * step_time;
    newb->y = oldb->y + newb->vy * step_time;
}

void draw_back()
{
    XSetForeground(display, gc, black);
    XFillRectangle(display, d_backBuf, gc, 0, 0, window_len, window_len);
}

void draw_points()
{
    int i;
    XSetForeground(display, gc, white);
    for(i=0; i<num_body; ++i)
        XDrawPoint(display, d_backBuf, gc, (Nbody[i].x - xmin) * ratio, (Nbody[i].y - ymin) * ratio);
}

void draw_lines_dark(double sx, double sy, double ex, double ey)
{
    XSetForeground(display, gc, 0x2A2A2A);
    XDrawLine(display, d_backBuf, gc, (sx-xmin) * ratio, (sy-ymin) * ratio, 
        (ex-xmin) * ratio, (ey-ymin) * ratio);
}

void draw_lines_light(double sx, double sy, double ex, double ey)
{
    XSetForeground(display, gc, 0x3A3A3A);
    XDrawLine(display, d_backBuf, gc, (sx-xmin) * ratio, (sy-ymin) * ratio, 
        (ex-xmin) * ratio, (ey-ymin) * ratio);
}

void buildQTree(QTree *QT)
{
    int i;
    tmin_x = 0, tmin_y = 0, tmax_x = 0, tmax_y = 0;
    for(i=0; i<num_body; ++i){
        tmin_x = MIN(tmin_x, Nbody[i].x);
        tmin_y = MIN(tmin_y, Nbody[i].y);
        tmax_x = MAX(tmax_x, Nbody[i].x);
        tmax_y = MAX(tmax_y, Nbody[i].y);
    }
    if(Xwindow){
        draw_lines_light(tmin_x, tmin_y, tmin_x, tmax_y);
        draw_lines_light(tmin_x, tmax_y, tmax_x, tmax_y);
        draw_lines_light(tmax_x, tmax_y, tmax_x, tmin_y);
        draw_lines_light(tmax_x, tmin_y, tmin_x, tmin_y);
    }
    init_QTree(QT, tmin_x, tmin_y, tmax_x, tmax_y);
    for(i=0; i<num_body; ++i)
        insertNode(QT, &Nbody[i]);        
}

void init_QTree(QTree *QT, double lx, double ly, double rx, double ry)
{
    QT->num_body = 0;
    QT->CM_x = 0, QT->CM_y = 0;
    QT->mid_x = (lx+rx)/2, QT->mid_y = (ly+ry)/2;
    QT->ll_x = lx, QT->ll_y = ly;
    QT->ru_x = rx, QT->ru_y = ry;
    QT->sum_mass = 0, QT->r_width = fabs(ly-ry), QT->theta = theta;
    QT->childbody = NULL, QT->childtree = NULL, QT->node_type = External;
}

void insertNode(QTree *QT, Body *b)
{
    if(QT->node_type == Internal)
        insertNode(&QT->childtree[getQuardant(QT, b)], b);
    else if(QT->node_type == External && QT->num_body == 1){
        setQuadrant(QT);
        QT->node_type = Internal;
        insertNode(&QT->childtree[getQuardant(QT, QT->childbody)], QT->childbody);
        insertNode(&QT->childtree[getQuardant(QT, b)], b);
    }
    else
        QT->childbody = b;
    double tx = QT->sum_mass * QT->CM_x + mass * b->x,
           ty = QT->sum_mass * QT->CM_y + mass * b->y;
    QT->sum_mass += mass;
    QT->CM_x = tx / QT->sum_mass, QT->CM_y = ty / QT->sum_mass;
    QT->num_body = QT->num_body + 1;
}

void setQuadrant(QTree *QT)
{
    if(Xwindow){
        draw_lines_dark(QT->ll_x, QT->mid_y, QT->ru_x, QT->mid_y);
        draw_lines_dark(QT->mid_x, QT->ru_y, QT->mid_x, QT->ll_y);
    }
    QT->childtree = (QTree*) malloc(4 * sizeof(QTree));
    init_QTree(&QT->childtree[NE], QT->mid_x, QT->mid_y, QT->ru_x,  QT->ru_y);
    init_QTree(&QT->childtree[SE], QT->mid_x, QT->ll_y,  QT->ru_x,  QT->mid_y);
    init_QTree(&QT->childtree[SW], QT->ll_x,  QT->ll_y,  QT->mid_x, QT->mid_y);
    init_QTree(&QT->childtree[NW], QT->ll_x,  QT->mid_y, QT->mid_x, QT->ru_y);
}

Quadrant getQuardant(QTree *QT, Body* b)
{
    double tx = b->x, ty = b->y,
           midx = QT->mid_x, midy = QT->mid_y;
    if(tx >= midx && ty >= midy) return NE;
    else if(tx >= midx && ty < midy) return SE;
    else if(tx < midx && ty < midy) return SW;
    else return NW;
}

void force_compute(QTree *QT, Body *b, Force *tf)
{
    if(QT->node_type == External){
        if(QT->childbody == b) return;
        force_value(QT->childbody->x, QT->childbody->y, b, mass, tf);
    }
    else if((QT->r_width/sqrt(SQUARE(QT->CM_x-b->x) + SQUARE(QT->CM_y-b->y))) < theta){
        force_value(QT->CM_x, QT->CM_y, b, QT->sum_mass, tf);
    }
    else{
        int i;
        for(i=0; i<4; ++i){
            if(QT->childtree[i].num_body < 1)
                continue;
            force_compute(&QT->childtree[i], b, tf);
        }
    }
}

void force_value(double cm_x, double cm_y, Body *b, double M, Force *tf)
{
    double delta_x = cm_x - b->x, 
           delta_y = cm_y - b->y,
           dis_cube = CUBE(sqrt(pow(delta_x, 2) + pow(delta_y, 2))) + 10e-7,
           GMm = Gm * M / dis_cube;
    tf->sum_fx += GMm * delta_x, tf->sum_fy += GMm * delta_y;
}