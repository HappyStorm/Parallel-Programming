#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <pthread.h>
#include <unistd.h>
using namespace std;
using namespace std::chrono;

class p_thread{
	public:
		int ID, capacity, wander_time;
		p_thread(int index, int C){
			ID = index, capacity = C;
		}
};
class c_thread{
	public:
		int capacity, track_time, total_track;
		c_thread(int C, int T, int N){
			capacity = C, track_time = T, total_track = N;
		}
};

high_resolution_clock::time_point start;
milliseconds avg_wait_time;
int n, C, T, N, onticket, finish, systime;
pthread_mutex_t mutex_onticket, mutex_output, mutex_one;
pthread_cond_t boarding, riding;
vector<int> onboard;

void Init();
void *PassengerFunction(void* p_thread);
void *CarFunction(void* c_thread);
void TakeRide(int);
milliseconds calculateTime(high_resolution_clock::time_point);


int main(int argc, char const *argv[])
{
	n = atoi(argv[1]);
	C = atoi(argv[2]);
	T = atoi(argv[3]);
	N = atoi(argv[4]);

	// Initial condition variable & mutex
	Init();

	// Initial on-board array & on-board counter
	// onboard = (int*) malloc(C * sizeof(int));
	finish = 0;
	onticket = C;

	// Initial srand() & time
	srand(time(NULL));
	start = high_resolution_clock::now();
	systime = 0;

	// Initial car thread
	pthread_t car;
	c_thread *car_thread = new c_thread(C, T, N);

	// Run the car thread
	pthread_create(&car, NULL, CarFunction, (void *) car_thread);

	// Initial passenger thread
	pthread_t *thread_arr = (pthread_t*) malloc(n * sizeof(pthread_t));
	p_thread *p_arr = (p_thread*) malloc(n * sizeof(p_thread));
	for(int i=0, j=1; i<n; ++i, ++j){
		p_arr[i] = *(new p_thread(j, C));
		pthread_create(&thread_arr[i], NULL, PassengerFunction, (void *)&p_arr[i]);
	}
	pthread_join(car, NULL);

	free(thread_arr);
    free(p_arr);
    pthread_mutex_destroy(&mutex_onticket);
    pthread_mutex_destroy(&mutex_output);
    pthread_mutex_destroy(&mutex_one);    
    pthread_exit(NULL);
	return 0;
}


void Init()
{
	pthread_mutex_init(&mutex_onticket, NULL);
	pthread_mutex_init(&mutex_output, NULL);
	pthread_mutex_init(&mutex_one, NULL);
	pthread_cond_init(&boarding, NULL);
	pthread_cond_init(&riding, NULL);
}

void *PassengerFunction(void *data)
{
	p_thread *p = (p_thread*) data;
	while(1){
		p->wander_time = (rand()%100+1)*1000;
		
		pthread_mutex_lock(&mutex_output);
		printf("No.%d passenger wanders around the park.(%dms)\n", p->ID, p->wander_time/1000);
		pthread_mutex_unlock(&mutex_output);
		
		usleep(p->wander_time);

		pthread_mutex_lock(&mutex_output);
		printf("No.%d passenger returns for another ride.\n", p->ID);
		pthread_mutex_unlock(&mutex_output);

		TakeRide(p->ID);
		if(finish) break;
	}
	pthread_exit(NULL);
}

void TakeRide(int p_ID)
{
	pthread_mutex_lock(&mutex_onticket);

	if(finish){
		pthread_mutex_unlock(&mutex_onticket);
		pthread_exit(NULL);
	}

	onboard.push_back(p_ID);
	while(onticket==0){
		auto tw = high_resolution_clock::now();
		pthread_cond_wait(&riding, &mutex_onticket);
		avg_wait_time += calculateTime(tw);
	}

	if(finish){
		pthread_mutex_unlock(&mutex_onticket);
		pthread_exit(NULL);
	}

	pthread_mutex_lock(&mutex_one);
	onticket--;
	if(onticket==0)
		pthread_cond_signal(&boarding);
	pthread_mutex_unlock(&mutex_onticket);
	pthread_mutex_unlock(&mutex_one);

	pthread_mutex_lock(&mutex_onticket);
	pthread_cond_wait(&riding, &mutex_onticket);
	pthread_mutex_unlock(&mutex_onticket);
}

void *CarFunction(void *data)
{
	c_thread *c = (c_thread*) data;

	for(int i=0; i<c->total_track; ++i){ 
		pthread_mutex_lock(&mutex_onticket);
		while(onticket!=0)
			pthread_cond_wait(&boarding, &mutex_onticket);

		int s = calculateTime(start).count();
		usleep(c->track_time*1000);
		int e = floor(calculateTime(start).count());

		cout << "Car departures at " << s << " millisec.";

		for(int j=0; j<c->capacity; ++j){
			if(j!=((c->capacity)-1))
				printf(" No.%d,", onboard[j]);
			else
				printf(" No.%d passengers are in the car.\n", onboard[j]);
		}


		if(s-e!=c->track_time)
			cout << "Car departures at " << s+c->track_time << " millisec.";	
		else
		cout << "Car arrives at " << e << " millisec.";

		pthread_mutex_unlock(&mutex_onticket);

		pthread_mutex_lock(&mutex_onticket);
		
		for(int j=0; j<c->capacity; ++j){
			if(j!=((c->capacity)-1))
				printf(" No.%d,", onboard[j]);
			else
				printf(" No.%d passengers get off.\n", onboard[j]);
		}

		onboard.erase(onboard.begin(), onboard.begin()+(c->capacity));
		onticket = c->capacity;
    	pthread_cond_broadcast(&riding);   
		pthread_mutex_unlock(&mutex_onticket);
	}
	finish = 1;

	// FILE *output = fopen("srccTest.txt", "a");
	// fprintf(output, "Average wait time = %d(ms)\n", avg_wait_time.count());
	// fclose(output);
	pthread_exit(NULL);
}

milliseconds calculateTime(high_resolution_clock::time_point s)
{
    auto t = duration_cast<milliseconds>(high_resolution_clock::now() - s);
    return t;
}
