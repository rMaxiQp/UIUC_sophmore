/**
* Critical Concurrency Lab
* CS 241 - Spring 2018
*/
#include <stdio.h>
#include "barrier.h"

// The returns are just for errors if you want to check for them.
int barrier_destroy(barrier_t *barrier) {
    pthread_mutex_destroy(&barrier->mtx);
    pthread_cond_destroy(&barrier->cv);
    return 0;
}

int barrier_init(barrier_t *barrier, unsigned int num_threads) {
    pthread_mutex_init(&barrier->mtx, NULL);
    pthread_cond_init(&barrier->cv, NULL);
    barrier->n_threads = num_threads;
    barrier->times_used = 0;
    barrier->count = 0;
    return 0;
}

int barrier_wait(barrier_t *barrier) {
   pthread_mutex_lock(&barrier->mtx);
   //fprintf(stderr, "thread:%d\n", barrier->n_threads);
   if(++barrier->count == barrier->n_threads) {
      barrier->times_used++;
      barrier->count = 0;
      pthread_cond_broadcast(&barrier->cv);
   }
   //fprintf(stderr, "count:%d\n", barrier->count);
   unsigned int t = barrier->times_used;
   while(barrier->count % barrier->n_threads && t == barrier->times_used)
      pthread_cond_wait(&barrier->cv, &barrier->mtx);
   //fprintf(stderr, "exit\n");
   pthread_mutex_unlock(&barrier->mtx);
   return 0;
}
