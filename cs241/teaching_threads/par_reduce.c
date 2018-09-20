/**
* Teaching Threads Lab
* CS 241 - Spring 2018
*/

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "reduce.h"
#include "reducers.h"

#define min(a,b) ((a < b) ? a : b)
#define max(a,b) ((a > b) ? a : b)
/* You might need a struct for each task ... */
typedef struct _thread {
   int start;
   int end;
   int idx;
   reducer reduce_func;
}thread;

static int* array;
static int arr[1024];
static int base;
/* You should create a start routine for your threads. */

void* busy(void *ptr) {
   thread* p = (thread*)ptr;
   int result = base;
   for(int i = p->start; i < p->end;i++)
      result = p->reduce_func(result, array[i]);
   arr[p->idx] = result;
   pthread_exit(0);
}


int par_reduce(int *list, size_t list_len, reducer reduce_func, int base_case,
               size_t num_threads) {
    /* Your implementation goes here */
   array = list;
   base = base_case;
   pthread_t* list_t[num_threads];
   //thread* td = calloc(num_threads, sizeof(thread));
   thread td[num_threads];
   int interval = max(list_len / num_threads, 1);
   size_t size = min(num_threads, list_len);
   for(size_t t = 0; t < size; t++) {
      pthread_t pt;
      td[t].start = min(t * interval, list_len);
      td[t].end = min((t + 1)*interval, list_len);
      td[t].reduce_func = reduce_func;
      list_t[t] = &pt;
      td[t].idx = t;
      pthread_create(list_t[t], NULL, busy, &td[t]);
   }

   int result = base_case;

   for(size_t t = 0; t < size; t++) {
      pthread_join(*list_t[t], NULL);
      result = reduce_func(result, arr[t]);
   }

   for(size_t t = num_threads * interval; t < list_len; t++) {
      result = reduce_func(result, list[t]);
   }

   return result;
}

