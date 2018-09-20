/**
* Savvy_scheduler Lab
* CS 241 - Spring 2018
*/

#include "libpriqueue/libpriqueue.h"
#include "libscheduler.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct _job_info {
    int id;
    int priority;
    double arrival;
    double runtime;
    double start;
    double remain;
    double last_time;
    /* Add whatever other bookkeeping you need into this struct. */
} job_info;

double turnaround = 0.0;//endtime - arrival
double waiting = 0.0;//(endtime - arrival) - runtime
double response = 0.0;// start_time - arrival_time;
int count = 0;

priqueue_t pqueue;
scheme_t pqueue_scheme;
comparer_t comparision_func;

void scheduler_start_up(scheme_t s) {
    switch (s) {
    case FCFS:
        comparision_func = comparer_fcfs;
        break;
    case PRI:
        comparision_func = comparer_pri;
        break;
    case PPRI:
        comparision_func = comparer_ppri;
        break;
    case PSRTF:
        comparision_func = comparer_psrtf;
        break;
    case RR:
        comparision_func = comparer_rr;
        break;
    case SJF:
        comparision_func = comparer_sjf;
        break;
    default:
        printf("Did not recognize scheme\n");
        exit(1);
    }
    priqueue_init(&pqueue, comparision_func);
    pqueue_scheme = s;
    // Put any set up code you may need here
}

static int break_tie(const void *a, const void *b) {
    return comparer_fcfs(a, b);
}

int comparer_fcfs(const void *a, const void *b) {
   job_info *a_job = ((job *) a)->metadata;
   job_info *b_job = ((job *) b)->metadata;
   if( a_job->arrival > b_job->arrival)
      return 1;
   if( a_job->arrival < b_job->arrival)
      return -1;
   return 0;
}

int comparer_ppri(const void *a, const void *b) {
    // Complete as is
    return comparer_pri(a, b);
}

int comparer_pri(const void *a, const void *b) {
   job_info *a_job = ((job *) a)->metadata;
   job_info *b_job = ((job *) b)->metadata;
   if( a_job->priority > b_job->priority)
      return 1;
   if( a_job->priority < b_job->priority)
      return -1;
   return break_tie(a, b);
}

int comparer_psrtf(const void *a, const void *b) {
   job_info *a_job = ((job *) a)->metadata;
   job_info *b_job = ((job *) b)->metadata;
   if( a_job->remain > b_job->remain)
      return 1;
   if( a_job->remain < b_job->remain)
      return -1;
   return break_tie(a, b);
}

int comparer_rr(const void *a, const void *b) {
   job_info *a_job = ((job *) a)->metadata;
   job_info *b_job = ((job *) b)->metadata;
   //fprintf(stderr ,"a: %lf, b: %lf\n", a_job->last_time, b_job->last_time);
   if( a_job->last_time > b_job->last_time)
      return 1;
   if( a_job->last_time < b_job->last_time)
      return -1;
   return break_tie(a, b);
}

int comparer_sjf(const void *a, const void *b) {
   job_info *a_job = ((job *) a)->metadata;
   job_info *b_job = ((job *) b)->metadata;
   if( a_job->runtime > b_job->runtime)
      return 1;
   if( a_job->runtime < b_job->runtime)
      return -1;
   return break_tie(a, b);
}

// Do not allocate stack space or initialize ctx. These will be overwritten by
// gtgo
void scheduler_new_job(job *newjob, int job_number, double time,
                       scheduler_info *sched_data) {
    // TODO complete me!
    job_info * info = malloc(sizeof(job_info));
    newjob->metadata = info;
    info->arrival = time;
    info->id = job_number;
    info->priority = sched_data->priority;
    info->runtime = sched_data->running_time;
    info->start = 0.0;
    info->remain = sched_data->running_time;
    info->last_time = 0.0;
    priqueue_offer(&pqueue, (void *)newjob);
}

job *scheduler_quantum_expired(job *job_evicted, double time) {
   // TODO complete me!
   if(pqueue_scheme != RR && pqueue_scheme != PPRI && pqueue_scheme != PSRTF && job_evicted)//not premptive && job_evicted != NULL
      return job_evicted;

   //premptive || RR || NULL
   if(job_evicted) {//premptive || RR
      job_info *job_i = (job_info *) job_evicted->metadata;
      job_i->remain -= time - job_i->start;
      job_i->last_time = time;
      priqueue_offer(&pqueue, (void *)job_evicted);
   }

   job *next_elem = priqueue_poll(&pqueue);
   if(NULL == next_elem)//NO OUTPUT
      return NULL;

   job_info *info = (job_info *) next_elem->metadata;//WITH OUTPUT
   if(info->start == 0.0) {//new Process ==> response
      info->start = time;
      response += time - info->arrival;
   }
   //info->last_time = time;
   return next_elem;
}

void scheduler_job_finished(job *job_done, double time) {
   // TODO complete me!
   if(NULL == job_done)
      return;
   count++;
   job_info *info = (job_info *) job_done->metadata;
   turnaround += time - info->arrival;
   waiting += time - info->arrival - info->runtime;
   free(info);
}

static void print_stats() {
    fprintf(stderr, "turnaround     %f\n", scheduler_average_turnaround_time());
    fprintf(stderr, "total_waiting  %f\n", scheduler_average_waiting_time());
    fprintf(stderr, "total_response %f\n", scheduler_average_response_time());
}

double scheduler_average_waiting_time() {
    // TODO complete me!
    if(0 == count)
       return 0.0;
    return (double) (waiting / count);
}

double scheduler_average_turnaround_time() {
    // TODO complete me!
    if(0 == count)
       return 0.0;
    return (double) (turnaround / count);
}

double scheduler_average_response_time() {
    // TODO complete me!
   if(0 == count)
      return 0.0;
   return (double) (response / count);
}

void scheduler_show_queue() {
    // Implement this if you need it!
    job *node = priqueue_poll(&pqueue);
    if(node != NULL) {
       job_info *node_info = (job_info *)(node->metadata);
       fprintf(stderr, "arrival: %lf, id: %d, priority: %d, runtime: %lf \n", node_info->arrival, node_info->id, node_info->priority, node_info->runtime);
       scheduler_show_queue();
       priqueue_offer(&pqueue, node);
    }
}

void scheduler_clean_up() {
    priqueue_destroy(&pqueue);
    print_stats();
}
