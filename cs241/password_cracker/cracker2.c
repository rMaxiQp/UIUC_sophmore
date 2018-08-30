/**
* Password Cracker Lab
* CS 241 - Spring 2018
*/

#include "cracker2.h"
#include "libs/format.h"
#include "libs/utils.h"
#include "libs/thread_status.h"
#include "libs/queue.h"

#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <crypt.h>
#include <pthread.h>

static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
static pthread_barrier_t pbt;
static queue *q = NULL;
static int total = 0;
static char *hash = NULL;
static char *password = NULL;
static char *usr = NULL;
static int unknown = 0;
static int success = 1;
static int hash_count = 0;
static double CPU_TIME = 0.0;
static double elaspsedTime = 0.0;
static int offset = 0;
static char bit = '\0';
static int time_to_exit = 0;
static char * password_correct = NULL;

typedef struct _crack {
   pthread_t tid;
   struct crypt_data cdata;
   char * start;
   size_t idx;
} crack;

void *bar(void *ptr) {
   crack *p = (crack *)ptr;
   char * not_parsed = NULL;
   pthread_barrier_wait(&pbt);
   while(1) {
      pthread_mutex_lock(&m);
      if(p->idx == 1) { //one thread only...
         if(hash) { //skip for the first iteration
            v2_print_summary(usr, password_correct, hash_count, getTime() - elaspsedTime, getCPUTime() - CPU_TIME, success);
            hash_count = 0;
            CPU_TIME = getCPUTime();
            elaspsedTime = getTime();
            success = 1; //set the flag back to "fail"
            hash = NULL;
            free(not_parsed);
         }

         queue_push(q, "\0");
         not_parsed = queue_pull(q);
         if(!strcmp(not_parsed, "\0")) {
            time_to_exit = 1;
         }
         else {
            char *saveptr = NULL;
            usr = strtok_r(not_parsed, " ", &saveptr);
            hash = strtok_r(NULL, " ", &saveptr);
            password = strtok_r(NULL, "\n", &saveptr);
            size_t s = strlen(password);
            offset = getPrefixLength(password);
            unknown = s - offset;
            setStringPosition(password + offset, 0);
            bit = password[offset-1];
            v2_print_start_user(usr);
         }
      }
      pthread_mutex_unlock(&m);
      pthread_barrier_wait(&pbt);
      if(p->start)
         free(p->start);
      if(time_to_exit) {
         break;
      }
      //assign start and end
      long potential = 0, ending = 0;
      getSubrange(unknown, total, p->idx, &potential, &ending);
      p->start = strdup(password);
      setStringPosition(p->start + offset, potential);
      v2_print_thread_start(p->idx, usr, potential, p->start);


      int l = success;
      //loop
      int count = 0;
      while(1) {
         count++;
         pthread_mutex_lock(&m);
         l = success;
         pthread_mutex_unlock(&m);

         //pthread_mutex_lock(&m);
         if(!strcmp(hash, crypt_r(p->start, "xx", &p->cdata))) {
            pthread_mutex_lock(&m);
            success = 0;
            v2_print_thread_result(p->idx, count, 0);//success
            pthread_mutex_unlock(&m);
            password_correct = p->start;
            break;
         }
         //pthread_mutex_unlock(&m);

         if(!l) {
            pthread_mutex_lock(&m);
            v2_print_thread_result(p->idx, count, 1);//cancelled
            pthread_mutex_unlock(&m);
            break;
         }

         if(!incrementString(p->start) || p->start[offset-1] != bit) {
            pthread_mutex_lock(&m);
            v2_print_thread_result(p->idx, count, 2);
            pthread_mutex_unlock(&m);
            break;
         }
      }



      pthread_mutex_lock(&m);
      hash_count += count;
      pthread_mutex_unlock(&m);
      pthread_barrier_wait(&pbt);
   }

   return NULL;
}

int start(size_t thread_count) {
    // TODO your code here, make sure to use thread_count!
    // Remember to ONLY crack passwords in other threads
   elaspsedTime = getTime();
   CPU_TIME = getCPUTime();
   char *buffer = NULL;
   size_t size = 0;
   q = queue_create(-1);
   pthread_barrier_init(&pbt, NULL, thread_count);

   while(getline(&buffer, &size, stdin) != -1)
      queue_push(q, strdup(buffer));

   total = thread_count;
   crack threads [thread_count];
   for(size_t t = 0; t < thread_count; t++) {
      threads[t].idx = t + 1;
      threads[t].cdata.initialized = 0;
      threads[t].start = NULL;
      pthread_create(&threads[t].tid, NULL, bar, &threads[t]);
   }

   for(size_t t = 0; t < thread_count; t++)
      pthread_join(threads[t].tid, NULL);

   pthread_barrier_destroy(&pbt);
   queue_destroy(q);
   free(buffer);
   return 0; // DO NOT change the return code since AG uses it to check if your
             // program exited normally
}
