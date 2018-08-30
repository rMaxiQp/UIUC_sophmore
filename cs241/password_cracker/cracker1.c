/**
* Password Cracker Lab
* CS 241 - Spring 2018
*/

#include "cracker1.h"
#include "libs/format.h"
#include "libs/utils.h"
#include "libs/queue.h"

#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <crypt.h>

static queue* q;
static int ff = 0;
static int success = 0;
static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;

typedef struct _crack {
   pthread_t tid;
   //struct crypt_data cdata;
   size_t idx;
}crack;

void* foo(void* ptr) {
   double time = getThreadCPUTime();
   crack *p = (crack*)ptr;

   queue_push(q, "\0");
   char *original = queue_pull(q);

   if(strcmp(original, "\0") == 0)
      return NULL;

   int fail = 0;
  int count = 0;

   char *saveptr = NULL;
   char *usr = strtok_r(original, " ", &saveptr);
   char *hash = strtok_r(NULL, " ", &saveptr);
   char *password = strtok_r(NULL, "\n", &saveptr);
   v1_print_thread_start(p->idx, usr);

   struct crypt_data cdata;
   cdata.initialized = 0;
   int s = getPrefixLength(password);
   int t = strlen(password);
   char bit = password[s-1];

   for(int i = s; i < t; i++)
      password[i] = 'a';

   //fprintf(stderr, "pass: %s, hash: %s, usr: %s\n", password, hash, usr);
   while(1){

      count++;
      if(strcmp(hash,crypt_r(password, "xx", &cdata)) == 0) {
         //fprintf(st
         break;
      }

      //count ++;
      if(!incrementString(password) || bit != password[s-1]) {
         fail = 1;
         break;
      }

      //count ++;
   }

   time = getThreadCPUTime() - time;
   v1_print_thread_result(p->idx, usr, password, count, time, fail);

   pthread_mutex_lock(&m);
   if(fail)
      ff++;
   else
      success++;
   pthread_mutex_unlock(&m);

   free(original);
   return foo(ptr);
}


int start(size_t thread_count) {
   // TODO your code here, make sure to use thread_count!
   // Remember to ONLY crack passwords in other threads

   char * buffer = NULL;
   size_t size = 0;
   q = queue_create(-1);
   //size_t tot = 0;

   while(getline(&buffer, &size, stdin) != -1) {
      queue_push(q, strdup(buffer));
      //tot ++;
   }

   crack* thread_arr = malloc(thread_count * sizeof(crack));
   for(size_t i = 0; i < thread_count; i ++) {
      thread_arr[i].idx = i + 1;
      //thread_arr[i].cdata.initialized = 0;
      pthread_create(&thread_arr[i].tid, NULL, foo, &thread_arr[i]);
   }

   for(size_t i = 0; i < thread_count; i ++)
      pthread_join(thread_arr[i].tid, NULL);

   free(thread_arr);
   v1_print_summary(success, ff);
   queue_destroy(q);
   free(buffer);
   return 0;
   // DO NOT change the return code since AG uses it to check if your
   // program exited normally
}
