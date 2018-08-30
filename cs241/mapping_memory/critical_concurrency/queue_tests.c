/**
* Critical Concurrency Lab
* CS 241 - Spring 2018
*/

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "queue.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("usage: %s test_number\n", argv[0]);
        exit(1);
    }
    printf("Please write tests cases\n");
    queue *q = queue_create(10);
    //fprintf(stderr, "before push\n");
    for(int i = 0; i < 10; i++)
       queue_push(q, (void*)&i);
    //fprintf(stderr, "after push\n");
    for(int i = 0; i < 20; i++) {
       fprintf(stderr, "%d\n", *(int *)queue_pull(q));
      // fprintf(stderr, "aa\n");
    }
    fprintf(stderr, "end\n");
    return 0;
}
