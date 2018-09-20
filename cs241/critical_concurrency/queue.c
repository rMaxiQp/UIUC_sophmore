/**
* Critical Concurrency Lab
* CS 241 - Spring 2018
*/

#include "queue.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * This queue is implemented with a linked list of queue_nodes.
 */
typedef struct queue_node {
    void *data;
    struct queue_node *next;
} queue_node;

struct queue {
    /* queue_node pointers to the head and tail of the queue */
    queue_node *head, *tail;

    /* The number of elements in the queue */
    ssize_t size;

    /**
     * The maximum number of elements the queue can hold.
     * max_size is non-positive if the queue does not have a max size.
     */
    ssize_t max_size;

    /* Mutex and Condition Variable for thread-safety */
    pthread_cond_t cv;
    pthread_mutex_t m;
};

queue *queue_create(ssize_t max_size) {
   /* Your code here */
   if (max_size == 0)
      return NULL;
   queue *retval = malloc(sizeof(queue));
   retval->head = NULL;
   retval->tail = NULL;
   retval->max_size = max_size;
   retval->size = 0;
   pthread_mutex_init(&retval->m, NULL);
   pthread_cond_init(&retval->cv, NULL);
   return retval;
}

void queue_destroy(queue *this) {
    /* Your code here */
   if(!this)
      return;
   queue_node *temp = this->head;
   while(this->size) {
      free(temp);
      temp = temp->next;
      this->size--;
   }

   pthread_cond_destroy(&this->cv);
   pthread_mutex_destroy(&this->m);
   free(this);
}

void queue_push(queue *this, void *data) {
    /* Your code here */
   pthread_mutex_lock(&this->m);
   while(this->max_size > 0 && this->max_size == this->size)
      pthread_cond_wait(&this->cv, &this->m);

   queue_node* qn = malloc(sizeof(queue_node));
   qn->data = data;
   qn->next = NULL;
   if(!this->size) {
      this->head = qn;
      this->tail = qn;
   }
   else {
      this->tail->next = qn;
      this->tail = qn;
   }
   this->size++;
   pthread_cond_signal(&this->cv);
   pthread_mutex_unlock(&this->m);
}

void *queue_pull(queue *this) {
    /* Your code here */
   pthread_mutex_lock(&this->m);
   //fprintf(stderr, "this->size:%zu\n", this->size);
   while(this->size == 0)
      pthread_cond_wait(&this->cv, &this->m);

   queue_node *qn = this->head;
   this->head = this->head->next;
   qn->next = NULL;
   //this->head = this->head->next;
   this->size--;

   if(!this->size)
      this->tail = NULL;

   pthread_cond_signal(&this->cv);
   //fprintf(stderr, "here\n");
   pthread_mutex_unlock(&this->m);
   return qn->data;
}
