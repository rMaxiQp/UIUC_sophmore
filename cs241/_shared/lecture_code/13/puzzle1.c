#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

int your_starting_value[10];
pthread_t tid[10]; 

void* myfunc(void*ptr) {
  int myvalue = * (int*) ptr;
  printf("My thread id is %ld and I'm starting at %d\n", (long) pthread_self(), myvalue);

  return NULL;
}
int main() {
  // Each thread needs a different value of i 
  
  for(int i =0; i < 10; i++) {
     your_starting_value[i] = i;
     pthread_create( &tid[i], 0, myfunc, & your_starting_value[i] );
   }
   for ... pthread_join on all threads
     pthread_exit(NULL) ; // this is a one way trip - never returns !!!!!!!!!
   return 0;
}

