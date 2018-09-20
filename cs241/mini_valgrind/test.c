/**
* Mini Valgrind Lab
* CS 241 - Spring 2018
*/

#include <stdio.h>
#include <stdlib.h>

int main() {
    // Your tests here using malloc and free
   int* a = malloc(sizeof(int));

   double *b = calloc(5, sizeof(double));

   a = realloc(a, sizeof(double) * 2);

   free(a);

   free(b);

   return 0;
}
