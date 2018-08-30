/**
* Pointers Gone Wild Lab
* CS 241 - Spring 2018
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "callbacks.h"
#include "part3-functions.h"
#include "vector.h"

vector *vector_map(vector *input, mapper map) {
   vector* result = string_vector_create();
    VECTOR_FOR_EACH(input, elem, {
          elem = map(elem);
          vector_push_back(result, elem);
          free(elem);
          });
   // vector_destroy(input);
   // free(input);
   // input = NULL;
    return result;
}

void *vector_reduce(vector *input, reducer reduce, void *acc) {
    VECTOR_FOR_EACH(input, elem, {
          acc = reduce(elem, acc);
          });
    return acc;
}

void *length_reducer(char *input, void *output) {
    if(output == NULL) output = calloc(1, sizeof(int));
    *(int *)output += strlen(input);
    return output;
}

void *concat_reducer(char *input, void *output) {
    if(!output) output = calloc(strlen(input) + 1, sizeof(char));
    else output = realloc(output,strlen((char*)output) + strlen(input) + 1);
    strcat((char*)output, input);
    return output;
}
