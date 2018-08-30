/**
* Extreme Edge Cases Lab
* CS 241 - Spring 2018
*/

#include "camelCaser.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>

char **camel_caser(const char *input_str) {
    if(!input_str) return NULL;
    if(!*input_str)
    {
       char** output = malloc(sizeof(char*));
       output[0] = NULL;
       return output;
    }

    size_t size = 0;
    int punct = 0;
    while(input_str[size])
    {
       if(ispunct(input_str[size++])) punct++;
    }
   //printf("%zu, %zu\n", size, strlen(input_str));
    char input_s[punct+1][size+1];
    int l_array[punct+1];
    for(int i = 0; i <= punct; i++) {
       for(size_t j = 0; j <= size; j++)
          input_s[i][j] = 0;
          l_array[i] = 0;
    }

    size_t cursor = 0;
    int idx = 0, length = 0;
    int space = 0;//flag to detect space
    while(cursor < size) {
	if(ispunct(input_str[cursor])) { //end of the camel_case
            input_s[idx][length++] = '\0';
	    l_array[idx++] = length;//store the length
            length = 0;
            space = 0;
	}
        else if(length > 0 && isspace(input_str[cursor])) {
            space = 1;//next character will be capitalized
        }
        else if(space && isalpha(input_str[cursor])) {
                space = 0;
		input_s[idx][length++] = toupper(input_str[cursor]);//capitalize the first character of the word
            }
        else if(!isspace(input_str[cursor])) {
		input_s[idx][length++] = tolower(input_str[cursor]);
	    }
        cursor++;//increment cursor on input_str
    }

    char** output_s = (char**) malloc((punct+1) * sizeof(char*));
    for(int i = 0; i < punct; i++) {
       output_s[i] = (char*) malloc(l_array[i] * sizeof(char));
       memcpy(output_s[i], input_s[i], l_array[i]);
      // printf("%s\n", output_s[i]);
      // printf("copid\n");
    }
    output_s[punct] = NULL;
    /*void* p = output_s;
   printf("%d\n", punct);
    while(*output_s) {
       printf("output_s: %s", *output_s);
       output_s++;
    }
    output_s = p;
    */
    return output_s;
}

void destroy(char **result) {
    if(!result) return;
    int i = 0;
    while(result[i] != NULL) {
	free(result[i]);
        result[i] = NULL;
	i++;
    }
   free(result);
   result = NULL;
}
