/**
* Pied Piper Lab
* CS 241 - Spring 2018
*/

#include "pied_piper.h"
#include "utils.h"
#include <fcntl.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#define TRIES 3

//input_fd  overall input
//output_fd overall output
//executables overall processes
int pied_piper(int input_fd, int output_fd, char **executables) {
    size_t arr_size = 0;
    size_t try = TRIES;
    while(executables[arr_size]) {
       arr_size++;
    }
    failure_information **fail = calloc(arr_size, sizeof(failure_information*));

    //fprintf(stderr, "enter pip\n");
    while(try) {
       int error = 0;
       int pips[arr_size][2];
       int backup[arr_size][2];//in case of failure
       pid_t children[arr_size];

       //fprintf(stderr, "start pipe\n");
       for(size_t t = 0; t < arr_size; t++) {
          pipe(pips[t]);
          pipe(backup[t]);
       }

       //fprintf(stderr, "fork process\n");
       //setup && child process executation
       for(size_t t = 0; t < arr_size; t++) {
          if(!(children[t] = fork())) { //child
             dup2(backup[t][1], 2);
             if(t == 0) {//first pipeline input
                dup2(input_fd, 0);
                close(pips[t][0]);
             }
             else
                dup2(pips[t][0], 0);

             if(t == 0)
                reset_file(input_fd);

             close(pips[t][1]);
             if(t == arr_size - 1) { //last pipeline output
                dup2(output_fd, 1);
                close(pips[t][1]);
                reset_file(output_fd);
             }
             else {
                dup2(pips[t + 1][1], 1);
                close(pips[t][0]);
             }

             exec_command(executables[t]);
             if(t != 0)
                close(pips[t][0]);

             if( t != arr_size - 1)
                close(pips[t][1]);

             exit(EXIT_FAILURE);
          }
          else { //parent
             close(pips[t][0]);
             close(pips[t][1]);
             close(backup[t][1]);
          }
       }

       //fprintf(stderr, "waitpid\n");
       //waitpid
       for(size_t t = 0; t <arr_size; t++) {
          dup2(backup[t][0], 0);
          fail[t] = calloc(1, sizeof(failure_information));
          waitpid(children[t], &fail[t]->status, 0);
          if(fail[t]->status)
             error = 1;
          fail[t]->command = executables[t];
          char * temp = calloc(1, 1024 * sizeof(char));
          fgets(temp, 1024, stdin);
          int len = strlen(temp);
          if(len > 0 && temp[len - 1] == '\n')
             temp[len - 1] = '\0';
          fail[t]->error_message = strdup(temp);
          free(temp);
          close(backup[t][0]);
       }

       if(try != 1) {
          for(size_t t = 0; t < arr_size; t++) {
             destroy_failure(fail[t]);
             free(fail[t]);
          }
       }


       if(error) //continue
          try--;
       else { //complete
          //fprintf(stderr, "great work\n");
          free(fail);
          return EXIT_SUCCESS;
       }
    }

    for(size_t t = 0; t < arr_size; t++) {
       print_failure_report(fail[t], 1);
       destroy_failure(fail[t]);
       free(fail[t]);
    }
    free(fail);
    return EXIT_FAILURE;
}
