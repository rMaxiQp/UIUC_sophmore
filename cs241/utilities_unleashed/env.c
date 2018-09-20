/**
* Utilities Unleashed Lab
* CS 241 - Spring 2018
*/

#include <stdlib.h>
#include <unistd.h>
#include "format.h"
#include <string.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <ctype.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
   if(argc < 4) {
      print_env_usage();
   }

   int ptr = 1;
   int limit = 1;
   if(!strcmp(argv[1],"-n")) {
      if(!atoi(argv[2]))
         print_env_usage();
      limit = atoi(argv[2]);
      ptr += 2;
   }

   char* cmd;
   int start = ptr;
   while(ptr < argc) {
      if(!strcmp(argv[ptr], "--")) {
         if(ptr + 1 < argc) {
            cmd = argv[ptr+1];
            break;//take everything after ptr to arg
         }
         else
            print_env_usage();
      }
      ptr++;
   }

   if(ptr == argc) print_env_usage(); //no "--" print_usage

   for(int i = 0; i < limit; i++) {
      char* string = "";
      for(int j = start; j < ptr; j++) {
         size_t size = strlen(argv[j]) + 1;
         string = malloc(sizeof(char) * size);
         memcpy(string, argv[j], size);
         char* temp_key = strtok(string, "=");
         char* temp_var;
         for(int k = 0; k <= i; k ++){
            temp_var = strtok(NULL, ",");
            if(!temp_var) {
               memcpy(string, argv[j], size);
               strtok(string,"=");
               temp_var = strtok(NULL, ",");
            }
         }
         if(!temp_var) {
            free(string);
            print_env_usage();
         }
         if(temp_var[0] ==  '%') {
            char *get_env = getenv(temp_var+1);
            if(!get_env) {
               get_env = "";
            }
            temp_var = get_env;
         }
         int result = setenv(temp_key, temp_var, 1);
         if(result == -1) {
            free(string);
            print_environment_change_failed();
         }
      }
         pid_t pt = fork();
         if(pt == -1) {
            free(string);
            print_fork_failed();
         }
         else if(!pt) {
            execvp(cmd, argv + ptr + 1);
            free(string);
            print_exec_failed();
         }
         else {
            int status;
            wait(&status);
         }
         free(string);
   }
   return 0;
}
