/**
* Utilities Unleashed Lab
* CS 241 - Spring 2018
*/
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/wait.h>
#include <stdlib.h>
#include "format.h"

int main(int argc, char *argv[]) {
   if(argc < 2) {
      print_time_usage();
      exit(2);
   }
   pid_t id = fork();
   if(id == -1) {
      print_fork_failed();
      exit(1);
   }
   else if (!id) {
      execvp(argv[1], argv + 1);
      print_exec_failed();
   } else {
      struct timespec tp;
      struct timespec ts;
      clockid_t clkid = CLOCK_MONOTONIC;
      int current = clock_gettime(clkid, &tp);
      int status = 0;
      waitpid(id, &status, 0);
      current = clock_gettime(clkid, &ts);
      display_results(argv, ts.tv_sec - tp.tv_sec + (ts.tv_nsec - tp.tv_nsec)/1E9);
   }
   /*
   else {
     int err = execvp(argv[1], argv + 1);
     print_exec_failed();
     exit(1);
   }*/
   return 0;
}
