/**
*  Lab
* CS 241 - Spring 2018
*/

#include "core/utils.h"
#include <alloca.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

void setup_fds(int new_stdin, int new_stdout);
void close_and_exec(char *exe, char *const *params);
pid_t start_reducer(char *reducer_exec, int in_fd, char *output_filename);
pid_t *read_input_chunked(char *filename, int *fds_to_write_to,
                          int num_mappers);
pid_t *start_mappers(char *mapper_exec, int num_mappers, int *read_mapper,
                     int write_reducer);
size_t count_lines(const char *filename);

void usage() {
    print_usage();
}

int main(int argc, char **argv) {
   if(argc != 6) {
      print_usage();
      return 0;
   }

   int numMap= 0;

   if( 0 == (numMap = atoi(argv[5]))) {
      print_usage();
      return 0;
   }

   char *input = argv[1];
   char *output = argv[2];
   char *map_exec = argv[3];
   char *reduce_exec = argv[4];
   char *split_cmd = "./splitter";

   //fprintf(stderr, "%s %s %s %s\n", input, output, map_exec, reduce_exec);

   int mapper_pip[numMap][2];
   int reducer_pip[2];
   int splitters[numMap];
   int map_receive[numMap];
   // Create an input pipe for each mapper.
   for(int i = 0; i < numMap; i++) {
      pipe2(mapper_pip[i], O_CLOEXEC);
      map_receive[i] = 0;
      splitters[i] = 0;
   }

   // Create one input pipe for the reducer.
   pipe2(reducer_pip, O_CLOEXEC);

   // Open the output file.
   int out_fd = open(output, O_TRUNC | O_CREAT | O_WRONLY | O_CLOEXEC, S_IRWXU | S_IRWXG);
   if(out_fd == -1)
      exit(-1);

   // Start a splitter process for each mapper.
   for(int i = 0; i < numMap; i ++) {
      if(0 == (splitters[i] = fork())) {//child
         for(int j = 0; j < numMap; j ++) {
            if(j != i)
               close(mapper_pip[j][1]);
            close(mapper_pip[j][0]);
         }
         close(reducer_pip[0]);
         close(reducer_pip[1]);
         dup2(mapper_pip[i][1], 1);
         char idx[5];
         sprintf(idx, "%d", i);
         execlp(split_cmd, split_cmd, input, argv[5], idx, (char*) NULL);
         exit(-1);
      }
   }

   // Start all the mapper processes.
   for(int i = 0; i < numMap; i ++) {
      if(0 == (map_receive[i] = fork())) {//chlid
         for(int j = 0; j < numMap; j ++) {
            if(j != i)
               close(mapper_pip[j][0]);
            close(mapper_pip[j][1]);
         }
         close(reducer_pip[0]);
         close(out_fd);
         dup2(mapper_pip[i][0], 0);
         dup2(reducer_pip[1], 1);
         execlp(map_exec, map_exec, (char*) NULL);
         //fprintf(stderr, "mapper\n");
         exit(-1);
      }
   }

   close(reducer_pip[1]);
   for(int i = 0; i < numMap; i ++) {
      close(mapper_pip[i][0]);
      close(mapper_pip[i][1]);
   }

   // Start the reducer process.
   pid_t chlid = fork();
   if(0 == chlid) {//chlid
      close(reducer_pip[1]);
      dup2(reducer_pip[0], 0);
      dup2(out_fd, 1);
      execlp(reduce_exec, reduce_exec, (char*) NULL);
      exit(-1);
   }

   int status = 0;
   // Wait for the reducer to finish.
   for(int i = 0; i < numMap; i++)
      waitpid(splitters[i], &status, 0);

   // Print nonzero subprocess exit codes.
   for(int i = 0; i < numMap; i++) {
      status = 0;
      waitpid(map_receive[i], &status, 0);
      if(!status && WIFEXITED(status) && WEXITSTATUS(status))
         print_nonzero_exit_status(map_exec, status);
   }

   //fprintf(stderr, "status : %d\n", status);

   status = 0;
   if(-1 == waitpid(chlid, &status, 0)) {
      perror("reduce:");
      exit(-1);
   }

   // Count the number of lines in the output file.
   print_num_lines(output);
   close(out_fd);
   close(reducer_pip[0]);
   close(reducer_pip[1]);
   return 0;
}
