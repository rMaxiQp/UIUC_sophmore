// Lawrence Angrave CS241 Lecture Demo

#include <unistd.h>
#include <stdio.h>
int main(int argc, char** argv) {
  
   char* lineptr = NULL;
   size_t size = 0 ;
   
   ssize_t bytesread  = getline(&lineptr, &size, stdin);
   lineptr[ bytesread -1]  = 0; 
   // Assumes bytesread>0. Assumes line is terminated with a newline (not always true!)

   fprintf(stderr,"About to execute %s\n",lineptr);

   execlp(lineptr, lineptr, "." , (char*)NULL );
   perror("failed.. try again");
   return 0;
}
