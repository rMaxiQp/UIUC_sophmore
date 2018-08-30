#include <unistd.h>
#include <stdio.h>
int main(int argc, char** argv) {
   puts("Oh Master whatwould you like to run.");
   char* lineptr = NULL;
   size_t size = 0;
   ssize_t bytes_read = getline(&lineptr, &size, stdin);
   if(bytes_read >0) {
     lineptr[bytes_read-1]= '\0';
   fprintf(stderr,"OK I will run %s for you.", lineptr);
   execlp(lineptr, lineptr, (char*) NULL);
   
   printf("Sorry Dave I cannot run %s.\n", lineptr);
 }
   return 0;
}
