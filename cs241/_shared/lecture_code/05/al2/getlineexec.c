#include <unistd.h>
#include <stdio.h>
int main(int argc, char** argv) {
  puts("Oh Friday master. What is your desire??");
   char* lineptr = NULL;
   size_t size;
   ssize_t bytes_read = getline( &lineptr, &size, stdin);
   if(bytes_read >0 ) {
     lineptr[bytes_read-1] = '\0';
     printf("OK I shall run %s for thee.", lineptr);
     fflush(stdout);
     execlp(lineptr,lineptr, (char*) NULL);
     printf("No cookies for you %s would not exec!\n", lineptr);
   }
   return 0;
}
