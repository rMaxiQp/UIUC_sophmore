#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

int main() {
   close(1); // close standard out
   // opens a new file
   open("log.txt", O_RDWR | O_CREAT | O_APPEND, S_IRUSR | S_IWUSR);
   
   puts("Captain's log"); // later... write(1, ...,...)
   fflush(stdout); // write(fd=1,)
   
   chdir("/usr/bin");
   execl("/bin/ls", "/bin/ls",".",(char*)NULL); // "ls ."
   perror("exec failed");
   return 0; // Not expected
}
