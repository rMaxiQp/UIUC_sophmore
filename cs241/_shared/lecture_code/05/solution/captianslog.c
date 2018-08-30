#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

int main() {
   close(1); // close standard out
   // open will use the smallest unused non-negative integer filedescriptor i.e. 1
   open("log.txt", O_RDWR | O_CREAT | O_APPEND, S_IRUSR | S_IWUSR);
   puts("Captain's log");
   chdir("/usr/bin");
   // fflush before exec- otherwise you won't see "captains log"
   fflush(stdout); // now write(fd=1,....)  is called
   
   execl("/bin/ls", "/bin/ls",".",(char*)NULL); // "ls ."
   perror("exec failed");
   return 0; // Not expected
}
