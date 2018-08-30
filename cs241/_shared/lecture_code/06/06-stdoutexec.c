#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
   close(1); // close standard out
   open("log.txt", O_RDWR | O_CREAT | O_APPEND, S_IRUSR | S_IWUSR);
   // Append selected - so if you run it again the output will be appended

   puts("Captain's log");
   fflush(stdout); // otherwise no 'Captains Log' is sent to the file

   chdir("/usr/include");
   execl("/bin/ls", "/bin/ls",".",(char*)NULL); // "ls ."
   perror("exec failed");
   return 0; // Not expected
}
