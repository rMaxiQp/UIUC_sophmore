#include <unistd.h>
#include <stdio.h>

int main(int argc, char**argv) {
  char mesg[50];
  while(1) {
    ssize_t bytes_read = read( 0, mesg, sizeof(mesg) );
  //write( 1, mesg, bytes_read );
    if(bytes_read == 0) {
      return 0;
    }
    mesg[bytes_read] = '\0';
    puts(mesg);
  }
  
  return 0;
}
