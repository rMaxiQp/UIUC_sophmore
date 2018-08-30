#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <signal.h>


void myfunc(int signal) {
  write(1,"Oh No!",6);
  pleasequit=1;
}

int main() {
  signal(SIGINT, myfunc);
  printf("My pid is %d\n", getpid() );
  int i = 60;
  while(--i) { 
    write(1, ".",1);
    if(i == 56) {
      //kill( getpid(), SIGINT); // CRTL-C
      raise(SIGINT);
    }
    sleep(1);
  }
  write(1, "Done!",5);
  return 0;
}
