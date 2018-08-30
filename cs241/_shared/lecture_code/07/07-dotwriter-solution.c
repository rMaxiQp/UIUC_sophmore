#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <signal.h>

void ohnoyoudont(int s) {
   write(1,"No go away!",11);
}

int main() {
  signal( SIGINT, ohnoyoudont );

  printf("My pid is %d\n", getpid() );
  int i = 60;
  while(--i) { 
    write(1, ".",1);
    if(i==55 ) {
      kill( getpid() , SIGINT );
    }
    sleep(1);
  }
  write(1, "Done!",5);
  return 0;
}
