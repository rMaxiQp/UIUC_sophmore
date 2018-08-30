// Lawrence Angrave CS241 Lecture Demo
#include <stdio.h>

extern char**environ;

int main(int argc, char**argv) {
  int i =0;
  while(environ[i]) {
    puts(environ[i]);
    i++;
  }
  return 0;
}
