#include <stdio.h>

int main(int argc, char** argv) {
  argv++;
  while( *argv) {
  //printf("%s\n", *argv);
    puts(*argv);
    argv++;
  }
  return 0;
}
