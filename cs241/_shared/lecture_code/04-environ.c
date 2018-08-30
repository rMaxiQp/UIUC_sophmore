#include <stdio.h>

extern char **environ;
int main(int a, char**b) {
  fprintf(stderr,"evironment is at %p\nargv is at %p\n", environ, b);
  puts(environ[0]);
  return 0;
}