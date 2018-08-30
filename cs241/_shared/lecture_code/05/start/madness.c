// Lawrence Angrave CS241 Lecture Demo

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

int main(int c, char **v)
{
  while (--c > 1 && !fork());
  int val  = atoi(v[c]);
  sleep(val);
  printf("%d\n", val);
  return 0;
}
