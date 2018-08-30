#include <stdio.h>

int main() {
  printf("String at address %p \n", "!");
  
  int bad = (int) "Hi";
  puts( (char*) bad);
  return 0;
}
