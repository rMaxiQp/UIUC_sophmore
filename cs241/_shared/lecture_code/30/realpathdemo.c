 #include <stdlib.h>
 #include <stdio.h>
int main() {
  char* path = realpath("./../../",NULL); 
  puts(path);
  free(path);
  return 0;
}
