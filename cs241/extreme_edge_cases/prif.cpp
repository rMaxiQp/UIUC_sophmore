#include <iostream>
#include <stdio.h>

int main() {
  for(int i = 0; i < 127; i++)
  {
    if(i != 10 || i != 11)
    {
      printf("%c",i);
    }
  }
}
