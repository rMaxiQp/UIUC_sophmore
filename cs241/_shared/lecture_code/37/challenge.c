#include <stdio.h>
#include <limits.h>

#define N (20)
int admin, debug;
int histogram[N];

static int hash(char* str) {
   int c, h = 0; // sdbm hash
   while ( (c = *str++))
       h = c + (h << 6) + (h << 16) - h;
   return h;
}

int main(int argc, char**argv){
   printf(" INT_MIN : %d %d \n", INT_MIN,  -INT_MIN);
   while(argc>1) {
      char*word= argv[ --argc];
      int h = hash(word);
      histogram[ (h<0?-h:h) % N ] ++;
   }
   if(admin || debug) puts("Admin/Debug rights");
   return 0;
}

