#include <stdio.h>
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
   char* p1 = (char*) histogram;
   char* p2 = (char*) &admin;
   printf("Difference: %d bytes\n", (int) (p1 - p2));
   while(argc>1) {
      char*word= argv[ --argc];
      int h = hash(word);
      printf("Hash value: %d %d  index:%d\n",h, -h, (h<0?-h:h) % N);
      histogram[ (h<0?-h:h) % N ] ++;
   }
   if(admin || debug) puts("Admin/Debug rights");
   return 0;
}

