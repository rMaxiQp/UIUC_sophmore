#include <openssl/sha.h>
//On Ubuntu sudo apt-get install libssl-dev
//Centos. Try sudo yum install openssl-devel
//Mac. Instead node

#include <pthread.h>
// gcc mine-threaded.c -std=c99 -lcrypto -pthread
// On OSX with node installed, I already had openssl headers here...
// gcc mine-threaded.c -std=c99 -lcrypto -pthread -I/usr/local/include/node/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>



void search(long start, long end) {
  printf("Searching from 0x%lx to 0x%lx\n", start , end);
  for(long i = start; i <end; i++) {
    char message[100];
    sprintf(message,"AngraveCoin:%lx", i);
    
    unsigned char *res = SHA256(message, strlen(message), NULL);
    int found = (res[0] == 0) && (res[1] == 0) && (res[2] == 0);

    if(found) // || (i & 0xfffff) ==0) 
        printf("%d %lx %02x %02x %02x '%s'\n", found, i, res[0], res[1], res[2] , message);
  }
  printf("Finished %lx to %lx\n", start, end);
}


long array[] = {0L, 1L <<25, 1L <<27, 1L <<33};
int main() {
  search(array[0], array[1]);
  search(array[1], array[2]);
  search(array[2], array[3]);
  return 0;
}
