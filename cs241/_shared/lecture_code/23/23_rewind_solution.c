#include <stdio.h>

typedef struct _FILE {
   int fd;
   void* buffer;
   size_t capacity, size;
   int mode; // _IONBF _IOLBF _IOFBF, see setvbuf
} FILE;
// hint  fseek or fflush and lseek are useful here
void rewind(FILE*f) {


}

//hint: use and reset the buffer content
// Use write
void fflush(FILE*f) {

}

