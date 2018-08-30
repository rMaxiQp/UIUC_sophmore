/**
* Finding Filesystems Lab
* CS 241 - Spring 2018
*/

#include "minixfs.h"
#include "minixfs_utils.h"
#include <assert.h>

int main(int argc, char *argv[]) {
    // Write tests here!

   char *str = " World!";
   off_t off = 5;
   file_system * fs = open_fs("../");
   minixfs_write(fs, "/text", str, 7, &off);
   close_fs(&fs);
}
