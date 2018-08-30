#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

void dirlist(char*path) {
  
  struct dirent* dp;
  struct stat statstruct;
  
  DIR* dirp = opendir(path);
  if( ! dirp) {
    perror("opendir failed!");
    return;
  }

  while ((dp = readdir(dirp)) != NULL) {
    if( strcmp(".",dp->d_name) == 0  || strcmp("..",dp->d_name) == 0 ) {
      continue;
    }
     char newpath[strlen(path) + strlen(dp->d_name) + 2]; // or use asprintf

     sprintf(newpath,"%s/%s", path, dp->d_name);

     printf("newpath: %s \n", newpath);

     if( 0 == stat(newpath, & statstruct) && S_ISDIR( statstruct.st_mode)) {
       dirlist(newpath);
     } 
  }
  closedir(dirp);
}

int main(int argc, char**argv) { dirlist(argv[1]);return 0; }
