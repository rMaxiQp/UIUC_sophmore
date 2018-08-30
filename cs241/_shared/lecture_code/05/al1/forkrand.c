// Lawrence Angrave CS241 Lecture Demo
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char**argv) {
    srand(time(NULL));
    pid_t child = fork();
    printf("My fork value is %d\n", (int) child );
    int r = rand() & 0xf;
    printf("%d: My random number is %d\n", getpid(), r);
    return 0;
}
