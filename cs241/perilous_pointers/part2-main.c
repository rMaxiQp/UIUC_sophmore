/**
* Pointers Gone Wild Lab
* CS 241 - Spring 2018
*/

#include "part2-functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * (Edit this function to print out the "Illinois" lines in
 * part2-functions.c in order.)
 */
int main() {

   int num = 81;
   first_step(num);

    int *second = malloc(sizeof(int));
    *second = 132;
    second_step(second);
    free(second);
    second = NULL;

    int **d = malloc(sizeof(int*));
    d[0] = malloc(sizeof(int));
    *d[0] = 8942;
    double_step(d);
    free(d[0]);
    free(d);
    d = NULL;
    //4
    int jk = 15;
    char zzz[6] = {'0','1','2','3','z' , jk};
    //printf("%d",* (int*)(zzz+5));
    //char * strange = malloc(sizeof(char) * 8);
    //strcpy(strange, s);
    strange_step(zzz);
    //free(strange);
    //strange = NULL;
   //5 6
    char * ety = "abc\0";
    void* p = ety;
    empty_step(p);
    ety = "accu\0";
    p = ety;
    two_step(p, ety);
    //7 8
    char* a = "a";
    char* z = "b";
    char* c = "c";

    three_step(a, z, c);
    //printf("%p\n",a);
   // printf("%p\n", a+2);

    char f[4] = {'3', (char)0, '1', '1'};
    char se[4] = {'3', 'a', (char)8, 'o'};
    char t[4] = {'3', '2', '1', (char)16};
    char *three[5] = {f, "09z0", se, "azzz", t};
    step_step_step(three[0], three[2], three[4]);
    //9
    char* v = malloc(sizeof(char));
    *v = 1;
    it_may_be_odd(v, 1);
    free(v);
    v =NULL;
    //10
    char* tok = malloc(sizeof(char) * 10);
    memcpy(tok, "a,CS241,i",10);
    tok_step(tok);
    free(tok);
    tok = NULL;
   //11
    int b = 0x801;
    void * blue = &b;
    the_end(blue, blue);

    return 0;
}
