/**
* Extreme Edge Cases Lab
* CS 241 - Spring 2018
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "camelCaser.h"
#include "camelCaser_tests.h"

/*
 * Testing function for various implementations of camelCaser.
 *
 * @param  camelCaser   A pointer to the target camelCaser function.
 * @param  destroy      A pointer to the function that destroys camelCaser
 * output.
 * @return              Correctness of the program (0 for wrong, 1 for correct).
 */

const char* input[7] = {"thisis 1 sINGle test cases.", "./?><|-=\\|]", "                           ", "a\ab\tc\nd\r", "this. is. 1. 12 b ingle. test about cAes.", "tHIS_is_1 single tes_t cA]ses", ""};
const char* output[7][13] = {{"thisis1SingleTestCases", NULL}, {"", "", "", "", "", "", "", "", "", "", "", NULL}, {NULL}, {NULL}, {"this","is","1","12BIngle","testAboutCaes",NULL},{"this","is", "1SingleTes", "tCa", NULL}, {NULL}};

const char* test_1 = "ABC  r,t\ny u.i ,.zzji ooi o   /p --=  o\t    ' f  g    PPMU__NU _Ma  121{32 l  }    zx[ ] \\m";
const char* solution_1[] =   {"abcR","tYU","i","","zzjiOoiO","p","","",  "o", "fGPpmu","","nu","ma121","32L",  "zx","","", NULL};

int test_camelCaser(char **(*camelCaser)(const char *),
                    void (*destroy)(char **)) {
    int i = 0;
    while(i<7) {
//    printf("%d", i);
	char** holder = camelCaser(input[i]);
	int j = 0;
	while(output[i][j]) {
	//content test
        //printf("%s",holder[j]);
	    if(holder[j] == NULL || strcmp(holder[j], output[i][j])) {
		destroy(holder);
		return 0;
	    }
	    j++;
	}
        //printf("end of the strcmp\n");
        if(holder[j]) {
         destroy(holder);
         return 0;
        }
        destroy(holder);
	i++;
   }

    //null ptr test
    char** expect_null = camel_caser(NULL);
    if(expect_null) {
	destroy(expect_null);
	return 0;
    }
   //printf("after NULL ptr test");
    //destroy
    destroy(expect_null);
    if (expect_null != NULL) return 0;
  //  printf("asd\n");


 // printf("before stress\n");
    char** test = camel_caser("cg,|X}K+o4Z?t_?=<aR3[tzO|!nwX&sp]<(GRP# P*u%NaX13=Pw@ol][mnj[2-Ii0 eJw8B@3h@^AhQi< i3shI$dQF=v@nBli5;WOX|8Cj`JlRWT[VN.xJe_q;As'*&Y,(>PS&6*%<m[Vc-v4X'X`Zey@B^m?wZ#l|Czr[eqOcjLaJXKr2clF`PA35GnCs<ihJ8`ozSa*qjDK=<_B$z)1'WK4?cR5{sbPLJvAcV38Z3.#7t0I|$V-q];}.4a7&");
    const char* so[] = {"cg","","x","k","o4z","t","","","","ar3","tzo","","nwx", "sp","","","grp","p","u","nax13","pw","ol","","mnj","2","ii0Ejw8b","3h","","ahqi","i3shi","dqf","v","nbli5","wox","8cj","jlrwt",
    "vn","xje","q","as",  "", "", "y", "","","ps","6","", "", "m","vc", "v4x","x","zey","b","m","wz","l",  "czr","eqocjlajxkr2clf","pa35gncs","ihj8","ozsa","qjdk","","","b","z","1","wk4",  "cr5","sbpljvacv38z3","", "7t0i","","v","q","","", "","4a7",NULL};
    int p = 0;
    while(so[p])
    {
       if(strcmp(so[p], test[p]))
       {
          destroy(test);
          return 0;
       }
       p++;
    }
    destroy(test);
//printf("<<\n");
    char** test_1_cc = camel_caser(test_1);
    p = 0;
    while(test_1_cc[p] && test_1[p]) {
       if(strcmp(solution_1[p], test_1_cc[p]))
       {
          destroy(test_1_cc);
          return 0;
       }
       p++;
    }
    if(solution_1[p] != NULL || test_1_cc[p] != NULL) {
       destroy(test_1_cc);
       return 0;
    }
    destroy(test_1_cc);

   //printf("before ascii");
    char** ascii = camel_caser("\x7fS <\x1 | \x23 .!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~");
    char* s[] = {	"\x7Fs",	"\x01",	"",	"",	"",	"",	"",	"",	"",	"",	"",	"",	"",	"",	"",	"",	"",	"",	"",	"0123456789",
	"",	"",	"",	"",	"",	"",	"abcdefghijklmnopqrstuvwxyz",
	"",	"",	"",	"",	"",	"abcdefghijklmnopqrstuvwxyz","","","",NULL};
  p = 0;
  while(s[p])
  {
    if(!ascii[p])
    {
      destroy(ascii);
      return 0;
    }
    if(strcmp(ascii[p], s[p]))
    {
      destroy(ascii);
      return 0;
    }
    p++;
  }
  destroy(ascii);
    return 1;
}
