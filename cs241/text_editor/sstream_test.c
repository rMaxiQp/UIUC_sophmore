/**
* Text Editor Lab
* CS 241 - Spring 2018
*/

#include "sstream.h"
#include <limits.h>
int main(int argc, char *argv[]) {
    // TODO create some tes
    bytestring b = (bytestring){"9999999999-999898991309274091240971502", -1};
    //bytestring s = (bytestring){"--891230903", -1};
    bytestring z = (bytestring){"-92233720368547758080", -1};
    sstream *stream = sstream_create(b);
    long* l = malloc(sizeof(long));
    //sstream_parse_long(stream, l);
   //printf("%ld for min\n", LONG_MIN);
    //printf("%ld for max\n", LONG_MAX);
   // printf("%s\n", b.str);
    //printf("%ld start ptr\n", sstream_tell(stream));
    //printf("%d  %ld\n", sstream_parse_long(stream, l), *l);
    //printf("%d   %ld\n", sstream_parse_long(stream, l), *l);
    //printf("%ld\n", sstream_tell(stream));
    //printf("%s\n", b.str);
    //sstream_str(stream, s);
    //sstream_parse_long(stream, l);
    //printf("%ld for min\n", LONG_MIN);
    //printf("%ld for max\n", LONG_MAX);
   // printf("%ld\n", *l);
    //printf("%d  %ld\n", sstream_parse_long(stream, l), *l);
    //printf("%ld\n", sstream_tell(stream));
    //printf("%s\n", s.str);
    sstream_str(stream, z);
    printf("%ld %s\n", sstream_tell(stream), z.str);
    printf("%d %ld\n", sstream_parse_long(stream, l), *l);
    printf("diving....\n");
    printf("%ld %s\n", sstream_tell(stream), z.str);
//;x
printf("%d %ld\n", sstream_parse_long(stream, l), *l);

    free(l);
    if(sstream_eos(stream)) printf("yes\n");
    else printf("%zu with ptr at %zu\n", sstream_remain(stream), sstream_tell(stream));
    sstream_destroy(stream);
    return 0;
}

