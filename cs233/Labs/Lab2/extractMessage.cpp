/**
 * @file
 * Contains the implementation of the extractMessage function.
 */

#include <iostream> // might be useful for debugging
#include <assert.h>
#include "extractMessage.h"

using namespace std;

char *extractMessage(const char *message_in, int length) {
   // length must be a multiple of 8
   assert((length % 8) == 0);

   // allocate an array for the output
   char *message_out = new char[length];

   unsigned a = 0;
   while(a < length){
     for(unsigned x = 0; x < 8; x++){
       for(unsigned i = 0; i < 8; i++){
         message_out[a+7-x] |= ((message_in[a+i] >> (7-x)) & 1) << i;
      }
    }
  a += 8;
  }
	return message_out;
}
