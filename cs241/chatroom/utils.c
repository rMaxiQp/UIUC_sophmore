/**
* Chatroom Lab
* CS 241 - Spring 2018
*/

#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

#include "includes/utils.h"
static const size_t MESSAGE_SIZE_DIGITS = 4;

char *create_message(char *name, char *message) {
    int name_len = strlen(name);
    int msg_len = strlen(message);
    char *msg = calloc(1, msg_len + name_len + 4);
    sprintf(msg, "%s: %s", name, message);

    return msg;
}

ssize_t get_message_size(int socket) {
    int32_t size;
    ssize_t read_bytes =
        read_all_from_socket(socket, (char *)&size, MESSAGE_SIZE_DIGITS);
    if (read_bytes == 0 || read_bytes == -1)
        return read_bytes;

    return (ssize_t)ntohl(size);
}

// You may assume size won't be larger than a 4 byte integer
ssize_t write_message_size(size_t size, int socket) {
   // Your code here
   int32_t ssize = htonl((ssize_t) size);
   ssize_t ret = write_all_to_socket(socket, (char *) &ssize, MESSAGE_SIZE_DIGITS);
   return ret;
}

ssize_t read_all_from_socket(int socket, char *buffer, size_t count) {
    // Your Code Here
   size_t cur = 0;
   while(cur < count) {
      ssize_t total = read(socket, buffer + cur, count - cur);
      if(!total)
         return 0;

      if(total == -1 && errno == EINTR)
         continue;
      else if(total == -1)
         return -1;
      cur += total;
   }
   return cur;
}

ssize_t write_all_to_socket(int socket, const char *buffer, size_t count) {
    // Your Code Here
   size_t cur = 0;
   while(cur < count) {
      ssize_t total = write(socket, buffer + cur, count - cur);
      if(!total)
         return 0;

      if(total == -1 && errno == EINTR)
         continue;
      else if(total == -1)
         return -1;

      cur += total;
   }
   return cur;
}
