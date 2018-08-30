/**
* Networking Lab
* CS 241 - Spring 2018
*/

#include "common.h"
#include "format.h"

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
         return cur;

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
      if(!total) {
         print_connection_closed();
         return cur;
      }

      if(-1 == total && errno == EINTR) //total == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN))
         continue;
      else if(total == -1)
         return -1;

      cur += total;
   }
   return cur;
}

/*
 * read from reader
 *
 * write to writer
 * */
ssize_t file_to_file(int reader, int writer, size_t file_size){
   ssize_t cur = file_size;
   ssize_t sent_size = 0;
   while(cur > 0) {
      char transfer[CAPACITY] = {0};
      sent_size += read_all_from_socket(reader, transfer, MIN(CAPACITY, cur));
      CHECK_WRITE(write_all_to_socket(writer, transfer, MIN(CAPACITY, cur)));
      cur -= CAPACITY;
   }
   return sent_size;
}

/*
 * read from reader
 *
 * write to writer
 *
 * NONBLOCK
 *
 * return offset
 * */
ssize_t ff_nonblock(int reader, int writer, size_t file_size){

   char transfer[CAPACITY] = {0};
   ssize_t size = read_all_from_socket(reader, transfer, MIN(CAPACITY, file_size));

   //size == 0 || size = -1
   if(size < 1 )
      return size;

   ssize_t temp = write_all_to_socket(writer, transfer, size);

   return temp; //MIN(temp, MAX(size, (ssize_t)strlen(transfer)));
}
