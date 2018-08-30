/**
* Networking Lab
* CS 241 - Spring 2018
*/

#include "format.h"
#include "common.h"
#include "includes/set.h"
#include "includes/dictionary.h"

#include <stdio.h>
#include <errno.h>
#include <netdb.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>

#define MAX_CLIENTS 256
#define WAIT_TIME -1

typedef struct _client_info {
   char header[276]; //256(file_name) + 6("DELETE") + 1( ) + 1(\0) = 274
   char *file_name;
   int fd;
   int file_fd;
   verb request;
   int idx; //for LIST only
   size_t file_size;
   size_t finished_size;
   size_t tmp;
   size_t col;
} client_info;

int create_server(char *port);
void close_server(int retval);
void create_dir();
int get_file_name(void* event);
void epoll_loop();
int parse_header(void *event);
int put_function(void *event);
int get_function(void *event);
int list_function(void *event);
int delete_function(void *event);
void send_error(code c, int client_fd);
void print_status(client_info *c);
int check_data_size(void *event);
int execute(struct epoll_event ep);
void delete_client(client_info *ci);

static set *file_name_set = NULL; //store file_names, for LIST
static char *dir_name = NULL; //temp directory name
static ssize_t total_file_size = 0; //keep track of file nums in directory
static int fd = 0;
static int epoll_fd = 0;
static dictionary *fd_to_struct = NULL;

/*
 * main function
 * */
int main(int argc, char **argv) {
   // good luck!
   if(argc != 2) {//false input
      fprintf(stderr, "%s <port>\n", argv[0]);
      return -1;
   }

   signal(SIGINT, close_server);
   signal(SIGPIPE, SIG_IGN);
   fd = create_server(argv[1]);//create socket

   file_name_set = string_set_create();
   fd_to_struct = int_to_shallow_dictionary_create();

   create_dir();
   epoll_loop();
   close_server(EXIT_SUCCESS);
}

/* main component of the server
 * */
void epoll_loop() {
   struct epoll_event ev, instance[MAX_CLIENTS];

   memset(&ev, 0, sizeof(struct epoll_event));
   memset(&instance, 0, sizeof(struct epoll_event) * MAX_CLIENTS);

   //create epfd
   epoll_fd = epoll_create(MAX_CLIENTS);

   SERVER_COND(epoll_fd);

   ev.data.fd = fd;
   ev.events = EPOLLIN;

   SERVER_COND(epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &ev));

   printf("Ready to accept incomming connections\n");

   //infinite loop
   while(1) {
      int nd;
      if((nd = epoll_wait(epoll_fd, instance, MAX_CLIENTS, WAIT_TIME)) > 0) {
         for(int i = 0; i < nd; i++){
            if(instance[i].data.fd == fd) {
               struct sockaddr_in local;
               socklen_t addr_len = sizeof(struct sockaddr_in);
               int client_fd = accept(fd, &local, &addr_len);

               if(CHECK_BLOCK(client_fd))
                  break;

               if(-1 == client_fd) {
                  perror("");
                  break;
               }

               int flags = fcntl(client_fd, F_GETFL, 0);
               fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);
               instance[i].data.fd = client_fd;
               instance[i].events = EPOLLIN;

               SERVER_COND(epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &instance[i]));

               client_info *ci = calloc(1, sizeof(client_info));
               ci->fd = client_fd;
               ci->request = V_UNKNOWN;
               ci->tmp = sizeof(size_t);
               dictionary_set(fd_to_struct,  &ci->fd, ci);
            }
            else
               while(!execute(instance[i]));
         }
      }
   }
}

/******************
 * general set up *
 ******************/

/* create_temp_dir
 * */
void create_dir() {
   char temp_dir[] = "XXXXXX";
   dir_name = mkdtemp(temp_dir);
   print_temp_directory(dir_name);
   CHECK_ERROR(chdir(dir_name));
}

/*
 * parse head
 * */
int get_file_name(void *event) {
   client_info *ci = (client_info *)event;
   char *start = strstr(ci->header, " ");
   char *end = strstr(ci->header, "\n");
   *end = 0;
   size_t file_name_length = end - start;

   ci->file_name = calloc(1, sizeof(char) * file_name_length);
   for(size_t i = 0; i < file_name_length; i++) {
      if(' ' == *(start+ i + 1))
         return -1;
      *(ci->file_name + i) = *(start + i + 1);
   }

   return 0;
}

/*
 * compare data size
 * */
int check_data_size(void * event) {
   client_info *ci = (client_info *) event;

   if(ci->finished_size > ci->file_size ) {
      send_error(TMD, ci->fd);
      return 1;
   }

   if(ci->finished_size < ci->file_size) {
      send_error(TLD, ci->fd);
     // print_status(ci);
      return 1;
   }
   return 0;
}

/*
 * print status of given client_info
 * */
void print_status(client_info *c) {
   fprintf(stderr, "fd: %d\n", c->fd);
   fprintf(stderr, "file_fd: %d\n", c->file_fd);
   fprintf(stderr, "file_name: %s\n", c->file_name);
   fprintf(stderr, "file_size: %ld\n", c->file_size);
   fprintf(stderr, "finished_size: %ld\n",c->finished_size);
   fprintf(stderr, "header: %s\n", c->header);
 }

/*
 * set up general connection
 * */
int create_server(char *port) {
   fprintf(stderr, "Initializing server\n");

   int s = 0;
   int sock_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);

   int optval = 1;

   //reuse address
   SERVER_COND(setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)));

   //reuse port
   SERVER_COND(setsockopt(sock_fd, SOL_SOCKET, SO_REUSEPORT, &optval, sizeof(optval)));

   struct addrinfo hints, *result;
   memset(&hints, 0, sizeof(struct addrinfo));
   hints.ai_family = AF_INET; //IPv4
   hints.ai_socktype = SOCK_STREAM; //TCP
   hints.ai_flags = AI_PASSIVE;

   //getaddrinfo
   s = getaddrinfo(NULL, port, &hints, &result);
   if (s != 0) {
      fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(s));
      close_server(EXIT_FAILURE);
   }

   //bind
   SERVER_COND(bind(sock_fd, result->ai_addr, result->ai_addrlen));

   //listen
   SERVER_COND(listen(sock_fd, MAX_CLIENTS));

   fprintf(stderr, "Listening on port %s\n", port);

   freeaddrinfo(result);
   return sock_fd;
}

/*
 * parse and run command
 * */
int execute(struct epoll_event ep) {

   client_info * ci = (client_info *)dictionary_get(fd_to_struct, &ep.data.fd);
   if(V_UNKNOWN == ci->request) {
      if(-1 == parse_header(ci)) {
         ci = (client_info *)dictionary_get(fd_to_struct, &ep.data.fd);
         send_error(BR, ci->fd);
         delete_client(ci);
         return 1;
      }
   }

   ci = (client_info *)dictionary_get(fd_to_struct, &ep.data.fd);
   //choose VERB
   switch(ci->request) {
      case PUT:
         return put_function(ci);
      case GET:
         return get_function(ci);
      case DELETE:
         return delete_function(ci);
      case LIST:
         return list_function(ci);
      default:
         delete_client(ci);
         return 1;
   }
}

/*
 * remove client from dictionary
 * */
void delete_client(client_info *ci) {

   if(ci->file_name)
      free(ci->file_name);

   dictionary_remove(fd_to_struct, &ci->fd);
   close(ci->fd);
   free(ci);
}

/* clean up and exit
 * */
void close_server(int retval) {
   retval = (retval == EXIT_FAILURE);

   if (dir_name) {
      //clean files
      if (file_name_set) {
         SET_FOR_EACH(file_name_set, f, {
            CHECK_ERROR(unlink(f));
         });
         set_destroy(file_name_set);
      }

      //move to the parent directory
      CHECK_ERROR(chdir("../"));

      //delete directory
      CHECK_ERROR(rmdir(dir_name));
   }

   if(fd_to_struct) {
      vector *values = dictionary_values(fd_to_struct);
      VECTOR_FOR_EACH(values, value, {
         client_info* c = (client_info *) value;
         delete_client(c);
      });
      dictionary_destroy(fd_to_struct);
      vector_destroy(values);
   }

   if(fd) {
      shutdown(fd, SHUT_RDWR);
      close(fd);
   }

   fprintf(stderr, "Shutting down...\n");
   exit(retval);
}


/****************
 * verb handles *
 ****************/

/*
 * handle DELETE request
 * */

int delete_function(void* event) {
   client_info * ci = (client_info *)event;
   if(!set_contains(file_name_set, ci->file_name)) {
      send_error(NSF, ci->fd);
      delete_client(ci);
      return 1;
   }

   set_remove(file_name_set, ci->file_name);
   total_file_size -= strlen(ci->file_name);

   if(-1 == unlink(ci->file_name))
      perror("");
   else
      write_all_to_socket(ci->fd, "OK\n", 3);
   delete_client(ci);
   return 1;
}

/*
 * handle PUT request
 * */

int put_function(void *event) {
   client_info *ci = (client_info *)event;

   if(ci->tmp) {
      ssize_t r = read_all_from_socket(ci->fd, (char *) (&(ci->file_size) + sizeof(size_t) - ci->tmp), ci->tmp);
      if(CHECK_BLOCK(r))
         return 0;
      ci->tmp -= r;
      if(ci->tmp)
         return 0;
   }

   //fprintf(stderr, "file_name: %s\n", ci->file_name);
   if(ci->finished_size)
      ci->file_fd = open(ci->file_name, O_APPEND | O_WRONLY, 0777);
   else
      ci->file_fd = open(ci->file_name, O_CREAT | O_TRUNC | O_WRONLY, 0777);

   if(-1 == ci->file_fd) {
      send_error(BR, ci->fd);
      delete_client(ci);
      return 1;
   }

   if(!set_contains(file_name_set, ci->file_name)) {
      set_add(file_name_set, ci->file_name);
      total_file_size += strlen(ci->file_name);
   }



   //increment written size
   ssize_t retval = 1;
   while(retval > 0) {
      retval = ff_nonblock(ci->fd, ci->file_fd, ci->file_size - ci->finished_size);

      if(retval > 0)
         ci->finished_size += retval;
   }

   close(ci->file_fd);

   //block case
   if(CHECK_BLOCK(retval))
      return 0;

   //finished case
   if(retval < 1) {
      if(0 == check_data_size(event))
         write_all_to_socket(ci->fd, "OK\n", 3);
      shutdown(ci->fd, SHUT_RDWR);
      delete_client(ci);
      return 1;
   }

   //ci->finished_size += retval;
   return 0;
}


/*
 * handle GET request
 * */
int get_function(void *event) {
   client_info *ci = (client_info *) event;
   if(!set_contains(file_name_set, ci->file_name)) {
      send_error(NSF, ci->fd);
      shutdown(ci->fd, SHUT_RDWR);
      delete_client(ci);
      return 1;
   }

   ci->file_fd = open(ci->file_name, O_RDONLY, 0777);
   if(-1 == ci->file_fd) {
      send_error(NSF, ci->fd);
      shutdown(ci->fd, SHUT_RDWR);
      delete_client(ci);
      return 1;
   }

   if(0 == ci->file_size) {
      struct stat sb;
      stat(ci->file_name, &sb);
      ci->file_size = sb.st_size;
      shutdown(ci->fd, SHUT_RD);
   }

   if(ci->col < 3) {
      if(0 == ci->col) {
         ssize_t r = write_all_to_socket(ci->fd, "OK\n", 3);
         if(CHECK_BLOCK(r))
            return 0;
         if(-1 == r) {
            shutdown(ci->fd, SHUT_RDWR);
            delete_client(ci);
            return 1;
         }
         ci->col += r;
      }
      else if(1 == ci->col) {
         ssize_t r = write_all_to_socket(ci->fd, "K\n", 2);
         if(CHECK_BLOCK(r))
            return 0;
         if(-1 == r) {
            shutdown(ci->fd, SHUT_RDWR);
            delete_client(ci);
            return 1;
         }
         ci->col += r;
      }
      else {
         ssize_t r = write_all_to_socket(ci->fd, "\n", 1);
         if(CHECK_BLOCK(r))
            return 0;
         if(-1 == r) {
            shutdown(ci->fd, SHUT_RDWR);
            delete_client(ci);
            return 1;
         }
         ci->col += r;
      }
      if(ci->col < 3)
         return 0;
   }

   //write_size();
   if(ci->tmp) {
      ssize_t r = write_all_to_socket(ci->fd, (char *) (&(ci->file_size) + sizeof(size_t) - ci->tmp), ci->tmp);
      if(CHECK_BLOCK(r))
         return 0;
      if(-1 == r) {
         shutdown(ci->fd, SHUT_RDWR);
         delete_client(ci);
         return 1;
      }
      ci->tmp -= r;
      if(ci->tmp)
         return 0;
   }

   lseek(ci->file_fd, ci->finished_size, SEEK_SET);

   ssize_t retval = 1;
   while(retval) {
      retval = ff_nonblock(ci->file_fd, ci->fd, ci->file_size - ci->finished_size);
      if(retval > 0)
         ci->finished_size += retval;
   }

   close(ci->file_fd);

   if(CHECK_BLOCK(retval))
      return 0;

   if(retval < 1) {
      shutdown(ci->fd, SHUT_WR);
      delete_client(ci);
      return 1;
   }

   //ci->finished_size += retval;

   return 0;
}

/*
 * handle LIST request
 * */
int list_function(void* event) {
   client_info *ci = (client_info *)event;
   int set_nums = set_cardinality(file_name_set);
   vector * files = set_elements(file_name_set);
   size_t retval = MAX((total_file_size + set_nums - 1), 0);

   if(0 == ci->file_fd) {
      while(CHECK_BLOCK(write_all_to_socket(ci->fd, "OK\n", 3)));
      ci->file_fd = 1;
      shutdown(ci->fd, SHUT_RD);
   }

   if(ci->tmp) {
      ssize_t r = write_all_to_socket(ci->fd, (char *) &(retval) + sizeof(size_t) - ci->tmp, ci->tmp);
      if(CHECK_BLOCK(r))
         return 0;
      ci->tmp -= r;
      if(ci->tmp)
         return 0;
   }

   while(ci->idx < set_nums) {
      char *file = vector_get(files, ci->idx);
      ssize_t v = 0;
      if((size_t)-1 != ci->col) {
         v = write_all_to_socket(ci->fd, file + ci->col, strlen(file) - ci->col);
         if(CHECK_BLOCK(v))
            return 0;
         if(-1 == v) {
            shutdown(ci->fd, SHUT_WR);
            delete_client(ci);
            return 1;
         }
         ci->col += v;
         if(ci->col != strlen(file))
            return 0;
         ci->idx++;
      }

      if(ci->idx < set_nums) {
         v = write_all_to_socket(ci->fd, "\n", 1);
         if(CHECK_BLOCK(v)) {
            ci->col = -1;
            return 0;
         }
         if(-1 == v) {
            shutdown(ci->fd, SHUT_WR);
            delete_client(ci);
            return 1;
         }
      }
      ci->col = 0;
   }

   shutdown(ci->fd, SHUT_WR);
   delete_client(ci);
   return 1;
}

/*
 * parse header
 * */
int parse_header(void *event) {

   //request: V_UNKNOWN
   client_info * ci = (client_info *) event;

   int client_fd = ci->fd;
   int idx = ci->idx;
   char current = 0;

   while(current != '\n' && idx < 276) {
      if(CHECK_BLOCK(read_all_from_socket(client_fd, &current, 1))) {
         ci->idx = idx;
         return 0;
      }
      else if(current)
         ci->header[idx++] = current;
   }

   if(276 == idx) {
      send_error(BR, ci->fd);
      return -1;
   }

   ci->idx = idx;

   //request: V_UNKNOWN, LIST, GET, PUT, DELETE
   if('\n' == current) {
      ci->idx = 0;
      if (0 == strncmp(ci->header, "LIST", 4))
         ci->request = LIST;
      else if (0 == strncmp(ci->header, "GET", 3))
         ci->request = GET;
      else if (0 == strncmp(ci->header, "PUT", 3))
         ci->request = PUT;
      else if (0 == strncmp(ci->header, "DELETE", 6))
         ci->request = DELETE;
      else {
         send_error(BR, ci->fd);
         return -1;
      }
   }

   //request: LIST, GET, PUT, DELETE
   if(ci->request != LIST)
      return get_file_name(event);

   return 0;
}

/*
 * send error to client
 * */
void send_error(code c, int client_fd) {
   write_all_to_socket(client_fd, "ERROR\n", 6);
   switch(c) {
      case TLD:
         print_too_little_data();
         write_all_to_socket(client_fd, err_bad_file_size, strlen(err_bad_file_size));
         break;
      case TMD:
         print_received_too_much_data();
         write_all_to_socket(client_fd, err_bad_file_size, strlen(err_bad_file_size));
         break;
      case BR:
         write_all_to_socket(client_fd, err_bad_request, strlen(err_bad_request));
         break;
      case NSF:
         write_all_to_socket(client_fd, err_no_such_file, strlen(err_no_such_file));
         break;
   }
}
