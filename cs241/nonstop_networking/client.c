/**
* Networking Lab
* CS 241 - Spring 2018
*/

#include "common.h"
#include "format.h"
#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <netdb.h>
#include <fcntl.h>

char **parse_args(int argc, char **argv);
verb check_args(char **args);
void execute(verb v, char **args);
void put(int fd, char **args);
void list(int fd, char **args);
void get(int fd, char **args);
void delete(int fd, char **args);
int connect_to_server(char *host, char *port);
char *parse_request(char *method, char *remote, char *local);
size_t get_response(int fd);

int main(int argc, char **argv) {
   // Good luck!
   char **parsed = parse_args(argc, argv);
   if(NULL == parsed) {
      print_client_usage();
      exit(1);
   }

   verb method = check_args(argv);
   execute(method, parsed);
   free(parsed);
}

/**
 * Given commandline argc and argv, parses argv.
 *
 * argc argc from main()
 * argv argv from main()
 *
 * Returns char* array in form of {host, port, method, remote, local, NULL}
 * where `method` is ALL CAPS
 */
char **parse_args(int argc, char **argv) {
    if (argc < 3) {
        return NULL;
    }

    char *host = strtok(argv[1], ":");
    char *port = strtok(NULL, ":");
    if (port == NULL) {
        return NULL;
    }

    char **args = calloc(1, 6 * sizeof(char *));
    args[0] = host;
    args[1] = port;
    args[2] = argv[2];
    char *temp = args[2];
    while (*temp) {
        *temp = toupper((unsigned char)*temp);
        temp++;
    }
    if (argc > 3) {
        args[3] = argv[3];
    }
    if (argc > 4) {
        args[4] = argv[4];
    }

    return args;
}

/**
 * Validates args to program.  If `args` are not valid, help information for the
 * program is printed.
 *
 * args     arguments to parse
 *
 * Returns a verb which corresponds to the request method
 */
verb check_args(char **args) {
    if (args == NULL) {
        print_client_usage();
        exit(1);
    }

    char *command = args[2];

    if (strcmp(command, "LIST") == 0) {
        return LIST;
    }

    if (strcmp(command, "GET") == 0) {
        if (args[3] != NULL && args[4] != NULL) {
            return GET;
        }
        print_client_help();
        exit(1);
    }

    if (strcmp(command, "DELETE") == 0) {
        if (args[3] != NULL) {
            return DELETE;
        }
        print_client_help();
        exit(1);
    }

    if (strcmp(command, "PUT") == 0) {
        if (args[3] == NULL || args[4] == NULL) {
            print_client_help();
            exit(1);
        }
        return PUT;
    }

    // Not a valid Method
    print_client_help();
    exit(1);
}


/*
 * execute each method
 * */
void execute(verb v, char** args) {
   int fd = connect_to_server(args[0], args[1]);
   switch(v) {
      case LIST:
         list(fd, args);
         break;
      case GET:
         get(fd, args);
         break;
      case PUT:
         put(fd, args);
         break;
      case DELETE:
         delete(fd, args);
         break;
      default:
         print_client_help();
         break;
   }
}

//char* array in form of {host, port, method, remote, local, NULL}

/*
 * function for LIST
 * */
void list(int fd, char **args) {
   char *method = args[2];
   char *request = parse_request(method, NULL, NULL);
   CHECK_WRITE(write_all_to_socket(fd, request, strlen(request)));
   free(request);
   shutdown(fd, SHUT_WR);
   fprintf(stderr, "Wrote 5 bytes for LIST\n");
   size_t file_size = get_response(fd);

   fprintf(stderr, "Expecting %ld bytes from server\n", file_size);
   size_t s = 0;
   size_t size = 0;
   char temp[CAPACITY];
   for(ssize_t i = file_size; i > 0; i -= CAPACITY) {
      s = read_all_from_socket(fd, temp, MIN(CAPACITY, i));
      size += s;
      write_all_to_socket(STDOUT_FILENO, temp, s);
   }

   fprintf(stderr, "received %ld bytes from server\n", size);

   if(size < file_size)
      print_too_little_data();
   char try = 0;
   if(size > file_size || (1 == read_all_from_socket(fd, &try, 1)))
      print_received_too_much_data();
}

/*
 * function for GET
 * */
void get(int fd, char **args) {
   char *method = args[2];
   char *remote = args[3];
   char *local = args[4];
   char *request = parse_request(method, remote, NULL);
   CHECK_WRITE(write_all_to_socket(fd, request, strlen(request)));
   free(request);
   shutdown(fd, SHUT_WR);
   size_t file_size = get_response(fd);
   fprintf(stderr, "Expecting %ld bytes from server\n", file_size);
   int file = open(local, O_CREAT | O_WRONLY | O_TRUNC, S_IRWXU | S_IRWXG | S_IWOTH);
   if(-1 == file) {
      perror("open");
      exit(0);
   }

   size_t size = file_to_file(fd, file, file_size);

   if(size < file_size)
      print_too_little_data();

   if(size > file_size) //|| 0 !=  read_all_from_socket(fd, &buffer, 1))
      print_received_too_much_data();

   close(file);
}

/*
 * function for PUT
 * */
void put(int fd, char **args) {
   char *remote = args[3];
   char *local = args[4];

   printf("Sent %lu bytes for first line of header\n", strlen(remote) + 5);
   struct stat f_st;
   if(-1 == stat(local, &f_st)) {
      perror("open");
      exit(0);
   }
   int file = open(local, O_RDONLY);
   size_t file_size = f_st.st_size;

   printf("File_size: %ld\n", file_size);
   //VERB [filename]\n
   size_t request_size = 5 + strlen(remote); //3(PUT) + 1( ) + 1(\n = 5
   char *request = calloc(1, request_size * sizeof(char));
   memcpy(request, "PUT ", 4);
   memcpy(request + 4, remote, strlen(remote));
   memcpy(request + 4 + strlen(remote), "\n", 1);

   write_all_to_socket(fd, request, request_size);
   write_all_to_socket(fd, (char *)&f_st.st_size, sizeof(size_t));

   file_to_file(file, fd, file_size);

   printf("Sent %ld bytes of file\n", file_size);
   free(request);
   close(file);
   shutdown(fd, SHUT_WR);
   get_response(fd);
   print_success();
}

/*
 * function for DELETE
 * */
void delete(int fd, char **args) {
   char *method = args[2];
   char *remote = args[3];
   char *request = parse_request(method, remote, NULL);
   CHECK_WRITE(write_all_to_socket(fd, request, strlen(request)));
   printf("Sent %lu bytes for first line of header\n", strlen(remote) + 6);
   free(request);
   shutdown(fd, SHUT_WR);
   get_response(fd);
   print_success();
}

/*
 * let client connect to server
 * */
int connect_to_server(char *host, char *port) {
   int s = 0;
   int sock_fd = socket(AF_INET, SOCK_STREAM, 0);

   struct addrinfo hints, *result;
   memset(&hints, 0, sizeof(struct addrinfo));
   hints.ai_family = AF_INET; //IPv4
   hints.ai_socktype = SOCK_STREAM; //TCP

   s = getaddrinfo(host, port, &hints, &result);
   if(0 != s) {
      fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(s));
      exit(EXIT_FAILURE);
   }

   if(-1 == connect(sock_fd, result->ai_addr, result->ai_addrlen)) {
      perror("connect");
      exit(EXIT_FAILURE);
   }

   freeaddrinfo(result);
   return sock_fd;
}

/*
 * create client request (No PUT case)
 *
 * char* array in form of {host, port, method, remote, local, NULL}
 * */
char *parse_request(char *method, char *remote, char *local) {
   // VERB [filename]\n

   // ||VERB ||
   int request_length = strlen(method) + 3;//( ) + (\n) + (\0)
   char *request = malloc(request_length * sizeof(char));
   memcpy(request, method, strlen(method));
   request[strlen(method)] = ' ';

   // ||VERB [filename]||
   if(local) {//GET
      request_length += strlen(local);
      request = realloc(request, request_length);
      memcpy(request + strlen(method) + 1, local, strlen(local));
   } else if(remote) {//DELETE
      request_length += strlen(remote);
      request = realloc(request, request_length);
      memcpy(request + strlen(method) + 1, remote, strlen(remote));
   }

   // ||VERB [filename]\n||
   memcpy(request + request_length - 2, "\n", 1);
   request[request_length-1] = '\0';
   return request;
}

/*
 * receive response from server
 * */
size_t get_response(int fd) {
   fprintf(stderr, "processing response\n");

   char buffer[3];
   CHECK_READ(read_all_from_socket(fd, buffer, 3));

   if(0 == strncmp(buffer, "OK\n", 3)) {//print OK
      fprintf(stderr, "STATUS_OK\n");

      size_t msg_size = 0;
      CHECK_READ(read_all_from_socket(fd, (char*)&msg_size, sizeof(size_t)));

      return msg_size;
   }

   //print error
   fprintf(stderr, "STATUS_ERROR\n");
   char temp = 0;
   size_t retval = 0;
   while('\n' != temp) {
      retval = read_all_from_socket(fd, &temp, 1);
      if(0 == retval) {
	 print_connection_closed();
	 exit(EXIT_FAILURE);
      } else if((size_t)-1 == retval) {
	 perror("");
	 exit(EXIT_FAILURE);
      }
   }
   char error_msg[256];
   int size = 0;
   while(size < 256) {
      CHECK_READ(read_all_from_socket(fd, error_msg + size, 1));
      if('\n' == error_msg[size]) {
	 error_msg[size] = '\0';
	 break;
      }
      size++;
   }

   print_error_message(error_msg);
   exit(EXIT_FAILURE);
}

