/**
* Chatroom Lab
* CS 241 - Spring 2018
*/

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "includes/utils.h"

#define MAX_CLIENTS 8

void *process_client(void *p);

static volatile int serverSocket;
static volatile int endSession;

static volatile int clientsCount;
static volatile int clients[MAX_CLIENTS];

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

/**
 * Signal handler for SIGINT.
 * Used to set flag to end server.
 */
void close_server() {
    endSession = 1;
    // add any additional flags here you want.
}

/**
 * Cleanup function called in main after `run_server` exits.
 * Server ending clean up (such as shutting down clients) should be handled
 * here.
 */
void cleanup() {
    if (shutdown(serverSocket, SHUT_RDWR) != 0) {
        perror("shutdown():");
    }
    close(serverSocket);

    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (clients[i] != -1) {
            if (shutdown(clients[i], SHUT_RDWR) != 0) {
                perror("shutdown(): ");
            }
            close(clients[i]);
        }
    }
}

/**
 * Sets up a server connection.
 * Does not accept more than MAX_CLIENTS connections.  If more than MAX_CLIENTS
 * clients attempts to connects, simply shuts down
 * the new client and continues accepting.
 * Per client, a thread should be created and 'process_client' should handle
 * that client.
 * Makes use of 'endSession', 'clientsCount', 'client', and 'mutex'.
 *
 * port - port server will run on.
 *
 * If any networking call fails, the appropriate error is printed and the
 * function calls exit(1):
 *    - fprtinf to stderr for getaddrinfo
 *    - perror() for any other call
 */
void run_server(char *port) {
   /*QUESTION 1*/ //an internal endpoit for sending or receiving data
   /*QUESTION 2*/ //AF_INET is bound to an IP address, while AF_UNIX is bound to a special file on the filesystem
   /*QUESTION 3*/ //SOCK_STREAM is for TCP and SOCK_DGRAM is for UDP

   /*QUESTION 8*/ //set option, protocol level and value to the specified socket_fd

   /*QUESTION 4*/ //to clear out the garbage value
   /*QUESTION 5*/ //ai_family specifies the desired address family for the returned address, ai_socktypespecifies the desired socket type

   /*QUESTION 6*/ //getaddrinfo takes node and service, which identify an Internet host and service, and returns addrinfo structures, each contains an Internet address that can be specified in bind or connect

   /*QUESTION 9*/ //assign a local socket address to an identified socket which has no local socket address

   /*QUESTION 10*/ //mark a connection-mode socket as accepting connections
   int s = 0;
   int sock_fd = socket(AF_INET, SOCK_STREAM, 0);

   serverSocket = sock_fd;

   int optval = 1;
   if(setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &optval,sizeof(optval))  < 0) {
      perror("socketopt(REUSEADDR)");
   }
   if(setsockopt(sock_fd, SOL_SOCKET, SO_REUSEPORT, &optval,sizeof(optval))  < 0) {
      perror("socketopt(REUSEPORT)");
   }
   struct addrinfo hints, *result;
   memset(&hints, 0, sizeof(struct addrinfo));
   hints.ai_family = AF_INET; //IPv4
   hints.ai_socktype = SOCK_STREAM; //TCP
   hints.ai_flags = AI_PASSIVE;

   //getaddrinfo
   s = getaddrinfo(NULL, port, &hints, &result);
   if(s != 0) {
      fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(s));
      exit(1);
   }

   //bind
   if(bind(sock_fd, result->ai_addr, result->ai_addrlen) != 0) {
      perror("bind()");
      exit(1);
   }

   //listen
   if(listen(sock_fd, MAX_CLIENTS) != 0) {
      perror("listen()");
      exit(1);
   }

   struct sockaddr_in *result_addr = (struct sockaddr_in *) result->ai_addr;
   printf("listening on file descriptor %d, port %d\n", sock_fd, ntohs(result_addr->sin_port));

   for(int i = 0; i < MAX_CLIENTS; i++)
      clients[i] = -1;

   while(!endSession) {
      printf("waiting for connection...\n");

      struct sockaddr_in client;
      memset(&client, 0, sizeof(struct sockaddr_in));
      socklen_t size = sizeof(struct sockaddr_in);
      int client_fd = accept(sock_fd, &client, &size);
      if(endSession)
         break;

      pthread_mutex_lock(&mutex);
      if(clientsCount == MAX_CLIENTS) {
         close(client_fd);
         pthread_mutex_unlock(&mutex);
         continue;
      }
      ssize_t idx = 0;
      for(; idx < clientsCount; idx++) {
         if(clients[idx] == -1) break;
      }
      printf("Client %lu joined on %s\n", idx, inet_ntoa(client.sin_addr));
      clients[idx] = client_fd;
      clientsCount++;
      printf("Currently serving %d clients\n", clientsCount);
      pthread_mutex_unlock(&mutex);

      pthread_t id;
      pthread_create(&id, NULL, process_client, (void*)idx);
   }
   freeaddrinfo(result);
}

/**
 * Broadcasts the message to all connected clients.
 *
 * message  - the message to send to all clients.
 * size     - length in bytes of message to send.
 */
void write_to_clients(const char *message, size_t size) {
    pthread_mutex_lock(&mutex);
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (clients[i] != -1) {
            ssize_t retval = write_message_size(size, clients[i]);
            if (retval > 0) {
                retval = write_all_to_socket(clients[i], message, size);
            }
            if (retval == -1) {
                perror("write(): ");
            }
        }
    }
    pthread_mutex_unlock(&mutex);
}

/**
 * Handles the reading to and writing from clients.
 *
 * p  - (void*)intptr_t index where clients[(intptr_t)p] is the file descriptor
 * for this client.
 *
 * Return value not used.
 */
void *process_client(void *p) {
    pthread_detach(pthread_self());
    intptr_t clientId = (intptr_t)p;
    ssize_t retval = 1;
    char *buffer = NULL;

    while (retval > 0 && endSession == 0) {
        retval = get_message_size(clients[clientId]);
        if (retval > 0) {
            buffer = calloc(1, retval);
            retval = read_all_from_socket(clients[clientId], buffer, retval);
        }
        if (retval > 0)
            write_to_clients(buffer, retval);

        free(buffer);
        buffer = NULL;
    }

    printf("User %d left\n", (int)clientId);
    close(clients[clientId]);

    pthread_mutex_lock(&mutex);
    clients[clientId] = -1;
    clientsCount--;
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "%s <port>\n", argv[0]);
        return -1;
    }

    struct sigaction act;
    memset(&act, '\0', sizeof(act));
    act.sa_handler = close_server;
    if (sigaction(SIGINT, &act, NULL) < 0) {
        perror("sigaction");
        return 1;
    }

    // signal(SIGINT, close_server);
    run_server(argv[1]);
    cleanup();
    pthread_exit(NULL);
}
