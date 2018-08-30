/**
* Networking Lab
* CS 241 - Spring 2018
*/

#pragma once
#include <stddef.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

#define LOG(...)                      \
    do {                              \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n");        \
    } while (0);

#define MAX(a, b) (a > b ? a : b)
#define MIN(a, b) (a < b ? a : b)
#define CHECK_READ(a) if(-1 == a) { 			\
			perror("read_all_from_socket");   \
			exit(1);}
#define CHECK_WRITE(a) if(-1 == a) { 			\
			perror("write_all_to_socket");   \
			exit(1);}
#define CHECK_BLOCK(a) ((ssize_t)-1 == a && ( EAGAIN == errno || EWOULDBLOCK ==  errno))

#define CHECK_ERROR(a) if(-1 == a) {     \
                        perror("");        \
                        exit(EXIT_FAILURE);}
#define SERVER_COND(a)  if(a < 0) {     \
                        perror("");        \
                        close_server(EXIT_FAILURE);} \

#define CAPACITY 2048

typedef enum { GET, PUT, DELETE, LIST, V_UNKNOWN } verb;
/*
 * NSF No such file
 * TMD Too much data
 * TLD Too little data
 * BR Bad request
 * */
typedef enum { NSF, TMD, TLD, BR } code;

static const size_t MESSAGE_SIZE_DIGITS = 4;

/**
 * The largest size the message can be that a client
 * sends to the server.
 */
#define MSG_SIZE (256)

/**
 * Builds a message in the form of
 * <name>: <message>\n
 *
 * Returns a char* to the created message on the heap
 */
char *create_message(char *name, char *message);

/**
 * Read the first four bytes from socket and transform it into ssize_t
 *
 * Returns the size of the incomming message,
 * 0 if socket is disconnected, -1 on failure
 */
ssize_t get_message_size(int socket);

/**
 * Writes the bytes of size to the socket
 *
 * Returns the number of bytes successfully written,
 * 0 if socket is disconnected, or -1 on failure
 */
ssize_t write_message_size(size_t size, int socket);

/**
 * Attempts to read all count bytes from socket into buffer.
 * Assumes buffer is large enough.
 *
 * Returns the number of bytes read, 0 if socket is disconnected,
 * or -1 on failure.
 */
ssize_t read_all_from_socket(int socket, char *buffer, size_t count);

/**
 * Attempts to write all count bytes from buffer to socket.
 * Assumes buffer contains at least count bytes.
 *
 * Returns the number of bytes written, 0 if socket is disconnected,
 * or -1 on failure.
 */
ssize_t write_all_to_socket(int socket, const char *buffer, size_t count);


/**
 * read from reader and write to writer
 *
 * Returns the number of bytes transferred
 */

ssize_t file_to_file(int reader, int writer, size_t file_size);

/**
 * read from reader and write to writer
 *
 * Returns the number of bytes transferred
 *
 * NONBLOCKING
 */
ssize_t ff_nonblock(int reader, int writer, size_t file_size);
