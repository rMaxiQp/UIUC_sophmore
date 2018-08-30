#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
main ()
{
  struct addrinfo hints, *result;

  memset (&hints, 0, sizeof (struct addrinfo));

  hints.ai_family = AF_INET6;
  hints.ai_socktype = SOCK_STREAM;

  int s = getaddrinfo ("illinois.edu", "80", &hints, &result);
  if (s != 0)
    {
      fprintf (stderr, "getaddrinfo: %s\n", gai_strerror (s));
      exit (1);
    }

  int sock_fd = socket (hints.ai_family, hints.ai_socktype, 0);
  connect (sock_fd, result->ai_addr, result->ai_addrlen);

//#define HELLO "GET /nosuchpagemwamwamwa.html HTTP/1.0\r\nHOST: illinois.edu\r\n\r\n"

//  write (sock_fd, HELLO, strlen (HELLO));
  dprintf(sock_fd, "GET / HTTP/1.0\r\n\r\n");
  char buffer[1000];
while(1) {
  ssize_t bytes = read (sock_fd, buffer, sizeof (buffer));
  if( bytes <1) break; // TODO Handle -1 and errno is EINTR
  write (1, buffer, bytes);
}  
  return 0;
}
