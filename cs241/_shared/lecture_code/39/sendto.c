#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
// gcc -std=gnu99    sendto.c -o sendto

void quit(char* mesg) {
    perror(mesg);
    exit(1);
}

int main(int argc, char**argv) {
    if(argc !=4 ) {
        printf("Usage: %s host port message\n", argv[0]);
        exit(1);
    }

    struct addrinfo hints, *result= 0;
    memset(&hints,0,sizeof(hints));

    hints.ai_family=AF_INET;
    hints.ai_socktype=SOCK_DGRAM;
    hints.ai_flags=0;

    char* hostname = argv[1];
    char *portname = argv[2];
    char *message = argv[3];

    int err=getaddrinfo(hostname,portname,&hints,&result);
    if(err) quit("getaddrinfo");

    int fd=socket(result->ai_family,result->ai_socktype,result->ai_protocol);
    if (fd==-1) quit("socket");

    int flags = 0;

    size_t bytes_sent = sendto(fd,message,strlen(message) ,flags, result->ai_addr,result->ai_addrlen);
    if(bytes_sent==-1) {
        quit("sendto");
    }
    printf("Datagram sent (%zu bytes)!\nWaiting for response...\n", bytes_sent) ;

    char buffer[1024];
    struct sockaddr_storage source;

    socklen_t  source_len = sizeof(source);

    size_t bytes_recd = recvfrom(fd,buffer,sizeof(buffer),0,(struct sockaddr*)&source,&source_len);

    if(bytes_recd>0) write(1,buffer,bytes_recd);
    else quit("recvfrom");

    freeaddrinfo(result);

    puts("\nFinished");
    return 0;
}


