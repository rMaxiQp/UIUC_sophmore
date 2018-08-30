#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
// gcc -std=gnu99 recvfrom.c -o recvfrom

void quit(char* mesg) {
    perror(mesg);
    exit(1);
}

int main(int argc, char** argv) {
    if(argc != 2) {
        printf("Usage: %s port\n", argv[0]);
        exit(1);
    }

    struct addrinfo hints, *result= 0;
    memset(&hints,0,sizeof(hints));
    hints.ai_family=AF_INET;
    hints.ai_socktype=SOCK_DGRAM;
    hints.ai_flags=AI_PASSIVE;

    char* hostname = NULL;
    char *portname = argv[1];

    int err=getaddrinfo(hostname,portname,&hints,&result);
    if(err) quit("getaddrinfo");

    int fd=socket(result->ai_family,result->ai_socktype,result->ai_protocol);
    if (fd==-1) quit("socket");

    if (bind(fd,result->ai_addr,result->ai_addrlen)==-1)
        quit("bind");

    freeaddrinfo(result);

    char buffer[1024];

    struct sockaddr_storage source;
    socklen_t  source_len = sizeof(source);

    while(1) {
        printf("Listening on port %s\n", portname);

        ssize_t bytes_recd=recvfrom(fd,buffer,sizeof(buffer),0,(struct sockaddr*)&source,&source_len);
        if (bytes_recd==-1) quit("recvfrom");
        if(bytes_recd == source_len)
            printf("Datagram > buffer - message truncated\n");

        // Print buffer contents
        write(1, buffer, bytes_recd);
        write(1, "\n",1);

        // Encrypt the message
        for(int i=0; i < bytes_recd; i++) {
            if( buffer[i] >= 64) buffer[i] ^= 1;
        }

        int flags = 0;

        size_t bytes_sent = sendto(fd, buffer, bytes_recd, flags, (struct sockaddr*) &source, source_len);
        if(bytes_sent==-1) {
            quit("sendto");
        }

        if(bytes_sent == bytes_recd ) printf("Replied\n");
        else  quit("write");
    }
    close(fd);
    puts("\nFinished");
    return 0;
}


