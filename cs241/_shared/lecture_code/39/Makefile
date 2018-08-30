#OPTS= -fno-common -std=gnu99 -Wall -g -D_GNU_SOURCE
OPTS= -std=gnu99 -Wall -g -D_GNU_SOURCE
CC=gcc

all: recvfrom sendto challenge hashfind

recvfrom: recvfrom.c
	$(CC) $(OPTS) recvfrom.c -o recvfrom

sendto: sendto.c
	$(CC)  $(OPTS)  sendto.c -o sendto

challenge: challenge.c
	$(CC)  $(OPTS)  challenge.c -o challenge 

hashfind: hashfind.c
	$(CC)  $(OPTS)  hashfind.c -o hashfind 

clean:
	rm -rf recvfrom sendto hashfind challenge *.dSYM

pretty:
	astyle -n *.c
