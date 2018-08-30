/**
* Text Editor Lab
* CS 241 - Spring 2018
*/

#include "sstream.h"
#include <string.h>
#include <assert.h>
#include <ctype.h>
#include <stdlib.h>

#define min(a,b) (a <= b ? a : b)
#define max(a,b) (a >= b ? a : b)

struct sstream {
   char* stream;
   size_t size;
   ssize_t ptr;
};

sstream *sstream_create(bytestring bytes) {
    // TODO implement
    sstream* ss = malloc(sizeof(sstream));
    assert(ss);
    //sstream_str(ss, bytes);
    if(bytes.str == NULL) {
       ss->stream = NULL;
       ss->size = 0;
    }
    else if (bytes.size < 0) {
       ss->size =  strlen(bytes.str);
       ss->stream = malloc(sizeof(char) * ss->size);
       memcpy(ss->stream, bytes.str, ss->size);
    }
    else {
       ss->size = bytes.size;
       ss->stream = malloc(sizeof(char) * ss->size);
       memcpy(ss->stream, bytes.str, ss->size);
    }
    ss->ptr = 0;
    return ss;
}

void sstream_destroy(sstream *this) {
    // TODO implement
    assert(this);
    free(this->stream);
    this->stream = NULL;
    free(this);
    this = NULL;
}

void sstream_str(sstream *this, bytestring bytes) {
    // TODO implement
    if(this == NULL || this->stream == NULL) {
       if(!this)
          this = malloc(sizeof(sstream));
       if (bytes.str == NULL) {
          this->stream = NULL;
          this->size = 0;
       }
       else if(bytes.size < 0) {
          this->size = strlen(bytes.str);
          this->stream = malloc(sizeof(char) * this->size);
          this->stream = memcpy(this->stream, bytes.str, this->size);
          //this->stream[this->size-1] = '\0';
       }
       else {
          this->size = bytes.size;
          this->stream = malloc(sizeof(char) * this->size);
          this->stream = memcpy(this->stream, bytes.str, this->size);
          //this->stream[this->size-1] = '\0';
       }
    }
    else if(bytes.str == NULL) {
       if(!this->stream)
          free(this->stream);
       this->stream = NULL;
       this->size = 0;
    }
    else if (bytes.size < 0) {
       this->size =  strlen(bytes.str);
       this->stream = realloc(this->stream, sizeof(char) * this->size);
       this->stream = memcpy(this->stream, bytes.str, this->size);
       //this->stream[this->size-1] = '\0';
    }
    else {
       this->size = bytes.size + 1;
       this->stream = realloc(this->stream, sizeof(char) * this->size);
       this->stream = memcpy(this->stream, bytes.str, this->size);
       //this->stream[this->size-1] = '\0';
    }
    this->ptr = 0;
}

bool sstream_eos(sstream *this) {
    // TODO implement
    return this->ptr >= (ssize_t) this->size;
}

char sstream_peek(sstream *this, ssize_t offset) {
    // TODO implement
    ssize_t going = this->ptr + offset;
    if(going < (ssize_t)this->size && going >= 0)
       return this->stream[going];
    return 0;
}

char sstream_getch(sstream *this) {
    // TODO implement
    assert(!sstream_eos(this));
    return this->stream[this->ptr++];
}

size_t sstream_size(sstream *this) {
    // TODO implement
    return this->size;
}

size_t sstream_tell(sstream *this) {
    // TODO implement
    return this->ptr;
}

int sstream_seek(sstream *this, ssize_t offset, int whence) {
    // TODO implement
    assert(this);
    ssize_t new_position = 0;
    if(whence == SEEK_END) new_position = this->size + offset;
    else if (whence == SEEK_CUR) new_position = this->ptr + offset;// + whence;
    else if (whence == SEEK_SET) new_position = offset;// + whence;

    if((ssize_t)this->size < new_position || new_position < 0) return -1;
    this->ptr = new_position;
    return 0;
}


size_t sstream_remain(sstream *this) {
    // TODO implement
    return this->size - this->ptr;
}

size_t sstream_read(sstream *this, bytestring *out, ssize_t count) {
    // TODO implement
    size_t count_t = 0;
    if(count >= 0) {
       count_t = min((size_t)labs(count) , sstream_remain(this));
    }
    else {
       count_t = min((size_t)labs(count), sstream_tell(this));
    }

    if(out->str == NULL) {
       out->str = calloc(1, (count_t + 1) *sizeof(char));
       if(!out->str) return -1;
       out->size = count_t;
    }
    else if (out->size < (ssize_t) count_t) {
       out->str = realloc(out->str, (count_t + 1)* sizeof(char));
       if(!out->str) return -1;
       out->size = count_t;
    }
    out->str[count_t] = '\0';

    if(count < 0) {
       size_t pos = this->ptr - count_t;
       memcpy(out->str, this->stream + pos, count_t);
    }
    else {
       memcpy(out->str, this->stream + this->ptr, count_t);
       this->ptr += count_t;
    }
    return count_t;
}

void sstream_append(sstream *this, bytestring bytes) {
    // TODO implement
    ssize_t bs = 0;
    if(bytes.size > 0) bs = bytes.size;
    else bs = strlen(bytes.str);
    this->stream = realloc(this->stream, (this->size + bs) * sizeof(char));
    for(size_t t = this->size; t < this->size + bs; t++) {
       this->stream[t] = bytes.str[t-this->size];
    }
    this->size += bs;
}

ssize_t sstream_subseq(sstream *this, bytestring bytes) {
    // TODO implement
    if(bytes.str == 0) return -1;
    size_t bs = strlen(bytes.str);
    if(bytes.size > 0) bs = bytes.size;

    if(bs > sstream_remain(this)) return -1;

    for(size_t t = this->ptr; t < this->size; t++) {
       if(this->stream[t] == bytes.str[0]) {
          int fail = 0;
          for(size_t i = 0; i < bs; i++) {
             if(this->stream[i+t] != bytes.str[i]) {
                fail = 1;
                break;
             }
          }
          if(!fail) {
             return t - this->ptr;
          }
       }
    }
    return -1;
}

size_t sstream_erase(sstream *this, ssize_t number) {
    // TODO implement
    assert(this);

    char* new_stream = NULL;
    ssize_t erase = 0;
    if(number > 0) {
       erase = min((size_t) number , sstream_remain(this));
       new_stream = malloc((this->size - erase) * sizeof(char));
       for(size_t t = 0; t < (size_t)this->ptr; t++) {
          new_stream[t] = this->stream[t];
       }
       for(size_t t = this->ptr + erase; t < this->size; t++) {
          new_stream[t - erase] = this->stream[t];
       }
    }
    else if(number < 0) {
       number = 0 - number;
       erase = min(number, this->ptr);
       new_stream = malloc(sizeof(char) * (this->size - erase + 1));
       for(size_t t = 0; t < (size_t)this->ptr - erase; t++) {
          new_stream[t] = this->stream[t];
       }
       for(size_t t = this->ptr - erase; t < this->size - erase; t++) {
          new_stream[t] = this->stream[t + erase];
       }
       this->ptr -= erase;
    }
    else
       return erase;
    this->size -= erase;
    free(this->stream);
    this->stream = new_stream;
    return erase;
}

void sstream_write(sstream *this, bytestring bytes) {
    // TODO implement
    assert(this);
    assert(bytes.str);

    ssize_t bs = 0;
    if(bytes.size > 0) bs = bytes.size;
    else bs = strlen(bytes.str);

    if(this->size < (size_t) (bs + this->ptr)) {
       this->stream = realloc(this->stream, (bs + this->ptr) * sizeof(char));
       this->size = bs + this->ptr;
    }
    for(ssize_t t = 0; t < bs; t++) {
       this->stream[this->ptr + t] = bytes.str[t];
    }
    //memcpy(this->stream + this->ptr, bytes.str, bs);
}

void sstream_insert(sstream *this, bytestring bytes) {
    // TODO implement
    assert(this);
    ssize_t bs = 0;
    if(bytes.size > 0) bs = bytes.size;
    else bs = strlen(bytes.str);

    char* dup = calloc(1, sizeof(char) * (this->size + bs) + 1);
    for(size_t t = 0; t < (size_t)this->ptr; t++)
    {
       dup[t] = this->stream[t];
    }
    for(size_t t = 0; t < (size_t)bs; t++) {
       dup[t+this->ptr] = bytes.str[t];
    }
    for(size_t t = this->ptr + bs; t < this->size + bs; t++) {
       dup[t] = this->stream[t - bs];
    }
    dup[bs + this->size] = 0;
    free(this->stream);
    this->stream = dup;
    this->size += bs;
}

int sstream_parse_long(sstream *this, long *out) {
    // TODO implement
    assert(this);

    char* current = this->stream + this->ptr;
    int to = 0;
    int signal = 1;
    if(isdigit(*current) || *current == '-') {
       if(*current == '-') {
          signal = 0;
          current++;
       }

       char replace = '\0';
       char temp = '\0';
       unsigned long long int var = 0;
       unsigned long long int prev = 0;
       while( (size_t)to <= sstream_remain(this) && isdigit(*(current + to))){
          temp = current[to];
          current[to] = replace;
          prev = var;
          var = strtoull(current, NULL, 10);
          current[to++] = temp;
          if(var > LONG_MAX) {
             if(signal)
                var = prev;
             else
                var = ~(prev) + 1;
             to--;
             break;
          }
       }
       if(isdigit(*(current + to)))  to--;
       else var = strtoull(current, NULL, 10);
       *out = var;
       this->ptr += to;
       return to;
    }
    return -1;
}
