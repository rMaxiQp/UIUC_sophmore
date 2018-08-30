/**
* Text Editor Lab
* CS 241 - Spring 2018
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "document.h"
#include "includes/vector.h"

struct document {
    vector *vector;
};

document *document_create() {
    document *this = (document *)malloc(sizeof(document));
    assert(this);
    this->vector = vector_create(string_copy_constructor, string_destructor,string_default_constructor);
    return this;
}

void document_write_to_file(document *this, const char *path_to_file) {
    assert(this);
    assert(path_to_file);
    // see the comment in the header file for a description of how to do this!
    FILE *dest = fopen(path_to_file, "w+");
    size_t size = vector_size(this->vector);
    char* string = "";
    for(size_t i = 0; i < size; i++) {
       string = (char*)vector_get(this->vector, i);
       fprintf(dest, "%s\n",  string);
    }
    //string = vector_get(this->vector, size - 1);
    //fprintf(dest, "%s", string);
    fclose(dest);
}

document *document_create_from_file(const char *path_to_file) {
    assert(path_to_file);
    // this function will read a file which is created by document_write_to_file
    FILE *path = fopen(path_to_file, "r");
    document *new_doc = document_create();
    if(!path) return document_create();
    char* buffer = NULL;
    size_t size = 0;
    ssize_t read = 0;
    while((read = getline(&buffer, &size, path)) != -1) {
       //fprintf(stderr, "while\n");
       if(read>0 && buffer[read-1] == '\n')
          buffer[read-1] = '\0';
       vector_push_back(new_doc->vector ,buffer);
    }
    free(buffer);
    fclose(path);
    return new_doc;
}

void document_destroy(document *this) {
    assert(this);
    vector_destroy(this->vector);
    free(this);
}

size_t document_size(document *this) {
    assert(this);
    return vector_size(this->vector);
}

void document_set_line(document *this, size_t line_number, const char *str) {
    assert(this);
    assert(str);
    size_t index = line_number - 1;
    vector_set(this->vector, index, (void *)str);
}

const char *document_get_line(document *this, size_t line_number) {
    assert(this);
    assert(line_number > 0);
    size_t index = line_number - 1;
    return (const char *)vector_get(this->vector, index);
}

void document_insert_line(document *this, size_t line_number, const char *str) {
    assert(this);
    assert(str);
    // TODO: your code here!
    // How are you going to handle the case when the user wants to
    // insert a line past the end of the document?
    size_t idx = line_number - 1;
    //char* dupe = strdup(str);
    if(document_size(this) <= line_number) {
       idx = document_size(this);
       vector_push_back(this->vector, (void*) str);
    }
    else {
       vector_insert(this->vector, line_number, (void*) str);
    }
    //free(dupe);
}

void document_delete_line(document *this, size_t line_number) {
    assert(this);
    assert(line_number > 0);
    size_t index = line_number - 1;
    vector_erase(this->vector, index);
}
