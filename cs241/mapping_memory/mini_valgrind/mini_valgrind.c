/**
* Mini Valgrind Lab
* CS 241 - Spring 2018
*/

#include "mini_valgrind.h"
#include <stdio.h>
#include <string.h>

meta_data *head = NULL;
size_t total_memory_requested = 0;
size_t total_memory_freed = 0;
size_t invalid_addresses = 0;

void *mini_malloc(size_t request_size, const char *filename,
                  void *instruction) {
    // your code here
    if(!request_size) return NULL;
    void* ptr = malloc(sizeof(meta_data) + request_size);
    if(!ptr) return NULL;
    meta_data* md = ptr;
    md->request_size = request_size;
    md->filename = filename;
    md->instruction = instruction;
    total_memory_requested += request_size;
    md->next = head;
    head = md;
    return ptr + sizeof(meta_data);
}

void *mini_calloc(size_t num_elements, size_t element_size,
                  const char *filename, void *instruction) {
    // your code here
    if(!num_elements || !element_size) return NULL;
    void* ptr = calloc(num_elements, sizeof(meta_data) + element_size);
    if(!ptr) return NULL;
    meta_data *md = ptr;
    md->request_size = num_elements * element_size;
    md->filename = filename;
    md->instruction = instruction;
    md->next = head;
    head = md;
    total_memory_requested += num_elements * element_size;
    return ptr + sizeof(meta_data);
}

void *mini_realloc(void *payload, size_t request_size, const char *filename,
                   void *instruction) {
    // your code here
    if(!payload)
       return mini_malloc(request_size, filename, instruction);
    if(!request_size) {
       mini_free(payload);
       return NULL;
    }
    meta_data *md = (meta_data *)(payload) - 1;
    meta_data *ptr = head;
    meta_data *ptr_prev = head;
    while(ptr) {
       if(ptr == md) {
          //fprintf(stderr, "payload%p , head%p,  ptr%p,   ptr_prev%p\n", payload, head, ptr, ptr_prev);
          md = realloc(md, request_size + sizeof(meta_data));
          if(!md) return NULL;
          //fprintf(stderr, "%zu\n",temp->request_size);
          if (md->request_size > request_size)
             total_memory_freed += (md->request_size - request_size);
          else
             total_memory_requested += (request_size - md->request_size);
          md->filename = filename;
          md->instruction = instruction;
          md->request_size = request_size;
          md->next = ptr->next;
          if(ptr_prev != ptr) ptr_prev->next = md;
          //fprintf(stderr, "temp addr: %p md addr: %p\n", temp, &md);
          return md + 1;
       }
       ptr_prev = ptr;
       ptr = ptr->next;
    }

    invalid_addresses ++;
    return NULL;
}

void mini_free(void *payload) {
    // your code here
    if(!payload) return;
    meta_data *ptr = payload - sizeof(meta_data);
    if(!ptr || !head) {
       invalid_addresses++;
       return;
    }

    meta_data *temp = head;
    meta_data *temp_prev = head;
    while(temp) {
       if(temp == ptr) {
          total_memory_freed += ptr->request_size;
          if(temp == temp_prev)
             head = head->next;
          else
             temp_prev->next = temp->next;
          free(ptr);
          return;
       }
       temp_prev = temp;
       temp = temp->next;
    }
    invalid_addresses++;
}
