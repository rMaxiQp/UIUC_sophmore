/**
* Malloc Lab
* CS 241 - Spring 2018
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct _meta_data {
   size_t size;
   void* addr;
   struct _meta_data *next;
   struct _meta_data *prev;
   int valid;
}meta_data;

static meta_data *free_list = NULL;
static int count = 0;

#define FREE 1
#define ALLOC 0

meta_data * split(size_t size) {
   meta_data* temp = free_list;
   meta_data* result = NULL;
   while(temp) {
      if(size <= temp->size) {
         result = temp;
         result->valid = ALLOC;
         if(free_list == result)
            free_list = NULL;
         else {
            if(result->next)
               result->next->prev = result->prev;
            else
               free_list = NULL;

            if(result->prev)
               result->prev->next = result->next;
         }
         break;
      }
      temp = temp->next;
   }
   return result;
}

void merge() {

   count --;
   if(count < 1)
      return;

    while(free_list && free_list->addr + free_list->size == sbrk(0)){
       meta_data * swap = free_list;
       if(free_list->next)
          free_list->next->prev = NULL;
       free_list = free_list->next;
       swap->next = NULL;
       if(brk(swap) == -1)
          break;
    }
}

/**
 * Allocate space for array in memory
 *
 * Allocates a block of memory for an array of num elements, each of them size
 * bytes long, and initializes all its bits to zero. The effective result is
 * the allocation of an zero-initialized memory block of (num * size) bytes.
 *
 * @param num
 *    Number of elements to be allocated.
 * @param size
 *    Size of elements.
 *
 * @return
 *    A pointer to the memory block allocated by the function.
 *
 *    The type of this pointer is always void*, which can be cast to the
 *    desired type of data pointer in order to be dereferenceable.
 *
 *    If the function failed to allocate the requested block of memory, a
 *    NULL pointer is returned.
 *
 * @see http://www.cplusplus.com/reference/clibrary/cstdlib/calloc/
  */
void *calloc(size_t num, size_t size) {
    size_t total = num * size;
    void * p = malloc(total);

    if(!p)
       return NULL;

    return memset(p, 0, total);
}

/**
 * Allocate memory block
 *
 * Allocates a block of size bytes of memory, returning a pointer to the
 * beginning of the block.  The content of the newly allocated block of
 * memory is not initialized, remaining with indeterminate values.
 *
 * @param size
 *    Size of the memory block, in bytes.
 *
 * @return
 *    On success, a pointer to the memory block allocated by the function.
 *
 *    The type of this pointer is always void*, which can be cast to the
 *    desired type of data pointer in order to be dereferenceable.
 *
 *    If the function failed to allocate the requested block of memory,
 *    a null pointer is returned.
 *
 * @see http://www.cplusplus.com/reference/clibrary/cstdlib/malloc/
 */
void *malloc(size_t size) {
   if(size == 0)
      return NULL;

   meta_data * result = split(size);
   count++;
   if(result == NULL) {
      result = sbrk(0);
      void* md = sbrk(size + sizeof(meta_data));
      if(md == (void*) -1)
         return NULL;
      result->size = size;
      result->addr = result + 1;
      result->valid = ALLOC;
      result->prev = NULL;
      result->next = NULL;
      return result->addr;
   }

   return result->addr;
}

/**
 * Deallocate space in memory
 i*
 * A block of memory previously allocated using a call to malloc(),
 * calloc() or realloc() is deallocated, making it available again for
 * further allocations.
 *
 * Notice that this function leaves the value of ptr unchanged, hence
 * it still points to the same (now invalid) location, and not to the
 * null pointer.
 *
 * @param ptr
 *    Pointer to a memory block previously allocated with malloc(),
 *    calloc() or realloc() to be deallocated.  If a null pointer is
 *    passed as argument, no action occurs.
 */
void free(void *ptr) {
    // implement free!
    if(!ptr) return;

    meta_data * to_free = ptr - sizeof(meta_data);

    if(to_free->valid == FREE)
       return;

    to_free->valid = FREE;
    to_free->prev = NULL;
    to_free->next = free_list;

    if(free_list)
       free_list->prev = to_free;
    free_list = to_free;

    merge();
}

/**
 * Reallocate memory block
 *
 * The size of the memory block pointed to by the ptr parameter is changed
 * to the size bytes, expanding or reducing the amount of memory available
 * in the block.
 *
 * The function may move the memory block to a new location, in which case
 * the new location is returned. The content of the memory block is preserved
 * up to the lesser of the new and old sizes, even if the block is moved. If
 * the new size is larger, the value of the newly allocated portion is
 * indeterminate.
 *
 * In case that ptr is NULL, the function behaves exactly as malloc, assigning
 * a new block of size bytes and returning a pointer to the beginning of it.
 *
 * In case that the size is 0, the memory previously allocated in ptr is
 * deallocated as if a call to free was made, and a NULL pointer is returned.
 *
 * @param ptr
 *    Pointer to a memory block previously allocated with malloc(), calloc()
 *    or realloc() to be reallocated.
 *
 *    If this is NULL, a new block is allocated and a pointer to it is
 *    returned by the function.
 *
 * @param size
 *    New size for the memory block, in bytes.


 *    If it is 0 and ptr points to an existing block of memory, the memory
 *    block pointed by ptr is deallocated and a NULL pointer is returned.
 *
 * @return
 *    A pointer to the reallocated memory block, which may be either the
 *    same as the ptr argument or a new location.
 *
 *    The type of this pointer is void*, which can be cast to the desired
 *    type of data pointer in order to be dereferenceable.
 *
 *    If the function failed to allocate the requested block of memory,
 *    a NULL pointer is returned, and the memory block pointed to by
 *    argument ptr is left unchanged.
 *
 * @see http://www.cplusplus.com/reference/clibrary/cstdlib/realloc/
 */
void *realloc(void *ptr, size_t size) {
    // implement realloc!
    if(!size) {
       free(ptr);
       return NULL;
    }
    if(!ptr)
       return malloc(size);

    meta_data *temp = (meta_data*) ptr - 1;

    if(temp->size > size)
       return ptr;

    void* result = malloc(size);

    if(!result)
       return NULL;

    memcpy(result, ptr, temp->size);

    free(ptr);

    return result;
}

