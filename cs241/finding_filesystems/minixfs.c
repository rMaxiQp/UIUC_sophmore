/**
* Finding Filesystems Lab
* CS 241 - Spring 2018
*/

#include "minixfs.h"
#include <stdio.h>
#include <errno.h>
#include <string.h>

#define MAX(a,b)((a > b) ? a : b)
void *get_block_start(file_system *fs, inode *file, uint64_t block_index);
//with help from hetian huo, yuchen li

int minixfs_chmod(file_system *fs, char *path, int new_permissions) {
    // Thar she blows!
    inode *i = get_inode(fs, path);
    if(NULL == i) {
       errno = ENOENT;
       return -1;
    }
    i->mode = new_permissions | ((i->mode >> RWX_BITS_NUMBER )<< RWX_BITS_NUMBER);

    return clock_gettime(CLOCK_REALTIME, &(i->ctim));
}

int minixfs_chown(file_system *fs, char *path, uid_t owner, gid_t group) {
    // Land ahoy!
    inode *i = get_inode(fs, path);
    if(NULL == i) {
       errno = ENOENT;
       return -1;
    }
    if(owner != (uid_t) - 1)
       i->uid = owner;
    if(group != (gid_t) - 1)
       i->gid = group;
    return clock_gettime(CLOCK_REALTIME, &i->ctim);
}

//with the help from haozhe wang
/*    minixfs_min_blockcount
 *    add_data_block_to_inode
 *    add_single_indirect_block
 *    add_data_block_to_indirect_block
 * */
ssize_t minixfs_write(file_system *fs, const char *path, const void *buf,
                      size_t count, off_t *off) {
    // X marks the spot
    size_t count_default = count;
    unsigned long block_count = (count + *off) / (16 * KILOBYTE);
    if((count + *off) % (16 * KILOBYTE) != 0) block_count++;
    if(block_count > NUM_DIRECT_INODES + NUM_INDIRECT_INODES) {
       errno = ENOSPC;
       return -1;
    }
    if(-1 == minixfs_min_blockcount(fs, path, block_count)) {
       errno = ENOSPC;
       return -1;
    }

    inode *f_inode = get_inode(fs, path);
    int data_block_id = *off / (16 * KILOBYTE);
    int completed = 0;
    if(data_block_id < NUM_DIRECT_INODES) {
      int true_offset = *off / (16 * KILOBYTE);
      for(int i = data_block_id; i < NUM_DIRECT_INODES; i++) {
         size_t write_amount = 16 * KILOBYTE - true_offset;
         if(count > 16 * KILOBYTE) {
            memcpy(&fs->data_root[f_inode->direct[i]].data[true_offset], buf + completed, write_amount);
            completed += write_amount;
            true_offset = 0;
            count -= write_amount;
         }
         else {
            memcpy(&fs->data_root[f_inode->direct[i]].data[true_offset], buf + completed, count);
            completed += count;
            true_offset = 0;
            count -= count;
            break;
         }
      }
      if(count > 0) {
         data_block indir = fs->data_root[f_inode->indirect];
         for(int i = 0; i < (int) NUM_INDIRECT_INODES; i++) {
            size_t write_amount = 16 * KILOBYTE - true_offset;
            int this_block_num = indir.data[i * 4];
            if(count > 16 * KILOBYTE) {
               memcpy(&fs->data_root[this_block_num].data[0], buf + completed, write_amount);
               completed += write_amount;
               count -= write_amount;
            }
            else {
               memcpy(&fs->data_root[this_block_num].data[0], buf + completed, count);
               completed += count;
               count -= count;
               break;
            }
         }
      }
    }
    else {
       int remain = *off - NUM_DIRECT_INODES * 16 *KILOBYTE;
       data_block indir = fs->data_root[f_inode->indirect];
       int indire_block_id = remain / (16 * KILOBYTE);
       int true_offset = remain % (16 * KILOBYTE);
       for(int i = indire_block_id; i <(int) NUM_INDIRECT_INODES;i++) {
          size_t write_amount = 16 * KILOBYTE - true_offset;
          int this_block_num = indir.data[i*4];
          if(count > 16 * KILOBYTE) {
             memcpy(&fs->data_root[this_block_num].data[true_offset], buf + completed, write_amount);
             completed += write_amount;
             true_offset = 0;
             count -= write_amount;
          }
          else {
             memcpy(&fs->data_root[this_block_num].data[true_offset], buf + completed, count);
             completed += count;
             true_offset = 0;
             count -= count;
             break;
          }
       }
    }
    f_inode->size = MAX(*off + count_default, f_inode->size);
    if(clock_gettime(CLOCK_REALTIME, &f_inode->mtim) == -1)
       return -1;
    if(clock_gettime(CLOCK_REALTIME, &f_inode->atim) == -1)
       return -1;
    *off += completed;
    return completed;
}

ssize_t minixfs_read(file_system *fs, const char *path, void *buf, size_t count,
                     off_t *off) {
    // 'ere be treasure!
    inode *i = get_inode(fs, path);
    if(NULL == i) {
       errno = ENOENT;
       return -1;
    }
    if(i->size < (uint32_t)*off)
       return 0;

    count = (*off + count) > i->size ? (i->size - *off) : count;
    uint64_t read = 0;
    uint64_t block_idx  = *off / sizeof(data_block);
    size_t read_size = sizeof(data_block) - (*off) % sizeof(data_block);
    read_size = (read_size < count) ? read_size : count;
    char * start =  (char *)get_block_start(fs, i, block_idx) + (*off) % sizeof(data_block);
    read_size = ((*off + read_size) < i->size) ? read_size : i->size - *off;
    memmove(buf, start, read_size);
    read += read_size;
    block_idx++;
    *off += read_size;

    while(read < count) {
      read_size = (count - read) > sizeof(data_block) ? sizeof(data_block): (count-read);
      start = (char *)get_block_start(fs, i, block_idx);
      memmove(buf + read, start, read_size);
      read += read_size;
      block_idx++;
      *off += read_size;
    }


    //clock_gettime(CLOCK_REALTIME, &i->mtim);
    clock_gettime(CLOCK_REALTIME, &i->atim);
    return read;
}

void *get_block_start(file_system *fs, inode *file, uint64_t block_index){
   data_block_number *block_array;
   if(block_index < NUM_DIRECT_INODES)
      block_array = file->direct;
   else {
      block_index -= NUM_DIRECT_INODES;
      block_array = (data_block_number *) (fs->data_root + file->indirect);
   }
   return(void *) (fs->data_root + block_array[block_index]);
}
