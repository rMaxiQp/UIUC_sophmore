/**
* Finding Filesystems Lab
* CS 241 - Spring 2018
*/

#include "minixfs.h"
#include "minixfs_utils.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/stat.h>
#include <unistd.h>
#define MIN(x, y) (x < y ? x : y)
#define MAX(x, y) (x > y ? x : y)

int minixfs_readdir(file_system *fs, const char *path,
                    struct dirent **entries) {
    inode *node = get_inode(fs, path);
    if (!node) {
        errno = ENOENT;
        return -1;
    }
    clock_gettime(CLOCK_REALTIME, &node->atim);
    if (is_file(node)) {
        errno = ENOTDIR;
        return -1;
    }
    assert(*entries == NULL);
    uint64_t size = node->size, max_entries = node->size / FILE_NAME_ENTRY;
    *entries = malloc(sizeof(struct dirent) * max_entries);
    int num_entries = 0;
    size_t count = 0;
    minixfs_dirent dirents;
    data_block_number *block_array = node->direct;
    while (size > 0) {
        uint64_t temp = 0;
        data_block *blocky = fs->data_root + block_array[count];
        while (temp < sizeof(data_block) && temp < size) {
            make_dirent_from_string(((char *)blocky) + temp, &dirents);

            int length = MIN(strlen(dirents.name) + 1, MAX_DIR_NAME_LEN);
            memcpy((*entries)[num_entries].d_name, dirents.name, length);
            (*entries)[num_entries].d_name[length] = '\0';
            (*entries)[num_entries].d_ino = dirents.inode_num;
            num_entries++;

            temp += FILE_NAME_ENTRY;
        }
        count++;
        size -= temp;
        if (count == NUM_DIRECT_INODES) {
            if (node->indirect == UNASSIGNED_NODE)
                break;
            block_array = (data_block_number *)(fs->data_root + node->indirect);
            count = 0;
        }
    }
    return num_entries;
}

data_block_number add_data_block_to_inode(file_system *fs_pointer,
                                          inode *node) {
    assert(fs_pointer);
    assert(node);

    int i;
    for (i = 0; i < NUM_DIRECT_INODES; ++i) {
        if (node->direct[i] == -1) {
            data_block_number first_data = first_unused_data(fs_pointer);
            if (first_data == -1) {
                return -1;
            }
            node->direct[i] = first_data;
            set_data_used(fs_pointer, first_data, 1);
            return first_data;
        }
    }
    return 0;
}

data_block_number add_data_block_to_indirect_block(file_system *fs_pointer,
                                                   data_block_number *blocks) {
    assert(fs_pointer);
    assert(blocks);

    size_t i;
    for (i = 0; i < NUM_INDIRECT_INODES; ++i) {
        if (blocks[i] == UNASSIGNED_NODE) {
            data_block_number first_data = first_unused_data(fs_pointer);
            if (first_data == -1) {
                return -1;
            }
            blocks[i] = first_data;
            set_data_used(fs_pointer, first_data, 1);
            return first_data;
        }
    }
    return 0;
}

inode_number add_single_indirect_block(file_system *fs_pointer, inode *node) {
    assert(fs_pointer);
    assert(node);

    if (node->indirect != UNASSIGNED_NODE)
        return 0;
    data_block_number first_data = first_unused_data(fs_pointer);
    if (first_data == -1) {
        return -1;
    }
    node->indirect = first_data;
    set_data_used(fs_pointer, first_data, 1);
    node->nlink = 1;
    int i;
    data_block_number *block_array =
        (data_block_number *)(fs_pointer->data_root + first_data);
    for (i = 0; i < NUM_DIRECT_INODES; ++i) {
        block_array[i] = UNASSIGNED_NODE;
    }
    return 0;
}
int minixfs_min_blockcount(file_system *fs, const char *path, int block_count) {
    inode *nody = get_inode(fs, path);
    if (!nody) {
        nody = minixfs_touch(fs, path);
        if (!nody)
            return -1;
    }

    data_block_number *block_array = nody->direct;
    int err = 0;
    if (block_count < NUM_DIRECT_INODES) {
        block_array = nody->direct;
        for (int i = 0; i < block_count; i++) {
            if (block_array[i] == -1) {
                err = add_data_block_to_inode(fs, nody);
                if (err == -1)
                    return -1;
                memset(fs->data_root + block_array[i], 0, sizeof(data_block));
            }
        }
    } else {
        for (int i = 0; i < NUM_DIRECT_INODES; i++) {
            if (block_array[i] == -1) {
                err = add_data_block_to_inode(fs, nody);
                if (err == -1)
                    return -1;
                memset(fs->data_root + block_array[i], 0, sizeof(data_block));
            }
        }
        err = add_single_indirect_block(fs, nody);
        if (err == -1)
            return -1;
        block_array = (data_block_number *)(fs->data_root + nody->indirect);
        block_count -= NUM_DIRECT_INODES;
        for (int i = 0; i < block_count; i++) {
            if (block_array[i] == -1) {
                err = add_data_block_to_indirect_block(fs, block_array);
                if (err == -1)
                    return -1;
                memset(fs->data_root + block_array[i], 0, sizeof(data_block));
            }
        }
    }
    return 0;
}

int minixfs_stat(file_system *fs, char *path, struct stat *buf) {
    inode *node = get_inode(fs, path);
    if (!node)
        return -1;
    buf->st_size = node->size;
    buf->st_blksize = sizeof(data_block);
    buf->st_blocks = (node->size + sizeof(data_block) - 1) / sizeof(data_block);

    buf->st_nlink = node->nlink;

    buf->st_mode = node->mode & 0777;
    if (is_directory(node)) {
        buf->st_mode |= S_IFDIR;
    } else {
        buf->st_mode |= S_IFREG;
    }

    buf->st_uid = node->uid;
    buf->st_gid = node->gid;

    buf->st_atim = node->atim;
    buf->st_mtim = node->mtim;
    buf->st_ctim = node->ctim;

    buf->st_ino = node - fs->inode_root;
    buf->st_dev = 0;

    return 0;
}

inode *minixfs_touch(file_system *fs, const char *path) {
    if (*path != '/') {
        fprintf(stderr, "Path not absolute");
        return NULL;
    }
    if (get_inode(fs, path) != NULL) {
        fprintf(stderr, "Filename not unique");
        return NULL;
    }
    const char *filename;
    inode *nody = parent_directory(fs, path, &filename);
    inode *parent_node = nody;
    size_t name_length = strlen(filename);
    if (!valid_filename(filename) || name_length > FILE_NAME_LENGTH) {
        return NULL;
    }
    if (!is_directory(nody)) {
        fprintf(stderr, "Parent Not Directory");
        return NULL;
    }

    data_block_number data_last = (nody->size) / sizeof(data_block);
    data_block_number *block_array = nody->direct;
    int is_indirect = 0;
    if (data_last >= NUM_DIRECT_INODES) {
        if (nody->indirect == UNASSIGNED_NODE) {
            inode_number block = add_single_indirect_block(fs, nody);
            if (block == -1) {
                fprintf(stderr, "Filesystem has no more space");
                return NULL;
            }
        }
        block_array = (data_block_number *)(fs->data_root + nody->indirect);
        is_indirect = 1;
        data_last -= NUM_DIRECT_INODES;
    }

    int new_inode = first_unused_inode(fs);
    init_inode(nody, fs->inode_root + new_inode);
    size_t last_offset = parent_node->size % sizeof(data_block);
    data_block *blocky = fs->data_root + block_array[data_last];
    if (last_offset == 0) { // We are at the end of a block
        data_block_number data_block;
        if (is_indirect)
            data_block = add_data_block_to_indirect_block(fs, block_array);
        else
            data_block = add_data_block_to_inode(fs, nody);
        if (data_block == -1) {
            fprintf(stderr, "Filesystem has no more space");
            return NULL;
        }
        blocky = fs->data_root + data_block;
    }
    char *cpy = calloc(1, FILE_NAME_LENGTH);
    strncpy(cpy, filename, FILE_NAME_LENGTH);
    memcpy(((char *)blocky) + last_offset, cpy, FILE_NAME_LENGTH);
    sprintf(((char *)blocky) + last_offset + FILE_NAME_LENGTH, "%08zx",
            (size_t)new_inode);
    parent_node->size += FILE_NAME_ENTRY; // Add one directory to parent only
    free(cpy);
    return fs->inode_root + new_inode;
}
