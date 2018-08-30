/**
* Finding Filesystems Lab
* CS 241 - Spring 2018
*/

#include <sys/types.h>
typedef int (*access_fn)(const char *pathname, int mode);

typedef int (*creat_fn)(const char *pathname, mode_t mode);
typedef int (*open_fn)(const char *pathname, int flags, ...);
typedef ssize_t (*read_fn)(int fd, void *buf, size_t count);
typedef ssize_t (*write_fn)(int fd, const void *buf, size_t count);
typedef int (*close_fn)(int fd);
typedef off_t (*lseek_fn)(int fd, off_t offset, int whence);
typedef int (*unlink_fn)(const char *pathname);
typedef int (*mkdir_fn)(const char *pathname, mode_t mode);
typedef int (*fchmod_fn)(int fd, mode_t mode);
typedef int (*fchmodat_fn)(int dirfd, const char *pathname, mode_t mode,
                           int flags);

typedef int (*fchown_fn)(int fd, uid_t owner, gid_t group);
typedef int (*fchownat_fn)(int dirfd, const char *pathname, uid_t owner,
                           gid_t group, int flags);

typedef int (*stat_fn)(int mode, const char *pathname, struct stat *buf);
typedef int (*fstat_fn)(int mode, int fd, struct stat *buf);
typedef int (*fstatat_fn)(int mode, int dirfd, const char *pathname,
                          struct stat *buf, int flags);
typedef int (*stat64_fn)(int mode, const char *pathname, struct stat64 *buf);
typedef int (*fstat64_fn)(int mode, int fd, struct stat64 *buf);
typedef int (*fstatat64_fn)(int mode, int dirfd, const char *pathname,
                            struct stat64 *buf, int flags);

typedef int (*fsync_fn)(int oldfd);

typedef int (*dup_fn)(int oldfd);
typedef int (*dup2_fn)(int oldfd, int newfd);

typedef DIR *(*opendir_fn)(const char *name);
typedef DIR *(*fdopendir_fn)(int fd);
typedef struct dirent64 *(*readdir64_fn)(DIR *dirp);
typedef struct dirent *(*readdir_fn)(DIR *dirp);
typedef int (*readdir_r_fn)(DIR *dirp, struct dirent *entry,
                            struct dirent **result);
typedef int (*closedir_fn)(DIR *dirp);

typedef void *(*mmap_fn)(void *addr, size_t length, int prot, int flags, int fd,
                         off_t offset);

typedef struct {
    int fd;
    int flags;
    char *path;
    size_t refcount;
    long offset;
} fakefile;

typedef struct {
    int fd;
    int entries_read;
    int max_entries;
    struct dirent *entry;
} fakedir;

static void destroy_fakefile(fakefile *f);
static void destroy_fakedir(fakedir *d);

static access_fn orig_access;

static creat_fn orig_creat;
static open_fn orig_open;
static read_fn orig_read;
static write_fn orig_write;
static close_fn orig_close;
static lseek_fn orig_lseek;

static unlink_fn orig_unlink;
static mkdir_fn orig_mkdir;
static fchmod_fn orig_fchmod;
static fchmodat_fn orig_fchmodat;
static fchown_fn orig_fchown;
static fchownat_fn orig_fchownat;

static fsync_fn orig_fsync;
static fsync_fn orig_fdatasync;

static dup_fn orig_dup;
static dup2_fn orig_dup2;

static stat_fn orig_stat;
static stat_fn orig_lstat;
static fstat_fn orig_fstat;
static fstatat_fn orig_fstatat;
static stat64_fn orig_stat64;
static stat64_fn orig_lstat64;
static fstat64_fn orig_fstat64;
static fstatat64_fn orig_fstatat64;

static fdopendir_fn orig_fdopendir;
static opendir_fn orig_opendir;
static readdir64_fn orig_readdir64;
static readdir_fn orig_readdir;
static readdir_r_fn orig_readdir_r;
static closedir_fn orig_closedir;

static mmap_fn orig_mmap;
