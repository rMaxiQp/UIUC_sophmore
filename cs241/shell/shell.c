/**
* Shell Lab
* CS 241 - Spring 2018
*/

#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <ctype.h>
#include "format.h"
#include "shell.h"
#include "includes/vector.h"

#define AND 13
#define OR 17
#define BACK 19

typedef struct process {
    char *command;
    char *status;
    pid_t pid;
    struct process* next;
} process;

int exec_cmd(char* cmd);
int shell(int argc, char** argv);
int kill_pid(pid_t pid);
void delete_process(pid_t pid);

//keep track of history with linked list
static vector* history_file = NULL;
static char* directory = NULL;
static char* file_path = NULL;
static pid_t shell_pid = 0;
static process* process_head;

// ---------------- helper--------------
void create_process(char* line, pid_t pid) {
   process *node = calloc(1, sizeof(process));
   node->command = strdup(line);
   node->pid = pid;
   node->status = STATUS_RUNNING;
   node->next = process_head;
   process_head = node;
}


char* find_process(pid_t pid) {
   process *cur = process_head;
   process *prev = process_head;
   while(cur && cur->pid != pid) {
      prev = cur;
      cur = cur->next;
   }
   if(cur) {
      prev->next = cur->next;
      char* string = cur->command;
      return string;
   }
   else
      print_no_process_found(pid);
   return NULL;
}

void handle_child() {
   int status = 0;
   pid_t pid = waitpid(-1 , &status, WNOHANG);
   if WIFEXITED(status)
      delete_process(pid);
}

void delete_process(pid_t pid) {
   process* temp = process_head;
   process* prev = process_head;
   while(temp) {
      if(temp->pid == pid) {
         prev->next = temp->next;
         if(prev == temp)
            process_head = prev->next;
         free(temp->command);
         free(temp);
         return;
      }
      prev = temp;
      temp = temp->next;
   }
}

void handle_int() {
   pid_t pid =getpid();
   int flag = 0;
   process* temp = process_head;
   while(temp) {
      if(pid == temp->pid){
         flag = 1;
      }
      temp = temp->next;
   }
   if(!flag)
      exit(0);
}

char* str_check(char* decode) {
   size_t size = strlen(decode);
   char output[size];
   size_t idx = 0, t = 0;
   char peek = '\0';
   for(; t < size; t++) {
      if(idx == size) break;
      if(idx + 1 < size)
         peek = decode[idx+1];
      else
         peek = '\0';

      if(decode[idx] == '\\' && (peek == '&' || peek == '|' || peek == ';')){
         output[t] = peek;
         idx++;
      }
      else
         output[t] = decode[idx];
      idx++;
   }
   char *o = calloc(1, (t+1) * sizeof(char));
   o = memcpy(o, output, t);
   o[t] = 0;
   return o;
}


// ----------------- SHELL FUNCTION------------------

/*
 * cd <path>
 *
 * Changes the current working directory of the shell to <path>.
 * Paths not starting with / should be followed relative to the current directory.
 * If the directory does not exist, then print the appropriate error.
 * Unlike your regular shell, the <path> argument is mandatory here.
 * A missing path should be treated as a nonexistent directory.
 * */
int cd(char* path) {
   if(!path) {
      print_no_directory("\0");
      return 1;
   }

   if(path[strlen(path) - 1] == '\n')
      path[strlen(path) - 1] = '\0';

   if(chdir(path) == -1) {
      print_no_directory(path);
      return 1;
   }

   free(directory);
   directory = get_current_dir_name();
   return 0;
}

/*
 * !history
 *
 * Prints out each command in the history, in order.
 * This command is not stored in history
 * */
int history() {
   if(vector_empty(history_file))
      return 1;
   vector_pop_back(history_file);
   size_t size = vector_size(history_file);
   for(size_t t = 0; t < size; t++)
      print_history_line(t,(char*) vector_get(history_file, t));
   return 0;
}

/*
 * #<n>
 *
 * Prints and executes the nth command in history (in chronological order,
 * from earliest to most recent), where n is a non-negative integer.
 * Other values of n will not be tested.
 * The command run should be stored in the history.
 * If n is not a valid index, then print the appropriate error and do not store anything in the history.
 *
 * Print out the command before executing if there is a match
 * The #<n> command itself is not stored in history, but the command being executed (if any) is.
 * */
int n_th(size_t n) {
   vector_pop_back(history_file);
   if(n < vector_size(history_file)) {
      char* string = (char*)vector_get(history_file, n);
      print_command(string);
      return exec_cmd(string);
   }
   print_invalid_index();
   return 1;
}

/*
 * !<prefix>
 *
 * Prints and executes the last command that has the specified prefix.
 * If no match is found, print the appropriate error and do not store anything in the history.
 * The prefix may be empty.
 *
 * Print out the command before executing if there is a match
 * The <!prefix> command itself is not stored in history, but the command being executed (if any) is.
 * */
int prefix(char* prefix) {
   vector_pop_back(history_file);
   size_t size = vector_size(history_file);
   size_t len = strlen(prefix) - 1;
   for(size_t t = size; t > 0; t--) {
      size_t i = 0;
      char* string = (char*) vector_get(history_file, t - 1);
      for(; i < len; i++)
         if(prefix[i+1] != string[i])
            break;
      if(i == len) {
         print_command(string);
         return exec_cmd(string);
      }
   }
   print_no_history_match();
   return 1;
}

/*
 * ps
 *
 * print out information about all currently executing processes.
 * You should include the shell and its immediate children, but don't worry about grandchildren or other processes.
 * Make sure you use print_process_info()
 *
 * while ps is normally a seperate binary, it is a built-in command for your shell
 *
 * The order in which you print the processes does not matter.
 * The 'command' for print_process_info should be the full command you executed, with escape sequences expanded
 * The & for background processes is optional.
 * For the main shell process only, you do not need to include the command-line flags
 * */
int ps() {
   process* current = process_head;
   while(current) {
      print_process_info(current->status, current->pid, current->command);
      current = current->next;
   }
   return 0;
}

/*
 * kill<pid>
 *
 * Send SIGTERM to the specified process.
 *
 * Use the appropriate prints from format.h for:
 *
 * Successfully sending SIGTERM to process
 * No process with pid exists
 * kill was ran without a pid
 * */
int kill_pid(pid_t pid) {
   char* string = find_process(pid);
   if(string){
      print_killed_process(pid, string);
      delete_process(pid);
      kill(pid, SIGTERM);
      return 0;
   }
   return 1;
}

/*
 * stop<pid>
 *
 * This command will allow your shell to stop a currently executing process by sending it the SIGSTP signal.
 * It may be resumed by using the command cont.
 *
 * Use the appropriate prints from format.h for:
 *
 * Process was successfully sent SIGSTP
 * No such process exists
 * stop was ran without a pid
 * */
int stop_pid(pid_t pid) {
   process *cur = process_head;
   while(cur && cur->pid != pid) {
      cur = cur->next;
   }
   if(!cur) {
      print_no_process_found(pid);
      return 1;
   }
   cur->status = STATUS_STOPPED;
   kill(pid, SIGSTOP);
   print_stopped_process(pid, cur->command);
   return 0;
}

/*
 * cont<pid>
 *
 * This command resumes the specified process by sending it SIGCONT.
 *
 * Use the appriate prints from format.h for:
 *
 * No such process exists
 * cont was ran without a pid
 *
 * Any <pid> used in kill, stop, or, cont will either be a process that is a direct child of your shell or a non-existent process.
 * You do not have to worry about killing other processes.
 * */
int cont_pid(pid_t pid) {
   process *cur = process_head;
   while(cur && cur->pid != pid) {
      cur = cur->next;
   }
   if(!cur) {
      print_no_process_found(pid);
      return 1;
   }
   cur->status = STATUS_RUNNING;
   kill(pid, SIGCONT);
   return 0;
}


/*
 * exit
 *
 * The shell will exit once it receives the exit command or an EOF.
 * The latter is sent by typing Ctrl+D on an empty line, and from a script file (as used with the -f flag) this is sent once the end of the file is reached.
 * This should cause your shell to exit with exit status 0. You should make sure that all processes you’ve started have been cleaned up.
 *
 * If there are currently stopped or running background processes when your shell receives exit or Control-D (EOF), you should kill and cleanup each of those children before your shell exits. You do not need to worry about SIGTERM.
 * (Think, what function lets you cleanup information about child processes?)
 *
 * You should not let your shell’s children be zombies, but your children’s children might turn into zombies.
 * You don’t have to handle those.
 *
 * exit should not be stored in history.
 * */
void exit_cmd() {
   while(!process_head && process_head->pid != shell_pid) {
      kill(process_head->pid, SIGTERM);
      free(process_head->command);
      free(process_head);
      process_head = process_head->next;
   }

   if(file_path) {
      FILE *f = fopen(file_path, "w");
      size_t size = vector_size(history_file);
      for(size_t t = 0; t < size; t++)
         fprintf(f, "%s\n", (char*)vector_get(history_file, t));
      fclose(f);
      free(file_path);
   }

   vector_destroy(history_file);
   free(process_head->command);
   free(process_head);
}



/*
 * For commands that are not built-in, the shell should consider the command name to be the name of a file that contains executable binary code.
 * Such a code must be executed in a process different from the one executing the shell.
 * You must use fork, exec, and wait/waitpid.
 *
 * The fork/exec/wait paradigm is as follows:
 *
 * fork a child process. The child process must execute the command with exec*, while the parent must wait for the child to terminate before printing the next prompt.
 *
 * You are responsible of cleaning up all the child processes upon termination of your program.
 * It is important to note that, upon a successful execution of the command, exec never returns to the child process.
 * exec only returns to the child process when the command fails to execute successfully.
 * If any of fork, exec, or wait fail, the appropriate error should be printed and your program should exit with exit status 1
 * */
int external_cmd(char* line, int bg) {
   size_t num = 0;
   int status = 0;
   char** arg = strsplit(line, " ", &num);
   fflush(stdout);
   pid_t pid = fork();
   if(pid == -1)
      print_fork_failed();
   else if(!pid) {
      if(execvp(*arg, arg) == -1) {
         print_exec_failed(*arg);
         exit(EXIT_FAILURE);
      }
      exit(EXIT_SUCCESS);
   }
   else {
      print_command_executed(pid);
      if(!bg){
         if(waitpid(pid, &status, 0) == -1) {
            print_wait_failed();
            free_args(arg);
            return(EXIT_FAILURE);
         }
         free_args(arg);
      }
      else {
         if(setpgid(pid, pid) == -1) {
            print_setpgid_failed();
            free_args(arg);
            return(EXIT_FAILURE);
         }
         create_process((char*)vector_get(history_file, vector_size(history_file) - 1), pid);
         waitpid(pid, &status, WNOHANG);
         free_args(arg);
      }
   }
   return WEXITSTATUS(status);
}

int pick_cmd(char* arg, char* end, int flag) {
   if(!(end - arg))
      return 1;
   size_t num;
   int result = 1;
   char* string = strndup(arg, end - arg);
   char* escape = str_check(string);
   char** cmd = strsplit(escape, " ", &num);

   char* first = strdup(cmd[0]);
   char* second = NULL;
   if(num > 1)
      second = strdup(cmd[1]);
   free_args(cmd);

   if(!strcmp(first, "cd")) {
      result = cd(second);
   }
   else if(!strcmp(first, "ps")) {
      result = ps();
   }
   else if(!strncmp(first, "#", 1)) {
      ssize_t temp = strtol(first + 1, NULL, 10);
      if(!isdigit(first[1]) || temp < 0 || num > 1) {
         print_invalid_command(arg);
      }
      else {
         result = n_th(temp);
      }
   }
   else if(!strcmp(first, "!history")) {
      result = history();
   }
   else if(!strcmp(first, "kill")) {
      if(num > 1) {
         result = kill_pid(strtol(second, NULL, 10));
      }
      else
         print_invalid_command(arg);
   }
   else if(!strcmp(first, "stop")) {
      if(num > 1) {
         result = stop_pid(strtol(second, NULL, 10));
      }
      else
         print_invalid_command(arg);
   }
   else if(!strcmp(first, "cont")) {
      if(num > 1) {
         result = cont_pid(strtol(second, NULL, 10));
      }
      else
         print_invalid_command(arg);
   }
   else if(!strncmp(first, "!", 1)) {
      result = prefix(arg);
   }
   else if(!strcmp(first, "exit")) {
      result = 233;
   }
   else if(flag == BACK) {
      result = external_cmd(escape, BACK);
   }
   else {
      result = external_cmd(escape, 0);
   }

   free(first);
   if(second)
      free(second);
   free(string);
   free(escape);
   return result;
}
/*
 * handle cmd
 *
 * Logical Operators:
 *
 * && is the AND operator.
 *
 * Input: x && y
 *
 * The shell first runs x, then checks the exit status.
 *
 * If x exited successfully (status = 0), run y.
 * If x did not exit successfully (status ≠ 0), do not run y
 * This is also known as short-circuiting.
 *
 * This mimics short-circuiting AND in boolean algebra: if x is false, we know the result will be false without having to run y.
 *
 *
 * || is the OR operator.
 *
 * Input: x || y
 *
 * The shell first runs x, then checks the exit status.
 * If x exited successfully, the shell does not run y.
 * This is short-circuiting.
 * If x did not exit successfully, run y.
 *
 * Boolean algebra: if x is true, we can return true right away without having to run y.
 *
 * ; is the command separator.
 *
 * Input: x; y
 *
 * The shell first runs x.
 * The shell then runs y.
 *
 * An external command suffixed with & should be run in the background.
 * In other words, the shell should be ready to take the next command before the given command has finished running.
 * There is no limit on the number of background processes you can have running at one time (aside from any limits set by the system).
 *
 * There may or may not be a single space between the rest of the command and &.
 * For example, pwd& and pwd & are both valid.
 *
 * While the shell should be usable after calling the command, after the process finishes, the parent is still responsible for waiting on the child (hint: catch a signal).
 * Avoid creating zombies!
 * Think about what happens when multiple children finish around the same time.
 * Backgrounding will not be chained with the logical operators.
 * */
int exec_cmd(char* cmd) {
   if(!history_file)
      history_file = string_vector_create();

   char* start = cmd;
   char* end = start + strlen(cmd);
   char* escape = str_check(cmd);
   vector_push_back(history_file, escape);
   free(escape);

   while(*(start) == ' ' || *(end-1) == ' ') {
      if(start == end || start + 1 == end) return 0;
      if(*(start) == ' ') start++;
      if(*(end - 1) == ' ') end--;
   }
   end = 0;

   if(!strcmp(start, "exit")) {
      vector_pop_back(history_file);
      return 233;
   }

   size_t length = strlen(start);
   int result = -1;
   char* needle_and = " && ";
   char* needle_or = " || ";
   char* needle_separator = ";";
   char needle_back = '&';
   char* ret = NULL;
   ret = strstr(start, needle_and);

   if(ret) {
      if(!pick_cmd(start, ret, AND) && length >(size_t) (ret - start) + 2)
         result = pick_cmd(ret + 3, end, AND);
      return result;
   }
   else if(!ret) {
      ret = strrchr(start, needle_back);
      if(ret) {
         if(*(ret-1) != '\\') {
            result = pick_cmd(start, ret, BACK);
            return result;
         }
      }
   }

   ret = strstr(start, needle_or);
   if(ret) {
      if(!pick_cmd(start, ret + 1, OR))
         result = 0;
      else if(length > (size_t) (ret - start) + 2) {
         pick_cmd(ret + 3, end, OR);
      }
      return result;
   }

   ret = strstr(start, needle_separator);
   if(ret && *(ret-1) != '\\') {
      result = pick_cmd(start, ret, 0);
      if(length > (size_t) (ret - start) + 1) {
         result = pick_cmd(ret + 1, end, 0);
      }
      return result;
   }
   else
      result = pick_cmd(start, NULL, 0);

   return result;
}

/*
 * internal_loop
 *
 * take cmd and exectue
 * */
void internal_loop() {
   if(!history_file)
      history_file = string_vector_create();
   char* cmd = NULL;
   int status = 0;
   size_t size = 0;
   int read = 0;
   if(directory)
      free(directory);
   directory = get_current_dir_name();
   print_prompt(directory, shell_pid);
   while( (read = getline(&cmd, &size, stdin)) != -1){
      char* new = strstr(cmd, "\n");
      *new = 0;

      if(strlen(cmd) == 0) {
         print_prompt(directory, shell_pid);
         continue;
      }

      if(cmd[read-1] == '\n')
         cmd[read-1] = 0;

      if((status = exec_cmd(cmd) == 233))
         break;
      print_prompt(directory, shell_pid);
   }
   free(cmd);
   exit_cmd();
}


// -------------- SHELL FLAG  ------------------

/*
 * argument with -h, take history file.
 * */

void take_history(char* path) {
   FILE *f = fopen(path, "r");
   if(!f) {
      exit(EXIT_FAILURE);
   }
   history_file = string_vector_create();
   char* line = NULL;
   size_t size = 0;
   ssize_t read = 0;
   while((read = getline(&line, &size,f) != -1)) {
      if(read > 0 && line[read-1] == '\n')
         line[read-1] = '\0';

      if(*line)
         vector_push_back(history_file, line);
   }
   file_path = get_full_path(path);
   free(line);
   fclose(f);
}

/**
 * argument with -f, execute commands and exit
 * */

void print_file(char* file_name) {
   FILE * f = fopen(file_name, "r");
   if(!f) {
      print_script_file_error();
      exit(EXIT_SUCCESS);
   }
   char* line = NULL;
   size_t size = 0;
   ssize_t read = 0;
   vector* temp = string_vector_create();
   while((read = getline(&line, &size, f) != -1)) {

      if(read  > 0 && line[strlen(line)-1] == '\n')
         line[strlen(line)-1] = '\0';

      if(!*line) continue;

      vector_push_back(temp, line);
   }
   size_t s = vector_size(temp);
   for(size_t t = 0; t < s; t++) {
      exec_cmd(vector_get(temp, t));
   }

   vector_destroy(temp);
   exit(EXIT_SUCCESS);
}

// ---------------- SHELL MAIN--------------------

int shell(int argc, char *argv[]) {
    // TODO: This is the entry point for your shell.

   int opt;
   int flag = 1;
   if(argc > 2) {
      while((opt = getopt(argc, argv, "h:f:")) != -1) {
         if(optind > argc) {
            print_usage();
            exit(EXIT_SUCCESS);
         }
         switch(opt) {
            case 'h':
               flag = 0;
               take_history(optarg);
               break;
            case 'f':
               flag = 0;
               print_file(optarg);
               break;
            default:
               print_usage();
               exit(EXIT_SUCCESS);
         }
      }
   }

   signal(SIGINT, handle_int);
   signal(SIGCHLD, handle_child);
   if(argc > 1 && flag) {
      print_usage();
      exit(EXIT_SUCCESS);
   }
   shell_pid = getpid();
   process_head = calloc(1, sizeof(process));
   process_head->command = strdup("./shell");
   process_head->pid = shell_pid;
   process_head->status = STATUS_RUNNING;
   process_head->next = NULL;
   internal_loop();
   exit(0);
}
