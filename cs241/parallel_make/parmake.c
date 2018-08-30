/**
* Parallel Make Lab
* CS 241 - Spring 2018
*/


#include "format.h"
#include "includes/graph.h"
#include "includes/set.h"
#include "includes/queue.h"
#include "parmake.h"
#include "parser.h"


#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>

#define VISITED 1
#define CYCLE 2
#define FINISHED 3
#define FAIL 4
#define ON_PROCESS 5
#define RUNNING 6

graph *dependency_graph = NULL;
set *in_stack = NULL;
pthread_mutex_t m_node = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t m_data = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t m_state = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t m_parent = PTHREAD_MUTEX_INITIALIZER;

//entry
void single_thread(char **targets);
void multiple_thread(size_t num_threads, char **targets);

//general setup
void create_graph(char *makefile, char **targets);
void cycle_detection(char *vertex);
void on_disk_file(char *rule);
void run_cmd(rule_t *rule);

//single
void run(char *rule);

//parallel
void *par_run();
char *fetch_node();
void wait_n_th(char *node);
void update_parents(char *node);
void sort(char *node);
/* ========
 * = main =
 * ========
 * */
int parmake(char *makefile, size_t num_threads, char **targets) {
   // good luck!
   create_graph(makefile, targets);

   if(num_threads == 1)
      single_thread(targets);
   else
      multiple_thread(num_threads, targets);

   pthread_mutex_destroy(&m_node);
   pthread_mutex_destroy(&m_data);
   pthread_mutex_destroy(&m_state);
   pthread_mutex_destroy(&m_parent);
   graph_destroy(dependency_graph);
   return 0;
}

/* =================
 * = general setup =
 * =================
 * */

void create_graph(char *makefile, char **targets) {
   dependency_graph = parser_parse_makefile(makefile, targets);
   in_stack = string_set_create();
   cycle_detection("");
   set_destroy(in_stack);
}

/*
 * Cycle detection(DFS)
 *
 *  DFS(Vertex v) {
 *     mark v visited
 *     for each successor v' of v {
 *       if v' not yet visited {
 *          DFS(v')
 *          }
 *       }
 *    }
 * */
void cycle_detection(char * vertex) {

   rule_t *parent = (rule_t *)graph_get_vertex_value(dependency_graph, vertex);//get parent
   parent->state = VISITED;

   vector *list = graph_neighbors(dependency_graph, vertex);
   size_t size = vector_size(list);
   for(size_t t = 0; t < size; t++) {
      char *node = vector_get(list, t);
      rule_t * rule = (rule_t *)graph_get_vertex_value(dependency_graph, node);

      if(!set_contains(in_stack, node)) { //not in set
         set_add(in_stack, node);//push
         if(!rule->state) //node not visited
            cycle_detection(node);//recursion
      }
      else //in set ==> CYCLE
         rule->state = CYCLE;

      set_remove(in_stack, node);//pop

      if(rule->state == CYCLE)
         parent->state = CYCLE;
   }

   vector_destroy(list);
}


/*
 * check if the rule is on disk and if the dependencies are on disk
 * */
void on_disk_file(char *rule) {
   vector *dependency = graph_neighbors(dependency_graph, rule);//dependency
   rule_t * current = (rule_t *) graph_get_vertex_value(dependency_graph, rule);//current node
   size_t size = vector_size(dependency);
   int on_disk = !access(current->target, F_OK);//on_disk = 1 ==> access() == 0

   //the rule is the name of a file on disk
   if(on_disk) {
      for(size_t t = 0; t < size; t++) {
         char * node = vector_get(dependency, t);
         rule_t * r = (rule_t *) graph_get_vertex_value(dependency_graph, node);//get target

         //the rule depend on another rule that is not a file on disk
         if(!access(r->target, F_OK)) {
            struct stat r_stat;
            struct stat current_stat;
            //any of the rule's dependencies has a newer modification time than the rule's modification time
            if(!stat(rule, &current_stat) && !stat(node, &r_stat)) {
               //dependency time is older than rule time
               if(r_stat.st_mtime > current_stat.st_mtime) {
                  on_disk = 0;
                  break;
               }
            }
         }
         else
            on_disk = 0;
      }

      pthread_mutex_lock(&m_state);
      if(on_disk)
         current->state = FINISHED;
      pthread_mutex_unlock(&m_state);
   }

   vector_destroy(dependency);
}

void run_cmd(rule_t *rule) {
   size_t num_cmds = vector_size(rule->commands);
   for(size_t t = 0; t < num_cmds; t++) {
      char * cmd = vector_get(rule->commands, t);
      if(cmd && system(cmd)) { //cmd exist and the return is nonzero
         pthread_mutex_lock(&m_state);
         rule->state = FAIL;
         pthread_mutex_unlock(&m_state);
         break;
      }
   }

   pthread_mutex_lock(&m_state);
   if(rule->state != FAIL)
      rule->state = FINISHED;
   pthread_mutex_unlock(&m_state);
}

/* =======================
 * = single thread entry =
 * =======================
 * */
void single_thread(char **targets) {
   //get size
   size_t size = 0;
   while(targets[size]) size++;

   if(!size) { //no target
      run("");
   } else {
      for(size_t t = 0; t < size; t++) {
         rule_t * rule = (rule_t *) graph_get_vertex_value(dependency_graph, targets[t]);

         if(rule->state == CYCLE)
            print_cycle_failure(targets[t]);
         else if(rule->state != FINISHED)
            run(targets[t]);
      }
   }
}

/* ==========================================
 * = single thread version helper function  =
 * ==========================================
 * */

/*
 * recursive make(DFS)
 * */
void run(char * rule) {
   vector *dependency = graph_neighbors(dependency_graph, rule);//dependency
   rule_t * current = (rule_t *) graph_get_vertex_value(dependency_graph, rule);//current node
   size_t size = vector_size(dependency);

   if(size > 0) {//there are dependencies
      for(size_t t = 0; t < size; t++) {
         char * node = vector_get(dependency, t);
         rule_t * r = (rule_t *) graph_get_vertex_value(dependency_graph, node);

         if(r->state == CYCLE) {//cycle
            print_cycle_failure(node);
            current->state = CYCLE;
            return;
         }

         if(r->state != FAIL && r->state != FINISHED)
            run(node);

         if(r->state == FAIL) //dependency fail
            current->state = FAIL;
      }
   }

   //fprintf(stderr, "run %s in single thread\n", rule);
   on_disk_file(rule);
   current = (rule_t *) graph_get_vertex_value(dependency_graph, rule);
   if(current->state == VISITED || !current->state) {
      //base case
      //run comand one by one
      run_cmd(current);
   }

   vector_destroy(dependency);
}

/* ==========================
 * = multiple threads entry =
 * ==========================
 * */

int queue_size = 0;
int targets = 0;
queue *q = NULL;

void multiple_thread(size_t num_threads, char **targets) {

   size_t size = 0;
   while(targets[size]) size++;

   q = queue_create(-1);
   if(size) {
      for(size_t t = 0; t < size; t++)
         sort(targets[t]);
   } else {
      vector * v = graph_neighbors(dependency_graph, "");
      VECTOR_FOR_EACH(v, chlid, {
            sort((char *)chlid);
            });
      vector_destroy(v);
   }

   pthread_t threads[num_threads];
   for(size_t t = 0; t < num_threads; t++)
      pthread_create(&threads[t], NULL, par_run, NULL);

   for(size_t t = 0; t < num_threads; t++)
      pthread_join(threads[t], NULL);

   queue_destroy(q);
}

/* ============================================
 * = multiple threads version helper function =
 * ============================================
 * */

/*
 * entry function of worker thread
 * */
void *par_run() {
   char *node = NULL;

   while(1) {
      node = fetch_node();
      if(node == NULL)
         break;

      //possible state: RUNNING
      on_disk_file(node);

      //possible state: RUNNING || FINISH
      pthread_mutex_lock(&m_state);
      rule_t *rule = (rule_t *) graph_get_vertex_value(dependency_graph, node);
      pthread_mutex_unlock(&m_state);

      if(rule->state == RUNNING)
         run_cmd(rule);
      //possible state: FINISH || FAIL
      update_parents(node);
   }
   return NULL;
}

/*
 * fetch the running node
 * */
char *fetch_node() {
   char *node = NULL;
   rule_t *node_rule = NULL;
   while(1) {
      //possible state: FAIL || ON_PROCESS

      pthread_mutex_lock(&m_parent);
      if(targets > 0) {
         targets--;
         pthread_mutex_unlock(&m_parent);
         node = queue_pull(q);

         pthread_mutex_lock(&m_node);
         queue_size--;
         pthread_mutex_unlock(&m_node);

      }
      else {
         pthread_mutex_unlock(&m_parent);
         return NULL;
      }

      pthread_mutex_lock(&m_state);
      node_rule = (rule_t *) graph_get_vertex_value(dependency_graph, node);
      int state = node_rule->state;
      pthread_mutex_unlock(&m_state);

      if(state == FAIL) {
         update_parents(node);
      }
      else {
         pthread_mutex_lock(&m_state);
         node_rule->state = RUNNING;
         pthread_mutex_unlock(&m_state);
         return node;
      }
   }
}

/*
 * push dependency into queue (can only be called for VISITED or DEFAULT)
 * */
void wait_n_th(char *node) {
   vector *v = graph_neighbors(dependency_graph, node);
   size_t size = vector_size(v);

   //no dependency
   pthread_mutex_lock(&m_state);
   rule_t *rule = (rule_t *) graph_get_vertex_value(dependency_graph, node);
   if(!size) {
      rule->state = ON_PROCESS;
      pthread_mutex_unlock(&m_state);
      queue_push(q, node);
      pthread_mutex_lock(&m_node);
      queue_size++;
      pthread_mutex_unlock(&m_node);
   }
   else {
         pthread_mutex_unlock(&m_state);
         //push dependencies into queue
         size_t* count = calloc(1, sizeof(size_t));
         for(size_t t = 0; t < size; t++) {
            char * node = vector_get(v, t);
            pthread_mutex_lock(&m_state);
            rule_t * rule = (rule_t *) graph_get_vertex_value(dependency_graph, node);
            int state = rule->state;
            pthread_mutex_unlock(&m_state);
            if(state != CYCLE && state != FINISHED && state != FAIL) {
               (*count)++;
               queue_push(q, node);
               pthread_mutex_lock(&m_node);
               queue_size++;
               pthread_mutex_unlock(&m_node);
            }
         }
         pthread_mutex_lock(&m_data);
         rule->data = (void *)count;
         pthread_mutex_unlock(&m_data);
   }
   vector_destroy(v);
}

/*
 * update antineighbor's data
 * */
void update_parents(char *node) {

   vector *v = graph_antineighbors(dependency_graph, node);
   size_t size = vector_size(v);

   pthread_mutex_lock(&m_data);
   rule_t * r = (rule_t *) graph_get_vertex_value(dependency_graph, node);
   pthread_mutex_unlock(&m_data);

   //no parent
   if(size > 0 && strcmp(vector_get(v, 0), "")) {
       //update parents
      for(size_t t = 0; t < size; t++) {
         char *parent = vector_get(v, t);



         pthread_mutex_lock(&m_state);
         rule_t *rule = (rule_t *) graph_get_vertex_value(dependency_graph, parent);
         if(r->state == FAIL) {

            pthread_mutex_lock(&m_data);
            if(rule->state == ON_PROCESS) {
               free(rule->data);
               rule->data = NULL;
            }
            pthread_mutex_unlock(&m_data);

            pthread_mutex_lock(&m_node);
            queue_push(q, parent);
            queue_size++;
            pthread_mutex_unlock(&m_node);

            rule->state = FAIL;
         } else {
            size_t * data = (size_t *)rule->data;
            if(data != NULL) {
               *data -= 1;
               if((*data) == 0) {
                  pthread_mutex_lock(&m_data);
                  free(rule->data);
                  rule->data = NULL;
                  pthread_mutex_unlock(&m_data);

                  pthread_mutex_lock(&m_node);
                  queue_push(q, parent);
                  queue_size++;
                  pthread_mutex_unlock(&m_node);
               }
            }
         }
         pthread_mutex_unlock(&m_state);
      }
   }
   vector_destroy(v);
}

/*
 * push no_dependencies into the queue
 * */
void sort(char *node) {
   rule_t * rule = (rule_t *) graph_get_vertex_value(dependency_graph, node);
   if(rule->state == CYCLE)
      print_cycle_failure(node);
   else if(rule->state != ON_PROCESS) {
      vector * v = graph_neighbors(dependency_graph, node);
      rule->state = ON_PROCESS;
      targets++;
      if(vector_size(v)) {
         size_t* count = calloc(1, sizeof(size_t));
         *count = vector_size(v);
         rule->data = (void *)count;
         VECTOR_FOR_EACH(v, chlid, {
            sort(chlid);
         });
      }
      else {
         queue_push(q, node);
         queue_size++;
      }
      vector_destroy(v);
   }
}
