/**
* Deadlocked Diners Lab
* CS 241 - Spring 2018
*/

#include "company.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
void *work_interns(void *p) {
   Company * c = (Company*)p;
   pthread_mutex_t *left_intern, *right_intern;
   while(running) {
      int fail_l = 0, fail_r = 0;
      left_intern = Company_get_left_intern(c);
      right_intern = Company_get_right_intern(c);
      fail_l = pthread_mutex_trylock(left_intern);
      if(!fail_l && !(fail_r = pthread_mutex_trylock(right_intern))) {
         Company_hire_interns(c);
         pthread_mutex_unlock(left_intern);
         pthread_mutex_unlock(right_intern);
      }
      else if (!fail_l)
         pthread_mutex_unlock(left_intern);
      Company_have_board_meeting(c);
   }
   return NULL;
}
