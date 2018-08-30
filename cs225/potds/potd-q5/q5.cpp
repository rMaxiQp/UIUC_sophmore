#include "q5.h"

void increase_quantity(food* foo){
  unsigned temp = foo->get_quantity();
  foo->set_quantity(temp+1);
}
