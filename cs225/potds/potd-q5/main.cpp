#include <iostream>

#include "q5.h"
#include "food.h"

int main(){
  food* i = new food();
  cout << i->get_quantity() << endl;
  increase_quantity(i);
  cout << i->get_quantity() << endl;
  return 0;
}
