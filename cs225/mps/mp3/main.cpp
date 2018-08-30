#include "list.h"
#include <iostream>
#include <ostream>
int main() {

  List<int> list;
  for (unsigned i = 0; i < 3; i++) {
     list.insertFront(i);
  }
  list.reverseNth(2);
  return 0;
}
