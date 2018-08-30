#include "potd.h"
#include <iostream>
using namespace std;

int main() {
  // Test 1: An empty list
  Node * head = NULL;
  cout << stringList(head) << endl;

  // Test 2: A list with exactly one node
  Node * h = new Node();
  h->data_ = 8;
  head = new Node();
  head->data_ = 1000;

  head->next_ = h;
  std::cout << stringList(head) << '\n';

  // Test 3: A list with more than one node

  return 0;
}
