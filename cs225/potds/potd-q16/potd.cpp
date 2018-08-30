#include "potd.h"
#include <iostream>

using namespace std;

string stringList(Node *head) {
  Node* temp = head;
  string stuff;
  int i = 0;
  for(; temp != NULL; i++){
    if(i != 0) stuff += " -> ";
    cout << "hi" <<endl;
    stuff += "Node " + to_string(i) + ": " + to_string(temp->data_);
    temp = temp->next_;
  }
  if(i == 0) return "Empty list";
  return stuff;
}
