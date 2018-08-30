#include "potd.h"
#include <iostream>

using namespace std;

void insertSorted(Node **head, Node *insert) {
  if(*head == NULL){
    *head = insert;
    insert->next_ = NULL;
    return;
    }
  else if((*head)->data_ > insert->data_){
    Node* temp = *head;
    *head = insert;
    insert->next_ = temp;
    return;
  }
  insertSorted(&((*head)->next_),  insert);
}
