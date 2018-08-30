#include "Node.h"
#include <iostream>
using namespace std;

void mergeList(Node *first, Node *second) {

  if(first == NULL && second == NULL) return;
  else if(first == NULL){
    first = second;
    return;
  }
  else if(second == NULL) return;

  Node* f = first->next_;
  Node* s = second;
  Node* temp = first;

  while(f && s){
    temp->next_ = s;
    temp = temp->next_;
    s = s->next_;
    temp->next_ = f;
    temp = temp->next_;
    f = f->next_;
  }
  if(s != NULL){
    temp->next_ = s;
  }else if(f != NULL){
    temp->next_ = f;
  }
  //first = temp;
}

Node::Node() {
    numNodes++;
}

Node::Node(Node &other) {
    this->data_ = other.data_;
    this->next_ = other.next_;
    numNodes++;
}

Node::~Node() {
    numNodes--;
}

int Node::numNodes = 0;
