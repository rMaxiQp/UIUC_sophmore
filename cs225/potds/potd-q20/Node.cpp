#include "Node.h"
#include <iostream>
using namespace std;

Node *listUnion(Node *first, Node *second) {

  Node* head = NULL;
  Node* temp;
  Node* f = first;
  Node* s = second;

  if(f == NULL && s == NULL) return NULL;
  else if(first == NULL){ return s;
    while(s != NULL){
      if(head == NULL) {
        head = new Node(*s);
        temp = head;
      }else{
        if(temp->data_ != s->data_){
        temp ->next_ = new Node(*s);
        temp = temp ->next_;
      }
    }
      s = s->next_;
    }
  } else if(second == NULL){
    while(f){
      if(head == NULL) {
        head = new Node(*f);
        temp = head;
      }else{
        if(temp->data_ != f->data_){
        temp ->next_ = new Node(*f);
        temp = temp ->next_;}
      }
      f = f->next_;
    }
  } else{

  if(f->data_ > s->data_){
    head = new Node(*s);
    s = s->next_;
  } else{
    head = new Node(*f);
    f = f->next_;
  }

  temp = head;

  while(f && s){
    if(f->data_ < s->data_){
      if(f->data_ != temp->data_){
        temp->next_ = new Node(*f);
        temp = temp->next_;
      }
      f = f->next_;
    } else if (f->data_ >= s->data_){
      if(s->data_ != temp->data_){
        temp->next_ = new Node(*s);
        temp = temp->next_;
      }
      s = s->next_;
    }else{
      s = s->next_;
      f = f->next_;
  }
}

while(f){
  if(f->data_ != temp->data_){
    temp->next_ = new Node(*f);
    temp = temp->next_;
  }
  f = f->next_;
}

while(s){
  if(s->data_ != temp->data_){
    temp->next_ = new Node(*s);
    temp = temp->next_;
  }
  s = s->next_;
}

}
return head;
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
